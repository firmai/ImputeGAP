# ===============================================================================================================
# SOURCE: https://github.com/Graph-Machine-Learning-Group/grin
#
# THIS CODE HAS BEEN MODIFIED TO ALIGN WITH THE REQUIREMENTS OF IMPUTEGAP (https://arxiv.org/abs/2503.15250),
#   WHILE STRIVING TO REMAIN AS FAITHFUL AS POSSIBLE TO THE ORIGINAL IMPLEMENTATION.
#
# FOR ADDITIONAL DETAILS, PLEASE REFER TO THE ORIGINAL PAPER:
# https://openreview.net/pdf?id=kOu3-S3wJ7
# ===============================================================================================================

import torch

from . import Filler
from ..nn.models import GRINet


class GraphFiller(Filler):

    def __init__(self,
                 model_class,
                 model_kwargs,
                 optim_class,
                 optim_kwargs,
                 loss_fn,
                 scaled_target=False,
                 whiten_prob=0.05,
                 pred_loss_weight=1.,
                 warm_up=0,
                 metrics=None,
                 scheduler_class=None,
                 scheduler_kwargs=None,
                 verbose=False):
        super(GraphFiller, self).__init__(model_class=model_class,
                                          model_kwargs=model_kwargs,
                                          optim_class=optim_class,
                                          optim_kwargs=optim_kwargs,
                                          loss_fn=loss_fn,
                                          scaled_target=scaled_target,
                                          whiten_prob=whiten_prob,
                                          metrics=metrics,
                                          scheduler_class=scheduler_class,
                                          scheduler_kwargs=scheduler_kwargs)

        self.tradeoff = pred_loss_weight
        self.verbose = verbose
        if model_class in [GRINet]:
            self.trimming = (warm_up, warm_up)

    def trim_seq(self, *seq):
        seq = [s[:, self.trimming[0]:s.size(1) - self.trimming[1]] for s in seq]
        if len(seq) == 1:
            return seq[0]
        return seq

    def training_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and ensure it is boolean
        mask = batch_data['mask'].clone().detach().bool()

        # Randomly drop values using keep_prob
        batch_data['mask'] = torch.bernoulli(mask.float() * self.keep_prob).bool()

        eval_mask = batch_data.pop('eval_mask').bool()

        # ✅ Fix: Ensure all masks are correctly cast to boolean
        batch_data['mask'] = batch_data['mask'].bool()

        # ✅ Fix: Replace subtraction with logical AND & NOT
        eval_mask = (mask | eval_mask)  # ✅ Logical NOT instead of subtraction

        y = batch_data.pop('y')

        # Compute predictions and compute loss
        res = self.predict_batch(batch, preprocess=False, postprocess=False)
        imputation, predictions = (res[0], res[1:]) if isinstance(res, (list, tuple)) else (res, [])

        # trim to imputation horizon len
        imputation, mask, eval_mask, y = self.trim_seq(imputation, mask, eval_mask, y)
        predictions = self.trim_seq(*predictions)

        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            imputation = self._postprocess(imputation, batch_preprocessing)
            for i, _ in enumerate(predictions):
                predictions[i] = self._postprocess(predictions[i], batch_preprocessing)

        loss = self.loss_fn(imputation, target, mask)

        for pred in predictions:
            loss += self.tradeoff * self.loss_fn(pred, target, mask)


        # Logging
        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)
        self.train_metrics.update(imputation.detach(), y, eval_mask)  # all unseen data
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_loss', loss.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract and convert mask
        mask = batch_data['mask'].clone().detach().bool()
        eval_mask = batch_data.pop('eval_mask', torch.zeros_like(mask)).bool()
        batch_data['mask'] = batch_data['mask'].bool()

        # Apply same masking logic as in training
        eval_mask = (mask | eval_mask)

        # Extract target
        y = batch_data.pop('y')

        # Predict
        imputation = self.predict_batch(batch, preprocess=False, postprocess=False)

        # Process targets
        if self.scaled_target:
            target = self._preprocess(y, batch_preprocessing)
        else:
            target = y
            #target = self._preprocess(y, batch_preprocessing)
            imputation = self._postprocess(imputation, batch_preprocessing)

        # Compute losses
        val_loss = self.loss_fn(imputation, target, eval_mask)
        val_mae = torch.nn.functional.l1_loss(imputation, target, reduction='mean')

        if self.verbose:
            print(f"Validation Step - Batch {batch_idx}:")
            print(f"  - Target (y): {y.mean().item()}")
            print(f"  - Imputation: {imputation.mean().item()}")
            print(f"  - val_loss: {val_loss.item()}")
            print(f"  - val_mae: {val_mae.item()}")
            print(f"  - y stats: mean={y.mean():.5f}, std={y.std():.5f}, min={y.min():.5f}, max={y.max():.5f}")
            print(f"  - mask true count: {mask.sum().item()}")
            print(f"  - eval_mask true count: {eval_mask.sum().item()}")
            print(f"  - values count: {y.numel()}")

        if self.scaled_target:
            imputation = self._postprocess(imputation, batch_preprocessing)

        self.val_metrics.update(imputation.detach(), y, mask)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_loss', val_mae.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return val_mae

    def test_step(self, batch, batch_idx):
        # Unpack batch
        batch_data, batch_preprocessing = self._unpack_batch(batch)

        # Extract mask and target
        mask = batch_data['mask'].clone().detach().bool()

        y = batch_data.pop('y')

        # Compute outputs and rescale
        imputation = self.predict_batch(batch, preprocess=False, postprocess=True)
        test_loss = self.loss_fn(imputation, y, mask)
        val_mae = torch.nn.functional.l1_loss(imputation, y, reduction='mean')

        # Logging
        self.test_metrics.update(imputation.detach(), y, mask)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        self.log('test_loss', val_mae.detach(), on_step=False, on_epoch=True, logger=True, prog_bar=False)

        return test_loss