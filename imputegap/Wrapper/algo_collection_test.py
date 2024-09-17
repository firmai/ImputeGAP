#!/usr/bin/python


### Imports

import algo_collection
import numpy as np

### Data prep

# rmat = np.random.randn(10,5)
#
# mat = np.loadtxt(
#     "../../../Google Drive/_Master Thesis Info/Zakhar/collection_wrap_v3/collection_wrap_v3/airq_normal.txt")
# mat_miss = np.copy(mat)
# mat_miss[50:149,0] = np.nan

### Tests

# L,R = algo_collection.native_cd(rmat, 5)
# print(np.linalg.norm(rmat - L @ R.T))
#
# mat_rec = algo_collection.native_cdrec(mat_miss, 2)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
#
# mat_rec = algo_collection.native_cdrec_param(mat_miss, 2, 1E-6, 100)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
#
# mat_rec = algo_collection.native_stmvl(mat_miss)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
#
# mat_rec = algo_collection.native_stmvl_param(mat_miss, 7, 0.85, 2.0)
# print(np.sqrt(np.power(np.abs(mat_rec - mat), 2).sum() / 100))
