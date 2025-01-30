import matlab.engine
import numpy as np
import pandas as pd


class HCTSAWrapper:
    def __init__(self, matlab_path="matlab"):
        """
        Initialize HCTSA wrapper.
        :param matlab_path: Path to the MATLAB installation (default: 'matlab')
        """
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(matlab_path), nargout=0)  # Add HCTSA path

    def extract_features(self, time_series):
        """
        Extracts HCTSA features from a time series.
        :param time_series: 1D NumPy array representing the time series.
        :return: Pandas DataFrame with extracted HCTSA features.
        """
        if not isinstance(time_series, np.ndarray) or time_series.ndim != 1:
            raise ValueError("Input must be a 1D NumPy array.")

        # Convert NumPy array to Matlab format
        ts_matlab = matlab.double(time_series.tolist())

        # Call HCTSA feature extraction
        print("Extracting features using HCTSA...")
        feature_values, feature_names = self.eng.extract_HCTSA_features(ts_matlab, nargout=2)

        # Convert Matlab data to Python lists
        feature_values = np.array(feature_values).flatten()
        feature_names = [str(name) for name in feature_names]

        # Create DataFrame with extracted features
        features_df = pd.DataFrame([feature_values], columns=feature_names)
        return features_df

    def close(self):
        """ Close the MATLAB engine. """
        self.eng.quit()


# Example Usage
if __name__ == "__main__":
    # Example time series
    ts = np.sin(np.linspace(0, 10, 100))  # Example sine wave

    # Initialize and extract features
    hctsa = HCTSAWrapper(matlab_path="path_to_HCTSA")
    features = hctsa.extract_features(ts)
    hctsa.close()

    print("Extracted HCTSA Features:")
    print(features)
