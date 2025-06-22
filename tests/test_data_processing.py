import numpy as np
import pandas as pd

from src.data_processing import (
    DataProcessor,
    create_sample_dataset,
    create_classification_dataset,
    generate_time_series,
)


class TestDataProcessor:
    def setup_method(self):
        self.processor = DataProcessor()
        # Create sample dataframe for testing
        self.df = pd.DataFrame(
            {
                "numeric_col": [1, 2, np.nan, 4, 5],
                "categorical_col": ["A", "B", "A", "B", np.nan],
                "float_col": [1.1, 2.2, 3.3, np.nan, 5.5],
                "text_col": ["cat", "dog", "cat", "bird", "dog"],
            }
        )

    def test_initialization(self):
        processor = DataProcessor()
        assert processor.scalers == {}
        assert processor.encoders == {}
        assert processor.missing_strategies == {}

    def test_handle_missing_values_mean(self):
        result = self.processor.handle_missing_values(
            self.df, strategy="mean", columns=["numeric_col"]
        )

        # Check that missing value is filled with mean
        expected_mean = (1 + 2 + 4 + 5) / 4  # 3.0
        assert result["numeric_col"].iloc[2] == expected_mean
        assert not result["numeric_col"].isna().any()

        # Check that strategy is recorded
        assert "numeric_col" in self.processor.missing_strategies
        assert self.processor.missing_strategies["numeric_col"]["strategy"] == "mean"

    def test_handle_missing_values_median(self):
        result = self.processor.handle_missing_values(
            self.df, strategy="median", columns=["numeric_col"]
        )

        # Check that missing value is filled with median
        expected_median = np.median([1, 2, 4, 5])  # 3.0
        assert result["numeric_col"].iloc[2] == expected_median
        assert not result["numeric_col"].isna().any()

    def test_handle_missing_values_mode(self):
        result = self.processor.handle_missing_values(
            self.df, strategy="mode", columns=["categorical_col"]
        )

        # Mode of ['A', 'B', 'A', 'B'] should be 'A' or 'B' (first occurrence)
        filled_value = result["categorical_col"].iloc[4]
        assert filled_value in ["A", "B"]
        assert not result["categorical_col"].isna().any()

    def test_encode_categorical_onehot(self):
        result = self.processor.encode_categorical(
            self.df, columns=["text_col"], method="onehot"
        )

        # Check that original column is removed and new columns are added
        assert "text_col" not in result.columns
        expected_cols = ["text_col_bird", "text_col_cat", "text_col_dog"]
        for col in expected_cols:
            assert col in result.columns

        # Check that encoding information is stored
        assert "text_col" in self.processor.encoders
        assert self.processor.encoders["text_col"]["method"] == "onehot"

    def test_encode_categorical_label(self):
        result = self.processor.encode_categorical(
            self.df, columns=["text_col"], method="label"
        )

        # Check that column still exists but values are encoded
        assert "text_col" in result.columns
        assert result["text_col"].dtype in ["int64", "int32"]

        # Check that encoding information is stored
        assert "text_col" in self.processor.encoders
        assert self.processor.encoders["text_col"]["method"] == "label"
        assert "mapping" in self.processor.encoders["text_col"]

    def test_scale_features_standard(self):
        # Create dataframe with numeric columns only
        df_numeric = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]}
        )

        result = self.processor.scale_features(
            df_numeric, columns=["col1", "col2"], method="standard"
        )

        # Check that columns are standardized (mean ≈ 0, std ≈ 1)
        assert abs(result["col1"].mean()) < 1e-10
        assert abs(result["col1"].std() - 1.0) < 0.2  # More tolerant for small datasets
        assert abs(result["col2"].mean()) < 1e-10
        assert abs(result["col2"].std() - 1.0) < 0.2

        # Check that scalers are stored
        assert "col1" in self.processor.scalers
        assert "col2" in self.processor.scalers

    def test_scale_features_minmax(self):
        df_numeric = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "col2": [10, 20, 30, 40, 50]}
        )

        result = self.processor.scale_features(
            df_numeric, columns=["col1"], method="minmax"
        )

        # Check that column is scaled to [0, 1] range
        assert result["col1"].min() == 0.0
        assert result["col1"].max() == 1.0

        # Check that scaler is stored
        assert "col1" in self.processor.scalers

    def test_remove_outliers_iqr(self):
        # Create data with outliers
        df_with_outliers = pd.DataFrame(
            {"values": [1, 2, 3, 4, 5, 100]}  # 100 is an outlier
        )

        result = self.processor.remove_outliers(
            df_with_outliers, columns=["values"], method="iqr"
        )

        # Check that outlier is removed
        assert len(result) < len(df_with_outliers)
        assert 100 not in result["values"].values

    def test_remove_outliers_z_score(self):
        # Create data with outliers
        df_with_outliers = pd.DataFrame(
            {"values": [1, 2, 3, 4, 5, 100]}  # 100 is an outlier
        )

        result = self.processor.remove_outliers(
            df_with_outliers, columns=["values"], method="z_score", threshold=2.0
        )

        # Check that outlier is removed
        assert len(result) < len(df_with_outliers)
        assert 100 not in result["values"].values

    def test_invalid_columns(self):
        # Test with non-existent columns
        result = self.processor.handle_missing_values(
            self.df, columns=["nonexistent_col"]
        )

        # Should return original dataframe unchanged
        pd.testing.assert_frame_equal(result, self.df)


class TestDatasetCreation:
    def test_create_sample_dataset(self):
        X, y = create_sample_dataset(n_samples=100, n_features=5, random_state=42)

        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_create_sample_dataset_reproducible(self):
        X1, y1 = create_sample_dataset(n_samples=50, n_features=3, random_state=42)
        X2, y2 = create_sample_dataset(n_samples=50, n_features=3, random_state=42)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_create_classification_dataset(self):
        X, y = create_classification_dataset(
            n_samples=100, n_features=3, n_classes=4, random_state=42
        )

        assert X.shape == (100, 3)
        assert y.shape == (100,)
        assert len(np.unique(y)) <= 4  # Should have at most n_classes unique values
        assert np.min(y) >= 0
        assert np.max(y) < 4

    def test_create_classification_dataset_reproducible(self):
        X1, y1 = create_classification_dataset(
            n_samples=50, n_classes=2, random_state=42
        )
        X2, y2 = create_classification_dataset(
            n_samples=50, n_classes=2, random_state=42
        )

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_generate_time_series(self):
        ts = generate_time_series(n_points=100, random_state=42)

        assert len(ts) == 100
        assert isinstance(ts, np.ndarray)

        # Check that time series has some variation (not constant)
        assert np.std(ts) > 0

    def test_generate_time_series_components(self):
        # Test with strong trend and no noise
        ts_trend = generate_time_series(
            n_points=100, trend=0.1, seasonality=0, noise=0, random_state=42
        )

        # Should be monotonically increasing
        assert np.all(np.diff(ts_trend) > 0)

        # Test with seasonality only
        ts_seasonal = generate_time_series(
            n_points=24, trend=0, seasonality=1, noise=0, random_state=42
        )

        # Should have some periodic behavior (not monotonic)
        assert not np.all(np.diff(ts_seasonal) > 0)
        assert not np.all(np.diff(ts_seasonal) < 0)

    def test_generate_time_series_reproducible(self):
        ts1 = generate_time_series(n_points=50, random_state=42)
        ts2 = generate_time_series(n_points=50, random_state=42)

        np.testing.assert_array_equal(ts1, ts2)


class TestDataProcessorIntegration:
    def test_full_preprocessing_pipeline(self):
        # Create a more realistic dataset
        df = pd.DataFrame(
            {
                "age": [25, 30, np.nan, 40, 35, 28, 45, np.nan, 32, 29],
                "income": [
                    50000,
                    60000,
                    55000,
                    80000,
                    np.nan,
                    52000,
                    90000,
                    65000,
                    58000,
                    54000,
                ],
                "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
                "city": [
                    "NYC",
                    "LA",
                    "NYC",
                    "Chicago",
                    "LA",
                    "NYC",
                    "Chicago",
                    "LA",
                    "NYC",
                    "Chicago",
                ],
            }
        )

        processor = DataProcessor()

        # Step 1: Handle missing values
        df_filled = processor.handle_missing_values(
            df, strategy="mean", columns=["age", "income"]
        )
        assert not df_filled[["age", "income"]].isna().any().any()

        # Step 2: Encode categorical variables
        df_encoded = processor.encode_categorical(
            df_filled, columns=["category", "city"], method="onehot"
        )

        # Check that categorical columns are encoded
        assert "category" not in df_encoded.columns
        assert "city" not in df_encoded.columns
        categorical_cols = [
            col for col in df_encoded.columns if col.startswith(("category_", "city_"))
        ]
        assert len(categorical_cols) > 0

        # Step 3: Scale numerical features
        numerical_cols = ["age", "income"]
        df_scaled = processor.scale_features(
            df_encoded, columns=numerical_cols, method="standard"
        )

        # Check that numerical columns are scaled
        for col in numerical_cols:
            if col in df_scaled.columns:
                assert abs(df_scaled[col].mean()) < 1e-10
                assert (
                    abs(df_scaled[col].std() - 1.0) < 0.2
                )  # More tolerant for small datasets

        # Final checks
        assert len(df_scaled) == len(df)  # Should not lose any rows in this pipeline
        assert not df_scaled.isna().any().any()  # Should have no missing values


DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
