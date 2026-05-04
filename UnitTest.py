import numpy as np

class TestMethods():
    """
        Description: This class contains unit test for all cells in the notebook.
        Author: Tim Liu
    """

    def test_load_data(dataSet):
        assert dataSet is not None, "Data set should not be None"
        assert dataSet.shape[0] > 0, "Data set should have at least one row"
        assert dataSet.shape[1] > 0, "Data set should have at least one column"
        assert not dataSet.isnull().values.any(), "Data set should not contain null values"

    def test_ordinal_encoder(dataset):
        assert dataset is not None, "Data set should not be None"
        for col in dataset.dtypes.values:
            assert col in ['int64', 'float64'], "All columns should be numeric after encoding"

    def test_target_class_imbalance(target_column):
        assert target_column is not None, "Target column should not be None"
        target_series = target_column.value_counts()
        assert target_series.min() > 0, "Each class should have at least one instance"
        assert target_series.max() / target_series.min() == 1, "Classes should be balanced"

    def test_feature_Identification(target_x, target_y):
        assert target_x is not None, "Feature set should not be None"
        assert target_y is not None, "Target variable should not be None"
        assert target_x.shape[0] == target_y.shape[0], "Number of samples in features and target should be the same"

    def test_feature_generation(dataset):
        assert dataset is not None, "Data set should not be None"
        assert dataset.shape[0] == 98990, "The amount of rows in the dataset should not have changed"
        assert dataset.shape[1] > 17, "Data set should have more than 17 columns"
    
    def test_feature_Selection(dataset):
        assert dataset is not None, "Data set should not be None"
        assert dataset.shape[0] == 98990, "The amount of rows in the dataset should not have changed"
        assert dataset.shape[1] < 17, "Data set should have less than 17 columns"

    def test_feature_normalization(Stand_X_train, Stand_Y_train, Stand_X_test, Stand_Y_test):
        assert Stand_X_train is not None, "Standardized training features should not be None"
        assert Stand_Y_train is not None, "Standardized training target should not be None"
        assert Stand_X_test is not None, "Standardized testing features should not be None"
        assert Stand_Y_test is not None, "Standardized testing target should not be None"
        assert np.all((Stand_X_train >= 0) & (Stand_X_train <= 1)), "All training features should be normalized between 0 and 1"
        assert np.all((Stand_X_test >= 0) & (Stand_X_test <= 1)), "All testing features should be normalized between 0 and 1"
        assert np.all((Stand_Y_train >= 0) & (Stand_Y_train <= 1)), "All training target should be normalized between 0 and 1"
        assert np.all((Stand_Y_test >= 0) & (Stand_Y_test <= 1)), "All testing target should be normalized between 0 and 1"

    
    
    