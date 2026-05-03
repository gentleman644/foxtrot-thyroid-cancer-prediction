

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
    