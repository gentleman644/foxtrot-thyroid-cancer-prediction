

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
    