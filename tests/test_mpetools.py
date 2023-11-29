import unittest
from mpetools import TimeSeriesAnalysis
from sklearn.preprocessing import LabelEncoder

class TestYourClass(unittest.TestCase):
    def test_all_tests_agree(self):
        # Arrange
        instance = TimeSeriesAnalysis()
        results = ['category1', 'category1', 'category1']

        # Act
        result = instance.consistency_check(results)

        # Assert
        self.assertEqual(result, 'category1', 'All tests should agree')

    def test_tests_agree_at_least_min_ratio(self):
        # Arrange
        instance = TimeSeriesAnalysis()
        results = ['category1', 'category1', 'category2', 'category2']

        # Act
        result = instance.consistency_check(results)

        # Assert
        self.assertEqual(result, 'category1', 'Tests should agree at least 60% of the time')

    def test_tests_do_not_agree_at_least_min_ratio(self):
        # Arrange
        instance = TimeSeriesAnalysis()
        results = ['category1', 'category2', 'category3']

        # Act
        result = instance.consistency_check(results)

        # Assert
        self.assertIsNone(result, 'Tests should not agree')

if __name__ == '__main__':
    unittest.main()