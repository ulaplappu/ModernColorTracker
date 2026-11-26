import unittest
import numpy as np

from color_tracker.utils import helpers


class TestHelpers(unittest.TestCase):

    def test_empty_contours(self):
        result = helpers.filter_contours_by_area([])
        self.assertEqual(len(result), 0)

    def test_single_contour(self):
        contour = np.array(
            [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]],
            dtype=np.int32
        )

        result = helpers.filter_contours_by_area([contour])
        self.assertEqual(len(result), 1)

    def test_multiple_contours(self):
     rectangle = np.array(
         [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]],
         dtype=np.int32
     )

     triangle = np.array(
         [[[0, 0]], [[20, 0]], [[10, 20]]],
         dtype=np.int32
     )

     result = helpers.filter_contours_by_area(
         [rectangle, triangle]
     )

     self.assertEqual(len(result), 2)

if __name__ == "__main__":
    unittest.main()