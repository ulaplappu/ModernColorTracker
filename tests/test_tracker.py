import unittest
from color_tracker import ColorTracker


class TestTracker(unittest.TestCase):

    def test_default_initialization(self):
        tracker = ColorTracker()
        self.assertEqual(tracker.tracked_objects, [])

    def test_custom_object_limit(self):
        tracker = ColorTracker(max_nb_of_objects=10)
        self.assertIsInstance(tracker, ColorTracker)

    def test_zero_object_limit(self):
        with self.assertRaises(ValueError):
            ColorTracker(max_nb_of_objects=0)

    def test_negative_object_limit(self):
        with self.assertRaises(ValueError):
            ColorTracker(max_nb_of_objects=-1)


if __name__ == "__main__":
    unittest.main()