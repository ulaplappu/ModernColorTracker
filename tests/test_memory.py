import unittest
from color_tracker.utils.tracker_object import TrackedObject


class TestMemory(unittest.TestCase):

    def test_history_limit(self):
        obj = TrackedObject(0, max_nb_of_points=120)

        for i in range(1000):
            obj.add_point((i, i))

        self.assertEqual(len(obj.tracked_points), 120)

    def test_old_points_removed(self):
        obj = TrackedObject(0, max_nb_of_points=10)

        for i in range(20):
            obj.add_point((i, i))

        self.assertEqual(obj.tracked_points[0], (10, 10))
        self.assertEqual(obj.last_point, (19, 19))


if __name__ == "__main__":
    unittest.main()