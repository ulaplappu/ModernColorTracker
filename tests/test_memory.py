import unittest
from color_tracker.utils.tracker_object import TrackedObject
from color_tracker import ColorTracker


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

    def test_default_history_limit(self):
        obj = TrackedObject(0)

        for i in range(150):
            obj.add_point((i, i))

        self.assertEqual(len(obj.tracked_points), 120)
        self.assertEqual(obj.tracked_points[0], (30, 30))

    def test_tracker_default_history_limit_is_120(self):
        tracker = ColorTracker(debug=False)
        tracker._init_new_tracked_object((0, 0))

        for i in range(150):
            tracker.tracked_objects[0].add_point((i, i))

        self.assertEqual(len(tracker.tracked_objects[0].tracked_points), 120)

    def test_none_keeps_unbounded_history(self):
        obj = TrackedObject(0, max_nb_of_points=None)

        for i in range(150):
            obj.add_point((i, i))

        self.assertEqual(len(obj.tracked_points), 150)

    def test_non_positive_history_limit_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    TrackedObject(0, max_nb_of_points=limit)


if __name__ == "__main__":
    unittest.main()