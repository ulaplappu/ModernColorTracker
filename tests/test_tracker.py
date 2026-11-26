import unittest
import cv2
import numpy as np

from color_tracker import ColorTracker


class SyntheticCamera:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        return True, next(self.frames)


class TestTracker(unittest.TestCase):

    def test_default_initialization(self):
        tracker = ColorTracker()
        self.assertEqual(tracker.tracked_objects, [])

    def test_default_max_nb_of_objects_is_five(self):
        frame = np.zeros((40, 120, 3), dtype=np.uint8)
        for x in (4, 24, 44, 64, 84, 104):
            cv2.rectangle(frame, (x, 10), (x + 8, 18), (0, 0, 255), -1)

        tracker = ColorTracker(debug=False)
        tracker.set_tracking_callback(lambda current: current.stop_tracking())
        tracker.track(
            SyntheticCamera([frame]),
            hsv_lower_value=[0, 200, 200],
            hsv_upper_value=[10, 255, 255],
            horizontal_flip=False,
        )

        self.assertEqual(len(tracker.tracked_objects), 5)

    def test_custom_object_limit(self):
        tracker = ColorTracker(max_nb_of_objects=10)
        self.assertIsInstance(tracker, ColorTracker)

    def test_zero_object_limit(self):
        with self.assertRaises(ValueError):
            ColorTracker(max_nb_of_objects=0)

    def test_negative_object_limit(self):
        with self.assertRaises(ValueError):
            ColorTracker(max_nb_of_objects=-1)

    def test_tracking_synthetic_frames_updates_each_detection_once(self):
        frames = []
        for x in (10, 12):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.rectangle(frame, (x, 12), (x + 9, 21), (0, 0, 255), -1)
            frames.append(frame)

        tracker = ColorTracker(max_nb_of_objects=2, debug=True)
        callback_count = []

        def stop_after_two_frames(current_tracker):
            callback_count.append(None)
            if len(callback_count) == 2:
                current_tracker.stop_tracking()

        tracker.set_tracking_callback(stop_after_two_frames)
        tracker.track(
            SyntheticCamera(frames),
            hsv_lower_value=[0, 200, 200],
            hsv_upper_value=[10, 255, 255],
            horizontal_flip=False,
        )

        self.assertEqual(len(tracker.tracked_objects), 1)
        self.assertEqual(
            list(tracker.tracked_objects[0].tracked_points), [(14, 16), (16, 16)]
        )
        self.assertEqual(tracker.tracked_objects[0].last_bbox, (12, 12, 22, 22))
        self.assertEqual(tracker.debug_frame.shape, frames[-1].shape)
        self.assertFalse(tracker._is_running)

    def test_multi_object_assignment_tracks_detections_after_area_sorting(self):
        frames = []
        for small_x, large_x in ((10, 50), (12, 48)):
            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            cv2.rectangle(frame, (small_x, 12), (small_x + 9, 21), (0, 0, 255), -1)
            cv2.rectangle(frame, (large_x, 30), (large_x + 13, 43), (0, 0, 255), -1)
            frames.append(frame)

        tracker = ColorTracker(max_nb_of_objects=2, debug=False)
        callback_count = []

        def stop_after_two_frames(current_tracker):
            callback_count.append(None)
            if len(callback_count) == 2:
                current_tracker.stop_tracking()

        tracker.set_tracking_callback(stop_after_two_frames)
        tracker.track(
            SyntheticCamera(frames),
            hsv_lower_value=[0, 200, 200],
            hsv_upper_value=[10, 255, 255],
            horizontal_flip=False,
        )

        self.assertEqual(
            [list(obj.tracked_points) for obj in tracker.tracked_objects],
            [[(56, 36), (54, 36)], [(14, 16), (16, 16)]],
        )

    def test_tracking_stops_cleanly_when_camera_read_fails(self):
        class FailedCamera:
            def read(self):
                return False, None

        tracker = ColorTracker(debug=False)

        with self.assertRaisesRegex(ValueError, "There is no camera feed"):
            tracker.track(
                FailedCamera(),
                hsv_lower_value=[0, 0, 0],
                hsv_upper_value=[255, 255, 255],
            )

        self.assertFalse(tracker._is_running)

    def test_tracking_stops_cleanly_when_callback_fails(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        tracker = ColorTracker(debug=False)

        def fail_callback(_):
            raise RuntimeError("synthetic callback failure")

        tracker.set_tracking_callback(fail_callback)

        with self.assertRaisesRegex(RuntimeError, "synthetic callback failure"):
            tracker.track(
                SyntheticCamera([frame]),
                hsv_lower_value=[0, 0, 0],
                hsv_upper_value=[255, 255, 255],
                horizontal_flip=False,
            )

        self.assertFalse(tracker._is_running)


if __name__ == "__main__":
    unittest.main()