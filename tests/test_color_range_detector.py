import unittest
from unittest.mock import patch

from color_tracker.utils.color_range_detector import HSVColorRangeDetector


class FailedCamera:
    def __init__(self):
        self.released = False

    def read(self):
        return False, None

    def release(self):
        self.released = True


class TestColorRangeDetector(unittest.TestCase):
    def test_camera_and_windows_are_released_when_frame_read_fails(self):
        camera = FailedCamera()
        with patch("cv2.namedWindow"), patch("cv2.createTrackbar"), patch(
            "cv2.destroyAllWindows"
        ) as destroy_windows:
            detector = HSVColorRangeDetector(camera)

            with self.assertRaisesRegex(RuntimeError, "There is no camera feed"):
                detector.detect()

        self.assertTrue(camera.released)
        destroy_windows.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
