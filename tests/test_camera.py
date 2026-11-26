import threading
import unittest
from unittest.mock import patch

from color_tracker.utils.camera.base_camera import Camera
from color_tracker.utils.camera.web_camera import WebCamera


class BlockingCamera(Camera):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()

    def _init_camera(self):
        self._cam = object()

    def _read_from_camera(self):
        self.started.set()
        self._stop_event.wait()
        return False, None

    def _release_camera(self):
        self._cam = None


class FailingCamera(BlockingCamera):
    def _read_from_camera(self):
        self.started.set()
        raise OSError("synthetic capture failure")


class StuckCamera(BlockingCamera):
    _shutdown_timeout_seconds = 0.05

    def __init__(self):
        super().__init__()
        self.allow_read_to_finish = threading.Event()
        self.reading = False
        self.release_count = 0
        self.released_during_read = False

    def _read_from_camera(self):
        self.reading = True
        self.started.set()
        self.allow_read_to_finish.wait()
        self.reading = False
        return False, None

    def _release_camera(self):
        self.released_during_read = self.reading
        self.release_count += 1
        self._cam = None


class RetryReleaseCamera(BlockingCamera):
    def __init__(self):
        super().__init__()
        self.release_count = 0

    def _release_camera(self):
        if self._cam is None:
            return
        self.release_count += 1
        if self.release_count == 1:
            raise OSError("synthetic release failure")
        self._cam = None


class ReusableCamera(BlockingCamera):
    def __init__(self):
        super().__init__()
        self.init_count = 0
        self.release_count = 0

    def _init_camera(self):
        self.init_count += 1
        self.started.clear()
        self._cam = object()

    def _release_camera(self):
        if self._cam is not None:
            self.release_count += 1
            self._cam = None


class TestCamera(unittest.TestCase):
    def test_release_stops_and_joins_capture_thread(self):
        camera = BlockingCamera()
        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))

        camera.release()

        self.assertFalse(camera.is_running())
        self.assertIsNone(camera._camera_thread)

    def test_shutdown_timeout_preserves_resources_for_safe_retry(self):
        camera = StuckCamera()
        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))

        with self.assertRaisesRegex(RuntimeError, "did not stop within"):
            camera.release()

        self.assertEqual(camera.release_count, 0)
        self.assertIsNotNone(camera._cam)
        self.assertTrue(camera.is_running())
        with self.assertRaisesRegex(RuntimeError, "still shutting down"):
            camera.start_camera()

        camera.allow_read_to_finish.set()
        camera._camera_thread.join(timeout=2)
        camera.release()

        self.assertEqual(camera.release_count, 1)
        self.assertFalse(camera.released_during_read)
        self.assertIsNone(camera._cam)
        self.assertIsNone(camera._camera_thread)

    def test_release_failure_is_surfaced_and_can_be_retried(self):
        camera = RetryReleaseCamera()
        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))

        with self.assertRaisesRegex(OSError, "synthetic release failure"):
            camera.release()

        self.assertIsNotNone(camera._camera_thread)
        self.assertIsNotNone(camera._cam)

        camera.release()

        self.assertEqual(camera.release_count, 2)
        self.assertIsNone(camera._cam)
        self.assertIsNone(camera._camera_thread)

    def test_repeated_start_and_release_are_idempotent(self):
        camera = ReusableCamera()
        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))
        camera.start_camera()
        self.assertEqual(camera.init_count, 1)

        camera.release()
        camera.release()
        self.assertEqual(camera.release_count, 1)

        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))
        camera.release()

        self.assertEqual(camera.init_count, 2)
        self.assertEqual(camera.release_count, 2)
        self.assertIsNone(camera._cam)

    def test_capture_thread_error_is_reported_by_read(self):
        camera = FailingCamera()
        camera.start_camera()
        self.assertTrue(camera.started.wait(timeout=2))
        camera._camera_thread.join(timeout=2)

        with self.assertRaisesRegex(RuntimeError, "Camera frame capture failed"):
            camera.read()

        camera.release()

    def test_web_camera_open_failure_releases_capture(self):
        class ClosedCapture:
            released = False

            def isOpened(self):
                return False

            def release(self):
                self.released = True

        capture = ClosedCapture()
        camera = WebCamera()
        with patch("cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(RuntimeError, "Unable to open"):
                camera.start_camera()

        self.assertTrue(capture.released)
        self.assertIsNone(camera._cam)


if __name__ == "__main__":
    unittest.main()
