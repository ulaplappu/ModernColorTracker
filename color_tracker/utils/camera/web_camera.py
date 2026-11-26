import cv2
from typing import Union

from color_tracker.utils.camera.base_camera import Camera


class WebCamera(Camera):
    """
    Simple Webcamera
    """

    def __init__(self, video_src: Union[int, str] = 0, start: bool = False):
        """
        :param video_src (int): camera source code. It can be an integer or the name of the video file.
        """

        super().__init__()
        self._video_src = video_src

        if start:
            self.start_camera()

    def _init_camera(self) -> None:
        self._cam = cv2.VideoCapture(self._video_src)
        if not self._cam.isOpened():
            self._release_camera()
            raise RuntimeError("Unable to open the camera source")
        ret, frame = self._cam.read()
        if not ret or frame is None:
            self._release_camera()
            raise RuntimeError("No camera feed")
        with self._frame_lock:
            self._ret, self._frame = ret, frame
            self._frame_height, self._frame_width = frame.shape[:2]

    def _read_from_camera(self):
        if self._cam is None:
            raise RuntimeError("Camera is not initialized")
        ret, frame = self._cam.read()
        if ret and frame is not None:
            with self._frame_lock:
                auto_undistortion = self._auto_undistortion
            if auto_undistortion:
                frame = self._undistort_image(frame)
            return True, frame
        return False, None

    def _release_camera(self) -> None:
        if self._cam is not None:
            self._cam.release()
            self._cam = None
