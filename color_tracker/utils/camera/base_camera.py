import threading
import warnings
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera(object):
    """
    Base Camera object
    """

    _shutdown_timeout_seconds = 5.0

    def __init__(self):
        self._cam = None
        self._frame: Optional[np.ndarray] = None
        self._frame_width: Optional[int] = None
        self._frame_height: Optional[int] = None
        self._ret = False

        self._auto_undistortion = False
        self._camera_matrix = None
        self._distortion_coefficients = None

        self._is_running = False
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._camera_thread: Optional[threading.Thread] = None
        self._thread_error: Optional[Exception] = None

    def _init_camera(self) -> None:
        """
        This is the first for creating our camera
        We should override this!
        """
        raise NotImplementedError("Camera subclasses must implement _init_camera")

    def start_camera(self) -> None:
        """
        Start the running of the camera, without this we can't capture frames
        Camera runs on a separate thread so we can reach a higher FPS
        """
        with self._lifecycle_lock:
            if self._camera_thread is not None and self._camera_thread.is_alive():
                if self._stop_event.is_set():
                    raise RuntimeError(
                        "Camera capture is still shutting down; retry release() "
                        "after the capture thread stops"
                    )
                return
            if self._camera_thread is not None or self._cam is not None:
                self._release_camera()
            with self._frame_lock:
                self._camera_thread = None
                self._ret = False
                self._frame = None
                self._thread_error = None
            try:
                self._init_camera()
            except Exception:
                self._release_camera()
                raise
            self._stop_event.clear()
            with self._frame_lock:
                self._is_running = True
                self._camera_thread = threading.Thread(
                    target=self._update_camera,
                    name="color-tracker-camera",
                    daemon=True,
                )
            try:
                self._camera_thread.start()
            except Exception:
                self._stop_event.set()
                self._release_camera()
                with self._frame_lock:
                    self._is_running = False
                    self._camera_thread = None
                raise

    def _read_from_camera(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        This method is responsible for grabbing frames from the camera
        We should override this!
        """
        if self._cam is None:
            raise RuntimeError("Camera is not initialized")
        raise NotImplementedError("Camera subclasses must implement _read_from_camera")

    def _update_camera(self) -> None:
        """
        Grabs the frames from the camera
        """
        try:
            while not self._stop_event.is_set():
                ret, frame = self._read_from_camera()
                with self._frame_lock:
                    self._ret = ret
                    self._frame = frame
                if not ret:
                    break
        except Exception as error:
            with self._frame_lock:
                self._thread_error = error
        finally:
            with self._frame_lock:
                self._is_running = False

    def get_frame_width_and_height(self) -> Tuple[Optional[int], Optional[int]]:
        """
        Returns the width and height of the grabbed images
        :return (int int): width and height
        """
        with self._frame_lock:
            return self._frame_width, self._frame_height

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        With this you can grab the last frame from the camera
        :return (boolean, np.array): return value and frame
        """
        with self._frame_lock:
            error = self._thread_error
            ret, frame = self._ret, self._frame
            camera_thread = self._camera_thread
        if error is not None:
            raise RuntimeError("Camera frame capture failed") from error
        if camera_thread is not None:
            return ret, frame
        warnings.warn(
            "Camera is not started, you should start it with start_camera()",
            RuntimeWarning,
            stacklevel=2,
        )
        return False, None

    def _release_camera(self) -> None:
        """Release resources owned by the camera implementation."""

    def release(self) -> None:
        """
        Stop the camera thread and release the camera resources.

        Raises RuntimeError on timeout without releasing a resource still in use;
        call release() again after the capture thread has stopped.
        """
        with self._lifecycle_lock:
            thread = self._camera_thread
            if thread is threading.current_thread():
                raise RuntimeError("Cannot release a camera from its capture thread")
            self._stop_event.set()
            if thread is not None:
                thread.join(timeout=self._shutdown_timeout_seconds)
                if thread.is_alive():
                    raise RuntimeError(
                        "Camera capture thread did not stop within "
                        f"{self._shutdown_timeout_seconds} seconds; resources remain "
                        "open so release() can be retried safely"
                    )

            self._release_camera()
            with self._frame_lock:
                self._is_running = False
                self._camera_thread = None
                self._ret = False
                self._frame = None
                self._thread_error = None

    def is_running(self) -> bool:
        with self._frame_lock:
            return self._is_running

    def set_calibration_matrices(self, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray) -> None:
        with self._frame_lock:
            self._camera_matrix = camera_matrix
            self._distortion_coefficients = distortion_coefficients

    def activate_auto_undistortion(self) -> None:
        with self._frame_lock:
            self._auto_undistortion = True

    def deactivate_auto_undistortion(self) -> None:
        with self._frame_lock:
            self._auto_undistortion = False

    def _undistort_image(self, image: np.ndarray) -> np.ndarray:
        with self._frame_lock:
            camera_matrix = self._camera_matrix
            distortion_coefficients = self._distortion_coefficients
        if camera_matrix is None or distortion_coefficients is None:
            warnings.warn(
                "Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!",
                RuntimeWarning,
                stacklevel=2,
            )
            return image

        height, width = image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion_coefficients,
            (width, height),
            1,
            (width, height),
        )
        return cv2.undistort(
            image,
            camera_matrix,
            distortion_coefficients,
            None,
            new_camera_matrix,
        )

    def __enter__(self):
        self.start_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
