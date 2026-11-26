from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy import optimize

from color_tracker.utils.tracker_object import TrackedObject


def crop_out_polygon_convex(image: np.ndarray, point_array: np.ndarray) -> np.ndarray:
    """
    Crops out a convex polygon given from a list of points from an image
    :param image: Opencv BGR image
    :param point_array: list of points that defines a convex polygon
    :return: Cropped out image
    """

    point_array = np.reshape(cv2.convexHull(point_array), point_array.shape)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    roi_corners = np.array([point_array], dtype=np.int32)
    cv2.fillConvexPoly(mask, roi_corners, 255)
    return cv2.bitwise_and(image, image, mask=mask)


def resize_img(image: np.ndarray, min_width: int, min_height: int) -> np.ndarray:
    """
    Resize the image with keeping the aspect ratio.
    :param image: image
    :param min_width: minimum width of the image
    :param min_height: minimum height of the image
    :return: resized image
    """

    if min_width <= 0 or min_height <= 0:
        raise ValueError("min_width and min_height must be positive")
    height, width = image.shape[:2]
    scale = min(1.0, min_width / width, min_height / height)
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size)


def sort_contours_by_area(
    contours: List[np.ndarray], descending: bool = True
) -> List[np.ndarray]:
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=descending)
    return contours


def filter_contours_by_area(
    contours: List[np.ndarray],
    min_area: float = 0,
    max_area: float = np.inf,
) -> List[np.ndarray]:
    if len(contours) == 0:
        return []

    def _keep_contour(c):
        area = cv2.contourArea(c)
        if area <= min_area:
            return False
        if area >= max_area:
            return False
        return True

    return list(filter(_keep_contour, contours))


def get_contour_centers(contours: List[np.ndarray]) -> np.ndarray:
    """
    Calculate the centers of the contours
    :param contours: Contours detected with find_contours
    :return: object centers as numpy array
    """

    centers = []
    for contour in contours:
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centers.append(
            (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        )
    return np.asarray(centers, dtype=np.int32).reshape((-1, 2))


def find_object_contours(
    image: np.ndarray,
    hsv_lower_value: Sequence[int],
    hsv_upper_value: Sequence[int],
    kernel: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, tuple(hsv_lower_value), tuple(hsv_upper_value))
    if kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def get_bbox_for_contours(contours: List[np.ndarray]) -> np.ndarray:
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return np.asarray(bboxes, dtype=np.int32).reshape((-1, 4))


def calculate_distance_mtx(
    tracked_objects: List[TrackedObject], points: np.ndarray
) -> np.ndarray:
    # (nb_tracked_objects, nb_current_detected_points)
    if not tracked_objects or len(points) == 0:
        return np.zeros((len(tracked_objects), len(points)), dtype=float)
    previous_points = np.asarray(
        [tracked_object.last_point for tracked_object in tracked_objects], dtype=float
    )
    current_points = np.asarray(points, dtype=float)
    differences = previous_points[:, np.newaxis, :] - current_points[np.newaxis, :, :]
    return np.linalg.norm(differences, axis=2)


def solve_assignment(cost_mtx: np.ndarray) -> List[int]:
    if cost_mtx.ndim != 2:
        raise ValueError("cost_mtx must be a two-dimensional array")
    nb_tracked_objects, _ = cost_mtx.shape
    assignment = [-1] * nb_tracked_objects
    if cost_mtx.size == 0:
        return assignment
    row_index, column_index = optimize.linear_sum_assignment(cost_mtx)
    for i in range(len(row_index)):
        assignment[row_index[i]] = column_index[i]
    return assignment


def remove_object_if_too_many_frames_skipped(
    tracked_objects: List[TrackedObject],
    assignment: List[int],
    max_skipped_frames: int,
) -> None:
    if len(tracked_objects) != len(assignment):
        raise ValueError("tracked_objects and assignment must have equal lengths")
    retained = [
        (tracked_object, assigned_detection)
        for tracked_object, assigned_detection in zip(tracked_objects, assignment)
        if tracked_object.skipped_frames <= max_skipped_frames
    ]
    tracked_objects[:] = [tracked_object for tracked_object, _ in retained]
    assignment[:] = [assigned_detection for _, assigned_detection in retained]
