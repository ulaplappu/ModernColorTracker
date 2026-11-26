import collections
from typing import Deque, Optional, Sequence, Tuple

import numpy as np


DEFAULT_MAX_NB_OF_POINTS = 120


class TrackedObject:
    def __init__(
        self,
        id: int,
        max_nb_of_points: Optional[int] = DEFAULT_MAX_NB_OF_POINTS,
    ) -> None:
        if max_nb_of_points is not None and (
            not isinstance(max_nb_of_points, int)
            or isinstance(max_nb_of_points, bool)
            or max_nb_of_points <= 0
        ):
            raise ValueError("max_nb_of_points must be a positive integer or None")

        self._id = id
        self._tracked_points: Deque[Tuple[int, int]] = collections.deque(
            maxlen=max_nb_of_points
        )
        self._skipped_frames = 0

        self._last_object_contour: Optional[np.ndarray] = None
        self._last_bounding_box: Optional[Tuple[int, int, int, int]] = None

    @property
    def id(self) -> int:
        return self._id

    @property
    def skipped_frames(self) -> int:
        return self._skipped_frames

    @skipped_frames.setter
    def skipped_frames(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError("skipped_frames must be a non-negative integer")
        self._skipped_frames = value

    @property
    def tracked_points(self) -> Deque[Tuple[int, int]]:
        return self._tracked_points

    @property
    def last_point(self) -> Tuple[int, int]:
        return self.tracked_points[-1]

    @property
    def last_object_contour(self) -> Optional[np.ndarray]:
        return self._last_object_contour

    @last_object_contour.setter
    def last_object_contour(self, value: Optional[np.ndarray]) -> None:
        self._last_object_contour = value

    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._last_bounding_box

    @last_bbox.setter
    def last_bbox(self, value: Optional[Tuple[int, int, int, int]]) -> None:
        self._last_bounding_box = value

    def add_point(self, point: Sequence[int]) -> None:
        if len(point) != 2:
            raise ValueError("point must contain exactly two coordinates")
        self._tracked_points.append((int(point[0]), int(point[1])))
