import numpy as np
import unittest

from color_tracker.utils import helpers


class TestHelpers(unittest.TestCase):

    def test_empty_contours(self):
        result = helpers.filter_contours_by_area([])
        self.assertEqual(len(result), 0)

    def test_single_contour(self):
        contour = np.array(
            [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]],
            dtype=np.int32
        )

        result = helpers.filter_contours_by_area([contour])
        self.assertEqual(len(result), 1)

    def test_multiple_contours(self):
        rectangle = np.array(
            [[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]], dtype=np.int32
        )
        triangle = np.array(
            [[[0, 0]], [[20, 0]], [[10, 20]]], dtype=np.int32
        )

        result = helpers.filter_contours_by_area([rectangle, triangle])

        self.assertEqual(len(result), 2)

    def test_crop_polygon_preserves_image_shape_and_masks_outside(self):
        image = np.full((20, 20, 3), 255, dtype=np.uint8)
        points = np.array([[5, 5], [14, 5], [14, 14], [5, 14]], dtype=np.int32)

        cropped = helpers.crop_out_polygon_convex(image, points)

        self.assertEqual(cropped.shape, image.shape)
        np.testing.assert_array_equal(cropped[10, 10], [255, 255, 255])
        np.testing.assert_array_equal(cropped[0, 0], [0, 0, 0])

    def test_resize_img_caps_dimensions_without_upscaling(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        resized = helpers.resize_img(image, min_width=80, min_height=60)
        unchanged = helpers.resize_img(image, min_width=300, min_height=200)

        self.assertEqual(resized.shape[:2], (40, 80))
        np.testing.assert_array_equal(unchanged, image)

    def test_resize_img_rejects_non_positive_dimensions(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)

        with self.assertRaises(ValueError):
            helpers.resize_img(image, min_width=0, min_height=10)

    def test_synthetic_image_contour_detection(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        image[10:20, 15:25] = (0, 0, 255)

        contours = helpers.find_object_contours(
            image,
            hsv_lower_value=[0, 200, 200],
            hsv_upper_value=[10, 255, 255],
        )

        self.assertEqual(len(contours), 1)
        np.testing.assert_array_equal(
            helpers.get_contour_centers(contours), np.array([[19, 14]], dtype=np.int32)
        )

    def test_degenerate_contour_has_no_center(self):
        contour = np.array([[[0, 0]], [[1, 1]]], dtype=np.int32)

        centers = helpers.get_contour_centers([contour])

        self.assertEqual(centers.shape, (0, 2))

    def test_empty_assignment_matrix(self):
        assignment = helpers.solve_assignment(np.zeros((3, 0)))

        self.assertEqual(assignment, [-1, -1, -1])

    def test_skipped_objects_are_removed_without_index_shifts(self):
        from color_tracker.utils.tracker_object import TrackedObject

        objects = [TrackedObject(i) for i in range(4)]
        for obj in objects:
            obj.add_point((obj.id, obj.id))
        objects[1].skipped_frames = 2
        objects[3].skipped_frames = 2
        assignment = [0, -1, 2, -1]

        helpers.remove_object_if_too_many_frames_skipped(objects, assignment, 1)

        self.assertEqual([obj.id for obj in objects], [0, 2])
        self.assertEqual(assignment, [0, 2])

if __name__ == "__main__":
    unittest.main()