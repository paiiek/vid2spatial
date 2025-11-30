"""
Unit tests for vision.py module - FIXED VERSION.
Tests core camera geometry and tracking functions.

Coordinate system (from actual implementation):
- pixel_to_ray outputs [x, y, z] where z=1 points forward (into scene)
- ray_to_angles: az = atan2(y, x), el = atan2(z, sqrt(x^2+y^2))
- This means:
  - x axis: left (-) to right (+) in image
  - y axis: top (-) to bottom (+) in image
  - z axis: depth (forward into scene)
  - Center pixel (cx, cy) → ray [0, 0, 1] → az=0°, el=90°
"""
import pytest
import numpy as np
import math
from mmhoa.vid2spatial.vision import (
    CameraIntrinsics,
    pixel_to_ray,
    ray_to_angles,
)


class TestCameraIntrinsics:
    """Test CameraIntrinsics dataclass and its computed properties."""

    def test_camera_intrinsics_basic(self):
        """Test basic camera intrinsics initialization."""
        K = CameraIntrinsics(width=1920, height=1080, fov_deg=60.0)
        assert K.width == 1920
        assert K.height == 1080
        assert K.fov_deg == 60.0

    def test_principal_point_center(self):
        """Test that principal point is at image center."""
        K = CameraIntrinsics(width=1920, height=1080, fov_deg=60.0)
        assert K.cx == 960.0
        assert K.cy == 540.0

    def test_focal_length_calculation(self):
        """Test focal length calculation from FOV."""
        K = CameraIntrinsics(width=1920, height=1080, fov_deg=60.0)
        # For 60° FOV: f = 0.5 * W / tan(FOV/2)
        expected_fx = 0.5 * 1920 / math.tan(math.radians(60.0) / 2.0)
        assert abs(K.fx - expected_fx) < 1e-6
        # Square pixels assumption
        assert abs(K.fy - K.fx) < 1e-6

    def test_different_fov(self):
        """Test focal length changes with different FOV."""
        K_wide = CameraIntrinsics(width=1920, height=1080, fov_deg=90.0)
        K_narrow = CameraIntrinsics(width=1920, height=1080, fov_deg=30.0)
        # Wider FOV should have smaller focal length
        assert K_wide.fx < K_narrow.fx


class TestPixelToRay:
    """Test pixel to 3D ray conversion."""

    def test_center_pixel_points_forward(self, sample_camera_intrinsics):
        """Test that center pixel produces forward ray [0, 0, 1]."""
        K = sample_camera_intrinsics
        ray = pixel_to_ray(K.cx, K.cy, K)
        # Should be normalized [0, 0, 1]
        np.testing.assert_allclose(ray, [0, 0, 1], atol=1e-6)

    def test_ray_normalization(self, sample_camera_intrinsics):
        """Test that output ray is normalized."""
        K = sample_camera_intrinsics
        # Test arbitrary pixel
        ray = pixel_to_ray(100, 200, K)
        norm = np.linalg.norm(ray)
        assert abs(norm - 1.0) < 1e-6

    def test_right_pixel_positive_x(self, sample_camera_intrinsics):
        """Test that pixel to the right has positive x component."""
        K = sample_camera_intrinsics
        # Pixel to the right of center
        ray = pixel_to_ray(K.cx + 100, K.cy, K)
        assert ray[0] > 0  # x component positive (right)

    def test_left_pixel_negative_x(self, sample_camera_intrinsics):
        """Test that pixel to the left has negative x component."""
        K = sample_camera_intrinsics
        # Pixel to the left of center
        ray = pixel_to_ray(K.cx - 100, K.cy, K)
        assert ray[0] < 0  # x component negative (left)

    def test_top_pixel_negative_y(self, sample_camera_intrinsics):
        """Test that pixel at top has negative y component."""
        K = sample_camera_intrinsics
        # Pixel above center
        ray = pixel_to_ray(K.cx, K.cy - 100, K)
        assert ray[1] < 0  # y component negative (up)

    def test_bottom_pixel_positive_y(self, sample_camera_intrinsics):
        """Test that pixel at bottom has positive y component."""
        K = sample_camera_intrinsics
        # Pixel below center
        ray = pixel_to_ray(K.cx, K.cy + 100, K)
        assert ray[1] > 0  # y component positive (down)


class TestRayToAngles:
    """Test 3D ray to spherical angles conversion."""

    def test_center_ray_elevation_90(self):
        """Test that center ray [0,0,1] gives el=90° (straight up in z)."""
        ray = np.array([0, 0, 1], dtype=np.float32)
        az, el = ray_to_angles(ray)
        # az can be anything when x=y=0, elevation should be 90°
        assert abs(el - math.pi/2) < 1e-6

    def test_right_ray_positive_azimuth(self):
        """Test that ray pointing right has positive azimuth."""
        # Ray with positive x, zero y: atan2(0, +x) = 0
        ray = np.array([1.0, 0.0, 0.1], dtype=np.float32)
        ray = ray / np.linalg.norm(ray)
        az, el = ray_to_angles(ray)
        assert abs(az) < 1e-6  # Should be near 0

    def test_left_ray_negative_or_pi(self):
        """Test that ray pointing left has az near ±180°."""
        # Ray with negative x, zero y: atan2(0, -x) = ±π
        ray = np.array([-1.0, 0.0, 0.1], dtype=np.float32)
        ray = ray / np.linalg.norm(ray)
        az, el = ray_to_angles(ray)
        # Should be near ±π
        assert abs(abs(az) - math.pi) < 1e-6

    def test_forward_horizontal_ray(self):
        """Test ray in horizontal plane (z=0)."""
        ray = np.array([1, 0, 0], dtype=np.float32)
        az, el = ray_to_angles(ray)
        assert abs(az) < 1e-6  # az=0
        assert abs(el) < 1e-6  # el=0 (horizontal)

    def test_known_angle_45deg_azimuth(self):
        """Test conversion for known 45° azimuth in xy plane."""
        # Ray at 45° in xy plane: y/x = 1
        ray = np.array([1, 1, 0], dtype=np.float32)
        ray = ray / np.linalg.norm(ray)
        az, el = ray_to_angles(ray)
        # atan2(1, 1) = π/4 = 45°
        assert abs(az - math.pi / 4) < 1e-6
        assert abs(el) < 1e-6  # Horizontal


class TestPixelToAnglesPipeline:
    """Test the full pipeline from pixel to angles."""

    def test_center_pixel_to_angles(self, sample_camera_intrinsics):
        """Test center pixel → ray → angles."""
        K = sample_camera_intrinsics
        ray = pixel_to_ray(K.cx, K.cy, K)
        az, el = ray_to_angles(ray)
        # Center ray is [0, 0, 1] → el=90°
        assert abs(el - math.pi/2) < 1e-6

    def test_horizontal_center_vs_edge(self, sample_camera_intrinsics):
        """Test that pixels at same vertical position have similar elevation."""
        K = sample_camera_intrinsics
        # All at cy → similar y component → similar elevation
        ray_center = pixel_to_ray(K.cx, K.cy, K)
        ray_right = pixel_to_ray(K.cx + 200, K.cy, K)
        ray_left = pixel_to_ray(K.cx - 200, K.cy, K)

        az_c, el_c = ray_to_angles(ray_center)
        az_r, el_r = ray_to_angles(ray_right)
        az_l, el_l = ray_to_angles(ray_left)

        # Center has highest elevation (most z)
        # Edges have lower elevation as they point more sideways
        assert el_c > el_r
        assert el_c > el_l


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_pixel_to_ray_float_coordinates(self, sample_camera_intrinsics):
        """Test that pixel_to_ray accepts float coordinates."""
        K = sample_camera_intrinsics
        ray = pixel_to_ray(100.5, 200.7, K)
        assert ray.shape == (3,)
        assert abs(np.linalg.norm(ray) - 1.0) < 1e-6

    def test_ray_to_angles_zero_vector(self):
        """Test ray_to_angles with near-zero vector."""
        ray = np.array([1e-10, 1e-10, 1e-10], dtype=np.float32)
        az, el = ray_to_angles(ray)
        # Should not crash
        assert isinstance(az, float)
        assert isinstance(el, float)

    def test_ray_to_angles_unit_vectors(self):
        """Test ray_to_angles with unit axis vectors."""
        # X axis: [1, 0, 0]
        ray = np.array([1, 0, 0], dtype=np.float32)
        az, el = ray_to_angles(ray)
        assert abs(az) < 1e-6  # atan2(0, 1) = 0
        assert abs(el) < 1e-6  # atan2(0, 1) = 0

        # Y axis: [0, 1, 0]
        ray = np.array([0, 1, 0], dtype=np.float32)
        az, el = ray_to_angles(ray)
        assert abs(az - math.pi / 2) < 1e-6  # atan2(1, 0) = π/2

        # Z axis: [0, 0, 1]
        ray = np.array([0, 0, 1], dtype=np.float32)
        az, el = ray_to_angles(ray)
        assert abs(el - math.pi / 2) < 1e-6  # Pointing straight up


class TestCameraGeometryConsistency:
    """Test consistency of camera geometry calculations."""

    def test_reciprocal_projection(self, sample_camera_intrinsics):
        """Test that ray → angles → ray is consistent."""
        test_rays = [
            [1, 0, 0],   # Right
            [0, 1, 0],   # Down
            [0, 0, 1],   # Forward/up
            [1, 1, 0],   # Diagonal
            [0.5, 0.3, 0.8],  # Arbitrary
        ]

        for ray_vec in test_rays:
            ray = np.array(ray_vec, dtype=np.float32)
            ray = ray / (np.linalg.norm(ray) + 1e-9)

            az, el = ray_to_angles(ray)

            # Reconstruct ray from angles
            # Standard spherical: x = cos(az)cos(el), y = sin(az)cos(el), z = sin(el)
            x = math.cos(az) * math.cos(el)
            y = math.sin(az) * math.cos(el)
            z = math.sin(el)
            ray_reconstructed = np.array([x, y, z], dtype=np.float32)

            # Should match original ray
            np.testing.assert_allclose(ray, ray_reconstructed, atol=1e-6)

    def test_symmetry_left_right(self, sample_camera_intrinsics):
        """Test that left/right pixels are symmetric in x."""
        K = sample_camera_intrinsics
        offset = 200

        ray_left = pixel_to_ray(K.cx - offset, K.cy, K)
        ray_right = pixel_to_ray(K.cx + offset, K.cy, K)

        # X components should be opposite
        assert abs(ray_left[0] + ray_right[0]) < 1e-6
        # Y components should be same (both at cy)
        assert abs(ray_left[1] - ray_right[1]) < 1e-6
        # Z components should be same
        assert abs(ray_left[2] - ray_right[2]) < 1e-6

    def test_symmetry_up_down(self, sample_camera_intrinsics):
        """Test that up/down pixels are symmetric in y."""
        K = sample_camera_intrinsics
        offset = 100

        ray_up = pixel_to_ray(K.cx, K.cy - offset, K)
        ray_down = pixel_to_ray(K.cx, K.cy + offset, K)

        # X components should be same (both at cx)
        assert abs(ray_up[0] - ray_down[0]) < 1e-6
        # Y components should be opposite
        assert abs(ray_up[1] + ray_down[1]) < 1e-6
        # Z components should be same
        assert abs(ray_up[2] - ray_down[2]) < 1e-6
