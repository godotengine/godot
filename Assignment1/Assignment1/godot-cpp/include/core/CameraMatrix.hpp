#ifndef CAMERA_MATRIX_H
#define CAMERA_MATRIX_H

#include "Defs.hpp"
#include "Plane.hpp"
#include "Rect2.hpp"
#include "Transform.hpp"

#include <vector>

namespace {
using namespace godot;
} // namespace

struct CameraMatrix {

	enum Planes {
		PLANE_NEAR,
		PLANE_FAR,
		PLANE_LEFT,
		PLANE_TOP,
		PLANE_RIGHT,
		PLANE_BOTTOM
	};

	real_t matrix[4][4];

	void set_identity();
	void set_zero();
	void set_light_bias();
	void set_light_atlas_rect(const Rect2 &p_rect);
	void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov = false);
	void set_perspective(real_t p_fovy_degrees, real_t p_aspect, real_t p_z_near, real_t p_z_far, bool p_flip_fov, int p_eye, real_t p_intraocular_dist, real_t p_convergence_dist);
	void set_for_hmd(int p_eye, real_t p_aspect, real_t p_intraocular_dist, real_t p_display_width, real_t p_display_to_lens, real_t p_oversample, real_t p_z_near, real_t p_z_far);
	void set_orthogonal(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_znear, real_t p_zfar);
	void set_orthogonal(real_t p_size, real_t p_aspect, real_t p_znear, real_t p_zfar, bool p_flip_fov = false);
	void set_frustum(real_t p_left, real_t p_right, real_t p_bottom, real_t p_top, real_t p_near, real_t p_far);
	void set_frustum(real_t p_size, real_t p_aspect, Vector2 p_offset, real_t p_near, real_t p_far, bool p_flip_fov = false);

	static real_t get_fovy(real_t p_fovx, real_t p_aspect) {

		return rad2deg(atan(p_aspect * tan(deg2rad(p_fovx) * 0.5)) * 2.0);
	}

	static inline double deg2rad(double p_y) { return p_y * Math_PI / 180.0; }
	static inline float deg2rad(float p_y) { return p_y * Math_PI / 180.0; }

	static inline double rad2deg(double p_y) { return p_y * 180.0 / Math_PI; }
	static inline float rad2deg(float p_y) { return p_y * 180.0 / Math_PI; }

	static inline double absd(double g) {

		union {
			double d;
			uint64_t i;
		} u;
		u.d = g;
		u.i &= (uint64_t)9223372036854775807ll;
		return u.d;
	}

	real_t get_z_far() const;
	real_t get_z_near() const;
	real_t get_aspect() const;
	real_t get_fov() const;
	bool is_orthogonal() const;

	std::vector<Plane> get_projection_planes(const Transform &p_transform) const;

	bool get_endpoints(const Transform &p_transform, Vector3 *p_8points) const;
	Vector2 get_viewport_half_extents() const;

	void invert();
	CameraMatrix inverse() const;

	CameraMatrix operator*(const CameraMatrix &p_matrix) const;

	Plane xform4(const Plane &p_vec4) const;
	inline Vector3 xform(const Vector3 &p_vec3) const;

	operator String() const;

	void scale_translate_to_fit(const AABB &p_aabb);
	void make_scale(const Vector3 &p_scale);
	int get_pixels_per_meter(int p_for_pixel_width) const;
	operator Transform() const;

	CameraMatrix();
	CameraMatrix(const Transform &p_transform);
	~CameraMatrix();
};

Vector3 CameraMatrix::xform(const Vector3 &p_vec3) const {

	Vector3 ret;
	ret.x = matrix[0][0] * p_vec3.x + matrix[1][0] * p_vec3.y + matrix[2][0] * p_vec3.z + matrix[3][0];
	ret.y = matrix[0][1] * p_vec3.x + matrix[1][1] * p_vec3.y + matrix[2][1] * p_vec3.z + matrix[3][1];
	ret.z = matrix[0][2] * p_vec3.x + matrix[1][2] * p_vec3.y + matrix[2][2] * p_vec3.z + matrix[3][2];
	real_t w = matrix[0][3] * p_vec3.x + matrix[1][3] * p_vec3.y + matrix[2][3] * p_vec3.z + matrix[3][3];
	return ret / w;
}

#endif
