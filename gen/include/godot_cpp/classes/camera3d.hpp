/**************************************************************************/
/*  camera3d.hpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/node3d.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/plane.hpp>
#include <godot_cpp/variant/projection.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/typed_array.hpp>
#include <godot_cpp/variant/vector2.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class CameraAttributes;
class Compositor;
class Environment;

class Camera3D : public Node3D {
	GDEXTENSION_CLASS(Camera3D, Node3D)

public:
	enum ProjectionType {
		PROJECTION_PERSPECTIVE = 0,
		PROJECTION_ORTHOGONAL = 1,
		PROJECTION_FRUSTUM = 2,
	};

	enum KeepAspect {
		KEEP_WIDTH = 0,
		KEEP_HEIGHT = 1,
	};

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED = 0,
		DOPPLER_TRACKING_IDLE_STEP = 1,
		DOPPLER_TRACKING_PHYSICS_STEP = 2,
	};

	Vector3 project_ray_normal(const Vector2 &p_screen_point) const;
	Vector3 project_local_ray_normal(const Vector2 &p_screen_point) const;
	Vector3 project_ray_origin(const Vector2 &p_screen_point) const;
	Vector2 unproject_position(const Vector3 &p_world_point) const;
	bool is_position_behind(const Vector3 &p_world_point) const;
	Vector3 project_position(const Vector2 &p_screen_point, float p_z_depth) const;
	void set_perspective(float p_fov, float p_z_near, float p_z_far);
	void set_orthogonal(float p_size, float p_z_near, float p_z_far);
	void set_frustum(float p_size, const Vector2 &p_offset, float p_z_near, float p_z_far);
	void make_current();
	void clear_current(bool p_enable_next = true);
	void set_current(bool p_enabled);
	bool is_current() const;
	Transform3D get_camera_transform() const;
	Projection get_camera_projection() const;
	float get_fov() const;
	Vector2 get_frustum_offset() const;
	float get_size() const;
	float get_far() const;
	float get_near() const;
	void set_fov(float p_fov);
	void set_frustum_offset(const Vector2 &p_offset);
	void set_size(float p_size);
	void set_far(float p_far);
	void set_near(float p_near);
	Camera3D::ProjectionType get_projection() const;
	void set_projection(Camera3D::ProjectionType p_mode);
	void set_h_offset(float p_offset);
	float get_h_offset() const;
	void set_v_offset(float p_offset);
	float get_v_offset() const;
	void set_cull_mask(uint32_t p_mask);
	uint32_t get_cull_mask() const;
	void set_environment(const Ref<Environment> &p_env);
	Ref<Environment> get_environment() const;
	void set_attributes(const Ref<CameraAttributes> &p_env);
	Ref<CameraAttributes> get_attributes() const;
	void set_compositor(const Ref<Compositor> &p_compositor);
	Ref<Compositor> get_compositor() const;
	void set_keep_aspect_mode(Camera3D::KeepAspect p_mode);
	Camera3D::KeepAspect get_keep_aspect_mode() const;
	void set_doppler_tracking(Camera3D::DopplerTracking p_mode);
	Camera3D::DopplerTracking get_doppler_tracking() const;
	TypedArray<Plane> get_frustum() const;
	bool is_position_in_frustum(const Vector3 &p_world_point) const;
	RID get_camera_rid() const;
	RID get_pyramid_shape_rid();
	void set_cull_mask_value(int32_t p_layer_number, bool p_value);
	bool get_cull_mask_value(int32_t p_layer_number) const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Node3D::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(Camera3D::ProjectionType);
VARIANT_ENUM_CAST(Camera3D::KeepAspect);
VARIANT_ENUM_CAST(Camera3D::DopplerTracking);

