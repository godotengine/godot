/*************************************************************************/
/*  camera_3d.h                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef CAMERA_3D_H
#define CAMERA_3D_H

#include "scene/3d/node_3d.h"
#include "scene/3d/velocity_tracker_3d.h"
#include "scene/main/window.h"
#include "scene/resources/camera_effects.h"
#include "scene/resources/environment.h"

class Camera3D : public Node3D {
	GDCLASS(Camera3D, Node3D);

public:
	enum Projection {
		PROJECTION_PERSPECTIVE,
		PROJECTION_ORTHOGONAL,
		PROJECTION_FRUSTUM
	};

	enum KeepAspect { KEEP_WIDTH,
		KEEP_HEIGHT };

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED,
		DOPPLER_TRACKING_IDLE_STEP,
		DOPPLER_TRACKING_PHYSICS_STEP
	};

private:
	bool force_change = false;
	bool current = false;
	Viewport *viewport = nullptr;

	Projection mode = PROJECTION_PERSPECTIVE;

	float fov = 0.0;
	float size = 1.0;
	Vector2 frustum_offset;
	float near = 0.0;
	float far = 0.0;
	float v_offset = 0.0;
	float h_offset = 0.0;
	KeepAspect keep_aspect = KEEP_HEIGHT;

	RID camera;
	RID scenario_id;

	// String camera_group;

	uint32_t layers = 0xfffff;

	Ref<Environment> environment;
	Ref<CameraEffects> effects;

	virtual bool _can_gizmo_scale() const;

	// void _camera_make_current(Node *p_camera);
	friend class Viewport;
	void _update_audio_listener_state();

	DopplerTracking doppler_tracking = DOPPLER_TRACKING_DISABLED;
	Ref<VelocityTracker3D> velocity_tracker;

protected:
	void _update_camera();
	virtual void _request_camera_update();
	void _update_camera_mode();

	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &p_property) const override;

	static void _bind_methods();

public:
	enum {
		NOTIFICATION_BECAME_CURRENT = 50,
		NOTIFICATION_LOST_CURRENT = 51
	};

	void set_perspective(float p_fovy_degrees, float p_z_near, float p_z_far);
	void set_orthogonal(float p_size, float p_z_near, float p_z_far);
	void set_frustum(float p_size, Vector2 p_offset, float p_z_near,
			float p_z_far);
	void set_projection(Camera3D::Projection p_mode);

	void make_current();
	void clear_current(bool p_enable_next = true);
	void set_current(bool p_current);
	bool is_current() const;

	RID get_camera() const;

	float get_fov() const;
	float get_size() const;
	float get_far() const;
	float get_near() const;
	Vector2 get_frustum_offset() const;

	Projection get_projection() const;

	void set_fov(float p_fov);
	void set_size(float p_size);
	void set_far(float p_far);
	void set_near(float p_near);
	void set_frustum_offset(Vector2 p_offset);

	virtual Transform get_camera_transform() const;

	virtual Vector3 project_ray_normal(const Point2 &p_pos) const;
	virtual Vector3 project_ray_origin(const Point2 &p_pos) const;
	virtual Vector3 project_local_ray_normal(const Point2 &p_pos) const;
	virtual Point2 unproject_position(const Vector3 &p_pos) const;
	bool is_position_behind(const Vector3 &p_pos) const;
	virtual Vector3 project_position(const Point2 &p_point,
			float p_z_depth) const;

	Vector<Vector3> get_near_plane_points() const;

	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;

	void set_cull_mask_bit(int p_layer, bool p_enable);
	bool get_cull_mask_bit(int p_layer) const;

	virtual Vector<Plane> get_frustum() const;

	void set_environment(const Ref<Environment> &p_environment);
	Ref<Environment> get_environment() const;

	void set_effects(const Ref<CameraEffects> &p_effects);
	Ref<CameraEffects> get_effects() const;

	void set_keep_aspect_mode(KeepAspect p_aspect);
	KeepAspect get_keep_aspect_mode() const;

	void set_v_offset(float p_offset);
	float get_v_offset() const;

	void set_h_offset(float p_offset);
	float get_h_offset() const;

	void set_doppler_tracking(DopplerTracking p_tracking);
	DopplerTracking get_doppler_tracking() const;

	Vector3 get_doppler_tracked_velocity() const;

	Camera3D();
	~Camera3D();
};

VARIANT_ENUM_CAST(Camera3D::Projection);
VARIANT_ENUM_CAST(Camera3D::KeepAspect);
VARIANT_ENUM_CAST(Camera3D::DopplerTracking);

class ClippedCamera3D : public Camera3D {
	GDCLASS(ClippedCamera3D, Camera3D);

public:
	enum ClipProcessCallback {
		CLIP_PROCESS_PHYSICS,
		CLIP_PROCESS_IDLE,
	};

private:
	ClipProcessCallback process_callback = CLIP_PROCESS_PHYSICS;
	RID pyramid_shape;
	float margin = 0.0;
	float clip_offset = 0.0;
	uint32_t collision_mask = 1;
	bool clip_to_areas = false;
	bool clip_to_bodies = true;

	Set<RID> exclude;

	Vector<Vector3> points;

protected:
	void _notification(int p_what);
	static void _bind_methods();
	virtual Transform get_camera_transform() const override;

public:
	void set_clip_to_areas(bool p_clip);
	bool is_clip_to_areas_enabled() const;

	void set_clip_to_bodies(bool p_clip);
	bool is_clip_to_bodies_enabled() const;

	void set_margin(float p_margin);
	float get_margin() const;

	void set_process_callback(ClipProcessCallback p_mode);
	ClipProcessCallback get_process_callback() const;

	void set_collision_mask(uint32_t p_mask);
	uint32_t get_collision_mask() const;

	void set_collision_mask_bit(int p_bit, bool p_value);
	bool get_collision_mask_bit(int p_bit) const;

	void add_exception_rid(const RID &p_rid);
	void add_exception(const Object *p_object);
	void remove_exception_rid(const RID &p_rid);
	void remove_exception(const Object *p_object);
	void clear_exceptions();

	float get_clip_offset() const;

	ClippedCamera3D();
	~ClippedCamera3D();
};

VARIANT_ENUM_CAST(ClippedCamera3D::ClipProcessCallback);
#endif
