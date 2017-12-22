/*************************************************************************/
/*  camera.h                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#ifndef CAMERA_H
#define CAMERA_H

#include "scene/3d/spatial.h"
#include "scene/3d/spatial_velocity_tracker.h"
#include "scene/main/viewport.h"
#include "scene/resources/environment.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Camera : public Spatial {

	GDCLASS(Camera, Spatial);

public:
	enum Projection {

		PROJECTION_PERSPECTIVE,
		PROJECTION_ORTHOGONAL
	};

	enum KeepAspect {
		KEEP_WIDTH,
		KEEP_HEIGHT
	};

	enum DopplerTracking {
		DOPPLER_TRACKING_DISABLED,
		DOPPLER_TRACKING_IDLE_STEP,
		DOPPLER_TRACKING_PHYSICS_STEP
	};

private:
	bool force_change;
	bool current;

	Projection mode;

	float fov;
	float size;
	float near, far;
	float v_offset;
	float h_offset;
	KeepAspect keep_aspect;

	RID camera;
	RID scenario_id;

	//String camera_group;

	uint32_t layers;

	Ref<Environment> environment;

	virtual bool _can_gizmo_scale() const;

	//void _camera_make_current(Node *p_camera);
	friend class Viewport;
	void _update_audio_listener_state();

	DopplerTracking doppler_tracking;
	Ref<SpatialVelocityTracker> velocity_tracker;

protected:
	void _update_camera();
	virtual void _request_camera_update();
	void _update_camera_mode();

	void _notification(int p_what);
	virtual void _validate_property(PropertyInfo &property) const;

	static void _bind_methods();

public:
	enum {

		NOTIFICATION_BECAME_CURRENT = 50,
		NOTIFICATION_LOST_CURRENT = 51
	};

	void set_perspective(float p_fovy_degrees, float p_z_near, float p_z_far);
	void set_orthogonal(float p_size, float p_z_near, float p_z_far);
	void set_projection(Camera::Projection p_mode);

	void make_current();
	void clear_current();
	void set_current(bool p_current);
	bool is_current() const;

	RID get_camera() const;

	float get_fov() const;
	float get_size() const;
	float get_zfar() const;
	float get_znear() const;
	Projection get_projection() const;

	void set_fov(float p_fov);
	void set_size(float p_size);
	void set_zfar(float p_zfar);
	void set_znear(float p_znear);

	virtual Transform get_camera_transform() const;

	Vector3 project_ray_normal(const Point2 &p_pos) const;
	virtual Vector3 project_ray_origin(const Point2 &p_pos) const;
	Vector3 project_local_ray_normal(const Point2 &p_pos) const;
	virtual Point2 unproject_position(const Vector3 &p_pos) const;
	bool is_position_behind(const Vector3 &p_pos) const;
	virtual Vector3 project_position(const Point2 &p_point) const;

	void set_cull_mask(uint32_t p_layers);
	uint32_t get_cull_mask() const;

	virtual Vector<Plane> get_frustum() const;

	void set_environment(const Ref<Environment> &p_environment);
	Ref<Environment> get_environment() const;

	void set_keep_aspect_mode(KeepAspect p_aspect);
	KeepAspect get_keep_aspect_mode() const;

	void set_v_offset(float p_offset);
	float get_v_offset() const;

	void set_h_offset(float p_offset);
	float get_h_offset() const;

	void set_doppler_tracking(DopplerTracking p_tracking);
	DopplerTracking get_doppler_tracking() const;

	Vector3 get_doppler_tracked_velocity() const;

	Camera();
	~Camera();
};

VARIANT_ENUM_CAST(Camera::Projection);
VARIANT_ENUM_CAST(Camera::KeepAspect);
VARIANT_ENUM_CAST(Camera::DopplerTracking);

#endif
