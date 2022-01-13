/*************************************************************************/
/*  camera.cpp                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "camera.h"

#include "collision_object.h"
#include "core/engine.h"
#include "core/math/camera_matrix.h"
#include "scene/resources/material.h"
#include "scene/resources/surface_tool.h"
void Camera::_update_audio_listener_state() {
}

void Camera::_request_camera_update() {
	_update_camera();
}

void Camera::_update_camera_mode() {
	force_change = true;
	switch (mode) {
		case PROJECTION_PERSPECTIVE: {
			set_perspective(fov, near, far);

		} break;
		case PROJECTION_ORTHOGONAL: {
			set_orthogonal(size, near, far);
		} break;
		case PROJECTION_FRUSTUM: {
			set_frustum(size, frustum_offset, near, far);
		} break;
	}
}

void Camera::_validate_property(PropertyInfo &p_property) const {
	if (p_property.name == "fov") {
		if (mode != PROJECTION_PERSPECTIVE) {
			p_property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	} else if (p_property.name == "size") {
		if (mode != PROJECTION_ORTHOGONAL && mode != PROJECTION_FRUSTUM) {
			p_property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	} else if (p_property.name == "frustum_offset") {
		if (mode != PROJECTION_FRUSTUM) {
			p_property.usage = PROPERTY_USAGE_NOEDITOR;
		}
	}
}

void Camera::_update_camera() {
	if (!is_inside_tree()) {
		return;
	}

	VisualServer::get_singleton()->camera_set_transform(camera, get_camera_transform());

	// here goes listener stuff
	/*
	if (viewport_ptr && is_inside_scene() && is_current())
		get_viewport()->_camera_transform_changed_notify();
	*/

	if (get_tree()->is_node_being_edited(this) || !is_current()) {
		return;
	}

	get_viewport()->_camera_transform_changed_notify();

	if (get_world().is_valid()) {
		get_world()->_update_camera(this);
	}
}

void Camera::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			// Needs to track the Viewport  because it's needed on NOTIFICATION_EXIT_WORLD
			// and Spatial will handle it first, including clearing its reference to the Viewport,
			// therefore making it impossible to subclasses to access it
			viewport = get_viewport();
			ERR_FAIL_COND(!viewport);

			bool first_camera = viewport->_camera_add(this);
			if (current || first_camera) {
				viewport->_camera_set(this);
			}

		} break;
		case NOTIFICATION_TRANSFORM_CHANGED: {
			_request_camera_update();
			if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
				velocity_tracker->update_position(get_global_transform().origin);
			}
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			if (!get_tree()->is_node_being_edited(this)) {
				if (is_current()) {
					clear_current();
					current = true; //keep it true

				} else {
					current = false;
				}
			}

			if (viewport) {
				viewport->_camera_remove(this);
				viewport = nullptr;
			}

		} break;
		case NOTIFICATION_BECAME_CURRENT: {
			if (viewport) {
				viewport->find_world()->_register_camera(this);
			}
		} break;
		case NOTIFICATION_LOST_CURRENT: {
			if (viewport) {
				viewport->find_world()->_remove_camera(this);
			}
		} break;
	}
}

Transform Camera::get_camera_transform() const {
	Transform tr = get_global_transform().orthonormalized();
	tr.origin += tr.basis.get_axis(1) * v_offset;
	tr.origin += tr.basis.get_axis(0) * h_offset;
	return tr;
}

void Camera::set_perspective(float p_fovy_degrees, float p_z_near, float p_z_far) {
	if (!force_change && fov == p_fovy_degrees && p_z_near == near && p_z_far == far && mode == PROJECTION_PERSPECTIVE) {
		return;
	}

	fov = p_fovy_degrees;
	near = p_z_near;
	far = p_z_far;
	mode = PROJECTION_PERSPECTIVE;

	VisualServer::get_singleton()->camera_set_perspective(camera, fov, near, far);
	update_gizmo();
	force_change = false;
}
void Camera::set_orthogonal(float p_size, float p_z_near, float p_z_far) {
	if (!force_change && size == p_size && p_z_near == near && p_z_far == far && mode == PROJECTION_ORTHOGONAL) {
		return;
	}

	size = p_size;

	near = p_z_near;
	far = p_z_far;
	mode = PROJECTION_ORTHOGONAL;
	force_change = false;

	VisualServer::get_singleton()->camera_set_orthogonal(camera, size, near, far);
	update_gizmo();
}

void Camera::set_frustum(float p_size, Vector2 p_offset, float p_z_near, float p_z_far) {
	if (!force_change && size == p_size && frustum_offset == p_offset && p_z_near == near && p_z_far == far && mode == PROJECTION_FRUSTUM) {
		return;
	}

	size = p_size;
	frustum_offset = p_offset;

	near = p_z_near;
	far = p_z_far;
	mode = PROJECTION_FRUSTUM;
	force_change = false;

	VisualServer::get_singleton()->camera_set_frustum(camera, size, frustum_offset, near, far);
	update_gizmo();
}

void Camera::set_projection(Camera::Projection p_mode) {
	if (p_mode == PROJECTION_PERSPECTIVE || p_mode == PROJECTION_ORTHOGONAL || p_mode == PROJECTION_FRUSTUM) {
		mode = p_mode;
		_update_camera_mode();
		_change_notify();
	}
}

RID Camera::get_camera() const {
	return camera;
};

void Camera::make_current() {
	current = true;

	if (!is_inside_tree()) {
		return;
	}

	get_viewport()->_camera_set(this);

	//get_scene()->call_group(SceneMainLoop::GROUP_CALL_REALTIME,camera_group,"_camera_make_current",this);
}

void Camera::clear_current(bool p_enable_next) {
	current = false;
	if (!is_inside_tree()) {
		return;
	}

	if (get_viewport()->get_camera() == this) {
		get_viewport()->_camera_set(nullptr);

		if (p_enable_next) {
			get_viewport()->_camera_make_next_current(this);
		}
	}
}

void Camera::set_current(bool p_current) {
	if (p_current) {
		make_current();
	} else {
		clear_current();
	}
}

bool Camera::is_current() const {
	if (is_inside_tree() && !get_tree()->is_node_being_edited(this)) {
		return get_viewport()->get_camera() == this;
	} else {
		return current;
	}
}

Vector3 Camera::project_ray_normal(const Point2 &p_pos) const {
	Vector3 ray = project_local_ray_normal(p_pos);
	return get_camera_transform().basis.xform(ray).normalized();
};

Vector3 Camera::project_local_ray_normal(const Point2 &p_pos) const {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector3(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_camera_rect_size();
	Vector2 cpos = get_viewport()->get_camera_coords(p_pos);
	Vector3 ray;

	if (mode == PROJECTION_ORTHOGONAL) {
		ray = Vector3(0, 0, -1);
	} else {
		CameraMatrix cm;
		cm.set_perspective(fov, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
		Vector2 screen_he = cm.get_viewport_half_extents();
		ray = Vector3(((cpos.x / viewport_size.width) * 2.0 - 1.0) * screen_he.x, ((1.0 - (cpos.y / viewport_size.height)) * 2.0 - 1.0) * screen_he.y, -near).normalized();
	}

	return ray;
};

Vector3 Camera::project_ray_origin(const Point2 &p_pos) const {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector3(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_camera_rect_size();
	Vector2 cpos = get_viewport()->get_camera_coords(p_pos);
	ERR_FAIL_COND_V(viewport_size.y == 0, Vector3());

	if (mode == PROJECTION_PERSPECTIVE) {
		return get_camera_transform().origin;
	} else {
		Vector2 pos = cpos / viewport_size;
		float vsize, hsize;
		if (keep_aspect == KEEP_WIDTH) {
			vsize = size / viewport_size.aspect();
			hsize = size;
		} else {
			hsize = size * viewport_size.aspect();
			vsize = size;
		}

		Vector3 ray;
		ray.x = pos.x * (hsize)-hsize / 2;
		ray.y = (1.0 - pos.y) * (vsize)-vsize / 2;
		ray.z = -near;
		ray = get_camera_transform().xform(ray);
		return ray;
	};
};

bool Camera::is_position_behind(const Vector3 &p_pos) const {
	Transform t = get_global_transform();
	Vector3 eyedir = -t.basis.get_axis(2).normalized();
	return eyedir.dot(p_pos - t.origin) < near;
}

Vector<Vector3> Camera::get_near_plane_points() const {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector<Vector3>(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_visible_rect().size;

	CameraMatrix cm;

	if (mode == PROJECTION_ORTHOGONAL) {
		cm.set_orthogonal(size, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	} else {
		cm.set_perspective(fov, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	}

	Vector3 endpoints[8];
	cm.get_endpoints(Transform(), endpoints);

	Vector<Vector3> points;
	points.push_back(Vector3());
	for (int i = 0; i < 4; i++) {
		points.push_back(endpoints[i + 4]);
	}
	return points;
}

Point2 Camera::unproject_position(const Vector3 &p_pos) const {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector2(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_visible_rect().size;

	CameraMatrix cm;

	if (mode == PROJECTION_ORTHOGONAL) {
		cm.set_orthogonal(size, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	} else {
		cm.set_perspective(fov, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	}

	Plane p(get_camera_transform().xform_inv(p_pos), 1.0);

	p = cm.xform4(p);
	p.normal /= p.d;

	Point2 res;
	res.x = (p.normal.x * 0.5 + 0.5) * viewport_size.x;
	res.y = (-p.normal.y * 0.5 + 0.5) * viewport_size.y;

	return res;
}

Vector3 Camera::project_position(const Point2 &p_point, float p_z_depth) const {
	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector3(), "Camera is not inside scene.");

	if (p_z_depth == 0 && mode != PROJECTION_ORTHOGONAL) {
		return get_global_transform().origin;
	}
	Size2 viewport_size = get_viewport()->get_visible_rect().size;

	CameraMatrix cm;

	if (mode == PROJECTION_ORTHOGONAL) {
		cm.set_orthogonal(size, viewport_size.aspect(), p_z_depth, far, keep_aspect == KEEP_WIDTH);
	} else {
		cm.set_perspective(fov, viewport_size.aspect(), p_z_depth, far, keep_aspect == KEEP_WIDTH);
	}

	Vector2 vp_he = cm.get_viewport_half_extents();

	Vector2 point;
	point.x = (p_point.x / viewport_size.x) * 2.0 - 1.0;
	point.y = (1.0 - (p_point.y / viewport_size.y)) * 2.0 - 1.0;
	point *= vp_he;

	Vector3 p(point.x, point.y, -p_z_depth);

	return get_camera_transform().xform(p);
}

/*
void Camera::_camera_make_current(Node *p_camera) {


	if (p_camera==this) {
		VisualServer::get_singleton()->viewport_attach_camera(viewport_id,camera);
		active=true;
	} else {
		if (active && p_camera==NULL) {
			//detech camera because no one else will claim it
			VisualServer::get_singleton()->viewport_attach_camera(viewport_id,RID());
		}
		active=false;
	}
}
*/

void Camera::set_environment(const Ref<Environment> &p_environment) {
	environment = p_environment;
	if (environment.is_valid()) {
		VS::get_singleton()->camera_set_environment(camera, environment->get_rid());
	} else {
		VS::get_singleton()->camera_set_environment(camera, RID());
	}
	_update_camera_mode();
}

Ref<Environment> Camera::get_environment() const {
	return environment;
}

void Camera::set_keep_aspect_mode(KeepAspect p_aspect) {
	keep_aspect = p_aspect;
	VisualServer::get_singleton()->camera_set_use_vertical_aspect(camera, p_aspect == KEEP_WIDTH);
	_update_camera_mode();
	_change_notify();
}

Camera::KeepAspect Camera::get_keep_aspect_mode() const {
	return keep_aspect;
}

void Camera::set_doppler_tracking(DopplerTracking p_tracking) {
	if (doppler_tracking == p_tracking) {
		return;
	}

	doppler_tracking = p_tracking;
	if (p_tracking != DOPPLER_TRACKING_DISABLED) {
		velocity_tracker->set_track_physics_step(doppler_tracking == DOPPLER_TRACKING_PHYSICS_STEP);
		if (is_inside_tree()) {
			velocity_tracker->reset(get_global_transform().origin);
		}
	}
	_update_camera_mode();
}

Camera::DopplerTracking Camera::get_doppler_tracking() const {
	return doppler_tracking;
}

void Camera::_bind_methods() {
	ClassDB::bind_method(D_METHOD("project_ray_normal", "screen_point"), &Camera::project_ray_normal);
	ClassDB::bind_method(D_METHOD("project_local_ray_normal", "screen_point"), &Camera::project_local_ray_normal);
	ClassDB::bind_method(D_METHOD("project_ray_origin", "screen_point"), &Camera::project_ray_origin);
	ClassDB::bind_method(D_METHOD("unproject_position", "world_point"), &Camera::unproject_position);
	ClassDB::bind_method(D_METHOD("is_position_behind", "world_point"), &Camera::is_position_behind);
	ClassDB::bind_method(D_METHOD("project_position", "screen_point", "z_depth"), &Camera::project_position);
	ClassDB::bind_method(D_METHOD("set_perspective", "fov", "z_near", "z_far"), &Camera::set_perspective);
	ClassDB::bind_method(D_METHOD("set_orthogonal", "size", "z_near", "z_far"), &Camera::set_orthogonal);
	ClassDB::bind_method(D_METHOD("set_frustum", "size", "offset", "z_near", "z_far"), &Camera::set_frustum);
	ClassDB::bind_method(D_METHOD("make_current"), &Camera::make_current);
	ClassDB::bind_method(D_METHOD("clear_current", "enable_next"), &Camera::clear_current, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_current"), &Camera::set_current);
	ClassDB::bind_method(D_METHOD("is_current"), &Camera::is_current);
	ClassDB::bind_method(D_METHOD("get_camera_transform"), &Camera::get_camera_transform);
	ClassDB::bind_method(D_METHOD("get_fov"), &Camera::get_fov);
	ClassDB::bind_method(D_METHOD("get_frustum_offset"), &Camera::get_frustum_offset);
	ClassDB::bind_method(D_METHOD("get_size"), &Camera::get_size);
	ClassDB::bind_method(D_METHOD("get_zfar"), &Camera::get_zfar);
	ClassDB::bind_method(D_METHOD("get_znear"), &Camera::get_znear);
	ClassDB::bind_method(D_METHOD("set_fov"), &Camera::set_fov);
	ClassDB::bind_method(D_METHOD("set_frustum_offset"), &Camera::set_frustum_offset);
	ClassDB::bind_method(D_METHOD("set_size"), &Camera::set_size);
	ClassDB::bind_method(D_METHOD("set_zfar"), &Camera::set_zfar);
	ClassDB::bind_method(D_METHOD("set_znear"), &Camera::set_znear);
	ClassDB::bind_method(D_METHOD("get_projection"), &Camera::get_projection);
	ClassDB::bind_method(D_METHOD("set_projection"), &Camera::set_projection);
	ClassDB::bind_method(D_METHOD("set_h_offset", "ofs"), &Camera::set_h_offset);
	ClassDB::bind_method(D_METHOD("get_h_offset"), &Camera::get_h_offset);
	ClassDB::bind_method(D_METHOD("set_v_offset", "ofs"), &Camera::set_v_offset);
	ClassDB::bind_method(D_METHOD("get_v_offset"), &Camera::get_v_offset);
	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &Camera::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &Camera::get_cull_mask);
	ClassDB::bind_method(D_METHOD("set_environment", "env"), &Camera::set_environment);
	ClassDB::bind_method(D_METHOD("get_environment"), &Camera::get_environment);
	ClassDB::bind_method(D_METHOD("set_keep_aspect_mode", "mode"), &Camera::set_keep_aspect_mode);
	ClassDB::bind_method(D_METHOD("get_keep_aspect_mode"), &Camera::get_keep_aspect_mode);
	ClassDB::bind_method(D_METHOD("set_doppler_tracking", "mode"), &Camera::set_doppler_tracking);
	ClassDB::bind_method(D_METHOD("get_doppler_tracking"), &Camera::get_doppler_tracking);
	ClassDB::bind_method(D_METHOD("get_frustum"), &Camera::get_frustum);
	ClassDB::bind_method(D_METHOD("get_camera_rid"), &Camera::get_camera);

	ClassDB::bind_method(D_METHOD("set_cull_mask_bit", "layer", "enable"), &Camera::set_cull_mask_bit);
	ClassDB::bind_method(D_METHOD("get_cull_mask_bit", "layer"), &Camera::get_cull_mask_bit);

	//ClassDB::bind_method(D_METHOD("_camera_make_current"),&Camera::_camera_make_current );

	ADD_PROPERTY(PropertyInfo(Variant::INT, "keep_aspect", PROPERTY_HINT_ENUM, "Keep Width,Keep Height"), "set_keep_aspect_mode", "get_keep_aspect_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "environment", PROPERTY_HINT_RESOURCE_TYPE, "Environment"), "set_environment", "get_environment");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "h_offset"), "set_h_offset", "get_h_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "v_offset"), "set_v_offset", "get_v_offset");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "doppler_tracking", PROPERTY_HINT_ENUM, "Disabled,Idle,Physics"), "set_doppler_tracking", "get_doppler_tracking");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "projection", PROPERTY_HINT_ENUM, "Perspective,Orthogonal,Frustum"), "set_projection", "get_projection");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "current"), "set_current", "is_current");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "fov", PROPERTY_HINT_RANGE, "1,179,0.1"), "set_fov", "get_fov");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "size", PROPERTY_HINT_RANGE, "0.1,16384,0.01"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "frustum_offset"), "set_frustum_offset", "get_frustum_offset");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "near", PROPERTY_HINT_EXP_RANGE, "0.01,8192,0.01,or_greater"), "set_znear", "get_znear");
	ADD_PROPERTY(PropertyInfo(Variant::REAL, "far", PROPERTY_HINT_EXP_RANGE, "0.1,8192,0.1,or_greater"), "set_zfar", "get_zfar");

	BIND_ENUM_CONSTANT(PROJECTION_PERSPECTIVE);
	BIND_ENUM_CONSTANT(PROJECTION_ORTHOGONAL);
	BIND_ENUM_CONSTANT(PROJECTION_FRUSTUM);

	BIND_ENUM_CONSTANT(KEEP_WIDTH);
	BIND_ENUM_CONSTANT(KEEP_HEIGHT);

	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_DISABLED);
	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_IDLE_STEP);
	BIND_ENUM_CONSTANT(DOPPLER_TRACKING_PHYSICS_STEP);
}

float Camera::get_fov() const {
	return fov;
}

float Camera::get_size() const {
	return size;
}

float Camera::get_znear() const {
	return near;
}

Vector2 Camera::get_frustum_offset() const {
	return frustum_offset;
}

float Camera::get_zfar() const {
	return far;
}

Camera::Projection Camera::get_projection() const {
	return mode;
}

void Camera::set_fov(float p_fov) {
	ERR_FAIL_COND(p_fov < 1 || p_fov > 179);
	fov = p_fov;
	_update_camera_mode();
	_change_notify("fov");
}

void Camera::set_size(float p_size) {
	ERR_FAIL_COND(p_size < 0.1 || p_size > 16384);
	size = p_size;
	_update_camera_mode();
	_change_notify("size");
}

void Camera::set_znear(float p_znear) {
	near = p_znear;
	_update_camera_mode();
}

void Camera::set_frustum_offset(Vector2 p_offset) {
	frustum_offset = p_offset;
	_update_camera_mode();
}

void Camera::set_zfar(float p_zfar) {
	far = p_zfar;
	_update_camera_mode();
}

void Camera::set_cull_mask(uint32_t p_layers) {
	layers = p_layers;
	VisualServer::get_singleton()->camera_set_cull_mask(camera, layers);
	_update_camera_mode();
}

uint32_t Camera::get_cull_mask() const {
	return layers;
}

void Camera::set_cull_mask_bit(int p_layer, bool p_enable) {
	ERR_FAIL_INDEX(p_layer, 32);
	if (p_enable) {
		set_cull_mask(layers | (1 << p_layer));
	} else {
		set_cull_mask(layers & (~(1 << p_layer)));
	}
}

bool Camera::get_cull_mask_bit(int p_layer) const {
	ERR_FAIL_INDEX_V(p_layer, 32, false);
	return (layers & (1 << p_layer));
}

Vector<Plane> Camera::get_frustum() const {
	ERR_FAIL_COND_V(!is_inside_world(), Vector<Plane>());

	Size2 viewport_size = get_viewport()->get_visible_rect().size;
	CameraMatrix cm;
	if (mode == PROJECTION_PERSPECTIVE) {
		cm.set_perspective(fov, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	} else {
		cm.set_orthogonal(size, viewport_size.aspect(), near, far, keep_aspect == KEEP_WIDTH);
	}

	return cm.get_projection_planes(get_camera_transform());
}

void Camera::set_v_offset(float p_offset) {
	v_offset = p_offset;
	_update_camera();
}

float Camera::get_v_offset() const {
	return v_offset;
}

void Camera::set_h_offset(float p_offset) {
	h_offset = p_offset;
	_update_camera();
}

float Camera::get_h_offset() const {
	return h_offset;
}

Vector3 Camera::get_doppler_tracked_velocity() const {
	if (doppler_tracking != DOPPLER_TRACKING_DISABLED) {
		return velocity_tracker->get_tracked_linear_velocity();
	} else {
		return Vector3();
	}
}
Camera::Camera() {
	camera = VisualServer::get_singleton()->camera_create();
	size = 1;
	fov = 0;
	frustum_offset = Vector2();
	near = 0;
	far = 0;
	current = false;
	viewport = nullptr;
	force_change = false;
	mode = PROJECTION_PERSPECTIVE;
	set_perspective(70.0, 0.05, 100.0);
	keep_aspect = KEEP_HEIGHT;
	layers = 0xfffff;
	v_offset = 0;
	h_offset = 0;
	VisualServer::get_singleton()->camera_set_cull_mask(camera, layers);
	//active=false;
	velocity_tracker.instance();
	doppler_tracking = DOPPLER_TRACKING_DISABLED;
	set_notify_transform(true);
	set_disable_scale(true);
}

Camera::~Camera() {
	VisualServer::get_singleton()->free(camera);
}

////////////////////////////////////////

void ClippedCamera::set_margin(float p_margin) {
	margin = p_margin;
}
float ClippedCamera::get_margin() const {
	return margin;
}
void ClippedCamera::set_process_mode(ProcessMode p_mode) {
	if (process_mode == p_mode) {
		return;
	}
	process_mode = p_mode;
	set_process_internal(process_mode == CLIP_PROCESS_IDLE);
	set_physics_process_internal(process_mode == CLIP_PROCESS_PHYSICS);
}
ClippedCamera::ProcessMode ClippedCamera::get_process_mode() const {
	return process_mode;
}

Transform ClippedCamera::get_camera_transform() const {
	Transform t = Camera::get_camera_transform();
	t.origin += -t.basis.get_axis(Vector3::AXIS_Z).normalized() * clip_offset;
	return t;
}

void ClippedCamera::_notification(int p_what) {
	if (p_what == NOTIFICATION_INTERNAL_PROCESS || p_what == NOTIFICATION_INTERNAL_PHYSICS_PROCESS) {
		Spatial *parent = Object::cast_to<Spatial>(get_parent());
		if (!parent) {
			return;
		}

		PhysicsDirectSpaceState *dspace = get_world()->get_direct_space_state();
		ERR_FAIL_COND(!dspace); // most likely physics set to threads

		Vector3 cam_fw = -get_global_transform().basis.get_axis(Vector3::AXIS_Z).normalized();
		Vector3 cam_pos = get_global_transform().origin;
		Vector3 parent_pos = parent->get_global_transform().origin;

		Plane parent_plane(parent_pos, cam_fw);

		if (parent_plane.is_point_over(cam_pos)) {
			//cam is beyond parent plane
			return;
		}

		Vector3 ray_from = parent_plane.project(cam_pos);

		clip_offset = 0; //reset by defau;t

		{ //check if points changed
			Vector<Vector3> local_points = get_near_plane_points();

			bool all_equal = true;

			for (int i = 0; i < 5; i++) {
				if (points[i] != local_points[i]) {
					all_equal = false;
					break;
				}
			}

			if (!all_equal) {
				PhysicsServer::get_singleton()->shape_set_data(pyramid_shape, local_points);
				points = local_points;
			}
		}

		Transform xf = get_global_transform();
		xf.origin = ray_from;
		xf.orthonormalize();

		float closest_safe = 1.0f, closest_unsafe = 1.0f;
		if (dspace->cast_motion(pyramid_shape, xf, cam_pos - ray_from, margin, closest_safe, closest_unsafe, exclude, collision_mask, clip_to_bodies, clip_to_areas)) {
			clip_offset = cam_pos.distance_to(ray_from + (cam_pos - ray_from) * closest_safe);
		}

		_update_camera();
	}

	if (p_what == NOTIFICATION_LOCAL_TRANSFORM_CHANGED) {
		update_gizmo();
	}
}

void ClippedCamera::set_collision_mask(uint32_t p_mask) {
	collision_mask = p_mask;
}

uint32_t ClippedCamera::get_collision_mask() const {
	return collision_mask;
}

void ClippedCamera::set_collision_mask_bit(int p_bit, bool p_value) {
	ERR_FAIL_INDEX_MSG(p_bit, 32, "Collision layer bit must be between 0 and 31 inclusive.");
	uint32_t mask = get_collision_mask();
	if (p_value) {
		mask |= 1 << p_bit;
	} else {
		mask &= ~(1 << p_bit);
	}
	set_collision_mask(mask);
}

bool ClippedCamera::get_collision_mask_bit(int p_bit) const {
	ERR_FAIL_INDEX_V_MSG(p_bit, 32, false, "Collision mask bit must be between 0 and 31 inclusive.");
	return get_collision_mask() & (1 << p_bit);
}

void ClippedCamera::add_exception_rid(const RID &p_rid) {
	exclude.insert(p_rid);
}

void ClippedCamera::add_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	if (!co) {
		return;
	}
	add_exception_rid(co->get_rid());
}

void ClippedCamera::remove_exception_rid(const RID &p_rid) {
	exclude.erase(p_rid);
}

void ClippedCamera::remove_exception(const Object *p_object) {
	ERR_FAIL_NULL(p_object);
	const CollisionObject *co = Object::cast_to<CollisionObject>(p_object);
	if (!co) {
		return;
	}
	remove_exception_rid(co->get_rid());
}

void ClippedCamera::clear_exceptions() {
	exclude.clear();
}

float ClippedCamera::get_clip_offset() const {
	return clip_offset;
}

void ClippedCamera::set_clip_to_areas(bool p_clip) {
	clip_to_areas = p_clip;
}

bool ClippedCamera::is_clip_to_areas_enabled() const {
	return clip_to_areas;
}

void ClippedCamera::set_clip_to_bodies(bool p_clip) {
	clip_to_bodies = p_clip;
}

bool ClippedCamera::is_clip_to_bodies_enabled() const {
	return clip_to_bodies;
}

void ClippedCamera::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &ClippedCamera::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &ClippedCamera::get_margin);

	ClassDB::bind_method(D_METHOD("set_process_mode", "process_mode"), &ClippedCamera::set_process_mode);
	ClassDB::bind_method(D_METHOD("get_process_mode"), &ClippedCamera::get_process_mode);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "mask"), &ClippedCamera::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &ClippedCamera::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_collision_mask_bit", "bit", "value"), &ClippedCamera::set_collision_mask_bit);
	ClassDB::bind_method(D_METHOD("get_collision_mask_bit", "bit"), &ClippedCamera::get_collision_mask_bit);

	ClassDB::bind_method(D_METHOD("add_exception_rid", "rid"), &ClippedCamera::add_exception_rid);
	ClassDB::bind_method(D_METHOD("add_exception", "node"), &ClippedCamera::add_exception);

	ClassDB::bind_method(D_METHOD("remove_exception_rid", "rid"), &ClippedCamera::remove_exception_rid);
	ClassDB::bind_method(D_METHOD("remove_exception", "node"), &ClippedCamera::remove_exception);

	ClassDB::bind_method(D_METHOD("set_clip_to_areas", "enable"), &ClippedCamera::set_clip_to_areas);
	ClassDB::bind_method(D_METHOD("is_clip_to_areas_enabled"), &ClippedCamera::is_clip_to_areas_enabled);

	ClassDB::bind_method(D_METHOD("get_clip_offset"), &ClippedCamera::get_clip_offset);

	ClassDB::bind_method(D_METHOD("set_clip_to_bodies", "enable"), &ClippedCamera::set_clip_to_bodies);
	ClassDB::bind_method(D_METHOD("is_clip_to_bodies_enabled"), &ClippedCamera::is_clip_to_bodies_enabled);

	ClassDB::bind_method(D_METHOD("clear_exceptions"), &ClippedCamera::clear_exceptions);

	ADD_PROPERTY(PropertyInfo(Variant::REAL, "margin", PROPERTY_HINT_RANGE, "0,32,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_mode", PROPERTY_HINT_ENUM, "Physics,Idle"), "set_process_mode", "get_process_mode");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");

	ADD_GROUP("Clip To", "clip_to");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_to_areas", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_clip_to_areas", "is_clip_to_areas_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "clip_to_bodies", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_clip_to_bodies", "is_clip_to_bodies_enabled");

	BIND_ENUM_CONSTANT(CLIP_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(CLIP_PROCESS_IDLE);
}
ClippedCamera::ClippedCamera() {
	margin = 0;
	clip_offset = 0;
	process_mode = CLIP_PROCESS_PHYSICS;
	set_physics_process_internal(true);
	collision_mask = 1;
	set_notify_local_transform(Engine::get_singleton()->is_editor_hint());
	points.resize(5);
	pyramid_shape = PhysicsServer::get_singleton()->shape_create(PhysicsServer::SHAPE_CONVEX_POLYGON);
	clip_to_areas = false;
	clip_to_bodies = true;
}
ClippedCamera::~ClippedCamera() {
	PhysicsServer::get_singleton()->free(pyramid_shape);
}
