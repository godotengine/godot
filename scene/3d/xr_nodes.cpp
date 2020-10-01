/*************************************************************************/
/*  xr_nodes.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "xr_nodes.h"

#include "core/input/input.h"
#include "servers/xr/xr_interface.h"
#include "servers/xr_server.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

void XRCamera3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			// need to find our XROrigin3D parent and let it know we're its camera!
			XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
			if (origin != nullptr) {
				origin->set_tracked_camera(this);
			}
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			// need to find our XROrigin3D parent and let it know we're no longer its camera!
			XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
			if (origin != nullptr) {
				origin->clear_tracked_camera_if(this);
			}
		}; break;
	};
};

String XRCamera3D::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree()) {
		return String();
	}

	String warning = Camera3D::get_configuration_warning();

	// must be child node of XROrigin3D!
	XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
	if (origin == nullptr) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("XRCamera3D must have an XROrigin3D node as its parent.");
	};

	return warning;
};

Vector3 XRCamera3D::project_local_ray_normal(const Point2 &p_pos) const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Vector3());

	Ref<XRInterface> xr_interface = xr_server->get_primary_interface();
	if (xr_interface.is_null()) {
		// we might be in the editor or have VR turned off, just call superclass
		return Camera3D::project_local_ray_normal(p_pos);
	}

	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector3(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_camera_rect_size();
	Vector2 cpos = get_viewport()->get_camera_coords(p_pos);
	Vector3 ray;

	CameraMatrix cm = xr_interface->get_projection_for_eye(XRInterface::EYE_MONO, viewport_size.aspect(), get_znear(), get_zfar());
	Vector2 screen_he = cm.get_viewport_half_extents();
	ray = Vector3(((cpos.x / viewport_size.width) * 2.0 - 1.0) * screen_he.x, ((1.0 - (cpos.y / viewport_size.height)) * 2.0 - 1.0) * screen_he.y, -get_znear()).normalized();

	return ray;
};

Point2 XRCamera3D::unproject_position(const Vector3 &p_pos) const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Vector2());

	Ref<XRInterface> xr_interface = xr_server->get_primary_interface();
	if (xr_interface.is_null()) {
		// we might be in the editor or have VR turned off, just call superclass
		return Camera3D::unproject_position(p_pos);
	}

	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector2(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_visible_rect().size;

	CameraMatrix cm = xr_interface->get_projection_for_eye(XRInterface::EYE_MONO, viewport_size.aspect(), get_znear(), get_zfar());

	Plane p(get_camera_transform().xform_inv(p_pos), 1.0);

	p = cm.xform4(p);
	p.normal /= p.d;

	Point2 res;
	res.x = (p.normal.x * 0.5 + 0.5) * viewport_size.x;
	res.y = (-p.normal.y * 0.5 + 0.5) * viewport_size.y;

	return res;
};

Vector3 XRCamera3D::project_position(const Point2 &p_point, float p_z_depth) const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Vector3());

	Ref<XRInterface> xr_interface = xr_server->get_primary_interface();
	if (xr_interface.is_null()) {
		// we might be in the editor or have VR turned off, just call superclass
		return Camera3D::project_position(p_point, p_z_depth);
	}

	ERR_FAIL_COND_V_MSG(!is_inside_tree(), Vector3(), "Camera is not inside scene.");

	Size2 viewport_size = get_viewport()->get_visible_rect().size;

	CameraMatrix cm = xr_interface->get_projection_for_eye(XRInterface::EYE_MONO, viewport_size.aspect(), get_znear(), get_zfar());

	Vector2 vp_he = cm.get_viewport_half_extents();

	Vector2 point;
	point.x = (p_point.x / viewport_size.x) * 2.0 - 1.0;
	point.y = (1.0 - (p_point.y / viewport_size.y)) * 2.0 - 1.0;
	point *= vp_he;

	Vector3 p(point.x, point.y, -p_z_depth);

	return get_camera_transform().xform(p);
};

Vector<Plane> XRCamera3D::get_frustum() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Vector<Plane>());

	Ref<XRInterface> xr_interface = xr_server->get_primary_interface();
	if (xr_interface.is_null()) {
		// we might be in the editor or have VR turned off, just call superclass
		return Camera3D::get_frustum();
	}

	ERR_FAIL_COND_V(!is_inside_world(), Vector<Plane>());

	Size2 viewport_size = get_viewport()->get_visible_rect().size;
	CameraMatrix cm = xr_interface->get_projection_for_eye(XRInterface::EYE_MONO, viewport_size.aspect(), get_znear(), get_zfar());
	return cm.get_projection_planes(get_camera_transform());
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void XRController3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// get our XRServer
			XRServer *xr_server = XRServer::get_singleton();
			ERR_FAIL_NULL(xr_server);

			// find the tracker for our controller
			XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
			if (tracker == nullptr) {
				// this controller is currently turned off
				is_active = false;
				button_states = 0;
			} else {
				is_active = true;
				set_transform(tracker->get_transform(true));

				int joy_id = tracker->get_joy_id();
				if (joy_id >= 0) {
					int mask = 1;
					// check button states
					for (int i = 0; i < 16; i++) {
						bool was_pressed = (button_states & mask) == mask;
						bool is_pressed = Input::get_singleton()->is_joy_button_pressed(joy_id, i);

						if (!was_pressed && is_pressed) {
							emit_signal("button_pressed", i);
							button_states += mask;
						} else if (was_pressed && !is_pressed) {
							emit_signal("button_release", i);
							button_states -= mask;
						};

						mask = mask << 1;
					};

				} else {
					button_states = 0;
				};

				// check for an updated mesh
				Ref<Mesh> trackerMesh = tracker->get_mesh();
				if (mesh != trackerMesh) {
					mesh = trackerMesh;
					emit_signal("mesh_updated", mesh);
				}
			};
		}; break;
		default:
			break;
	};
};

void XRController3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_controller_id", "controller_id"), &XRController3D::set_controller_id);
	ClassDB::bind_method(D_METHOD("get_controller_id"), &XRController3D::get_controller_id);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "controller_id", PROPERTY_HINT_RANGE, "0,32,1"), "set_controller_id", "get_controller_id");
	ClassDB::bind_method(D_METHOD("get_controller_name"), &XRController3D::get_controller_name);

	// passthroughs to information about our related joystick
	ClassDB::bind_method(D_METHOD("get_joystick_id"), &XRController3D::get_joystick_id);
	ClassDB::bind_method(D_METHOD("is_button_pressed", "button"), &XRController3D::is_button_pressed);
	ClassDB::bind_method(D_METHOD("get_joystick_axis", "axis"), &XRController3D::get_joystick_axis);

	ClassDB::bind_method(D_METHOD("get_is_active"), &XRController3D::get_is_active);
	ClassDB::bind_method(D_METHOD("get_hand"), &XRController3D::get_hand);

	ClassDB::bind_method(D_METHOD("get_rumble"), &XRController3D::get_rumble);
	ClassDB::bind_method(D_METHOD("set_rumble", "rumble"), &XRController3D::set_rumble);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "rumble", PROPERTY_HINT_RANGE, "0.0,1.0,0.01"), "set_rumble", "get_rumble");
	ADD_PROPERTY_DEFAULT("rumble", 0.0);

	ClassDB::bind_method(D_METHOD("get_mesh"), &XRController3D::get_mesh);

	ADD_SIGNAL(MethodInfo("button_pressed", PropertyInfo(Variant::INT, "button")));
	ADD_SIGNAL(MethodInfo("button_release", PropertyInfo(Variant::INT, "button")));
	ADD_SIGNAL(MethodInfo("mesh_updated", PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh")));
};

void XRController3D::set_controller_id(int p_controller_id) {
	// We don't check any bounds here, this controller may not yet be active and just be a place holder until it is.
	// Note that setting this to 0 means this node is not bound to a controller yet.
	controller_id = p_controller_id;
	update_configuration_warning();
};

int XRController3D::get_controller_id() const {
	return controller_id;
};

String XRController3D::get_controller_name() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, String());

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == nullptr) {
		return String("Not connected");
	};

	return tracker->get_name();
};

int XRController3D::get_joystick_id() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 0);

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == nullptr) {
		// No tracker? no joystick id... (0 is our first joystick)
		return -1;
	};

	return tracker->get_joy_id();
};

bool XRController3D::is_button_pressed(int p_button) const {
	int joy_id = get_joystick_id();
	if (joy_id == -1) {
		return false;
	};

	return Input::get_singleton()->is_joy_button_pressed(joy_id, p_button);
};

float XRController3D::get_joystick_axis(int p_axis) const {
	int joy_id = get_joystick_id();
	if (joy_id == -1) {
		return 0.0;
	};

	return Input::get_singleton()->get_joy_axis(joy_id, p_axis);
};

real_t XRController3D::get_rumble() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 0.0);

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == nullptr) {
		return 0.0;
	};

	return tracker->get_rumble();
};

void XRController3D::set_rumble(real_t p_rumble) {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker != nullptr) {
		tracker->set_rumble(p_rumble);
	};
};

Ref<Mesh> XRController3D::get_mesh() const {
	return mesh;
}

bool XRController3D::get_is_active() const {
	return is_active;
};

XRPositionalTracker::TrackerHand XRController3D::get_hand() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, XRPositionalTracker::TRACKER_HAND_UNKNOWN);

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_CONTROLLER, controller_id);
	if (tracker == nullptr) {
		return XRPositionalTracker::TRACKER_HAND_UNKNOWN;
	};

	return tracker->get_hand();
};

String XRController3D::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree()) {
		return String();
	}

	String warning = Node3D::get_configuration_warning();

	// must be child node of XROrigin!
	XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
	if (origin == nullptr) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("XRController3D must have an XROrigin3D node as its parent.");
	};

	if (controller_id == 0) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("The controller ID must not be 0 or this controller won't be bound to an actual controller.");
	};

	return warning;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void XRAnchor3D::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// get our XRServer
			XRServer *xr_server = XRServer::get_singleton();
			ERR_FAIL_NULL(xr_server);

			// find the tracker for our anchor
			XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_ANCHOR, anchor_id);
			if (tracker == nullptr) {
				// this anchor is currently not available
				is_active = false;
			} else {
				is_active = true;
				Transform transform;

				// we'll need our world_scale
				real_t world_scale = xr_server->get_world_scale();

				// get our info from our tracker
				transform.basis = tracker->get_orientation();
				transform.origin = tracker->get_position(); // <-- already adjusted to world scale

				// our basis is scaled to the size of the plane the anchor is tracking
				// extract the size from our basis and reset the scale
				size = transform.basis.get_scale() * world_scale;
				transform.basis.orthonormalize();

				// apply our reference frame and set our transform
				set_transform(xr_server->get_reference_frame() * transform);

				// check for an updated mesh
				Ref<Mesh> trackerMesh = tracker->get_mesh();
				if (mesh != trackerMesh) {
					mesh = trackerMesh;
					emit_signal("mesh_updated", mesh);
				}
			};
		}; break;
		default:
			break;
	};
};

void XRAnchor3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_anchor_id", "anchor_id"), &XRAnchor3D::set_anchor_id);
	ClassDB::bind_method(D_METHOD("get_anchor_id"), &XRAnchor3D::get_anchor_id);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "anchor_id", PROPERTY_HINT_RANGE, "0,32,1"), "set_anchor_id", "get_anchor_id");
	ClassDB::bind_method(D_METHOD("get_anchor_name"), &XRAnchor3D::get_anchor_name);

	ClassDB::bind_method(D_METHOD("get_is_active"), &XRAnchor3D::get_is_active);
	ClassDB::bind_method(D_METHOD("get_size"), &XRAnchor3D::get_size);

	ClassDB::bind_method(D_METHOD("get_plane"), &XRAnchor3D::get_plane);

	ClassDB::bind_method(D_METHOD("get_mesh"), &XRAnchor3D::get_mesh);
	ADD_SIGNAL(MethodInfo("mesh_updated", PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh")));
};

void XRAnchor3D::set_anchor_id(int p_anchor_id) {
	// We don't check any bounds here, this anchor may not yet be active and just be a place holder until it is.
	// Note that setting this to 0 means this node is not bound to an anchor yet.
	anchor_id = p_anchor_id;
	update_configuration_warning();
};

int XRAnchor3D::get_anchor_id() const {
	return anchor_id;
};

Vector3 XRAnchor3D::get_size() const {
	return size;
};

String XRAnchor3D::get_anchor_name() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, String());

	XRPositionalTracker *tracker = xr_server->find_by_type_and_id(XRServer::TRACKER_ANCHOR, anchor_id);
	if (tracker == nullptr) {
		return String("Not connected");
	};

	return tracker->get_name();
};

bool XRAnchor3D::get_is_active() const {
	return is_active;
};

String XRAnchor3D::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree()) {
		return String();
	}

	String warning = Node3D::get_configuration_warning();

	// must be child node of XROrigin3D!
	XROrigin3D *origin = Object::cast_to<XROrigin3D>(get_parent());
	if (origin == nullptr) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("XRAnchor3D must have an XROrigin3D node as its parent.");
	};

	if (anchor_id == 0) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("The anchor ID must not be 0 or this anchor won't be bound to an actual anchor.");
	};

	return warning;
};

Plane XRAnchor3D::get_plane() const {
	Vector3 location = get_translation();
	Basis orientation = get_transform().basis;

	Plane plane(location, orientation.get_axis(1).normalized());

	return plane;
};

Ref<Mesh> XRAnchor3D::get_mesh() const {
	return mesh;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

String XROrigin3D::get_configuration_warning() const {
	if (!is_visible() || !is_inside_tree()) {
		return String();
	}

	String warning = Node3D::get_configuration_warning();

	if (tracked_camera == nullptr) {
		if (!warning.empty()) {
			warning += "\n\n";
		}
		warning += TTR("XROrigin3D requires an XRCamera3D child node.");
	}

	return warning;
};

void XROrigin3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_world_scale", "world_scale"), &XROrigin3D::set_world_scale);
	ClassDB::bind_method(D_METHOD("get_world_scale"), &XROrigin3D::get_world_scale);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "world_scale"), "set_world_scale", "get_world_scale");
};

void XROrigin3D::set_tracked_camera(XRCamera3D *p_tracked_camera) {
	tracked_camera = p_tracked_camera;
};

void XROrigin3D::clear_tracked_camera_if(XRCamera3D *p_tracked_camera) {
	if (tracked_camera == p_tracked_camera) {
		tracked_camera = nullptr;
	};
};

float XROrigin3D::get_world_scale() const {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, 1.0);

	return xr_server->get_world_scale();
};

void XROrigin3D::set_world_scale(float p_world_scale) {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	xr_server->set_world_scale(p_world_scale);
};

void XROrigin3D::_notification(int p_what) {
	// get our XRServer
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			set_process_internal(true);
		}; break;
		case NOTIFICATION_EXIT_TREE: {
			set_process_internal(false);
		}; break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			// set our world origin to our node transform
			xr_server->set_world_origin(get_global_transform());

			// check if we have a primary interface
			Ref<XRInterface> xr_interface = xr_server->get_primary_interface();
			if (xr_interface.is_valid() && tracked_camera != nullptr) {
				// get our positioning transform for our headset
				Transform t = xr_interface->get_transform_for_eye(XRInterface::EYE_MONO, Transform());

				// now apply this to our camera
				tracked_camera->set_transform(t);
			};
		}; break;
		default:
			break;
	};

	// send our notification to all active XE interfaces, they may need to react to it also
	for (int i = 0; i < xr_server->get_interface_count(); i++) {
		Ref<XRInterface> interface = xr_server->get_interface(i);
		if (interface.is_valid() && interface->is_initialized()) {
			interface->notification(p_what);
		}
	}
};
