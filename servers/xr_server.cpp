/*************************************************************************/
/*  xr_server.cpp                                                        */
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

#include "xr_server.h"
#include "core/project_settings.h"
#include "xr/xr_interface.h"
#include "xr/xr_positional_tracker.h"

XRServer *XRServer::singleton = nullptr;

XRServer *XRServer::get_singleton() {
	return singleton;
};

void XRServer::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_world_scale"), &XRServer::get_world_scale);
	ClassDB::bind_method(D_METHOD("set_world_scale"), &XRServer::set_world_scale);
	ClassDB::bind_method(D_METHOD("get_reference_frame"), &XRServer::get_reference_frame);
	ClassDB::bind_method(D_METHOD("center_on_hmd", "rotation_mode", "keep_height"), &XRServer::center_on_hmd);
	ClassDB::bind_method(D_METHOD("get_hmd_transform"), &XRServer::get_hmd_transform);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "world_scale"), "set_world_scale", "get_world_scale");

	ClassDB::bind_method(D_METHOD("get_interface_count"), &XRServer::get_interface_count);
	ClassDB::bind_method(D_METHOD("get_interface", "idx"), &XRServer::get_interface);
	ClassDB::bind_method(D_METHOD("get_interfaces"), &XRServer::get_interfaces);
	ClassDB::bind_method(D_METHOD("find_interface", "name"), &XRServer::find_interface);
	ClassDB::bind_method(D_METHOD("get_tracker_count"), &XRServer::get_tracker_count);
	ClassDB::bind_method(D_METHOD("get_tracker", "idx"), &XRServer::get_tracker);

	ClassDB::bind_method(D_METHOD("get_primary_interface"), &XRServer::get_primary_interface);
	ClassDB::bind_method(D_METHOD("set_primary_interface", "interface"), &XRServer::set_primary_interface);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "primary_interface"), "set_primary_interface", "get_primary_interface");

	ClassDB::bind_method(D_METHOD("get_last_process_usec"), &XRServer::get_last_process_usec);
	ClassDB::bind_method(D_METHOD("get_last_commit_usec"), &XRServer::get_last_commit_usec);
	ClassDB::bind_method(D_METHOD("get_last_frame_usec"), &XRServer::get_last_frame_usec);

	BIND_ENUM_CONSTANT(TRACKER_CONTROLLER);
	BIND_ENUM_CONSTANT(TRACKER_BASESTATION);
	BIND_ENUM_CONSTANT(TRACKER_ANCHOR);
	BIND_ENUM_CONSTANT(TRACKER_ANY_KNOWN);
	BIND_ENUM_CONSTANT(TRACKER_UNKNOWN);
	BIND_ENUM_CONSTANT(TRACKER_ANY);

	BIND_ENUM_CONSTANT(RESET_FULL_ROTATION);
	BIND_ENUM_CONSTANT(RESET_BUT_KEEP_TILT);
	BIND_ENUM_CONSTANT(DONT_RESET_ROTATION);

	ADD_SIGNAL(MethodInfo("interface_added", PropertyInfo(Variant::STRING_NAME, "interface_name")));
	ADD_SIGNAL(MethodInfo("interface_removed", PropertyInfo(Variant::STRING_NAME, "interface_name")));

	ADD_SIGNAL(MethodInfo("tracker_added", PropertyInfo(Variant::STRING_NAME, "tracker_name"), PropertyInfo(Variant::INT, "type"), PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("tracker_removed", PropertyInfo(Variant::STRING_NAME, "tracker_name"), PropertyInfo(Variant::INT, "type"), PropertyInfo(Variant::INT, "id")));
};

real_t XRServer::get_world_scale() const {
	return world_scale;
};

void XRServer::set_world_scale(real_t p_world_scale) {
	if (p_world_scale < 0.01) {
		p_world_scale = 0.01;
	} else if (p_world_scale > 1000.0) {
		p_world_scale = 1000.0;
	}

	world_scale = p_world_scale;
};

Transform XRServer::get_world_origin() const {
	return world_origin;
};

void XRServer::set_world_origin(const Transform &p_world_origin) {
	world_origin = p_world_origin;
};

Transform XRServer::get_reference_frame() const {
	return reference_frame;
};

void XRServer::center_on_hmd(RotationMode p_rotation_mode, bool p_keep_height) {
	if (primary_interface != nullptr) {
		// clear our current reference frame or we'll end up double adjusting it
		reference_frame = Transform();

		// requesting our EYE_MONO transform should return our current HMD position
		Transform new_reference_frame = primary_interface->get_transform_for_eye(XRInterface::EYE_MONO, Transform());

		// remove our tilt
		if (p_rotation_mode == 1) {
			// take the Y out of our Z
			new_reference_frame.basis.set_axis(2, Vector3(new_reference_frame.basis.elements[0][2], 0.0, new_reference_frame.basis.elements[2][2]).normalized());

			// Y is straight up
			new_reference_frame.basis.set_axis(1, Vector3(0.0, 1.0, 0.0));

			// and X is our cross reference
			new_reference_frame.basis.set_axis(0, new_reference_frame.basis.get_axis(1).cross(new_reference_frame.basis.get_axis(2)).normalized());
		} else if (p_rotation_mode == 2) {
			// remove our rotation, we're only interesting in centering on position
			new_reference_frame.basis = Basis();
		};

		// don't negate our height
		if (p_keep_height) {
			new_reference_frame.origin.y = 0.0;
		};

		reference_frame = new_reference_frame.inverse();
	};
};

Transform XRServer::get_hmd_transform() {
	Transform hmd_transform;
	if (primary_interface != nullptr) {
		hmd_transform = primary_interface->get_transform_for_eye(XRInterface::EYE_MONO, hmd_transform);
	};
	return hmd_transform;
};

void XRServer::add_interface(const Ref<XRInterface> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());

	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			ERR_PRINT("Interface was already added");
			return;
		};
	};

	interfaces.push_back(p_interface);
	emit_signal("interface_added", p_interface->get_name());
};

void XRServer::remove_interface(const Ref<XRInterface> &p_interface) {
	ERR_FAIL_COND(p_interface.is_null());

	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i] == p_interface) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND(idx == -1);

	print_verbose("XR: Removed interface" + p_interface->get_name());

	emit_signal("interface_removed", p_interface->get_name());
	interfaces.remove(idx);
};

int XRServer::get_interface_count() const {
	return interfaces.size();
};

Ref<XRInterface> XRServer::get_interface(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, interfaces.size(), nullptr);

	return interfaces[p_index];
};

Ref<XRInterface> XRServer::find_interface(const String &p_name) const {
	int idx = -1;
	for (int i = 0; i < interfaces.size(); i++) {
		if (interfaces[i]->get_name() == p_name) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND_V(idx == -1, nullptr);

	return interfaces[idx];
};

Array XRServer::get_interfaces() const {
	Array ret;

	for (int i = 0; i < interfaces.size(); i++) {
		Dictionary iface_info;

		iface_info["id"] = i;
		iface_info["name"] = interfaces[i]->get_name();

		ret.push_back(iface_info);
	};

	return ret;
};

/*
	A little extra info on the tracker ids, these are unique per tracker type so we get some consistency in recognising our trackers, specifically controllers.

	The first controller that is turned of will get ID 1, the second will get ID 2, etc.
	The magic happens when one of the controllers is turned off, say controller 1 turns off, controller 2 will remain controller 2, controller 3 will remain controller 3.
	If controller number 1 is turned on again it again gets ID 1 unless another new controller was turned on since.

	The most likely scenario however is a controller that runs out of battery and another controller being used to replace it.
	Because the controllers are often linked to physical objects, say you're holding a shield in controller 1, your left hand, and a gun in controller 2, your right hand, and controller 1 dies:
	- using our tracker index would suddenly make the gun disappear and the shield jump into your right hand because controller 2 becomes controller 1.
	- using this approach the shield disappears or is no longer tracked, but the gun stays firmly in your right hand because that is still controller 2, further more, if controller 1 is replaced the shield will return.
*/

bool XRServer::is_tracker_id_in_use_for_type(TrackerType p_tracker_type, int p_tracker_id) const {
	for (int i = 0; i < trackers.size(); i++) {
		if (trackers[i]->get_type() == p_tracker_type && trackers[i]->get_tracker_id() == p_tracker_id) {
			return true;
		};
	};

	// all good
	return false;
};

int XRServer::get_free_tracker_id_for_type(TrackerType p_tracker_type) {
	// We start checking at 1, 0 means that it's not a controller..
	// Note that for controller we reserve:
	// - 1 for the left hand controller and
	// - 2 for the right hand controller
	// so we start at 3 :)
	int tracker_id = p_tracker_type == XRServer::TRACKER_CONTROLLER ? 3 : 1;

	while (is_tracker_id_in_use_for_type(p_tracker_type, tracker_id)) {
		// try the next one
		tracker_id++;
	};

	return tracker_id;
};

void XRServer::add_tracker(XRPositionalTracker *p_tracker) {
	ERR_FAIL_NULL(p_tracker);

	trackers.push_back(p_tracker);
	emit_signal("tracker_added", p_tracker->get_name(), p_tracker->get_type(), p_tracker->get_tracker_id());
};

void XRServer::remove_tracker(XRPositionalTracker *p_tracker) {
	ERR_FAIL_NULL(p_tracker);

	int idx = -1;
	for (int i = 0; i < trackers.size(); i++) {
		if (trackers[i] == p_tracker) {
			idx = i;
			break;
		};
	};

	ERR_FAIL_COND(idx == -1);

	emit_signal("tracker_removed", p_tracker->get_name(), p_tracker->get_type(), p_tracker->get_tracker_id());
	trackers.remove(idx);
};

int XRServer::get_tracker_count() const {
	return trackers.size();
};

XRPositionalTracker *XRServer::get_tracker(int p_index) const {
	ERR_FAIL_INDEX_V(p_index, trackers.size(), nullptr);

	return trackers[p_index];
};

XRPositionalTracker *XRServer::find_by_type_and_id(TrackerType p_tracker_type, int p_tracker_id) const {
	ERR_FAIL_COND_V(p_tracker_id == 0, nullptr);

	for (int i = 0; i < trackers.size(); i++) {
		if (trackers[i]->get_type() == p_tracker_type && trackers[i]->get_tracker_id() == p_tracker_id) {
			return trackers[i];
		};
	};

	return nullptr;
};

Ref<XRInterface> XRServer::get_primary_interface() const {
	return primary_interface;
};

void XRServer::set_primary_interface(const Ref<XRInterface> &p_primary_interface) {
	primary_interface = p_primary_interface;

	print_verbose("XR: Primary interface set to: " + primary_interface->get_name());
};

void XRServer::clear_primary_interface_if(const Ref<XRInterface> &p_primary_interface) {
	if (primary_interface == p_primary_interface) {
		print_verbose("XR: Clearing primary interface");
		primary_interface.unref();
	};
};

uint64_t XRServer::get_last_process_usec() {
	return last_process_usec;
};

uint64_t XRServer::get_last_commit_usec() {
	return last_commit_usec;
};

uint64_t XRServer::get_last_frame_usec() {
	return last_frame_usec;
};

void XRServer::_process() {
	/* called from rendering_server_viewport.draw_viewports right before we start drawing our viewports */

	/* mark for our frame timing */
	last_process_usec = OS::get_singleton()->get_ticks_usec();

	/* process all active interfaces */
	for (int i = 0; i < interfaces.size(); i++) {
		if (!interfaces[i].is_valid()) {
			// ignore, not a valid reference
		} else if (interfaces[i]->is_initialized()) {
			interfaces.write[i]->process();
		};
	};
};

void XRServer::_mark_commit() {
	/* time this */
	last_commit_usec = OS::get_singleton()->get_ticks_usec();

	/* now store our difference as we may overwrite last_process_usec before this is accessed */
	last_frame_usec = last_commit_usec - last_process_usec;
};

XRServer::XRServer() {
	singleton = this;
	world_scale = 1.0;
};

XRServer::~XRServer() {
	primary_interface.unref();

	while (interfaces.size() > 0) {
		interfaces.remove(0);
	}

	while (trackers.size() > 0) {
		trackers.remove(0);
	}

	singleton = nullptr;
};
