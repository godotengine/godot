/*************************************************************************/
/*  webxr_interface_js.cpp                                               */
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

#ifdef JAVASCRIPT_ENABLED

#include "webxr_interface_js.h"
#include "core/input/input.h"
#include "core/os/os.h"
#include "emscripten.h"
#include "godot_webxr.h"
#include "servers/rendering/renderer_compositor.h"
#include <stdlib.h>

void _emwebxr_on_session_supported(char *p_session_mode, int p_supported) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	String session_mode = String(p_session_mode);
	interface->emit_signal(SNAME("session_supported"), session_mode, p_supported ? true : false);
}

void _emwebxr_on_session_started(char *p_reference_space_type) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	String reference_space_type = String(p_reference_space_type);
	((WebXRInterfaceJS *)interface.ptr())->_set_reference_space_type(reference_space_type);
	interface->emit_signal(SNAME("session_started"));
}

void _emwebxr_on_session_ended() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	interface->uninitialize();
	interface->emit_signal(SNAME("session_ended"));
}

void _emwebxr_on_session_failed(char *p_message) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	interface->uninitialize();

	String message = String(p_message);
	interface->emit_signal(SNAME("session_failed"), message);
}

void _emwebxr_on_controller_changed() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	((WebXRInterfaceJS *)interface.ptr())->_on_controller_changed();
}

extern "C" EMSCRIPTEN_KEEPALIVE void _emwebxr_on_input_event(char *p_signal_name, int p_input_source) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	StringName signal_name = StringName(p_signal_name);
	interface->emit_signal(signal_name, p_input_source + 1);
}

extern "C" EMSCRIPTEN_KEEPALIVE void _emwebxr_on_simple_event(char *p_signal_name) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	Ref<XRInterface> interface = xr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	StringName signal_name = StringName(p_signal_name);
	interface->emit_signal(signal_name);
}

void WebXRInterfaceJS::is_session_supported(const String &p_session_mode) {
	godot_webxr_is_session_supported(p_session_mode.utf8().get_data(), &_emwebxr_on_session_supported);
}

void WebXRInterfaceJS::set_session_mode(String p_session_mode) {
	session_mode = p_session_mode;
}

String WebXRInterfaceJS::get_session_mode() const {
	return session_mode;
}

void WebXRInterfaceJS::set_required_features(String p_required_features) {
	required_features = p_required_features;
}

String WebXRInterfaceJS::get_required_features() const {
	return required_features;
}

void WebXRInterfaceJS::set_optional_features(String p_optional_features) {
	optional_features = p_optional_features;
}

String WebXRInterfaceJS::get_optional_features() const {
	return optional_features;
}

void WebXRInterfaceJS::set_requested_reference_space_types(String p_requested_reference_space_types) {
	requested_reference_space_types = p_requested_reference_space_types;
}

String WebXRInterfaceJS::get_requested_reference_space_types() const {
	return requested_reference_space_types;
}

void WebXRInterfaceJS::_set_reference_space_type(String p_reference_space_type) {
	reference_space_type = p_reference_space_type;
}

String WebXRInterfaceJS::get_reference_space_type() const {
	return reference_space_type;
}

Ref<XRPositionalTracker> WebXRInterfaceJS::get_controller(int p_controller_id) const {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, Ref<XRPositionalTracker>());

	// TODO support more then two controllers
	if (p_controller_id >= 0 && p_controller_id < 2) {
		return controllers[p_controller_id];
	};

	return Ref<XRPositionalTracker>();
}

String WebXRInterfaceJS::get_visibility_state() const {
	char *c_str = godot_webxr_get_visibility_state();
	if (c_str) {
		String visibility_state = String(c_str);
		free(c_str);

		return visibility_state;
	}
	return String();
}

PackedVector3Array WebXRInterfaceJS::get_bounds_geometry() const {
	PackedVector3Array ret;

	int *js_bounds = godot_webxr_get_bounds_geometry();
	if (js_bounds) {
		ret.resize(js_bounds[0]);
		for (int i = 0; i < js_bounds[0]; i++) {
			float *js_vector3 = ((float *)js_bounds) + (i * 3) + 1;
			ret.set(i, Vector3(js_vector3[0], js_vector3[1], js_vector3[2]));
		}
		free(js_bounds);
	}

	return ret;
}

StringName WebXRInterfaceJS::get_name() const {
	return "WebXR";
};

uint32_t WebXRInterfaceJS::get_capabilities() const {
	return XRInterface::XR_STEREO | XRInterface::XR_MONO;
};

uint32_t WebXRInterfaceJS::get_view_count() {
	return godot_webxr_get_view_count();
};

bool WebXRInterfaceJS::is_initialized() const {
	return (initialized);
};

bool WebXRInterfaceJS::initialize() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	if (!initialized) {
		if (!godot_webxr_is_supported()) {
			return false;
		}

		if (requested_reference_space_types.size() == 0) {
			return false;
		}

		// we must create a tracker for our head
		head_tracker.instantiate();
		head_tracker->set_tracker_type(XRServer::TRACKER_HEAD);
		head_tracker->set_tracker_name("head");
		head_tracker->set_tracker_desc("Players head");
		xr_server->add_tracker(head_tracker);

		// make this our primary interface
		xr_server->set_primary_interface(this);

		// Clear render_targetsize to make sure it gets reset to the new size.
		// Clearing in uninitialize() doesn't work because a frame can still be
		// rendered after it's called, which will fill render_targetsize again.
		render_targetsize.width = 0;
		render_targetsize.height = 0;

		initialized = true;

		godot_webxr_initialize(
				session_mode.utf8().get_data(),
				required_features.utf8().get_data(),
				optional_features.utf8().get_data(),
				requested_reference_space_types.utf8().get_data(),
				&_emwebxr_on_session_started,
				&_emwebxr_on_session_ended,
				&_emwebxr_on_session_failed,
				&_emwebxr_on_controller_changed,
				&_emwebxr_on_input_event,
				&_emwebxr_on_simple_event);
	};

	return true;
};

void WebXRInterfaceJS::uninitialize() {
	if (initialized) {
		XRServer *xr_server = XRServer::get_singleton();
		if (xr_server != nullptr) {
			if (head_tracker.is_valid()) {
				xr_server->remove_tracker(head_tracker);

				head_tracker.unref();
			}

			if (xr_server->get_primary_interface() == this) {
				// no longer our primary interface
				xr_server->set_primary_interface(nullptr);
			}
		}

		godot_webxr_uninitialize();

		reference_space_type = "";
		initialized = false;
	};
};

Transform3D WebXRInterfaceJS::_js_matrix_to_transform(float *p_js_matrix) {
	Transform3D transform;

	transform.basis.elements[0].x = p_js_matrix[0];
	transform.basis.elements[1].x = p_js_matrix[1];
	transform.basis.elements[2].x = p_js_matrix[2];
	transform.basis.elements[0].y = p_js_matrix[4];
	transform.basis.elements[1].y = p_js_matrix[5];
	transform.basis.elements[2].y = p_js_matrix[6];
	transform.basis.elements[0].z = p_js_matrix[8];
	transform.basis.elements[1].z = p_js_matrix[9];
	transform.basis.elements[2].z = p_js_matrix[10];
	transform.origin.x = p_js_matrix[12];
	transform.origin.y = p_js_matrix[13];
	transform.origin.z = p_js_matrix[14];

	return transform;
}

Size2 WebXRInterfaceJS::get_render_target_size() {
	if (render_targetsize.width != 0 && render_targetsize.height != 0) {
		return render_targetsize;
	}

	int *js_size = godot_webxr_get_render_target_size();
	if (!initialized || js_size == nullptr) {
		// As a temporary default (until WebXR is fully initialized), use half the window size.
		Size2 temp = DisplayServer::get_singleton()->window_get_size();
		temp.width /= 2.0;
		return temp;
	}

	render_targetsize.width = js_size[0];
	render_targetsize.height = js_size[1];

	free(js_size);

	return render_targetsize;
};

Transform3D WebXRInterfaceJS::get_camera_transform() {
	Transform3D transform_for_eye;

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, transform_for_eye);

	float *js_matrix = godot_webxr_get_transform_for_eye(0);
	if (!initialized || js_matrix == nullptr) {
		return transform_for_eye;
	}

	transform_for_eye = _js_matrix_to_transform(js_matrix);
	free(js_matrix);

	return xr_server->get_reference_frame() * transform_for_eye;
};

Transform3D WebXRInterfaceJS::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	Transform3D transform_for_eye;

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, transform_for_eye);

	float *js_matrix = godot_webxr_get_transform_for_eye(p_view + 1);
	if (!initialized || js_matrix == nullptr) {
		transform_for_eye = p_cam_transform;
		return transform_for_eye;
	}

	transform_for_eye = _js_matrix_to_transform(js_matrix);
	free(js_matrix);

	return p_cam_transform * xr_server->get_reference_frame() * transform_for_eye;
};

CameraMatrix WebXRInterfaceJS::get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) {
	CameraMatrix eye;

	float *js_matrix = godot_webxr_get_projection_for_eye(p_view + 1);
	if (!initialized || js_matrix == nullptr) {
		return eye;
	}

	int k = 0;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			eye.matrix[i][j] = js_matrix[k++];
		}
	}

	free(js_matrix);

	// Copied from godot_oculus_mobile's ovr_mobile_session.cpp
	eye.matrix[2][2] = -(p_z_far + p_z_near) / (p_z_far - p_z_near);
	eye.matrix[3][2] = -(2.0f * p_z_far * p_z_near) / (p_z_far - p_z_near);

	return eye;
}

Vector<BlitToScreen> WebXRInterfaceJS::commit_views(RID p_render_target, const Rect2 &p_screen_rect) {
	Vector<BlitToScreen> blit_to_screen;

	if (!initialized) {
		return blit_to_screen;
	}

	// @todo Refactor this to be based on "views" rather than "eyes".
	godot_webxr_commit_for_eye(1);
	if (godot_webxr_get_view_count() > 1) {
		godot_webxr_commit_for_eye(2);
	}

	return blit_to_screen;
};

void WebXRInterfaceJS::process() {
	if (initialized) {
		godot_webxr_sample_controller_data();

		if (head_tracker.is_valid()) {
			// TODO set default pose to our head location (i.e. get_camera_transform without world scale and reference frame applied)
			// head_tracker->set_pose("default", head_transform, Vector3(), Vector3());
		}

		int controller_count = godot_webxr_get_controller_count();
		if (controller_count == 0) {
			return;
		}

		for (int i = 0; i < controller_count; i++) {
			_update_tracker(i);
		}
	};
};

void WebXRInterfaceJS::_update_tracker(int p_controller_id) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	// need to support more then two controllers...
	if (p_controller_id < 0 || p_controller_id > 1) {
		return;
	}

	Ref<XRPositionalTracker> tracker = controllers[p_controller_id];
	if (godot_webxr_is_controller_connected(p_controller_id)) {
		if (tracker.is_null()) {
			tracker.instantiate();
			tracker->set_tracker_type(XRServer::TRACKER_CONTROLLER);
			// Controller id's 0 and 1 are always the left and right hands.
			if (p_controller_id < 2) {
				tracker->set_tracker_name(p_controller_id == 0 ? "left_hand" : "right_hand");
				tracker->set_tracker_desc(p_controller_id == 0 ? "Left hand controller" : "Right hand controller");
				tracker->set_tracker_hand(p_controller_id == 0 ? XRPositionalTracker::TRACKER_HAND_LEFT : XRPositionalTracker::TRACKER_HAND_RIGHT);
			} else {
				char name[1024];
				sprintf(name, "tracker_%i", p_controller_id);
				tracker->set_tracker_name(name);
				tracker->set_tracker_desc(name);
			}
			xr_server->add_tracker(tracker);
		}

		float *tracker_matrix = godot_webxr_get_controller_transform(p_controller_id);
		if (tracker_matrix) {
			// Note, poses should NOT have world scale and our reference frame applied!
			Transform3D transform = _js_matrix_to_transform(tracker_matrix);
			tracker->set_pose("default", transform, Vector3(), Vector3());
			free(tracker_matrix);
		}

		// TODO implement additional poses such as "aim" and "grip"

		int *buttons = godot_webxr_get_controller_buttons(p_controller_id);
		if (buttons) {
			// TODO buttons should be named properly, this is just a temporary fix
			for (int i = 0; i < buttons[0]; i++) {
				char name[1024];
				sprintf(name, "button_%i", i);

				float value = *((float *)buttons + (i + 1));
				bool state = value > 0.0;
				tracker->set_input(name, state);
			}
			free(buttons);
		}

		int *axes = godot_webxr_get_controller_axes(p_controller_id);
		if (axes) {
			// TODO again just a temporary fix, split these between proper float and vector2 inputs
			for (int i = 0; i < axes[0]; i++) {
				char name[1024];
				sprintf(name, "axis_%i", i);

				float value = *((float *)axes + (i + 1));
				;
				tracker->set_input(name, value);
			}
			free(axes);
		}
	} else if (tracker.is_valid()) {
		xr_server->remove_tracker(tracker);
		controllers[p_controller_id].unref();
	}
}

void WebXRInterfaceJS::_on_controller_changed() {
	// Register "virtual" gamepads with Godot for the ones we get from WebXR.
	godot_webxr_sample_controller_data();
	for (int i = 0; i < 2; i++) {
		bool controller_connected = godot_webxr_is_controller_connected(i);
		if (controllers_state[i] != controller_connected) {
			// Input::get_singleton()->joy_connection_changed(i + 100, controller_connected, i == 0 ? "Left" : "Right", "");
			controllers_state[i] = controller_connected;
		}
	}
}

WebXRInterfaceJS::WebXRInterfaceJS() {
	initialized = false;
	session_mode = "inline";
	requested_reference_space_types = "local";
};

WebXRInterfaceJS::~WebXRInterfaceJS() {
	// and make sure we cleanup if we haven't already
	if (initialized) {
		uninitialize();
	};
};

#endif // JAVASCRIPT_ENABLED
