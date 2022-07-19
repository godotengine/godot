/*************************************************************************/
/*  webxr_interface_js.cpp                                               */
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

#ifdef JAVASCRIPT_ENABLED

#include "webxr_interface_js.h"
#include "core/os/input.h"
#include "core/os/os.h"
#include "emscripten.h"
#include "godot_webxr.h"
#include "main/input_default.h"
#include "servers/visual/visual_server_globals.h"
#include <stdlib.h>

void _emwebxr_on_session_supported(char *p_session_mode, int p_supported) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	String session_mode = String(p_session_mode);
	interface->emit_signal("session_supported", session_mode, p_supported ? true : false);
}

void _emwebxr_on_session_started(char *p_reference_space_type) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	String reference_space_type = String(p_reference_space_type);
	((WebXRInterfaceJS *)interface.ptr())->_set_reference_space_type(reference_space_type);
	interface->emit_signal("session_started");
}

void _emwebxr_on_session_ended() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	interface->uninitialize();
	interface->emit_signal("session_ended");
}

void _emwebxr_on_session_failed(char *p_message) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	interface->uninitialize();

	String message = String(p_message);
	interface->emit_signal("session_failed", message);
}

void _emwebxr_on_controller_changed() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	((WebXRInterfaceJS *)interface.ptr())->_on_controller_changed();
}

extern "C" EMSCRIPTEN_KEEPALIVE void _emwebxr_on_input_event(char *p_signal_name, int p_input_source) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
	ERR_FAIL_COND(interface.is_null());

	StringName signal_name = StringName(p_signal_name);
	interface->emit_signal(signal_name, p_input_source + 1);
}

extern "C" EMSCRIPTEN_KEEPALIVE void _emwebxr_on_simple_event(char *p_signal_name) {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	Ref<ARVRInterface> interface = arvr_server->find_interface("WebXR");
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

ARVRPositionalTracker *WebXRInterfaceJS::get_controller(int p_controller_id) const {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, nullptr);

	return arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, p_controller_id);
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

PoolVector3Array WebXRInterfaceJS::get_bounds_geometry() const {
	PoolVector3Array ret;

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

int WebXRInterfaceJS::get_capabilities() const {
	return ARVRInterface::ARVR_STEREO | ARVRInterface::ARVR_MONO;
};

bool WebXRInterfaceJS::is_stereo() {
	return godot_webxr_get_view_count() == 2;
};

bool WebXRInterfaceJS::is_initialized() const {
	return (initialized);
};

bool WebXRInterfaceJS::initialize() {
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, false);

	if (!initialized) {
		if (!godot_webxr_is_supported()) {
			return false;
		}

		if (requested_reference_space_types.size() == 0) {
			return false;
		}

		// make this our primary interface
		arvr_server->set_primary_interface(this);

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
		ARVRServer *arvr_server = ARVRServer::get_singleton();
		if (arvr_server != NULL) {
			// no longer our primary interface
			arvr_server->clear_primary_interface_if(this);
		}

		godot_webxr_uninitialize();

		reference_space_type = "";
		initialized = false;
	};
};

Transform WebXRInterfaceJS::_js_matrix_to_transform(float *p_js_matrix) {
	Transform transform;

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

Size2 WebXRInterfaceJS::get_render_targetsize() {
	if (render_targetsize.width != 0 && render_targetsize.height != 0) {
		return render_targetsize;
	}

	int *js_size = godot_webxr_get_render_targetsize();
	if (!initialized || js_size == nullptr) {
		// As a temporary default (until WebXR is fully initialized), use half the window size.
		Size2 temp = OS::get_singleton()->get_window_size();
		temp.width /= 2.0;
		return temp;
	}

	render_targetsize.width = js_size[0];
	render_targetsize.height = js_size[1];

	free(js_size);

	return render_targetsize;
};

Transform WebXRInterfaceJS::get_transform_for_eye(ARVRInterface::Eyes p_eye, const Transform &p_cam_transform) {
	Transform transform_for_eye;

	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL_V(arvr_server, transform_for_eye);

	float *js_matrix = godot_webxr_get_transform_for_eye(p_eye);
	if (!initialized || js_matrix == nullptr) {
		transform_for_eye = p_cam_transform;
		return transform_for_eye;
	}

	transform_for_eye = _js_matrix_to_transform(js_matrix);
	free(js_matrix);

	return p_cam_transform * arvr_server->get_reference_frame() * transform_for_eye;
};

CameraMatrix WebXRInterfaceJS::get_projection_for_eye(ARVRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	CameraMatrix eye;

	float *js_matrix = godot_webxr_get_projection_for_eye(p_eye);
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

unsigned int WebXRInterfaceJS::get_external_texture_for_eye(ARVRInterface::Eyes p_eye) {
	if (!initialized) {
		return 0;
	}
	return godot_webxr_get_external_texture_for_eye(p_eye);
}

void WebXRInterfaceJS::commit_for_eye(ARVRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) {
	if (!initialized) {
		return;
	}
	godot_webxr_commit_for_eye(p_eye);
};

void WebXRInterfaceJS::process() {
	if (initialized) {
		godot_webxr_sample_controller_data();

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
	ARVRServer *arvr_server = ARVRServer::get_singleton();
	ERR_FAIL_NULL(arvr_server);

	ARVRPositionalTracker *tracker = arvr_server->find_by_type_and_id(ARVRServer::TRACKER_CONTROLLER, p_controller_id + 1);
	if (godot_webxr_is_controller_connected(p_controller_id)) {
		if (tracker == nullptr) {
			tracker = memnew(ARVRPositionalTracker);
			tracker->set_type(ARVRServer::TRACKER_CONTROLLER);
			// Controller id's 0 and 1 are always the left and right hands.
			if (p_controller_id < 2) {
				tracker->set_name(p_controller_id == 0 ? "Left" : "Right");
				tracker->set_hand(p_controller_id == 0 ? ARVRPositionalTracker::TRACKER_LEFT_HAND : ARVRPositionalTracker::TRACKER_RIGHT_HAND);
			}
			// Use the ids we're giving to our "virtual" gamepads.
			tracker->set_joy_id(p_controller_id + 100);
			arvr_server->add_tracker(tracker);
		}

		InputDefault *input = (InputDefault *)Input::get_singleton();

		float *tracker_matrix = godot_webxr_get_controller_transform(p_controller_id);
		if (tracker_matrix) {
			Transform transform = _js_matrix_to_transform(tracker_matrix);
			tracker->set_position(transform.origin);
			tracker->set_orientation(transform.basis);
			free(tracker_matrix);
		}

		int *buttons = godot_webxr_get_controller_buttons(p_controller_id);
		if (buttons) {
			for (int i = 0; i < buttons[0]; i++) {
				input->joy_button(p_controller_id + 100, i, *((float *)buttons + (i + 1)));
			}
			free(buttons);
		}

		int *axes = godot_webxr_get_controller_axes(p_controller_id);
		if (axes) {
			for (int i = 0; i < axes[0]; i++) {
				InputDefault::JoyAxis joy_axis;
				joy_axis.min = -1;
				joy_axis.value = *((float *)axes + (i + 1));
				input->joy_axis(p_controller_id + 100, i, joy_axis);
			}
			free(axes);
		}
	} else if (tracker) {
		arvr_server->remove_tracker(tracker);
	}
}

void WebXRInterfaceJS::_on_controller_changed() {
	// Register "virtual" gamepads with Godot for the ones we get from WebXR.
	godot_webxr_sample_controller_data();
	for (int i = 0; i < 2; i++) {
		bool controller_connected = godot_webxr_is_controller_connected(i);
		if (controllers_state[i] != controller_connected) {
			Input::get_singleton()->joy_connection_changed(i + 100, controller_connected, i == 0 ? "Left" : "Right", "");
			controllers_state[i] = controller_connected;
		}
	}
}

void WebXRInterfaceJS::notification(int p_what) {
	// Nothing to do here.
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
