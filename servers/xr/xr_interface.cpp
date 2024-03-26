/**************************************************************************/
/*  xr_interface.cpp                                                      */
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

#include "xr_interface.h"
#include "servers/rendering/renderer_compositor.h"

void XRInterface::_bind_methods() {
	ADD_SIGNAL(MethodInfo("play_area_changed", PropertyInfo(Variant::INT, "mode")));

	ClassDB::bind_method(D_METHOD("get_name"), &XRInterface::get_name);
	ClassDB::bind_method(D_METHOD("get_capabilities"), &XRInterface::get_capabilities);

	ClassDB::bind_method(D_METHOD("is_primary"), &XRInterface::is_primary);
	ClassDB::bind_method(D_METHOD("set_primary", "primary"), &XRInterface::set_primary);

	ClassDB::bind_method(D_METHOD("is_initialized"), &XRInterface::is_initialized);
	ClassDB::bind_method(D_METHOD("initialize"), &XRInterface::initialize);
	ClassDB::bind_method(D_METHOD("uninitialize"), &XRInterface::uninitialize);
	ClassDB::bind_method(D_METHOD("get_system_info"), &XRInterface::get_system_info);

	ClassDB::bind_method(D_METHOD("get_tracking_status"), &XRInterface::get_tracking_status);

	ClassDB::bind_method(D_METHOD("get_render_target_size"), &XRInterface::get_render_target_size);
	ClassDB::bind_method(D_METHOD("get_view_count"), &XRInterface::get_view_count);

	ClassDB::bind_method(D_METHOD("trigger_haptic_pulse", "action_name", "tracker_name", "frequency", "amplitude", "duration_sec", "delay_sec"), &XRInterface::trigger_haptic_pulse);

	ADD_GROUP("Interface", "interface_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "interface_is_primary"), "set_primary", "is_primary");

	// methods and properties specific to VR...
	ClassDB::bind_method(D_METHOD("supports_play_area_mode", "mode"), &XRInterface::supports_play_area_mode);
	ClassDB::bind_method(D_METHOD("get_play_area_mode"), &XRInterface::get_play_area_mode);
	ClassDB::bind_method(D_METHOD("set_play_area_mode", "mode"), &XRInterface::set_play_area_mode);
	ClassDB::bind_method(D_METHOD("get_play_area"), &XRInterface::get_play_area);

	ClassDB::bind_method(D_METHOD("get_vrs_min_radius"), &XRInterface::get_vrs_min_radius);
	ClassDB::bind_method(D_METHOD("set_vrs_min_radius", "radius"), &XRInterface::set_vrs_min_radius);

	ClassDB::bind_method(D_METHOD("get_vrs_strength"), &XRInterface::get_vrs_strength);
	ClassDB::bind_method(D_METHOD("set_vrs_strength", "strength"), &XRInterface::set_vrs_strength);

	ADD_GROUP("XR", "xr_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "xr_play_area_mode", PROPERTY_HINT_ENUM, "Unknown,3DOF,Sitting,Roomscale,Stage"), "set_play_area_mode", "get_play_area_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "xr_vrs_min_radius", PROPERTY_HINT_RANGE, "1.0,100.0,1.0"), "set_vrs_min_radius", "get_vrs_min_radius");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "xr_vrs_strength", PROPERTY_HINT_RANGE, "0.1,10.0,0.1"), "set_vrs_strength", "get_vrs_strength");

	// methods and properties specific to AR....
	ClassDB::bind_method(D_METHOD("get_anchor_detection_is_enabled"), &XRInterface::get_anchor_detection_is_enabled);
	ClassDB::bind_method(D_METHOD("set_anchor_detection_is_enabled", "enable"), &XRInterface::set_anchor_detection_is_enabled);
	ClassDB::bind_method(D_METHOD("get_camera_feed_id"), &XRInterface::get_camera_feed_id);

	ClassDB::bind_method(D_METHOD("is_passthrough_supported"), &XRInterface::is_passthrough_supported);
	ClassDB::bind_method(D_METHOD("is_passthrough_enabled"), &XRInterface::is_passthrough_enabled);
	ClassDB::bind_method(D_METHOD("start_passthrough"), &XRInterface::start_passthrough);
	ClassDB::bind_method(D_METHOD("stop_passthrough"), &XRInterface::stop_passthrough);
	ClassDB::bind_method(D_METHOD("get_transform_for_view", "view", "cam_transform"), &XRInterface::get_transform_for_view);
	ClassDB::bind_method(D_METHOD("get_projection_for_view", "view", "aspect", "near", "far"), &XRInterface::get_projection_for_view);

	/** environment blend mode. */
	ClassDB::bind_method(D_METHOD("get_supported_environment_blend_modes"), &XRInterface::get_supported_environment_blend_modes);
	ClassDB::bind_method(D_METHOD("set_environment_blend_mode", "mode"), &XRInterface::set_environment_blend_mode);
	ClassDB::bind_method(D_METHOD("get_environment_blend_mode"), &XRInterface::get_environment_blend_mode);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "environment_blend_mode"), "set_environment_blend_mode", "get_environment_blend_mode");

	ADD_GROUP("AR", "ar_");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "ar_is_anchor_detection_enabled"), "set_anchor_detection_is_enabled", "get_anchor_detection_is_enabled");

	BIND_ENUM_CONSTANT(XR_NONE);
	BIND_ENUM_CONSTANT(XR_MONO);
	BIND_ENUM_CONSTANT(XR_STEREO);
	BIND_ENUM_CONSTANT(XR_QUAD);
	BIND_ENUM_CONSTANT(XR_VR);
	BIND_ENUM_CONSTANT(XR_AR);
	BIND_ENUM_CONSTANT(XR_EXTERNAL);

	BIND_ENUM_CONSTANT(XR_NORMAL_TRACKING);
	BIND_ENUM_CONSTANT(XR_EXCESSIVE_MOTION);
	BIND_ENUM_CONSTANT(XR_INSUFFICIENT_FEATURES);
	BIND_ENUM_CONSTANT(XR_UNKNOWN_TRACKING);
	BIND_ENUM_CONSTANT(XR_NOT_TRACKING);

	BIND_ENUM_CONSTANT(XR_PLAY_AREA_UNKNOWN);
	BIND_ENUM_CONSTANT(XR_PLAY_AREA_3DOF);
	BIND_ENUM_CONSTANT(XR_PLAY_AREA_SITTING);
	BIND_ENUM_CONSTANT(XR_PLAY_AREA_ROOMSCALE);
	BIND_ENUM_CONSTANT(XR_PLAY_AREA_STAGE);

	BIND_ENUM_CONSTANT(XR_ENV_BLEND_MODE_OPAQUE);
	BIND_ENUM_CONSTANT(XR_ENV_BLEND_MODE_ADDITIVE);
	BIND_ENUM_CONSTANT(XR_ENV_BLEND_MODE_ALPHA_BLEND);
};

bool XRInterface::is_primary() {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	return xr_server->get_primary_interface() == this;
}

void XRInterface::set_primary(bool p_primary) {
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL(xr_server);

	if (p_primary) {
		ERR_FAIL_COND(!is_initialized());

		xr_server->set_primary_interface(this);
	} else if (xr_server->get_primary_interface() == this) {
		xr_server->set_primary_interface(nullptr);
	}
}

XRInterface::XRInterface() {}

XRInterface::~XRInterface() {
	if (vrs.vrs_texture.is_valid()) {
		ERR_FAIL_NULL(RenderingServer::get_singleton());
		RS::get_singleton()->free(vrs.vrs_texture);
		vrs.vrs_texture = RID();
	}
}

// query if this interface supports this play area mode
bool XRInterface::supports_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_UNKNOWN;
}

// get the current play area mode
XRInterface::PlayAreaMode XRInterface::get_play_area_mode() const {
	return XR_PLAY_AREA_UNKNOWN;
}

// change the play area mode, note that this should return false if the mode is not available
bool XRInterface::set_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_UNKNOWN;
}

// if available, returns an array of vectors denoting the play area the player can move around in
PackedVector3Array XRInterface::get_play_area() const {
	// Return an empty array by default.
	// Note implementation is responsible for applying our reference frame and world scale to the raw data.
	// `play_area_changed` should be emitted if play area data is available and either the reference frame or world scale changes.
	return PackedVector3Array();
};

/** these will only be implemented on AR interfaces, so we want dummies for VR **/
bool XRInterface::get_anchor_detection_is_enabled() const {
	return false;
}

void XRInterface::set_anchor_detection_is_enabled(bool p_enable) {
}

int XRInterface::get_camera_feed_id() {
	return 0;
}

float XRInterface::get_vrs_min_radius() const {
	return vrs_min_radius;
}

void XRInterface::set_vrs_min_radius(float p_vrs_min_radius) {
	if (p_vrs_min_radius < 1.0) {
		WARN_PRINT_ONCE("VRS minimum radius can not be set below 1.0");
		vrs_min_radius = 1.0;
	} else if (p_vrs_min_radius > 100.0) {
		WARN_PRINT_ONCE("VRS minimum radius can not be set above 100.0");
		vrs_min_radius = 100.0;
	} else {
		vrs_min_radius = p_vrs_min_radius;
		vrs_dirty = true;
	}
}

float XRInterface::get_vrs_strength() const {
	return vrs_strength;
}

void XRInterface::set_vrs_strength(float p_vrs_strength) {
	if (p_vrs_strength < 0.1) {
		WARN_PRINT_ONCE("VRS strength can not be set below 0.1");
		vrs_strength = 0.1;
	} else if (p_vrs_strength > 10.0) {
		WARN_PRINT_ONCE("VRS strength can not be set above 10.0");
		vrs_strength = 10.0;
	} else {
		vrs_strength = p_vrs_strength;
		vrs_dirty = true;
	}
}

Vector2 XRInterface::get_eye_focus(uint32_t p_view, float p_aspect) {
	// Our near and far don't matter much for what we're doing here,
	// but there are some interfaces that will remember this as the near and far and may fail as a result...

	Projection cm = get_projection_for_view(p_view, p_aspect, 0.1, 1000.0);
	Vector3 center = cm.xform(Vector3(0.0, 0.0, 999.0));

	return Vector2(center.x, center.y);
}

RID XRInterface::get_vrs_texture() {
	// Default logic will return a standard VRS image based on our target size and default projections.
	// Note that this only gets called if VRS is supported on the hardware.

	int32_t texel_width = RD::get_singleton()->limit_get(RD::LIMIT_VRS_TEXEL_WIDTH);
	int32_t texel_height = RD::get_singleton()->limit_get(RD::LIMIT_VRS_TEXEL_HEIGHT);
	int view_count = get_view_count();
	Size2 target_size = get_render_target_size();

	float texel_size = MAX(texel_width, texel_height);
	float min_radius = vrs_min_radius / texel_size;

	real_t aspect = target_size.x / target_size.y; // is this y/x ?
	Size2 vrs_size = Size2(round(0.5 + target_size.x / texel_width), round(0.5 + target_size.y / texel_height));
	real_t radius = MAX(1.0, (MAX(vrs_size.x, vrs_size.y) * 0.5 / vrs_strength) - min_radius);
	Size2 vrs_sizei = vrs_size;

	// Our density map is now unified, with a value of (0.0, 0.0) meaning a 1x1 texel size and (1.0, 1.0) an max texel size.
	// For our standard VRS extension on Vulkan this means a maximum of 8x8.
	// For the density map extension this scales depending on the max texel size.

	if (vrs.size != vrs_sizei || vrs_dirty) {
		// Out with the old.
		if (vrs.vrs_texture.is_valid()) {
			RS::get_singleton()->free(vrs.vrs_texture);
			vrs.vrs_texture = RID();
		}

		// In with the new.
		Vector<Ref<Image>> images;
		vrs.size = vrs_sizei;

		for (int i = 0; i < view_count && i < 2; i++) {
			PackedByteArray data;
			data.resize(vrs_sizei.x * vrs_sizei.y * 2);
			uint8_t *data_ptr = data.ptrw();

			Vector2 eye_focus = get_eye_focus(i, aspect);

			Vector2i view_center;
			view_center.x = int(vrs_size.x * (eye_focus.x + 1.0) * 0.5);
			view_center.y = int(vrs_size.y * (eye_focus.y + 1.0) * 0.5);

			int d = 0;
			for (int y = 0; y < vrs_sizei.y; y++) {
				for (int x = 0; x < vrs_sizei.x; x++) {
					Vector2 offset = Vector2(x - view_center.x, y - view_center.y);
					offset.y *= aspect;
					real_t density = 255.0 * MAX(0.0, abs(offset.x) - min_radius) / radius;
					data_ptr[d++] = MIN(255, density);
					density = 255.0 * MAX(0.0, abs(offset.y) - min_radius) / radius;
					data_ptr[d++] = MIN(255, density);
				}
			}
			images.push_back(Image::create_from_data(vrs_sizei.x, vrs_sizei.y, false, Image::FORMAT_RG8, data));
		}

		if (images.size() == 1) {
			vrs.vrs_texture = RS::get_singleton()->texture_2d_create(images[0]);
		} else {
			vrs.vrs_texture = RS::get_singleton()->texture_2d_layered_create(images, RS::TEXTURE_LAYERED_2D_ARRAY);
		}

		vrs_dirty = false;
	}

	return vrs.vrs_texture;
}

/** these are optional, so we want dummies **/

RID XRInterface::get_color_texture() {
	return RID();
}

RID XRInterface::get_depth_texture() {
	return RID();
}

RID XRInterface::get_velocity_texture() {
	return RID();
}

PackedStringArray XRInterface::get_suggested_tracker_names() const {
	PackedStringArray arr;

	return arr;
}

PackedStringArray XRInterface::get_suggested_pose_names(const StringName &p_tracker_name) const {
	PackedStringArray arr;

	return arr;
}

XRInterface::TrackingStatus XRInterface::get_tracking_status() const {
	return XR_UNKNOWN_TRACKING;
}

void XRInterface::trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec) {
}

Array XRInterface::get_supported_environment_blend_modes() {
	Array default_blend_modes;
	default_blend_modes.push_back(XR_ENV_BLEND_MODE_OPAQUE);
	return default_blend_modes;
}
