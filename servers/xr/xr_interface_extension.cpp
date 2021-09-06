/*************************************************************************/
/*  xr_interface_extension.cpp                                           */
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

#include "xr_interface_extension.h"
#include "servers/rendering/renderer_compositor.h"

void XRInterfaceExtension::_bind_methods() {
	GDVIRTUAL_BIND(_get_name);
	GDVIRTUAL_BIND(_get_capabilities);

	GDVIRTUAL_BIND(_is_initialized);
	GDVIRTUAL_BIND(_initialize);
	GDVIRTUAL_BIND(_uninitialize);

	GDVIRTUAL_BIND(_get_tracking_status);

	ClassDB::bind_method(D_METHOD("add_blit", "render_target", "src_rect", "dst_rect", "use_layer", "layer", "apply_lens_distortion", "eye_center", "k1", "k2", "upscale", "aspect_ratio"), &XRInterfaceExtension::add_blit);

	GDVIRTUAL_BIND(_get_render_target_size);
	GDVIRTUAL_BIND(_get_view_count);
	GDVIRTUAL_BIND(_get_camera_transform);
	GDVIRTUAL_BIND(_get_transform_for_view, "view", "cam_transform");
	GDVIRTUAL_BIND(_get_projection_for_view, "view", "aspect", "z_near", "z_far");

	GDVIRTUAL_BIND(_commit_views, "render_target", "screen_rect");

	GDVIRTUAL_BIND(_process);
	GDVIRTUAL_BIND(_notification, "what");

	// we don't have any properties specific to VR yet....

	// but we do have properties specific to AR....
	GDVIRTUAL_BIND(_get_anchor_detection_is_enabled);
	GDVIRTUAL_BIND(_set_anchor_detection_is_enabled, "enabled");
	GDVIRTUAL_BIND(_get_camera_feed_id);
}

StringName XRInterfaceExtension::get_name() const {
	StringName name;

	if (GDVIRTUAL_CALL(_get_name, name)) {
		return name;
	}

	return "Unknown";
}

uint32_t XRInterfaceExtension::get_capabilities() const {
	uint32_t capabilities;

	if (GDVIRTUAL_CALL(_get_capabilities, capabilities)) {
		return capabilities;
	}

	return 0;
}

bool XRInterfaceExtension::is_initialized() const {
	bool initialised = false;

	if (GDVIRTUAL_CALL(_is_initialized, initialised)) {
		return initialised;
	}

	return false;
}

bool XRInterfaceExtension::initialize() {
	bool initialised = false;

	if (GDVIRTUAL_CALL(_initialize, initialised)) {
		return initialised;
	}

	return false;
}

void XRInterfaceExtension::uninitialize() {
	GDVIRTUAL_CALL(_uninitialize);
}

XRInterface::TrackingStatus XRInterfaceExtension::get_tracking_status() const {
	uint32_t status;

	if (GDVIRTUAL_CALL(_get_tracking_status, status)) {
		return TrackingStatus(status);
	}

	return XR_UNKNOWN_TRACKING;
}

/** these will only be implemented on AR interfaces, so we want dummies for VR **/
bool XRInterfaceExtension::get_anchor_detection_is_enabled() const {
	bool enabled;

	if (GDVIRTUAL_CALL(_get_anchor_detection_is_enabled, enabled)) {
		return enabled;
	}

	return false;
}

void XRInterfaceExtension::set_anchor_detection_is_enabled(bool p_enable) {
	// don't do anything here, this needs to be implemented on AR interface to enable/disable things like plane detection etc.
	GDVIRTUAL_CALL(_set_anchor_detection_is_enabled, p_enable);
}

int XRInterfaceExtension::get_camera_feed_id() {
	int feed_id;

	if (GDVIRTUAL_CALL(_get_camera_feed_id, feed_id)) {
		return feed_id;
	}

	return 0;
}

Size2 XRInterfaceExtension::get_render_target_size() {
	Size2 size;

	if (GDVIRTUAL_CALL(_get_render_target_size, size)) {
		return size;
	}

	return Size2(0, 0);
}

uint32_t XRInterfaceExtension::get_view_count() {
	uint32_t view_count;

	if (GDVIRTUAL_CALL(_get_view_count, view_count)) {
		return view_count;
	}

	return 1;
}

Transform3D XRInterfaceExtension::get_camera_transform() {
	Transform3D transform;

	if (GDVIRTUAL_CALL(_get_camera_transform, transform)) {
		return transform;
	}

	return Transform3D();
}

Transform3D XRInterfaceExtension::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	Transform3D transform;

	if (GDVIRTUAL_CALL(_get_transform_for_view, p_view, p_cam_transform, transform)) {
		return transform;
	}

	return Transform3D();
}

CameraMatrix XRInterfaceExtension::get_projection_for_view(uint32_t p_view, real_t p_aspect, real_t p_z_near, real_t p_z_far) {
	CameraMatrix cm;
	PackedFloat64Array arr;

	if (GDVIRTUAL_CALL(_get_projection_for_view, p_view, p_aspect, p_z_near, p_z_far, arr)) {
		ERR_FAIL_COND_V_MSG(arr.size() != 16, CameraMatrix(), "Projection matrix must contain 16 floats");
		real_t *m = (real_t *)cm.matrix;
		for (int i = 0; i < 16; i++) {
			m[i] = arr[i];
		}
		return cm;
	}

	return CameraMatrix();
}

void XRInterfaceExtension::add_blit(RID p_render_target, Rect2 p_src_rect, Rect2i p_dst_rect, bool p_use_layer, uint32_t p_layer, bool p_apply_lens_distortion, Vector2 p_eye_center, float p_k1, float p_k2, float p_upscale, float p_aspect_ratio) {
	BlitToScreen blit;

	ERR_FAIL_COND_MSG(!can_add_blits, "add_blit can only be called from an XR plugin from within _commit_views!");

	blit.render_target = p_render_target;
	blit.src_rect = p_src_rect;
	blit.dst_rect = p_dst_rect;

	blit.multi_view.use_layer = p_use_layer;
	blit.multi_view.layer = p_layer;

	blit.lens_distortion.apply = p_apply_lens_distortion;
	blit.lens_distortion.eye_center = p_eye_center;
	blit.lens_distortion.k1 = p_k1;
	blit.lens_distortion.k2 = p_k2;
	blit.lens_distortion.upscale = p_upscale;
	blit.lens_distortion.aspect_ratio = p_aspect_ratio;

	blits.push_back(blit);
}

Vector<BlitToScreen> XRInterfaceExtension::commit_views(RID p_render_target, const Rect2 &p_screen_rect) {
	// This is just so our XR plugin can add blits...
	blits.clear();
	can_add_blits = true;

	if (GDVIRTUAL_CALL(_commit_views, p_render_target, p_screen_rect)) {
		return blits;
	}

	can_add_blits = false;
	return blits;
}

void XRInterfaceExtension::process() {
	GDVIRTUAL_CALL(_process);
}

void XRInterfaceExtension::notification(int p_what) {
	GDVIRTUAL_CALL(_notification, p_what);
}
