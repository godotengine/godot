/*************************************************************************/
/*  xr_interface_extension.h                                             */
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

#ifndef XR_INTERFACE_EXTENSION_H
#define XR_INTERFACE_EXTENSION_H

#include "servers/xr/xr_interface.h"

class XRInterfaceExtension : public XRInterface {
	GDCLASS(XRInterfaceExtension, XRInterface);

public:
private:
	bool can_add_blits = false;
	Vector<BlitToScreen> blits;

protected:
	_THREAD_SAFE_CLASS_

	static void _bind_methods();

public:
	/** general interface information **/
	virtual StringName get_name() const override;
	virtual uint32_t get_capabilities() const override;

	GDVIRTUAL0RC(StringName, _get_name);
	GDVIRTUAL0RC(uint32_t, _get_capabilities);

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	GDVIRTUAL0RC(bool, _is_initialized);
	GDVIRTUAL0R(bool, _initialize);
	GDVIRTUAL0(_uninitialize);

	/** input and output **/

	virtual PackedStringArray get_suggested_tracker_names() const override; /* return a list of likely/suggested tracker names */
	virtual PackedStringArray get_suggested_pose_names(const StringName &p_tracker_name) const override; /* return a list of likely/suggested action names for this tracker */
	virtual TrackingStatus get_tracking_status() const override;
	virtual void trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec = 0) override;

	GDVIRTUAL0RC(PackedStringArray, _get_suggested_tracker_names);
	GDVIRTUAL1RC(PackedStringArray, _get_suggested_pose_names, const StringName &);
	GDVIRTUAL0RC(uint32_t, _get_tracking_status);
	GDVIRTUAL6(_trigger_haptic_pulse, const String &, const StringName &, double, double, double, double);

	/** specific to VR **/
	// nothing yet

	/** specific to AR **/
	virtual bool get_anchor_detection_is_enabled() const override;
	virtual void set_anchor_detection_is_enabled(bool p_enable) override;
	virtual int get_camera_feed_id() override;

	GDVIRTUAL0RC(bool, _get_anchor_detection_is_enabled);
	GDVIRTUAL1(_set_anchor_detection_is_enabled, bool);
	GDVIRTUAL0RC(int, _get_camera_feed_id);

	/** rendering and internal **/

	virtual Size2 get_render_target_size() override;
	virtual uint32_t get_view_count() override;
	virtual Transform3D get_camera_transform() override;
	virtual Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) override;
	virtual CameraMatrix get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) override;

	GDVIRTUAL0R(Size2, _get_render_target_size);
	GDVIRTUAL0R(uint32_t, _get_view_count);
	GDVIRTUAL0R(Transform3D, _get_camera_transform);
	GDVIRTUAL2R(Transform3D, _get_transform_for_view, uint32_t, const Transform3D &);
	GDVIRTUAL4R(PackedFloat64Array, _get_projection_for_view, uint32_t, double, double, double);

	void add_blit(RID p_render_target, Rect2 p_src_rect, Rect2i p_dst_rect, bool p_use_layer = false, uint32_t p_layer = 0, bool p_apply_lens_distortion = false, Vector2 p_eye_center = Vector2(), double p_k1 = 0.0, double p_k2 = 0.0, double p_upscale = 1.0, double p_aspect_ratio = 1.0);
	virtual Vector<BlitToScreen> commit_views(RID p_render_target, const Rect2 &p_screen_rect) override;
	GDVIRTUAL2(_commit_views, RID, const Rect2 &);

	virtual void process() override;
	virtual void notification(int p_what) override;

	GDVIRTUAL0(_process);
	GDVIRTUAL1(_notification, int);

	/* access to some internals we need */
	RID get_render_target_texture(RID p_render_target);
	// RID get_render_target_depth(RID p_render_target);
};

#endif // !XR_INTERFACE_EXTENSION_H
