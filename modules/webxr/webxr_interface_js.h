/*************************************************************************/
/*  webxr_interface_js.h                                                 */
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

#ifndef WEBXR_INTERFACE_JS_H
#define WEBXR_INTERFACE_JS_H

#ifdef JAVASCRIPT_ENABLED

#include "webxr_interface.h"

/**
	@author David Snopek <david.snopek@snopekgames.com>

	The WebXR interface is a VR/AR interface that can be used on the web.
*/

class WebXRInterfaceJS : public WebXRInterface {
	GDCLASS(WebXRInterfaceJS, WebXRInterface);

private:
	bool initialized;

	String session_mode;
	String required_features;
	String optional_features;
	String requested_reference_space_types;
	String reference_space_type;

	bool controllers_state[2];
	Size2 render_targetsize;

	Transform _js_matrix_to_transform(float *p_js_matrix);
	void _update_tracker(int p_controller_id);

public:
	virtual void is_session_supported(const String &p_session_mode) override;
	virtual void set_session_mode(String p_session_mode) override;
	virtual String get_session_mode() const override;
	virtual void set_required_features(String p_required_features) override;
	virtual String get_required_features() const override;
	virtual void set_optional_features(String p_optional_features) override;
	virtual String get_optional_features() const override;
	virtual void set_requested_reference_space_types(String p_requested_reference_space_types) override;
	virtual String get_requested_reference_space_types() const override;
	void _set_reference_space_type(String p_reference_space_type);
	virtual String get_reference_space_type() const override;
	virtual Ref<XRPositionalTracker> get_controller(int p_controller_id) const override;
	virtual String get_visibility_state() const override;
	virtual PackedVector3Array get_bounds_geometry() const override;

	virtual StringName get_name() const override;
	virtual int get_capabilities() const override;

	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	virtual Size2 get_render_targetsize() override;
	virtual bool is_stereo() override;
	virtual Transform get_transform_for_eye(XRInterface::Eyes p_eye, const Transform &p_cam_transform) override;
	virtual CameraMatrix get_projection_for_eye(XRInterface::Eyes p_eye, real_t p_aspect, real_t p_z_near, real_t p_z_far) override;
	virtual unsigned int get_external_texture_for_eye(XRInterface::Eyes p_eye) override;
	virtual void commit_for_eye(XRInterface::Eyes p_eye, RID p_render_target, const Rect2 &p_screen_rect) override;

	virtual void process() override;
	virtual void notification(int p_what) override;

	void _on_controller_changed();

	WebXRInterfaceJS();
	~WebXRInterfaceJS();
};

#endif // JAVASCRIPT_ENABLED

#endif // WEBXR_INTERFACE_JS_H
