/**************************************************************************/
/*  openxr_foveated_inset_extension.h                                     */
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

#pragma once

#include "../openxr_api.h"
#include "openxr_extension_wrapper.h"

#include "servers/xr/xr_positional_tracker.h"

// Foveated inset is a core feature in OpenXR 1.1 but we'll encapsulate
// the functionality in this "extension".
//
// When used we render 4 views instead of 2.
// The first two views are our context view and rendered by our default logic.
// The second two views are our focus view (inset) and handled by the
// implementation here.
// This order is guaranteed by the OpenXR specification.
//
// In OpenXR 1.0 XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET
// was a Varjo only extension implementing this functionality equal to how
// the core support in OpenXR 1.1 defines this.
// We enable it as a fallback if supported.
//
// For more detail on how this works, Varjo has an excellent information page:
// https://varjo.com/blog/make-the-best-out-of-your-varjo-experience-update-on-varjo-quad-view-eye-tracked-foveation-and-varjo-base-settings

#define XR_TRACKER_INSET SNAME("foveated_inset")

class OpenXRFoveatedInsetViewport;

class OpenXRFoveatedInsetExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRFoveatedInsetExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods() {}

public:
	static OpenXRFoveatedInsetExtension *get_singleton();

	OpenXRFoveatedInsetExtension();
	virtual ~OpenXRFoveatedInsetExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;

	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;
	virtual void on_state_ready() override;
	virtual void on_process() override;

	// Note, call `OpenXR::is_view_configuration_supported(XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO_WITH_FOVEATED_INSET)`
	// to check if our foveated inset is supported.
	// Even if OpenXR 1.1 is initialized or the Varjo extension is available,
	// this does not guarantee the current hardware configuration actually makes foveated inset available.
	// Hence we do not expose an `is_available` function here.

	virtual TypedArray<Projection> get_camera_projections(const StringName &p_tracker_name, double p_aspect, double p_z_near, double p_z_far) override;
	virtual TypedArray<Transform3D> get_camera_offsets(const StringName &p_tracker_name) override;

	Size2i get_render_size();

	void register_viewport(RID p_viewport);
	void unregister_viewport(RID p_viewport);
	virtual void on_pre_render() override;
	virtual void on_post_render() override;

private:
	static OpenXRFoveatedInsetExtension *singleton;
	OpenXRFoveatedInsetViewport *inset_viewport = nullptr;

	bool varjo_ext_available = false;
	bool varjo_foveated_rendering_ext_available = false;

	Ref<XRPositionalTracker> foveated_inset;
	Transform3D head_transform;
	Vector3 head_linear_velocity;
	Vector3 head_angular_velocity;
	XRPose::TrackingConfidence head_confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;

	enum SwapChainTypes {
		SWAPCHAIN_COLOR,
		SWAPCHAIN_DEPTH,
		SWAPCHAIN_MAX
	};

	// Render state, Only accessible in rendering thread
	struct RenderState {
		LocalVector<RID> viewports;
		Size2i size;
		OpenXRAPI::OpenXRSwapChainInfo swapchains[SWAPCHAIN_MAX];
	} render_state;

	void _register_viewport_rt(RID p_viewport);
	void _unregister_viewport_rt(RID p_viewport);

	void _free_swapchains();
};
