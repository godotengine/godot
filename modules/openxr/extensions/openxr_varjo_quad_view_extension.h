/**************************************************************************/
/*  openxr_varjo_quad_view_extension.h                                    */
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

#ifndef OPENXR_VARJO_QUAD_VIEW_EXTENSION_H
#define OPENXR_VARJO_QUAD_VIEW_EXTENSION_H

#include "../openxr_api.h"
#include "openxr_extension_wrapper.h"

// This extension enables the XR_VIEW_CONFIGURATION_TYPE_PRIMARY_QUAD_VARJO
// view configuration type used for quad buffer rendering on Varjo devices.

class OpenXRVarjoQuadViewExtension : public OpenXRExtensionWrapper {
public:
	virtual HashMap<String, bool *> get_requested_extensions() override;

	virtual void on_session_created(const XrSession p_instance) override;
	virtual void on_session_destroyed() override;

	virtual bool owns_viewport(RID p_render_target) override;
	virtual bool on_pre_draw_viewport(RID p_render_target) override;
	virtual uint32_t get_viewport_view_count() override;
	virtual Size2i get_viewport_size() override;
	virtual bool get_view_transform(uint32_t p_view, XrTime p_display_time, Transform3D &r_transform) override;
	virtual bool get_view_projection(uint32_t p_view, double p_z_near, double p_z_far, XrTime p_display_time, Projection &r_projection) override;
	virtual XrSwapchain get_color_swapchain() override;
	virtual RID get_color_texture() override;
	virtual RID get_depth_texture() override;
	virtual void on_post_draw_viewport(RID p_render_target) override;
	virtual void on_end_frame() override;

	bool is_available();
	bool is_enabled();

private:
	bool available = false;
	bool enabled = false;
	bool have_primary_viewport = false;

	RID secondary_viewport;
	RID secondary_camera;
	Size2i swapchain_size = Size2i();
	OpenXRAPI::OpenXRSwapChainInfo swapchains[OpenXRAPI::OPENXR_SWAPCHAIN_MAX];

	void create_swapchains(Size2i p_size);
	void free_swapchains();
};

#endif // OPENXR_VARJO_QUAD_VIEW_EXTENSION_H
