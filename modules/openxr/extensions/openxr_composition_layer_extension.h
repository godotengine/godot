/**************************************************************************/
/*  openxr_composition_layer_extension.h                                  */
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

#include "openxr_extension_wrapper.h"

#include "../openxr_api.h"

#ifdef ANDROID_ENABLED
#include <jni.h>

// Copied here from openxr_platform.h, in order to avoid including that whole header,
// which can cause compilation issues on some platforms.
typedef XrResult(XRAPI_PTR *PFN_xrCreateSwapchainAndroidSurfaceKHR)(XrSession session, const XrSwapchainCreateInfo *info, XrSwapchain *swapchain, jobject *surface);
#endif

class JavaObject;

// This extension provides access to composition layers for displaying 2D content through the XR compositor.

#define OPENXR_LAYER_FUNC1(m_name, m_arg1)                                                                                                                                \
	void _composition_layer_##m_name##_rt(RID p_layer, m_arg1 p1) {                                                                                                       \
		CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);                                                                                           \
		ERR_FAIL_NULL(layer);                                                                                                                                             \
		layer->m_name(p1);                                                                                                                                                \
	}                                                                                                                                                                     \
	void composition_layer_##m_name(RID p_layer, m_arg1 p1) {                                                                                                             \
		RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &OpenXRCompositionLayerExtension::_composition_layer_##m_name##_rt).bind(p_layer, p1)); \
	}

#define OPENXR_LAYER_FUNC2(m_name, m_arg1, m_arg2)                                                                                                                            \
	void _composition_layer_##m_name##_rt(RID p_layer, m_arg1 p1, m_arg2 p2) {                                                                                                \
		CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);                                                                                               \
		ERR_FAIL_NULL(layer);                                                                                                                                                 \
		layer->m_name(p1, p2);                                                                                                                                                \
	}                                                                                                                                                                         \
	void composition_layer_##m_name(RID p_layer, m_arg1 p1, m_arg2 p2) {                                                                                                      \
		RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &OpenXRCompositionLayerExtension::_composition_layer_##m_name##_rt).bind(p_layer, p1, p2)); \
	}

// OpenXRCompositionLayerExtension enables the extensions related to this functionality
class OpenXRCompositionLayerExtension : public OpenXRExtensionWrapper {
	GDCLASS(OpenXRCompositionLayerExtension, OpenXRExtensionWrapper);

protected:
	static void _bind_methods() {}

public:
	// Must be identical to Filter enum definition in OpenXRCompositionLayer.
	enum Filter {
		FILTER_NEAREST,
		FILTER_LINEAR,
		FILTER_CUBIC,
	};

	// Must be identical to MipmapMode enum definition in OpenXRCompositionLayer.
	enum MipmapMode {
		MIPMAP_MODE_DISABLED,
		MIPMAP_MODE_NEAREST,
		MIPMAP_MODE_LINEAR,
	};

	// Must be identical to Wrap enum definition in OpenXRCompositionLayer.
	enum Wrap {
		WRAP_CLAMP_TO_BORDER,
		WRAP_CLAMP_TO_EDGE,
		WRAP_REPEAT,
		WRAP_MIRRORED_REPEAT,
		WRAP_MIRROR_CLAMP_TO_EDGE,
	};

	// Must be identical to Swizzle enum definition in OpenXRCompositionLayer.
	enum Swizzle {
		SWIZZLE_RED,
		SWIZZLE_GREEN,
		SWIZZLE_BLUE,
		SWIZZLE_ALPHA,
		SWIZZLE_ZERO,
		SWIZZLE_ONE,
	};

	struct SwapchainState {
		Filter min_filter = Filter::FILTER_LINEAR;
		Filter mag_filter = Filter::FILTER_LINEAR;
		MipmapMode mipmap_mode = MipmapMode::MIPMAP_MODE_LINEAR;
		Wrap horizontal_wrap = Wrap::WRAP_CLAMP_TO_BORDER;
		Wrap vertical_wrap = Wrap::WRAP_CLAMP_TO_BORDER;
		Swizzle red_swizzle = Swizzle::SWIZZLE_RED;
		Swizzle green_swizzle = Swizzle::SWIZZLE_GREEN;
		Swizzle blue_swizzle = Swizzle::SWIZZLE_BLUE;
		Swizzle alpha_swizzle = Swizzle::SWIZZLE_ALPHA;
		float max_anisotropy = 1.0;
		Color border_color = { 0.0, 0.0, 0.0, 0.0 };
	};

	static OpenXRCompositionLayerExtension *get_singleton();

	OpenXRCompositionLayerExtension();
	virtual ~OpenXRCompositionLayerExtension() override;

	virtual HashMap<String, bool *> get_requested_extensions(XrVersion p_version) override;
	virtual void on_instance_created(const XrInstance p_instance) override;
	virtual void on_session_created(const XrSession p_session) override;
	virtual void on_session_destroyed() override;
	virtual void on_pre_render() override;

	virtual int get_composition_layer_count() override;
	virtual XrCompositionLayerBaseHeader *get_composition_layer(int p_index) override;
	virtual int get_composition_layer_order(int p_index) override;

	// The data on p_openxr_layer will be copied - there is no need to keep it valid after this call.
	RID composition_layer_create(XrCompositionLayerBaseHeader *p_openxr_layer);
	void composition_layer_free(RID p_layer);

	void composition_layer_register(RID p_layer);
	void composition_layer_unregister(RID p_layer);

	OPENXR_LAYER_FUNC2(set_viewport, RID, const Size2i &);
	OPENXR_LAYER_FUNC2(set_use_android_surface, bool, const Size2i &);
	OPENXR_LAYER_FUNC1(set_sort_order, int);
	OPENXR_LAYER_FUNC1(set_alpha_blend, bool);
	OPENXR_LAYER_FUNC1(set_transform, const Transform3D &);
	OPENXR_LAYER_FUNC1(set_protected_content, bool);
	OPENXR_LAYER_FUNC1(set_extension_property_values, Dictionary);

	OPENXR_LAYER_FUNC1(set_min_filter, Filter);
	OPENXR_LAYER_FUNC1(set_mag_filter, Filter);
	OPENXR_LAYER_FUNC1(set_mipmap_mode, MipmapMode);
	OPENXR_LAYER_FUNC1(set_horizontal_wrap, Wrap);
	OPENXR_LAYER_FUNC1(set_vertical_wrap, Wrap);
	OPENXR_LAYER_FUNC1(set_red_swizzle, Swizzle);
	OPENXR_LAYER_FUNC1(set_blue_swizzle, Swizzle);
	OPENXR_LAYER_FUNC1(set_green_swizzle, Swizzle);
	OPENXR_LAYER_FUNC1(set_alpha_swizzle, Swizzle);
	OPENXR_LAYER_FUNC1(set_max_anisotropy, float);
	OPENXR_LAYER_FUNC1(set_border_color, const Color &);

	OPENXR_LAYER_FUNC1(set_quad_size, const Size2 &);

	OPENXR_LAYER_FUNC1(set_cylinder_radius, float);
	OPENXR_LAYER_FUNC1(set_cylinder_aspect_ratio, float);
	OPENXR_LAYER_FUNC1(set_cylinder_central_angle, float);

	OPENXR_LAYER_FUNC1(set_equirect_radius, float);
	OPENXR_LAYER_FUNC1(set_equirect_central_horizontal_angle, float);
	OPENXR_LAYER_FUNC1(set_equirect_upper_vertical_angle, float);
	OPENXR_LAYER_FUNC1(set_equirect_lower_vertical_angle, float);

	Ref<JavaObject> composition_layer_get_android_surface(RID p_layer);

	bool is_available(XrStructureType p_which);
	bool is_android_surface_swapchain_available() { return android_surface_ext_available; }

private:
	static OpenXRCompositionLayerExtension *singleton;

	bool cylinder_ext_available = false;
	bool equirect_ext_available = false;
	bool android_surface_ext_available = false;

	void _composition_layer_free_rt(RID p_layer);
	void _composition_layer_register_rt(RID p_layer);
	void _composition_layer_unregister_rt(RID p_layer);

#ifdef ANDROID_ENABLED
	bool create_android_surface_swapchain(XrSwapchainCreateInfo *p_info, XrSwapchain *r_swapchain, jobject *r_surface);

	EXT_PROTO_XRRESULT_FUNC1(xrDestroySwapchain, (XrSwapchain), swapchain)
	EXT_PROTO_XRRESULT_FUNC4(xrCreateSwapchainAndroidSurfaceKHR, (XrSession), session, (const XrSwapchainCreateInfo *), info, (XrSwapchain *), swapchain, (jobject *), surface)
#endif

	struct CompositionLayer {
		union {
			XrCompositionLayerBaseHeader composition_layer;
			XrCompositionLayerQuad composition_layer_quad;
			XrCompositionLayerCylinderKHR composition_layer_cylinder;
			XrCompositionLayerEquirect2KHR composition_layer_equirect;
		};

		int sort_order = 1;
		bool alpha_blend = false;
		Dictionary extension_property_values;
		bool extension_property_values_changed = true;

		struct {
			RID viewport;
			Size2i viewport_size;
			OpenXRAPI::OpenXRSwapChainInfo swapchain_info;
			bool static_image = false;
			bool swapchain_protected_content = false;
		} subviewport;

#ifdef ANDROID_ENABLED
		struct {
			XrSwapchain swapchain = XR_NULL_HANDLE;
			Ref<JavaObject> surface;
		} android_surface;
#endif

		bool use_android_surface = false;
		bool protected_content = false;
		Size2i swapchain_size;

		SwapchainState swapchain_state;
		bool swapchain_state_is_dirty = false;

		void set_viewport(RID p_viewport, const Size2i &p_size);
		void set_use_android_surface(bool p_use_android_surface, const Size2i &p_size);

		void set_sort_order(int p_sort_order) { sort_order = p_sort_order; }
		void set_alpha_blend(bool p_alpha_blend);
		void set_protected_content(bool p_protected_content) { protected_content = p_protected_content; }
		void set_transform(const Transform3D &p_transform);
		void set_extension_property_values(const Dictionary &p_extension_property_values);

		void set_min_filter(Filter p_mode);
		void set_mag_filter(Filter p_mode);
		void set_mipmap_mode(MipmapMode p_mode);
		void set_horizontal_wrap(Wrap p_mode);
		void set_vertical_wrap(Wrap p_mode);
		void set_red_swizzle(Swizzle p_mode);
		void set_green_swizzle(Swizzle p_mode);
		void set_blue_swizzle(Swizzle p_mode);
		void set_alpha_swizzle(Swizzle p_mode);
		void set_max_anisotropy(float p_value);
		void set_border_color(const Color &p_color);

		void set_quad_size(const Size2 &p_size);

		void set_cylinder_radius(float p_radius);
		void set_cylinder_aspect_ratio(float p_aspect_ratio);
		void set_cylinder_central_angle(float p_central_angle);

		void set_equirect_radius(float p_radius);
		void set_equirect_central_horizontal_angle(float p_angle);
		void set_equirect_upper_vertical_angle(float p_angle);
		void set_equirect_lower_vertical_angle(float p_angle);

		Ref<JavaObject> get_android_surface();
		void on_pre_render();
		XrCompositionLayerBaseHeader *get_composition_layer();
		void free();

	private:
		void update_swapchain_state();
		void update_swapchain_sub_image(XrSwapchainSubImage &r_subimage);
		bool update_and_acquire_swapchain(bool p_static_image);
		RID get_current_swapchain_texture();
		void free_swapchain();

#ifdef ANDROID_ENABLED
		void create_android_surface();
#endif
	};

	Mutex composition_layer_mutex;
	RID_Owner<CompositionLayer, true> composition_layer_owner;
	LocalVector<CompositionLayer *> registered_composition_layers;
};

VARIANT_ENUM_CAST(OpenXRCompositionLayerExtension::Filter);
VARIANT_ENUM_CAST(OpenXRCompositionLayerExtension::MipmapMode);
VARIANT_ENUM_CAST(OpenXRCompositionLayerExtension::Wrap);
VARIANT_ENUM_CAST(OpenXRCompositionLayerExtension::Swizzle);

#undef OPENXR_LAYER_FUNC1
#undef OPENXR_LAYER_FUNC2
