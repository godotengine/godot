/**************************************************************************/
/*  openxr_composition_layer_extension.cpp                                */
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

#include "openxr_composition_layer_extension.h"

#ifdef ANDROID_ENABLED
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#endif

#include "openxr_fb_update_swapchain_extension.h"
#include "platform/android/api/java_class_wrapper.h"
#include "servers/rendering/rendering_server_globals.h"

////////////////////////////////////////////////////////////////////////////
// OpenXRCompositionLayerExtension

OpenXRCompositionLayerExtension *OpenXRCompositionLayerExtension::singleton = nullptr;

OpenXRCompositionLayerExtension *OpenXRCompositionLayerExtension::get_singleton() {
	return singleton;
}

OpenXRCompositionLayerExtension::OpenXRCompositionLayerExtension() {
	singleton = this;
}

OpenXRCompositionLayerExtension::~OpenXRCompositionLayerExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRCompositionLayerExtension::get_requested_extensions(XrVersion p_version) {
	HashMap<String, bool *> request_extensions;

	request_extensions[XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME] = &cylinder_ext_available;
	request_extensions[XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME] = &equirect_ext_available;

#ifdef ANDROID_ENABLED
	request_extensions[XR_KHR_ANDROID_SURFACE_SWAPCHAIN_EXTENSION_NAME] = &android_surface_ext_available;
#endif

	return request_extensions;
}

void OpenXRCompositionLayerExtension::on_instance_created(const XrInstance p_instance) {
#ifdef ANDROID_ENABLED
	EXT_INIT_XR_FUNC(xrDestroySwapchain);
	if (android_surface_ext_available) {
		EXT_INIT_XR_FUNC(xrCreateSwapchainAndroidSurfaceKHR);
	}
#endif
}

void OpenXRCompositionLayerExtension::on_session_created(const XrSession p_session) {
	OpenXRAPI::get_singleton()->register_composition_layer_provider(this);
}

void OpenXRCompositionLayerExtension::on_session_destroyed() {
	OpenXRAPI::get_singleton()->unregister_composition_layer_provider(this);
}

void OpenXRCompositionLayerExtension::on_pre_render() {
	for (CompositionLayer *composition_layer : registered_composition_layers) {
		composition_layer->on_pre_render();
	}
}

int OpenXRCompositionLayerExtension::get_composition_layer_count() {
	return registered_composition_layers.size();
}

XrCompositionLayerBaseHeader *OpenXRCompositionLayerExtension::get_composition_layer(int p_index) {
	ERR_FAIL_UNSIGNED_INDEX_V((unsigned int)p_index, registered_composition_layers.size(), nullptr);
	return registered_composition_layers[p_index]->get_composition_layer();
}

int OpenXRCompositionLayerExtension::get_composition_layer_order(int p_index) {
	ERR_FAIL_UNSIGNED_INDEX_V((unsigned int)p_index, registered_composition_layers.size(), 1);
	return registered_composition_layers[p_index]->sort_order;
}

RID OpenXRCompositionLayerExtension::composition_layer_create(XrCompositionLayerBaseHeader *p_openxr_layer) {
	RID rid = composition_layer_owner.make_rid();
	CompositionLayer *layer = composition_layer_owner.get_or_null(rid);

	switch (p_openxr_layer->type) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			layer->composition_layer_quad = *(XrCompositionLayerQuad *)p_openxr_layer;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			layer->composition_layer_cylinder = *(XrCompositionLayerCylinderKHR *)p_openxr_layer;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			layer->composition_layer_equirect = *(XrCompositionLayerEquirect2KHR *)p_openxr_layer;
		} break;
		default: {
			ERR_PRINT(vformat("Invalid OpenXR composition layer type: %s", p_openxr_layer->type));
			composition_layer_owner.free(rid);
			return RID();
		}
	}

	return rid;
}

void OpenXRCompositionLayerExtension::composition_layer_free(RID p_layer) {
	RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &OpenXRCompositionLayerExtension::_composition_layer_free_rt).bind(p_layer));
}

void OpenXRCompositionLayerExtension::composition_layer_register(RID p_layer) {
	RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &OpenXRCompositionLayerExtension::_composition_layer_register_rt).bind(p_layer));
}

void OpenXRCompositionLayerExtension::composition_layer_unregister(RID p_layer) {
	RenderingServer::get_singleton()->call_on_render_thread(callable_mp(this, &OpenXRCompositionLayerExtension::_composition_layer_unregister_rt).bind(p_layer));
}

Ref<JavaObject> OpenXRCompositionLayerExtension::composition_layer_get_android_surface(RID p_layer) {
	MutexLock lock(composition_layer_mutex);
	CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);
	ERR_FAIL_NULL_V(layer, Ref<JavaObject>());
	return layer->get_android_surface();
}

void OpenXRCompositionLayerExtension::_composition_layer_free_rt(RID p_layer) {
	_composition_layer_unregister_rt(p_layer);

	MutexLock lock(composition_layer_mutex);
	CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);
	if (layer) {
		for (OpenXRExtensionWrapper *extension : OpenXRAPI::get_registered_extension_wrappers()) {
			extension->on_viewport_composition_layer_destroyed(&layer->composition_layer);
		}
		layer->free();
	}

	composition_layer_owner.free(p_layer);
}

void OpenXRCompositionLayerExtension::_composition_layer_register_rt(RID p_layer) {
	CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);
	ERR_FAIL_NULL(layer);
	registered_composition_layers.push_back(layer);
}

void OpenXRCompositionLayerExtension::_composition_layer_unregister_rt(RID p_layer) {
	CompositionLayer *layer = composition_layer_owner.get_or_null(p_layer);
	ERR_FAIL_NULL(layer);
	registered_composition_layers.erase(layer);
}

bool OpenXRCompositionLayerExtension::is_available(XrStructureType p_which) {
	switch (p_which) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			// Doesn't require an extension.
			return true;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			return cylinder_ext_available;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			return equirect_ext_available;
		} break;
		default: {
			ERR_PRINT(vformat("Unsupported composition layer type: %s", p_which));
			return false;
		}
	}
}

#ifdef ANDROID_ENABLED
bool OpenXRCompositionLayerExtension::create_android_surface_swapchain(XrSwapchainCreateInfo *p_info, XrSwapchain *r_swapchain, jobject *r_surface) {
	if (android_surface_ext_available) {
		OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
		ERR_FAIL_NULL_V(openxr_api, false);

		XrResult result = xrCreateSwapchainAndroidSurfaceKHR(openxr_api->get_session(), p_info, r_swapchain, r_surface);
		if (XR_FAILED(result)) {
			print_line("OpenXR: Failed to create Android surface swapchain [", openxr_api->get_error_string(result), "]");
			return false;
		}

		return true;
	}

	return false;
}
#endif

////////////////////////////////////////////////////////////////////////////
// OpenXRCompositionLayerExtension::CompositionLayer

void OpenXRCompositionLayerExtension::CompositionLayer::set_viewport(RID p_viewport, const Size2i &p_size) {
	ERR_FAIL_COND(use_android_surface);

	if (subviewport.viewport != p_viewport) {
		if (subviewport.viewport.is_valid()) {
			RID rt = RenderingServer::get_singleton()->viewport_get_render_target(subviewport.viewport);
			RSG::texture_storage->render_target_set_override(rt, RID(), RID(), RID(), RID());
		}

		subviewport.viewport = p_viewport;

		if (subviewport.viewport.is_valid()) {
			subviewport.viewport_size = p_size;
		} else {
			free_swapchain();
			subviewport.viewport_size = Size2i();
		}
	} else if (subviewport.viewport_size != p_size) {
		subviewport.viewport_size = p_size;
	}
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_use_android_surface(bool p_use_android_surface, const Size2i &p_size) {
#ifdef ANDROID_ENABLED
	if (p_use_android_surface == use_android_surface) {
		if (use_android_surface && swapchain_size != p_size) {
			OpenXRFBUpdateSwapchainExtension *fb_update_swapchain_ext = OpenXRFBUpdateSwapchainExtension::get_singleton();
			if (fb_update_swapchain_ext && fb_update_swapchain_ext->is_android_ext_enabled()) {
				swapchain_size = p_size;
				fb_update_swapchain_ext->update_swapchain_surface_size(android_surface.swapchain, swapchain_size);
			}
		}
		return;
	}

	use_android_surface = p_use_android_surface;

	if (use_android_surface) {
		if (!OpenXRCompositionLayerExtension::get_singleton()->is_android_surface_swapchain_available()) {
			ERR_PRINT_ONCE("OpenXR: Cannot use Android surface for composition layer because the extension isn't available");
		}

		if (subviewport.viewport.is_valid()) {
			set_viewport(RID(), Size2i());
		}

		swapchain_size = p_size;
	} else {
		free_swapchain();
	}
#endif
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_alpha_blend(bool p_alpha_blend) {
	if (alpha_blend != p_alpha_blend) {
		alpha_blend = p_alpha_blend;
		if (alpha_blend) {
			composition_layer.layerFlags |= XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
		} else {
			composition_layer.layerFlags &= ~XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
		}
	}
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_transform(const Transform3D &p_transform) {
	Transform3D reference_frame = XRServer::get_singleton()->get_reference_frame();
	Transform3D transform = reference_frame.inverse() * p_transform;
	Quaternion quat(transform.basis.orthonormalized());

	XrPosef pose = {
		{ (float)quat.x, (float)quat.y, (float)quat.z, (float)quat.w },
		{ (float)transform.origin.x, (float)transform.origin.y, (float)transform.origin.z }
	};

	switch (composition_layer.type) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			composition_layer_quad.pose = pose;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			composition_layer_cylinder.pose = pose;
		} break;
		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			composition_layer_equirect.pose = pose;
		} break;
		default: {
			ERR_PRINT(vformat("Cannot set transform on unsupported composition layer type: %s", composition_layer.type));
		}
	}
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_extension_property_values(const Dictionary &p_property_values) {
	extension_property_values = p_property_values;
	extension_property_values_changed = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_min_filter(Filter p_mode) {
	swapchain_state.min_filter = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_mag_filter(Filter p_mode) {
	swapchain_state.mag_filter = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_mipmap_mode(MipmapMode p_mode) {
	swapchain_state.mipmap_mode = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_horizontal_wrap(Wrap p_mode) {
	swapchain_state.horizontal_wrap = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_vertical_wrap(Wrap p_mode) {
	swapchain_state.vertical_wrap = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_red_swizzle(Swizzle p_mode) {
	swapchain_state.red_swizzle = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_green_swizzle(Swizzle p_mode) {
	swapchain_state.green_swizzle = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_blue_swizzle(Swizzle p_mode) {
	swapchain_state.blue_swizzle = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_alpha_swizzle(Swizzle p_mode) {
	swapchain_state.alpha_swizzle = p_mode;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_max_anisotropy(float p_value) {
	swapchain_state.max_anisotropy = p_value;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_border_color(const Color &p_color) {
	swapchain_state.border_color = p_color;
	swapchain_state_is_dirty = true;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_quad_size(const Size2 &p_size) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_QUAD);
	composition_layer_quad.size = { (float)p_size.x, (float)p_size.y };
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_cylinder_radius(float p_radius) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR);
	composition_layer_cylinder.radius = p_radius;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_cylinder_aspect_ratio(float p_aspect_ratio) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR);
	composition_layer_cylinder.aspectRatio = p_aspect_ratio;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_cylinder_central_angle(float p_central_angle) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR);
	composition_layer_cylinder.centralAngle = p_central_angle;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_equirect_radius(float p_radius) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR);
	composition_layer_equirect.radius = p_radius;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_equirect_central_horizontal_angle(float p_angle) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR);
	composition_layer_equirect.centralHorizontalAngle = p_angle;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_equirect_upper_vertical_angle(float p_angle) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR);
	composition_layer_equirect.upperVerticalAngle = p_angle;
}

void OpenXRCompositionLayerExtension::CompositionLayer::set_equirect_lower_vertical_angle(float p_angle) {
	ERR_FAIL_COND(composition_layer.type != XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR);
	composition_layer_equirect.lowerVerticalAngle = p_angle;
}

Ref<JavaObject> OpenXRCompositionLayerExtension::CompositionLayer::get_android_surface() {
#ifdef ANDROID_ENABLED
	if (use_android_surface) {
		MutexLock lock(OpenXRCompositionLayerExtension::get_singleton()->composition_layer_mutex);
		if (android_surface.surface.is_null()) {
			create_android_surface();
		}
		return android_surface.surface;
	}
#endif
	return Ref<JavaObject>();
}

void OpenXRCompositionLayerExtension::CompositionLayer::on_pre_render() {
#ifdef ANDROID_ENABLED
	if (use_android_surface) {
		MutexLock lock(OpenXRCompositionLayerExtension::get_singleton()->composition_layer_mutex);
		if (android_surface.surface.is_null()) {
			create_android_surface();
		}
		return;
	}
#endif

	RenderingServer *rs = RenderingServer::get_singleton();
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();

	if (subviewport.viewport.is_valid() && openxr_api && openxr_api->is_running()) {
		RS::ViewportUpdateMode update_mode = rs->viewport_get_update_mode(subviewport.viewport);
		if (update_mode == RS::VIEWPORT_UPDATE_ONCE || update_mode == RS::VIEWPORT_UPDATE_ALWAYS) {
			// Update our XR swapchain
			if (update_and_acquire_swapchain(update_mode == RS::VIEWPORT_UPDATE_ONCE)) {
				// Render to our XR swapchain image.
				RID rt = rs->viewport_get_render_target(subviewport.viewport);
				RSG::texture_storage->render_target_set_override(rt, get_current_swapchain_texture(), RID(), RID(), RID());
			}
		}
	}

	if (swapchain_state_is_dirty) {
		update_swapchain_state();
		swapchain_state_is_dirty = false;
	}
}

XrCompositionLayerBaseHeader *OpenXRCompositionLayerExtension::CompositionLayer::get_composition_layer() {
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();
	OpenXRCompositionLayerExtension *composition_layer_extension = OpenXRCompositionLayerExtension::get_singleton();

	if (openxr_api == nullptr || composition_layer_extension == nullptr) {
		// OpenXR not initialized or we're in the editor?
		return nullptr;
	}

	if (!composition_layer_extension->is_available(composition_layer.type)) {
		// Selected type is not supported, ignore our layer.
		return nullptr;
	}

	XrSwapchainSubImage subimage = {
		0, // swapchain // NOLINT(modernize-use-nullptr) - 32-bit uses non-pointer uint64
		{ { 0, 0 }, { 0, 0 } }, // imageRect
		0, // imageArrayIndex
	};
	update_swapchain_sub_image(subimage);

	if (subimage.swapchain == XR_NULL_HANDLE) {
		// Don't have a swapchain to display? Ignore our layer.
		return nullptr;
	}

	// Update the layer struct for the swapchain.
	switch (composition_layer.type) {
		case XR_TYPE_COMPOSITION_LAYER_QUAD: {
			composition_layer_quad.space = openxr_api->get_play_space();
			composition_layer_quad.subImage = subimage;
		} break;

		case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
			composition_layer_cylinder.space = openxr_api->get_play_space();
			composition_layer_cylinder.subImage = subimage;
		} break;

		case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
			composition_layer_equirect.space = openxr_api->get_play_space();
			composition_layer_equirect.subImage = subimage;
		} break;

		default: {
			return nullptr;
		} break;
	}

	if (extension_property_values_changed) {
		extension_property_values_changed = false;

		void *next_pointer = nullptr;
		for (OpenXRExtensionWrapper *extension : OpenXRAPI::get_registered_extension_wrappers()) {
			void *np = extension->set_viewport_composition_layer_and_get_next_pointer(&composition_layer, extension_property_values, next_pointer);
			if (np) {
				next_pointer = np;
			}
		}
		composition_layer.next = next_pointer;
	}

	return &composition_layer;
}

void OpenXRCompositionLayerExtension::CompositionLayer::free() {
	if (use_android_surface) {
		free_swapchain();
	} else {
		// This will reset the viewport and free the swapchain too.
		set_viewport(RID(), Size2i());
	}
}

void OpenXRCompositionLayerExtension::CompositionLayer::update_swapchain_state() {
	OpenXRFBUpdateSwapchainExtension *fb_update_swapchain_ext = OpenXRFBUpdateSwapchainExtension::get_singleton();
	if (!fb_update_swapchain_ext) {
		return;
	}

#ifdef ANDROID_ENABLED
	if (use_android_surface) {
		if (android_surface.swapchain == XR_NULL_HANDLE) {
			return;
		}

		fb_update_swapchain_ext->update_swapchain_state(android_surface.swapchain, &swapchain_state);
	} else
#endif
	{
		if (subviewport.swapchain_info.get_swapchain() == XR_NULL_HANDLE) {
			return;
		}

		fb_update_swapchain_ext->update_swapchain_state(subviewport.swapchain_info.get_swapchain(), &swapchain_state);
	}
}

void OpenXRCompositionLayerExtension::CompositionLayer::update_swapchain_sub_image(XrSwapchainSubImage &r_subimage) {
#ifdef ANDROID_ENABLED
	if (use_android_surface) {
		r_subimage.swapchain = android_surface.swapchain;
	} else
#endif
	{
		XrSwapchain swapchain = subviewport.swapchain_info.get_swapchain();

		if (swapchain && subviewport.swapchain_info.is_image_acquired()) {
			subviewport.swapchain_info.release();
		}

		r_subimage.swapchain = swapchain;
	}

	r_subimage.imageRect.extent.width = swapchain_size.width;
	r_subimage.imageRect.extent.height = swapchain_size.height;
}

bool OpenXRCompositionLayerExtension::CompositionLayer::update_and_acquire_swapchain(bool p_static_image) {
	ERR_FAIL_COND_V(use_android_surface, false);

	OpenXRCompositionLayerExtension *composition_layer_extension = OpenXRCompositionLayerExtension::get_singleton();
	OpenXRAPI *openxr_api = OpenXRAPI::get_singleton();

	if (openxr_api == nullptr || composition_layer_extension == nullptr) {
		// OpenXR not initialized or we're in the editor?
		return false;
	}
	if (!composition_layer_extension->is_available(composition_layer.type)) {
		// Selected type is not supported?
		return false;
	}

	// See if our current swapchain is outdated.
	if (subviewport.swapchain_info.get_swapchain() != XR_NULL_HANDLE) {
		// If this swap chain, or the previous one, were static, then we can't reuse it.
		if (swapchain_size == subviewport.viewport_size && !p_static_image && !subviewport.static_image && protected_content == subviewport.swapchain_protected_content) {
			// We're all good! Just acquire it.
			// We can ignore should_render here, return will be false.
			bool should_render = true;
			return subviewport.swapchain_info.acquire(should_render);
		}

		subviewport.swapchain_info.queue_free();
	}

	// Create our new swap chain
	int64_t swapchain_format = openxr_api->get_color_swapchain_format();
	const uint32_t sample_count = 1;
	const uint32_t array_size = 1;
	XrSwapchainCreateFlags create_flags = 0;
	if (p_static_image) {
		create_flags |= XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT;
	}
	if (protected_content) {
		create_flags |= XR_SWAPCHAIN_CREATE_PROTECTED_CONTENT_BIT;
	}
	if (!subviewport.swapchain_info.create(create_flags, XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, swapchain_format, subviewport.viewport_size.width, subviewport.viewport_size.height, sample_count, array_size)) {
		swapchain_size = Size2i();
		return false;
	}

	swapchain_state_is_dirty = true;

	// Acquire our image so we can start rendering into it,
	// we can ignore should_render here, ret will be false.
	bool should_render = true;
	bool ret = subviewport.swapchain_info.acquire(should_render);

	swapchain_size = subviewport.viewport_size;
	subviewport.static_image = p_static_image;
	subviewport.swapchain_protected_content = protected_content;
	return ret;
}

RID OpenXRCompositionLayerExtension::CompositionLayer::get_current_swapchain_texture() {
	ERR_FAIL_COND_V(use_android_surface, RID());

	if (OpenXRAPI::get_singleton() == nullptr) {
		return RID();
	}

	return subviewport.swapchain_info.get_image();
}

void OpenXRCompositionLayerExtension::CompositionLayer::free_swapchain() {
#ifdef ANDROID_ENABLED
	if (use_android_surface) {
		if (android_surface.swapchain != XR_NULL_HANDLE) {
			OpenXRCompositionLayerExtension::get_singleton()->xrDestroySwapchain(android_surface.swapchain);
			android_surface.swapchain = XR_NULL_HANDLE;
			android_surface.surface.unref();
		}
	} else
#endif
	{
		if (subviewport.swapchain_info.get_swapchain() != XR_NULL_HANDLE) {
			subviewport.swapchain_info.queue_free();
		}
		subviewport.static_image = false;
	}

	swapchain_size = Size2i();
}

#ifdef ANDROID_ENABLED
void OpenXRCompositionLayerExtension::CompositionLayer::create_android_surface() {
	ERR_FAIL_COND(android_surface.swapchain != XR_NULL_HANDLE || android_surface.surface.is_valid());

	void *next_pointer = nullptr;
	for (OpenXRExtensionWrapper *wrapper : OpenXRAPI::get_registered_extension_wrappers()) {
		void *np = wrapper->set_android_surface_swapchain_create_info_and_get_next_pointer(extension_property_values, next_pointer);
		if (np != nullptr) {
			next_pointer = np;
		}
	}

	// Check to see if content should be protected.
	XrSwapchainCreateFlags create_flags = 0;

	if (protected_content) {
		create_flags = XR_SWAPCHAIN_CREATE_PROTECTED_CONTENT_BIT;
	}

	// The XR_FB_android_surface_swapchain_create extension mandates that format, sampleCount,
	// faceCount, arraySize, and mipCount must be zero.
	XrSwapchainCreateInfo info = {
		XR_TYPE_SWAPCHAIN_CREATE_INFO, // type
		next_pointer, // next
		create_flags, // createFlags
		XR_SWAPCHAIN_USAGE_SAMPLED_BIT | XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT, // usageFlags
		0, // format
		0, // sampleCount
		(uint32_t)swapchain_size.x, // width
		(uint32_t)swapchain_size.y, // height
		0, // faceCount
		0, // arraySize
		0, // mipCount
	};

	jobject surface;
	OpenXRCompositionLayerExtension::get_singleton()->create_android_surface_swapchain(&info, &android_surface.swapchain, &surface);

	swapchain_state_is_dirty = true;

	if (surface) {
		android_surface.surface.instantiate(JavaClassWrapper::get_singleton()->wrap("android.view.Surface"), surface);
	}
}
#endif
