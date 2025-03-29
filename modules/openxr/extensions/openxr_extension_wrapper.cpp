/**************************************************************************/
/*  openxr_extension_wrapper.cpp                                          */
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

#include "openxr_extension_wrapper.h"

#include "../openxr_api.h"
#include "../openxr_api_extension.h"

void OpenXRExtensionWrapper::_bind_methods() {
	GDVIRTUAL_BIND(_get_requested_extensions);
	GDVIRTUAL_BIND(_set_system_properties_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_instance_create_info_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_session_create_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_swapchain_create_info_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_hand_joint_locations_and_get_next_pointer, "hand_index", "next_pointer");
	GDVIRTUAL_BIND(_set_projection_views_and_get_next_pointer, "view_index", "next_pointer");
	GDVIRTUAL_BIND(_set_frame_wait_info_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_frame_end_info_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_view_locate_info_and_get_next_pointer, "next_pointer");
	GDVIRTUAL_BIND(_set_reference_space_create_info_and_get_next_pointer, "reference_space_type", "next_pointer");
	GDVIRTUAL_BIND(_get_composition_layer_count);
	GDVIRTUAL_BIND(_get_composition_layer, "index");
	GDVIRTUAL_BIND(_get_composition_layer_order, "index");
	GDVIRTUAL_BIND(_get_suggested_tracker_names);
	GDVIRTUAL_BIND(_on_register_metadata);
	GDVIRTUAL_BIND(_on_before_instance_created);
	GDVIRTUAL_BIND(_on_instance_created, "instance");
	GDVIRTUAL_BIND(_on_instance_destroyed);
	GDVIRTUAL_BIND(_on_session_created, "session");
	GDVIRTUAL_BIND(_on_process);
	GDVIRTUAL_BIND(_on_pre_render);
	GDVIRTUAL_BIND(_on_main_swapchains_created);
	GDVIRTUAL_BIND(_on_pre_draw_viewport, "viewport");
	GDVIRTUAL_BIND(_on_post_draw_viewport, "viewport");
	GDVIRTUAL_BIND(_on_session_destroyed);
	GDVIRTUAL_BIND(_on_state_idle);
	GDVIRTUAL_BIND(_on_state_ready);
	GDVIRTUAL_BIND(_on_state_synchronized);
	GDVIRTUAL_BIND(_on_state_visible);
	GDVIRTUAL_BIND(_on_state_focused);
	GDVIRTUAL_BIND(_on_state_stopping);
	GDVIRTUAL_BIND(_on_state_loss_pending);
	GDVIRTUAL_BIND(_on_state_exiting);
	GDVIRTUAL_BIND(_on_event_polled, "event");
	GDVIRTUAL_BIND(_set_viewport_composition_layer_and_get_next_pointer, "layer", "property_values", "next_pointer");
	GDVIRTUAL_BIND(_get_viewport_composition_layer_extension_properties);
	GDVIRTUAL_BIND(_get_viewport_composition_layer_extension_property_defaults);
	GDVIRTUAL_BIND(_on_viewport_composition_layer_destroyed, "layer");
	GDVIRTUAL_BIND(_set_android_surface_swapchain_create_info_and_get_next_pointer, "property_values", "next_pointer");

	ClassDB::bind_method(D_METHOD("get_openxr_api"), &OpenXRExtensionWrapper::_gdextension_get_openxr_api);
	ClassDB::bind_method(D_METHOD("register_extension_wrapper"), &OpenXRExtensionWrapper::_gdextension_register_extension_wrapper);
}

HashMap<String, bool *> OpenXRExtensionWrapper::get_requested_extensions() {
	Dictionary request_extension;

	if (GDVIRTUAL_CALL(_get_requested_extensions, request_extension)) {
		HashMap<String, bool *> result;
		Array keys = request_extension.keys();
		for (int i = 0; i < keys.size(); i++) {
			String key = keys.get(i);
			GDExtensionPtr<bool> value = VariantCaster<GDExtensionPtr<bool>>::cast(request_extension.get(key, GDExtensionPtr<bool>(nullptr)));
			result.insert(key, value);
		}
		return result;
	}

	return HashMap<String, bool *>();
}

void *OpenXRExtensionWrapper::set_system_properties_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_set_system_properties_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_instance_create_info_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_set_instance_create_info_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_session_create_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_set_session_create_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_swapchain_create_info_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_set_swapchain_create_info_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_hand_joint_locations_and_get_next_pointer(int p_hand_index, void *p_next_pointer) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_set_hand_joint_locations_and_get_next_pointer, p_hand_index, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_projection_views_and_get_next_pointer(int p_view_index, void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_projection_views_and_get_next_pointer, p_view_index, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_reference_space_create_info_and_get_next_pointer(int p_reference_space_type, void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_reference_space_create_info_and_get_next_pointer, p_reference_space_type, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_frame_wait_info_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_frame_wait_info_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_frame_end_info_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_frame_end_info_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

void *OpenXRExtensionWrapper::set_view_locate_info_and_get_next_pointer(void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_view_locate_info_and_get_next_pointer, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return nullptr;
}

PackedStringArray OpenXRExtensionWrapper::get_suggested_tracker_names() {
	PackedStringArray ret;

	if (GDVIRTUAL_CALL(_get_suggested_tracker_names, ret)) {
		return ret;
	}

	return PackedStringArray();
}

int OpenXRExtensionWrapper::get_composition_layer_count() {
	int count = 0;
	GDVIRTUAL_CALL(_get_composition_layer_count, count);
	return count;
}

XrCompositionLayerBaseHeader *OpenXRExtensionWrapper::get_composition_layer(int p_index) {
	uint64_t pointer;

	if (GDVIRTUAL_CALL(_get_composition_layer, p_index, pointer)) {
		return reinterpret_cast<XrCompositionLayerBaseHeader *>(pointer);
	}

	return nullptr;
}

int OpenXRExtensionWrapper::get_composition_layer_order(int p_index) {
	int order = 0;
	GDVIRTUAL_CALL(_get_composition_layer_order, p_index, order);
	return order;
}

void OpenXRExtensionWrapper::on_register_metadata() {
	GDVIRTUAL_CALL(_on_register_metadata);
}

void OpenXRExtensionWrapper::on_before_instance_created() {
	GDVIRTUAL_CALL(_on_before_instance_created);
}

void OpenXRExtensionWrapper::on_instance_created(const XrInstance p_instance) {
	uint64_t instance = (uint64_t)p_instance;
	GDVIRTUAL_CALL(_on_instance_created, instance);
}

void OpenXRExtensionWrapper::on_instance_destroyed() {
	GDVIRTUAL_CALL(_on_instance_destroyed);
}

void OpenXRExtensionWrapper::on_session_created(const XrSession p_session) {
	uint64_t session = (uint64_t)p_session;
	GDVIRTUAL_CALL(_on_session_created, session);
}

void OpenXRExtensionWrapper::on_process() {
	GDVIRTUAL_CALL(_on_process);
}

void OpenXRExtensionWrapper::on_pre_render() {
	GDVIRTUAL_CALL(_on_pre_render);
}

void OpenXRExtensionWrapper::on_main_swapchains_created() {
	GDVIRTUAL_CALL(_on_main_swapchains_created);
}

void OpenXRExtensionWrapper::on_session_destroyed() {
	GDVIRTUAL_CALL(_on_session_destroyed);
}

void OpenXRExtensionWrapper::on_pre_draw_viewport(RID p_render_target) {
	GDVIRTUAL_CALL(_on_pre_draw_viewport, p_render_target);
}

void OpenXRExtensionWrapper::on_post_draw_viewport(RID p_render_target) {
	GDVIRTUAL_CALL(_on_post_draw_viewport, p_render_target);
}

void OpenXRExtensionWrapper::on_state_idle() {
	GDVIRTUAL_CALL(_on_state_idle);
}

void OpenXRExtensionWrapper::on_state_ready() {
	GDVIRTUAL_CALL(_on_state_ready);
}

void OpenXRExtensionWrapper::on_state_synchronized() {
	GDVIRTUAL_CALL(_on_state_synchronized);
}

void OpenXRExtensionWrapper::on_state_visible() {
	GDVIRTUAL_CALL(_on_state_visible);
}

void OpenXRExtensionWrapper::on_state_focused() {
	GDVIRTUAL_CALL(_on_state_focused);
}

void OpenXRExtensionWrapper::on_state_stopping() {
	GDVIRTUAL_CALL(_on_state_stopping);
}

void OpenXRExtensionWrapper::on_state_loss_pending() {
	GDVIRTUAL_CALL(_on_state_loss_pending);
}

void OpenXRExtensionWrapper::on_state_exiting() {
	GDVIRTUAL_CALL(_on_state_exiting);
}

bool OpenXRExtensionWrapper::on_event_polled(const XrEventDataBuffer &p_event) {
	bool event_polled;

	if (GDVIRTUAL_CALL(_on_event_polled, GDExtensionConstPtr<void>(&p_event), event_polled)) {
		return event_polled;
	}

	return false;
}

void *OpenXRExtensionWrapper::set_viewport_composition_layer_and_get_next_pointer(const XrCompositionLayerBaseHeader *p_layer, const Dictionary &p_property_values, void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_viewport_composition_layer_and_get_next_pointer, GDExtensionConstPtr<void>(p_layer), p_property_values, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return p_next_pointer;
}

void OpenXRExtensionWrapper::on_viewport_composition_layer_destroyed(const XrCompositionLayerBaseHeader *p_layer) {
	GDVIRTUAL_CALL(_on_viewport_composition_layer_destroyed, GDExtensionConstPtr<void>(p_layer));
}

void OpenXRExtensionWrapper::get_viewport_composition_layer_extension_properties(List<PropertyInfo> *p_property_list) {
	TypedArray<Dictionary> properties;

	if (GDVIRTUAL_CALL(_get_viewport_composition_layer_extension_properties, properties)) {
		for (int i = 0; i < properties.size(); i++) {
			p_property_list->push_back(PropertyInfo::from_dict(properties[i]));
		}
	}
}

Dictionary OpenXRExtensionWrapper::get_viewport_composition_layer_extension_property_defaults() {
	Dictionary property_defaults;
	GDVIRTUAL_CALL(_get_viewport_composition_layer_extension_property_defaults, property_defaults);
	return property_defaults;
}

void *OpenXRExtensionWrapper::set_android_surface_swapchain_create_info_and_get_next_pointer(const Dictionary &p_property_values, void *p_next_pointer) {
	uint64_t pointer = 0;

	if (GDVIRTUAL_CALL(_set_android_surface_swapchain_create_info_and_get_next_pointer, p_property_values, GDExtensionPtr<void>(p_next_pointer), pointer)) {
		return reinterpret_cast<void *>(pointer);
	}

	return p_next_pointer;
}

Ref<OpenXRAPIExtension> OpenXRExtensionWrapper::_gdextension_get_openxr_api() {
	static Ref<OpenXRAPIExtension> openxr_api_extension;
	if (unlikely(openxr_api_extension.is_null())) {
		openxr_api_extension.instantiate();
	}
	return openxr_api_extension;
}

void OpenXRExtensionWrapper::_gdextension_register_extension_wrapper() {
	OpenXRAPI::register_extension_wrapper(this);
}
