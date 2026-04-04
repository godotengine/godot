/**************************************************************************/
/*  open_xr_extension_wrapper.hpp                                         */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/core/object.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/typed_array.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class OpenXRAPIExtension;
class RID;

class OpenXRExtensionWrapper : public Object {
	GDEXTENSION_CLASS(OpenXRExtensionWrapper, Object)

public:
	Ref<OpenXRAPIExtension> get_openxr_api();
	void register_extension_wrapper();
	virtual Dictionary _get_requested_extensions(uint64_t p_xr_version);
	virtual uint64_t _set_system_properties_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_instance_create_info_and_get_next_pointer(uint64_t p_xr_version, void *p_next_pointer);
	virtual uint64_t _set_session_create_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_swapchain_create_info_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_hand_joint_locations_and_get_next_pointer(int32_t p_hand_index, void *p_next_pointer);
	virtual uint64_t _set_projection_views_and_get_next_pointer(int32_t p_view_index, void *p_next_pointer);
	virtual uint64_t _set_frame_wait_info_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_frame_end_info_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_view_locate_info_and_get_next_pointer(void *p_next_pointer);
	virtual uint64_t _set_reference_space_create_info_and_get_next_pointer(int32_t p_reference_space_type, void *p_next_pointer);
	virtual void _prepare_view_configuration(int32_t p_view_count);
	virtual uint64_t _set_view_configuration_and_get_next_pointer(uint32_t p_view, void *p_next_pointer);
	virtual void _print_view_configuration_info(int32_t p_view) const;
	virtual int32_t _get_composition_layer_count();
	virtual uint64_t _get_composition_layer(int32_t p_index);
	virtual int32_t _get_composition_layer_order(int32_t p_index);
	virtual PackedStringArray _get_suggested_tracker_names();
	virtual void _on_register_metadata();
	virtual void _on_before_instance_created();
	virtual void _on_instance_created(uint64_t p_instance);
	virtual void _on_instance_destroyed();
	virtual void _on_session_created(uint64_t p_session);
	virtual void _on_process();
	virtual void _on_sync_actions();
	virtual void _on_pre_render();
	virtual void _on_main_swapchains_created();
	virtual void _on_pre_draw_viewport(const RID &p_viewport);
	virtual void _on_post_draw_viewport(const RID &p_viewport);
	virtual void _on_session_destroyed();
	virtual void _on_state_idle();
	virtual void _on_state_ready();
	virtual void _on_state_synchronized();
	virtual void _on_state_visible();
	virtual void _on_state_focused();
	virtual void _on_state_stopping();
	virtual void _on_state_loss_pending();
	virtual void _on_state_exiting();
	virtual bool _on_event_polled(const void *p_event);
	virtual uint64_t _set_viewport_composition_layer_and_get_next_pointer(const void *p_layer, const Dictionary &p_property_values, void *p_next_pointer);
	virtual TypedArray<Dictionary> _get_viewport_composition_layer_extension_properties();
	virtual Dictionary _get_viewport_composition_layer_extension_property_defaults();
	virtual void _on_viewport_composition_layer_destroyed(const void *p_layer);
	virtual uint64_t _set_android_surface_swapchain_create_info_and_get_next_pointer(const Dictionary &p_property_values, void *p_next_pointer);

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		Object::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_requested_extensions), decltype(&T::_get_requested_extensions)>) {
			BIND_VIRTUAL_METHOD(T, _get_requested_extensions, 3554694381);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_system_properties_and_get_next_pointer), decltype(&T::_set_system_properties_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_system_properties_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_instance_create_info_and_get_next_pointer), decltype(&T::_set_instance_create_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_instance_create_info_and_get_next_pointer, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_session_create_and_get_next_pointer), decltype(&T::_set_session_create_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_session_create_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_swapchain_create_info_and_get_next_pointer), decltype(&T::_set_swapchain_create_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_swapchain_create_info_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_hand_joint_locations_and_get_next_pointer), decltype(&T::_set_hand_joint_locations_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_hand_joint_locations_and_get_next_pointer, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_projection_views_and_get_next_pointer), decltype(&T::_set_projection_views_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_projection_views_and_get_next_pointer, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_frame_wait_info_and_get_next_pointer), decltype(&T::_set_frame_wait_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_frame_wait_info_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_frame_end_info_and_get_next_pointer), decltype(&T::_set_frame_end_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_frame_end_info_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_view_locate_info_and_get_next_pointer), decltype(&T::_set_view_locate_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_view_locate_info_and_get_next_pointer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_reference_space_create_info_and_get_next_pointer), decltype(&T::_set_reference_space_create_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_reference_space_create_info_and_get_next_pointer, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_prepare_view_configuration), decltype(&T::_prepare_view_configuration)>) {
			BIND_VIRTUAL_METHOD(T, _prepare_view_configuration, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_view_configuration_and_get_next_pointer), decltype(&T::_set_view_configuration_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_view_configuration_and_get_next_pointer, 50157827);
		}
		if constexpr (!std::is_same_v<decltype(&B::_print_view_configuration_info), decltype(&T::_print_view_configuration_info)>) {
			BIND_VIRTUAL_METHOD(T, _print_view_configuration_info, 998575451);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_composition_layer_count), decltype(&T::_get_composition_layer_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_composition_layer_count, 2455072627);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_composition_layer), decltype(&T::_get_composition_layer)>) {
			BIND_VIRTUAL_METHOD(T, _get_composition_layer, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_composition_layer_order), decltype(&T::_get_composition_layer_order)>) {
			BIND_VIRTUAL_METHOD(T, _get_composition_layer_order, 3744713108);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_suggested_tracker_names), decltype(&T::_get_suggested_tracker_names)>) {
			BIND_VIRTUAL_METHOD(T, _get_suggested_tracker_names, 2981934095);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_register_metadata), decltype(&T::_on_register_metadata)>) {
			BIND_VIRTUAL_METHOD(T, _on_register_metadata, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_before_instance_created), decltype(&T::_on_before_instance_created)>) {
			BIND_VIRTUAL_METHOD(T, _on_before_instance_created, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_instance_created), decltype(&T::_on_instance_created)>) {
			BIND_VIRTUAL_METHOD(T, _on_instance_created, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_instance_destroyed), decltype(&T::_on_instance_destroyed)>) {
			BIND_VIRTUAL_METHOD(T, _on_instance_destroyed, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_session_created), decltype(&T::_on_session_created)>) {
			BIND_VIRTUAL_METHOD(T, _on_session_created, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_process), decltype(&T::_on_process)>) {
			BIND_VIRTUAL_METHOD(T, _on_process, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_sync_actions), decltype(&T::_on_sync_actions)>) {
			BIND_VIRTUAL_METHOD(T, _on_sync_actions, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_pre_render), decltype(&T::_on_pre_render)>) {
			BIND_VIRTUAL_METHOD(T, _on_pre_render, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_main_swapchains_created), decltype(&T::_on_main_swapchains_created)>) {
			BIND_VIRTUAL_METHOD(T, _on_main_swapchains_created, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_pre_draw_viewport), decltype(&T::_on_pre_draw_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _on_pre_draw_viewport, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_post_draw_viewport), decltype(&T::_on_post_draw_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _on_post_draw_viewport, 2722037293);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_session_destroyed), decltype(&T::_on_session_destroyed)>) {
			BIND_VIRTUAL_METHOD(T, _on_session_destroyed, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_idle), decltype(&T::_on_state_idle)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_idle, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_ready), decltype(&T::_on_state_ready)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_ready, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_synchronized), decltype(&T::_on_state_synchronized)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_synchronized, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_visible), decltype(&T::_on_state_visible)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_visible, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_focused), decltype(&T::_on_state_focused)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_focused, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_stopping), decltype(&T::_on_state_stopping)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_stopping, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_loss_pending), decltype(&T::_on_state_loss_pending)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_loss_pending, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_state_exiting), decltype(&T::_on_state_exiting)>) {
			BIND_VIRTUAL_METHOD(T, _on_state_exiting, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_event_polled), decltype(&T::_on_event_polled)>) {
			BIND_VIRTUAL_METHOD(T, _on_event_polled, 3067735520);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_viewport_composition_layer_and_get_next_pointer), decltype(&T::_set_viewport_composition_layer_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_viewport_composition_layer_and_get_next_pointer, 2250464348);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_viewport_composition_layer_extension_properties), decltype(&T::_get_viewport_composition_layer_extension_properties)>) {
			BIND_VIRTUAL_METHOD(T, _get_viewport_composition_layer_extension_properties, 2915620761);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_viewport_composition_layer_extension_property_defaults), decltype(&T::_get_viewport_composition_layer_extension_property_defaults)>) {
			BIND_VIRTUAL_METHOD(T, _get_viewport_composition_layer_extension_property_defaults, 2382534195);
		}
		if constexpr (!std::is_same_v<decltype(&B::_on_viewport_composition_layer_destroyed), decltype(&T::_on_viewport_composition_layer_destroyed)>) {
			BIND_VIRTUAL_METHOD(T, _on_viewport_composition_layer_destroyed, 1286410249);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_android_surface_swapchain_create_info_and_get_next_pointer), decltype(&T::_set_android_surface_swapchain_create_info_and_get_next_pointer)>) {
			BIND_VIRTUAL_METHOD(T, _set_android_surface_swapchain_create_info_and_get_next_pointer, 3726637545);
		}
	}

public:
};

} // namespace godot

