/**************************************************************************/
/*  open_xrapi_extension.hpp                                              */
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
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/packed_int64_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/transform3d.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Array;
class OpenXRExtensionWrapper;
struct Rect2i;
struct Vector2i;

class OpenXRAPIExtension : public RefCounted {
	GDEXTENSION_CLASS(OpenXRAPIExtension, RefCounted)

public:
	enum OpenXRAlphaBlendModeSupport {
		OPENXR_ALPHA_BLEND_MODE_SUPPORT_NONE = 0,
		OPENXR_ALPHA_BLEND_MODE_SUPPORT_REAL = 1,
		OPENXR_ALPHA_BLEND_MODE_SUPPORT_EMULATING = 2,
	};

	uint64_t get_openxr_version();
	uint64_t get_instance();
	uint64_t get_system_id();
	uint64_t get_session();
	Transform3D transform_from_pose(const void *p_pose);
	bool xr_result(uint64_t p_result, const String &p_format, const Array &p_args);
	static bool openxr_is_enabled(bool p_check_run_in_editor);
	uint64_t get_instance_proc_addr(const String &p_name);
	String get_error_string(uint64_t p_result);
	String get_swapchain_format_name(int64_t p_swapchain_format);
	void set_object_name(int64_t p_object_type, uint64_t p_object_handle, const String &p_object_name);
	void begin_debug_label_region(const String &p_label_name);
	void end_debug_label_region();
	void insert_debug_label(const String &p_label_name);
	bool is_initialized();
	bool is_running();
	void set_custom_play_space(const void *p_space);
	uint64_t get_play_space();
	int64_t get_predicted_display_time();
	int64_t get_next_frame_time();
	bool can_render();
	RID find_action(const String &p_name, const RID &p_action_set);
	uint64_t action_get_handle(const RID &p_action);
	uint64_t get_hand_tracker(int32_t p_hand_index);
	void register_composition_layer_provider(OpenXRExtensionWrapper *p_extension);
	void unregister_composition_layer_provider(OpenXRExtensionWrapper *p_extension);
	void register_projection_views_extension(OpenXRExtensionWrapper *p_extension);
	void unregister_projection_views_extension(OpenXRExtensionWrapper *p_extension);
	void register_frame_info_extension(OpenXRExtensionWrapper *p_extension);
	void unregister_frame_info_extension(OpenXRExtensionWrapper *p_extension);
	double get_render_state_z_near();
	double get_render_state_z_far();
	void set_velocity_texture(const RID &p_render_target);
	void set_velocity_depth_texture(const RID &p_render_target);
	void set_velocity_target_size(const Vector2i &p_target_size);
	PackedInt64Array get_supported_swapchain_formats();
	uint64_t openxr_swapchain_create(uint64_t p_create_flags, uint64_t p_usage_flags, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size);
	void openxr_swapchain_free(uint64_t p_swapchain);
	uint64_t openxr_swapchain_get_swapchain(uint64_t p_swapchain);
	void openxr_swapchain_acquire(uint64_t p_swapchain);
	RID openxr_swapchain_get_image(uint64_t p_swapchain);
	void openxr_swapchain_release(uint64_t p_swapchain);
	uint64_t get_projection_layer();
	void set_render_region(const Rect2i &p_render_region);
	void set_emulate_environment_blend_mode_alpha_blend(bool p_enabled);
	OpenXRAPIExtension::OpenXRAlphaBlendModeSupport is_environment_blend_mode_alpha_supported();
	void update_main_swapchain_size();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(OpenXRAPIExtension::OpenXRAlphaBlendModeSupport);

