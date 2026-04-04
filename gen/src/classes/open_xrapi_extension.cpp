/**************************************************************************/
/*  open_xrapi_extension.cpp                                              */
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

#include <godot_cpp/classes/open_xrapi_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/open_xr_extension_wrapper.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/rect2i.hpp>
#include <godot_cpp/variant/vector2i.hpp>

namespace godot {

uint64_t OpenXRAPIExtension::get_openxr_version() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_openxr_version")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t OpenXRAPIExtension::get_instance() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_instance")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t OpenXRAPIExtension::get_system_id() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_system_id")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

uint64_t OpenXRAPIExtension::get_session() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_session")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

Transform3D OpenXRAPIExtension::transform_from_pose(const void *p_pose) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("transform_from_pose")._native_ptr(), 2963875352);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Transform3D()));
	return ::godot::internal::_call_native_mb_ret<Transform3D>(_gde_method_bind, _owner, &p_pose);
}

bool OpenXRAPIExtension::xr_result(uint64_t p_result, const String &p_format, const Array &p_args) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("xr_result")._native_ptr(), 3886436197);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_result_encoded;
	PtrToArg<int64_t>::encode(p_result, &p_result_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_result_encoded, &p_format, &p_args);
}

bool OpenXRAPIExtension::openxr_is_enabled(bool p_check_run_in_editor) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_is_enabled")._native_ptr(), 2703660260);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int8_t p_check_run_in_editor_encoded;
	PtrToArg<bool>::encode(p_check_run_in_editor, &p_check_run_in_editor_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, nullptr, &p_check_run_in_editor_encoded);
}

uint64_t OpenXRAPIExtension::get_instance_proc_addr(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_instance_proc_addr")._native_ptr(), 1597066294);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_name);
}

String OpenXRAPIExtension::get_error_string(uint64_t p_result) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_error_string")._native_ptr(), 990163283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_result_encoded;
	PtrToArg<int64_t>::encode(p_result, &p_result_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_result_encoded);
}

String OpenXRAPIExtension::get_swapchain_format_name(int64_t p_swapchain_format) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_swapchain_format_name")._native_ptr(), 990163283);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_swapchain_format_encoded;
	PtrToArg<int64_t>::encode(p_swapchain_format, &p_swapchain_format_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_swapchain_format_encoded);
}

void OpenXRAPIExtension::set_object_name(int64_t p_object_type, uint64_t p_object_handle, const String &p_object_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_object_name")._native_ptr(), 2285447957);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_object_type_encoded;
	PtrToArg<int64_t>::encode(p_object_type, &p_object_type_encoded);
	int64_t p_object_handle_encoded;
	PtrToArg<int64_t>::encode(p_object_handle, &p_object_handle_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_object_type_encoded, &p_object_handle_encoded, &p_object_name);
}

void OpenXRAPIExtension::begin_debug_label_region(const String &p_label_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("begin_debug_label_region")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label_name);
}

void OpenXRAPIExtension::end_debug_label_region() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("end_debug_label_region")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

void OpenXRAPIExtension::insert_debug_label(const String &p_label_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("insert_debug_label")._native_ptr(), 83702148);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_label_name);
}

bool OpenXRAPIExtension::is_initialized() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("is_initialized")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

bool OpenXRAPIExtension::is_running() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("is_running")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void OpenXRAPIExtension::set_custom_play_space(const void *p_space) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_custom_play_space")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_space);
}

uint64_t OpenXRAPIExtension::get_play_space() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_play_space")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

int64_t OpenXRAPIExtension::get_predicted_display_time() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_predicted_display_time")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int64_t OpenXRAPIExtension::get_next_frame_time() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_next_frame_time")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

bool OpenXRAPIExtension::can_render() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("can_render")._native_ptr(), 2240911060);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

RID OpenXRAPIExtension::find_action(const String &p_name, const RID &p_action_set) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("find_action")._native_ptr(), 4106179378);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_name, &p_action_set);
}

uint64_t OpenXRAPIExtension::action_get_handle(const RID &p_action) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("action_get_handle")._native_ptr(), 3917799429);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_action);
}

uint64_t OpenXRAPIExtension::get_hand_tracker(int32_t p_hand_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_hand_tracker")._native_ptr(), 3744713108);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_hand_index_encoded;
	PtrToArg<int64_t>::encode(p_hand_index, &p_hand_index_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_hand_index_encoded);
}

void OpenXRAPIExtension::register_composition_layer_provider(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("register_composition_layer_provider")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

void OpenXRAPIExtension::unregister_composition_layer_provider(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("unregister_composition_layer_provider")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

void OpenXRAPIExtension::register_projection_views_extension(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("register_projection_views_extension")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

void OpenXRAPIExtension::unregister_projection_views_extension(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("unregister_projection_views_extension")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

void OpenXRAPIExtension::register_frame_info_extension(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("register_frame_info_extension")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

void OpenXRAPIExtension::unregister_frame_info_extension(OpenXRExtensionWrapper *p_extension) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("unregister_frame_info_extension")._native_ptr(), 1477360496);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, (p_extension != nullptr ? &p_extension->_owner : nullptr));
}

double OpenXRAPIExtension::get_render_state_z_near() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_render_state_z_near")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

double OpenXRAPIExtension::get_render_state_z_far() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_render_state_z_far")._native_ptr(), 191475506);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner);
}

void OpenXRAPIExtension::set_velocity_texture(const RID &p_render_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_velocity_texture")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_target);
}

void OpenXRAPIExtension::set_velocity_depth_texture(const RID &p_render_target) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_velocity_depth_texture")._native_ptr(), 2722037293);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_target);
}

void OpenXRAPIExtension::set_velocity_target_size(const Vector2i &p_target_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_velocity_target_size")._native_ptr(), 1130785943);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_target_size);
}

PackedInt64Array OpenXRAPIExtension::get_supported_swapchain_formats() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_supported_swapchain_formats")._native_ptr(), 3851388692);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (PackedInt64Array()));
	return ::godot::internal::_call_native_mb_ret<PackedInt64Array>(_gde_method_bind, _owner);
}

uint64_t OpenXRAPIExtension::openxr_swapchain_create(uint64_t p_create_flags, uint64_t p_usage_flags, int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_create")._native_ptr(), 2162228999);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_create_flags_encoded;
	PtrToArg<int64_t>::encode(p_create_flags, &p_create_flags_encoded);
	int64_t p_usage_flags_encoded;
	PtrToArg<int64_t>::encode(p_usage_flags, &p_usage_flags_encoded);
	int64_t p_swapchain_format_encoded;
	PtrToArg<int64_t>::encode(p_swapchain_format, &p_swapchain_format_encoded);
	int64_t p_width_encoded;
	PtrToArg<int64_t>::encode(p_width, &p_width_encoded);
	int64_t p_height_encoded;
	PtrToArg<int64_t>::encode(p_height, &p_height_encoded);
	int64_t p_sample_count_encoded;
	PtrToArg<int64_t>::encode(p_sample_count, &p_sample_count_encoded);
	int64_t p_array_size_encoded;
	PtrToArg<int64_t>::encode(p_array_size, &p_array_size_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_create_flags_encoded, &p_usage_flags_encoded, &p_swapchain_format_encoded, &p_width_encoded, &p_height_encoded, &p_sample_count_encoded, &p_array_size_encoded);
}

void OpenXRAPIExtension::openxr_swapchain_free(uint64_t p_swapchain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_free")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_swapchain_encoded;
	PtrToArg<int64_t>::encode(p_swapchain, &p_swapchain_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_swapchain_encoded);
}

uint64_t OpenXRAPIExtension::openxr_swapchain_get_swapchain(uint64_t p_swapchain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_get_swapchain")._native_ptr(), 3744713108);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	int64_t p_swapchain_encoded;
	PtrToArg<int64_t>::encode(p_swapchain, &p_swapchain_encoded);
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner, &p_swapchain_encoded);
}

void OpenXRAPIExtension::openxr_swapchain_acquire(uint64_t p_swapchain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_acquire")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_swapchain_encoded;
	PtrToArg<int64_t>::encode(p_swapchain, &p_swapchain_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_swapchain_encoded);
}

RID OpenXRAPIExtension::openxr_swapchain_get_image(uint64_t p_swapchain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_get_image")._native_ptr(), 937000113);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (RID()));
	int64_t p_swapchain_encoded;
	PtrToArg<int64_t>::encode(p_swapchain, &p_swapchain_encoded);
	return ::godot::internal::_call_native_mb_ret<RID>(_gde_method_bind, _owner, &p_swapchain_encoded);
}

void OpenXRAPIExtension::openxr_swapchain_release(uint64_t p_swapchain) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("openxr_swapchain_release")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_swapchain_encoded;
	PtrToArg<int64_t>::encode(p_swapchain, &p_swapchain_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_swapchain_encoded);
}

uint64_t OpenXRAPIExtension::get_projection_layer() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("get_projection_layer")._native_ptr(), 2455072627);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

void OpenXRAPIExtension::set_render_region(const Rect2i &p_render_region) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_render_region")._native_ptr(), 1763793166);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_render_region);
}

void OpenXRAPIExtension::set_emulate_environment_blend_mode_alpha_blend(bool p_enabled) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("set_emulate_environment_blend_mode_alpha_blend")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enabled_encoded;
	PtrToArg<bool>::encode(p_enabled, &p_enabled_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enabled_encoded);
}

OpenXRAPIExtension::OpenXRAlphaBlendModeSupport OpenXRAPIExtension::is_environment_blend_mode_alpha_supported() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("is_environment_blend_mode_alpha_supported")._native_ptr(), 1579290861);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (OpenXRAPIExtension::OpenXRAlphaBlendModeSupport(0)));
	return (OpenXRAPIExtension::OpenXRAlphaBlendModeSupport)::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

void OpenXRAPIExtension::update_main_swapchain_size() {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(OpenXRAPIExtension::get_class_static()._native_ptr(), StringName("update_main_swapchain_size")._native_ptr(), 3218959716);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner);
}

} // namespace godot
