/**************************************************************************/
/*  animation_node.cpp                                                    */
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

#include <godot_cpp/classes/animation_node.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/variant/node_path.hpp>
#include <godot_cpp/variant/string_name.hpp>

namespace godot {

bool AnimationNode::add_input(const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("add_input")._native_ptr(), 2323990056);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_name);
}

void AnimationNode::remove_input(int32_t p_index) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("remove_input")._native_ptr(), 1286410249);
	CHECK_METHOD_BIND(_gde_method_bind);
	int64_t p_index_encoded;
	PtrToArg<int64_t>::encode(p_index, &p_index_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_index_encoded);
}

bool AnimationNode::set_input_name(int32_t p_input, const String &p_name) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("set_input_name")._native_ptr(), 215573526);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	int64_t p_input_encoded;
	PtrToArg<int64_t>::encode(p_input, &p_input_encoded);
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_input_encoded, &p_name);
}

String AnimationNode::get_input_name(int32_t p_input) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("get_input_name")._native_ptr(), 844755477);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (String()));
	int64_t p_input_encoded;
	PtrToArg<int64_t>::encode(p_input, &p_input_encoded);
	return ::godot::internal::_call_native_mb_ret<String>(_gde_method_bind, _owner, &p_input_encoded);
}

int32_t AnimationNode::get_input_count() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("get_input_count")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner);
}

int32_t AnimationNode::find_input(const String &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("find_input")._native_ptr(), 1321353865);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<int64_t>(_gde_method_bind, _owner, &p_name);
}

void AnimationNode::set_filter_path(const NodePath &p_path, bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("set_filter_path")._native_ptr(), 3868023870);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_path, &p_enable_encoded);
}

bool AnimationNode::is_path_filtered(const NodePath &p_path) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("is_path_filtered")._native_ptr(), 861721659);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner, &p_path);
}

void AnimationNode::set_filter_enabled(bool p_enable) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("set_filter_enabled")._native_ptr(), 2586408642);
	CHECK_METHOD_BIND(_gde_method_bind);
	int8_t p_enable_encoded;
	PtrToArg<bool>::encode(p_enable, &p_enable_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_enable_encoded);
}

bool AnimationNode::is_filter_enabled() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("is_filter_enabled")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

uint64_t AnimationNode::get_processing_animation_tree_instance_id() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("get_processing_animation_tree_instance_id")._native_ptr(), 3905245786);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0));
	return ::godot::internal::_call_native_mb_ret<uint64_t>(_gde_method_bind, _owner);
}

bool AnimationNode::is_process_testing() const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("is_process_testing")._native_ptr(), 36873697);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (false));
	return ::godot::internal::_call_native_mb_ret<int8_t>(_gde_method_bind, _owner);
}

void AnimationNode::blend_animation(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, float p_blend, Animation::LoopedFlag p_looped_flag) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("blend_animation")._native_ptr(), 1630801826);
	CHECK_METHOD_BIND(_gde_method_bind);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	double p_delta_encoded;
	PtrToArg<double>::encode(p_delta, &p_delta_encoded);
	int8_t p_seeked_encoded;
	PtrToArg<bool>::encode(p_seeked, &p_seeked_encoded);
	int8_t p_is_external_seeking_encoded;
	PtrToArg<bool>::encode(p_is_external_seeking, &p_is_external_seeking_encoded);
	double p_blend_encoded;
	PtrToArg<double>::encode(p_blend, &p_blend_encoded);
	int64_t p_looped_flag_encoded;
	PtrToArg<int64_t>::encode(p_looped_flag, &p_looped_flag_encoded);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_animation, &p_time_encoded, &p_delta_encoded, &p_seeked_encoded, &p_is_external_seeking_encoded, &p_blend_encoded, &p_looped_flag_encoded);
}

double AnimationNode::blend_node(const StringName &p_name, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, float p_blend, AnimationNode::FilterAction p_filter, bool p_sync, bool p_test_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("blend_node")._native_ptr(), 1746075988);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	int8_t p_seek_encoded;
	PtrToArg<bool>::encode(p_seek, &p_seek_encoded);
	int8_t p_is_external_seeking_encoded;
	PtrToArg<bool>::encode(p_is_external_seeking, &p_is_external_seeking_encoded);
	double p_blend_encoded;
	PtrToArg<double>::encode(p_blend, &p_blend_encoded);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	int8_t p_sync_encoded;
	PtrToArg<bool>::encode(p_sync, &p_sync_encoded);
	int8_t p_test_only_encoded;
	PtrToArg<bool>::encode(p_test_only, &p_test_only_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_name, (p_node != nullptr ? &p_node->_owner : nullptr), &p_time_encoded, &p_seek_encoded, &p_is_external_seeking_encoded, &p_blend_encoded, &p_filter_encoded, &p_sync_encoded, &p_test_only_encoded);
}

double AnimationNode::blend_input(int32_t p_input_index, double p_time, bool p_seek, bool p_is_external_seeking, float p_blend, AnimationNode::FilterAction p_filter, bool p_sync, bool p_test_only) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("blend_input")._native_ptr(), 1361527350);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (0.0));
	int64_t p_input_index_encoded;
	PtrToArg<int64_t>::encode(p_input_index, &p_input_index_encoded);
	double p_time_encoded;
	PtrToArg<double>::encode(p_time, &p_time_encoded);
	int8_t p_seek_encoded;
	PtrToArg<bool>::encode(p_seek, &p_seek_encoded);
	int8_t p_is_external_seeking_encoded;
	PtrToArg<bool>::encode(p_is_external_seeking, &p_is_external_seeking_encoded);
	double p_blend_encoded;
	PtrToArg<double>::encode(p_blend, &p_blend_encoded);
	int64_t p_filter_encoded;
	PtrToArg<int64_t>::encode(p_filter, &p_filter_encoded);
	int8_t p_sync_encoded;
	PtrToArg<bool>::encode(p_sync, &p_sync_encoded);
	int8_t p_test_only_encoded;
	PtrToArg<bool>::encode(p_test_only, &p_test_only_encoded);
	return ::godot::internal::_call_native_mb_ret<double>(_gde_method_bind, _owner, &p_input_index_encoded, &p_time_encoded, &p_seek_encoded, &p_is_external_seeking_encoded, &p_blend_encoded, &p_filter_encoded, &p_sync_encoded, &p_test_only_encoded);
}

void AnimationNode::set_parameter(const StringName &p_name, const Variant &p_value) {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("set_parameter")._native_ptr(), 3776071444);
	CHECK_METHOD_BIND(_gde_method_bind);
	::godot::internal::_call_native_mb_no_ret(_gde_method_bind, _owner, &p_name, &p_value);
}

Variant AnimationNode::get_parameter(const StringName &p_name) const {
	static GDExtensionMethodBindPtr _gde_method_bind = ::godot::gdextension_interface::classdb_get_method_bind(AnimationNode::get_class_static()._native_ptr(), StringName("get_parameter")._native_ptr(), 2760726917);
	CHECK_METHOD_BIND_RET(_gde_method_bind, (Variant()));
	return ::godot::internal::_call_native_mb_ret<Variant>(_gde_method_bind, _owner, &p_name);
}

Dictionary AnimationNode::_get_child_nodes() const {
	return Dictionary();
}

Array AnimationNode::_get_parameter_list() const {
	return Array();
}

Ref<AnimationNode> AnimationNode::_get_child_by_name(const StringName &p_name) const {
	return Ref<AnimationNode>();
}

Variant AnimationNode::_get_parameter_default_value(const StringName &p_parameter) const {
	return Variant();
}

bool AnimationNode::_is_parameter_read_only(const StringName &p_parameter) const {
	return false;
}

double AnimationNode::_process(double p_time, bool p_seek, bool p_is_external_seeking, bool p_test_only) {
	return 0.0;
}

String AnimationNode::_get_caption() const {
	return String();
}

bool AnimationNode::_has_filter() const {
	return false;
}

} // namespace godot
