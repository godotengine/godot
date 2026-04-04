/**************************************************************************/
/*  node_path.cpp                                                         */
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

#include <godot_cpp/variant/node_path.hpp>

#include <godot_cpp/core/binder_common.hpp>

#include <godot_cpp/godot.hpp>

#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/variant.hpp>

#include <godot_cpp/core/builtin_ptrcall.hpp>

#include <utility>

namespace godot {

NodePath::_MethodBindings NodePath::_method_bindings;

void NodePath::_init_bindings_constructors_destructor() {
	_method_bindings.from_variant_constructor = ::godot::gdextension_interface::get_variant_to_type_constructor(GDEXTENSION_VARIANT_TYPE_NODE_PATH);
	_method_bindings.constructor_0 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_NODE_PATH, 0);
	_method_bindings.constructor_1 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_NODE_PATH, 1);
	_method_bindings.constructor_2 = ::godot::gdextension_interface::variant_get_ptr_constructor(GDEXTENSION_VARIANT_TYPE_NODE_PATH, 2);
	_method_bindings.destructor = ::godot::gdextension_interface::variant_get_ptr_destructor(GDEXTENSION_VARIANT_TYPE_NODE_PATH);
}
void NodePath::init_bindings() {
	NodePath::_init_bindings_constructors_destructor();
	StringName _gde_name;
	_gde_name = StringName("is_absolute");
	_method_bindings.method_is_absolute = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 3918633141);
	_gde_name = StringName("get_name_count");
	_method_bindings.method_get_name_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_name");
	_method_bindings.method_get_name = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 2948586938);
	_gde_name = StringName("get_subname_count");
	_method_bindings.method_get_subname_count = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("hash");
	_method_bindings.method_hash = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 3173160232);
	_gde_name = StringName("get_subname");
	_method_bindings.method_get_subname = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 2948586938);
	_gde_name = StringName("get_concatenated_names");
	_method_bindings.method_get_concatenated_names = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("get_concatenated_subnames");
	_method_bindings.method_get_concatenated_subnames = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 1825232092);
	_gde_name = StringName("slice");
	_method_bindings.method_slice = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 421628484);
	_gde_name = StringName("get_as_property_path");
	_method_bindings.method_get_as_property_path = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 1598598043);
	_gde_name = StringName("is_empty");
	_method_bindings.method_is_empty = ::godot::gdextension_interface::variant_get_ptr_builtin_method(GDEXTENSION_VARIANT_TYPE_NODE_PATH, _gde_name._native_ptr(), 3918633141);
	_method_bindings.operator_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not_equal_Variant = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_not = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_NIL);
	_method_bindings.operator_equal_NodePath = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_EQUAL, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_NODE_PATH);
	_method_bindings.operator_not_equal_NodePath = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_NOT_EQUAL, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_NODE_PATH);
	_method_bindings.operator_in_Dictionary = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_DICTIONARY);
	_method_bindings.operator_in_Array = ::godot::gdextension_interface::variant_get_ptr_operator_evaluator(GDEXTENSION_VARIANT_OP_IN, GDEXTENSION_VARIANT_TYPE_NODE_PATH, GDEXTENSION_VARIANT_TYPE_ARRAY);
}

NodePath::NodePath(const Variant *p_variant) {
	_method_bindings.from_variant_constructor(&opaque, p_variant->_native_ptr());
}

NodePath::NodePath() {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_0, &opaque);
}

NodePath::NodePath(const NodePath &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_from);
}

NodePath::NodePath(const String &p_from) {
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_2, &opaque, &p_from);
}

NodePath::NodePath(NodePath &&p_other) {
	std::swap(opaque, p_other.opaque);
}

NodePath::~NodePath() {
	_method_bindings.destructor(&opaque);
}

bool NodePath::is_absolute() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_absolute, (GDExtensionTypePtr)&opaque);
}

int64_t NodePath::get_name_count() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_name_count, (GDExtensionTypePtr)&opaque);
}

StringName NodePath::get_name(int64_t p_idx) const {
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_name, (GDExtensionTypePtr)&opaque, &p_idx_encoded);
}

int64_t NodePath::get_subname_count() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_get_subname_count, (GDExtensionTypePtr)&opaque);
}

int64_t NodePath::hash() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int64_t>(_method_bindings.method_hash, (GDExtensionTypePtr)&opaque);
}

StringName NodePath::get_subname(int64_t p_idx) const {
	int64_t p_idx_encoded;
	PtrToArg<int64_t>::encode(p_idx, &p_idx_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_subname, (GDExtensionTypePtr)&opaque, &p_idx_encoded);
}

StringName NodePath::get_concatenated_names() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_concatenated_names, (GDExtensionTypePtr)&opaque);
}

StringName NodePath::get_concatenated_subnames() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<StringName>(_method_bindings.method_get_concatenated_subnames, (GDExtensionTypePtr)&opaque);
}

NodePath NodePath::slice(int64_t p_begin, int64_t p_end) const {
	int64_t p_begin_encoded;
	PtrToArg<int64_t>::encode(p_begin, &p_begin_encoded);
	int64_t p_end_encoded;
	PtrToArg<int64_t>::encode(p_end, &p_end_encoded);
	return ::godot::internal::_call_builtin_method_ptr_ret<NodePath>(_method_bindings.method_slice, (GDExtensionTypePtr)&opaque, &p_begin_encoded, &p_end_encoded);
}

NodePath NodePath::get_as_property_path() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<NodePath>(_method_bindings.method_get_as_property_path, (GDExtensionTypePtr)&opaque);
}

bool NodePath::is_empty() const {
	return ::godot::internal::_call_builtin_method_ptr_ret<int8_t>(_method_bindings.method_is_empty, (GDExtensionTypePtr)&opaque);
}

bool NodePath::operator==(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool NodePath::operator!=(const Variant &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_Variant, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool NodePath::operator!() const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr) nullptr);
}

bool NodePath::operator==(const NodePath &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_equal_NodePath, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

bool NodePath::operator!=(const NodePath &p_other) const {
	return ::godot::internal::_call_builtin_operator_ptr<int8_t>(_method_bindings.operator_not_equal_NodePath, (GDExtensionConstTypePtr)&opaque, (GDExtensionConstTypePtr)&p_other);
}

NodePath &NodePath::operator=(const NodePath &p_other) {
	_method_bindings.destructor(&opaque);
	::godot::internal::_call_builtin_constructor(_method_bindings.constructor_1, &opaque, &p_other);
	return *this;
}

NodePath &NodePath::operator=(NodePath &&p_other) {
	std::swap(opaque, p_other.opaque);
	return *this;
}

} //namespace godot
