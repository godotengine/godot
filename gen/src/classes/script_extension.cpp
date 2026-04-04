/**************************************************************************/
/*  script_extension.cpp                                                  */
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

#include <godot_cpp/classes/script_extension.hpp>

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/core/engine_ptrcall.hpp>
#include <godot_cpp/core/error_macros.hpp>

#include <godot_cpp/classes/script_language.hpp>
#include <godot_cpp/core/object.hpp>

namespace godot {

bool ScriptExtension::_editor_can_reload_from_file() {
	return false;
}

void ScriptExtension::_placeholder_erased(void *p_placeholder) {}

bool ScriptExtension::_can_instantiate() const {
	return false;
}

Ref<Script> ScriptExtension::_get_base_script() const {
	return Ref<Script>();
}

StringName ScriptExtension::_get_global_name() const {
	return StringName();
}

bool ScriptExtension::_inherits_script(const Ref<Script> &p_script) const {
	return false;
}

StringName ScriptExtension::_get_instance_base_type() const {
	return StringName();
}

void *ScriptExtension::_instance_create(Object *p_for_object) const {
	return nullptr;
}

void *ScriptExtension::_placeholder_instance_create(Object *p_for_object) const {
	return nullptr;
}

bool ScriptExtension::_instance_has(Object *p_object) const {
	return false;
}

bool ScriptExtension::_has_source_code() const {
	return false;
}

String ScriptExtension::_get_source_code() const {
	return String();
}

void ScriptExtension::_set_source_code(const String &p_code) {}

Error ScriptExtension::_reload(bool p_keep_state) {
	return Error(0);
}

StringName ScriptExtension::_get_doc_class_name() const {
	return StringName();
}

TypedArray<Dictionary> ScriptExtension::_get_documentation() const {
	return TypedArray<Dictionary>();
}

String ScriptExtension::_get_class_icon_path() const {
	return String();
}

bool ScriptExtension::_has_method(const StringName &p_method) const {
	return false;
}

bool ScriptExtension::_has_static_method(const StringName &p_method) const {
	return false;
}

Variant ScriptExtension::_get_script_method_argument_count(const StringName &p_method) const {
	return Variant();
}

Dictionary ScriptExtension::_get_method_info(const StringName &p_method) const {
	return Dictionary();
}

bool ScriptExtension::_is_tool() const {
	return false;
}

bool ScriptExtension::_is_valid() const {
	return false;
}

bool ScriptExtension::_is_abstract() const {
	return false;
}

ScriptLanguage *ScriptExtension::_get_language() const {
	return nullptr;
}

bool ScriptExtension::_has_script_signal(const StringName &p_signal) const {
	return false;
}

TypedArray<Dictionary> ScriptExtension::_get_script_signal_list() const {
	return TypedArray<Dictionary>();
}

bool ScriptExtension::_has_property_default_value(const StringName &p_property) const {
	return false;
}

Variant ScriptExtension::_get_property_default_value(const StringName &p_property) const {
	return Variant();
}

void ScriptExtension::_update_exports() {}

TypedArray<Dictionary> ScriptExtension::_get_script_method_list() const {
	return TypedArray<Dictionary>();
}

TypedArray<Dictionary> ScriptExtension::_get_script_property_list() const {
	return TypedArray<Dictionary>();
}

int32_t ScriptExtension::_get_member_line(const StringName &p_member) const {
	return 0;
}

Dictionary ScriptExtension::_get_constants() const {
	return Dictionary();
}

TypedArray<StringName> ScriptExtension::_get_members() const {
	return TypedArray<StringName>();
}

bool ScriptExtension::_is_placeholder_fallback_enabled() const {
	return false;
}

Variant ScriptExtension::_get_rpc_config() const {
	return Variant();
}

} // namespace godot
