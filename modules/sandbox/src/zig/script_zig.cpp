/**************************************************************************/
/*  script_zig.cpp                                                        */
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

#include "script_zig.h"

#include "script_language_zig.h"
#include <godot_cpp/classes/file_access.hpp>
#include <godot_cpp/classes/resource_loader.hpp>

bool ZigScript::_editor_can_reload_from_file() {
	return true;
}
void ZigScript::_placeholder_erased(void *p_placeholder) {}
bool ZigScript::_can_instantiate() const {
	return false;
}
Ref<Script> ZigScript::_get_base_script() const {
	return Ref<Script>();
}
StringName ZigScript::_get_global_name() const {
	return StringName();
}
bool ZigScript::_inherits_script(const Ref<Script> &p_script) const {
	return false;
}
StringName ZigScript::_get_instance_base_type() const {
	return StringName();
}
void *ZigScript::_instance_create(Object *p_for_object) const {
	return nullptr;
}
void *ZigScript::_placeholder_instance_create(Object *p_for_object) const {
	return nullptr;
}
bool ZigScript::_instance_has(Object *p_object) const {
	return false;
}
bool ZigScript::_has_source_code() const {
	return true;
}
String ZigScript::_get_source_code() const {
	return source_code;
}
void ZigScript::_set_source_code(const String &p_code) {
	source_code = p_code;
}
Error ZigScript::_reload(bool p_keep_state) {
	return Error::OK;
}
TypedArray<Dictionary> ZigScript::_get_documentation() const {
	return TypedArray<Dictionary>();
}
String ZigScript::_get_class_icon_path() const {
	return String("res://addons/godot_sandbox/ZigScript.svg");
}
bool ZigScript::_has_method(const StringName &p_method) const {
	return false;
}
bool ZigScript::_has_static_method(const StringName &p_method) const {
	return false;
}
Dictionary ZigScript::_get_method_info(const StringName &p_method) const {
	return Dictionary();
}
bool ZigScript::_is_tool() const {
	return true;
}
bool ZigScript::_is_valid() const {
	return true;
}
bool ZigScript::_is_abstract() const {
	return true;
}
ScriptLanguage *ZigScript::_get_language() const {
	return ZigScriptLanguage::get_singleton();
}
bool ZigScript::_has_script_signal(const StringName &p_signal) const {
	return false;
}
TypedArray<Dictionary> ZigScript::_get_script_signal_list() const {
	return TypedArray<Dictionary>();
}
bool ZigScript::_has_property_default_value(const StringName &p_property) const {
	return false;
}
Variant ZigScript::_get_property_default_value(const StringName &p_property) const {
	return Variant();
}
void ZigScript::_update_exports() {}
TypedArray<Dictionary> ZigScript::_get_script_method_list() const {
	return TypedArray<Dictionary>();
}
TypedArray<Dictionary> ZigScript::_get_script_property_list() const {
	return TypedArray<Dictionary>();
}
int32_t ZigScript::_get_member_line(const StringName &p_member) const {
	return 0;
}
Dictionary ZigScript::_get_constants() const {
	return Dictionary();
}
TypedArray<StringName> ZigScript::_get_members() const {
	return TypedArray<StringName>();
}
bool ZigScript::_is_placeholder_fallback_enabled() const {
	return false;
}
Variant ZigScript::_get_rpc_config() const {
	return Variant();
}

ZigScript::ZigScript() {
	source_code = R"C0D3(
// TODO: Implement me.
)C0D3";
}
