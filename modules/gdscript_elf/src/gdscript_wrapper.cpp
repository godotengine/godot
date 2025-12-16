/**************************************************************************/
/*  gdscript_wrapper.cpp                                                  */
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

#include "gdscript_wrapper.h"

#include "gdscript_bytecode_c_codegen.h"
#include "modules/gdscript/gdscript.h"

void GDScriptWrapper::_bind_methods() {
}

GDScriptWrapper::GDScriptWrapper() {
	original_script = Ref<GDScript>();
}

GDScriptWrapper::~GDScriptWrapper() {
	original_script.unref();
}

void GDScriptWrapper::set_original_script(const Ref<GDScript> &p_script) {
	original_script = p_script;
}

// Script interface - all methods delegate to original
bool GDScriptWrapper::can_instantiate() const {
	if (original_script.is_null()) { return false; }
	return original_script->can_instantiate();
}

Ref<Script> GDScriptWrapper::get_base_script() const {
	if (original_script.is_null()) { return Ref<Script>(); }
	return original_script->get_base_script();
}

StringName GDScriptWrapper::get_global_name() const {
	if (original_script.is_null()) { return StringName(); }
	return original_script->get_global_name();
}

bool GDScriptWrapper::inherits_script(const Ref<Script> &p_script) const {
	if (original_script.is_null()) { return false; }
	return original_script->inherits_script(p_script);
}

StringName GDScriptWrapper::get_instance_base_type() const {
	if (original_script.is_null()) { return StringName(); }
	return original_script->get_instance_base_type();
}

ScriptInstance *GDScriptWrapper::instance_create(Object *p_this) {
	if (original_script.is_null()) { return nullptr; }
	return original_script->instance_create(p_this);
}

PlaceHolderScriptInstance *GDScriptWrapper::placeholder_instance_create(Object *p_this) {
	if (original_script.is_null()) { return nullptr; }
	return original_script->placeholder_instance_create(p_this);
}

bool GDScriptWrapper::instance_has(const Object *p_this) const {
	if (original_script.is_null()) { return false; }
	return original_script->instance_has(p_this);
}

bool GDScriptWrapper::has_source_code() const {
	if (original_script.is_null()) { return false; }
	return original_script->has_source_code();
}

String GDScriptWrapper::get_source_code() const {
	if (original_script.is_null()) { return String(); }
	return original_script->get_source_code();
}

void GDScriptWrapper::set_source_code(const String &p_code) {
	if (original_script.is_null()) { return; }
	original_script->set_source_code(p_code);
}

Error GDScriptWrapper::reload(bool p_keep_state) {
	ERR_FAIL_COND_V(original_script.is_null(), ERR_INVALID_DATA);
	Error err = original_script->reload(p_keep_state);
	if (err == OK && original_script->is_valid()) {
		// Generate C99 code for each function (no ELF compilation)
		GDScriptBytecodeCCodeGenerator codegen;
		for (const KeyValue<StringName, GDScriptFunction *> &E : original_script->get_member_functions()) {
			GDScriptFunction *func = E.value;
			if (func) {
				String c_code = codegen.generate_c_code(func);
				// C code generated but not compiled - just for inspection/debugging
			}
		}
	}
	return err;
}

bool GDScriptWrapper::has_method(const StringName &p_method) const {
	if (original_script.is_null()) { return false; }
	return original_script->has_method(p_method);
}

bool GDScriptWrapper::has_static_method(const StringName &p_method) const {
	if (original_script.is_null()) { return false; }
	return original_script->has_static_method(p_method);
}

MethodInfo GDScriptWrapper::get_method_info(const StringName &p_method) const {
	if (original_script.is_null()) { return MethodInfo(); }
	return original_script->get_method_info(p_method);
}

bool GDScriptWrapper::is_tool() const {
	if (original_script.is_null()) { return false; }
	return original_script->is_tool();
}

bool GDScriptWrapper::is_valid() const {
	if (original_script.is_null()) { return false; }
	return original_script->is_valid();
}

bool GDScriptWrapper::is_abstract() const {
	if (original_script.is_null()) { return false; }
	return original_script->is_abstract();
}

ScriptLanguage *GDScriptWrapper::get_language() const {
	if (original_script.is_null()) { return nullptr; }
	return original_script->get_language();
}

bool GDScriptWrapper::has_script_signal(const StringName &p_signal) const {
	if (original_script.is_null()) { return false; }
	return original_script->has_script_signal(p_signal);
}

void GDScriptWrapper::get_script_signal_list(List<MethodInfo> *p_signals) const {
	if (original_script.is_null()) { return; }
	original_script->get_script_signal_list(p_signals);
}

bool GDScriptWrapper::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	if (original_script.is_null()) { return false; }
	return original_script->get_property_default_value(p_property, r_value);
}

void GDScriptWrapper::update_exports() {
	if (original_script.is_null()) { return; }
	original_script->update_exports();
}

void GDScriptWrapper::get_script_method_list(List<MethodInfo> *p_list) const {
	if (original_script.is_null()) { return; }
	original_script->get_script_method_list(p_list);
}

void GDScriptWrapper::get_script_property_list(List<PropertyInfo> *p_list) const {
	if (original_script.is_null()) { return; }
	original_script->get_script_property_list(p_list);
}

int GDScriptWrapper::get_member_line(const StringName &p_member) const {
	if (original_script.is_null()) { return -1; }
	return original_script->get_member_line(p_member);
}

void GDScriptWrapper::get_constants(HashMap<StringName, Variant> *p_constants) {
	if (original_script.is_null()) { return; }
	original_script->get_constants(p_constants);
}

void GDScriptWrapper::get_members(HashSet<StringName> *p_members) {
	if (original_script.is_null()) { return; }
	original_script->get_members(p_members);
}

bool GDScriptWrapper::is_placeholder_fallback_enabled() const {
	if (original_script.is_null()) { return false; }
	return original_script->is_placeholder_fallback_enabled();
}

const Variant GDScriptWrapper::get_rpc_config() const {
	if (original_script.is_null()) { return Variant(); }
	return original_script->get_rpc_config();
}

#ifdef TOOLS_ENABLED
StringName GDScriptWrapper::get_doc_class_name() const {
	if (original_script.is_null()) { return StringName(); }
	return original_script->get_doc_class_name();
}

Vector<DocData::ClassDoc> GDScriptWrapper::get_documentation() const {
	if (original_script.is_null()) { return Vector<DocData::ClassDoc>(); }
	return original_script->get_documentation();
}

String GDScriptWrapper::get_class_icon_path() const {
	if (original_script.is_null()) { return String(); }
	return original_script->get_class_icon_path();
}
#endif
