/*************************************************************************/
/*  proxy_script.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "proxy_script.h"

ProxyScript::ProxyScript(RefPtr script) {
	m_script = script;
}

bool ProxyScript::ProxyScript::editor_can_reload_from_file() {
	return false;
}

void ProxyScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
	}
}

bool ProxyScript::can_instance() const {
	return true;
}

Ref<Script> ProxyScript::get_base_script() const {
	return m_script;
}

StringName ProxyScript::get_instance_base_type() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_instance_base_type();
	}
	return "ProxyScript";
}
ScriptInstance *ProxyScript::instance_create(Object *p_this) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return memnew(ProxyScriptInstance(this, script->instance_create(p_this)));
	} else {
		return memnew(ProxyScriptInstance(this, NULL));
	}
	return NULL;
}
PlaceHolderScriptInstance *ProxyScript::placeholder_instance_create(Object *p_this) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->placeholder_instance_create(p_this);
	}
	return NULL;
}
bool ProxyScript::instance_has(const Object *p_this) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->instance_has(p_this);
	}
	return false;
}

bool ProxyScript::has_source_code() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->has_source_code();
	}
	return false;
}
String ProxyScript::get_source_code() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_source_code();
	}
	return "";
}
void ProxyScript::set_source_code(const String &p_code) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->set_source_code(p_code);
	}
}
Error ProxyScript::reload(bool p_keep_state) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->reload(p_keep_state);
	}
	return ::Error::OK;
}

bool ProxyScript::has_method(const StringName &p_method) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->has_method(p_method);
	}
	return false;
}
MethodInfo ProxyScript::get_method_info(const StringName &p_method) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_method_info(p_method);
	}
	MethodInfo mi;
	return mi;
}

bool ProxyScript::is_tool() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->is_tool();
	}
	return false;
}
bool ProxyScript::is_valid() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->is_valid();
	}
	return true;
}

ScriptLanguage *ProxyScript::get_language() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_language();
	}
	return NULL;
}

bool ProxyScript::has_script_signal(const StringName &p_signal) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->has_script_signal(p_signal);
	}
	return false;
}
void ProxyScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_script_signal_list(r_signals);
	}
}

bool ProxyScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_property_default_value(p_property, r_value);
	}
	return false;
}

void ProxyScript::update_exports() {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->update_exports();
	}
}
void ProxyScript::get_script_method_list(List<MethodInfo> *p_list) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_script_method_list(p_list);
	}
}
void ProxyScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_script_property_list(p_list);
	}
}

int ProxyScript::get_member_line(const StringName &p_member) const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_member_line(p_member);
	}
	return -1;
}

void ProxyScript::get_constants(Map<StringName, Variant> *p_constants) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_constants(p_constants);
	}
}
void ProxyScript::get_members(Set<StringName> *p_constants) {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->get_members(p_constants);
	}
}

bool ProxyScript::is_placeholder_fallback_enabled() const {
	Ref<Script> script = m_script;
	if (script.is_valid()) {
		return script->is_placeholder_fallback_enabled();
	}
	return false;
}

Variant ProxyScriptInstance::_bind_method(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (p_argcount < 2) {
		r_error.argument = p_argcount;
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		return Variant();
	}
	if (p_argcount > 2) {
		r_error.argument = p_argcount;
		r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		return Variant();
	}
	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		return Variant();
	}
	const String &name = *p_args[0];
	bind_method(name, *p_args[1]);
	return Variant();
}
Variant ProxyScriptInstance::_add_property(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (p_argcount < 3) {
		r_error.argument = p_argcount;
		r_error.error = Variant::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
		return Variant();
	}
	if (p_argcount > 3) {
		r_error.argument = p_argcount;
		r_error.error = Variant::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
		return Variant();
	}
	if (p_args[0]->get_type() != Variant::STRING) {
		r_error.argument = 0;
		r_error.expected = Variant::STRING;
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		return Variant();
	}
	if (p_args[1]->get_type() != Variant::STRING) {
		r_error.argument = 1;
		r_error.expected = Variant::STRING;
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		return Variant();
	}
	if (p_args[2]->get_type() != Variant::STRING) {
		r_error.argument = 2;
		r_error.expected = Variant::STRING;
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_ARGUMENT;
		return Variant();
	}
	const String &name = *p_args[0];
	const String &setter = *p_args[1];
	const String &getter = *p_args[2];
	add_property(name, setter, getter);
	return Variant();
}

void ProxyScriptInstance::bind_method(const String &p_name, const Variant &p_return) {
	m_method_watcher.bind_method(p_name, p_return);
}
void ProxyScriptInstance::add_property(const String &p_name, const StringName p_setter, const StringName p_getter) {
	m_method_watcher.add_property(p_name, p_setter, p_getter);
}

ProxyScriptInstance::ProxyScriptInstance(Ref<ProxyScript> script, ScriptInstance *script_instance) {
	m_script = script;
	m_script_instance = script_instance;
}

bool ProxyScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	bool valid = false;
	m_method_watcher.set(p_name, p_value, &valid);
	if (valid) {
		return true;
	}
	if (m_script_instance != NULL) {
		return m_script_instance->set(p_name, p_value);
	}
	return false;
}
bool ProxyScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	bool valid = false;
	r_ret = m_method_watcher.get(p_name, &valid);
	if (valid) {
		return true;
	}
	if (m_script_instance != NULL) {
		return m_script_instance->get(p_name, r_ret);
	}
	return false;
}
void ProxyScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	if (m_script_instance != NULL) {
		return m_script_instance->get_property_list(p_properties);
	}
}
Variant::Type ProxyScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if (m_script_instance != NULL) {
		return m_script_instance->get_property_type(p_name, r_is_valid);
	}
	return ::Variant::Type::NIL;
}

Object *ProxyScriptInstance::get_owner() {
	if (m_script_instance != NULL) {
		return m_script_instance->get_owner();
	}
	return NULL;
}
void ProxyScriptInstance::get_property_state(List<Pair<StringName, Variant> > &state) {
	if (m_script_instance != NULL) {
		return m_script_instance->get_property_state(state);
	}
}

void ProxyScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	if (m_script_instance != NULL) {
		return m_script_instance->get_method_list(p_list);
	}
}
bool ProxyScriptInstance::has_method(const StringName &p_method) const {
	if (m_method_watcher.has_method(p_method)) {
		return true;
	}
	if (m_script_instance != NULL) {
		return m_script_instance->has_method(p_method);
	}
	return false;
}
Variant ProxyScriptInstance::call(const StringName &p_method, VARIANT_ARG_DECLARE) {
	if (m_script_instance != NULL) {
		return m_script_instance->call(p_method, VARIANT_ARG_PASS);
	}
	return Variant();
}
Variant ProxyScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (p_method == "bind_method") {
		return _bind_method(p_args, p_argcount, r_error);
	}
	if (p_method == "add_property") {
		return _add_property(p_args, p_argcount, r_error);
	}
	Variant result = m_method_watcher.call(p_method, p_args, p_argcount, r_error);
	if (r_error.error == Variant::CallError::CALL_OK) {
		return result;
	}
	if (m_script_instance != NULL) {
		return m_script_instance->call(p_method, p_args, p_argcount, r_error);
	}
	return Variant();
}
void ProxyScriptInstance::call_multilevel(const StringName &p_method, VARIANT_ARG_DECLARE) {
	if (m_script_instance != NULL) {
		m_script_instance->call_multilevel(p_method, VARIANT_ARG_PASS);
	}
}
void ProxyScriptInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
	Variant::CallError ce;
	m_method_watcher.call(p_method, p_args, p_argcount, ce);
	if (ce.error != Variant::CallError::CALL_OK && m_script_instance != NULL) {
		m_script_instance->call_multilevel(p_method, p_args, p_argcount);
	}
}
void ProxyScriptInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
	Variant::CallError ce;
	m_method_watcher.call(p_method, p_args, p_argcount, ce);
	if (ce.error != Variant::CallError::CALL_OK && m_script_instance != NULL) {
		return m_script_instance->call_multilevel_reversed(p_method, p_args, p_argcount);
	}
}
void ProxyScriptInstance::notification(int p_notification) {
	if (m_script_instance != NULL) {
		m_script_instance->notification(p_notification);
	}
}

void ProxyScriptInstance::refcount_incremented() {
	if (m_script_instance != NULL) {
		m_script_instance->refcount_incremented();
	}
}
bool ProxyScriptInstance::refcount_decremented() {
	if (m_script_instance != NULL) {
		return m_script_instance->refcount_decremented();
	}
	return false;
}

Ref<Script> ProxyScriptInstance::get_script() const {
	return m_script;
}

bool ProxyScriptInstance::is_placeholder() const {
	if (m_script_instance != NULL) {
		return m_script_instance->is_placeholder();
	}
	return false;
}

void ProxyScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	if (m_script_instance != NULL) {
		m_script_instance->property_set_fallback(p_name, p_value, r_valid);
	}
}
Variant ProxyScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
	if (m_script_instance != NULL) {
		return m_script_instance->property_get_fallback(p_name, r_valid);
	}
	return Variant();
}

MultiplayerAPI::RPCMode ProxyScriptInstance::get_rpc_mode(const StringName &p_method) const {
	if (m_script_instance != NULL) {
		return m_script_instance->get_rpc_mode(p_method);
	}
	return MultiplayerAPI::RPC_MODE_DISABLED;
}
MultiplayerAPI::RPCMode ProxyScriptInstance::get_rset_mode(const StringName &p_variable) const {
	if (m_script_instance != NULL) {
		return m_script_instance->get_rset_mode(p_variable);
	}
	return MultiplayerAPI::RPC_MODE_DISABLED;
}

ScriptLanguage *ProxyScriptInstance::get_language() {
	if (m_script_instance != NULL) {
		return m_script_instance->get_language();
	}
	return NULL;
}
ProxyScriptInstance::~ProxyScriptInstance() {
	if (m_script_instance) {
		memfree(m_script_instance);
		m_script_instance = NULL;
	}
}

const Vector<MethodWatcher::Args> ProxyScriptInstance::get_calls(const String &p_name) const {
	return m_method_watcher.get_calls(p_name);
}
