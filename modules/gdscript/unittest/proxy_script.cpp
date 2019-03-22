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

bool ProxyScript::ProxyScript::editor_can_reload_from_file() {
    return false;
}
void ProxyScript::_bind_methods() {
}

void ProxyScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
}

bool ProxyScript::can_instance() const {
    return true;
}

Ref<Script> ProxyScript::get_base_script() const {
    return false;
}

StringName ProxyScript::get_instance_base_type() const {
    return "";
}
ScriptInstance *ProxyScript::instance_create(Object *p_this) {
    return false;
}
PlaceHolderScriptInstance *ProxyScript::placeholder_instance_create(Object *p_this) {
    return NULL;
}
bool ProxyScript::instance_has(const Object *p_this) const {
    return false;
}

bool ProxyScript::has_source_code() const { 
return false;
}
String ProxyScript::get_source_code() const {
    return "";
}
void ProxyScript::set_source_code(const String &p_code) {
}
Error ProxyScript::reload(bool p_keep_state) {
    return Error::OK;
}

bool ProxyScript::has_method(const StringName &p_method) const {
    return false;
}
MethodInfo ProxyScript::get_method_info(const StringName &p_method) const {
    MethodInfo mi;
    return mi;
}

bool ProxyScript::is_tool() const  {
    return false;
}
bool ProxyScript::is_valid() const {
    return false;
}

ScriptLanguage *ProxyScript::get_language() const {
    return NULL;
}

bool ProxyScript::has_script_signal(const StringName &p_signal) const {
    return false;
}
void ProxyScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
}

bool ProxyScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
    return false;
}

void ProxyScript::update_exports() {
}
void ProxyScript::get_script_method_list(List<MethodInfo> *p_list) const {
}
void ProxyScript::get_script_property_list(List<PropertyInfo> *p_list) const {
}

int ProxyScript::get_member_line(const StringName &p_member) const {
    return -1;
}

void ProxyScript::get_constants(Map<StringName, Variant> *p_constants) {
}
void ProxyScript::get_members(Set<StringName> *p_constants) {}

bool ProxyScript::is_placeholder_fallback_enabled() const {
    return false;
}

bool ProxyScriptInstance::set(const StringName &p_name, const Variant &p_value) {
    return false;
}
bool ProxyScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
    return false;
}
void ProxyScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
}
Variant::Type ProxyScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
    return Variant::Type::NIL;
}

Object *ProxyScriptInstance::get_owner() {
    return NULL;
}
void ProxyScriptInstance::get_property_state(List<Pair<StringName, Variant> > &state) {
}

void ProxyScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
}
bool ProxyScriptInstance::has_method(const StringName &p_method) const {
    return false;
}
Variant ProxyScriptInstance::call(const StringName &p_method, VARIANT_ARG_DECLARE) {
    return NULL;
}
Variant ProxyScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
    return NULL;
}
void ProxyScriptInstance::call_multilevel(const StringName &p_method, VARIANT_ARG_DECLARE) {
}
void ProxyScriptInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
}
void ProxyScriptInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
}
void ProxyScriptInstance::notification(int p_notification) {
}

void ProxyScriptInstance::refcount_incremented() {
}
bool ProxyScriptInstance::refcount_decremented() {
    return true;
}

Ref<Script> ProxyScriptInstance::get_script() const {
    return NULL;
}

bool ProxyScriptInstance::is_placeholder() const {
    return false;
}

void ProxyScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
}
Variant ProxyScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
    return NULL;
}

MultiplayerAPI::RPCMode ProxyScriptInstance::get_rpc_mode(const StringName &p_method) const {
    MultiplayerAPI::RPCMode mode;
    return mode;
}
MultiplayerAPI::RPCMode ProxyScriptInstance::get_rset_mode(const StringName &p_variable) const {
    MultiplayerAPI::RPCMode mode;
    return mode;
}

ScriptLanguage *ProxyScriptInstance::get_language() {
    return NULL;
}
ProxyScriptInstance::~ProxyScriptInstance() {
}
