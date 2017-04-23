/*************************************************************************/
/*  multiscript.cpp                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "multiscript.h"

bool MultiScriptInstance::set(const StringName &p_name, const Variant &p_value) {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		bool found = sarr[i]->set(p_name, p_value);
		if (found)
			return true;
	}

	if (String(p_name).begins_with("script_")) {
		bool valid;
		owner->set(p_name, p_value, &valid);
		return valid;
	}
	return false;
}

bool MultiScriptInstance::get(const StringName &p_name, Variant &r_ret) const {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		bool found = sarr[i]->get(p_name, r_ret);
		if (found)
			return true;
	}
	if (String(p_name).begins_with("script_")) {
		bool valid;
		r_ret = owner->get(p_name, &valid);
		return valid;
	}
	return false;
}
void MultiScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	Set<String> existing;

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		List<PropertyInfo> pl;
		sarr[i]->get_property_list(&pl);

		for (List<PropertyInfo>::Element *E = pl.front(); E; E = E->next()) {

			if (existing.has(E->get().name))
				continue;

			p_properties->push_back(E->get());
			existing.insert(E->get().name);
		}
	}

	p_properties->push_back(PropertyInfo(Variant::NIL, "Scripts", PROPERTY_HINT_NONE, String(), PROPERTY_USAGE_CATEGORY));

	for (int i = 0; i < owner->scripts.size(); i++) {

		p_properties->push_back(PropertyInfo(Variant::OBJECT, "script_" + String::chr('a' + i), PROPERTY_HINT_RESOURCE_TYPE, "Script", PROPERTY_USAGE_EDITOR));
	}

	if (owner->scripts.size() < 25) {

		p_properties->push_back(PropertyInfo(Variant::OBJECT, "script_" + String::chr('a' + (owner->scripts.size())), PROPERTY_HINT_RESOURCE_TYPE, "Script", PROPERTY_USAGE_EDITOR));
	}
}

void MultiScriptInstance::get_method_list(List<MethodInfo> *p_list) const {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	Set<StringName> existing;

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		List<MethodInfo> ml;
		sarr[i]->get_method_list(&ml);

		for (List<MethodInfo>::Element *E = ml.front(); E; E = E->next()) {

			if (existing.has(E->get().name))
				continue;

			p_list->push_back(E->get());
			existing.insert(E->get().name);
		}
	}
}
bool MultiScriptInstance::has_method(const StringName &p_method) const {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		if (sarr[i]->has_method(p_method))
			return true;
	}

	return false;
}

Variant MultiScriptInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		Variant r = sarr[i]->call(p_method, p_args, p_argcount, r_error);
		if (r_error.error == Variant::CallError::CALL_OK)
			return r;
		else if (r_error.error != Variant::CallError::CALL_ERROR_INVALID_METHOD)
			return r;
	}

	r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void MultiScriptInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		sarr[i]->call_multilevel(p_method, p_args, p_argcount);
	}
}
void MultiScriptInstance::notification(int p_notification) {

	// ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		ScriptInstance *instance = instances[i];

		if (!instance)
			continue;

		instance->notification(p_notification);
	}
}

Ref<Script> MultiScriptInstance::get_script() const {

	return owner;
}

ScriptLanguage *MultiScriptInstance::get_language() {

	return MultiScriptLanguage::get_singleton();
}

MultiScriptInstance::~MultiScriptInstance() {

	owner->remove_instance(object);
}

Variant::Type MultiScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	bool valid = false;
	Variant::Type type;

	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		type = sarr[i]->get_property_type(p_name, &valid);
		if (valid) {
			*r_is_valid = valid;
			return type;
		}
	}
	*r_is_valid = false;
	return Variant::NIL;
}

ScriptInstance::RPCMode MultiScriptInstance::get_rpc_mode(const StringName &p_method) const {
	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;
		if (sarr[i]->has_method(p_method))
			return sarr[i]->get_rpc_mode(p_method);
	}
	return RPC_MODE_DISABLED;
}

ScriptInstance::RPCMode MultiScriptInstance::get_rset_mode(const StringName &p_variable) const {
	ScriptInstance **sarr = instances.ptr();
	int sc = instances.size();

	for (int i = 0; i < sc; i++) {

		if (!sarr[i])
			continue;

		List<PropertyInfo> properties;
		sarr[i]->get_property_list(&properties);

		for (List<PropertyInfo>::Element *P = properties.front(); P; P = P->next()) {
			if (P->get().name == p_variable) {
				return sarr[i]->get_rset_mode(p_variable);
			}
		}
	}
	return RPC_MODE_DISABLED;
}

///////////////////

bool MultiScript::is_tool() const {

	for (int i = 0; i < scripts.size(); i++) {

		if (scripts[i]->is_tool())
			return true;
	}

	return false;
}

bool MultiScript::_set(const StringName &p_name, const Variant &p_value) {

	_THREAD_SAFE_METHOD_

	String s = String(p_name);
	if (s.begins_with("script_")) {

		int idx = s[7];
		if (idx == 0)
			return false;
		idx -= 'a';

		ERR_FAIL_COND_V(idx < 0, false);

		Ref<Script> s = p_value;

		if (idx < scripts.size()) {

			if (s.is_null())
				remove_script(idx);
			else
				set_script(idx, s);
		} else if (idx == scripts.size()) {
			if (s.is_null())
				return false;
			add_script(s);
		} else
			return false;

		return true;
	}

	return false;
}

bool MultiScript::_get(const StringName &p_name, Variant &r_ret) const {

	_THREAD_SAFE_METHOD_

	String s = String(p_name);
	if (s.begins_with("script_")) {

		int idx = s[7];
		if (idx == 0)
			return false;
		idx -= 'a';

		ERR_FAIL_COND_V(idx < 0, false);

		if (idx < scripts.size()) {

			r_ret = get_script(idx);
			return true;
		} else if (idx == scripts.size()) {
			r_ret = Ref<Script>();
			return true;
		}
	}

	return false;
}
void MultiScript::_get_property_list(List<PropertyInfo> *p_list) const {

	_THREAD_SAFE_METHOD_

	for (int i = 0; i < scripts.size(); i++) {

		p_list->push_back(PropertyInfo(Variant::OBJECT, "script_" + String::chr('a' + i), PROPERTY_HINT_RESOURCE_TYPE, "Script"));
	}

	if (scripts.size() < 25) {

		p_list->push_back(PropertyInfo(Variant::OBJECT, "script_" + String::chr('a' + (scripts.size())), PROPERTY_HINT_RESOURCE_TYPE, "Script"));
	}
}

void MultiScript::set_script(int p_idx, const Ref<Script> &p_script) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_idx, scripts.size());
	ERR_FAIL_COND(p_script.is_null());

	scripts[p_idx] = p_script;
	Ref<Script> s = p_script;

	for (Map<Object *, MultiScriptInstance *>::Element *E = instances.front(); E; E = E->next()) {

		MultiScriptInstance *msi = E->get();
		ScriptInstance *si = msi->instances[p_idx];
		if (si) {
			msi->instances[p_idx] = NULL;
			memdelete(si);
		}

		if (p_script->can_instance())
			msi->instances[p_idx] = s->instance_create(msi->object);
	}
}

Ref<Script> MultiScript::get_script(int p_idx) const {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX_V(p_idx, scripts.size(), Ref<Script>());

	return scripts[p_idx];
}
void MultiScript::add_script(const Ref<Script> &p_script) {

	_THREAD_SAFE_METHOD_
	ERR_FAIL_COND(p_script.is_null());
	Multi *script_owner = memnew(Multi);
	script_instances.push_back(script_owner);
	scripts.push_back(p_script);
	Ref<Script> s = p_script;

	for (Map<Object *, MultiScriptInstance *>::Element *E = instances.front(); E; E = E->next()) {

		MultiScriptInstance *msi = E->get();

		if (p_script->can_instance()) {
			script_owner->real_owner = msi->object;
			msi->instances.push_back(s->instance_create(script_owner));
		} else {
			msi->instances.push_back(NULL);
		}

		msi->object->_change_notify();
	}

	_change_notify();
}

void MultiScript::remove_script(int p_idx) {

	_THREAD_SAFE_METHOD_

	ERR_FAIL_INDEX(p_idx, scripts.size());

	scripts.remove(p_idx);
	script_instances.remove(p_idx);

	for (Map<Object *, MultiScriptInstance *>::Element *E = instances.front(); E; E = E->next()) {

		MultiScriptInstance *msi = E->get();
		ScriptInstance *si = msi->instances[p_idx];
		msi->instances.remove(p_idx);
		if (si) {
			memdelete(si);
		}

		msi->object->_change_notify();
	}
}

void MultiScript::remove_instance(Object *p_object) {

	_THREAD_SAFE_METHOD_
	instances.erase(p_object);
}

bool MultiScript::can_instance() const {

	return true;
}

StringName MultiScript::get_instance_base_type() const {

	return StringName();
}
ScriptInstance *MultiScript::instance_create(Object *p_this) {

	_THREAD_SAFE_METHOD_
	MultiScriptInstance *msi = memnew(MultiScriptInstance);
	msi->object = p_this;
	msi->owner = this;

	for (int i = 0; i < scripts.size(); i++) {

		ScriptInstance *si;

		if (scripts[i]->can_instance()) {
			script_instances[i]->real_owner = p_this;
			si = scripts[i]->instance_create(script_instances[i]);
		} else {
			si = NULL;
		}

		msi->instances.push_back(si);
	}

	instances[p_this] = msi;
	p_this->_change_notify();
	return msi;
}
bool MultiScript::instance_has(const Object *p_this) const {

	_THREAD_SAFE_METHOD_
	return instances.has((Object *)p_this);
}

bool MultiScript::has_source_code() const {

	return false;
}
String MultiScript::get_source_code() const {

	return "";
}
void MultiScript::set_source_code(const String &p_code) {
}
Error MultiScript::reload(bool p_keep_state) {

	for (int i = 0; i < scripts.size(); i++)
		scripts[i]->reload(p_keep_state);

	return OK;
}

String MultiScript::get_node_type() const {

	return "";
}

void MultiScript::_bind_methods() {
}

ScriptLanguage *MultiScript::get_language() const {

	return MultiScriptLanguage::get_singleton();
}

///////////////

MultiScript::MultiScript() {
}

MultiScript::~MultiScript() {
	for (int i = 0; i < script_instances.size(); i++) {
		memdelete(script_instances[i]);
	}

	script_instances.resize(0);
}

Ref<Script> MultiScript::get_base_script() const {
	Ref<MultiScript> base_script;
	return base_script;
}

bool MultiScript::has_method(const StringName &p_method) const {
	for (int i = 0; i < scripts.size(); i++) {
		if (scripts[i]->has_method(p_method)) {
			return true;
		}
	}
	return false;
}

MethodInfo MultiScript::get_method_info(const StringName &p_method) const {
	for (int i = 0; i < scripts.size(); i++) {
		if (scripts[i]->has_method(p_method)) {
			return scripts[i]->get_method_info(p_method);
		}
	}
	return MethodInfo();
}

bool MultiScript::has_script_signal(const StringName &p_signal) const {
	for (int i = 0; i < scripts.size(); i++) {
		if (scripts[i]->has_script_signal(p_signal)) {
			return true;
		}
	}
	return false;
}

void MultiScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	for (int i = 0; i < scripts.size(); i++) {
		scripts[i]->get_script_signal_list(r_signals);
	}
}

bool MultiScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	for (int i = 0; i < scripts.size(); i++) {

		if (scripts[i]->get_property_default_value(p_property, r_value)) {
			return true;
		}
	}
	return false;
}

void MultiScript::get_script_method_list(List<MethodInfo> *p_list) const {
	for (int i = 0; i < scripts.size(); i++) {
		scripts[i]->get_script_method_list(p_list);
	}
}

void MultiScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < scripts.size(); i++) {
		scripts[i]->get_script_property_list(p_list);
	}
}

void MultiScript::update_exports() {
	for (int i = 0; i < scripts.size(); i++) {
		scripts[i]->update_exports();
	}
}

MultiScriptLanguage *MultiScriptLanguage::singleton = NULL;

MultiScriptLanguage *MultiScriptLanguage::get_singleton() {
	return singleton;
}

String MultiScriptLanguage::get_name() const {
	return "MultiScript";
}

void MultiScriptLanguage::init() {}

String MultiScriptLanguage::get_type() const {
	return "MultiScript";
}

String MultiScriptLanguage::get_extension() const {
	return "";
}

Error MultiScriptLanguage::execute_file(const String &p_path) {
	return OK;
}

void MultiScriptLanguage::finish() {}

void MultiScriptLanguage::get_reserved_words(List<String> *p_words) const {}

void MultiScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {}

void MultiScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {}

Ref<Script> MultiScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	MultiScript *s = memnew(MultiScript);
	s->base_class_name = p_base_class_name;
	return Ref<MultiScript>(s);
}

bool MultiScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_fn) const {
	return true;
}

Script *MultiScriptLanguage::create_script() const {
	return memnew(MultiScript);
}

bool MultiScriptLanguage::has_named_classes() const {
	return false;
}

int MultiScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1;
}

String MultiScriptLanguage::make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const {
	return "";
}

String MultiScriptLanguage::debug_get_error() const {
	return "";
}

int MultiScriptLanguage::debug_get_stack_level_count() const {
	return 0;
}

int MultiScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return 0;
}

String MultiScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return "";
}

String MultiScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return "";
}

void MultiScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}

void MultiScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}

void MultiScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}

String MultiScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return "";
}

void MultiScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {}

void MultiScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {}

MultiScriptLanguage::MultiScriptLanguage() {
	singleton = this;
}

MultiScriptLanguage::~MultiScriptLanguage() {}

void MultiScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
}

void MultiScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
}

void MultiScriptLanguage::reload_all_scripts() {
}

void MultiScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
}

void MultiScriptLanguage::get_public_constants(List<Pair<String, Variant> > *p_constants) const {
}

void MultiScriptLanguage::profiling_start() {
}

void MultiScriptLanguage::profiling_stop() {
}

int MultiScriptLanguage::profiling_get_accumulated_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int MultiScriptLanguage::profiling_get_frame_data(ScriptLanguage::ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

void Multi::_bind_methods() {
	// ClassDB::bind_method("call", &Multi::call);
	// ClassDB::bind_method("call_multilevel", &Multi::call_multilevel);
	// ClassDB::bind_method("call_multilevel_reversed", &Multi::call_multilevel_reversed);
}

Variant Multi::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	if (real_owner)
		return real_owner->call(p_method, p_args, p_argcount, r_error);
	return Variant();
}

void Multi::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
	if (real_owner)
		real_owner->call_multilevel(p_method, p_args, p_argcount);
}

void Multi::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
	if (real_owner)
		real_owner->call_multilevel_reversed(p_method, p_args, p_argcount);
}
