/**************************************************************************/
/*  script_language.cpp                                                   */
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

#include "script_language.h"

#include "core/core_string_names.h"
#include "core/project_settings.h"

ScriptLanguage *ScriptServer::_languages[MAX_LANGUAGES];
int ScriptServer::_language_count = 0;

bool ScriptServer::scripting_enabled = true;
bool ScriptServer::reload_scripts_on_save = false;
bool ScriptServer::languages_finished = false;
ScriptEditRequestFunction ScriptServer::edit_request_func = nullptr;

void Script::_notification(int p_what) {
	if (p_what == NOTIFICATION_POSTINITIALIZE) {
		if (ScriptDebugger::get_singleton()) {
			ScriptDebugger::get_singleton()->set_break_language(get_language());
		}
	}
}

Variant Script::_get_property_default_value(const StringName &p_property) {
	Variant ret;
	get_property_default_value(p_property, ret);
	return ret;
}

Array Script::_get_script_property_list() {
	Array ret;
	List<PropertyInfo> list;
	get_script_property_list(&list);
	for (List<PropertyInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.append(E->get().operator Dictionary());
	}
	return ret;
}

Array Script::_get_script_method_list() {
	Array ret;
	List<MethodInfo> list;
	get_script_method_list(&list);
	for (List<MethodInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.append(E->get().operator Dictionary());
	}
	return ret;
}

Array Script::_get_script_signal_list() {
	Array ret;
	List<MethodInfo> list;
	get_script_signal_list(&list);
	for (List<MethodInfo>::Element *E = list.front(); E; E = E->next()) {
		ret.append(E->get().operator Dictionary());
	}
	return ret;
}

Dictionary Script::_get_script_constant_map() {
	Dictionary ret;
	Map<StringName, Variant> map;
	get_constants(&map);
	for (Map<StringName, Variant>::Element *E = map.front(); E; E = E->next()) {
		ret[E->key()] = E->value();
	}
	return ret;
}

void Script::_bind_methods() {
	ClassDB::bind_method(D_METHOD("can_instance"), &Script::can_instance);
	//ClassDB::bind_method(D_METHOD("instance_create","base_object"),&Script::instance_create);
	ClassDB::bind_method(D_METHOD("instance_has", "base_object"), &Script::instance_has);
	ClassDB::bind_method(D_METHOD("has_source_code"), &Script::has_source_code);
	ClassDB::bind_method(D_METHOD("get_source_code"), &Script::get_source_code);
	ClassDB::bind_method(D_METHOD("set_source_code", "source"), &Script::set_source_code);
	ClassDB::bind_method(D_METHOD("reload", "keep_state"), &Script::reload, DEFVAL(false));
	ClassDB::bind_method(D_METHOD("get_base_script"), &Script::get_base_script);
	ClassDB::bind_method(D_METHOD("get_instance_base_type"), &Script::get_instance_base_type);

	ClassDB::bind_method(D_METHOD("has_script_signal", "signal_name"), &Script::has_script_signal);

	ClassDB::bind_method(D_METHOD("get_script_property_list"), &Script::_get_script_property_list);
	ClassDB::bind_method(D_METHOD("get_script_method_list"), &Script::_get_script_method_list);
	ClassDB::bind_method(D_METHOD("get_script_signal_list"), &Script::_get_script_signal_list);
	ClassDB::bind_method(D_METHOD("get_script_constant_map"), &Script::_get_script_constant_map);
	ClassDB::bind_method(D_METHOD("get_property_default_value", "property"), &Script::_get_property_default_value);

	ClassDB::bind_method(D_METHOD("is_tool"), &Script::is_tool);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "source_code", PROPERTY_HINT_NONE, "", 0), "set_source_code", "get_source_code");
}

void ScriptServer::set_scripting_enabled(bool p_enabled) {
	scripting_enabled = p_enabled;
}

bool ScriptServer::is_scripting_enabled() {
	return scripting_enabled;
}

ScriptLanguage *ScriptServer::get_language(int p_idx) {
	ERR_FAIL_INDEX_V(p_idx, _language_count, nullptr);

	return _languages[p_idx];
}

void ScriptServer::register_language(ScriptLanguage *p_language) {
	ERR_FAIL_COND(_language_count >= MAX_LANGUAGES);
	_languages[_language_count++] = p_language;
}

void ScriptServer::unregister_language(ScriptLanguage *p_language) {
	for (int i = 0; i < _language_count; i++) {
		if (_languages[i] == p_language) {
			_language_count--;
			if (i < _language_count) {
				SWAP(_languages[i], _languages[_language_count]);
			}
			return;
		}
	}
}

void ScriptServer::init_languages() {
	{ //load global classes
		global_classes_clear();
		if (ProjectSettings::get_singleton()->has_setting("_global_script_classes")) {
			Array script_classes = ProjectSettings::get_singleton()->get("_global_script_classes");

			for (int i = 0; i < script_classes.size(); i++) {
				Dictionary c = script_classes[i];
				if (!c.has("class") || !c.has("language") || !c.has("path") || !c.has("base")) {
					continue;
				}
				add_global_class(c["class"], c["base"], c["language"], c["path"]);
			}
		}
	}

	for (int i = 0; i < _language_count; i++) {
		_languages[i]->init();
	}
}

void ScriptServer::finish_languages() {
	for (int i = 0; i < _language_count; i++) {
		_languages[i]->finish();
	}
	global_classes_clear();
	languages_finished = true;
}

void ScriptServer::set_reload_scripts_on_save(bool p_enable) {
	reload_scripts_on_save = p_enable;
}

bool ScriptServer::is_reload_scripts_on_save_enabled() {
	return reload_scripts_on_save;
}

void ScriptServer::thread_enter() {
	for (int i = 0; i < _language_count; i++) {
		_languages[i]->thread_enter();
	}
}

void ScriptServer::thread_exit() {
	for (int i = 0; i < _language_count; i++) {
		_languages[i]->thread_exit();
	}
}

HashMap<StringName, ScriptServer::GlobalScriptClass> ScriptServer::global_classes;

void ScriptServer::global_classes_clear() {
	global_classes.clear();
}

void ScriptServer::add_global_class(const StringName &p_class, const StringName &p_base, const StringName &p_language, const String &p_path) {
	ERR_FAIL_COND_MSG(p_class == p_base || (global_classes.has(p_base) && get_global_class_native_base(p_base) == p_class), "Cyclic inheritance in script class.");
	GlobalScriptClass g;
	g.language = p_language;
	g.path = p_path;
	g.base = p_base;
	global_classes[p_class] = g;
}
void ScriptServer::remove_global_class(const StringName &p_class) {
	global_classes.erase(p_class);
}
bool ScriptServer::is_global_class(const StringName &p_class) {
	return global_classes.has(p_class);
}
StringName ScriptServer::get_global_class_language(const StringName &p_class) {
	ERR_FAIL_COND_V(!global_classes.has(p_class), StringName());
	return global_classes[p_class].language;
}
String ScriptServer::get_global_class_path(const String &p_class) {
	ERR_FAIL_COND_V(!global_classes.has(p_class), String());
	return global_classes[p_class].path;
}

StringName ScriptServer::get_global_class_base(const String &p_class) {
	ERR_FAIL_COND_V(!global_classes.has(p_class), String());
	return global_classes[p_class].base;
}
StringName ScriptServer::get_global_class_native_base(const String &p_class) {
	ERR_FAIL_COND_V(!global_classes.has(p_class), String());
	String base = global_classes[p_class].base;
	while (global_classes.has(base)) {
		base = global_classes[base].base;
	}
	return base;
}
void ScriptServer::get_global_class_list(List<StringName> *r_global_classes) {
	const StringName *K = nullptr;
	List<StringName> classes;
	while ((K = global_classes.next(K))) {
		classes.push_back(*K);
	}
	classes.sort_custom<StringName::AlphCompare>();
	for (List<StringName>::Element *E = classes.front(); E; E = E->next()) {
		r_global_classes->push_back(E->get());
	}
}
void ScriptServer::save_global_classes() {
	List<StringName> gc;
	get_global_class_list(&gc);
	Array gcarr;
	for (List<StringName>::Element *E = gc.front(); E; E = E->next()) {
		Dictionary d;
		d["class"] = E->get();
		d["language"] = global_classes[E->get()].language;
		d["path"] = global_classes[E->get()].path;
		d["base"] = global_classes[E->get()].base;
		gcarr.push_back(d);
	}

	Array old;
	if (ProjectSettings::get_singleton()->has_setting("_global_script_classes")) {
		old = ProjectSettings::get_singleton()->get("_global_script_classes");
	}
	if ((!old.empty() || gcarr.empty()) && gcarr.hash() == old.hash()) {
		return;
	}

	if (gcarr.empty()) {
		if (ProjectSettings::get_singleton()->has_setting("_global_script_classes")) {
			ProjectSettings::get_singleton()->clear("_global_script_classes");
		}
	} else {
		ProjectSettings::get_singleton()->set("_global_script_classes", gcarr);
	}
	ProjectSettings::get_singleton()->save();
}

////////////////////
void ScriptInstance::get_property_state(List<Pair<StringName, Variant>> &state) {
	List<PropertyInfo> pinfo;
	get_property_list(&pinfo);
	for (List<PropertyInfo>::Element *E = pinfo.front(); E; E = E->next()) {
		if (E->get().usage & PROPERTY_USAGE_STORAGE) {
			Pair<StringName, Variant> p;
			p.first = E->get().name;
			if (get(p.first, p.second)) {
				state.push_back(p);
			}
		}
	}
}

Variant ScriptInstance::call(const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;
	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	Variant::CallError error;
	return call(p_method, argptr, argc, error);
}

void ScriptInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
	Variant::CallError ce;
	call(p_method, p_args, p_argcount, ce); // script may not support multilevel calls
}

void ScriptInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
	Variant::CallError ce;
	call(p_method, p_args, p_argcount, ce); // script may not support multilevel calls
}

void ScriptInstance::property_set_fallback(const StringName &, const Variant &, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}
}

Variant ScriptInstance::property_get_fallback(const StringName &, bool *r_valid) {
	if (r_valid) {
		*r_valid = false;
	}
	return Variant();
}

void ScriptInstance::call_multilevel(const StringName &p_method, VARIANT_ARG_DECLARE) {
	VARIANT_ARGPTRS;
	int argc = 0;
	for (int i = 0; i < VARIANT_ARG_MAX; i++) {
		if (argptr[i]->get_type() == Variant::NIL) {
			break;
		}
		argc++;
	}

	call_multilevel(p_method, argptr, argc);
}

ScriptInstance::~ScriptInstance() {
}

ScriptCodeCompletionCache *ScriptCodeCompletionCache::singleton = nullptr;
ScriptCodeCompletionCache::ScriptCodeCompletionCache() {
	singleton = this;
}

void ScriptLanguage::frame() {
}

ScriptDebugger *ScriptDebugger::singleton = nullptr;

void ScriptDebugger::set_lines_left(int p_left) {
	lines_left = p_left;
}

int ScriptDebugger::get_lines_left() const {
	return lines_left;
}

void ScriptDebugger::set_depth(int p_depth) {
	depth = p_depth;
}

int ScriptDebugger::get_depth() const {
	return depth;
}

void ScriptDebugger::insert_breakpoint(int p_line, const StringName &p_source) {
	if (!breakpoints.has(p_line)) {
		breakpoints[p_line] = Set<StringName>();
	}
	breakpoints[p_line].insert(p_source);
}

void ScriptDebugger::remove_breakpoint(int p_line, const StringName &p_source) {
	if (!breakpoints.has(p_line)) {
		return;
	}

	breakpoints[p_line].erase(p_source);
	if (breakpoints[p_line].size() == 0) {
		breakpoints.erase(p_line);
	}
}
bool ScriptDebugger::is_breakpoint(int p_line, const StringName &p_source) const {
	if (!breakpoints.has(p_line)) {
		return false;
	}
	return breakpoints[p_line].has(p_source);
}
bool ScriptDebugger::is_breakpoint_line(int p_line) const {
	return breakpoints.has(p_line);
}

String ScriptDebugger::breakpoint_find_source(const String &p_source) const {
	return p_source;
}

void ScriptDebugger::clear_breakpoints() {
	breakpoints.clear();
}

void ScriptDebugger::idle_poll() {
}

void ScriptDebugger::line_poll() {
}

void ScriptDebugger::set_break_language(ScriptLanguage *p_lang) {
	break_lang = p_lang;
}

ScriptLanguage *ScriptDebugger::get_break_language() const {
	return break_lang;
}

ScriptDebugger::ScriptDebugger() {
	singleton = this;
	lines_left = -1;
	depth = -1;
	break_lang = nullptr;
}

bool PlaceHolderScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	if (script->is_placeholder_fallback_enabled()) {
		return false;
	}

	if (values.has(p_name)) {
		Variant defval;
		if (script->get_property_default_value(p_name, defval)) {
			if (defval == p_value) {
				values.erase(p_name);
				return true;
			}
		}
		values[p_name] = p_value;
		return true;
	} else {
		Variant defval;
		if (script->get_property_default_value(p_name, defval)) {
			if (defval != p_value) {
				values[p_name] = p_value;
			}
			return true;
		}
	}
	return false;
}
bool PlaceHolderScriptInstance::get(const StringName &p_name, Variant &r_ret) const {
	if (values.has(p_name)) {
		r_ret = values[p_name];
		return true;
	}

	if (constants.has(p_name)) {
		r_ret = constants[p_name];
		return true;
	}

	if (!script->is_placeholder_fallback_enabled()) {
		Variant defval;
		if (script->get_property_default_value(p_name, defval)) {
			r_ret = defval;
			return true;
		}
	}

	return false;
}

void PlaceHolderScriptInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	if (script->is_placeholder_fallback_enabled()) {
		for (const List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			p_properties->push_back(E->get());
		}
	} else {
		for (const List<PropertyInfo>::Element *E = properties.front(); E; E = E->next()) {
			PropertyInfo pinfo = E->get();
			if (!values.has(pinfo.name)) {
				pinfo.usage |= PROPERTY_USAGE_SCRIPT_DEFAULT_VALUE;
			}
			p_properties->push_back(E->get());
		}
	}
}

Variant::Type PlaceHolderScriptInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if (values.has(p_name)) {
		if (r_is_valid) {
			*r_is_valid = true;
		}
		return values[p_name].get_type();
	}

	if (constants.has(p_name)) {
		if (r_is_valid) {
			*r_is_valid = true;
		}
		return constants[p_name].get_type();
	}

	if (r_is_valid) {
		*r_is_valid = false;
	}

	return Variant::NIL;
}

void PlaceHolderScriptInstance::get_method_list(List<MethodInfo> *p_list) const {
	if (script->is_placeholder_fallback_enabled()) {
		return;
	}

	if (script.is_valid()) {
		script->get_script_method_list(p_list);
	}
}
bool PlaceHolderScriptInstance::has_method(const StringName &p_method) const {
	if (script->is_placeholder_fallback_enabled()) {
		return false;
	}

	if (script.is_valid()) {
		return script->has_method(p_method);
	}
	return false;
}

void PlaceHolderScriptInstance::update(const List<PropertyInfo> &p_properties, const Map<StringName, Variant> &p_values) {
	Set<StringName> new_values;
	for (const List<PropertyInfo>::Element *E = p_properties.front(); E; E = E->next()) {
		StringName n = E->get().name;
		new_values.insert(n);

		if (!values.has(n) || values[n].get_type() != E->get().type) {
			if (p_values.has(n)) {
				values[n] = p_values[n];
			}
		}
	}

	properties = p_properties;
	List<StringName> to_remove;

	for (Map<StringName, Variant>::Element *E = values.front(); E; E = E->next()) {
		if (!new_values.has(E->key())) {
			to_remove.push_back(E->key());
		}

		Variant defval;
		if (script->get_property_default_value(E->key(), defval)) {
			//remove because it's the same as the default value
			if (defval == E->get()) {
				to_remove.push_back(E->key());
			}
		}
	}

	while (to_remove.size()) {
		values.erase(to_remove.front()->get());
		to_remove.pop_front();
	}

	if (owner && owner->get_script_instance() == this) {
		owner->_change_notify();
	}
	//change notify

	constants.clear();
	script->get_constants(&constants);
}

void PlaceHolderScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	if (script->is_placeholder_fallback_enabled()) {
		Map<StringName, Variant>::Element *E = values.find(p_name);

		if (E) {
			E->value() = p_value;
		} else {
			values.insert(p_name, p_value);
		}

		bool found = false;
		for (const List<PropertyInfo>::Element *F = properties.front(); F; F = F->next()) {
			if (F->get().name == p_name) {
				found = true;
				break;
			}
		}
		if (!found) {
			properties.push_back(PropertyInfo(p_value.get_type(), p_name, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR | PROPERTY_USAGE_SCRIPT_VARIABLE));
		}
	}

	if (r_valid) {
		*r_valid = false; // Cannot change the value in either case
	}
}

Variant PlaceHolderScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
	if (script->is_placeholder_fallback_enabled()) {
		const Map<StringName, Variant>::Element *E = values.find(p_name);

		if (E) {
			if (r_valid) {
				*r_valid = true;
			}
			return E->value();
		}

		E = constants.find(p_name);
		if (E) {
			if (r_valid) {
				*r_valid = true;
			}
			return E->value();
		}
	}

	if (r_valid) {
		*r_valid = false;
	}

	return Variant();
}

PlaceHolderScriptInstance::PlaceHolderScriptInstance(ScriptLanguage *p_language, Ref<Script> p_script, Object *p_owner) :
		owner(p_owner),
		language(p_language),
		script(p_script) {
}

PlaceHolderScriptInstance::~PlaceHolderScriptInstance() {
	if (script.is_valid()) {
		script->_placeholder_erased(this);
	}
}
