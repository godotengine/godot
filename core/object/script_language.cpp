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

#include "core/config/project_settings.h"
#include "core/core_string_names.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"

#include <stdint.h>

ScriptLanguage *ScriptServer::_languages[MAX_LANGUAGES];
int ScriptServer::_language_count = 0;
bool ScriptServer::languages_ready = false;
Mutex ScriptServer::languages_mutex;

bool ScriptServer::scripting_enabled = true;
bool ScriptServer::reload_scripts_on_save = false;
ScriptEditRequestFunction ScriptServer::edit_request_func = nullptr;

void Script::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_POSTINITIALIZE: {
			if (EngineDebugger::is_active()) {
				EngineDebugger::get_script_debugger()->set_break_language(get_language());
			}
		} break;
	}
}

Variant Script::_get_property_default_value(const StringName &p_property) {
	Variant ret;
	get_property_default_value(p_property, ret);
	return ret;
}

TypedArray<Dictionary> Script::_get_script_property_list() {
	TypedArray<Dictionary> ret;
	List<PropertyInfo> list;
	get_script_property_list(&list);
	for (const PropertyInfo &E : list) {
		ret.append(E.operator Dictionary());
	}
	return ret;
}

TypedArray<Dictionary> Script::_get_script_method_list() {
	TypedArray<Dictionary> ret;
	List<MethodInfo> list;
	get_script_method_list(&list);
	for (const MethodInfo &E : list) {
		ret.append(E.operator Dictionary());
	}
	return ret;
}

TypedArray<Dictionary> Script::_get_script_signal_list() {
	TypedArray<Dictionary> ret;
	List<MethodInfo> list;
	get_script_signal_list(&list);
	for (const MethodInfo &E : list) {
		ret.append(E.operator Dictionary());
	}
	return ret;
}

Dictionary Script::_get_script_constant_map() {
	Dictionary ret;
	HashMap<StringName, Variant> map;
	get_constants(&map);
	for (const KeyValue<StringName, Variant> &E : map) {
		ret[E.key] = E.value;
	}
	return ret;
}

#ifdef TOOLS_ENABLED

PropertyInfo Script::get_class_category() const {
	String path = get_path();
	String scr_name;

	if (is_built_in()) {
		if (get_name().is_empty()) {
			scr_name = TTR("Built-in script");
		} else {
			scr_name = vformat("%s (%s)", get_name(), TTR("Built-in"));
		}
	} else {
		if (get_name().is_empty()) {
			scr_name = path.get_file();
		} else {
			scr_name = get_name();
		}
	}

	return PropertyInfo(Variant::NIL, scr_name, PROPERTY_HINT_NONE, path, PROPERTY_USAGE_CATEGORY);
}

#endif // TOOLS_ENABLED

void Script::_bind_methods() {
	ClassDB::bind_method(D_METHOD("can_instantiate"), &Script::can_instantiate);
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
	ClassDB::bind_method(D_METHOD("is_abstract"), &Script::is_abstract);

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "source_code", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_source_code", "get_source_code");
}

void ScriptServer::set_scripting_enabled(bool p_enabled) {
	scripting_enabled = p_enabled;
}

bool ScriptServer::is_scripting_enabled() {
	return scripting_enabled;
}

ScriptLanguage *ScriptServer::get_language(int p_idx) {
	MutexLock lock(languages_mutex);
	ERR_FAIL_INDEX_V(p_idx, _language_count, nullptr);
	return _languages[p_idx];
}

Error ScriptServer::register_language(ScriptLanguage *p_language) {
	MutexLock lock(languages_mutex);
	ERR_FAIL_NULL_V(p_language, ERR_INVALID_PARAMETER);
	ERR_FAIL_COND_V_MSG(_language_count >= MAX_LANGUAGES, ERR_UNAVAILABLE, "Script languages limit has been reach, cannot register more.");
	for (int i = 0; i < _language_count; i++) {
		const ScriptLanguage *other_language = _languages[i];
		ERR_FAIL_COND_V_MSG(other_language->get_extension() == p_language->get_extension(), ERR_ALREADY_EXISTS, "A script language with extension '" + p_language->get_extension() + "' is already registered.");
		ERR_FAIL_COND_V_MSG(other_language->get_name() == p_language->get_name(), ERR_ALREADY_EXISTS, "A script language with name '" + p_language->get_name() + "' is already registered.");
		ERR_FAIL_COND_V_MSG(other_language->get_type() == p_language->get_type(), ERR_ALREADY_EXISTS, "A script language with type '" + p_language->get_type() + "' is already registered.");
	}
	_languages[_language_count++] = p_language;
	return OK;
}

Error ScriptServer::unregister_language(const ScriptLanguage *p_language) {
	MutexLock lock(languages_mutex);

	for (int i = 0; i < _language_count; i++) {
		if (_languages[i] == p_language) {
			_language_count--;
			if (i < _language_count) {
				SWAP(_languages[i], _languages[_language_count]);
			}
			return OK;
		}
	}
	return ERR_DOES_NOT_EXIST;
}

void ScriptServer::init_languages() {
	{ // Load global classes.
		global_classes_clear();
#ifndef DISABLE_DEPRECATED
		if (ProjectSettings::get_singleton()->has_setting("_global_script_classes")) {
			Array script_classes = GLOBAL_GET("_global_script_classes");

			for (int i = 0; i < script_classes.size(); i++) {
				Dictionary c = script_classes[i];
				if (!c.has("class") || !c.has("language") || !c.has("path") || !c.has("base")) {
					continue;
				}
				add_global_class(c["class"], c["base"], c["language"], c["path"]);
			}
			ProjectSettings::get_singleton()->clear("_global_script_classes");
		}
#endif

		Array script_classes = ProjectSettings::get_singleton()->get_global_class_list();
		for (int i = 0; i < script_classes.size(); i++) {
			Dictionary c = script_classes[i];
			if (!c.has("class") || !c.has("language") || !c.has("path") || !c.has("base")) {
				continue;
			}
			add_global_class(c["class"], c["base"], c["language"], c["path"]);
		}
	}

	HashSet<ScriptLanguage *> langs_to_init;
	{
		MutexLock lock(languages_mutex);
		for (int i = 0; i < _language_count; i++) {
			if (_languages[i]) {
				langs_to_init.insert(_languages[i]);
			}
		}
	}

	for (ScriptLanguage *E : langs_to_init) {
		E->init();
	}

	{
		MutexLock lock(languages_mutex);
		languages_ready = true;
	}
}

void ScriptServer::finish_languages() {
	HashSet<ScriptLanguage *> langs_to_finish;

	{
		MutexLock lock(languages_mutex);
		for (int i = 0; i < _language_count; i++) {
			if (_languages[i]) {
				langs_to_finish.insert(_languages[i]);
			}
		}
	}

	for (ScriptLanguage *E : langs_to_finish) {
		E->finish();
	}

	{
		MutexLock lock(languages_mutex);
		languages_ready = false;
	}

	global_classes_clear();
}

bool ScriptServer::are_languages_initialized() {
	MutexLock lock(languages_mutex);
	return languages_ready;
}

void ScriptServer::set_reload_scripts_on_save(bool p_enable) {
	reload_scripts_on_save = p_enable;
}

bool ScriptServer::is_reload_scripts_on_save_enabled() {
	return reload_scripts_on_save;
}

void ScriptServer::thread_enter() {
	MutexLock lock(languages_mutex);
	if (!languages_ready) {
		return;
	}
	for (int i = 0; i < _language_count; i++) {
		_languages[i]->thread_enter();
	}
}

void ScriptServer::thread_exit() {
	MutexLock lock(languages_mutex);
	if (!languages_ready) {
		return;
	}
	for (int i = 0; i < _language_count; i++) {
		_languages[i]->thread_exit();
	}
}

HashMap<StringName, ScriptServer::GlobalScriptClass> ScriptServer::global_classes;
HashMap<StringName, Vector<StringName>> ScriptServer::inheriters_cache;
bool ScriptServer::inheriters_cache_dirty = true;

void ScriptServer::global_classes_clear() {
	global_classes.clear();
	inheriters_cache.clear();
}

void ScriptServer::add_global_class(const StringName &p_class, const StringName &p_base, const StringName &p_language, const String &p_path) {
	ERR_FAIL_COND_MSG(p_class == p_base || (global_classes.has(p_base) && get_global_class_native_base(p_base) == p_class), "Cyclic inheritance in script class.");
	GlobalScriptClass g;
	g.language = p_language;
	g.path = p_path;
	g.base = p_base;
	global_classes[p_class] = g;
	inheriters_cache_dirty = true;
}

void ScriptServer::remove_global_class(const StringName &p_class) {
	global_classes.erase(p_class);
	inheriters_cache_dirty = true;
}

void ScriptServer::get_inheriters_list(const StringName &p_base_type, List<StringName> *r_classes) {
	if (inheriters_cache_dirty) {
		inheriters_cache.clear();
		for (const KeyValue<StringName, GlobalScriptClass> &K : global_classes) {
			if (!inheriters_cache.has(K.value.base)) {
				inheriters_cache[K.value.base] = Vector<StringName>();
			}
			inheriters_cache[K.value.base].push_back(K.key);
		}
		for (KeyValue<StringName, Vector<StringName>> &K : inheriters_cache) {
			K.value.sort_custom<StringName::AlphCompare>();
		}
		inheriters_cache_dirty = false;
	}

	if (!inheriters_cache.has(p_base_type)) {
		return;
	}

	const Vector<StringName> &v = inheriters_cache[p_base_type];
	for (int i = 0; i < v.size(); i++) {
		r_classes->push_back(v[i]);
	}
}

void ScriptServer::remove_global_class_by_path(const String &p_path) {
	for (const KeyValue<StringName, GlobalScriptClass> &kv : global_classes) {
		if (kv.value.path == p_path) {
			global_classes.erase(kv.key);
			inheriters_cache_dirty = true;
			return;
		}
	}
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
	List<StringName> classes;
	for (const KeyValue<StringName, GlobalScriptClass> &E : global_classes) {
		classes.push_back(E.key);
	}
	classes.sort_custom<StringName::AlphCompare>();
	for (const StringName &E : classes) {
		r_global_classes->push_back(E);
	}
}

void ScriptServer::save_global_classes() {
	Dictionary class_icons;

	Array script_classes = ProjectSettings::get_singleton()->get_global_class_list();
	for (int i = 0; i < script_classes.size(); i++) {
		Dictionary d = script_classes[i];
		if (!d.has("name") || !d.has("icon")) {
			continue;
		}
		class_icons[d["name"]] = d["icon"];
	}

	List<StringName> gc;
	get_global_class_list(&gc);
	Array gcarr;
	for (const StringName &E : gc) {
		Dictionary d;
		d["class"] = E;
		d["language"] = global_classes[E].language;
		d["path"] = global_classes[E].path;
		d["base"] = global_classes[E].base;
		d["icon"] = class_icons.get(E, "");
		gcarr.push_back(d);
	}
	ProjectSettings::get_singleton()->store_global_class_list(gcarr);
}

String ScriptServer::get_global_class_cache_file_path() {
	return ProjectSettings::get_singleton()->get_global_class_list_path();
}

////////////////////

ScriptCodeCompletionCache *ScriptCodeCompletionCache::singleton = nullptr;
ScriptCodeCompletionCache::ScriptCodeCompletionCache() {
	singleton = this;
}

void ScriptLanguage::get_core_type_words(List<String> *p_core_type_words) const {
	p_core_type_words->push_back("String");
	p_core_type_words->push_back("Vector2");
	p_core_type_words->push_back("Vector2i");
	p_core_type_words->push_back("Rect2");
	p_core_type_words->push_back("Rect2i");
	p_core_type_words->push_back("Vector3");
	p_core_type_words->push_back("Vector3i");
	p_core_type_words->push_back("Transform2D");
	p_core_type_words->push_back("Vector4");
	p_core_type_words->push_back("Vector4i");
	p_core_type_words->push_back("Plane");
	p_core_type_words->push_back("Quaternion");
	p_core_type_words->push_back("AABB");
	p_core_type_words->push_back("Basis");
	p_core_type_words->push_back("Transform3D");
	p_core_type_words->push_back("Projection");
	p_core_type_words->push_back("Color");
	p_core_type_words->push_back("StringName");
	p_core_type_words->push_back("NodePath");
	p_core_type_words->push_back("RID");
	p_core_type_words->push_back("Callable");
	p_core_type_words->push_back("Signal");
	p_core_type_words->push_back("Dictionary");
	p_core_type_words->push_back("Array");
	p_core_type_words->push_back("PackedByteArray");
	p_core_type_words->push_back("PackedInt32Array");
	p_core_type_words->push_back("PackedInt64Array");
	p_core_type_words->push_back("PackedFloat32Array");
	p_core_type_words->push_back("PackedFloat64Array");
	p_core_type_words->push_back("PackedStringArray");
	p_core_type_words->push_back("PackedVector2Array");
	p_core_type_words->push_back("PackedVector3Array");
	p_core_type_words->push_back("PackedColorArray");
}

void ScriptLanguage::frame() {
}

TypedArray<int> ScriptLanguage::CodeCompletionOption::get_option_characteristics(const String &p_base) {
	// Return characacteristics of the match found by order of importance.
	// Matches will be ranked by a lexicographical order on the vector returned by this function.
	// The lower values indicate better matches and that they should go before in the order of appearance.
	if (last_matches == matches) {
		return charac;
	}
	charac.clear();
	// Ensure base is not empty and at the same time that matches is not empty too.
	if (p_base.length() == 0) {
		last_matches = matches;
		charac.push_back(location);
		return charac;
	}
	charac.push_back(matches.size());
	charac.push_back((matches[0].first == 0) ? 0 : 1);
	const char32_t *target_char = &p_base[0];
	int bad_case = 0;
	for (const Pair<int, int> &match_segment : matches) {
		const char32_t *string_to_complete_char = &display[match_segment.first];
		for (int j = 0; j < match_segment.second; j++, string_to_complete_char++, target_char++) {
			if (*string_to_complete_char != *target_char) {
				bad_case++;
			}
		}
	}
	charac.push_back(bad_case);
	charac.push_back(location);
	charac.push_back(matches[0].first);
	last_matches = matches;
	return charac;
}

void ScriptLanguage::CodeCompletionOption::clear_characteristics() {
	charac = TypedArray<int>();
}

TypedArray<int> ScriptLanguage::CodeCompletionOption::get_option_cached_characteristics() const {
	// Only returns the cached value and warns if it was not updated since the last change of matches.
	if (last_matches != matches) {
		WARN_PRINT("Characteristics are not up to date.");
	}

	return charac;
}

bool PlaceHolderScriptInstance::set(const StringName &p_name, const Variant &p_value) {
	if (script->is_placeholder_fallback_enabled()) {
		return false;
	}

	if (values.has(p_name)) {
		Variant defval;
		if (script->get_property_default_value(p_name, defval)) {
			// The evaluate function ensures that a NIL variant is equal to e.g. an empty Resource.
			// Simply doing defval == p_value does not do this.
			if (Variant::evaluate(Variant::OP_EQUAL, defval, p_value)) {
				values.erase(p_name);
				return true;
			}
		}
		values[p_name] = p_value;
		return true;
	} else {
		Variant defval;
		if (script->get_property_default_value(p_name, defval)) {
			if (Variant::evaluate(Variant::OP_NOT_EQUAL, defval, p_value)) {
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
		for (const PropertyInfo &E : properties) {
			p_properties->push_back(E);
		}
	} else {
		for (const PropertyInfo &E : properties) {
			PropertyInfo pinfo = E;
			p_properties->push_back(E);
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

void PlaceHolderScriptInstance::update(const List<PropertyInfo> &p_properties, const HashMap<StringName, Variant> &p_values) {
	HashSet<StringName> new_values;
	for (const PropertyInfo &E : p_properties) {
		StringName n = E.name;
		new_values.insert(n);

		if (!values.has(n) || values[n].get_type() != E.type) {
			if (p_values.has(n)) {
				values[n] = p_values[n];
			}
		}
	}

	properties = p_properties;
	List<StringName> to_remove;

	for (KeyValue<StringName, Variant> &E : values) {
		if (!new_values.has(E.key)) {
			to_remove.push_back(E.key);
		}

		Variant defval;
		if (script->get_property_default_value(E.key, defval)) {
			//remove because it's the same as the default value
			if (defval == E.value) {
				to_remove.push_back(E.key);
			}
		}
	}

	while (to_remove.size()) {
		values.erase(to_remove.front()->get());
		to_remove.pop_front();
	}

	if (owner && owner->get_script_instance() == this) {
		owner->notify_property_list_changed();
	}
	//change notify

	constants.clear();
	script->get_constants(&constants);
}

void PlaceHolderScriptInstance::property_set_fallback(const StringName &p_name, const Variant &p_value, bool *r_valid) {
	if (script->is_placeholder_fallback_enabled()) {
		HashMap<StringName, Variant>::Iterator E = values.find(p_name);

		if (E) {
			E->value = p_value;
		} else {
			values.insert(p_name, p_value);
		}

		bool found = false;
		for (const PropertyInfo &F : properties) {
			if (F.name == p_name) {
				found = true;
				break;
			}
		}
		if (!found) {
			properties.push_back(PropertyInfo(p_value.get_type(), p_name, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_SCRIPT_VARIABLE));
		}
	}

	if (r_valid) {
		*r_valid = false; // Cannot change the value in either case
	}
}

Variant PlaceHolderScriptInstance::property_get_fallback(const StringName &p_name, bool *r_valid) {
	if (script->is_placeholder_fallback_enabled()) {
		HashMap<StringName, Variant>::ConstIterator E = values.find(p_name);

		if (E) {
			if (r_valid) {
				*r_valid = true;
			}
			return E->value;
		}

		E = constants.find(p_name);
		if (E) {
			if (r_valid) {
				*r_valid = true;
			}
			return E->value;
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
