/*************************************************************************/
/*  pluginscript_script.cpp                                              */
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

// Godot imports
#include "core/os/file_access.h"
// PluginScript imports
#include "pluginscript_instance.h"
#include "pluginscript_script.h"

#if DEBUG_ENABLED
#define __ASSERT_SCRIPT_REASON "Cannot retrieve pluginscript class for this script, is you code correct ?"
#define ASSERT_SCRIPT_VALID()                \
	{                                        \
		ERR_EXPLAIN(__ASSERT_SCRIPT_REASON); \
		ERR_FAIL_COND(!can_instance())       \
	}
#define ASSERT_SCRIPT_VALID_V(ret)            \
	{                                         \
		ERR_EXPLAIN(__ASSERT_SCRIPT_REASON);  \
		ERR_FAIL_COND_V(!can_instance(), ret) \
	}
#else
#define ASSERT_SCRIPT_VALID()
#define ASSERT_SCRIPT_VALID_V(ret)
#endif

void PluginScript::_bind_methods() {
}

#ifdef TOOLS_ENABLED

void PluginScript::_placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {
	placeholders.erase(p_placeholder);
}

#endif

bool PluginScript::can_instance() const {
	bool can = _valid || (!_tool && !ScriptServer::is_scripting_enabled());
	return can;
}

Ref<Script> PluginScript::get_base_script() const {
	if (_ref_base_parent.is_valid()) {
		return Ref<PluginScript>(_ref_base_parent);
	} else {
		return Ref<Script>();
	}
}

StringName PluginScript::get_instance_base_type() const {
	if (_native_parent)
		return _native_parent;
	if (_ref_base_parent.is_valid())
		return _ref_base_parent->get_instance_base_type();
	return StringName();
}

void PluginScript::update_exports() {
// TODO
#ifdef TOOLS_ENABLED
#if 0
	ASSERT_SCRIPT_VALID();
	if (/*changed &&*/ placeholders.size()) { //hm :(

		//update placeholders if any
		Map<StringName, Variant> propdefvalues;
		List<PropertyInfo> propinfos;
		const String *props = (const String *)pybind_get_prop_list(_py_exposed_class);
		for (int i = 0; props[i] != ""; ++i) {
			const String propname = props[i];
			pybind_get_prop_default_value(_py_exposed_class, propname.c_str(), (godot_variant *)&propdefvalues[propname]);
			pybind_prop_info raw_info;
			pybind_get_prop_info(_py_exposed_class, propname.c_str(), &raw_info);
			PropertyInfo info;
			info.type = (Variant::Type)raw_info.type;
			info.name = propname;
			info.hint = (PropertyHint)raw_info.hint;
			info.hint_string = *(String *)&raw_info.hint_string;
			info.usage = raw_info.usage;
			propinfos.push_back(info);
		}
		for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
			E->get()->update(propinfos, propdefvalues);
		}
	}
#endif
#endif
}

// TODO: rename p_this "p_owner" ?
ScriptInstance *PluginScript::instance_create(Object *p_this) {
	ASSERT_SCRIPT_VALID_V(NULL);
	// TODO check script validity ?
	if (!_tool && !ScriptServer::is_scripting_enabled()) {
#ifdef TOOLS_ENABLED
		// Instance a fake script for editing the values
		PlaceHolderScriptInstance *si = memnew(PlaceHolderScriptInstance(get_language(), Ref<Script>(this), p_this));
		placeholders.insert(si);
		update_exports();
		return si;
#else
		return NULL;
#endif
	}

	PluginScript *top = this;
	// TODO: can be optimized by storing a PluginScript::_base_parent direct pointer
	while (top->_ref_base_parent.is_valid())
		top = top->_ref_base_parent.ptr();
	if (top->_native_parent) {
		if (!ClassDB::is_parent_class(p_this->get_class_name(), top->_native_parent)) {
			String msg = "Script inherits from native type '" + String(top->_native_parent) + "', so it can't be instanced in object of type: '" + p_this->get_class() + "'";
			// TODO: implement PluginscriptLanguage::debug_break_parse
			// if (ScriptDebugger::get_singleton()) {
			// 	_language->debug_break_parse(get_path(), 0, msg);
			// }
			ERR_EXPLAIN(msg);
			ERR_FAIL_V(NULL);
		}
	}

	PluginScriptInstance *instance = memnew(PluginScriptInstance());
	const bool success = instance->init(this, p_this);
	if (success) {
		_language->lock();
		_instances.insert(instance->get_owner());
		_language->unlock();
		return instance;
	} else {
		memdelete(instance);
		ERR_FAIL_V(NULL);
	}
}

bool PluginScript::instance_has(const Object *p_this) const {
	_language->lock();
	bool hasit = _instances.has((Object *)p_this);
	_language->unlock();
	return hasit;
}

bool PluginScript::has_source_code() const {
	bool has = _source != "";
	return has;
}

String PluginScript::get_source_code() const {
	return _source;
}

void PluginScript::set_source_code(const String &p_code) {
	if (_source == p_code)
		return;
	_source = p_code;
}

Error PluginScript::reload(bool p_keep_state) {
	_language->lock();
	ERR_FAIL_COND_V(!p_keep_state && _instances.size(), ERR_ALREADY_IN_USE);
	_language->unlock();

	_valid = false;
	String basedir = _path;

	if (basedir == "")
		basedir = get_path();

	if (basedir != "")
		basedir = basedir.get_base_dir();

	if (_data) {
		_desc->finish(_data);
	}

	Error err;
	godot_pluginscript_script_manifest manifest = _desc->init(
			_language->_data,
			(godot_string *)&_path,
			(godot_string *)&_source,
			(godot_error *)&err);
	if (err) {
		// TODO: GDscript uses `ScriptDebugger` here to jump into the parsing error
		return err;
	}
	_valid = true;
	// Use the manifest to configure this script object
	_data = manifest.data;
	_name = *(StringName *)&manifest.name;
	_tool = manifest.is_tool;
	// Base name is either another PluginScript or a regular class accessible
	// through ClassDB
	StringName *base_name = (StringName *)&manifest.base;
	for (SelfList<PluginScript> *e = _language->_script_list.first(); e != NULL; e = e->next()) {
		if (e->self()->_name == *base_name) {
			// Found you, base is a PluginScript !
			_ref_base_parent = Ref<PluginScript>(e->self());
			break;
		}
	}
	if (!_ref_base_parent.is_valid()) {
		// Base is a native ClassDB
		if (!ClassDB::class_exists(*base_name)) {
			ERR_EXPLAIN("Unknown script '" + String(_name) + "' parent '" + String(*base_name) + "'.");
			ERR_FAIL_V(ERR_PARSE_ERROR);
		}
		_native_parent = *base_name;
	}

	Dictionary *members = (Dictionary *)&manifest.member_lines;
	for (const Variant *key = members->next(); key != NULL; key = members->next(key)) {
		_member_lines[*key] = (*members)[key];
	}
	Array *methods = (Array *)&manifest.methods;
	for (int i = 0; i < methods->size(); ++i) {
		Dictionary v = (*methods)[i];
		MethodInfo mi = MethodInfo::from_dict(v);
		_methods_info[mi.name] = mi;
		// rpc_mode is passed as an optional field and is not part of MethodInfo
		Variant var = v["rpc_mode"];
		if (var == Variant()) {
			_methods_rpc_mode[mi.name] = ScriptInstance::RPC_MODE_DISABLED;
		} else {
			_methods_rpc_mode[mi.name] = ScriptInstance::RPCMode(int(var));
		}
	}
	Array *signals = (Array *)&manifest.signals;
	for (int i = 0; i < signals->size(); ++i) {
		Variant v = (*signals)[i];
		MethodInfo mi = MethodInfo::from_dict(v);
		_signals_info[mi.name] = mi;
	}
	Array *properties = (Array *)&manifest.properties;
	for (int i = 0; i < properties->size(); ++i) {
		Dictionary v = (*properties)[i];
		PropertyInfo pi = PropertyInfo::from_dict(v);
		_properties_info[pi.name] = pi;
		_properties_default_values[pi.name] = v["default_value"];
		// rset_mode is passed as an optional field and is not part of PropertyInfo
		Variant var = v["rset_mode"];
		if (var == Variant()) {
			_methods_rpc_mode[pi.name] = ScriptInstance::RPC_MODE_DISABLED;
		} else {
			_methods_rpc_mode[pi.name] = ScriptInstance::RPCMode(int(var));
		}
	}
	// Manifest's attributes must be explicitly freed
	godot_string_name_destroy(&manifest.name);
	godot_string_name_destroy(&manifest.base);
	godot_dictionary_destroy(&manifest.member_lines);
	godot_array_destroy(&manifest.methods);
	godot_array_destroy(&manifest.signals);
	godot_array_destroy(&manifest.properties);

#ifdef TOOLS_ENABLED
/*for (Set<PlaceHolderScriptInstance*>::Element *E=placeholders.front();E;E=E->next()) {

        _update_placeholder(E->get());
    }*/
#endif
	return OK;
}

void PluginScript::get_script_method_list(List<MethodInfo> *r_methods) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, MethodInfo>::Element *e = _methods_info.front(); e != NULL; e = e->next()) {
		r_methods->push_back(e->get());
	}
}

void PluginScript::get_script_property_list(List<PropertyInfo> *r_properties) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, PropertyInfo>::Element *e = _properties_info.front(); e != NULL; e = e->next()) {
		r_properties->push_back(e->get());
	}
}

bool PluginScript::has_method(const StringName &p_method) const {
	ASSERT_SCRIPT_VALID_V(false);
	return _methods_info.has(p_method);
}

MethodInfo PluginScript::get_method_info(const StringName &p_method) const {
	ASSERT_SCRIPT_VALID_V(MethodInfo());
	const Map<StringName, MethodInfo>::Element *e = _methods_info.find(p_method);
	if (e != NULL) {
		return e->get();
	} else {
		return MethodInfo();
	}
}

bool PluginScript::has_property(const StringName &p_method) const {
	ASSERT_SCRIPT_VALID_V(false);
	return _properties_info.has(p_method);
}

PropertyInfo PluginScript::get_property_info(const StringName &p_property) const {
	ASSERT_SCRIPT_VALID_V(PropertyInfo());
	const Map<StringName, PropertyInfo>::Element *e = _properties_info.find(p_property);
	if (e != NULL) {
		return e->get();
	} else {
		return PropertyInfo();
	}
}

bool PluginScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	ASSERT_SCRIPT_VALID_V(false);
#ifdef TOOLS_ENABLED
	const Map<StringName, Variant>::Element *e = _properties_default_values.find(p_property);
	if (e != NULL) {
		r_value = e->get();
		return true;
	} else {
		return false;
	}
#endif
	return false;
}

ScriptLanguage *PluginScript::get_language() const {
	return _language;
}

Error PluginScript::load_source_code(const String &p_path) {

	PoolVector<uint8_t> sourcef;
	Error err;
	FileAccess *f = FileAccess::open(p_path, FileAccess::READ, &err);
	if (err) {
		ERR_FAIL_COND_V(err, err);
	}

	int len = f->get_len();
	sourcef.resize(len + 1);
	PoolVector<uint8_t>::Write w = sourcef.write();
	int r = f->get_buffer(w.ptr(), len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r != len, ERR_CANT_OPEN);
	w[len] = 0;

	String s;
	if (s.parse_utf8((const char *)w.ptr())) {
		ERR_EXPLAIN("Script '" + p_path + "' contains invalid unicode (utf-8), so it was not loaded. Please ensure that scripts are saved in valid utf-8 unicode.");
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	_source = s;
#ifdef TOOLS_ENABLED
// source_changed_cache=true;
#endif
	_path = p_path;
	return OK;
}

bool PluginScript::has_script_signal(const StringName &p_signal) const {
	ASSERT_SCRIPT_VALID_V(false);
	return _signals_info.has(p_signal);
}

void PluginScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, MethodInfo>::Element *e = _signals_info.front(); e != NULL; e = e->next()) {
		r_signals->push_back(e->get());
	}
}

int PluginScript::get_member_line(const StringName &p_member) const {
#ifdef TOOLS_ENABLED
	if (_member_lines.has(p_member))
		return _member_lines[p_member];
	else
#endif
		return -1;
}

ScriptInstance::RPCMode PluginScript::get_rpc_mode(const StringName &p_method) const {
	ASSERT_SCRIPT_VALID_V(ScriptInstance::RPC_MODE_DISABLED);
	const Map<StringName, ScriptInstance::RPCMode>::Element *e = _methods_rpc_mode.find(p_method);
	if (e != NULL) {
		return e->get();
	} else {
		return ScriptInstance::RPC_MODE_DISABLED;
	}
}

ScriptInstance::RPCMode PluginScript::get_rset_mode(const StringName &p_variable) const {
	ASSERT_SCRIPT_VALID_V(ScriptInstance::RPC_MODE_DISABLED);
	const Map<StringName, ScriptInstance::RPCMode>::Element *e = _variables_rset_mode.find(p_variable);
	if (e != NULL) {
		return e->get();
	} else {
		return ScriptInstance::RPC_MODE_DISABLED;
	}
}

PluginScript::PluginScript() :
		_data(NULL),
		_tool(false),
		_valid(false),
		_script_list(this) {
}

void PluginScript::init(PluginScriptLanguage *language) {
	_desc = &language->_desc.script_desc;
	_language = language;

#ifdef DEBUG_ENABLED
	_language->lock();
	_language->_script_list.add(&_script_list);
	_language->unlock();
#endif
}

PluginScript::~PluginScript() {
	_desc->finish(_data);

#ifdef DEBUG_ENABLED
	_language->lock();
	_language->_script_list.remove(&_script_list);
	_language->unlock();
#endif
}
