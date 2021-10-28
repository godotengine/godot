/*************************************************************************/
/*  pluginscript_script.cpp                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifdef DEBUG_ENABLED
#define __ASSERT_SCRIPT_REASON "Cannot retrieve PluginScript class for this script, is your code correct?"
#define ASSERT_SCRIPT_VALID()                                       \
	{                                                               \
		ERR_FAIL_COND_MSG(!can_instance(), __ASSERT_SCRIPT_REASON); \
	}
#define ASSERT_SCRIPT_VALID_V(ret)                                         \
	{                                                                      \
		ERR_FAIL_COND_V_MSG(!can_instance(), ret, __ASSERT_SCRIPT_REASON); \
	}
#else
#define ASSERT_SCRIPT_VALID()
#define ASSERT_SCRIPT_VALID_V(ret)
#endif

void PluginScript::_bind_methods() {
	ClassDB::bind_vararg_method(METHOD_FLAGS_DEFAULT, "new", &PluginScript::_new, MethodInfo("new"));
}

PluginScriptInstance *PluginScript::_create_instance(const Variant **p_args, int p_argcount, Object *p_owner, Variant::CallError &r_error) {
	r_error.error = Variant::CallError::CALL_OK;

	// Create instance
	PluginScriptInstance *instance = memnew(PluginScriptInstance());

	if (instance->init(this, p_owner)) {
		_language->lock();
		_instances.insert(instance->get_owner());
		_language->unlock();
	} else {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		memdelete(instance);
		ERR_FAIL_V(nullptr);
	}

	// Construct
	// TODO: Support arguments in the constructor?
	// There is currently no way to get the constructor function name of the script.
	// instance->call("__init__", p_args, p_argcount, r_error);
	if (p_argcount > 0) {
		WARN_PRINT("PluginScript doesn't support arguments in the constructor");
	}

	return instance;
}

Variant PluginScript::_new(const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	r_error.error = Variant::CallError::CALL_OK;

	if (!_valid) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}

	REF ref;
	Object *owner = nullptr;

	if (get_instance_base_type() == "") {
		owner = memnew(Reference);
	} else {
		owner = ClassDB::instance(get_instance_base_type());
	}

	if (!owner) {
		r_error.error = Variant::CallError::CALL_ERROR_INSTANCE_IS_NULL;
		return Variant();
	}

	Reference *r = Object::cast_to<Reference>(owner);
	if (r) {
		ref = REF(r);
	}

	PluginScriptInstance *instance = _create_instance(p_args, p_argcount, owner, r_error);

	if (!instance) {
		if (ref.is_null()) {
			memdelete(owner); //no owner, sorry
		}
		return Variant();
	}

	if (ref.is_valid()) {
		return ref;
	} else {
		return owner;
	}
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

bool PluginScript::inherits_script(const Ref<Script> &p_script) const {
	Ref<PluginScript> ps = p_script;
	if (ps.is_null()) {
		return false;
	}

	const PluginScript *s = this;

	while (s) {
		if (s == p_script.ptr()) {
			return true;
		}
		s = Object::cast_to<PluginScript>(s->_ref_base_parent.ptr());
	}

	return false;
}

Ref<Script> PluginScript::get_base_script() const {
	if (_ref_base_parent.is_valid()) {
		return Ref<PluginScript>(_ref_base_parent);
	} else {
		return Ref<Script>();
	}
}

StringName PluginScript::get_instance_base_type() const {
	if (_native_parent) {
		return _native_parent;
	}
	if (_ref_base_parent.is_valid()) {
		return _ref_base_parent->get_instance_base_type();
	}
	return StringName();
}

void PluginScript::update_exports() {
#ifdef TOOLS_ENABLED
	ASSERT_SCRIPT_VALID();
	if (placeholders.size()) {
		//update placeholders if any
		Map<StringName, Variant> propdefvalues;
		List<PropertyInfo> propinfos;

		get_script_property_list(&propinfos);
		for (Set<PlaceHolderScriptInstance *>::Element *E = placeholders.front(); E; E = E->next()) {
			E->get()->update(propinfos, _properties_default_values);
		}
	}
#endif
}

// TODO: rename p_this "p_owner" ?
ScriptInstance *PluginScript::instance_create(Object *p_this) {
	ASSERT_SCRIPT_VALID_V(nullptr);
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

	StringName base_type = get_instance_base_type();
	if (base_type) {
		if (!ClassDB::is_parent_class(p_this->get_class_name(), base_type)) {
			String msg = "Script inherits from native type '" + String(base_type) + "', so it can't be instanced in object of type: '" + p_this->get_class() + "'";
			// TODO: implement PluginscriptLanguage::debug_break_parse
			// if (ScriptDebugger::get_singleton()) {
			// 	_language->debug_break_parse(get_path(), 0, msg);
			// }
			ERR_FAIL_V_MSG(nullptr, msg);
		}
	}

	Variant::CallError unchecked_error;
	return _create_instance(nullptr, 0, p_this, unchecked_error);
}

bool PluginScript::instance_has(const Object *p_this) const {
	ERR_FAIL_COND_V(!_language, false);

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
	if (_source == p_code) {
		return;
	}
	_source = p_code;
}

Error PluginScript::reload(bool p_keep_state) {
	ERR_FAIL_COND_V(!_language, ERR_UNCONFIGURED);

	_language->lock();
	ERR_FAIL_COND_V(!p_keep_state && _instances.size(), ERR_ALREADY_IN_USE);
	_language->unlock();

	_valid = false;
	String basedir = _path;

	if (basedir == "") {
		basedir = get_path();
	}

	if (basedir != "") {
		basedir = basedir.get_base_dir();
	}

	if (_data) {
		_desc->finish(_data);
	}

	Error err;
	godot_pluginscript_script_manifest manifest = _desc->init(
			_language->_data,
			(godot_string *)&_path,
			(godot_string *)&_source,
			(godot_error *)&err);
// Manifest's attributes must be explicitly freed
#define FREE_SCRIPT_MANIFEST(manifest)                    \
	{                                                     \
		godot_string_name_destroy(&manifest.name);        \
		godot_string_name_destroy(&manifest.base);        \
		godot_dictionary_destroy(&manifest.member_lines); \
		godot_array_destroy(&manifest.methods);           \
		godot_array_destroy(&manifest.signals);           \
		godot_array_destroy(&manifest.properties);        \
	}

	if (err) {
		FREE_SCRIPT_MANIFEST(manifest);
		// TODO: GDscript uses `ScriptDebugger` here to jump into the parsing error
		return err;
	}

	// Script's parent is passed as base_name which can make reference to a
	// ClassDB name (i.e. `Node2D`) or a resource path (i.e. `res://foo/bar.gd`)
	StringName *base_name = (StringName *)&manifest.base;
	if (*base_name) {
		if (ClassDB::class_exists(*base_name)) {
			_native_parent = *base_name;
		} else {
			Ref<Script> res = ResourceLoader::load(*base_name);
			if (res.is_valid()) {
				_ref_base_parent = res;
			} else {
				String name = *(StringName *)&manifest.name;
				FREE_SCRIPT_MANIFEST(manifest);
				ERR_FAIL_V_MSG(ERR_PARSE_ERROR, _path + ": Script '" + name + "' has an invalid parent '" + *base_name + "'.");
			}
		}
	}

	_valid = true;
	// Use the manifest to configure this script object
	_data = manifest.data;
	_name = *(StringName *)&manifest.name;
	_tool = manifest.is_tool;

	Dictionary *members = (Dictionary *)&manifest.member_lines;
	for (const Variant *key = members->next(); key != nullptr; key = members->next(key)) {
		_member_lines[*key] = (*members)[*key];
	}
	Array *methods = (Array *)&manifest.methods;
	for (int i = 0; i < methods->size(); ++i) {
		Dictionary v = (*methods)[i];
		MethodInfo mi = MethodInfo::from_dict(v);
		_methods_info[mi.name] = mi;
		// rpc_mode is passed as an optional field and is not part of MethodInfo
		Variant var = v["rpc_mode"];
		if (var == Variant()) {
			_methods_rpc_mode[mi.name] = MultiplayerAPI::RPC_MODE_DISABLED;
		} else {
			_methods_rpc_mode[mi.name] = MultiplayerAPI::RPCMode(int(var));
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
			_methods_rpc_mode[pi.name] = MultiplayerAPI::RPC_MODE_DISABLED;
		} else {
			_methods_rpc_mode[pi.name] = MultiplayerAPI::RPCMode(int(var));
		}
	}

	FREE_SCRIPT_MANIFEST(manifest);
	return OK;
#undef FREE_SCRIPT_MANIFEST
}

void PluginScript::get_script_method_list(List<MethodInfo> *r_methods) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, MethodInfo>::Element *e = _methods_info.front(); e != nullptr; e = e->next()) {
		r_methods->push_back(e->get());
	}
}

void PluginScript::get_script_property_list(List<PropertyInfo> *r_properties) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, PropertyInfo>::Element *e = _properties_info.front(); e != nullptr; e = e->next()) {
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
	if (e != nullptr) {
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
	if (e != nullptr) {
		return e->get();
	} else {
		return PropertyInfo();
	}
}

bool PluginScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	ASSERT_SCRIPT_VALID_V(false);
#ifdef TOOLS_ENABLED
	const Map<StringName, Variant>::Element *e = _properties_default_values.find(p_property);
	if (e != nullptr) {
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
	ERR_FAIL_COND_V_MSG(err, err, "Cannot open file '" + p_path + "'.");

	uint64_t len = f->get_len();
	sourcef.resize(len + 1);
	PoolVector<uint8_t>::Write w = sourcef.write();
	uint64_t r = f->get_buffer(w.ptr(), len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r != len, ERR_CANT_OPEN);
	w[len] = 0;

	String s;
	if (s.parse_utf8((const char *)w.ptr())) {
		ERR_FAIL_V_MSG(ERR_INVALID_DATA, "Script '" + p_path + "' contains invalid unicode (UTF-8), so it was not loaded. Please ensure that scripts are saved in valid UTF-8 unicode.");
	}

	_source = s;
	_path = p_path;
	return OK;
}

bool PluginScript::has_script_signal(const StringName &p_signal) const {
	ASSERT_SCRIPT_VALID_V(false);
	return _signals_info.has(p_signal);
}

void PluginScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	ASSERT_SCRIPT_VALID();
	for (Map<StringName, MethodInfo>::Element *e = _signals_info.front(); e != nullptr; e = e->next()) {
		r_signals->push_back(e->get());
	}
}

int PluginScript::get_member_line(const StringName &p_member) const {
#ifdef TOOLS_ENABLED
	if (_member_lines.has(p_member)) {
		return _member_lines[p_member];
	}
#endif
	return -1;
}

MultiplayerAPI::RPCMode PluginScript::get_rpc_mode(const StringName &p_method) const {
	ASSERT_SCRIPT_VALID_V(MultiplayerAPI::RPC_MODE_DISABLED);
	const Map<StringName, MultiplayerAPI::RPCMode>::Element *e = _methods_rpc_mode.find(p_method);
	if (e != nullptr) {
		return e->get();
	} else {
		return MultiplayerAPI::RPC_MODE_DISABLED;
	}
}

MultiplayerAPI::RPCMode PluginScript::get_rset_mode(const StringName &p_variable) const {
	ASSERT_SCRIPT_VALID_V(MultiplayerAPI::RPC_MODE_DISABLED);
	const Map<StringName, MultiplayerAPI::RPCMode>::Element *e = _variables_rset_mode.find(p_variable);
	if (e != nullptr) {
		return e->get();
	} else {
		return MultiplayerAPI::RPC_MODE_DISABLED;
	}
}

PluginScript::PluginScript() :
		_data(nullptr),
		_desc(nullptr),
		_language(nullptr),
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
	if (_desc && _data) {
		_desc->finish(_data);
	}

#ifdef DEBUG_ENABLED
	if (_language) {
		_language->lock();
		_language->_script_list.remove(&_script_list);
		_language->unlock();
	}
#endif
}
