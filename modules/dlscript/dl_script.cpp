/*************************************************************************/
/*  dl_script.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#include "dl_script.h"

#include "global_config.h"
#include "global_constants.h"
#include "io/file_access_encrypted.h"
#include "os/file_access.h"
#include "os/os.h"

#include "scene/resources/scene_format_text.h"

#ifdef TOOLS_ENABLED
// #include "editor/editor_import_export.h"
#endif

#if defined(TOOLS_ENABLED) && defined(DEBUG_METHODS_ENABLED)
#include "api_generator.h"
#endif

// Script

bool DLScript::can_instance() const {
#ifdef DLSCRIPT_EDITOR_FEATURES
	return script_data || (!is_tool() && !ScriptServer::is_scripting_enabled());
#else
	// allow defaultlibrary without editor features
	if (!library.is_valid()) {
		String path = GLOBAL_GET("dlscript/default_dllibrary");

		RES lib = ResourceLoader::load(path);

		if (lib.is_valid() && lib->cast_to<DLLibrary>()) {
			return true;
		}
	}

	return script_data;
#endif
	//return script_data || (!tool && !ScriptServer::is_scripting_enabled());
	// change to true enable in editor stuff.
}

Ref<Script> DLScript::get_base_script() const {
	Ref<DLScript> base_script;
	base_script->library = library;
	base_script->script_data = script_data;
	base_script->script_name = script_data->base;
	return base_script;
}

StringName DLScript::get_instance_base_type() const {
	return script_data->base_native_type;
}

ScriptInstance *DLScript::instance_create(Object *p_this) {

#ifdef TOOLS_ENABLED

// find a good way to initialize stuff in the editor
#ifdef DLSCRIPT_EDITOR_FEATURES
	if (!ScriptServer::is_scripting_enabled() && !is_tool()) {
		// placeholder, for nodes. But for tools we want the real thing

		PlaceHolderScriptInstance *sins = memnew(PlaceHolderScriptInstance(DLScriptLanguage::singleton, Ref<Script>((Script *)this), p_this));
		placeholders.insert(sins);

		List<PropertyInfo> pinfo;
		Map<StringName, Variant> values;

		if (!library.is_valid())
			return sins;

		if (!library->library_handle) {
			Error err = library->_initialize_handle(true);
			if (err != OK) {
				return sins;
			}
		}

		if (!script_data) {
			script_data = library->get_script_data(script_name);
		}
		if (script_data)
			script_data->create_func.create_func((godot_object *)p_this, script_data->create_func.method_data);

		if (script_data) {
			for (Map<StringName, DLScriptData::Property>::Element *E = script_data->properties.front(); E; E = E->next()) {

				PropertyInfo p = E->get().info;
				p.name = String(E->key());
				pinfo.push_back(p);
				values[p.name] = E->get().default_value;
			}
		}

		sins->update(pinfo, values);

		return sins;
	}
#endif

#endif

	if (!library.is_valid()) {
		String path = GLOBAL_GET("dlscript/default_dllibrary");

		RES lib = ResourceLoader::load(path);

		if (lib.is_valid() && lib->cast_to<DLLibrary>()) {
			set_library(lib);
		}
	}

	DLInstance *new_instance = memnew(DLInstance);

	new_instance->owner = p_this;
	new_instance->script = Ref<DLScript>(this);

#ifndef DLSCRIPT_EDITOR_FEATURES
	if (!ScriptServer::is_scripting_enabled()) {
		new_instance->userdata = 0;
	} else {
		new_instance->userdata = script_data->create_func.create_func((godot_object *)p_this, script_data->create_func.method_data);
	}
#else
	new_instance->userdata = script_data->create_func.create_func((godot_object *)p_this, script_data->create_func.method_data);
#endif

	instances.insert(p_this);
	return new_instance;
}

bool DLScript::instance_has(const Object *p_this) const {
	return instances.has((Object *)p_this); // TODO
}

bool DLScript::has_source_code() const {
	return false;
}

String DLScript::get_source_code() const {
	return "";
}

Error DLScript::reload(bool p_keep_state) {
	return FAILED;
}

bool DLScript::has_method(const StringName &p_method) const {
	if (!script_data)
		return false;
	return script_data->methods.has(p_method);
}

MethodInfo DLScript::get_method_info(const StringName &p_method) const {
	if (!script_data)
		return MethodInfo();
	ERR_FAIL_COND_V(!script_data->methods.has(p_method), MethodInfo());
	return script_data->methods[p_method].info;
}

void DLScript::get_script_method_list(List<MethodInfo> *p_list) const {
	if (!script_data) return;
	for (Map<StringName, DLScriptData::Method>::Element *E = script_data->methods.front(); E; E = E->next()) {
		p_list->push_back(E->get().info);
	}
}

void DLScript::get_script_property_list(List<PropertyInfo> *p_list) const {
	if (!script_data) return;
	for (Map<StringName, DLScriptData::Property>::Element *E = script_data->properties.front(); E; E = E->next()) {
		p_list->push_back(E->get().info);
	}
}

bool DLScript::get_property_default_value(const StringName &p_property, Variant &r_value) const {
	if (!script_data) return false;
	if (script_data->properties.has(p_property)) {
		r_value = script_data->properties[p_property].default_value;
		return true;
	}
	return false;
}

bool DLScript::is_tool() const {
	ERR_FAIL_COND_V(!script_data, false);
	return script_data->is_tool;
}

String DLScript::get_node_type() const {
	return ""; // ?
}

ScriptLanguage *DLScript::get_language() const {
	return DLScriptLanguage::singleton;
}

bool DLScript::has_script_signal(const StringName &p_signal) const {
	if (!script_data)
		return false;
	return script_data->signals_.has(p_signal);
}

void DLScript::get_script_signal_list(List<MethodInfo> *r_signals) const {
	if (!script_data)
		return;
	for (Map<StringName, DLScriptData::Signal>::Element *S = script_data->signals_.front(); S; S = S->next()) {
		r_signals->push_back(S->get().signal);
	}
}

Ref<DLLibrary> DLScript::get_library() const {
	return library;
}

void DLScript::set_library(Ref<DLLibrary> p_library) {
	library = p_library;

#ifdef TOOLS_ENABLED
	if (!ScriptServer::is_scripting_enabled())
		return;
#endif
	if (library.is_valid()) {
		Error initalize_status = library->_initialize_handle(!ScriptServer::is_scripting_enabled());
		ERR_FAIL_COND(initalize_status != OK);
		if (script_name) {
			script_data = library->get_script_data(script_name);
			ERR_FAIL_COND(!script_data);
		}
	}
}

StringName DLScript::get_script_name() const {
	return script_name;
}

void DLScript::set_script_name(StringName p_script_name) {
	script_name = p_script_name;

	if (library.is_valid()) {
		script_data = library->get_script_data(script_name);
		ERR_FAIL_COND(!script_data);
	}
}

void DLScript::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_library"), &DLScript::get_library);
	ClassDB::bind_method(D_METHOD("set_library", "library"), &DLScript::set_library);
	ClassDB::bind_method(D_METHOD("get_script_name"), &DLScript::get_script_name);
	ClassDB::bind_method(D_METHOD("set_script_name", "script_name"), &DLScript::set_script_name);

	ADD_PROPERTYNZ(PropertyInfo(Variant::OBJECT, "library", PROPERTY_HINT_RESOURCE_TYPE, "DLLibrary"), "set_library", "get_library");
	ADD_PROPERTYNZ(PropertyInfo(Variant::STRING, "script_name"), "set_script_name", "get_script_name");
}

DLScript::DLScript() {
	script_data = NULL;
}

DLScript::~DLScript() {
	//hmm
}

// Library

DLLibrary *DLLibrary::currently_initialized_library = NULL;

DLLibrary *DLLibrary::get_currently_initialized_library() {
	return currently_initialized_library;
}

static const char *_dl_platforms_info[] = {
	"|unix|so|Unix",
	"unix|x11|so|X11",
	"unix|server|so|Server",
	"unix|android|so|Android",
	"unix|blackberry|so|Blackberry 10",
	"unix|haiku|so|Haiku", // Right?
	"|mac|dynlib|Mac",
	"mac|ios|dynlib|iOS",
	"mac|osx|dynlib|OSX",
	"|html5|js|HTML5",
	"|windows|dll|Windows",
	"windows|uwp|dll|UWP",
	NULL // Finishing condition
};

void DLLibrary::set_platform_file(StringName p_platform, String p_file) {
	if (p_file.empty()) {
		platform_files.erase(p_platform);
	} else {
		platform_files[p_platform] = p_file;
	}
}

String DLLibrary::get_platform_file(StringName p_platform) const {
	if (platform_files.has(p_platform)) {
		return platform_files[p_platform];
	} else {
		return "";
	}
}

Error DLLibrary::_initialize_handle(bool p_in_editor) {
	_THREAD_SAFE_METHOD_

	void *_library_handle;

	// Get the file

	const String platform_name = OS::get_singleton()->get_name();
	String platform_file("");
	char **platform_info = (char **)_dl_platforms_info;

	if (platform_files.has(platform_name.to_lower())) {
		platform_file = platform_files[platform_name.to_lower()];
	}

	while (*platform_info) {
		String platform_info_string(*platform_info);

		if (platform_name == platform_info_string.get_slicec('|', 3)) {
			String platform_key = platform_info_string.get_slicec('|', 1);
			String fallback_platform_key = platform_info_string.get_slicec('|', 0);

			if (platform_files.has(platform_key)) {
				platform_file = platform_files[platform_key];
			} else if (!fallback_platform_key.empty() && platform_files.has(fallback_platform_key)) {
				platform_file = platform_files[fallback_platform_key];
			} else {
				return ERR_UNAVAILABLE;
			}
		}
		platform_info++;
	}
	ERR_FAIL_COND_V(platform_file == "", ERR_DOES_NOT_EXIST);

	library_path = GlobalConfig::get_singleton()->globalize_path(platform_file);

	if (DLScriptLanguage::get_singleton()->is_library_initialized(library_path)) {
		*this = *DLScriptLanguage::get_singleton()->get_library_dllibrary(library_path);
		return OK;
	}

	// Open the file

	Error error;
	error = OS::get_singleton()->open_dynamic_library(library_path, _library_handle);
	if (error) return error;
	ERR_FAIL_COND_V(!_library_handle, ERR_BUG);

	// Get the method

	void *library_init;
	error = OS::get_singleton()->get_dynamic_library_symbol_handle(_library_handle, DLScriptLanguage::get_init_symbol_name(), library_init);
	if (error) return error;
	ERR_FAIL_COND_V(!library_init, ERR_BUG);

	DLLibrary::currently_initialized_library = this;

	void (*library_init_fpointer)(godot_dlscript_init_options *) = (void (*)(godot_dlscript_init_options *))library_init;

	godot_dlscript_init_options options;

	options.in_editor = p_in_editor;
	options.core_api_hash = ClassDB::get_api_hash(ClassDB::API_CORE);
	options.editor_api_hash = ClassDB::get_api_hash(ClassDB::API_EDITOR);
	options.no_api_hash = ClassDB::get_api_hash(ClassDB::API_NONE);

	library_init_fpointer(&options); // Catch errors?
	/*{
		ERR_EXPLAIN("Couldn't initialize library");
		ERR_FAIL_V(ERR_SCRIPT_FAILED);
	}*/

	DLLibrary::currently_initialized_library = NULL;
	library_handle = _library_handle;

	DLScriptLanguage::get_singleton()->set_library_initialized(library_path, this);

	return OK;
}

Error DLLibrary::_free_handle(bool p_in_editor) {
	ERR_FAIL_COND_V(!library_handle, ERR_BUG);

	if (!DLScriptLanguage::get_singleton()->is_library_initialized(library_path)) {
		OS::get_singleton()->close_dynamic_library(library_handle);
		library_handle = 0;
		return OK;
	}

	Error error = OK;
	void *library_terminate;
	error = OS::get_singleton()->get_dynamic_library_symbol_handle(library_handle, DLScriptLanguage::get_terminate_symbol_name(), library_terminate);
	if (error)
		return OK; // no terminate? okay, not that important lol

	void (*library_terminate_pointer)(godot_dlscript_terminate_options *) = (void (*)(godot_dlscript_terminate_options *))library_terminate;

	godot_dlscript_terminate_options options;
	options.in_editor = p_in_editor;

	library_terminate_pointer(&options);

	DLScriptLanguage::get_singleton()->set_library_uninitialized(library_path);

	OS::get_singleton()->close_dynamic_library(library_handle);
	library_handle = 0;

	return OK;
}

void DLLibrary::_register_script(const StringName p_name, const StringName p_base, godot_instance_create_func p_instance_func, godot_instance_destroy_func p_destroy_func) {
	ERR_FAIL_COND(scripts.has(p_name));

	DLScriptData *s = memnew(DLScriptData);
	s->base = p_base;
	s->create_func = p_instance_func;
	s->destroy_func = p_destroy_func;
	Map<StringName, DLScriptData *>::Element *E = scripts.find(p_base);
	if (E) {
		s->base_data = E->get();
		s->base_native_type = s->base_data->base_native_type;
	} else {
		if (!ClassDB::class_exists(p_base)) {
			memdelete(s);
			ERR_EXPLAIN("Invalid base for registered type '" + p_name + "'");
			ERR_FAIL();
		}
		s->base_native_type = p_base;
	}

	scripts.insert(p_name, s);
}

void DLLibrary::_register_tool_script(const StringName p_name, const StringName p_base, godot_instance_create_func p_instance_func, godot_instance_destroy_func p_destroy_func) {
	ERR_FAIL_COND(scripts.has(p_name));

	DLScriptData *s = memnew(DLScriptData);
	s->base = p_base;
	s->create_func = p_instance_func;
	s->destroy_func = p_destroy_func;
	s->is_tool = true;
	Map<StringName, DLScriptData *>::Element *E = scripts.find(p_base);
	if (E) {
		s->base_data = E->get();
		s->base_native_type = s->base_data->base_native_type;
	} else {
		if (!ClassDB::class_exists(p_base)) {
			memdelete(s);
			ERR_EXPLAIN("Invalid base for registered type '" + p_name + "'");
			ERR_FAIL();
		}
		s->base_native_type = p_base;
	}

	scripts.insert(p_name, s);
}

void DLLibrary::_register_script_method(const StringName p_name, const StringName p_method, godot_method_attributes p_attr, godot_instance_method p_func, MethodInfo p_info) {
	ERR_FAIL_COND(!scripts.has(p_name));

	p_info.name = p_method;
	DLScriptData::Method method;

	method = DLScriptData::Method(p_func, p_info, p_attr.rpc_type);

	scripts[p_name]->methods.insert(p_method, method);
}

void DLLibrary::_register_script_property(const StringName p_name, const String p_path, godot_property_attributes *p_attr, godot_property_set_func p_setter, godot_property_get_func p_getter) {
	ERR_FAIL_COND(!scripts.has(p_name));

	DLScriptData::Property p;

	PropertyInfo pi;
	pi.name = p_path;

	if (p_attr != NULL) {
		pi = PropertyInfo((Variant::Type)p_attr->type, p_path, (PropertyHint)p_attr->hint, *(String *)&p_attr->hint_string, p_attr->usage);

		p = DLScriptData::Property(p_setter, p_getter, pi, *(Variant *)&p_attr->default_value, p_attr->rset_type);
	}

	scripts[p_name]->properties.insert(p_path, p);
}

void DLLibrary::_register_script_signal(const StringName p_name, const godot_signal *p_signal) {
	ERR_FAIL_COND(!scripts.has(p_name));
	ERR_FAIL_COND(!p_signal);

	DLScriptData::Signal signal;

	signal.signal.name = *(String *)&p_signal->name;

	{
		List<PropertyInfo> arguments;
		for (int i = 0; i < p_signal->num_args; i++) {
			PropertyInfo info;
			godot_signal_argument attrib = p_signal->args[i];

			String *name = (String *)&attrib.name;
			info.name = *name;
			info.type = (Variant::Type)attrib.type;
			info.hint = (PropertyHint)attrib.hint;
			info.hint_string = *(String *)&attrib.hint_string;
			info.usage = attrib.usage;

			arguments.push_back(info);
		}

		signal.signal.arguments = arguments;
	}

	{
		Vector<Variant> default_arguments;
		for (int i = 0; i < p_signal->num_default_args; i++) {
			Variant *v;
			godot_signal_argument attrib = p_signal->args[i];

			v = (Variant *)&attrib.default_value;

			default_arguments.push_back(*v);
		}

		signal.signal.default_arguments = default_arguments;
	}

	scripts[p_name]->signals_.insert(*(String *)&p_signal->name, signal);
}

DLScriptData *DLLibrary::get_script_data(const StringName p_name) {

	if (!scripts.has(p_name)) {
		if (DLScriptLanguage::get_singleton()->is_library_initialized(library_path)) {
			_update_library(*DLScriptLanguage::get_singleton()->get_library_dllibrary(library_path));
		}
		ERR_FAIL_COND_V(!scripts.has(p_name), NULL);
	}

	return scripts[p_name];
}

bool DLLibrary::_set(const StringName &p_name, const Variant &p_value) {
	String name = p_name;
	if (name.begins_with("platform/")) {
		set_platform_file(name.get_slice("/", 1), p_value);
		return true;
	}
	return false;
}

bool DLLibrary::_get(const StringName &p_name, Variant &r_ret) const {
	String name = p_name;
	if (name.begins_with("platform/")) {
		r_ret = get_platform_file(name.get_slice("/", 1));
		return true;
	}
	return false;
}

void DLLibrary::_get_property_list(List<PropertyInfo> *p_list) const {
	char **platform_info = (char **)_dl_platforms_info;

	Set<String> registered_platform_names;
	{
		List<StringName> ep;
		// ep.push_back("X11");
		// EditorImportExport::get_singleton()->get_export_platforms(&ep);
		for (List<StringName>::Element *E = ep.front(); E; E = E->next()) {
			registered_platform_names.insert(String(E->get()).to_lower());
		}
	}

	while (*platform_info) {
		String platform_info_string(*platform_info);
		String fallback_platform_key = platform_info_string.get_slicec('|', 0);
		String platform_key = platform_info_string.get_slicec('|', 1);
		String platform_extension = platform_info_string.get_slicec('|', 2);
		String platform_name = platform_info_string.get_slicec('|', 3);

		registered_platform_names.erase(platform_name);

		if (fallback_platform_key.empty()) {
			p_list->push_back(PropertyInfo(Variant::STRING, "platform/" + platform_key, PROPERTY_HINT_FILE, "*." + platform_extension));

		} else {
			if (platform_files.has(platform_key)) {
				p_list->push_back(PropertyInfo(Variant::STRING, "platform/" + platform_key, PROPERTY_HINT_FILE, "*." + platform_extension, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CHECKABLE | PROPERTY_USAGE_CHECKED));
			} else {
				p_list->push_back(PropertyInfo(Variant::STRING, "platform/" + platform_key, PROPERTY_HINT_FILE, "*." + platform_extension, PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_CHECKABLE));
			}
		}
		platform_info++;
	}

	while (registered_platform_names.size()) {
		const String platform_name = registered_platform_names.front()->get();
		registered_platform_names.erase(platform_name);
		p_list->push_back(PropertyInfo(Variant::STRING, "platform/" + platform_name.to_lower(), PROPERTY_HINT_FILE, "*"));
	}
}

void DLLibrary::_notification(int what) {
	// TODO
}

void DLLibrary::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_platform_file", "platform", "file"), &DLLibrary::set_platform_file);
	ClassDB::bind_method(D_METHOD("get_platform_file", "platform"), &DLLibrary::get_platform_file);
}

DLLibrary::DLLibrary() {
	library_handle = NULL;
}

DLLibrary::~DLLibrary() {

	for (Map<StringName, DLScriptData *>::Element *E = scripts.front(); E; E = E->next()) {
		for (Map<StringName, DLScriptData::Method>::Element *M = E->get()->methods.front(); M; M = M->next()) {
			if (M->get().method.free_func) {
				M->get().method.free_func(M->get().method.method_data);
			}
		}
		memdelete(E->get());
	}

	if (library_handle) {
		bool in_editor = false;
#ifdef TOOLS_ENABLED
		in_editor = !ScriptServer::is_scripting_enabled();
#endif
		_free_handle(in_editor);
	}
}

// Instance

bool DLInstance::set(const StringName &p_name, const Variant &p_value) {
	if (!script->script_data)
		return false;
	if (script->script_data->properties.has(p_name)) {
		script->script_data->properties[p_name].setter.set_func((godot_object *)owner, script->script_data->properties[p_name].setter.method_data, userdata, *(godot_variant *)&p_value);
		return true;
	}
	return false;
}

bool DLInstance::get(const StringName &p_name, Variant &r_ret) const {
	if (!script->script_data)
		return false;
	if (script->script_data->properties.has(p_name)) {
		godot_variant value = script->script_data->properties[p_name].getter.get_func((godot_object *)owner, script->script_data->properties[p_name].getter.method_data, userdata);
		r_ret = *(Variant *)&value;
		return true;
	}
	return false;
}

void DLInstance::get_property_list(List<PropertyInfo> *p_properties) const {
	script->get_script_property_list(p_properties);
	// TODO: dynamic properties
}

Variant::Type DLInstance::get_property_type(const StringName &p_name, bool *r_is_valid) const {
	if (script->script_data->properties.has(p_name)) {
		*r_is_valid = true;
		return script->script_data->properties[p_name].info.type;
	}
	*r_is_valid = false;
	return Variant::NIL;
}

void DLInstance::get_method_list(List<MethodInfo> *p_list) const {
	script->get_script_method_list(p_list);
}

bool DLInstance::has_method(const StringName &p_method) const {
	return script->has_method(p_method);
}

Variant DLInstance::call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
	// TODO: validated methods & errors

	DLScriptData *data_ptr = script->script_data;
	while (data_ptr) {
		Map<StringName, DLScriptData::Method>::Element *E = data_ptr->methods.find(p_method);
		if (E) {
			godot_variant result = E->get().method.method((godot_object *)owner, E->get().method.method_data, userdata, p_argcount, (godot_variant **)p_args);
			return *(Variant *)&result;
		}
		data_ptr = data_ptr->base_data;
	}
	r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
	return Variant();
}

void DLInstance::call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount) {
	// TODO: validated methods & errors

	DLScriptData *data_ptr = script->script_data;
	while (data_ptr) {
		Map<StringName, DLScriptData::Method>::Element *E = data_ptr->methods.find(p_method);
		if (E) {
			E->get().method.method((godot_object *)owner, E->get().method.method_data, userdata, p_argcount, (godot_variant **)p_args);
		}
		data_ptr = data_ptr->base_data;
	}
}

void DLInstance::_ml_call_reversed(DLScriptData *data_ptr, const StringName &p_method, const Variant **p_args, int p_argcount) {
	// TODO: validated methods & errors

	if (data_ptr->base_data)
		_ml_call_reversed(data_ptr->base_data, p_method, p_args, p_argcount);

	// Variant::CallError ce;

	Map<StringName, DLScriptData::Method>::Element *E = data_ptr->methods.find(p_method);
	if (E) {
		E->get().method.method((godot_object *)owner, E->get().method.method_data, userdata, p_argcount, (godot_variant **)p_args);
	}
}

void DLInstance::call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount) {
	if (script.ptr() && script->script_data) {
		_ml_call_reversed(script->script_data, p_method, p_args, p_argcount);
	}
}

void DLInstance::notification(int p_notification) {
	Variant value = p_notification;
	const Variant *args[1] = { &value };
	call_multilevel(DLScriptLanguage::singleton->strings._notification, args, 1);
}

Ref<Script> DLInstance::get_script() const {
	return script;
}

ScriptLanguage *DLInstance::get_language() {
	return DLScriptLanguage::singleton;
}

ScriptInstance::RPCMode DLInstance::get_rpc_mode(const StringName &p_method) const {
	DLScriptData::Method m = script->script_data->methods[p_method];
	switch (m.rpc_mode) {
		case GODOT_METHOD_RPC_MODE_DISABLED:
			return RPC_MODE_DISABLED;
		case GODOT_METHOD_RPC_MODE_REMOTE:
			return RPC_MODE_REMOTE;
		case GODOT_METHOD_RPC_MODE_SYNC:
			return RPC_MODE_SYNC;
		case GODOT_METHOD_RPC_MODE_MASTER:
			return RPC_MODE_MASTER;
		case GODOT_METHOD_RPC_MODE_SLAVE:
			return RPC_MODE_SLAVE;
		default:
			return RPC_MODE_DISABLED;
	}
}

ScriptInstance::RPCMode DLInstance::get_rset_mode(const StringName &p_variable) const {
	DLScriptData::Property p = script->script_data->properties[p_variable];
	switch (p.rset_mode) {
		case GODOT_METHOD_RPC_MODE_DISABLED:
			return RPC_MODE_DISABLED;
		case GODOT_METHOD_RPC_MODE_REMOTE:
			return RPC_MODE_REMOTE;
		case GODOT_METHOD_RPC_MODE_SYNC:
			return RPC_MODE_SYNC;
		case GODOT_METHOD_RPC_MODE_MASTER:
			return RPC_MODE_MASTER;
		case GODOT_METHOD_RPC_MODE_SLAVE:
			return RPC_MODE_SLAVE;
		default:
			return RPC_MODE_DISABLED;
	}
}

DLInstance::DLInstance() {
	owner = NULL;
	userdata = NULL;
}

DLInstance::~DLInstance() {
	if (script.is_valid()) {
		if (owner) {
			script->instances.erase(owner);
		}
		if (!script->script_data)
			return;
		script->script_data->destroy_func.destroy_func((godot_object *)owner, script->script_data->destroy_func.method_data, userdata);
		if (script->script_data->destroy_func.free_func)
			script->script_data->destroy_func.free_func(script->script_data->destroy_func.method_data);
		if (script->script_data->create_func.free_func)
			script->script_data->create_func.free_func(script->script_data->create_func.method_data);
	}
}

// Language

DLScriptLanguage *DLScriptLanguage::singleton = NULL;

String DLScriptLanguage::get_name() const {
	return "DLScript";
}

bool DLScriptLanguage::is_library_initialized(const String &p_path) {

	return initialized_libraries.has(p_path);
}

void DLScriptLanguage::set_library_initialized(const String &p_path, DLLibrary *p_dllibrary) {

	initialized_libraries[p_path] = p_dllibrary;
}

DLLibrary *DLScriptLanguage::get_library_dllibrary(const String &p_path) {

	return initialized_libraries[p_path];
}

void DLScriptLanguage::set_library_uninitialized(const String &p_path) {

	initialized_libraries.erase(p_path);
}

void DLScriptLanguage::init() {
	// TODO: Expose globals
	GLOBAL_DEF("dlscript/default_dllibrary", "");
	PropertyInfo prop_info(Variant::STRING, "dlscript/default_dllibrary", PROPERTY_HINT_FILE, "tres,res,dllib");
	GlobalConfig::get_singleton()->set_custom_property_info("dlscript/default_dllibrary", prop_info);

// generate bindings
#if defined(TOOLS_ENABLED) && defined(DEBUG_METHODS_ENABLED)

	List<String> args = OS::get_singleton()->get_cmdline_args();

	List<String>::Element *E = args.find("--dlscript-generate-json-api");

	if (E && E->next()) {
		if (generate_c_api(E->next()->get()) != OK) {
			ERR_PRINT("Failed to generate C API\n");
		}
	}
#endif
}

String DLScriptLanguage::get_type() const {
	return "DLScript";
}

String DLScriptLanguage::get_extension() const {
	return "dl";
}

Error DLScriptLanguage::execute_file(const String &p_path) {
	return OK; // ??
}

void DLScriptLanguage::finish() {
	// cleanup is for noobs
}

// scons doesn't want to link in the api source so we need to call a dummy function to cause it to link
extern "C" void _api_anchor();

void DLScriptLanguage::_compile_dummy_for_the_api() {
	_api_anchor();
}

Ref<Script> DLScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	DLScript *src = memnew(DLScript);
	src->set_script_name(p_class_name);
	return Ref<DLScript>(src);
}

bool DLScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions) const {
	return false; // TODO
}

Script *DLScriptLanguage::create_script() const {
	DLScript *scr = memnew(DLScript);
	return scr;
}

bool DLScriptLanguage::has_named_classes() const {
	return true;
}

int DLScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	return -1; // No source code!
}

String DLScriptLanguage::make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const {
	return ""; // No source code!
}

void DLScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
	// TODO TODO TODO
}

// TODO: Any debugging? (research)
String DLScriptLanguage::debug_get_error() const {
	return "";
}

int DLScriptLanguage::debug_get_stack_level_count() const {
	return 1; // ?
}

int DLScriptLanguage::debug_get_stack_level_line(int p_level) const {
	return -1;
}

String DLScriptLanguage::debug_get_stack_level_function(int p_level) const {
	return "[native code]"; // ?
}

String DLScriptLanguage::debug_get_stack_level_source(int p_level) const {
	return "";
}

void DLScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}

void DLScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}

String DLScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	return ""; // ??
}

void DLScriptLanguage::reload_all_scripts() {
	// @Todo
}

void DLScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
	// @Todo
	OS::get_singleton()->print("reload tool scripts\n");
}

void DLScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("dl"); // Container file format
}

void DLScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
}

void DLScriptLanguage::get_public_constants(List<Pair<String, Variant> > *p_constants) const {
}

// TODO: all profilling
void DLScriptLanguage::profiling_start() {
}

void DLScriptLanguage::profiling_stop() {
}

int DLScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

int DLScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	return 0;
}

void DLScriptLanguage::frame() {
}

String DLScriptLanguage::get_init_symbol_name() {
	return "godot_dlscript_init"; // TODO: Maybe make some internal function which would do the actual stuff
}

String DLScriptLanguage::get_terminate_symbol_name() {
	return "godot_dlscript_terminate";
}

DLScriptLanguage::DLScriptLanguage() {
	ERR_FAIL_COND(singleton);
	strings._notification = StringName("_notification");
	singleton = this;
	initialized_libraries = Map<String, DLLibrary *>();
}

DLScriptLanguage::~DLScriptLanguage() {
	singleton = NULL;
}

RES ResourceFormatLoaderDLScript::load(const String &p_path, const String &p_original_path, Error *r_error) {
	ResourceFormatLoaderText rsflt;
	return rsflt.load(p_path, p_original_path, r_error);
}

void ResourceFormatLoaderDLScript::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("dl");
}
bool ResourceFormatLoaderDLScript::handles_type(const String &p_type) const {
	return (p_type == "Script" || p_type == "DLScript");
}
String ResourceFormatLoaderDLScript::get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "dl")
		return "DLScript";
	return "";
}

Error ResourceFormatSaverDLScript::save(const String &p_path, const RES &p_resource, uint32_t p_flags) {
	ResourceFormatSaverText rfst;
	return rfst.save(p_path, p_resource, p_flags);
}

bool ResourceFormatSaverDLScript::recognize(const RES &p_resource) const {
	return p_resource->cast_to<DLScript>() != NULL;
}

void ResourceFormatSaverDLScript::get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const {
	if (p_resource->cast_to<DLScript>()) {
		p_extensions->push_back("dl");
	}
}
