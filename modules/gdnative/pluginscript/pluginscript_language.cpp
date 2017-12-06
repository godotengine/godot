/*************************************************************************/
/*  pluginscript_language.cpp                                            */
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
#include "core/os/os.h"
#include "core/project_settings.h"
// PluginScript imports
#include "pluginscript_language.h"
#include "pluginscript_script.h"

String PluginScriptLanguage::get_name() const {
	return String(_desc.name);
}

void PluginScriptLanguage::init() {
	_data = _desc.init();
}

String PluginScriptLanguage::get_type() const {
	return String(_desc.type);
}

String PluginScriptLanguage::get_extension() const {
	return String(_desc.extension);
}

Error PluginScriptLanguage::execute_file(const String &p_path) {
	// TODO: pretty sure this method is totally deprecated and should be removed...
	return OK;
}

void PluginScriptLanguage::finish() {
	_desc.finish(_data);
}

/* EDITOR FUNCTIONS */

void PluginScriptLanguage::get_reserved_words(List<String> *p_words) const {
	if (_desc.reserved_words) {
		const char **w = _desc.reserved_words;
		while (*w) {
			p_words->push_back(*w);
			w++;
		}
	}
}

void PluginScriptLanguage::get_comment_delimiters(List<String> *p_delimiters) const {
	if (_desc.comment_delimiters) {
		const char **w = _desc.comment_delimiters;
		while (*w) {
			p_delimiters->push_back(*w);
			w++;
		}
	}
}

void PluginScriptLanguage::get_string_delimiters(List<String> *p_delimiters) const {
	if (_desc.string_delimiters) {
		const char **w = _desc.string_delimiters;
		while (*w) {
			p_delimiters->push_back(*w);
			w++;
		}
	}
}

Ref<Script> PluginScriptLanguage::get_template(const String &p_class_name, const String &p_base_class_name) const {
	Script *ns = create_script();
	Ref<Script> script = Ref<Script>(ns);
	if (_desc.get_template_source_code) {
		godot_string src = _desc.get_template_source_code(_data, (godot_string *)&p_class_name, (godot_string *)&p_base_class_name);
		script->set_source_code(*(String *)&src);
	}
	return script;
}

bool PluginScriptLanguage::validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions) const {
	PoolStringArray functions;
	if (_desc.validate) {
		bool ret = _desc.validate(
				_data,
				(godot_string *)&p_script,
				&r_line_error,
				&r_col_error,
				(godot_string *)&r_test_error,
				(godot_string *)&p_path,
				(godot_pool_string_array *)&functions);
		for (int i = 0; i < functions.size(); i++) {
			r_functions->push_back(functions[i]);
		}
		return ret;
	}
	return true;
}

Script *PluginScriptLanguage::create_script() const {
	PluginScript *script = memnew(PluginScript());
	// I'm hurting kittens doing this I guess...
	script->init(const_cast<PluginScriptLanguage *>(this));
	return script;
}

bool PluginScriptLanguage::has_named_classes() const {
	return _desc.has_named_classes;
}

bool PluginScriptLanguage::supports_builtin_mode() const {
	return _desc.supports_builtin_mode;
}

int PluginScriptLanguage::find_function(const String &p_function, const String &p_code) const {
	if (_desc.find_function) {
		return _desc.find_function(_data, (godot_string *)&p_function, (godot_string *)&p_code);
	}
	return -1;
}

String PluginScriptLanguage::make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const {
	if (_desc.make_function) {
		godot_string tmp = _desc.make_function(_data, (godot_string *)&p_class, (godot_string *)&p_name, (godot_pool_string_array *)&p_args);
		String ret = *(String *)&tmp;
		godot_string_destroy(&tmp);
		return ret;
	}
	return String();
}

Error PluginScriptLanguage::complete_code(const String &p_code, const String &p_base_path, Object *p_owner, List<String> *r_options, bool &r_force, String &r_call_hint) {
	if (_desc.complete_code) {
		Array options;
		godot_error tmp = _desc.complete_code(
				_data,
				(godot_string *)&p_code,
				(godot_string *)&p_base_path,
				(godot_object *)p_owner,
				(godot_array *)&options,
				&r_force,
				(godot_string *)&r_call_hint);
		for (int i = 0; i < options.size(); i++) {
			r_options->push_back(String(options[i]));
		}
		Error err = *(Error *)tmp;
		return err;
	}
	return ERR_UNAVAILABLE;
}

void PluginScriptLanguage::auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {
	if (_desc.auto_indent_code) {
		_desc.auto_indent_code(_data, (godot_string *)&p_code, p_from_line, p_to_line);
	}
	return;
}

void PluginScriptLanguage::add_global_constant(const StringName &p_variable, const Variant &p_value) {
	const String variable = String(p_variable);
	_desc.add_global_constant(_data, (godot_string *)&variable, (godot_variant *)&p_value);
}

/* LOADER FUNCTIONS */

void PluginScriptLanguage::get_recognized_extensions(List<String> *p_extensions) const {
	for (int i = 0; _desc.recognized_extensions[i]; ++i) {
		p_extensions->push_back(String(_desc.recognized_extensions[i]));
	}
}

void PluginScriptLanguage::get_public_functions(List<MethodInfo> *p_functions) const {
	// TODO: provid this statically in `godot_pluginscript_language_desc` ?
	if (_desc.get_public_functions) {
		Array functions;
		_desc.get_public_functions(_data, (godot_array *)&functions);
		for (int i = 0; i < functions.size(); i++) {
			MethodInfo mi = MethodInfo::from_dict(functions[i]);
			p_functions->push_back(mi);
		}
	}
}

void PluginScriptLanguage::get_public_constants(List<Pair<String, Variant> > *p_constants) const {
	// TODO: provid this statically in `godot_pluginscript_language_desc` ?
	if (_desc.get_public_constants) {
		Dictionary constants;
		_desc.get_public_constants(_data, (godot_dictionary *)&constants);
		for (const Variant *key = constants.next(); key; key = constants.next(key)) {
			Variant value = constants[key];
			p_constants->push_back(Pair<String, Variant>(*key, value));
		}
	}
}

void PluginScriptLanguage::profiling_start() {
#ifdef DEBUG_ENABLED
	if (_desc.profiling_start) {
		lock();
		_desc.profiling_start(_data);
		unlock();
	}
#endif
}

void PluginScriptLanguage::profiling_stop() {
#ifdef DEBUG_ENABLED
	if (_desc.profiling_stop) {
		lock();
		_desc.profiling_stop(_data);
		unlock();
	}
#endif
}

int PluginScriptLanguage::profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int info_count = 0;
#ifdef DEBUG_ENABLED
	if (_desc.profiling_get_accumulated_data) {
		godot_pluginscript_profiling_data *info = (godot_pluginscript_profiling_data *)memalloc(
				sizeof(godot_pluginscript_profiling_data) * p_info_max);
		info_count = _desc.profiling_get_accumulated_data(_data, info, p_info_max);
		for (int i = 0; i < info_count; ++i) {
			p_info_arr[i].signature = *(StringName *)&info[i].signature;
			p_info_arr[i].call_count = info[i].call_count;
			p_info_arr[i].total_time = info[i].total_time;
			p_info_arr[i].self_time = info[i].self_time;
			godot_string_name_destroy(&info[i].signature);
		}
	}
#endif
	return info_count;
}

int PluginScriptLanguage::profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) {
	int info_count = 0;
#ifdef DEBUG_ENABLED
	if (_desc.profiling_get_frame_data) {
		godot_pluginscript_profiling_data *info = (godot_pluginscript_profiling_data *)memalloc(
				sizeof(godot_pluginscript_profiling_data) * p_info_max);
		info_count = _desc.profiling_get_frame_data(_data, info, p_info_max);
		for (int i = 0; i < info_count; ++i) {
			p_info_arr[i].signature = *(StringName *)&info[i].signature;
			p_info_arr[i].call_count = info[i].call_count;
			p_info_arr[i].total_time = info[i].total_time;
			p_info_arr[i].self_time = info[i].self_time;
			godot_string_name_destroy(&info[i].signature);
		}
	}
#endif
	return info_count;
}

void PluginScriptLanguage::frame() {
#ifdef DEBUG_ENABLED
	if (_desc.profiling_frame) {
		_desc.profiling_frame(_data);
	}
#endif
}

/* DEBUGGER FUNCTIONS */

String PluginScriptLanguage::debug_get_error() const {
	if (_desc.debug_get_error) {
		godot_string tmp = _desc.debug_get_error(_data);
		String ret = *(String *)&tmp;
		godot_string_destroy(&tmp);
		return ret;
	}
	return String("Nothing");
}

int PluginScriptLanguage::debug_get_stack_level_count() const {
	if (_desc.debug_get_stack_level_count) {
		return _desc.debug_get_stack_level_count(_data);
	}
	return 1;
}

int PluginScriptLanguage::debug_get_stack_level_line(int p_level) const {
	if (_desc.debug_get_stack_level_line) {
		return _desc.debug_get_stack_level_line(_data, p_level);
	}
	return 1;
}

String PluginScriptLanguage::debug_get_stack_level_function(int p_level) const {
	if (_desc.debug_get_stack_level_function) {
		godot_string tmp = _desc.debug_get_stack_level_function(_data, p_level);
		String ret = *(String *)&tmp;
		godot_string_destroy(&tmp);
		return ret;
	}
	return String("Nothing");
}

String PluginScriptLanguage::debug_get_stack_level_source(int p_level) const {
	if (_desc.debug_get_stack_level_source) {
		godot_string tmp = _desc.debug_get_stack_level_source(_data, p_level);
		String ret = *(String *)&tmp;
		godot_string_destroy(&tmp);
		return ret;
	}
	return String("Nothing");
}

void PluginScriptLanguage::debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_desc.debug_get_stack_level_locals) {
		PoolStringArray locals;
		Array values;
		_desc.debug_get_stack_level_locals(_data, p_level, (godot_pool_string_array *)&locals, (godot_array *)&values, p_max_subitems, p_max_depth);
		for (int i = 0; i < locals.size(); i++) {
			p_locals->push_back(locals[i]);
		}
		for (int i = 0; i < values.size(); i++) {
			p_values->push_back(values[i]);
		}
	}
}

void PluginScriptLanguage::debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_desc.debug_get_stack_level_members) {
		PoolStringArray members;
		Array values;
		_desc.debug_get_stack_level_members(_data, p_level, (godot_pool_string_array *)&members, (godot_array *)&values, p_max_subitems, p_max_depth);
		for (int i = 0; i < members.size(); i++) {
			p_members->push_back(members[i]);
		}
		for (int i = 0; i < values.size(); i++) {
			p_values->push_back(values[i]);
		}
	}
}

void PluginScriptLanguage::debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {
	if (_desc.debug_get_globals) {
		PoolStringArray locals;
		Array values;
		_desc.debug_get_globals(_data, (godot_pool_string_array *)&locals, (godot_array *)&values, p_max_subitems, p_max_depth);
		for (int i = 0; i < locals.size(); i++) {
			p_locals->push_back(locals[i]);
		}
		for (int i = 0; i < values.size(); i++) {
			p_values->push_back(values[i]);
		}
	}
}

String PluginScriptLanguage::debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) {
	if (_desc.debug_parse_stack_level_expression) {
		godot_string tmp = _desc.debug_parse_stack_level_expression(_data, p_level, (godot_string *)&p_expression, p_max_subitems, p_max_depth);
		String ret = *(String *)&tmp;
		godot_string_destroy(&tmp);
		return ret;
	}
	return String("Nothing");
}

void PluginScriptLanguage::reload_all_scripts() {
	// TODO
}

void PluginScriptLanguage::reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) {
#ifdef DEBUG_ENABLED
	lock();
	// TODO
	unlock();
#endif
}

void PluginScriptLanguage::lock() {
#ifndef NO_THREADS
	if (_lock) {
		_lock->lock();
	}
#endif
}

void PluginScriptLanguage::unlock() {
#ifndef NO_THREADS
	if (_lock) {
		_lock->unlock();
	}
#endif
}

PluginScriptLanguage::PluginScriptLanguage(const godot_pluginscript_language_desc *desc) :
		_desc(*desc) {
	_resource_loader = memnew(ResourceFormatLoaderPluginScript(this));
	_resource_saver = memnew(ResourceFormatSaverPluginScript(this));

// TODO: totally remove _lock attribute if NO_THREADS is set
#ifdef NO_THREADS
	_lock = NULL;
#else
	_lock = Mutex::create();
#endif
}

PluginScriptLanguage::~PluginScriptLanguage() {
	memdelete(_resource_loader);
	memdelete(_resource_saver);
#ifndef NO_THREADS
	if (_lock) {
		memdelete(_lock);
		_lock = NULL;
	}
#endif
}
