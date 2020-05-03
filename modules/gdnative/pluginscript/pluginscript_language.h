/*************************************************************************/
/*  pluginscript_language.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef PLUGINSCRIPT_LANGUAGE_H
#define PLUGINSCRIPT_LANGUAGE_H

// Godot imports
#include "core/io/resource_loader.h"
#include "core/io/resource_saver.h"
#include "core/map.h"
#include "core/script_language.h"
#include "core/self_list.h"
// PluginScript imports
#include "pluginscript_loader.h"
#include <pluginscript/godot_pluginscript.h>

class PluginScript;
class PluginScriptInstance;

class PluginScriptLanguage : public ScriptLanguage {
	friend class PluginScript;
	friend class PluginScriptInstance;

	Ref<ResourceFormatLoaderPluginScript> _resource_loader;
	Ref<ResourceFormatSaverPluginScript> _resource_saver;
	const godot_pluginscript_language_desc _desc;
	godot_pluginscript_language_data *_data;

	Mutex _lock;
	SelfList<PluginScript>::List _script_list;

public:
	virtual String get_name() const override;

	_FORCE_INLINE_ Ref<ResourceFormatLoaderPluginScript> get_resource_loader() { return _resource_loader; }
	_FORCE_INLINE_ Ref<ResourceFormatSaverPluginScript> get_resource_saver() { return _resource_saver; }

	/* LANGUAGE FUNCTIONS */
	virtual void init() override;
	virtual String get_type() const override;
	virtual String get_extension() const override;
	virtual Error execute_file(const String &p_path) override;
	virtual void finish() override;

	/* EDITOR FUNCTIONS */
	virtual void get_reserved_words(List<String> *p_words) const override;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const override;
	virtual void get_string_delimiters(List<String> *p_delimiters) const override;
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const override;
	virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path = "", List<String> *r_functions = nullptr, List<ScriptLanguage::Warning> *r_warnings = nullptr, Set<int> *r_safe_lines = nullptr) const override;
	virtual Script *create_script() const override;
	virtual bool has_named_classes() const override;
	virtual bool supports_builtin_mode() const override;
	virtual bool can_inherit_from_file() override { return true; }
	virtual int find_function(const String &p_function, const String &p_code) const override;
	virtual String make_function(const String &p_class, const String &p_name, const PackedStringArray &p_args) const override;
	virtual Error complete_code(const String &p_code, const String &p_path, Object *p_owner, List<ScriptCodeCompletionOption> *r_options, bool &r_force, String &r_call_hint) override;
	virtual void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const override;
	virtual void add_global_constant(const StringName &p_variable, const Variant &p_value) override;

	/* MULTITHREAD FUNCTIONS */

	//some VMs need to be notified of thread creation/exiting to allocate a stack
	// void thread_enter() {}
	// void thread_exit() {}

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const override;
	virtual int debug_get_stack_level_count() const override;
	virtual int debug_get_stack_level_line(int p_level) const override;
	virtual String debug_get_stack_level_function(int p_level) const override;
	virtual String debug_get_stack_level_source(int p_level) const override;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) override;
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems = -1, int p_max_depth = -1) override;

	// virtual Vector<StackInfo> debug_get_current_stack_info() { return Vector<StackInfo>(); }

	virtual void reload_all_scripts() override;
	virtual void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) override;

	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual void get_public_functions(List<MethodInfo> *p_functions) const override;
	virtual void get_public_constants(List<Pair<String, Variant>> *p_constants) const override;

	virtual void profiling_start() override;
	virtual void profiling_stop() override;

	virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) override;
	virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) override;

	virtual void frame() override;

	void lock();
	void unlock();

	PluginScriptLanguage(const godot_pluginscript_language_desc *desc);
	virtual ~PluginScriptLanguage();
};

#endif // PLUGINSCRIPT_LANGUAGE_H
