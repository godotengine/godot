/*************************************************************************/
/*  multiscript.h                                                        */
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
#ifndef MULTISCRIPT_H
#define MULTISCRIPT_H

#include "os/thread_safe.h"
#include "script_language.h"

class MultiScript;

class Multi : public Object {
	GDCLASS(Multi, Object)

	friend class MultiScript;

	Object *real_owner;

public:
	static void _bind_methods();

	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);
};

class MultiScriptInstance : public ScriptInstance {
	friend class MultiScript;
	mutable Vector<ScriptInstance *> instances;
	Object *object;
	mutable MultiScript *owner;

public:
	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName &p_method) const;
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void notification(int p_notification);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();
	virtual ~MultiScriptInstance();

	// ScriptInstance interface
public:
	Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const;
	RPCMode get_rpc_mode(const StringName &p_method) const;
	RPCMode get_rset_mode(const StringName &p_variable) const;
};

class MultiScript : public Script {

	_THREAD_SAFE_CLASS_
	friend class MultiScriptInstance;
	friend class MultiScriptLanguage;
	GDCLASS(MultiScript, Script)

	StringName base_class_name;

	Vector<Ref<Script> > scripts;
	Vector<Multi *> script_instances;

	Map<Object *, MultiScriptInstance *> instances;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	static void _bind_methods();

public:
	void remove_instance(Object *p_object);
	virtual bool can_instance() const;

	virtual StringName get_instance_base_type() const;
	virtual ScriptInstance *instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String &p_code);
	virtual Error reload(bool p_keep_state = false);

	virtual bool is_tool() const;

	virtual String get_node_type() const;

	void set_script(int p_idx, const Ref<Script> &p_script);
	Ref<Script> get_script(int p_idx) const;
	void remove_script(int p_idx);
	void add_script(const Ref<Script> &p_script);

	virtual ScriptLanguage *get_language() const;

	MultiScript();
	~MultiScript();

	virtual Ref<Script> get_base_script() const;
	virtual bool has_method(const StringName &p_method) const;
	virtual MethodInfo get_method_info(const StringName &p_method) const;
	virtual bool has_script_signal(const StringName &p_signal) const;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const;
	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const;
	virtual void get_script_method_list(List<MethodInfo> *p_list) const;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const;
	virtual void update_exports();
};

class MultiScriptLanguage : public ScriptLanguage {

	static MultiScriptLanguage *singleton;

public:
	static _FORCE_INLINE_ MultiScriptLanguage *get_singleton();
	virtual String get_name() const;

	/* LANGUAGE FUNCTIONS */
	virtual void init();
	virtual String get_type() const;
	virtual String get_extension() const;
	virtual Error execute_file(const String &p_path);
	virtual void finish();

	/* EDITOR FUNCTIONS */
	virtual void get_reserved_words(List<String> *p_words) const;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const;
	virtual void get_string_delimiters(List<String> *p_delimiters) const;
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const;
	virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path = "", List<String> *r_fn = NULL) const;
	virtual Script *create_script() const;
	virtual bool has_named_classes() const;
	virtual int find_function(const String &p_function, const String &p_code) const;
	virtual String make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const;

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const;
	virtual int debug_get_stack_level_count() const;
	virtual int debug_get_stack_level_line(int p_level) const;
	virtual String debug_get_stack_level_function(int p_level) const;
	virtual String debug_get_stack_level_source(int p_level) const;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1);
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1);
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1);
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems = -1, int p_max_depth = -1);

	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_public_functions(List<MethodInfo> *p_functions) const;

	MultiScriptLanguage();
	virtual ~MultiScriptLanguage();

	// ScriptLanguage interface
public:
	void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const;
	void add_global_constant(const StringName &p_variable, const Variant &p_value);
	void reload_all_scripts();
	void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload);
	void get_public_constants(List<Pair<String, Variant> > *p_constants) const;
	void profiling_start();
	void profiling_stop();
	int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max);
	int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max);
};

#endif // MULTISCRIPT_H
