/*************************************************************************/
/*  script_language.h                                                    */
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
#ifndef SCRIPT_LANGUAGE_H
#define SCRIPT_LANGUAGE_H

#include "map.h"
#include "pair.h"
#include "resource.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ScriptLanguage;

typedef void (*ScriptEditRequestFunction)(const String &p_path);

class ScriptServer {
	enum {

		MAX_LANGUAGES = 16
	};

	static ScriptLanguage *_languages[MAX_LANGUAGES];
	static int _language_count;
	static bool scripting_enabled;
	static bool reload_scripts_on_save;

public:
	static ScriptEditRequestFunction edit_request_func;

	static void set_scripting_enabled(bool p_enabled);
	static bool is_scripting_enabled();
	static int get_language_count();
	static ScriptLanguage *get_language(int p_idx);
	static void register_language(ScriptLanguage *p_language);
	static void unregister_language(ScriptLanguage *p_language);

	static void set_reload_scripts_on_save(bool p_enable);
	static bool is_reload_scripts_on_save_enabled();

	static void thread_enter();
	static void thread_exit();

	static void init_languages();
};

class ScriptInstance;
class PlaceHolderScriptInstance;

class Script : public Resource {

	GDCLASS(Script, Resource);
	OBJ_SAVE_TYPE(Script);

protected:
	virtual bool editor_can_reload_from_file() { return false; } // this is handled by editor better
	void _notification(int p_what);
	static void _bind_methods();

	friend class PlaceHolderScriptInstance;
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder) {}

public:
	virtual bool can_instance() const = 0;

	virtual Ref<Script> get_base_script() const = 0; //for script inheritance

	virtual StringName get_instance_base_type() const = 0; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance *instance_create(Object *p_this) = 0;
	virtual bool instance_has(const Object *p_this) const = 0;

	virtual bool has_source_code() const = 0;
	virtual String get_source_code() const = 0;
	virtual void set_source_code(const String &p_code) = 0;
	virtual Error reload(bool p_keep_state = false) = 0;

	virtual bool has_method(const StringName &p_method) const = 0;
	virtual MethodInfo get_method_info(const StringName &p_method) const = 0;

	virtual bool is_tool() const = 0;

	virtual String get_node_type() const = 0;

	virtual ScriptLanguage *get_language() const = 0;

	virtual bool has_script_signal(const StringName &p_signal) const = 0;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const = 0;

	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const = 0;

	virtual void update_exports() {} //editor tool
	virtual void get_script_method_list(List<MethodInfo> *p_list) const = 0;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const = 0;

	virtual int get_member_line(const StringName &p_member) const { return 0; }

	Script() {}
};

class ScriptInstance {
public:
	virtual bool set(const StringName &p_name, const Variant &p_value) = 0;
	virtual bool get(const StringName &p_name, Variant &r_ret) const = 0;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const = 0;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = NULL) const = 0;

	virtual void get_property_state(List<Pair<StringName, Variant> > &state);

	virtual void get_method_list(List<MethodInfo> *p_list) const = 0;
	virtual bool has_method(const StringName &p_method) const = 0;
	virtual Variant call(const StringName &p_method, VARIANT_ARG_LIST);
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) = 0;
	virtual void call_multilevel(const StringName &p_method, VARIANT_ARG_LIST);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void notification(int p_notification) = 0;

	//this is used by script languages that keep a reference counter of their own
	//you can make make Ref<> not die when it reaches zero, so deleting the reference
	//depends entirely from the script

	virtual void refcount_incremented() {}
	virtual bool refcount_decremented() { return true; } //return true if it can die

	virtual Ref<Script> get_script() const = 0;

	virtual bool is_placeholder() const { return false; }

	enum RPCMode {
		RPC_MODE_DISABLED,
		RPC_MODE_REMOTE,
		RPC_MODE_SYNC,
		RPC_MODE_MASTER,
		RPC_MODE_SLAVE,
	};

	virtual RPCMode get_rpc_mode(const StringName &p_method) const = 0;
	virtual RPCMode get_rset_mode(const StringName &p_variable) const = 0;

	virtual ScriptLanguage *get_language() = 0;
	virtual ~ScriptInstance();
};

class ScriptCodeCompletionCache {

	static ScriptCodeCompletionCache *singleton;

public:
	virtual RES get_cached_resource(const String &p_path) = 0;

	static ScriptCodeCompletionCache *get_sigleton() { return singleton; }

	ScriptCodeCompletionCache();
};

class ScriptLanguage {
public:
	virtual String get_name() const = 0;

	/* LANGUAGE FUNCTIONS */
	virtual void init() = 0;
	virtual String get_type() const = 0;
	virtual String get_extension() const = 0;
	virtual Error execute_file(const String &p_path) = 0;
	virtual void finish() = 0;

	/* EDITOR FUNCTIONS */
	virtual void get_reserved_words(List<String> *p_words) const = 0;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const = 0;
	virtual void get_string_delimiters(List<String> *p_delimiters) const = 0;
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const = 0;
	virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path = "", List<String> *r_functions = NULL) const = 0;
	virtual Script *create_script() const = 0;
	virtual bool has_named_classes() const = 0;
	virtual int find_function(const String &p_function, const String &p_code) const = 0;
	virtual String make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const = 0;

	virtual Error complete_code(const String &p_code, const String &p_base_path, Object *p_owner, List<String> *r_options, String &r_call_hint) { return ERR_UNAVAILABLE; }

	struct LookupResult {
		enum Type {
			RESULT_SCRIPT_LOCATION,
			RESULT_CLASS,
			RESULT_CLASS_CONSTANT,
			RESULT_CLASS_PROPERTY,
			RESULT_CLASS_METHOD
		};
		Type type;
		Ref<Script> script;
		String class_name;
		String class_member;
		int location;
	};

	virtual Error lookup_code(const String &p_code, const String &p_symbol, const String &p_base_path, Object *p_owner, LookupResult &r_result) { return ERR_UNAVAILABLE; }

	virtual void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const = 0;
	virtual void add_global_constant(const StringName &p_variable, const Variant &p_value) = 0;

	/* MULTITHREAD FUNCTIONS */

	//some VMs need to be notified of thread creation/exiting to allocate a stack
	virtual void thread_enter() {}
	virtual void thread_exit() {}

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const = 0;
	virtual int debug_get_stack_level_count() const = 0;
	virtual int debug_get_stack_level_line(int p_level) const = 0;
	virtual String debug_get_stack_level_function(int p_level) const = 0;
	virtual String debug_get_stack_level_source(int p_level) const = 0;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) = 0;
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) = 0;
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1) = 0;
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems = -1, int p_max_depth = -1) = 0;

	struct StackInfo {
		Ref<Script> script;
		int line;
	};

	virtual Vector<StackInfo> debug_get_current_stack_info() { return Vector<StackInfo>(); }

	virtual void reload_all_scripts() = 0;
	virtual void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload) = 0;
	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const = 0;
	virtual void get_public_functions(List<MethodInfo> *p_functions) const = 0;
	virtual void get_public_constants(List<Pair<String, Variant> > *p_constants) const = 0;

	struct ProfilingInfo {
		StringName signature;
		uint64_t call_count;
		uint64_t total_time;
		uint64_t self_time;
	};

	virtual void profiling_start() = 0;
	virtual void profiling_stop() = 0;

	virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) = 0;
	virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) = 0;

	virtual void frame();

	virtual ~ScriptLanguage(){};
};

extern uint8_t script_encryption_key[32];

class PlaceHolderScriptInstance : public ScriptInstance {

	Object *owner;
	List<PropertyInfo> properties;
	Map<StringName, Variant> values;
	ScriptLanguage *language;
	Ref<Script> script;

public:
	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = NULL) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const {}
	virtual bool has_method(const StringName &p_method) const { return false; }
	virtual Variant call(const StringName &p_method, VARIANT_ARG_LIST) { return Variant(); }
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error) {
		r_error.error = Variant::CallError::CALL_ERROR_INVALID_METHOD;
		return Variant();
	}
	//virtual void call_multilevel(const StringName& p_method,VARIANT_ARG_LIST) { return Variant(); }
	//virtual void call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error) { return Variant(); }
	virtual void notification(int p_notification) {}

	virtual Ref<Script> get_script() const { return script; }

	virtual ScriptLanguage *get_language() { return language; }

	Object *get_owner() { return owner; }

	void update(const List<PropertyInfo> &p_properties, const Map<StringName, Variant> &p_values); //likely changed in editor

	virtual bool is_placeholder() const { return true; }

	virtual RPCMode get_rpc_mode(const StringName &p_method) const { return RPC_MODE_DISABLED; }
	virtual RPCMode get_rset_mode(const StringName &p_variable) const { return RPC_MODE_DISABLED; }

	PlaceHolderScriptInstance(ScriptLanguage *p_language, Ref<Script> p_script, Object *p_owner);
	~PlaceHolderScriptInstance();
};

class ScriptDebugger {

	int lines_left;
	int depth;

	static ScriptDebugger *singleton;
	Map<int, Set<StringName> > breakpoints;

	ScriptLanguage *break_lang;

public:
	typedef void (*RequestSceneTreeMessageFunc)(void *);

	struct LiveEditFuncs {

		void *udata;
		void (*node_path_func)(void *, const NodePath &p_path, int p_id);
		void (*res_path_func)(void *, const String &p_path, int p_id);

		void (*node_set_func)(void *, int p_id, const StringName &p_prop, const Variant &p_value);
		void (*node_set_res_func)(void *, int p_id, const StringName &p_prop, const String &p_value);
		void (*node_call_func)(void *, int p_id, const StringName &p_method, VARIANT_ARG_DECLARE);
		void (*res_set_func)(void *, int p_id, const StringName &p_prop, const Variant &p_value);
		void (*res_set_res_func)(void *, int p_id, const StringName &p_prop, const String &p_value);
		void (*res_call_func)(void *, int p_id, const StringName &p_method, VARIANT_ARG_DECLARE);
		void (*root_func)(void *, const NodePath &p_scene_path, const String &p_scene_from);

		void (*tree_create_node_func)(void *, const NodePath &p_parent, const String &p_type, const String &p_name);
		void (*tree_instance_node_func)(void *, const NodePath &p_parent, const String &p_path, const String &p_name);
		void (*tree_remove_node_func)(void *, const NodePath &p_at);
		void (*tree_remove_and_keep_node_func)(void *, const NodePath &p_at, ObjectID p_keep_id);
		void (*tree_restore_node_func)(void *, ObjectID p_id, const NodePath &p_at, int p_at_pos);
		void (*tree_duplicate_node_func)(void *, const NodePath &p_at, const String &p_new_name);
		void (*tree_reparent_node_func)(void *, const NodePath &p_at, const NodePath &p_new_place, const String &p_new_name, int p_at_pos);
	};

	_FORCE_INLINE_ static ScriptDebugger *get_singleton() { return singleton; }
	void set_lines_left(int p_left);
	int get_lines_left() const;

	void set_depth(int p_depth);
	int get_depth() const;

	String breakpoint_find_source(const String &p_source) const;
	void insert_breakpoint(int p_line, const StringName &p_source);
	void remove_breakpoint(int p_line, const StringName &p_source);
	bool is_breakpoint(int p_line, const StringName &p_source) const;
	bool is_breakpoint_line(int p_line) const;
	void clear_breakpoints();

	virtual void debug(ScriptLanguage *p_script, bool p_can_continue = true) = 0;
	virtual void idle_poll();
	virtual void line_poll();

	void set_break_language(ScriptLanguage *p_lang);
	ScriptLanguage *get_break_language() const;

	virtual void send_message(const String &p_message, const Array &p_args) = 0;

	virtual bool is_remote() const { return false; }
	virtual void request_quit() {}

	virtual void set_request_scene_tree_message_func(RequestSceneTreeMessageFunc p_func, void *p_udata) {}
	virtual void set_live_edit_funcs(LiveEditFuncs *p_funcs) {}

	virtual bool is_profiling() const = 0;
	virtual void add_profiling_frame_data(const StringName &p_name, const Array &p_data) = 0;
	virtual void profiling_start() = 0;
	virtual void profiling_end() = 0;
	virtual void profiling_set_frame_times(float p_frame_time, float p_idle_time, float p_fixed_time, float p_fixed_frame_time) = 0;

	ScriptDebugger();
	virtual ~ScriptDebugger() { singleton = NULL; }
};

#endif
