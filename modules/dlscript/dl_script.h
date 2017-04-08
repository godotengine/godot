/*************************************************************************/
/*  dl_script.h                                                          */
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
#ifndef DL_SCRIPT_H
#define DL_SCRIPT_H

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/thread_safe.h"
#include "resource.h"
#include "scene/main/node.h"
#include "script_language.h"
#include "self_list.h"

#include "godot.h"

#ifdef TOOLS_ENABLED
#define DLSCRIPT_EDITOR_FEATURES
#endif

class DLScriptData;
class DLLibrary;

struct NativeLibrary {
	StringName path;
	void *handle;

	DLLibrary *dllib;

	Map<StringName, DLScriptData *> scripts;

	static Error initialize(NativeLibrary *&p_native_lib, const StringName p_path);
	static Error terminate(NativeLibrary *&p_native_lib);
};

struct DLScriptData {
	/* typedef void* (InstanceFunc)(godot_object* instance);
	typedef void (DestroyFunc)(godot_object* instance,void* userdata);
	typedef godot_variant (MethodFunc)(godot_object *instance, void *userdata, void *method_data, int arg_count,godot_variant **args);
	typedef void (MethodDataFreeFunc)(void *method_data);
	typedef void (SetterFunc)(godot_object* instance,void* userdata,godot_variant value);
	typedef godot_variant (GetterFunc)(godot_object* instance,void* userdata);*/

	struct Method {
		godot_instance_method method;
		MethodInfo info;
		int rpc_mode;

		Method() {
		}
		Method(godot_instance_method p_method, MethodInfo p_info, int p_rpc_mode) {
			method = p_method;
			info = p_info;
			rpc_mode = p_rpc_mode;
		}
	};
	struct Property {
		godot_property_set_func setter;
		godot_property_get_func getter;
		PropertyInfo info;
		Variant default_value;
		int rset_mode;

		Property() {
		}
		Property(godot_property_set_func p_setter, godot_property_get_func p_getter) {
			setter = p_setter;
			getter = p_getter;
		}
		Property(godot_property_set_func p_setter, godot_property_get_func p_getter, PropertyInfo p_info, Variant p_default_value, int p_rset_mode) {
			setter = p_setter;
			getter = p_getter;
			info = p_info;
			default_value = p_default_value;
			rset_mode = p_rset_mode;
		}
	};

	struct Signal {
		MethodInfo signal;
	};

	Map<StringName, Method> methods;
	Map<StringName, Property> properties;
	Map<StringName, Signal> signals_; // QtCreator doesn't like the name signals
	StringName base;
	StringName base_native_type;
	DLScriptData *base_data;
	godot_instance_create_func create_func;
	godot_instance_destroy_func destroy_func;

	bool is_tool;

	DLScriptData() {
		base = StringName();
		base_data = NULL;
		is_tool = false;
	}
	DLScriptData(StringName p_base, godot_instance_create_func p_instance, godot_instance_destroy_func p_free) {
		base = p_base;
		base_data = NULL;
		create_func = p_instance;
		destroy_func = p_free;
		is_tool = false;
	}
};

class DLLibrary;

class DLScript : public Script {
	GDCLASS(DLScript, Script);

	Ref<DLLibrary> library;
	StringName script_name;
	StringName base_native_type;
	Set<Object *> instances;
	DLScriptData *script_data;

#ifdef TOOLS_ENABLED
	Set<PlaceHolderScriptInstance *> placeholders;
	void _update_placeholder(PlaceHolderScriptInstance *p_placeholder);
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);
#endif

	friend class DLInstance;
	friend class DLScriptLanguage;
	friend class DLReloadNode;
	friend class DLLibrary;

protected:
	static void _bind_methods();

public:
	virtual bool can_instance() const;

	virtual Ref<Script> get_base_script() const; //for script inheritance

	virtual StringName get_instance_base_type() const; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance *instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String &p_code) {}
	virtual Error reload(bool p_keep_state = false);

	virtual bool has_method(const StringName &p_method) const;
	virtual MethodInfo get_method_info(const StringName &p_method) const;

	virtual bool is_tool() const;

	virtual String get_node_type() const;

	virtual ScriptLanguage *get_language() const;

	virtual bool has_script_signal(const StringName &p_signal) const;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const;

	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const;

	virtual void update_exports() {} //editor tool
	virtual void get_script_method_list(List<MethodInfo> *p_list) const;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const;

	Ref<DLLibrary> get_library() const;
	void set_library(Ref<DLLibrary> p_library);

	StringName get_script_name() const;
	void set_script_name(StringName p_script_name);

	DLScript();
	~DLScript();
};

class DLLibrary : public Resource {
	_THREAD_SAFE_CLASS_

	GDCLASS(DLLibrary, Resource);
	OBJ_SAVE_TYPE(DLLibrary);

	Map<StringName, String> platform_files;
	NativeLibrary *native_library;
	static DLLibrary *currently_initialized_library;

protected:
	friend class DLScript;
	friend class NativeLibrary;
	friend class DLReloadNode;

	DLScriptData *get_script_data(const StringName p_name);

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);
	static void _bind_methods();

public:
	Error _initialize();
	Error _terminate();

	static DLLibrary *get_currently_initialized_library();

	void _register_script(const StringName p_name, const StringName p_base, godot_instance_create_func p_instance_func, godot_instance_destroy_func p_destroy_func);
	void _register_tool_script(const StringName p_name, const StringName p_base, godot_instance_create_func p_instance_func, godot_instance_destroy_func p_destroy_func);
	void _register_script_method(const StringName p_name, const StringName p_method, godot_method_attributes p_attr, godot_instance_method p_func, MethodInfo p_info);
	void _register_script_property(const StringName p_name, const String p_path, godot_property_attributes *p_attr, godot_property_set_func p_setter, godot_property_get_func p_getter);
	void _register_script_signal(const StringName p_name, const godot_signal *p_signal);

	void set_platform_file(StringName p_platform, String p_file);
	String get_platform_file(StringName p_platform) const;

	DLLibrary();
	~DLLibrary();
};

class DLInstance : public ScriptInstance {
	friend class DLScript;

	Object *owner;
	Ref<DLScript> script;
	void *userdata;

	void _ml_call_reversed(DLScriptData *data_ptr, const StringName &p_method, const Variant **p_args, int p_argcount);

public:
	_FORCE_INLINE_ Object *get_owner() { return owner; }

	_FORCE_INLINE_ void *get_userdata() { return userdata; }

	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid = NULL) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName &p_method) const;
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);

	Variant debug_get_member_by_index(int p_idx) const { return Variant(); }

	virtual void notification(int p_notification);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();

	void set_path(const String &p_path);

	void reload_members();

	virtual RPCMode get_rpc_mode(const StringName &p_method) const;
	virtual RPCMode get_rset_mode(const StringName &p_variable) const;

	DLInstance();
	~DLInstance();
};

class DLReloadNode;

class DLScriptLanguage : public ScriptLanguage {
	friend class DLScript;
	friend class DLInstance;
	friend class DLReloadNode;
	friend class DLLibrary;

	static DLScriptLanguage *singleton;

	Variant *_global_array; // @Unused necessary?
	Vector<Variant> global_array; // @Unused necessary?
	Map<StringName, int> globals; // @Unused necessary?

	// @Unused necessary?
	void _add_global(const StringName &p_name, const Variant &p_value);

	Mutex *lock;

	Set<DLScript *> script_list;

	bool profiling;
	uint64_t script_frame_time;

	struct {

		StringName _notification;

	} strings;

public:
	Map<StringName, NativeLibrary *> initialized_libraries;

	_FORCE_INLINE_ static DLScriptLanguage *get_singleton() { return singleton; }

	virtual String get_name() const;

	/* LANGUAGE FUNCTIONS */
	virtual void init();
	virtual String get_type() const;
	virtual String get_extension() const;
	virtual Error execute_file(const String &p_path);
	virtual void finish();

	/* EDITOR FUNCTIONS */

	virtual void get_reserved_words(List<String> *p_words) const {};
	virtual void get_comment_delimiters(List<String> *p_delimiters) const {};
	virtual void get_string_delimiters(List<String> *p_delimiters) const {};
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const;
	virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path = "", List<String> *r_functions = NULL) const;
	virtual Script *create_script() const;
	virtual bool has_named_classes() const;
	virtual int find_function(const String &p_function, const String &p_code) const;
	virtual String make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const;

	virtual Error complete_code(const String &p_code, const String &p_base_path, Object *p_owner, List<String> *r_options, String &r_call_hint) { return ERR_UNAVAILABLE; }

	virtual Error lookup_code(const String &p_code, const String &p_symbol, const String &p_base_path, Object *p_owner, LookupResult &r_result) { return ERR_UNAVAILABLE; }

	virtual void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {};
	virtual void add_global_constant(const StringName &p_variable, const Variant &p_value);

	/* MULTITHREAD FUNCTIONS */

	//some VMs need to be notified of thread creation/exiting to allocate a stack
	virtual void thread_enter() {}
	virtual void thread_exit() {}

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const;
	virtual int debug_get_stack_level_count() const;
	virtual int debug_get_stack_level_line(int p_level) const;
	virtual String debug_get_stack_level_function(int p_level) const;
	virtual String debug_get_stack_level_source(int p_level) const;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1){};
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1);
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems = -1, int p_max_depth = -1);
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems = -1, int p_max_depth = -1);

	virtual Vector<StackInfo> debug_get_current_stack_info() { return Vector<StackInfo>(); }

	virtual void reload_all_scripts();
	virtual void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload);
	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_public_functions(List<MethodInfo> *p_functions) const;
	virtual void get_public_constants(List<Pair<String, Variant> > *p_constants) const;

	/* PROFILLER FUNCTIONS */

	virtual void profiling_start();
	virtual void profiling_stop();

	virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max);
	virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max);

	virtual void frame();

	static String get_init_symbol_name();
	static String get_terminate_symbol_name();

	/* HACKER FUNCTIONS */
	void _compile_dummy_for_the_api();

	DLScriptLanguage();
	~DLScriptLanguage();
};

class DLReloadNode : public Node {
	GDCLASS(DLReloadNode, Node)
public:
	static void _bind_methods();
	void _notification(int p_what);
};

class ResourceFormatLoaderDLScript : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class ResourceFormatSaverDLScript : public ResourceFormatSaver {
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	virtual bool recognize(const RES &p_resource) const;
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const;
};

// ugly, but hey

#endif // DL_SCRIPT_H
