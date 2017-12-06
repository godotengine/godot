/*************************************************************************/
/*  nativescript.h                                                       */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
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
#ifndef NATIVE_SCRIPT_H
#define NATIVE_SCRIPT_H

#include "core/resource.h"
#include "core/script_language.h"
#include "core/self_list.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "ordered_hash_map.h"
#include "os/thread_safe.h"
#include "scene/main/node.h"

#include "modules/gdnative/gdnative.h"
#include <nativescript/godot_nativescript.h>

#ifndef NO_THREADS
#include "os/mutex.h"
#endif

struct NativeScriptDesc {

	struct Method {
		godot_instance_method method;
		MethodInfo info;
		int rpc_mode;
	};
	struct Property {
		godot_property_set_func setter;
		godot_property_get_func getter;
		PropertyInfo info;
		Variant default_value;
		int rset_mode;
	};

	struct Signal {
		MethodInfo signal;
	};

	Map<StringName, Method> methods;
	OrderedHashMap<StringName, Property> properties;
	Map<StringName, Signal> signals_; // QtCreator doesn't like the name signals
	StringName base;
	StringName base_native_type;
	NativeScriptDesc *base_data;
	godot_instance_create_func create_func;
	godot_instance_destroy_func destroy_func;

	bool is_tool;

	inline NativeScriptDesc() :
			methods(),
			properties(),
			signals_(),
			base(),
			base_native_type() {
		zeromem(&create_func, sizeof(godot_instance_create_func));
		zeromem(&destroy_func, sizeof(godot_instance_destroy_func));
	}
};

class NativeScript : public Script {
	GDCLASS(NativeScript, Script)

#ifdef TOOLS_ENABLED
	Set<PlaceHolderScriptInstance *> placeholders;
	void _update_placeholder(PlaceHolderScriptInstance *p_placeholder);
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);
#endif

	friend class NativeScriptInstance;
	friend class NativeScriptLanguage;
	friend class NativeReloadNode;
	friend class GDNativeLibrary;

	Ref<GDNativeLibrary> library;

	String lib_path;

	String class_name;

#ifndef NO_THREADS
	Mutex *owners_lock;
#endif
	Set<Object *> instance_owners;

protected:
	static void _bind_methods();

public:
	inline NativeScriptDesc *get_script_desc() const;

	void set_class_name(String p_class_name);
	String get_class_name() const;

	void set_library(Ref<GDNativeLibrary> p_library);
	Ref<GDNativeLibrary> get_library() const;

	virtual bool can_instance() const;

	virtual Ref<Script> get_base_script() const; //for script inheritance

	virtual StringName get_instance_base_type() const; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance *instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String &p_code);
	virtual Error reload(bool p_keep_state = false);

	virtual bool has_method(const StringName &p_method) const;
	virtual MethodInfo get_method_info(const StringName &p_method) const;

	virtual bool is_tool() const;

	virtual ScriptLanguage *get_language() const;

	virtual bool has_script_signal(const StringName &p_signal) const;
	virtual void get_script_signal_list(List<MethodInfo> *r_signals) const;

	virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const;

	virtual void update_exports(); //editor tool
	virtual void get_script_method_list(List<MethodInfo> *p_list) const;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const;

	Variant _new(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	NativeScript();
	~NativeScript();
};

class NativeScriptInstance : public ScriptInstance {

	friend class NativeScript;

	Object *owner;
	Ref<NativeScript> script;

	void _ml_call_reversed(NativeScriptDesc *script_data, const StringName &p_method, const Variant **p_args, int p_argcount);

public:
	void *userdata;

	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const;
	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName &p_method) const;
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void notification(int p_notification);
	virtual Ref<Script> get_script() const;
	virtual RPCMode get_rpc_mode(const StringName &p_method) const;
	virtual RPCMode get_rset_mode(const StringName &p_variable) const;
	virtual ScriptLanguage *get_language();

	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);

	virtual void refcount_incremented();
	virtual bool refcount_decremented();

	~NativeScriptInstance();
};

class NativeReloadNode;

class NativeScriptLanguage : public ScriptLanguage {

	friend class NativeScript;
	friend class NativeScriptInstance;
	friend class NativeReloadNode;

private:
	static NativeScriptLanguage *singleton;

	void _unload_stuff();

#ifndef NO_THREADS
	Mutex *mutex;

	Set<Ref<GDNativeLibrary> > libs_to_init;
	Set<NativeScript *> scripts_to_register;
	volatile bool has_objects_to_register; // so that we don't lock mutex every frame - it's rarely needed
	void defer_init_library(Ref<GDNativeLibrary> lib, NativeScript *script);
#endif

	void init_library(const Ref<GDNativeLibrary> &lib);
	void register_script(NativeScript *script);
	void unregister_script(NativeScript *script);

	void call_libraries_cb(const StringName &name);

public:
	// These two maps must only be touched on the main thread
	Map<String, Map<StringName, NativeScriptDesc> > library_classes;
	Map<String, Ref<GDNative> > library_gdnatives;

	Map<String, Set<NativeScript *> > library_script_users;

	const StringName _init_call_type = "nativescript_init";
	const StringName _init_call_name = "nativescript_init";

	const StringName _noarg_call_type = "nativescript_no_arg";

	const StringName _frame_call_name = "nativescript_frame";

#ifndef NO_THREADS
	const StringName _thread_enter_call_name = "nativescript_thread_enter";
	const StringName _thread_exit_call_name = "nativescript_thread_exit";
#endif

	NativeScriptLanguage();
	~NativeScriptLanguage();

	inline static NativeScriptLanguage *get_singleton() {
		return singleton;
	}

	void _hacky_api_anchor();

#ifndef NO_THREADS
	virtual void thread_enter();
	virtual void thread_exit();
#endif

	virtual void frame();

	virtual String get_name() const;
	virtual void init();
	virtual String get_type() const;
	virtual String get_extension() const;
	virtual Error execute_file(const String &p_path);
	virtual void finish();
	virtual void get_reserved_words(List<String> *p_words) const;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const;
	virtual void get_string_delimiters(List<String> *p_delimiters) const;
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const;
	virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions) const;
	virtual Script *create_script() const;
	virtual bool has_named_classes() const;
	virtual bool supports_builtin_mode() const;
	virtual int find_function(const String &p_function, const String &p_code) const;
	virtual String make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const;
	virtual void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const;
	virtual void add_global_constant(const StringName &p_variable, const Variant &p_value);
	virtual String debug_get_error() const;
	virtual int debug_get_stack_level_count() const;
	virtual int debug_get_stack_level_line(int p_level) const;
	virtual String debug_get_stack_level_function(int p_level) const;
	virtual String debug_get_stack_level_source(int p_level) const;
	virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth);
	virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth);
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth);
	virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth);
	virtual void reload_all_scripts();
	virtual void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual void get_public_functions(List<MethodInfo> *p_functions) const;
	virtual void get_public_constants(List<Pair<String, Variant> > *p_constants) const;
	virtual void profiling_start();
	virtual void profiling_stop();
	virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max);
	virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max);
};

inline NativeScriptDesc *NativeScript::get_script_desc() const {
	Map<StringName, NativeScriptDesc>::Element *E = NativeScriptLanguage::singleton->library_classes[lib_path].find(class_name);
	return E ? &E->get() : NULL;
}

class NativeReloadNode : public Node {
	GDCLASS(NativeReloadNode, Node)
	bool unloaded = false;

public:
	static void _bind_methods();
	void _notification(int p_what);
};

class ResourceFormatLoaderNativeScript : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class ResourceFormatSaverNativeScript : public ResourceFormatSaver {
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	virtual bool recognize(const RES &p_resource) const;
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const;
};

#endif // GDNATIVE_H
