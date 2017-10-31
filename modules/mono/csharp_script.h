/*************************************************************************/
/*  csharp_script.h                                                      */
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
#ifndef CSHARP_SCRIPT_H
#define CSHARP_SCRIPT_H

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "script_language.h"
#include "self_list.h"

#include "mono_gc_handle.h"
#include "mono_gd/gd_mono.h"
#include "mono_gd/gd_mono_header.h"
#include "mono_gd/gd_mono_internals.h"

class CSharpScript;
class CSharpInstance;
class CSharpLanguage;

#ifdef NO_SAFE_CAST
template <typename TScriptInstance, typename TScriptLanguage>
TScriptInstance *cast_script_instance(ScriptInstance *p_inst) {
	return p_inst->get_language() == TScriptLanguage::get_singleton() ? static_cast<TScriptInstance *>(p_inst) : NULL;
}
#else
template <typename TScriptInstance, typename TScriptLanguage>
TScriptInstance *cast_script_instance(ScriptInstance *p_inst) {
	return dynamic_cast<TScriptInstance *>(p_inst);
}
#endif

#define CAST_CSHARP_INSTANCE(m_inst) (cast_script_instance<CSharpInstance, CSharpLanguage>(m_inst))

class CSharpScript : public Script {

	GDCLASS(CSharpScript, Script)

	friend class CSharpInstance;
	friend class CSharpLanguage;
	friend class CSharpScriptDepSort;

	bool tool;
	bool valid;

	bool builtin;

	GDMonoClass *base;
	GDMonoClass *native;
	GDMonoClass *script_class;

	Ref<CSharpScript> base_cache; // TODO what's this for?

	Set<Object *> instances;

	String source;
	StringName name;

	SelfList<CSharpScript> script_list;

#ifdef TOOLS_ENABLED
	List<PropertyInfo> exported_members_cache; // members_cache
	Map<StringName, Variant> exported_members_defval_cache; // member_default_values_cache
	Set<PlaceHolderScriptInstance *> placeholders;
	bool source_changed_cache;
	bool exports_invalidated;

	void _update_exports_values(Map<StringName, Variant> &values, List<PropertyInfo> &propnames);
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);
#endif

#ifdef DEBUG_ENABLED
	Map<ObjectID, List<Pair<StringName, Variant> > > pending_reload_state;
#endif

	Map<StringName, PropertyInfo> member_info;

	void _clear();

	bool _update_exports();
	CSharpInstance *_create_instance(const Variant **p_args, int p_argcount, Object *p_owner, bool p_isref, Variant::CallError &r_error);
	Variant _new(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

	// Do not use unless you know what you are doing
	friend void GDMonoInternals::tie_managed_to_unmanaged(MonoObject *, Object *);
	static Ref<CSharpScript> create_for_managed_type(GDMonoClass *p_class);

protected:
	static void _bind_methods();

	Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void _resource_path_changed();
	bool _get(const StringName &p_name, Variant &r_ret) const;
	bool _set(const StringName &p_name, const Variant &p_value);
	void _get_property_list(List<PropertyInfo> *p_properties) const;

public:
	virtual bool can_instance() const;
	virtual StringName get_instance_base_type() const;
	virtual ScriptInstance *instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String &p_code);

	virtual Error reload(bool p_keep_state = false);

	/* TODO */ virtual bool has_script_signal(const StringName &p_signal) const { return false; }
	/* TODO */ virtual void get_script_signal_list(List<MethodInfo> *r_signals) const {}

	/* TODO */ virtual bool get_property_default_value(const StringName &p_property, Variant &r_value) const;
	virtual void get_script_property_list(List<PropertyInfo> *p_list) const;
	virtual void update_exports();

	virtual bool is_tool() const { return tool; }
	virtual Ref<Script> get_base_script() const;
	virtual ScriptLanguage *get_language() const;

	/* TODO */ virtual void get_script_method_list(List<MethodInfo> *p_list) const {}
	bool has_method(const StringName &p_method) const;
	/* TODO */ MethodInfo get_method_info(const StringName &p_method) const { return MethodInfo(); }

	virtual int get_member_line(const StringName &p_member) const;

	Error load_source_code(const String &p_path);

	StringName get_script_name() const;

	CSharpScript();
	~CSharpScript();
};

class CSharpInstance : public ScriptInstance {

	friend class CSharpScript;
	friend class CSharpLanguage;
	Object *owner;
	Ref<CSharpScript> script;
	Ref<MonoGCHandle> gchandle;
	bool base_ref;
	bool ref_dying;

	void _ml_call_reversed(GDMonoClass *klass, const StringName &p_method, const Variant **p_args, int p_argcount);

	void _reference_owner_unsafe();
	void _unreference_owner_unsafe();

	// Do not use unless you know what you are doing
	friend void GDMonoInternals::tie_managed_to_unmanaged(MonoObject *, Object *);
	static CSharpInstance *create_for_managed_type(Object *p_owner, CSharpScript *p_script, const Ref<MonoGCHandle> &p_gchandle);

public:
	MonoObject *get_mono_object() const;

	virtual bool set(const StringName &p_name, const Variant &p_value);
	virtual bool get(const StringName &p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;
	virtual Variant::Type get_property_type(const StringName &p_name, bool *r_is_valid) const;

	/* TODO */ virtual void get_method_list(List<MethodInfo> *p_list) const {}
	virtual bool has_method(const StringName &p_method) const;
	virtual Variant call(const StringName &p_method, const Variant **p_args, int p_argcount, Variant::CallError &r_error);
	virtual void call_multilevel(const StringName &p_method, const Variant **p_args, int p_argcount);
	virtual void call_multilevel_reversed(const StringName &p_method, const Variant **p_args, int p_argcount);

	void mono_object_disposed();

	void refcount_incremented();
	bool refcount_decremented();

	RPCMode get_rpc_mode(const StringName &p_method) const;
	RPCMode get_rset_mode(const StringName &p_variable) const;

	virtual void notification(int p_notification);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();

	CSharpInstance();
	~CSharpInstance();
};

class CSharpLanguage : public ScriptLanguage {

	friend class CSharpScript;
	friend class CSharpInstance;

	static CSharpLanguage *singleton;

	GDMono *gdmono;
	SelfList<CSharpScript>::List script_list;

	Mutex *lock;
	Mutex *script_bind_lock;

	Map<Ref<CSharpScript>, Map<ObjectID, List<Pair<StringName, Variant> > > > to_reload;

	Map<Object *, Ref<MonoGCHandle> > gchandle_bindings;

	struct StringNameCache {

		StringName _signal_callback;
		StringName _set;
		StringName _get;
		StringName _notification;
		StringName _script_source;
		StringName dotctor; // .ctor

		StringNameCache();
	};

	int lang_idx;

public:
	StringNameCache string_names;

	_FORCE_INLINE_ int get_language_index() { return lang_idx; }
	void set_language_index(int p_idx);

	_FORCE_INLINE_ const StringNameCache &get_string_names() { return string_names; }

	_FORCE_INLINE_ static CSharpLanguage *get_singleton() { return singleton; }

	bool debug_break(const String &p_error, bool p_allow_continue = true);
	bool debug_break_parse(const String &p_file, int p_line, const String &p_error);

#ifdef TOOLS_ENABLED
	void reload_assemblies_if_needed(bool p_soft_reload);
#endif

	virtual String get_name() const;

	/* LANGUAGE FUNCTIONS */
	virtual String get_type() const;
	virtual String get_extension() const;
	virtual Error execute_file(const String &p_path);
	virtual void init();
	virtual void finish();

	/* EDITOR FUNCTIONS */
	virtual void get_reserved_words(List<String> *p_words) const;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const;
	virtual void get_string_delimiters(List<String> *p_delimiters) const;
	virtual Ref<Script> get_template(const String &p_class_name, const String &p_base_class_name) const;
	virtual bool is_using_templates();
	virtual void make_template(const String &p_class_name, const String &p_base_class_name, Ref<Script> &p_script);
	/* TODO */ virtual bool validate(const String &p_script, int &r_line_error, int &r_col_error, String &r_test_error, const String &p_path, List<String> *r_functions) const { return true; }
	virtual Script *create_script() const;
	virtual bool has_named_classes() const;
	virtual bool supports_builtin_mode() const;
	/* TODO? */ virtual int find_function(const String &p_function, const String &p_code) const { return -1; }
	virtual String make_function(const String &p_class, const String &p_name, const PoolStringArray &p_args) const;
	/* TODO? */ Error complete_code(const String &p_code, const String &p_base_path, Object *p_owner, List<String> *r_options, String &r_call_hint) { return ERR_UNAVAILABLE; }
	virtual String _get_indentation() const;
	/* TODO? */ virtual void auto_indent_code(String &p_code, int p_from_line, int p_to_line) const {}
	/* TODO */ virtual void add_global_constant(const StringName &p_variable, const Variant &p_value) {}

	/* DEBUGGER FUNCTIONS */
	/* TODO */ virtual String debug_get_error() const { return ""; }
	/* TODO */ virtual int debug_get_stack_level_count() const { return 1; }
	/* TODO */ virtual int debug_get_stack_level_line(int p_level) const { return 1; }
	/* TODO */ virtual String debug_get_stack_level_function(int p_level) const { return ""; }
	/* TODO */ virtual String debug_get_stack_level_source(int p_level) const { return ""; }
	/* TODO */ virtual void debug_get_stack_level_locals(int p_level, List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}
	/* TODO */ virtual void debug_get_stack_level_members(int p_level, List<String> *p_members, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}
	/* TODO */ virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems, int p_max_depth) {}
	/* TODO */ virtual String debug_parse_stack_level_expression(int p_level, const String &p_expression, int p_max_subitems, int p_max_depth) { return ""; }
	/* TODO */ virtual Vector<StackInfo> debug_get_current_stack_info() { return Vector<StackInfo>(); }

	/* PROFILING FUNCTIONS */
	/* TODO */ virtual void profiling_start() {}
	/* TODO */ virtual void profiling_stop() {}
	/* TODO */ virtual int profiling_get_accumulated_data(ProfilingInfo *p_info_arr, int p_info_max) { return 0; }
	/* TODO */ virtual int profiling_get_frame_data(ProfilingInfo *p_info_arr, int p_info_max) { return 0; }

	virtual void frame();

	/* TODO? */ virtual void get_public_functions(List<MethodInfo> *p_functions) const {}
	/* TODO? */ virtual void get_public_constants(List<Pair<String, Variant> > *p_constants) const {}

	virtual void reload_all_scripts();
	virtual void reload_tool_script(const Ref<Script> &p_script, bool p_soft_reload);

	/* LOADER FUNCTIONS */
	virtual void get_recognized_extensions(List<String> *p_extensions) const;

#ifdef TOOLS_ENABLED
	virtual Error open_in_external_editor(const Ref<Script> &p_script, int p_line, int p_col);
	virtual bool overrides_external_editor();
#endif

	/* THREAD ATTACHING */
	virtual void thread_enter();
	virtual void thread_exit();

	// Don't use these. I'm watching you
	virtual void *alloc_instance_binding_data(Object *p_object);
	virtual void free_instance_binding_data(void *p_data);

	CSharpLanguage();
	~CSharpLanguage();
};

class ResourceFormatLoaderCSharpScript : public ResourceFormatLoader {
public:
	virtual RES load(const String &p_path, const String &p_original_path = "", Error *r_error = NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String &p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};

class ResourceFormatSaverCSharpScript : public ResourceFormatSaver {
public:
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const;
	virtual bool recognize(const RES &p_resource) const;
};

#endif // CSHARP_SCRIPT_H
