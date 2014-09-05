/*************************************************************************/
/*  lua_script.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifdef LUASCRIPT_ENABLED

#ifndef LUA_SCRIPT_H
#define LUA_SCRIPT_H

#include "script_language.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/thread.h"
#include "pair.h"
#include "os/mutex.h"
extern "C" {
#include "lua/src/lua.h"
#include "lua/src/lauxlib.h"
#include "lua/src/lualib.h"
};

class LuaInstance;
class LuaScript;

class LuaNativeClass : public Reference {

	OBJ_TYPE(LuaNativeClass,Reference);

	StringName name;
protected:

	bool _get(const StringName& p_name,Variant &r_ret) const;
	static void _bind_methods();

public:

	_FORCE_INLINE_ const StringName& get_name() const { return name; }
	Variant _new();
	Object *instance();
	LuaNativeClass(const StringName& p_name);
	virtual ~LuaNativeClass();
};


class LuaScript : public Script {


	OBJ_TYPE(LuaScript,Script);
	bool tool;
	bool valid;

friend class LuaInstance;
friend class LuaScriptLanguage;

	Ref<LuaNativeClass> native;
	Ref<LuaScript> base;
	LuaScript *_base; //fast pointer access
//	LuaScript *_owner; //for subclasses
//
//	Set<StringName> members; //members are just indices to the instanced script.
//	Map<StringName,Variant> constants;
//	Map<StringName,GDFunction> member_functions;
//	Map<StringName,int> member_indices; //members are just indices to the instanced script.
//	Map<StringName,Ref<LuaScript> > subclasses;	
//
#ifdef TOOLS_ENABLED
	Map<StringName,Variant> member_default_values;
#endif
	Map<StringName,PropertyInfo> member_info;

//	int subclass_count;
	Set<Object*> instances;
	//exported members
	String source;
    Vector<uint8_t> bytecode;
	String path;
	String name;
    int ref; // ref to loaded lua script chunk(function)

	LuaInstance* _create_instance(const Variant** p_args,int p_argcount,Object *p_owner,bool p_isref);

//	void _set_subclass_path(Ref<LuaScript>& p_sc,const String& p_path);

#ifdef TOOLS_ENABLED
	Set<PlaceHolderScriptInstance*> placeholders;
	void _update_placeholder(PlaceHolderScriptInstance *p_placeholder);
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);
#endif

    void reset();
    // lua functions
    static int l_extends(lua_State *L);

    static bool preprocessHints(PropertyInfo& pi, Vector<String>& tokens);
    static int l_export(lua_State *L);

    // lua meta methods
    static int l_meta_index(lua_State *L);
    static int l_meta_gc(lua_State *L);


protected:
//	bool _get(const StringName& p_name,Variant &r_ret) const;
//	bool _set(const StringName& p_name, const Variant& p_value);
//	void _get_property_list(List<PropertyInfo> *p_properties) const;
//
//	Variant call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error);
//	void call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount);

	static void _bind_methods();
public:
//
//
//	const Map<StringName,Ref<LuaScript> >& get_subclasses() const { return subclasses; }
//	const Map<StringName,Variant >& get_constants() const { return constants; }
//	const Set<StringName>& get_members() const { return members; }
//	const Map<StringName,GDFunction>& get_member_functions() const { return member_functions; }
//	const Ref<LuaNativeClass>& get_native() const { return native; }


	bool is_tool() const { return tool; }
	Ref<LuaScript> get_base() const;

//	const Map<StringName,int>& debug_get_member_indices() const { return member_indices; }
//	const Map<StringName,GDFunction>& debug_get_member_functions() const; //this is debug only
//	StringName debug_get_member_by_index(int p_idx) const;
//
	Variant _new(const Variant** p_args,int p_argcount,Variant::CallError& r_error);
	virtual bool can_instance() const;

	virtual StringName get_instance_base_type() const; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance* instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String& p_code);
	virtual Error reload();

	virtual String get_node_type() const;
	void set_script_path(const String& p_path) { path=p_path; } //because subclasses need a path too...
	Error load_source_code(const String& p_path);
    Error load_byte_code(const String& p_path);

	virtual ScriptLanguage *get_language() const;

    void reportError(const char* fmt, ...) const;

	LuaScript();
    ~LuaScript();
};

class LuaInstance : public ScriptInstance {
friend class LuaScript;
friend class LuaScriptLanguage;
//friend class GDFunctions;

	Variant owner;
	Ref<LuaScript> script;
//	Vector<Variant> members;
	bool base_ref;
    bool gc_delete;
    int ref; // ref to object's lua table

	void _ml_call_reversed(LuaScript *sptr,const StringName& p_method,const Variant** p_args,int p_argcount);

    int _call_script(const LuaScript *sptr, const LuaInstance *inst, const char *p_method, const Variant** p_args, int p_argcount, bool p_ret) const;
    int _call_script_func(const LuaScript *sptr, const LuaInstance *inst, const char *p_method, const Variant** p_args, int p_argcount) const;
    int _call_script_func(const LuaScript *sptr, const LuaInstance *inst, const char *p_method, const Variant** p_args, int p_argcount, Variant& result) const;

    // lua methods
    static int l_extends(lua_State *L);
    //static int l_ratain(lua_State *L);
    //static int l_release(lua_State *L);

    static int l_methodbind_wrapper(lua_State *L);
    static int l_bultins_wrapper(lua_State *L);
    static int l_bultins_tostring(lua_State *L);
    static int l_bultins_index(lua_State *L);
    static int l_bultins_caller_wrapper(lua_State *L);

    static int l_push_bulltins_type(lua_State *L, const Variant& var);

    // GdObject lua meta methods
    static int meta__gc(lua_State *L);
    static int meta__tostring(lua_State *L);
    static int meta__index(lua_State *L);
    static int meta__newindex(lua_State *L);

    // Variant bultins lua meta methods
    static int meta_bultins__gc(lua_State *L);
    static int meta_bultins__tostring(lua_State *L);
    static int meta_bultins__index(lua_State *L);
    static int meta_bultins__newindex(lua_State *L);
    static int meta_bultins__evaluate(lua_State *L);


public:

	virtual bool set(const StringName& p_name, const Variant& p_value);
	virtual bool get(const StringName& p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName& p_method) const;
	virtual Variant call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error);
	virtual void call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount);
	virtual void call_multilevel_reversed(const StringName& p_method,const Variant** p_args,int p_argcount);

//    Variant debug_get_member_by_index(int p_idx) const { return members[p_idx]; }

	virtual void notification(int p_notification);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();

//	void set_path(const String& p_path);

    int init(bool p_ref = false);
    // helper lua functions
    static void l_push_variant(lua_State *L, const Variant& var);
    static void l_get_variant(lua_State *L, int idx, Variant& var);
    static void l_push_value(lua_State *L, int idx);
    static bool l_register_bultins_ctors(lua_State *L);

    bool l_get_object_table() const;

    static void setup();


	LuaInstance();
	~LuaInstance();

};

class LuaScriptLanguage : public ScriptLanguage {

	static LuaScriptLanguage *singleton;

	Variant* _global_array;
	Vector<Variant> global_array;
	Map<StringName,int> globals;

    lua_State *L;
    Mutex* lock;

    int _debug_parse_err_line;
    String _debug_parse_err_file;
    String _debug_error;

    bool _debug_in_coroutine;
    int _debug_running_level;
    int _debug_break_level;

    bool hitBreakPoint(lua_State *L, lua_Debug *ar);
    void onHook(lua_State *L, lua_Debug *ar);
    static void hookRoutine(lua_State *L, lua_Debug *ar);

	void _add_global(const StringName& p_name,const Variant& p_value);
    bool execute(const char *script);

public:

	int calls;

    bool debug_break(const String& p_error,bool p_allow_continue=true);
    bool debug_break_parse(const String& p_file, int p_line,const String& p_error);

	_FORCE_INLINE_ int get_global_array_size() const { return global_array.size(); }
	_FORCE_INLINE_ Variant* get_global_array() { return _global_array; }
	_FORCE_INLINE_ const Map<StringName,int>& get_global_map() { return globals; }

	_FORCE_INLINE_ static LuaScriptLanguage *get_singleton() { return singleton; }

	virtual String get_name() const;

	/* LANGUAGE FUNCTIONS */
	virtual void init();
	virtual String get_type() const;
	virtual String get_extension() const;
	virtual Error execute_file(const String& p_path) ;
	virtual void finish();

	/* EDITOR FUNCTIONS */
	virtual void get_reserved_words(List<String> *p_words) const;
	virtual void get_comment_delimiters(List<String> *p_delimiters) const;
	virtual void get_string_delimiters(List<String> *p_delimiters) const;
	virtual String get_template(const String& p_class_name, const String& p_base_class_name) const;
	virtual bool validate(const String& p_script,int &r_line_error,int &r_col_error,String& r_test_error, const String& p_path="",List<String> *r_functions=NULL) const;
	virtual Script *create_script() const;
	virtual bool has_named_classes() const;
	virtual int find_function(const String& p_function,const String& p_code) const;
	virtual String make_function(const String& p_class,const String& p_name,const StringArray& p_args) const;
	virtual void auto_indent_code(String& p_code,int p_from_line,int p_to_line) const;
	virtual Error complete_keyword(const String& p_code, int p_line, const String& p_base_path,const String& p_keyword, List<String>* r_options);

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const;
	virtual int debug_get_stack_level_count() const;
	virtual int debug_get_stack_level_line(int p_level) const;
	virtual String debug_get_stack_level_function(int p_level) const;
	virtual String debug_get_stack_level_source(int p_level) const;
	virtual void debug_get_stack_level_locals(int p_level,List<String> *p_locals, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual void debug_get_stack_level_members(int p_level,List<String> *p_members, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual String debug_parse_stack_level_expression(int p_level,const String& p_expression,int p_max_subitems=-1,int p_max_depth=-1,bool p_return=true);
    virtual void debug_status_changed();

	virtual void frame();

	virtual void get_public_functions(List<MethodInfo> *p_functions) const;
	virtual void get_public_constants(List<Pair<String,Variant> > *p_constants) const;

	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const;

    lua_State *get_state() const { return L; }
    Mutex* get_lock() const { return lock; }

    static int panic(lua_State *L);

	LuaScriptLanguage();
	~LuaScriptLanguage();
};


class ResourceFormatLoaderLuaScript : public ResourceFormatLoader {
public:

	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

class ResourceFormatSaverLuaScript : public ResourceFormatSaver {
public:

	virtual Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);
	virtual void get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const;
	virtual bool recognize(const RES& p_resource) const;

};

//#define LUA_MULTITHREAD_GUARD()\
//    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();
#define LUA_MULTITHREAD_GUARD()\
    LuaScriptLanguage *lang = LuaScriptLanguage::get_singleton();\
    MutexLock lklua(lang->get_lock());

#endif // LUA_SCRIPT_H

#endif
