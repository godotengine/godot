/*************************************************************************/
/*  gd_script.h                                                          */
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
#ifndef GD_SCRIPT_H
#define GD_SCRIPT_H

#include "script_language.h"
#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/thread.h"
#include "pair.h"
class GDInstance;
class GDScript;



class GDFunction {
public:

	enum Opcode {
		OPCODE_OPERATOR,
		OPCODE_EXTENDS_TEST,
		OPCODE_SET,
		OPCODE_GET,
		OPCODE_SET_NAMED,
		OPCODE_GET_NAMED,
		OPCODE_ASSIGN,
		OPCODE_ASSIGN_TRUE,
		OPCODE_ASSIGN_FALSE,
		OPCODE_CONSTRUCT, //only for basic types!!
		OPCODE_CONSTRUCT_ARRAY,
		OPCODE_CONSTRUCT_DICTIONARY,
		OPCODE_CALL,
		OPCODE_CALL_RETURN,
		OPCODE_CALL_BUILT_IN,
		OPCODE_CALL_SELF,
		OPCODE_CALL_SELF_BASE,
		OPCODE_YIELD,
		OPCODE_YIELD_SIGNAL,
		OPCODE_YIELD_RESUME,
		OPCODE_JUMP,
		OPCODE_JUMP_IF,
		OPCODE_JUMP_IF_NOT,
		OPCODE_JUMP_TO_DEF_ARGUMENT,
		OPCODE_RETURN,
		OPCODE_ITERATE_BEGIN,
		OPCODE_ITERATE,
		OPCODE_ASSERT,
		OPCODE_LINE,
		OPCODE_END
	};

	enum Address {
		ADDR_BITS=24,
		ADDR_MASK=((1<<ADDR_BITS)-1),
		ADDR_TYPE_MASK=~ADDR_MASK,
		ADDR_TYPE_SELF=0,
		ADDR_TYPE_CLASS=1,
		ADDR_TYPE_MEMBER=2,
		ADDR_TYPE_CLASS_CONSTANT=3,
		ADDR_TYPE_LOCAL_CONSTANT=4,
		ADDR_TYPE_STACK=5,
		ADDR_TYPE_STACK_VARIABLE=6,
		ADDR_TYPE_GLOBAL=7,
		ADDR_TYPE_NIL=8
	};

    struct StackDebug {

        int line;
        int pos;
        bool added;
        StringName identifier;
    };

private:
friend class GDCompiler;

	StringName source;

	mutable Variant nil;
	mutable Variant *_constants_ptr;
	int _constant_count;
	const StringName *_global_names_ptr;
	int _global_names_count;
	const int *_default_arg_ptr;
	int _default_arg_count;
	const int *_code_ptr;
	int _code_size;
	int _argument_count;
	int _stack_size;
	int _call_size;
	int _initial_line;
	bool _static;
	GDScript *_script;

	StringName name;
	Vector<Variant> constants;
	Vector<StringName> global_names;
	Vector<int> default_arguments;
	Vector<int> code;
#ifdef DEBUG_ENABLED
	CharString func_cname;
	const char*_func_cname;
#endif

#ifdef TOOLS_ENABLED
	Vector<StringName> arg_names;
#endif

	List<StackDebug> stack_debug;

	_FORCE_INLINE_ Variant *_get_variant(int p_address,GDInstance *p_instance,GDScript *p_script,Variant &self,Variant *p_stack,String& r_error) const;
	_FORCE_INLINE_ String _get_call_error(const Variant::CallError& p_err, const String& p_where,const Variant**argptrs) const;


public:

	struct CallState {

		GDInstance *instance;
		Vector<uint8_t> stack;
		int stack_size;
		Variant self;
		uint32_t alloca_size;
		GDScript *_class;
		int ip;
		int line;
		int defarg;
		Variant result;

	};

	_FORCE_INLINE_ bool is_static() const { return _static; }

	const int* get_code() const; //used for debug
	int get_code_size() const;
	Variant get_constant(int p_idx) const;
	StringName get_global_name(int p_idx) const;
	StringName get_name() const;
	int get_max_stack_size() const;
	int get_default_argument_count() const;
	int get_default_argument_addr(int p_idx) const;
	GDScript *get_script() const { return _script; }

	void debug_get_stack_member_state(int p_line,List<Pair<StringName,int> > *r_stackvars) const;

	_FORCE_INLINE_ bool is_empty() const { return _code_size==0; }

	int get_argument_count() const { return _argument_count; }
	StringName get_argument_name(int p_idx) const {
#ifdef TOOLS_ENABLED
		ERR_FAIL_INDEX_V(p_idx,arg_names.size(),StringName());
		return arg_names[p_idx];
#endif
		return StringName();

	}
	Variant get_default_argument(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx,default_arguments.size(),Variant());
		return default_arguments[p_idx];
	}

	Variant call(GDInstance *p_instance,const Variant **p_args, int p_argcount,Variant::CallError& r_err,CallState *p_state=NULL);

	GDFunction();
};


class GDFunctionState : public Reference {

	OBJ_TYPE(GDFunctionState,Reference);
friend class GDFunction;
	GDFunction *function;
	GDFunction::CallState state;
	Variant _signal_callback(const Variant** p_args, int p_argcount, Variant::CallError& r_error);
protected:
	static void _bind_methods();
public:

	bool is_valid() const;
	Variant resume(const Variant& p_arg=Variant());
	GDFunctionState();
	~GDFunctionState();
};


class GDNativeClass : public Reference {

	OBJ_TYPE(GDNativeClass,Reference);

	StringName name;
protected:

	bool _get(const StringName& p_name,Variant &r_ret) const;
	static void _bind_methods();

public:

	_FORCE_INLINE_ const StringName& get_name() const { return name; }
	Variant _new();
	Object *instance();
	GDNativeClass(const StringName& p_name);
};


class GDScript : public Script {


	OBJ_TYPE(GDScript,Script);
	bool tool;
	bool valid;


	struct MemberInfo {
		int index;
		StringName setter;
		StringName getter;
	};

friend class GDInstance;
friend class GDFunction;
friend class GDCompiler;
friend class GDFunctions;
friend class GDScriptLanguage;

	Variant _static_ref; //used for static call
	Ref<GDNativeClass> native;
	Ref<GDScript> base;
	GDScript *_base; //fast pointer access
	GDScript *_owner; //for subclasses

	Set<StringName> members; //members are just indices to the instanced script.
	Map<StringName,Variant> constants;
	Map<StringName,GDFunction> member_functions;
	Map<StringName,MemberInfo> member_indices; //members are just indices to the instanced script.
	Map<StringName,Ref<GDScript> > subclasses;	

#ifdef TOOLS_ENABLED

	Map<StringName,Variant> member_default_values;

	List<PropertyInfo> members_cache;
	Map<StringName,Variant> member_default_values_cache;
	Ref<GDScript> base_cache;
	Set<ObjectID> inheriters_cache;
	bool source_changed_cache;
	void _update_exports_values(Map<StringName,Variant>& values, List<PropertyInfo> &propnames);

#endif
	Map<StringName,PropertyInfo> member_info;

	GDFunction *initializer; //direct pointer to _init , faster to locate

	int subclass_count;
	Set<Object*> instances;
	//exported members
	String source;
	String path;
	String name;


	GDInstance* _create_instance(const Variant** p_args,int p_argcount,Object *p_owner,bool p_isref);

	void _set_subclass_path(Ref<GDScript>& p_sc,const String& p_path);

#ifdef TOOLS_ENABLED
	Set<PlaceHolderScriptInstance*> placeholders;
	//void _update_placeholder(PlaceHolderScriptInstance *p_placeholder);
	virtual void _placeholder_erased(PlaceHolderScriptInstance *p_placeholder);
#endif



	bool _update_exports();

protected:
	bool _get(const StringName& p_name,Variant &r_ret) const;
	bool _set(const StringName& p_name, const Variant& p_value);
	void _get_property_list(List<PropertyInfo> *p_properties) const;

	Variant call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error);
//	void call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount);

	static void _bind_methods();
public:

	bool is_valid() const { return valid; }

	const Map<StringName,Ref<GDScript> >& get_subclasses() const { return subclasses; }
	const Map<StringName,Variant >& get_constants() const { return constants; }
	const Set<StringName>& get_members() const { return members; }
	const Map<StringName,GDFunction>& get_member_functions() const { return member_functions; }
	const Ref<GDNativeClass>& get_native() const { return native; }


	bool is_tool() const { return tool; }
	Ref<GDScript> get_base() const;

	const Map<StringName,MemberInfo>& debug_get_member_indices() const { return member_indices; }
	const Map<StringName,GDFunction>& debug_get_member_functions() const; //this is debug only
	StringName debug_get_member_by_index(int p_idx) const;

	Variant _new(const Variant** p_args,int p_argcount,Variant::CallError& r_error);
	virtual bool can_instance() const;

	virtual StringName get_instance_base_type() const; // this may not work in all scripts, will return empty if so
	virtual ScriptInstance* instance_create(Object *p_this);
	virtual bool instance_has(const Object *p_this) const;

	virtual bool has_source_code() const;
	virtual String get_source_code() const;
	virtual void set_source_code(const String& p_code);
	virtual void update_exports();

	virtual Error reload();

	virtual String get_node_type() const;
	void set_script_path(const String& p_path) { path=p_path; } //because subclasses need a path too...
	Error load_source_code(const String& p_path);
	Error load_byte_code(const String& p_path);

	virtual ScriptLanguage *get_language() const;

	GDScript();
};

class GDInstance : public ScriptInstance {
friend class GDScript;
friend class GDFunction;
friend class GDFunctions;

	Object *owner;
	Ref<GDScript> script;
	Vector<Variant> members;
	bool base_ref;

	void _ml_call_reversed(GDScript *sptr,const StringName& p_method,const Variant** p_args,int p_argcount);

public:

	virtual bool set(const StringName& p_name, const Variant& p_value);
	virtual bool get(const StringName& p_name, Variant &r_ret) const;
	virtual void get_property_list(List<PropertyInfo> *p_properties) const;

	virtual void get_method_list(List<MethodInfo> *p_list) const;
	virtual bool has_method(const StringName& p_method) const;
	virtual Variant call(const StringName& p_method,const Variant** p_args,int p_argcount,Variant::CallError &r_error);
	virtual void call_multilevel(const StringName& p_method,const Variant** p_args,int p_argcount);
	virtual void call_multilevel_reversed(const StringName& p_method,const Variant** p_args,int p_argcount);

    Variant debug_get_member_by_index(int p_idx) const { return members[p_idx]; }

	virtual void notification(int p_notification);

	virtual Ref<Script> get_script() const;

	virtual ScriptLanguage *get_language();

	void set_path(const String& p_path);


	GDInstance();
	~GDInstance();

};

class GDScriptLanguage : public ScriptLanguage {

	static GDScriptLanguage *singleton;

	Variant* _global_array;
	Vector<Variant> global_array;
	Map<StringName,int> globals;


    struct CallLevel {

        Variant *stack;
        GDFunction *function;
        GDInstance *instance;
        int *ip;
        int *line;

    };


    int _debug_parse_err_line;
    String _debug_parse_err_file;
    String _debug_error;
    int _debug_call_stack_pos;
    int _debug_max_call_stack;
    CallLevel *_call_stack;

	void _add_global(const StringName& p_name,const Variant& p_value);


public:

	int calls;

    bool debug_break(const String& p_error,bool p_allow_continue=true);
    bool debug_break_parse(const String& p_file, int p_line,const String& p_error);

    _FORCE_INLINE_ void enter_function(GDInstance *p_instance,GDFunction *p_function, Variant *p_stack, int *p_ip, int *p_line) {

        if (Thread::get_main_ID()!=Thread::get_caller_ID())
            return; //no support for other threads than main for now

        if (ScriptDebugger::get_singleton()->get_lines_left()>0 && ScriptDebugger::get_singleton()->get_depth()>=0)
            ScriptDebugger::get_singleton()->set_depth( ScriptDebugger::get_singleton()->get_depth() +1 );

        if (_debug_call_stack_pos >= _debug_max_call_stack) {
            //stack overflow
            _debug_error="Stack Overflow (Stack Size: "+itos(_debug_max_call_stack)+")";
            ScriptDebugger::get_singleton()->debug(this);
            return;
	}

        _call_stack[_debug_call_stack_pos].stack=p_stack;
        _call_stack[_debug_call_stack_pos].instance=p_instance;
        _call_stack[_debug_call_stack_pos].function=p_function;
        _call_stack[_debug_call_stack_pos].ip=p_ip;
        _call_stack[_debug_call_stack_pos].line=p_line;
        _debug_call_stack_pos++;
    }

    _FORCE_INLINE_ void exit_function() {

        if (Thread::get_main_ID()!=Thread::get_caller_ID())
            return; //no support for other threads than main for now

        if (ScriptDebugger::get_singleton()->get_lines_left()>0 && ScriptDebugger::get_singleton()->get_depth()>=0)
	    ScriptDebugger::get_singleton()->set_depth( ScriptDebugger::get_singleton()->get_depth() -1 );

        if (_debug_call_stack_pos==0) {

            _debug_error="Stack Underflow (Engine Bug)";
            ScriptDebugger::get_singleton()->debug(this);
            return;
        }

        _debug_call_stack_pos--;
    }


	struct {

		StringName _init;
		StringName _notification;
		StringName _set;
		StringName _get;
		StringName _get_property_list;
		StringName _script_source;

	} strings;


	_FORCE_INLINE_ int get_global_array_size() const { return global_array.size(); }
	_FORCE_INLINE_ Variant* get_global_array() { return _global_array; }
	_FORCE_INLINE_ const Map<StringName,int>& get_global_map() { return globals; }

	_FORCE_INLINE_ static GDScriptLanguage *get_singleton() { return singleton; }

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
	virtual Error complete_code(const String& p_code, const String& p_base_path, Object*p_owner,List<String>* r_options,String& r_call_hint);
	virtual void auto_indent_code(String& p_code,int p_from_line,int p_to_line) const;

	/* DEBUGGER FUNCTIONS */

	virtual String debug_get_error() const;
	virtual int debug_get_stack_level_count() const;
	virtual int debug_get_stack_level_line(int p_level) const;
	virtual String debug_get_stack_level_function(int p_level) const;
	virtual String debug_get_stack_level_source(int p_level) const;
	virtual void debug_get_stack_level_locals(int p_level,List<String> *p_locals, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual void debug_get_stack_level_members(int p_level,List<String> *p_members, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual void debug_get_globals(List<String> *p_locals, List<Variant> *p_values, int p_max_subitems=-1,int p_max_depth=-1);
	virtual String debug_parse_stack_level_expression(int p_level,const String& p_expression,int p_max_subitems=-1,int p_max_depth=-1);

	virtual void frame();

	virtual void get_public_functions(List<MethodInfo> *p_functions) const;
	virtual void get_public_constants(List<Pair<String,Variant> > *p_constants) const;

	/* LOADER FUNCTIONS */

	virtual void get_recognized_extensions(List<String> *p_extensions) const;

	GDScriptLanguage();
	~GDScriptLanguage();
};


class ResourceFormatLoaderGDScript : public ResourceFormatLoader {
public:

	virtual RES load(const String &p_path,const String& p_original_path="");
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

};

class ResourceFormatSaverGDScript : public ResourceFormatSaver {
public:

	virtual Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);
	virtual void get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const;
	virtual bool recognize(const RES& p_resource) const;

};

#endif // GD_SCRIPT_H
