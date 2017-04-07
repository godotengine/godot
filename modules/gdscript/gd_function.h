/*************************************************************************/
/*  gd_function.h                                                        */
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
#ifndef GD_FUNCTION_H
#define GD_FUNCTION_H

#include "os/thread.h"
#include "pair.h"
#include "reference.h"
#include "script_language.h"
#include "self_list.h"
#include "string_db.h"
#include "variant.h"

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
		OPCODE_SET_MEMBER,
		OPCODE_GET_MEMBER,
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
		OPCODE_BREAKPOINT,
		OPCODE_LINE,
		OPCODE_END
	};

	enum Address {
		ADDR_BITS = 24,
		ADDR_MASK = ((1 << ADDR_BITS) - 1),
		ADDR_TYPE_MASK = ~ADDR_MASK,
		ADDR_TYPE_SELF = 0,
		ADDR_TYPE_CLASS = 1,
		ADDR_TYPE_MEMBER = 2,
		ADDR_TYPE_CLASS_CONSTANT = 3,
		ADDR_TYPE_LOCAL_CONSTANT = 4,
		ADDR_TYPE_STACK = 5,
		ADDR_TYPE_STACK_VARIABLE = 6,
		ADDR_TYPE_GLOBAL = 7,
		ADDR_TYPE_NIL = 8
	};

	enum RPCMode {
		RPC_DISABLED,
		RPC_ENABLED,
		RPC_SYNC,
		RPC_SYNC_MASTER,
		RPC_SYNC_SLAVE
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
	ScriptInstance::RPCMode rpc_mode;

	GDScript *_script;

	StringName name;
	Vector<Variant> constants;
	Vector<StringName> global_names;
	Vector<int> default_arguments;
	Vector<int> code;

#ifdef TOOLS_ENABLED
	Vector<StringName> arg_names;
#endif

	List<StackDebug> stack_debug;

	_FORCE_INLINE_ Variant *_get_variant(int p_address, GDInstance *p_instance, GDScript *p_script, Variant &self, Variant *p_stack, String &r_error) const;
	_FORCE_INLINE_ String _get_call_error(const Variant::CallError &p_err, const String &p_where, const Variant **argptrs) const;

	friend class GDScriptLanguage;

	SelfList<GDFunction> function_list;
#ifdef DEBUG_ENABLED
	CharString func_cname;
	const char *_func_cname;

	struct Profile {
		StringName signature;
		uint64_t call_count;
		uint64_t self_time;
		uint64_t total_time;
		uint64_t frame_call_count;
		uint64_t frame_self_time;
		uint64_t frame_total_time;
		uint64_t last_frame_call_count;
		uint64_t last_frame_self_time;
		uint64_t last_frame_total_time;
	} profile;

#endif

public:
	struct CallState {

		ObjectID instance_id; //by debug only
		ObjectID script_id;

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

	const int *get_code() const; //used for debug
	int get_code_size() const;
	Variant get_constant(int p_idx) const;
	StringName get_global_name(int p_idx) const;
	StringName get_name() const;
	int get_max_stack_size() const;
	int get_default_argument_count() const;
	int get_default_argument_addr(int p_idx) const;
	GDScript *get_script() const { return _script; }
	StringName get_source() const { return source; }

	void debug_get_stack_member_state(int p_line, List<Pair<StringName, int> > *r_stackvars) const;

	_FORCE_INLINE_ bool is_empty() const { return _code_size == 0; }

	int get_argument_count() const { return _argument_count; }
	StringName get_argument_name(int p_idx) const {
#ifdef TOOLS_ENABLED
		ERR_FAIL_INDEX_V(p_idx, arg_names.size(), StringName());
		return arg_names[p_idx];
#endif
		return StringName();
	}
	Variant get_default_argument(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, default_arguments.size(), Variant());
		return default_arguments[p_idx];
	}

	Variant call(GDInstance *p_instance, const Variant **p_args, int p_argcount, Variant::CallError &r_err, CallState *p_state = NULL);

	_FORCE_INLINE_ ScriptInstance::RPCMode get_rpc_mode() const { return rpc_mode; }
	GDFunction();
	~GDFunction();
};

class GDFunctionState : public Reference {

	GDCLASS(GDFunctionState, Reference);
	friend class GDFunction;
	GDFunction *function;
	GDFunction::CallState state;
	Variant _signal_callback(const Variant **p_args, int p_argcount, Variant::CallError &r_error);

protected:
	static void _bind_methods();

public:
	bool is_valid() const;
	Variant resume(const Variant &p_arg = Variant());
	GDFunctionState();
	~GDFunctionState();
};

#endif // GD_FUNCTION_H
