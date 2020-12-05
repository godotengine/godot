/*************************************************************************/
/*  gdscript_function.h                                                  */
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

#ifndef GDSCRIPT_FUNCTION_H
#define GDSCRIPT_FUNCTION_H

#include "core/object/reference.h"
#include "core/object/script_language.h"
#include "core/os/thread.h"
#include "core/string/string_name.h"
#include "core/templates/pair.h"
#include "core/templates/self_list.h"
#include "core/variant/variant.h"

class GDScriptInstance;
class GDScript;

struct GDScriptDataType {
	enum Kind {
		UNINITIALIZED,
		BUILTIN,
		NATIVE,
		SCRIPT,
		GDSCRIPT,
	};

	Kind kind = UNINITIALIZED;

	bool has_type = false;
	Variant::Type builtin_type = Variant::NIL;
	StringName native_type;
	Script *script_type = nullptr;
	Ref<Script> script_type_ref;

	bool is_type(const Variant &p_variant, bool p_allow_implicit_conversion = false) const {
		if (!has_type) {
			return true; // Can't type check
		}

		switch (kind) {
			case UNINITIALIZED:
				break;
			case BUILTIN: {
				Variant::Type var_type = p_variant.get_type();
				bool valid = builtin_type == var_type;
				if (!valid && p_allow_implicit_conversion) {
					valid = Variant::can_convert_strict(var_type, builtin_type);
				}
				return valid;
			} break;
			case NATIVE: {
				if (p_variant.get_type() == Variant::NIL) {
					return true;
				}
				if (p_variant.get_type() != Variant::OBJECT) {
					return false;
				}

				Object *obj = p_variant.get_validated_object();
				if (!obj) {
					return false;
				}

				if (!ClassDB::is_parent_class(obj->get_class_name(), native_type)) {
					// Try with underscore prefix
					StringName underscore_native_type = "_" + native_type;
					if (!ClassDB::is_parent_class(obj->get_class_name(), underscore_native_type)) {
						return false;
					}
				}
				return true;
			} break;
			case SCRIPT:
			case GDSCRIPT: {
				if (p_variant.get_type() == Variant::NIL) {
					return true;
				}
				if (p_variant.get_type() != Variant::OBJECT) {
					return false;
				}

				Object *obj = p_variant.get_validated_object();
				if (!obj) {
					return false;
				}

				Ref<Script> base = obj && obj->get_script_instance() ? obj->get_script_instance()->get_script() : nullptr;
				bool valid = false;
				while (base.is_valid()) {
					if (base == script_type) {
						valid = true;
						break;
					}
					base = base->get_base_script();
				}
				return valid;
			} break;
		}
		return false;
	}

	operator PropertyInfo() const {
		PropertyInfo info;
		if (has_type) {
			switch (kind) {
				case UNINITIALIZED:
					break;
				case BUILTIN: {
					info.type = builtin_type;
				} break;
				case NATIVE: {
					info.type = Variant::OBJECT;
					info.class_name = native_type;
				} break;
				case SCRIPT:
				case GDSCRIPT: {
					info.type = Variant::OBJECT;
					info.class_name = script_type->get_instance_base_type();
				} break;
			}
		} else {
			info.type = Variant::NIL;
			info.usage |= PROPERTY_USAGE_NIL_IS_VARIANT;
		}
		return info;
	}

	GDScriptDataType() {}
};

class GDScriptFunction {
public:
	enum Opcode {
		OPCODE_OPERATOR,
		OPCODE_OPERATOR_VALIDATED,
		OPCODE_EXTENDS_TEST,
		OPCODE_IS_BUILTIN,
		OPCODE_SET_KEYED,
		OPCODE_SET_KEYED_VALIDATED,
		OPCODE_SET_INDEXED_VALIDATED,
		OPCODE_GET_KEYED,
		OPCODE_GET_KEYED_VALIDATED,
		OPCODE_GET_INDEXED_VALIDATED,
		OPCODE_SET_NAMED,
		OPCODE_SET_NAMED_VALIDATED,
		OPCODE_GET_NAMED,
		OPCODE_GET_NAMED_VALIDATED,
		OPCODE_SET_MEMBER,
		OPCODE_GET_MEMBER,
		OPCODE_ASSIGN,
		OPCODE_ASSIGN_TRUE,
		OPCODE_ASSIGN_FALSE,
		OPCODE_ASSIGN_TYPED_BUILTIN,
		OPCODE_ASSIGN_TYPED_NATIVE,
		OPCODE_ASSIGN_TYPED_SCRIPT,
		OPCODE_CAST_TO_BUILTIN,
		OPCODE_CAST_TO_NATIVE,
		OPCODE_CAST_TO_SCRIPT,
		OPCODE_CONSTRUCT, // Only for basic types!
		OPCODE_CONSTRUCT_VALIDATED, // Only for basic types!
		OPCODE_CONSTRUCT_ARRAY,
		OPCODE_CONSTRUCT_DICTIONARY,
		OPCODE_CALL,
		OPCODE_CALL_RETURN,
		OPCODE_CALL_ASYNC,
		OPCODE_CALL_BUILT_IN,
		OPCODE_CALL_BUILTIN_TYPE_VALIDATED,
		OPCODE_CALL_SELF_BASE,
		OPCODE_CALL_METHOD_BIND,
		OPCODE_CALL_METHOD_BIND_RET,
		// ptrcall have one instruction per return type.
		OPCODE_CALL_PTRCALL_NO_RETURN,
		OPCODE_CALL_PTRCALL_BOOL,
		OPCODE_CALL_PTRCALL_INT,
		OPCODE_CALL_PTRCALL_FLOAT,
		OPCODE_CALL_PTRCALL_STRING,
		OPCODE_CALL_PTRCALL_VECTOR2,
		OPCODE_CALL_PTRCALL_VECTOR2I,
		OPCODE_CALL_PTRCALL_RECT2,
		OPCODE_CALL_PTRCALL_RECT2I,
		OPCODE_CALL_PTRCALL_VECTOR3,
		OPCODE_CALL_PTRCALL_VECTOR3I,
		OPCODE_CALL_PTRCALL_TRANSFORM2D,
		OPCODE_CALL_PTRCALL_PLANE,
		OPCODE_CALL_PTRCALL_QUAT,
		OPCODE_CALL_PTRCALL_AABB,
		OPCODE_CALL_PTRCALL_BASIS,
		OPCODE_CALL_PTRCALL_TRANSFORM,
		OPCODE_CALL_PTRCALL_COLOR,
		OPCODE_CALL_PTRCALL_STRING_NAME,
		OPCODE_CALL_PTRCALL_NODE_PATH,
		OPCODE_CALL_PTRCALL_RID,
		OPCODE_CALL_PTRCALL_OBJECT,
		OPCODE_CALL_PTRCALL_CALLABLE,
		OPCODE_CALL_PTRCALL_SIGNAL,
		OPCODE_CALL_PTRCALL_DICTIONARY,
		OPCODE_CALL_PTRCALL_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_BYTE_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_INT32_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_INT64_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_FLOAT32_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_FLOAT64_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_STRING_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_VECTOR2_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_VECTOR3_ARRAY,
		OPCODE_CALL_PTRCALL_PACKED_COLOR_ARRAY,
		OPCODE_AWAIT,
		OPCODE_AWAIT_RESUME,
		OPCODE_JUMP,
		OPCODE_JUMP_IF,
		OPCODE_JUMP_IF_NOT,
		OPCODE_JUMP_TO_DEF_ARGUMENT,
		OPCODE_RETURN,
		OPCODE_ITERATE_BEGIN,
		OPCODE_ITERATE_BEGIN_INT,
		OPCODE_ITERATE_BEGIN_FLOAT,
		OPCODE_ITERATE_BEGIN_VECTOR2,
		OPCODE_ITERATE_BEGIN_VECTOR2I,
		OPCODE_ITERATE_BEGIN_VECTOR3,
		OPCODE_ITERATE_BEGIN_VECTOR3I,
		OPCODE_ITERATE_BEGIN_STRING,
		OPCODE_ITERATE_BEGIN_DICTIONARY,
		OPCODE_ITERATE_BEGIN_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_BYTE_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_INT32_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_INT64_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_FLOAT32_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_FLOAT64_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_STRING_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_VECTOR2_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_VECTOR3_ARRAY,
		OPCODE_ITERATE_BEGIN_PACKED_COLOR_ARRAY,
		OPCODE_ITERATE_BEGIN_OBJECT,
		OPCODE_ITERATE,
		OPCODE_ITERATE_INT,
		OPCODE_ITERATE_FLOAT,
		OPCODE_ITERATE_VECTOR2,
		OPCODE_ITERATE_VECTOR2I,
		OPCODE_ITERATE_VECTOR3,
		OPCODE_ITERATE_VECTOR3I,
		OPCODE_ITERATE_STRING,
		OPCODE_ITERATE_DICTIONARY,
		OPCODE_ITERATE_ARRAY,
		OPCODE_ITERATE_PACKED_BYTE_ARRAY,
		OPCODE_ITERATE_PACKED_INT32_ARRAY,
		OPCODE_ITERATE_PACKED_INT64_ARRAY,
		OPCODE_ITERATE_PACKED_FLOAT32_ARRAY,
		OPCODE_ITERATE_PACKED_FLOAT64_ARRAY,
		OPCODE_ITERATE_PACKED_STRING_ARRAY,
		OPCODE_ITERATE_PACKED_VECTOR2_ARRAY,
		OPCODE_ITERATE_PACKED_VECTOR3_ARRAY,
		OPCODE_ITERATE_PACKED_COLOR_ARRAY,
		OPCODE_ITERATE_OBJECT,
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
		ADDR_TYPE_NAMED_GLOBAL = 8,
		ADDR_TYPE_NIL = 9
	};

	enum Instruction {
		INSTR_BITS = 20,
		INSTR_MASK = ((1 << INSTR_BITS) - 1),
		INSTR_ARGS_MASK = ~INSTR_MASK,
	};

	struct StackDebug {
		int line;
		int pos;
		bool added;
		StringName identifier;
	};

private:
	friend class GDScriptCompiler;
	friend class GDScriptByteCodeGenerator;

	StringName source;

	mutable Variant nil;
	mutable Variant *_constants_ptr = nullptr;
	int _constant_count = 0;
	const StringName *_global_names_ptr = nullptr;
	int _global_names_count = 0;
	const int *_default_arg_ptr = nullptr;
	int _default_arg_count = 0;
	int _operator_funcs_count = 0;
	const Variant::ValidatedOperatorEvaluator *_operator_funcs_ptr = nullptr;
	int _setters_count = 0;
	const Variant::ValidatedSetter *_setters_ptr = nullptr;
	int _getters_count = 0;
	const Variant::ValidatedGetter *_getters_ptr = nullptr;
	int _keyed_setters_count = 0;
	const Variant::ValidatedKeyedSetter *_keyed_setters_ptr = nullptr;
	int _keyed_getters_count = 0;
	const Variant::ValidatedKeyedGetter *_keyed_getters_ptr = nullptr;
	int _indexed_setters_count = 0;
	const Variant::ValidatedIndexedSetter *_indexed_setters_ptr = nullptr;
	int _indexed_getters_count = 0;
	const Variant::ValidatedIndexedGetter *_indexed_getters_ptr = nullptr;
	int _builtin_methods_count = 0;
	const Variant::ValidatedBuiltInMethod *_builtin_methods_ptr = nullptr;
	int _constructors_count = 0;
	const Variant::ValidatedConstructor *_constructors_ptr = nullptr;
	int _methods_count = 0;
	MethodBind **_methods_ptr = nullptr;
	const int *_code_ptr = nullptr;
	int _code_size = 0;
	int _argument_count = 0;
	int _stack_size = 0;
	int _instruction_args_size = 0;
	int _ptrcall_args_size = 0;

	int _initial_line = 0;
	bool _static = false;
	MultiplayerAPI::RPCMode rpc_mode = MultiplayerAPI::RPC_MODE_DISABLED;

	GDScript *_script = nullptr;

	StringName name;
	Vector<Variant> constants;
	Vector<StringName> global_names;
	Vector<int> default_arguments;
	Vector<Variant::ValidatedOperatorEvaluator> operator_funcs;
	Vector<Variant::ValidatedSetter> setters;
	Vector<Variant::ValidatedGetter> getters;
	Vector<Variant::ValidatedKeyedSetter> keyed_setters;
	Vector<Variant::ValidatedKeyedGetter> keyed_getters;
	Vector<Variant::ValidatedIndexedSetter> indexed_setters;
	Vector<Variant::ValidatedIndexedGetter> indexed_getters;
	Vector<Variant::ValidatedBuiltInMethod> builtin_methods;
	Vector<Variant::ValidatedConstructor> constructors;
	Vector<MethodBind *> methods;
	Vector<int> code;
	Vector<GDScriptDataType> argument_types;
	GDScriptDataType return_type;

#ifdef TOOLS_ENABLED
	Vector<StringName> arg_names;
	Vector<Variant> default_arg_values;
#endif

	List<StackDebug> stack_debug;

	_FORCE_INLINE_ Variant *_get_variant(int p_address, GDScriptInstance *p_instance, GDScript *p_script, Variant &self, Variant &static_ref, Variant *p_stack, String &r_error) const;
	_FORCE_INLINE_ String _get_call_error(const Callable::CallError &p_err, const String &p_where, const Variant **argptrs) const;

	friend class GDScriptLanguage;

	SelfList<GDScriptFunction> function_list{ this };
#ifdef DEBUG_ENABLED
	CharString func_cname;
	const char *_func_cname = nullptr;

	struct Profile {
		StringName signature;
		uint64_t call_count = 0;
		uint64_t self_time = 0;
		uint64_t total_time = 0;
		uint64_t frame_call_count = 0;
		uint64_t frame_self_time = 0;
		uint64_t frame_total_time = 0;
		uint64_t last_frame_call_count = 0;
		uint64_t last_frame_self_time = 0;
		uint64_t last_frame_total_time = 0;
	} profile;

#endif

public:
	struct CallState {
		GDScript *script;
		GDScriptInstance *instance;
#ifdef DEBUG_ENABLED
		StringName function_name;
		String script_path;
#endif
		Vector<uint8_t> stack;
		int stack_size;
		Variant self;
		uint32_t alloca_size;
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
	GDScriptDataType get_return_type() const;
	GDScriptDataType get_argument_type(int p_idx) const;
	GDScript *get_script() const { return _script; }
	StringName get_source() const { return source; }

	void debug_get_stack_member_state(int p_line, List<Pair<StringName, int>> *r_stackvars) const;

	_FORCE_INLINE_ bool is_empty() const { return _code_size == 0; }

	int get_argument_count() const { return _argument_count; }
	StringName get_argument_name(int p_idx) const {
#ifdef TOOLS_ENABLED
		ERR_FAIL_INDEX_V(p_idx, arg_names.size(), StringName());
		return arg_names[p_idx];
#else
		return StringName();
#endif
	}
	Variant get_default_argument(int p_idx) const {
		ERR_FAIL_INDEX_V(p_idx, default_arguments.size(), Variant());
		return default_arguments[p_idx];
	}
#ifdef TOOLS_ENABLED
	const Vector<Variant> &get_default_arg_values() const {
		return default_arg_values;
	}
#endif // TOOLS_ENABLED

	Variant call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, CallState *p_state = nullptr);

#ifdef DEBUG_ENABLED
	void disassemble(const Vector<String> &p_code_lines) const;
#endif

	_FORCE_INLINE_ MultiplayerAPI::RPCMode get_rpc_mode() const { return rpc_mode; }
	GDScriptFunction();
	~GDScriptFunction();
};

class GDScriptFunctionState : public Reference {
	GDCLASS(GDScriptFunctionState, Reference);
	friend class GDScriptFunction;
	GDScriptFunction *function;
	GDScriptFunction::CallState state;
	Variant _signal_callback(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
	Ref<GDScriptFunctionState> first_state;

	SelfList<GDScriptFunctionState> scripts_list;
	SelfList<GDScriptFunctionState> instances_list;

protected:
	static void _bind_methods();

public:
	bool is_valid(bool p_extended_check = false) const;
	Variant resume(const Variant &p_arg = Variant());

	void _clear_stack();

	GDScriptFunctionState();
	~GDScriptFunctionState();
};

#endif // GDSCRIPT_FUNCTION_H
