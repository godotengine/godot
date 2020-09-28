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

#include "core/os/thread.h"
#include "core/pair.h"
#include "core/reference.h"
#include "core/script_language.h"
#include "core/self_list.h"
#include "core/string_name.h"
#include "core/variant.h"

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

#define OPCODE_OP_NUMBER(m_op)            \
	OPCODE_OP_##m_op##_INT_INT,           \
			OPCODE_OP_##m_op##_INT_FLOAT, \
			OPCODE_OP_##m_op##_FLOAT_INT, \
			OPCODE_OP_##m_op##_FLOAT_FLOAT

#define OPCODE_OP_VECTOR(m_op)                    \
	OPCODE_OP_##m_op##_VECTOR2_VECTOR2,           \
			OPCODE_OP_##m_op##_VECTOR2I_VECTOR2I, \
			OPCODE_OP_##m_op##_VECTOR3_VECTOR3,   \
			OPCODE_OP_##m_op##_VECTOR3I_VECTOR3I

#define OPCODE_OP_ARRAYS(m_op)                                            \
	OPCODE_OP_##m_op##_ARRAY_ARRAY,                                       \
			OPCODE_OP_##m_op##_PACKED_BYTE_ARRAY_PACKED_BYTE_ARRAY,       \
			OPCODE_OP_##m_op##_PACKED_INT32_ARRAY_PACKED_INT32_ARRAY,     \
			OPCODE_OP_##m_op##_PACKED_INT64_ARRAY_PACKED_INT64_ARRAY,     \
			OPCODE_OP_##m_op##_PACKED_FLOAT32_ARRAY_PACKED_FLOAT32_ARRAY, \
			OPCODE_OP_##m_op##_PACKED_FLOAT64_ARRAY_PACKED_FLOAT64_ARRAY, \
			OPCODE_OP_##m_op##_PACKED_STRING_ARRAY_PACKED_STRING_ARRAY,   \
			OPCODE_OP_##m_op##_PACKED_VECTOR2_ARRAY_PACKED_VECTOR2_ARRAY, \
			OPCODE_OP_##m_op##_PACKED_VECTOR3_ARRAY_PACKED_VECTOR3_ARRAY, \
			OPCODE_OP_##m_op##_PACKED_COLOR_ARRAY_PACKED_COLOR_ARRAY

#define OPCODE_OP_TYPE_NUMBER(m_op, m_type)    \
	OPCODE_OP_##m_op##_##m_type##_##m_type,    \
			OPCODE_OP_##m_op##_##m_type##_INT, \
			OPCODE_OP_##m_op##_##m_type##_FLOAT

#define OPCODE_OP_TYPE_NUMBER_REV(m_op, m_type) \
	OPCODE_OP_##m_op##_INT_##m_type,            \
			OPCODE_OP_##m_op##_FLOAT_##m_type

#define OPCODE_ALL_TYPES(m_op)                    \
	OPCODE_##m_op##_BOOL,                         \
			OPCODE_##m_op##_INT,                  \
			OPCODE_##m_op##_FLOAT,                \
			OPCODE_##m_op##_STRING,               \
			OPCODE_##m_op##_VECTOR2,              \
			OPCODE_##m_op##_VECTOR2I,             \
			OPCODE_##m_op##_VECTOR3,              \
			OPCODE_##m_op##_VECTOR3I,             \
			OPCODE_##m_op##_TRANSFORM2D,          \
			OPCODE_##m_op##_PLANE,                \
			OPCODE_##m_op##_QUAT,                 \
			OPCODE_##m_op##_AABB,                 \
			OPCODE_##m_op##_BASIS,                \
			OPCODE_##m_op##_TRANSFORM,            \
			OPCODE_##m_op##_COLOR,                \
			OPCODE_##m_op##_STRING_NAME,          \
			OPCODE_##m_op##_RID,                  \
			OPCODE_##m_op##_OBJECT,               \
			OPCODE_##m_op##_CALLABLE,             \
			OPCODE_##m_op##_SIGNAL,               \
			OPCODE_##m_op##_DICTIONARY,           \
			OPCODE_##m_op##_ARRAY,                \
			OPCODE_##m_op##_PACKED_BYTE_ARRAY,    \
			OPCODE_##m_op##_PACKED_INT32_ARRAY,   \
			OPCODE_##m_op##_PACKED_INT64_ARRAY,   \
			OPCODE_##m_op##_PACKED_FLOAT32_ARRAY, \
			OPCODE_##m_op##_PACKED_FLOAT64_ARRAY, \
			OPCODE_##m_op##_PACKED_STRING_ARRAY,  \
			OPCODE_##m_op##_PACKED_VECTOR2_ARRAY, \
			OPCODE_##m_op##_PACKED_VECTOR3_ARRAY, \
			OPCODE_##m_op##_PACKED_COLOR_ARRAY

class GDScriptFunction {
public:
	enum Opcode {
		OPCODE_OPERATOR,

		// Typed operators.
		// Addition.
		OPCODE_OP_NUMBER(ADD),
		OPCODE_OP_VECTOR(ADD),
		OPCODE_OP_ADD_QUAT_QUAT,
		OPCODE_OP_ADD_COLOR_COLOR,
		OPCODE_OP_CONCAT_STRING_STRING,
		OPCODE_OP_ARRAYS(CONCAT),
		// Subtraction.
		OPCODE_OP_NUMBER(SUBTRACT),
		OPCODE_OP_VECTOR(SUBTRACT),
		OPCODE_OP_SUBTRACT_QUAT_QUAT,
		OPCODE_OP_SUBTRACT_COLOR_COLOR,
		// Multiplication.
		OPCODE_OP_NUMBER(MULTIPLY),
		OPCODE_OP_MULTIPLY_TRANSFORM2D_TRANSFORM2D,
		OPCODE_OP_MULTIPLY_TRANSFORM2D_VECTOR2,
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, QUAT),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, QUAT),
		OPCODE_OP_MULTIPLY_QUAT_VECTOR3,
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, VECTOR2),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, VECTOR2),
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, VECTOR2I),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, VECTOR2I),
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, VECTOR3),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, VECTOR3),
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, VECTOR3I),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, VECTOR3I),
		OPCODE_OP_TYPE_NUMBER(MULTIPLY, COLOR),
		OPCODE_OP_TYPE_NUMBER_REV(MULTIPLY, COLOR),
		// Division.
		OPCODE_OP_NUMBER(DIVIDE),
		OPCODE_OP_TYPE_NUMBER(DIVIDE, VECTOR2),
		OPCODE_OP_TYPE_NUMBER(DIVIDE, VECTOR2I),
		OPCODE_OP_TYPE_NUMBER(DIVIDE, VECTOR3),
		OPCODE_OP_TYPE_NUMBER(DIVIDE, VECTOR3I),
		OPCODE_OP_TYPE_NUMBER(DIVIDE, COLOR),
		OPCODE_OP_DIVIDE_QUAT_FLOAT,
		// Modulo.
		OPCODE_OP_MODULO_INT_INT,
		// Unary operators.
		// Negate.
		OPCODE_OP_NEGATE_INT,
		OPCODE_OP_NEGATE_FLOAT,
		OPCODE_OP_NEGATE_VECTOR2,
		OPCODE_OP_NEGATE_VECTOR2I,
		OPCODE_OP_NEGATE_VECTOR3,
		OPCODE_OP_NEGATE_VECTOR3I,
		OPCODE_OP_NEGATE_QUAT,
		OPCODE_OP_NEGATE_COLOR,
		// Positive does nothing, so it's not an operator.
		// Bitwise operators.
		OPCODE_OP_BIT_NEGATE_INT,
		OPCODE_OP_BIT_AND_INT_INT,
		OPCODE_OP_BIT_OR_INT_INT,
		OPCODE_OP_BIT_XOR_INT_INT,
		OPCODE_OP_SHIFT_LEFT,
		OPCODE_OP_SHIFT_RIGHT,
		// Logic operators.
		OPCODE_OP_NOT,
		OPCODE_OP_AND,
		OPCODE_OP_OR,
		// Comparison operators.
		OPCODE_ALL_TYPES(OP_EQUAL),
		// Equal.
		// Numbers can be swapped.
		OPCODE_OP_EQUAL_INT_FLOAT,
		OPCODE_OP_EQUAL_FLOAT_INT,
		// String is compatible with NodePath and StringName.
		OPCODE_OP_EQUAL_STRING_STRING_NAME,
		OPCODE_OP_EQUAL_STRING_NAME_STRING,
		OPCODE_OP_EQUAL_STRING_NODE_PATH,
		OPCODE_OP_EQUAL_NODE_PATH_STRING,
		// Not equal.
		OPCODE_ALL_TYPES(OP_NOT_EQUAL),
		// Numbers can be swapped.
		OPCODE_OP_NOT_EQUAL_INT_FLOAT,
		OPCODE_OP_NOT_EQUAL_FLOAT_INT,
		// String is compatible with NodePath and StringName.
		OPCODE_OP_NOT_EQUAL_STRING_STRING_NAME,
		OPCODE_OP_NOT_EQUAL_STRING_NAME_STRING,
		OPCODE_OP_NOT_EQUAL_STRING_NODE_PATH,
		OPCODE_OP_NOT_EQUAL_NODE_PATH_STRING,
		// Less than.
		OPCODE_OP_LESS_BOOL_BOOL,
		OPCODE_OP_NUMBER(LESS),
		OPCODE_OP_VECTOR(LESS),
		// Less than or equal to.
		OPCODE_OP_NUMBER(LESS_EQUAL),
		OPCODE_OP_VECTOR(LESS_EQUAL),
		// Greater than.
		OPCODE_OP_GREATER_BOOL_BOOL,
		OPCODE_OP_NUMBER(GREATER),
		OPCODE_OP_VECTOR(GREATER),
		// Greater than or equal to.
		OPCODE_OP_NUMBER(GREATER_EQUAL),
		OPCODE_OP_VECTOR(GREATER_EQUAL),

		OPCODE_EXTENDS_TEST,
		OPCODE_IS_BUILTIN,
		OPCODE_SET,
		OPCODE_GET,
		// Typed get: base and index types.
		OPCODE_GET_STRING_INT,
		OPCODE_GET_STRING_FLOAT,
		OPCODE_GET_VECTOR2_INT,
		OPCODE_GET_VECTOR2_FLOAT,
		OPCODE_GET_VECTOR2_STRING,
		OPCODE_GET_VECTOR2I_INT,
		OPCODE_GET_VECTOR2I_FLOAT,
		OPCODE_GET_VECTOR2I_STRING,
		OPCODE_GET_VECTOR3_INT,
		OPCODE_GET_VECTOR3_FLOAT,
		OPCODE_GET_VECTOR3_STRING,
		OPCODE_GET_VECTOR3I_INT,
		OPCODE_GET_VECTOR3I_FLOAT,
		OPCODE_GET_VECTOR3I_STRING,
		OPCODE_GET_RECT2_STRING,
		OPCODE_GET_RECT2I_STRING,
		OPCODE_GET_TRANSFORM_INT,
		OPCODE_GET_TRANSFORM_FLOAT,
		OPCODE_GET_TRANSFORM_STRING,
		OPCODE_GET_TRANSFORM2D_INT,
		OPCODE_GET_TRANSFORM2D_FLOAT,
		OPCODE_GET_TRANSFORM2D_STRING,
		OPCODE_GET_PLANE_STRING,
		OPCODE_GET_QUAT_STRING,
		OPCODE_GET_AABB_STRING,
		OPCODE_GET_BASIS_INT,
		OPCODE_GET_BASIS_FLOAT,
		OPCODE_GET_BASIS_STRING,
		OPCODE_GET_COLOR_INT,
		OPCODE_GET_COLOR_FLOAT,
		OPCODE_GET_COLOR_STRING,
		OPCODE_GET_OBJECT_STRING,

		OPCODE_SET_NAMED,
		OPCODE_GET_NAMED,
		// Typed get named: source type.
		OPCODE_GET_NAMED_VECTOR2,
		OPCODE_GET_NAMED_VECTOR2I,
		OPCODE_GET_NAMED_VECTOR3,
		OPCODE_GET_NAMED_VECTOR3I,
		OPCODE_GET_NAMED_RECT2,
		OPCODE_GET_NAMED_RECT2I,
		OPCODE_GET_NAMED_TRANSFORM,
		OPCODE_GET_NAMED_TRANSFORM2D,
		OPCODE_GET_NAMED_PLANE,
		OPCODE_GET_NAMED_QUAT,
		OPCODE_GET_NAMED_BASIS,
		OPCODE_GET_NAMED_AABB,
		OPCODE_GET_NAMED_COLOR,
		OPCODE_GET_NAMED_OBJECT,

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
		OPCODE_CONSTRUCT, //only for basic types!!
		OPCODE_CONSTRUCT_ARRAY,
		OPCODE_CONSTRUCT_DICTIONARY,
		OPCODE_CALL,
		OPCODE_CALL_RETURN,
		OPCODE_CALL_ASYNC,
		OPCODE_CALL_BUILT_IN,
		OPCODE_CALL_SELF_BASE,
		OPCODE_CALL_METHOD_BIND,
		OPCODE_CALL_METHOD_BIND_RET,
		OPCODE_AWAIT,
		OPCODE_AWAIT_RESUME,
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

#undef OPCODE_OP_NUMBER
#undef OPCODE_OP_VECTOR
#undef OPCODE_OP_ARRAYS
#undef OPCODE_OP_TYPE_NUMBER
#undef OPCODE_OP_TYPE_NUMBER_REV
#undef OPCODE_ALL_TYPES

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
	MultiplayerAPI::RPCMode rpc_mode;

	GDScript *_script;

	StringName name;
	Vector<Variant> constants;
	Vector<StringName> global_names;
	Vector<int> default_arguments;
	Vector<int> code;
	Vector<GDScriptDataType> argument_types;
	GDScriptDataType return_type;

#ifdef TOOLS_ENABLED
	Vector<StringName> arg_names;
#endif

	List<StackDebug> stack_debug;

	_FORCE_INLINE_ Variant *_get_variant(int p_address, GDScriptInstance *p_instance, GDScript *p_script, Variant &self, Variant &static_ref, Variant *p_stack, String &r_error) const;
	_FORCE_INLINE_ String _get_call_error(const Callable::CallError &p_err, const String &p_where, const Variant **argptrs) const;

	friend class GDScriptLanguage;

	SelfList<GDScriptFunction> function_list;
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
