/**************************************************************************/
/*  gdscript_function.h                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef GDSCRIPT_FUNCTION_H
#define GDSCRIPT_FUNCTION_H

#include "gdscript_utility_functions.h"

#include "core/object/ref_counted.h"
#include "core/object/script_language.h"
#include "core/os/thread.h"
#include "core/string/string_name.h"
#include "core/templates/pair.h"
#include "core/templates/self_list.h"
#include "core/variant/variant.h"

#define OP_ARGS (GDScriptInstance * p_instance,                        \
		int *p_variant_address_limits,                                 \
		Variant *p_variant_addresses[GDScriptFunction::ADDR_TYPE_MAX], \
		int p_ip,                                                      \
		String p_err_text)
#define OP_EXEC_H(m_opcode) void _exec_##m_opcode OP_ARGS

class GDScriptInstance;
class GDScript;

class GDScriptDataType {
public:
	Vector<GDScriptDataType> container_element_types;

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
				if (valid && builtin_type == Variant::ARRAY && has_container_element_type(0)) {
					Array array = p_variant;
					if (array.is_typed()) {
						const GDScriptDataType &elem_type = container_element_types[0];
						Variant::Type array_builtin_type = (Variant::Type)array.get_typed_builtin();
						StringName array_native_type = array.get_typed_class_name();
						Ref<Script> array_script_type_ref = array.get_typed_script();

						if (array_script_type_ref.is_valid()) {
							valid = (elem_type.kind == SCRIPT || elem_type.kind == GDSCRIPT) && elem_type.script_type == array_script_type_ref.ptr();
						} else if (array_native_type != StringName()) {
							valid = elem_type.kind == NATIVE && elem_type.native_type == array_native_type;
						} else {
							valid = elem_type.kind == BUILTIN && elem_type.builtin_type == array_builtin_type;
						}
					} else {
						valid = false;
					}
				} else if (valid && builtin_type == Variant::DICTIONARY && has_container_element_types()) {
					Dictionary dictionary = p_variant;
					if (dictionary.is_typed()) {
						if (dictionary.is_typed_key()) {
							GDScriptDataType key = get_container_element_type_or_variant(0);
							Variant::Type key_builtin_type = (Variant::Type)dictionary.get_typed_key_builtin();
							StringName key_native_type = dictionary.get_typed_key_class_name();
							Ref<Script> key_script_type_ref = dictionary.get_typed_key_script();

							if (key_script_type_ref.is_valid()) {
								valid = (key.kind == SCRIPT || key.kind == GDSCRIPT) && key.script_type == key_script_type_ref.ptr();
							} else if (key_native_type != StringName()) {
								valid = key.kind == NATIVE && key.native_type == key_native_type;
							} else {
								valid = key.kind == BUILTIN && key.builtin_type == key_builtin_type;
							}
						}

						if (valid && dictionary.is_typed_value()) {
							GDScriptDataType value = get_container_element_type_or_variant(1);
							Variant::Type value_builtin_type = (Variant::Type)dictionary.get_typed_value_builtin();
							StringName value_native_type = dictionary.get_typed_value_class_name();
							Ref<Script> value_script_type_ref = dictionary.get_typed_value_script();

							if (value_script_type_ref.is_valid()) {
								valid = (value.kind == SCRIPT || value.kind == GDSCRIPT) && value.script_type == value_script_type_ref.ptr();
							} else if (value_native_type != StringName()) {
								valid = value.kind == NATIVE && value.native_type == value_native_type;
							} else {
								valid = value.kind == BUILTIN && value.builtin_type == value_builtin_type;
							}
						}
					} else {
						valid = false;
					}
				} else if (!valid && p_allow_implicit_conversion) {
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

				bool was_freed = false;
				Object *obj = p_variant.get_validated_object_with_check(was_freed);
				if (!obj) {
					return !was_freed;
				}

				if (!ClassDB::is_parent_class(obj->get_class_name(), native_type)) {
					return false;
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

				bool was_freed = false;
				Object *obj = p_variant.get_validated_object_with_check(was_freed);
				if (!obj) {
					return !was_freed;
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

	bool can_contain_object() const {
		if (has_type && kind == BUILTIN) {
			switch (builtin_type) {
				case Variant::ARRAY:
					if (has_container_element_type(0)) {
						return container_element_types[0].can_contain_object();
					}
					return true;
				case Variant::DICTIONARY:
					if (has_container_element_types()) {
						return get_container_element_type_or_variant(0).can_contain_object() || get_container_element_type_or_variant(1).can_contain_object();
					}
					return true;
				case Variant::NIL:
				case Variant::OBJECT:
					return true;
				default:
					return false;
			}
		}
		return true;
	}

	void set_container_element_type(int p_index, const GDScriptDataType &p_element_type) {
		ERR_FAIL_COND(p_index < 0);
		while (p_index >= container_element_types.size()) {
			container_element_types.push_back(GDScriptDataType());
		}
		container_element_types.write[p_index] = GDScriptDataType(p_element_type);
	}

	GDScriptDataType get_container_element_type(int p_index) const {
		ERR_FAIL_INDEX_V(p_index, container_element_types.size(), GDScriptDataType());
		return container_element_types[p_index];
	}

	GDScriptDataType get_container_element_type_or_variant(int p_index) const {
		if (p_index < 0 || p_index >= container_element_types.size()) {
			return GDScriptDataType();
		}
		return container_element_types[p_index];
	}

	bool has_container_element_type(int p_index) const {
		return p_index >= 0 && p_index < container_element_types.size();
	}

	bool has_container_element_types() const {
		return !container_element_types.is_empty();
	}

	GDScriptDataType() = default;

	void operator=(const GDScriptDataType &p_other) {
		kind = p_other.kind;
		has_type = p_other.has_type;
		builtin_type = p_other.builtin_type;
		native_type = p_other.native_type;
		script_type = p_other.script_type;
		script_type_ref = p_other.script_type_ref;
		container_element_types = p_other.container_element_types;
	}

	GDScriptDataType(const GDScriptDataType &p_other) {
		*this = p_other;
	}

	~GDScriptDataType() {}
};

class GDScriptFunction {
public:
	/*
	TODO:
	- What does validated mean?
	*/
	enum Opcode {
		OPCODE_OPERATOR, // Can be Validated
		OPCODE_TYPE_TEST, // Args is TestArguments
		OPCODE_SET_KEYED, // Can be Validated
		OPCODE_SET_INDEXED, // Only Validated
		OPCODE_GET_KEYED, // Can be Validated
		OPCODE_GET_INDEXED, // Only Validated
		OPCODE_SET_NAMED, // Can be Validated
		OPCODE_GET_NAMED, // Can be Validated
		OPCODE_SET_MEMBER,
		OPCODE_GET_MEMBER,
		OPCODE_SET_STATIC_VARIABLE, // Only for GDScript.
		OPCODE_GET_STATIC_VARIABLE, // Only for GDScript.
		OPCODE_ASSIGN, // Args is AssignArguments
		OPCODE_CAST, // Args is CastArgs
		OPCODE_CONSTRUCT, // Only for basic types! Args is ConstructArguments
		OPCODE_CALL, // Args is CallArguments
		OPCODE_AWAIT,
		OPCODE_AWAIT_RESUME,
		OPCODE_CREATE_LAMBDA,
		OPCODE_CREATE_SELF_LAMBDA,
		OPCODE_JUMP, // Args is JumpArgs
		OPCODE_RETURN, // Args is ReturnArgs
		OPCODE_ITERATE_BEGIN, // Args is IterateArguments
		OPCODE_ITERATE, // Args is IterateArguments
		OPCODE_STORE_GLOBAL,
		OPCODE_STORE_NAMED_GLOBAL,
		OPCODE_TYPE_ADJUST, // Args is AdjustArguments
		OPCODE_ASSERT,
		OPCODE_BREAKPOINT,
		OPCODE_LINE,
		OPCODE_END
	};

	enum Address {
		ADDR_BITS = 24,
		ADDR_MASK = ((1 << ADDR_BITS) - 1),
		ADDR_TYPE_MASK = ~ADDR_MASK,
		ADDR_TYPE_STACK = 0,
		ADDR_TYPE_CONSTANT = 1,
		ADDR_TYPE_MEMBER = 2,
		ADDR_TYPE_MAX = 3,
	};

	enum FixedAddresses {
		ADDR_STACK_SELF = 0,
		ADDR_STACK_CLASS = 1,
		ADDR_STACK_NIL = 2,
		FIXED_ADDRESSES_MAX = 3,
		ADDR_SELF = ADDR_STACK_SELF | (ADDR_TYPE_STACK << ADDR_BITS),
		ADDR_CLASS = ADDR_STACK_CLASS | (ADDR_TYPE_STACK << ADDR_BITS),
		ADDR_NIL = ADDR_STACK_NIL | (ADDR_TYPE_STACK << ADDR_BITS),
	};

	struct StackDebug {
		int line;
		int pos;
		bool added;
		StringName identifier;
	};

private:
	friend class GDScript;
	friend class GDScriptCompiler;
	friend class GDScriptByteCodeGenerator;
	friend class GDScriptLanguage;

	// Used for bit masking to find arguments.
	enum ArgumentMask {
		ARGUMENT = 0xFF >> 1,
		IS_VALIDATED = 0xFF ^ ARGUMENT,
	};

	// Arguments for Opcode::OPCODE_JUMP.
	enum JumpArgs {
		DEFAULT,
		IF,
		IF_NOT,
		DEF_ARGUMENT,
		IF_SHARED,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_CAST.
	enum CastArgs {
		BUILTIN,
		NATIVE,
		SCRIPT,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_TEST.
	enum TestArguments {
		TEST_BUILTIN,
		TEST_ARRAY,
		TEST_DICTIONARY,
		TEST_NATIVE,
		TEST_SCRIPT,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_ASSIGN.
	enum AssignArguments {
		OTHER,
		NULL_VAL,
		TRUE,
		FALSE,
		TYPED_BUILTIN,
		TYPED_ARRAY,
		TYPED_DICTIONARY,
		TYPED_NATIVE,
		TYPED_SCRIPT,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_CONSTRUCT.
	enum ConstructArguments {
		OTHER, // Can be validated <- Only for basic types!
		ARRAY,
		TYPED_ARRAY,
		DICTIONARY,
		TYPED_DICTIONARY,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_RETURN.
	enum ReturnArguments {
		OTHER,
		TYPED_BUILTIN,
		TYPED_ARRAY,
		TYPED_DICTIONARY,
		TYPED_NATIVE,
		TYPED_SCRIPT,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_CALL.
	enum CallArguments {
		OTHER,
		RETURN,
		ASYNC,
		UTILITY, // Can be Validated
		GDSCRIPT_UTILITY,
		BUILTIN_TYPE, // Only Validated
		SELF_BASE,
		METHOD_BIND,
		METHOD_BIND_RET,
		BUILTIN_STATIC,
		NATIVE_STATIC,
		NATIVE_STATIC_RETURN, // Only Validated
		NATIVE_STATIC_NO_RETURN, // Only Validated
		METHOD_BIND_RETURN, // Only Validated
		METHOD_BIND_NO_RETURN, // Only Validated
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_ITERATE and Opcode::OPCODE_ITERATE_BEGIN.
	enum IterateArguments {
		OTHER,
		INT,
		FLOAT,
		VECTOR2,
		VECTOR2I,
		VECTOR3,
		VECTOR3I,
		STRING,
		DICTIONARY,
		ARRAY,
		PACKED_BYTE_ARRAY,
		PACKED_INT32_ARRAY,
		PACKED_INT64_ARRAY,
		PACKED_FLOAT32_ARRAY,
		PACKED_FLOAT64_ARRAY,
		PACKED_STRING_ARRAY,
		PACKED_VECTOR2_ARRAY,
		PACKED_VECTOR3_ARRAY,
		PACKED_VECTOR4_ARRAY,
		PACKED_COLOR_ARRAY,
		OBJECT,
		ARGS_MAX,
	};

	// Arguments for Opcode::OPCODE_ADJUST.
	enum AdjustArguments {
		BOOL,
		INT,
		FLOAT,
		STRING,
		VECTOR2,
		VECTOR2I,
		RECT2,
		RECT2I,
		VECTOR3,
		VECTOR3I,
		VECTOR4,
		VECTOR4I,
		PLANE,
		QUATERNION,
		AABB,
		BASIS,
		TRANSFORM2D,
		TRANSFORM3D,
		PROJECTION,
		COLOR,
		STRING_NAME,
		NODE_PATH,
		RID,
		OBJECT,
		CALLABLE,
		SIGNAL,
		DICTIONARY,
		ARRAY,
		PACKED_BYTE_ARRAY,
		PACKED_INT32_ARRAY,
		PACKED_INT64_ARRAY,
		PACKED_FLOAT32_ARRAY,
		PACKED_FLOAT64_ARRAY,
		PACKED_STRING_ARRAY,
		PACKED_VECTOR2_ARRAY,
		PACKED_VECTOR3_ARRAY,
		PACKED_VECTOR4_ARRAY,
		PACKED_COLOR_ARRAY,
		ARGS_MAX,
	};

	StringName name;
	StringName source;
	bool _static = false;
	Vector<GDScriptDataType> argument_types;
	GDScriptDataType return_type;
	MethodInfo method_info;
	Variant rpc_config;

	GDScript *_script = nullptr;
	int _initial_line = 0;
	int _argument_count = 0;
	int _stack_size = 0;
	int _instruction_args_size = 0;

	SelfList<GDScriptFunction> function_list{ this };
	mutable Variant nil;
	HashMap<int, Variant::Type> temporary_slots;
	List<StackDebug> stack_debug;

	Vector<int> code;
	Vector<int> default_arguments;
	Vector<Variant> constants;
	Vector<StringName> global_names;
	Vector<Variant::ValidatedOperatorEvaluator> operator_funcs;
	Vector<Variant::ValidatedSetter> setters;
	Vector<Variant::ValidatedGetter> getters;
	Vector<Variant::ValidatedKeyedSetter> keyed_setters;
	Vector<Variant::ValidatedKeyedGetter> keyed_getters;
	Vector<Variant::ValidatedIndexedSetter> indexed_setters;
	Vector<Variant::ValidatedIndexedGetter> indexed_getters;
	Vector<Variant::ValidatedBuiltInMethod> builtin_methods;
	Vector<Variant::ValidatedConstructor> constructors;
	Vector<Variant::ValidatedUtilityFunction> utilities;
	Vector<GDScriptUtilityFunctions::FunctionPtr> gds_utilities;
	Vector<MethodBind *> methods;
	Vector<GDScriptFunction *> lambdas;

	int _code_size = 0;
	int _default_arg_count = 0;
	int _constant_count = 0;
	int _global_names_count = 0;
	int _operator_funcs_count = 0;
	int _setters_count = 0;
	int _getters_count = 0;
	int _keyed_setters_count = 0;
	int _keyed_getters_count = 0;
	int _indexed_setters_count = 0;
	int _indexed_getters_count = 0;
	int _builtin_methods_count = 0;
	int _constructors_count = 0;
	int _utilities_count = 0;
	int _gds_utilities_count = 0;
	int _methods_count = 0;
	int _lambdas_count = 0;

	int *_code_ptr = nullptr;
	const int *_default_arg_ptr = nullptr;
	mutable Variant *_constants_ptr = nullptr;
	const StringName *_global_names_ptr = nullptr;
	const Variant::ValidatedOperatorEvaluator *_operator_funcs_ptr = nullptr;
	const Variant::ValidatedSetter *_setters_ptr = nullptr;
	const Variant::ValidatedGetter *_getters_ptr = nullptr;
	const Variant::ValidatedKeyedSetter *_keyed_setters_ptr = nullptr;
	const Variant::ValidatedKeyedGetter *_keyed_getters_ptr = nullptr;
	const Variant::ValidatedIndexedSetter *_indexed_setters_ptr = nullptr;
	const Variant::ValidatedIndexedGetter *_indexed_getters_ptr = nullptr;
	const Variant::ValidatedBuiltInMethod *_builtin_methods_ptr = nullptr;
	const Variant::ValidatedConstructor *_constructors_ptr = nullptr;
	const Variant::ValidatedUtilityFunction *_utilities_ptr = nullptr;
	const GDScriptUtilityFunctions::FunctionPtr *_gds_utilities_ptr = nullptr;
	MethodBind **_methods_ptr = nullptr;
	GDScriptFunction **_lambdas_ptr = nullptr;

	OP_EXEC_H(OPCODE_OPERATOR);
	OP_EXEC_H(OPCODE_TYPE_TEST);
	OP_EXEC_H(OPCODE_SET_KEYED);
	OP_EXEC_H(OPCODE_SET_INDEXED);
	OP_EXEC_H(OPCODE_GET_KEYED);
	OP_EXEC_H(OPCODE_GET_INDEXED);
	OP_EXEC_H(OPCODE_SET_NAMED);
	OP_EXEC_H(OPCODE_GET_NAMED);
	OP_EXEC_H(OPCODE_SET_MEMBER);
	OP_EXEC_H(OPCODE_GET_MEMBER);
	OP_EXEC_H(OPCODE_SET_STATIC_VARIABLE);
	OP_EXEC_H(OPCODE_GET_STATIC_VARIABLE);
	OP_EXEC_H(OPCODE_ASSIGN);
	OP_EXEC_H(OPCODE_CAST);
	OP_EXEC_H(OPCODE_CONSTRUCT);
	OP_EXEC_H(OPCODE_CALL);
	OP_EXEC_H(OPCODE_AWAIT);
	OP_EXEC_H(OPCODE_AWAIT_RESUME);
	OP_EXEC_H(OPCODE_CREATE_LAMBDA);
	OP_EXEC_H(OPCODE_CREATE_SELF_LAMBDA);
	OP_EXEC_H(OPCODE_JUMP);
	OP_EXEC_H(OPCODE_RETURN);
	OP_EXEC_H(OPCODE_ITERATE_BEGIN);
	OP_EXEC_H(OPCODE_ITERATE);
	OP_EXEC_H(OPCODE_STORE_GLOBAL);
	OP_EXEC_H(OPCODE_STORE_NAMED_GLOBAL);
	OP_EXEC_H(OPCODE_TYPE_ADJUST);
	OP_EXEC_H(OPCODE_ASSERT);
	OP_EXEC_H(OPCODE_BREAKPOINT);
	OP_EXEC_H(OPCODE_LINE);

#ifdef DEBUG_ENABLED
	CharString func_cname;
	const char *_func_cname = nullptr;

	Vector<String> operator_names;
	Vector<String> setter_names;
	Vector<String> getter_names;
	Vector<String> builtin_methods_names;
	Vector<String> constructors_names;
	Vector<String> utilities_names;
	Vector<String> gds_utilities_names;

	struct Profile {
		StringName signature;
		SafeNumeric<uint64_t> call_count;
		SafeNumeric<uint64_t> self_time;
		SafeNumeric<uint64_t> total_time;
		SafeNumeric<uint64_t> frame_call_count;
		SafeNumeric<uint64_t> frame_self_time;
		SafeNumeric<uint64_t> frame_total_time;
		uint64_t last_frame_call_count = 0;
		uint64_t last_frame_self_time = 0;
		uint64_t last_frame_total_time = 0;
		typedef struct NativeProfile {
			uint64_t call_count;
			uint64_t total_time;
			String signature;
		} NativeProfile;
		HashMap<String, NativeProfile> native_calls;
		HashMap<String, NativeProfile> last_native_calls;
	} profile;
#endif

	_FORCE_INLINE_ String _get_call_error(const String &p_where, const Variant **p_argptrs, const Variant &p_ret, const Callable::CallError &p_err) const;
	Variant _get_default_variant_for_data_type(const GDScriptDataType &p_data_type);

public:
	static constexpr int MAX_CALL_DEPTH = 2048; // Limit to try to avoid crash because of a stack overflow.

	struct CallState {
		GDScript *script = nullptr;
		GDScriptInstance *instance = nullptr;
#ifdef DEBUG_ENABLED
		StringName function_name;
		String script_path;
#endif
		Vector<uint8_t> stack;
		int stack_size = 0;
		uint32_t alloca_size = 0;
		int ip = 0;
		int line = 0;
		int defarg = 0;
		Variant result;
	};

	_FORCE_INLINE_ StringName get_name() const { return name; }
	_FORCE_INLINE_ StringName get_source() const { return source; }
	_FORCE_INLINE_ GDScript *get_script() const { return _script; }
	_FORCE_INLINE_ bool is_static() const { return _static; }
	_FORCE_INLINE_ MethodInfo get_method_info() const { return method_info; }
	_FORCE_INLINE_ int get_argument_count() const { return _argument_count; }
	_FORCE_INLINE_ Variant get_rpc_config() const { return rpc_config; }
	_FORCE_INLINE_ int get_max_stack_size() const { return _stack_size; }

	Variant get_constant(int p_idx) const;
	StringName get_global_name(int p_idx) const;

	Variant call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, CallState *p_state = nullptr);
	void debug_get_stack_member_state(int p_line, List<Pair<StringName, int>> *r_stackvars) const;

#ifdef DEBUG_ENABLED
	void _profile_native_call(uint64_t p_t_taken, const String &p_function_name, const String &p_instance_class_name = String());
	void disassemble(const Vector<String> &p_code_lines) const;
#endif

	GDScriptFunction();
	~GDScriptFunction();
};

class GDScriptFunctionState : public RefCounted {
	GDCLASS(GDScriptFunctionState, RefCounted);
	friend class GDScriptFunction;
	GDScriptFunction *function = nullptr;
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
	void _clear_connections();

	GDScriptFunctionState();
	~GDScriptFunctionState();
};

#endif // GDSCRIPT_FUNCTION_H
