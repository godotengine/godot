/**************************************************************************/
/*  gdscript_vm.cpp                                                       */
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

#include "gdscript.h"
#include "gdscript_function.h"
#include "gdscript_lambda_callable.h"

#include "core/os/os.h"

#ifdef DEBUG_ENABLED

static bool _profile_count_as_native(const Object *p_base_obj, const StringName &p_methodname) {
	if (!p_base_obj) {
		return false;
	}
	StringName cname = p_base_obj->get_class_name();
	if ((p_methodname == "new" && cname == "GDScript") || p_methodname == "call") {
		return false;
	}
	return ClassDB::class_exists(cname) && ClassDB::has_method(cname, p_methodname, false);
}

static String _get_element_type(Variant::Type builtin_type, const StringName &native_type, const Ref<Script> &script_type) {
	if (script_type.is_valid() && script_type->is_valid()) {
		return GDScript::debug_get_script_name(script_type);
	} else if (native_type != StringName()) {
		return native_type.operator String();
	} else {
		return Variant::get_type_name(builtin_type);
	}
}

static String _get_var_type(const Variant *p_var) {
	String basestr;

	if (p_var->get_type() == Variant::OBJECT) {
		bool was_freed;
		Object *bobj = p_var->get_validated_object_with_check(was_freed);
		if (!bobj) {
			if (was_freed) {
				basestr = "previously freed";
			} else {
				basestr = "null instance";
			}
		} else {
			if (bobj->is_class_ptr(GDScriptNativeClass::get_class_ptr_static())) {
				basestr = Object::cast_to<GDScriptNativeClass>(bobj)->get_name();
			} else {
				basestr = bobj->get_class();
				if (bobj->get_script_instance()) {
					basestr += " (" + GDScript::debug_get_script_name(bobj->get_script_instance()->get_script()) + ")";
				}
			}
		}

	} else {
		if (p_var->get_type() == Variant::ARRAY) {
			basestr = "Array";
			const Array *p_array = VariantInternal::get_array(p_var);
			if (p_array->is_typed()) {
				basestr += "[" + _get_element_type((Variant::Type)p_array->get_typed_builtin(), p_array->get_typed_class_name(), p_array->get_typed_script()) + "]";
			}
		} else if (p_var->get_type() == Variant::DICTIONARY) {
			basestr = "Dictionary";
			const Dictionary *p_dictionary = VariantInternal::get_dictionary(p_var);
			if (p_dictionary->is_typed()) {
				basestr += "[" + _get_element_type((Variant::Type)p_dictionary->get_typed_key_builtin(), p_dictionary->get_typed_key_class_name(), p_dictionary->get_typed_key_script()) +
						", " + _get_element_type((Variant::Type)p_dictionary->get_typed_value_builtin(), p_dictionary->get_typed_value_class_name(), p_dictionary->get_typed_value_script()) + "]";
			}
		} else {
			basestr = Variant::get_type_name(p_var->get_type());
		}
	}

	return basestr;
}

void GDScriptFunction::_profile_native_call(uint64_t p_t_taken, const String &p_func_name, const String &p_instance_class_name) {
	HashMap<String, Profile::NativeProfile>::Iterator inner_prof = profile.native_calls.find(p_func_name);
	if (inner_prof) {
		inner_prof->value.call_count += 1;
	} else {
		String sig = vformat("%s::0::%s%s%s", get_script()->get_script_path(), p_instance_class_name, p_instance_class_name.is_empty() ? "" : ".", p_func_name);
		inner_prof = profile.native_calls.insert(p_func_name, Profile::NativeProfile{ 1, 0, sig });
	}
	inner_prof->value.total_time += p_t_taken;
}

#endif // DEBUG_ENABLED

Variant GDScriptFunction::_get_default_variant_for_data_type(const GDScriptDataType &p_data_type) {
	if (p_data_type.kind == GDScriptDataType::BUILTIN) {
		if (p_data_type.builtin_type == Variant::ARRAY) {
			Array array;
			// Typed array.
			if (p_data_type.has_container_element_type(0)) {
				const GDScriptDataType &element_type = p_data_type.get_container_element_type(0);
				array.set_typed(element_type.builtin_type, element_type.native_type, element_type.script_type);
			}

			return array;
		} else if (p_data_type.builtin_type == Variant::DICTIONARY) {
			Dictionary dict;
			// Typed dictionary.
			if (p_data_type.has_container_element_types()) {
				const GDScriptDataType &key_type = p_data_type.get_container_element_type_or_variant(0);
				const GDScriptDataType &value_type = p_data_type.get_container_element_type_or_variant(1);
				dict.set_typed(key_type.builtin_type, key_type.native_type, key_type.script_type, value_type.builtin_type, value_type.native_type, value_type.script_type);
			}

			return dict;
		} else {
			Callable::CallError ce;
			Variant variant;
			Variant::construct(p_data_type.builtin_type, variant, nullptr, 0, ce);

			ERR_FAIL_COND_V(ce.error != Callable::CallError::CALL_OK, Variant());

			return variant;
		}
	}

	return Variant();
}

String GDScriptFunction::_get_call_error(const String &p_where, const Variant **p_argptrs, const Variant &p_ret, const Callable::CallError &p_err) const {
	switch (p_err.error) {
		case Callable::CallError::CALL_OK:
			return String();
		case Callable::CallError::CALL_ERROR_INVALID_METHOD:
			if (p_ret.get_type() == Variant::STRING && !p_ret.operator String().is_empty()) {
				return "Invalid call " + p_where + ": " + p_ret.operator String();
			}
			return "Invalid call. Nonexistent " + p_where + ".";
		case Callable::CallError::CALL_ERROR_INVALID_ARGUMENT:
			ERR_FAIL_COND_V_MSG(p_err.argument < 0 || p_argptrs[p_err.argument] == nullptr, "Bug: Invalid CallError argument index or null pointer.", "Bug: Invalid CallError argument index or null pointer.");
			// Handle the Object to Object case separately as we don't have further class details.
#ifdef DEBUG_ENABLED
			if (p_err.expected == Variant::OBJECT && p_argptrs[p_err.argument]->get_type() == p_err.expected) {
				return "Invalid type in " + p_where + ". The Object-derived class of argument " + itos(p_err.argument + 1) + " (" + _get_var_type(p_argptrs[p_err.argument]) + ") is not a subclass of the expected argument class.";
			}
			if (p_err.expected == Variant::ARRAY && p_argptrs[p_err.argument]->get_type() == p_err.expected) {
				return "Invalid type in " + p_where + ". The array of argument " + itos(p_err.argument + 1) + " (" + _get_var_type(p_argptrs[p_err.argument]) + ") does not have the same element type as the expected typed array argument.";
			}
			if (p_err.expected == Variant::DICTIONARY && p_argptrs[p_err.argument]->get_type() == p_err.expected) {
				return "Invalid type in " + p_where + ". The dictionary of argument " + itos(p_err.argument + 1) + " (" + _get_var_type(p_argptrs[p_err.argument]) + ") does not have the same element type as the expected typed dictionary argument.";
			}
#endif // DEBUG_ENABLED
			return "Invalid type in " + p_where + ". Cannot convert argument " + itos(p_err.argument + 1) + " from " + Variant::get_type_name(p_argptrs[p_err.argument]->get_type()) + " to " + Variant::get_type_name(Variant::Type(p_err.expected)) + ".";
		case Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS:
		case Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS:
			return "Invalid call to " + p_where + ". Expected " + itos(p_err.expected) + " arguments.";
		case Callable::CallError::CALL_ERROR_INSTANCE_IS_NULL:
			return "Attempt to call " + p_where + " on a null instance.";
		case Callable::CallError::CALL_ERROR_METHOD_NOT_CONST:
			return "Attempt to call " + p_where + " on a const instance.";
	}
	return "Bug: Invalid call error code " + itos(p_err.error) + ".";
}

void (*type_init_function_table[])(Variant *) = {
	nullptr, // NIL (shouldn't be called).
	&VariantInitializer<bool>::init, // BOOL.
	&VariantInitializer<int64_t>::init, // INT.
	&VariantInitializer<double>::init, // FLOAT.
	&VariantInitializer<String>::init, // STRING.
	&VariantInitializer<Vector2>::init, // VECTOR2.
	&VariantInitializer<Vector2i>::init, // VECTOR2I.
	&VariantInitializer<Rect2>::init, // RECT2.
	&VariantInitializer<Rect2i>::init, // RECT2I.
	&VariantInitializer<Vector3>::init, // VECTOR3.
	&VariantInitializer<Vector3i>::init, // VECTOR3I.
	&VariantInitializer<Transform2D>::init, // TRANSFORM2D.
	&VariantInitializer<Vector4>::init, // VECTOR4.
	&VariantInitializer<Vector4i>::init, // VECTOR4I.
	&VariantInitializer<Plane>::init, // PLANE.
	&VariantInitializer<Quaternion>::init, // QUATERNION.
	&VariantInitializer<AABB>::init, // AABB.
	&VariantInitializer<Basis>::init, // BASIS.
	&VariantInitializer<Transform3D>::init, // TRANSFORM3D.
	&VariantInitializer<Projection>::init, // PROJECTION.
	&VariantInitializer<Color>::init, // COLOR.
	&VariantInitializer<StringName>::init, // STRING_NAME.
	&VariantInitializer<NodePath>::init, // NODE_PATH.
	&VariantInitializer<RID>::init, // RID.
	&VariantInitializer<Object *>::init, // OBJECT.
	&VariantInitializer<Callable>::init, // CALLABLE.
	&VariantInitializer<Signal>::init, // SIGNAL.
	&VariantInitializer<Dictionary>::init, // DICTIONARY.
	&VariantInitializer<Array>::init, // ARRAY.
	&VariantInitializer<PackedByteArray>::init, // PACKED_BYTE_ARRAY.
	&VariantInitializer<PackedInt32Array>::init, // PACKED_INT32_ARRAY.
	&VariantInitializer<PackedInt64Array>::init, // PACKED_INT64_ARRAY.
	&VariantInitializer<PackedFloat32Array>::init, // PACKED_FLOAT32_ARRAY.
	&VariantInitializer<PackedFloat64Array>::init, // PACKED_FLOAT64_ARRAY.
	&VariantInitializer<PackedStringArray>::init, // PACKED_STRING_ARRAY.
	&VariantInitializer<PackedVector2Array>::init, // PACKED_VECTOR2_ARRAY.
	&VariantInitializer<PackedVector3Array>::init, // PACKED_VECTOR3_ARRAY.
	&VariantInitializer<PackedColorArray>::init, // PACKED_COLOR_ARRAY.
	&VariantInitializer<PackedVector4Array>::init, // PACKED_VECTOR4_ARRAY.
};

#if defined(__GNUC__) || defined(__clang__)
#define OPCODES_TABLE                                    \
	static const void *switch_table_ops[] = {            \
		&&OPCODE_OPERATOR,                               \
		&&OPCODE_OPERATOR_VALIDATED,                     \
		&&OPCODE_TYPE_TEST_BUILTIN,                      \
		&&OPCODE_TYPE_TEST_ARRAY,                        \
		&&OPCODE_TYPE_TEST_DICTIONARY,                   \
		&&OPCODE_TYPE_TEST_NATIVE,                       \
		&&OPCODE_TYPE_TEST_SCRIPT,                       \
		&&OPCODE_SET_KEYED,                              \
		&&OPCODE_SET_KEYED_VALIDATED,                    \
		&&OPCODE_SET_INDEXED_VALIDATED,                  \
		&&OPCODE_GET_KEYED,                              \
		&&OPCODE_GET_KEYED_VALIDATED,                    \
		&&OPCODE_GET_INDEXED_VALIDATED,                  \
		&&OPCODE_SET_NAMED,                              \
		&&OPCODE_SET_NAMED_VALIDATED,                    \
		&&OPCODE_GET_NAMED,                              \
		&&OPCODE_GET_NAMED_VALIDATED,                    \
		&&OPCODE_SET_MEMBER,                             \
		&&OPCODE_GET_MEMBER,                             \
		&&OPCODE_SET_STATIC_VARIABLE,                    \
		&&OPCODE_GET_STATIC_VARIABLE,                    \
		&&OPCODE_ASSIGN,                                 \
		&&OPCODE_ASSIGN_NULL,                            \
		&&OPCODE_ASSIGN_TRUE,                            \
		&&OPCODE_ASSIGN_FALSE,                           \
		&&OPCODE_ASSIGN_TYPED_BUILTIN,                   \
		&&OPCODE_ASSIGN_TYPED_ARRAY,                     \
		&&OPCODE_ASSIGN_TYPED_DICTIONARY,                \
		&&OPCODE_ASSIGN_TYPED_NATIVE,                    \
		&&OPCODE_ASSIGN_TYPED_SCRIPT,                    \
		&&OPCODE_CAST_TO_BUILTIN,                        \
		&&OPCODE_CAST_TO_NATIVE,                         \
		&&OPCODE_CAST_TO_SCRIPT,                         \
		&&OPCODE_CONSTRUCT,                              \
		&&OPCODE_CONSTRUCT_VALIDATED,                    \
		&&OPCODE_CONSTRUCT_ARRAY,                        \
		&&OPCODE_CONSTRUCT_TYPED_ARRAY,                  \
		&&OPCODE_CONSTRUCT_DICTIONARY,                   \
		&&OPCODE_CONSTRUCT_TYPED_DICTIONARY,             \
		&&OPCODE_CALL,                                   \
		&&OPCODE_CALL_RETURN,                            \
		&&OPCODE_CALL_ASYNC,                             \
		&&OPCODE_CALL_UTILITY,                           \
		&&OPCODE_CALL_UTILITY_VALIDATED,                 \
		&&OPCODE_CALL_GDSCRIPT_UTILITY,                  \
		&&OPCODE_CALL_BUILTIN_TYPE_VALIDATED,            \
		&&OPCODE_CALL_SELF_BASE,                         \
		&&OPCODE_CALL_METHOD_BIND,                       \
		&&OPCODE_CALL_METHOD_BIND_RET,                   \
		&&OPCODE_CALL_BUILTIN_STATIC,                    \
		&&OPCODE_CALL_NATIVE_STATIC,                     \
		&&OPCODE_CALL_NATIVE_STATIC_VALIDATED_RETURN,    \
		&&OPCODE_CALL_NATIVE_STATIC_VALIDATED_NO_RETURN, \
		&&OPCODE_CALL_METHOD_BIND_VALIDATED_RETURN,      \
		&&OPCODE_CALL_METHOD_BIND_VALIDATED_NO_RETURN,   \
		&&OPCODE_AWAIT,                                  \
		&&OPCODE_AWAIT_RESUME,                           \
		&&OPCODE_CREATE_LAMBDA,                          \
		&&OPCODE_CREATE_SELF_LAMBDA,                     \
		&&OPCODE_JUMP,                                   \
		&&OPCODE_JUMP_IF,                                \
		&&OPCODE_JUMP_IF_NOT,                            \
		&&OPCODE_JUMP_TO_DEF_ARGUMENT,                   \
		&&OPCODE_JUMP_IF_SHARED,                         \
		&&OPCODE_RETURN,                                 \
		&&OPCODE_RETURN_TYPED_BUILTIN,                   \
		&&OPCODE_RETURN_TYPED_ARRAY,                     \
		&&OPCODE_RETURN_TYPED_DICTIONARY,                \
		&&OPCODE_RETURN_TYPED_NATIVE,                    \
		&&OPCODE_RETURN_TYPED_SCRIPT,                    \
		&&OPCODE_ITERATE_BEGIN,                          \
		&&OPCODE_ITERATE_BEGIN_INT,                      \
		&&OPCODE_ITERATE_BEGIN_FLOAT,                    \
		&&OPCODE_ITERATE_BEGIN_VECTOR2,                  \
		&&OPCODE_ITERATE_BEGIN_VECTOR2I,                 \
		&&OPCODE_ITERATE_BEGIN_VECTOR3,                  \
		&&OPCODE_ITERATE_BEGIN_VECTOR3I,                 \
		&&OPCODE_ITERATE_BEGIN_STRING,                   \
		&&OPCODE_ITERATE_BEGIN_DICTIONARY,               \
		&&OPCODE_ITERATE_BEGIN_ARRAY,                    \
		&&OPCODE_ITERATE_BEGIN_PACKED_BYTE_ARRAY,        \
		&&OPCODE_ITERATE_BEGIN_PACKED_INT32_ARRAY,       \
		&&OPCODE_ITERATE_BEGIN_PACKED_INT64_ARRAY,       \
		&&OPCODE_ITERATE_BEGIN_PACKED_FLOAT32_ARRAY,     \
		&&OPCODE_ITERATE_BEGIN_PACKED_FLOAT64_ARRAY,     \
		&&OPCODE_ITERATE_BEGIN_PACKED_STRING_ARRAY,      \
		&&OPCODE_ITERATE_BEGIN_PACKED_VECTOR2_ARRAY,     \
		&&OPCODE_ITERATE_BEGIN_PACKED_VECTOR3_ARRAY,     \
		&&OPCODE_ITERATE_BEGIN_PACKED_COLOR_ARRAY,       \
		&&OPCODE_ITERATE_BEGIN_PACKED_VECTOR4_ARRAY,     \
		&&OPCODE_ITERATE_BEGIN_OBJECT,                   \
		&&OPCODE_ITERATE,                                \
		&&OPCODE_ITERATE_INT,                            \
		&&OPCODE_ITERATE_FLOAT,                          \
		&&OPCODE_ITERATE_VECTOR2,                        \
		&&OPCODE_ITERATE_VECTOR2I,                       \
		&&OPCODE_ITERATE_VECTOR3,                        \
		&&OPCODE_ITERATE_VECTOR3I,                       \
		&&OPCODE_ITERATE_STRING,                         \
		&&OPCODE_ITERATE_DICTIONARY,                     \
		&&OPCODE_ITERATE_ARRAY,                          \
		&&OPCODE_ITERATE_PACKED_BYTE_ARRAY,              \
		&&OPCODE_ITERATE_PACKED_INT32_ARRAY,             \
		&&OPCODE_ITERATE_PACKED_INT64_ARRAY,             \
		&&OPCODE_ITERATE_PACKED_FLOAT32_ARRAY,           \
		&&OPCODE_ITERATE_PACKED_FLOAT64_ARRAY,           \
		&&OPCODE_ITERATE_PACKED_STRING_ARRAY,            \
		&&OPCODE_ITERATE_PACKED_VECTOR2_ARRAY,           \
		&&OPCODE_ITERATE_PACKED_VECTOR3_ARRAY,           \
		&&OPCODE_ITERATE_PACKED_COLOR_ARRAY,             \
		&&OPCODE_ITERATE_PACKED_VECTOR4_ARRAY,           \
		&&OPCODE_ITERATE_OBJECT,                         \
		&&OPCODE_STORE_GLOBAL,                           \
		&&OPCODE_STORE_NAMED_GLOBAL,                     \
		&&OPCODE_TYPE_ADJUST_BOOL,                       \
		&&OPCODE_TYPE_ADJUST_INT,                        \
		&&OPCODE_TYPE_ADJUST_FLOAT,                      \
		&&OPCODE_TYPE_ADJUST_STRING,                     \
		&&OPCODE_TYPE_ADJUST_VECTOR2,                    \
		&&OPCODE_TYPE_ADJUST_VECTOR2I,                   \
		&&OPCODE_TYPE_ADJUST_RECT2,                      \
		&&OPCODE_TYPE_ADJUST_RECT2I,                     \
		&&OPCODE_TYPE_ADJUST_VECTOR3,                    \
		&&OPCODE_TYPE_ADJUST_VECTOR3I,                   \
		&&OPCODE_TYPE_ADJUST_TRANSFORM2D,                \
		&&OPCODE_TYPE_ADJUST_VECTOR4,                    \
		&&OPCODE_TYPE_ADJUST_VECTOR4I,                   \
		&&OPCODE_TYPE_ADJUST_PLANE,                      \
		&&OPCODE_TYPE_ADJUST_QUATERNION,                 \
		&&OPCODE_TYPE_ADJUST_AABB,                       \
		&&OPCODE_TYPE_ADJUST_BASIS,                      \
		&&OPCODE_TYPE_ADJUST_TRANSFORM3D,                \
		&&OPCODE_TYPE_ADJUST_PROJECTION,                 \
		&&OPCODE_TYPE_ADJUST_COLOR,                      \
		&&OPCODE_TYPE_ADJUST_STRING_NAME,                \
		&&OPCODE_TYPE_ADJUST_NODE_PATH,                  \
		&&OPCODE_TYPE_ADJUST_RID,                        \
		&&OPCODE_TYPE_ADJUST_OBJECT,                     \
		&&OPCODE_TYPE_ADJUST_CALLABLE,                   \
		&&OPCODE_TYPE_ADJUST_SIGNAL,                     \
		&&OPCODE_TYPE_ADJUST_DICTIONARY,                 \
		&&OPCODE_TYPE_ADJUST_ARRAY,                      \
		&&OPCODE_TYPE_ADJUST_PACKED_BYTE_ARRAY,          \
		&&OPCODE_TYPE_ADJUST_PACKED_INT32_ARRAY,         \
		&&OPCODE_TYPE_ADJUST_PACKED_INT64_ARRAY,         \
		&&OPCODE_TYPE_ADJUST_PACKED_FLOAT32_ARRAY,       \
		&&OPCODE_TYPE_ADJUST_PACKED_FLOAT64_ARRAY,       \
		&&OPCODE_TYPE_ADJUST_PACKED_STRING_ARRAY,        \
		&&OPCODE_TYPE_ADJUST_PACKED_VECTOR2_ARRAY,       \
		&&OPCODE_TYPE_ADJUST_PACKED_VECTOR3_ARRAY,       \
		&&OPCODE_TYPE_ADJUST_PACKED_COLOR_ARRAY,         \
		&&OPCODE_TYPE_ADJUST_PACKED_VECTOR4_ARRAY,       \
		&&OPCODE_ASSERT,                                 \
		&&OPCODE_BREAKPOINT,                             \
		&&OPCODE_LINE,                                   \
		&&OPCODE_END                                     \
	};                                                   \
	static_assert((sizeof(switch_table_ops) / sizeof(switch_table_ops[0]) == (OPCODE_END + 1)), "Opcodes in jump table aren't the same as opcodes in enum.");

#define OPCODE(m_op) \
	m_op:
#define OPCODE_WHILE(m_test)
#define OPCODES_END \
	OPSEXIT:
#define OPCODES_OUT \
	OPSOUT:
#define OPCODE_SWITCH(m_test) goto *switch_table_ops[m_test];

#ifdef DEBUG_ENABLED
#define DISPATCH_OPCODE          \
	last_opcode = _code_ptr[ip]; \
	goto *switch_table_ops[last_opcode]
#else // !DEBUG_ENABLED
#define DISPATCH_OPCODE goto *switch_table_ops[_code_ptr[ip]]
#endif // DEBUG_ENABLED

#define OPCODE_BREAK goto OPSEXIT
#define OPCODE_OUT goto OPSOUT
#else // !(defined(__GNUC__) || defined(__clang__))
#define OPCODES_TABLE
#define OPCODE(m_op) case m_op:
#define OPCODE_WHILE(m_test) while (m_test)
#define OPCODES_END
#define OPCODES_OUT
#define DISPATCH_OPCODE continue

#ifdef _MSC_VER
#define OPCODE_SWITCH(m_test)       \
	__assume(m_test <= OPCODE_END); \
	switch (m_test)
#else // !_MSC_VER
#define OPCODE_SWITCH(m_test) switch (m_test)
#endif // _MSC_VER

#define OPCODE_BREAK break
#define OPCODE_OUT break
#endif // defined(__GNUC__) || defined(__clang__)

// Helpers for VariantInternal methods in macros.
#define OP_GET_BOOL get_bool
#define OP_GET_INT get_int
#define OP_GET_FLOAT get_float
#define OP_GET_VECTOR2 get_vector2
#define OP_GET_VECTOR2I get_vector2i
#define OP_GET_VECTOR3 get_vector3
#define OP_GET_VECTOR3I get_vector3i
#define OP_GET_RECT2 get_rect2
#define OP_GET_VECTOR4 get_vector4
#define OP_GET_VECTOR4I get_vector4i
#define OP_GET_RECT2I get_rect2i
#define OP_GET_QUATERNION get_quaternion
#define OP_GET_COLOR get_color
#define OP_GET_STRING get_string
#define OP_GET_STRING_NAME get_string_name
#define OP_GET_NODE_PATH get_node_path
#define OP_GET_CALLABLE get_callable
#define OP_GET_SIGNAL get_signal
#define OP_GET_ARRAY get_array
#define OP_GET_DICTIONARY get_dictionary
#define OP_GET_PACKED_BYTE_ARRAY get_byte_array
#define OP_GET_PACKED_INT32_ARRAY get_int32_array
#define OP_GET_PACKED_INT64_ARRAY get_int64_array
#define OP_GET_PACKED_FLOAT32_ARRAY get_float32_array
#define OP_GET_PACKED_FLOAT64_ARRAY get_float64_array
#define OP_GET_PACKED_STRING_ARRAY get_string_array
#define OP_GET_PACKED_VECTOR2_ARRAY get_vector2_array
#define OP_GET_PACKED_VECTOR3_ARRAY get_vector3_array
#define OP_GET_PACKED_COLOR_ARRAY get_color_array
#define OP_GET_PACKED_VECTOR4_ARRAY get_vector4_array
#define OP_GET_TRANSFORM3D get_transform
#define OP_GET_TRANSFORM2D get_transform2d
#define OP_GET_PROJECTION get_projection
#define OP_GET_PLANE get_plane
#define OP_GET_AABB get_aabb
#define OP_GET_BASIS get_basis
#define OP_GET_RID get_rid

#define METHOD_CALL_ON_NULL_VALUE_ERROR(method_pointer) "Cannot call method '" + (method_pointer)->get_name() + "' on a null value."
#define METHOD_CALL_ON_FREED_INSTANCE_ERROR(method_pointer) "Cannot call method '" + (method_pointer)->get_name() + "' on a previously freed instance."

#ifdef DEBUG_ENABLED

#define GD_ERR_BREAK(m_cond)                                                                                           \
	{                                                                                                                  \
		if (unlikely(m_cond)) {                                                                                        \
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Condition ' " _STR(m_cond) " ' is true. Breaking..:"); \
			return;                                                                                                    \
		}                                                                                                              \
	}

#define CHECK_SPACE(m_space) \
	GD_ERR_BREAK((p_ip + m_space) > _code_size)

#define GET_VARIANT_PTR(m_v, m_code_ofs)                                                              \
	Variant *m_v;                                                                                     \
	{                                                                                                 \
		int address = _code_ptr[p_ip + 1 + (m_code_ofs)];                                             \
		int address_type = (address & ADDR_TYPE_MASK) >> ADDR_BITS;                                   \
		if (unlikely(address_type < 0 || address_type >= ADDR_TYPE_MAX)) {                            \
			p_err_text = "Bad address type.";                                                         \
			return;                                                                                   \
		}                                                                                             \
		int address_index = address & ADDR_MASK;                                                      \
		if (unlikely(address_index < 0 || address_index >= p_variant_address_limits[address_type])) { \
			if (address_type == ADDR_TYPE_MEMBER && !p_instance) {                                    \
				p_err_text = "Cannot access member without instance.";                                \
			} else {                                                                                  \
				p_err_text = "Bad address index.";                                                    \
			}                                                                                         \
			return;                                                                                   \
		}                                                                                             \
		m_v = &p_variant_addresses[address_type][address_index];                                      \
		if (unlikely(!m_v))                                                                           \
			return;                                                                                   \
	}

#else // !DEBUG_ENABLED
#define GD_ERR_BREAK(m_cond)
#define CHECK_SPACE(m_space)

#define GET_VARIANT_PTR(m_v, m_code_ofs)                                                                                                                \
	Variant *m_v;                                                                                                                                       \
	{                                                                                                                                                   \
		int address = p_code_ptr[p_ip + 1 + (m_code_ofs)];                                                                                              \
		m_v = &p_variant_addresses[(address & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS][address & GDScriptFunction::ADDR_MASK]; \
		if (unlikely(!m_v))                                                                                                                             \
			return;                                                                                                                                     \
	}

#endif // DEBUG_ENABLED

#define OP_EXEC_IMPLEMENT(m_opcode) void GDScriptFunction::_exec_##m_opcode OP_ARGS

#define GET_CALL_ARGUMENT(m_v) \
	int m_v = _code_ptr[p_ip + 1];

#define LOAD_INSTRUCTION_ARGS                   \
	int instr_arg_count = _code_ptr[p_ip + 1];  \
	for (int i = 0; i < instr_arg_count; i++) { \
		GET_VARIANT_PTR(v, i + 1);              \
		instruction_args[i] = v;                \
	}                                           \
	ip += 1; // Offset to skip instruction argcount.

#define GET_INSTRUCTION_ARG(m_v, m_idx) \
	Variant *m_v = instruction_args[m_idx]

Variant GDScriptFunction::call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, CallState *p_state) {
	OPCODES_TABLE;

	if (!_code_ptr) {
		return _get_default_variant_for_data_type(return_type);
	}

	r_err.error = Callable::CallError::CALL_OK;

	static thread_local int call_depth = 0;
	if (unlikely(++call_depth > MAX_CALL_DEPTH)) {
		call_depth--;
#ifdef DEBUG_ENABLED
		String err_file;
		if (p_instance && ObjectDB::get_instance(p_instance->owner_id) != nullptr && p_instance->script->is_valid() && !p_instance->script->path.is_empty()) {
			err_file = p_instance->script->path;
		} else if (_script) {
			err_file = _script->path;
		}
		if (err_file.is_empty()) {
			err_file = "<built-in>";
		}
		String err_func = name;
		if (p_instance && ObjectDB::get_instance(p_instance->owner_id) != nullptr && p_instance->script->is_valid() && p_instance->script->local_name != StringName()) {
			err_func = p_instance->script->local_name.operator String() + "." + err_func;
		}
		int err_line = _initial_line;
		const char *err_text = "Stack overflow. Check for infinite recursion in your script.";
		if (!GDScriptLanguage::get_singleton()->debug_break(err_text, false)) {
			// Debugger break did not happen.
			_err_print_error(err_func.utf8().get_data(), err_file.utf8().get_data(), err_line, err_text, false, ERR_HANDLER_SCRIPT);
		}
#endif
		return _get_default_variant_for_data_type(return_type);
	}

	Variant retvalue;
	Variant *stack = nullptr;
	Variant **instruction_args = nullptr;
	int defarg = 0;

	uint32_t alloca_size = 0;
	GDScript *script;
	int ip = 0;
	int line = _initial_line;

	if (p_state) {
		//use existing (supplied) state (awaited)
		stack = (Variant *)p_state->stack.ptr();
		instruction_args = (Variant **)&p_state->stack.ptr()[sizeof(Variant) * p_state->stack_size]; //ptr() to avoid bounds check
		line = p_state->line;
		ip = p_state->ip;
		alloca_size = p_state->stack.size();
		script = p_state->script;
		p_instance = p_state->instance;
		defarg = p_state->defarg;

	} else {
		if (p_argcount != _argument_count) {
			if (p_argcount > _argument_count) {
				r_err.error = Callable::CallError::CALL_ERROR_TOO_MANY_ARGUMENTS;
				r_err.expected = _argument_count;
				call_depth--;
				return _get_default_variant_for_data_type(return_type);
			} else if (p_argcount < _argument_count - _default_arg_count) {
				r_err.error = Callable::CallError::CALL_ERROR_TOO_FEW_ARGUMENTS;
				r_err.expected = _argument_count - _default_arg_count;
				call_depth--;
				return _get_default_variant_for_data_type(return_type);
			} else {
				defarg = _argument_count - p_argcount;
			}
		}

		// Add 3 here for self, class, and nil.
		alloca_size = sizeof(Variant *) * 3 + sizeof(Variant *) * _instruction_args_size + sizeof(Variant) * _stack_size;

		uint8_t *aptr = (uint8_t *)alloca(alloca_size);
		stack = (Variant *)aptr;

		for (int i = 0; i < p_argcount; i++) {
			if (!argument_types[i].has_type) {
				memnew_placement(&stack[i + 3], Variant(*p_args[i]));
				continue;
			}
			// If types already match, don't call Variant::construct(). Constructors of some types
			// (e.g. packed arrays) do copies, whereas they pass by reference when inside a Variant.
			if (argument_types[i].is_type(*p_args[i], false)) {
				memnew_placement(&stack[i + 3], Variant(*p_args[i]));
				continue;
			}
			if (!argument_types[i].is_type(*p_args[i], true)) {
				r_err.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
				r_err.argument = i;
				r_err.expected = argument_types[i].builtin_type;
				call_depth--;
				return _get_default_variant_for_data_type(return_type);
			}
			if (argument_types[i].kind == GDScriptDataType::BUILTIN) {
				if (argument_types[i].builtin_type == Variant::DICTIONARY && argument_types[i].has_container_element_types()) {
					const GDScriptDataType &arg_key_type = argument_types[i].get_container_element_type_or_variant(0);
					const GDScriptDataType &arg_value_type = argument_types[i].get_container_element_type_or_variant(1);
					Dictionary dict(p_args[i]->operator Dictionary(), arg_key_type.builtin_type, arg_key_type.native_type, arg_key_type.script_type, arg_value_type.builtin_type, arg_value_type.native_type, arg_value_type.script_type);
					memnew_placement(&stack[i + 3], Variant(dict));
				} else if (argument_types[i].builtin_type == Variant::ARRAY && argument_types[i].has_container_element_type(0)) {
					const GDScriptDataType &arg_type = argument_types[i].container_element_types[0];
					Array array(p_args[i]->operator Array(), arg_type.builtin_type, arg_type.native_type, arg_type.script_type);
					memnew_placement(&stack[i + 3], Variant(array));
				} else {
					Variant variant;
					Variant::construct(argument_types[i].builtin_type, variant, &p_args[i], 1, r_err);
					if (unlikely(r_err.error)) {
						r_err.error = Callable::CallError::CALL_ERROR_INVALID_ARGUMENT;
						r_err.argument = i;
						r_err.expected = argument_types[i].builtin_type;
						call_depth--;
						return _get_default_variant_for_data_type(return_type);
					}
					memnew_placement(&stack[i + 3], Variant(variant));
				}
			} else {
				memnew_placement(&stack[i + 3], Variant(*p_args[i]));
			}
		}
		for (int i = p_argcount + 3; i < _stack_size; i++) {
			memnew_placement(&stack[i], Variant);
		}

		if (_instruction_args_size) {
			instruction_args = (Variant **)&aptr[sizeof(Variant) * _stack_size];
		} else {
			instruction_args = nullptr;
		}

		for (const KeyValue<int, Variant::Type> &E : temporary_slots) {
			type_init_function_table[E.value](&stack[E.key]);
		}
	}

	if (p_instance) {
		memnew_placement(&stack[ADDR_STACK_SELF], Variant(p_instance->owner));
		script = p_instance->script.ptr();
	} else {
		memnew_placement(&stack[ADDR_STACK_SELF], Variant);
		script = _script;
	}
	memnew_placement(&stack[ADDR_STACK_CLASS], Variant(script));
	memnew_placement(&stack[ADDR_STACK_NIL], Variant);

	String err_text;

#ifdef DEBUG_ENABLED

	if (EngineDebugger::is_active()) {
		GDScriptLanguage::get_singleton()->enter_function(p_instance, this, stack, &ip, &line);
	}

	uint64_t function_start_time = 0;
	uint64_t function_call_time = 0;

	if (GDScriptLanguage::get_singleton()->profiling) {
		function_start_time = OS::get_singleton()->get_ticks_usec();
		function_call_time = 0;
		profile.call_count.increment();
		profile.frame_call_count.increment();
	}
	bool exit_ok = false;
	bool awaited = false;
	int variant_address_limits[ADDR_TYPE_MAX] = { _stack_size, _constant_count, p_instance ? (int)p_instance->members.size() : 0 };

#endif

	Variant *variant_addresses[ADDR_TYPE_MAX] = { stack, _constants_ptr, p_instance ? p_instance->members.ptrw() : nullptr };

#ifdef DEBUG_ENABLED
	OPCODE_WHILE(ip < _code_size) {
		int last_opcode = _code_ptr[ip];
#else
	OPCODE_WHILE(true) {
#endif

		OPCODE_SWITCH(_code_ptr[ip]) {
		}
	}
#ifdef DEBUG_ENABLED
	if (GDScriptLanguage::get_singleton()->profiling) {
		uint64_t time_taken = OS::get_singleton()->get_ticks_usec() - function_start_time;
		profile.total_time.add(time_taken);
		profile.self_time.add(time_taken - function_call_time);
		profile.frame_total_time.add(time_taken);
		profile.frame_self_time.add(time_taken - function_call_time);
		if (Thread::get_caller_id() == Thread::get_main_id()) {
			GDScriptLanguage::get_singleton()->script_frame_time += time_taken - function_call_time;
		}
	}

	// Check if this is not the last time it was interrupted by `await` or if it's the first time executing.
	// If that is the case then we exit the function as normal. Otherwise we postpone it until the last `await` is completed.
	// This ensures the call stack can be properly shown when using `await`, showing what resumed the function.
	if (!p_state || awaited) {
		if (EngineDebugger::is_active()) {
			GDScriptLanguage::get_singleton()->exit_function();
		}
#endif

		// Free stack, except reserved addresses.
		for (int i = FIXED_ADDRESSES_MAX; i < _stack_size; i++) {
			stack[i].~Variant();
		}
	}

	// Always free reserved addresses, since they are never copied.
	for (int i = 0; i < FIXED_ADDRESSES_MAX; i++) {
		stack[i].~Variant();
	}

	call_depth--;

	return retvalue;
}

OP_EXEC_IMPLEMENT(OPCODE_OPERATOR) {
	constexpr int _pointer_size = sizeof(Variant::ValidatedOperatorEvaluator) / sizeof(*_code_ptr);

	bool valid;
	Variant::Operator op = (Variant::Operator)_code_ptr[p_ip + 4];
	GD_ERR_BREAK(op >= Variant::OP_MAX);

	GET_CALL_ARGUMENT(arg);
	bool is_validated = (arg & GDScriptFunction::IS_VALIDATED) != 0;

	if (is_validated) {
		const int OP_SIZE = 6;
		const int LHS_INDEX = 1;
		const int RHS_INDEX = 2;
		const int DST_INDEX = 3;

		CHECK_SPACE(OP_SIZE);

		int operator_idx = _code_ptr[p_ip + 4];
		GD_ERR_BREAK(operator_idx < 0 || operator_idx >= _operator_funcs_count);
		Variant::ValidatedOperatorEvaluator operator_func = _operator_funcs_ptr[operator_idx];

		GET_VARIANT_PTR(a, 0);
		GET_VARIANT_PTR(b, 1);
		GET_VARIANT_PTR(dst, 2);

		operator_func(a, b, dst);

		p_ip += OP_SIZE;
	} else {
		const int OP_SIZE = 8;
		const int LHS_INDEX = 1;
		const int RHS_INDEX = 2;
		const int DST_INDEX = 3;
		CHECK_SPACE(OP_SIZE + _pointer_size);

		GET_VARIANT_PTR(a, LHS_INDEX);
		GET_VARIANT_PTR(b, RHS_INDEX);
		GET_VARIANT_PTR(dst, DST_INDEX);
		// Compute signatures (types of operands) so it can be optimized when matching.
		uint32_t op_signature = _code_ptr[p_ip + 5];
		uint32_t actual_signature = (a->get_type() << 8) | (b->get_type());

#ifdef DEBUG_ENABLED
		if (op == Variant::OP_DIVIDE || op == Variant::OP_MODULE) {
			// Don't optimize division and modulo since there's not check for division by zero with validated calls.
			op_signature = 0xFFFF;
			_code_ptr[p_ip + 5] = op_signature;
		}
#endif

		// Check if this is the first run. If so, store the current signature for the optimized path.
		if (unlikely(op_signature == 0)) {
			static Mutex initializer_mutex;
			initializer_mutex.lock();
			Variant::Type a_type = (Variant::Type)((actual_signature >> 8) & 0xFF);
			Variant::Type b_type = (Variant::Type)(actual_signature & 0xFF);

			Variant::ValidatedOperatorEvaluator op_func = Variant::get_validated_operator_evaluator(op, a_type, b_type);

			if (unlikely(!op_func)) {
#ifdef DEBUG_ENABLED
				p_err_text = "Invalid operands '" + Variant::get_type_name(a->get_type()) + "' and '" + Variant::get_type_name(b->get_type()) + "' in operator '" + Variant::get_operator_name(op) + "'.";
#endif
				initializer_mutex.unlock();
			} else {
				Variant::Type ret_type = Variant::get_operator_return_type(op, a_type, b_type);
				VariantInternal::initialize(dst, ret_type);
				op_func(a, b, dst);

				// Check again in case another thread already set it.
				if (_code_ptr[p_ip + 5] == 0) {
					_code_ptr[p_ip + 5] = actual_signature;
					_code_ptr[p_ip + 6] = static_cast<int>(ret_type);
					Variant::ValidatedOperatorEvaluator *tmp = reinterpret_cast<Variant::ValidatedOperatorEvaluator *>(&_code_ptr[p_ip + 7]);
					*tmp = op_func;
				}
			}
			initializer_mutex.unlock();
		} else if (likely(op_signature == actual_signature)) {
			// If the signature matches, we can use the optimized path.
			Variant::Type ret_type = static_cast<Variant::Type>(_code_ptr[p_ip + 6]);
			Variant::ValidatedOperatorEvaluator op_func = *reinterpret_cast<Variant::ValidatedOperatorEvaluator *>(&_code_ptr[p_ip + 7]);

			// Make sure the return value has the correct type.
			VariantInternal::initialize(dst, ret_type);
			op_func(a, b, dst);
		} else {
			// If the signature doesn't match, we have to use the slow path.
#ifdef DEBUG_ENABLED

			Variant ret;
			Variant::evaluate(op, *a, *b, ret, valid);
#else
			Variant::evaluate(op, *a, *b, *dst, valid);
#endif
#ifdef DEBUG_ENABLED
			if (!valid) {
				if (ret.get_type() == Variant::STRING) {
					//return a string when invalid with the error
					p_err_text = ret;
					p_err_text += " in operator '" + Variant::get_operator_name(op) + "'.";
				} else {
					p_err_text = "Invalid operands '" + Variant::get_type_name(a->get_type()) + "' and '" + Variant::get_type_name(b->get_type()) + "' in operator '" + Variant::get_operator_name(op) + "'.";
				}
				return;
			}
			*dst = ret;
#endif
		}
		p_ip += OP_SIZE + _pointer_size;
	}
}

OP_EXEC_IMPLEMENT(OPCODE_TYPE_TEST) {
	const int
			DEST_INDEX = 1,
			VALUE_INDEX = 2;
	GET_CALL_ARGUMENT(arg);

	switch (arg) {
		case TestArguments::TEST_BUILTIN: {
			const int
					OP_SIZE = 5, // ?
					DST_POS = 1,
					VAL_POS = 2,
					BUILTIN_POS = 4;
			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_ip + BUILTIN_POS];
			GD_ERR_BREAK(builtin_type < 0 || builtin_type >= Variant::VARIANT_MAX);

			*dst = value->get_type() == builtin_type;
			p_ip += OP_SIZE;
			break;
		}

		case TestArguments::TEST_ARRAY: {
			const int
					OP_SIZE = 7,
					DST_POS = 1,
					VAL_POS = 2,
					SCRIPT_POS = 3,
					BUILTIN_POS = 5,
					NATIVE_POS = 6;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			GET_VARIANT_PTR(script_type, SCRIPT_POS);
			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_ip + BUILTIN_POS];
			int native_type_idx = _code_ptr[p_ip + NATIVE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			bool result = false;
			if (value->get_type() == Variant::ARRAY) {
				Array *array = VariantInternal::get_array(value);
				result = array->get_typed_builtin() == ((uint32_t)builtin_type) && array->get_typed_class_name() == native_type && array->get_typed_script() == *script_type;
			}

			*dst = result;
			p_ip += OP_SIZE;
			break;
		}

		case TestArguments::TEST_DICTIONARY: {
			const int
					OP_SIZE = 11,
					DST_POS = 1,
					VAL_POS = 2,
					SCRIPT_A_POS = 3,
					SCRIPT_B_POS = 4,
					BUILTIN_A_POS = 6,
					BUILTIN_B_POS = 8,
					NATIVE_A_POS = 7,
					NATIVE_B_POS = 9;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			GET_VARIANT_PTR(key_script_type, SCRIPT_A_POS);
			Variant::Type key_builtin_type = (Variant::Type)_code_ptr[p_ip + BUILTIN_A_POS];
			int key_native_type_idx = _code_ptr[p_ip + NATIVE_B_POS];
			GD_ERR_BREAK(key_native_type_idx < 0 || key_native_type_idx >= _global_names_count);
			const StringName key_native_type = _global_names_ptr[key_native_type_idx];

			GET_VARIANT_PTR(value_script_type, SCRIPT_B_POS);
			Variant::Type value_builtin_type = (Variant::Type)_code_ptr[p_ip + BUILTIN_B_POS];
			int value_native_type_idx = _code_ptr[p_ip + NATIVE_B_POS];
			GD_ERR_BREAK(value_native_type_idx < 0 || value_native_type_idx >= _global_names_count);
			const StringName value_native_type = _global_names_ptr[value_native_type_idx];

			bool result = false;
			if (value->get_type() == Variant::DICTIONARY) {
				Dictionary *dictionary = VariantInternal::get_dictionary(value);
				result = dictionary->get_typed_key_builtin() == ((uint32_t)key_builtin_type) && dictionary->get_typed_key_class_name() == key_native_type && dictionary->get_typed_key_script() == *key_script_type &&
						dictionary->get_typed_value_builtin() == ((uint32_t)value_builtin_type) && dictionary->get_typed_value_class_name() == value_native_type && dictionary->get_typed_value_script() == *value_script_type;
			}

			*dst = result;
			p_ip += OP_SIZE;
			break;
		}

		case TestArguments::TEST_NATIVE: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					VAL_POS = 2,
					NATIVE_POS = 3;

			CHECK_SPACE(5);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			int native_type_idx = _code_ptr[p_ip + NATIVE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			bool was_freed = false;
			Object *object = value->get_validated_object_with_check(was_freed);
			if (was_freed) {
				p_err_text = "Left operand of 'is' is a previously freed instance.";
				return;
			}

			*dst = object && ClassDB::is_parent_class(object->get_class_name(), native_type);
			p_ip += OP_SIZE;
			break;
		}

		case TestArguments::TEST_SCRIPT: {
			const int
					OP_SIZE = 6,
					DST_POS = 1,
					VAL_POS = 2,
					TYPE_POS = 3;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			GET_VARIANT_PTR(type, TYPE_POS);
			Script *script_type = Object::cast_to<Script>(type->operator Object *());
			GD_ERR_BREAK(!script_type);

			bool was_freed = false;
			Object *object = value->get_validated_object_with_check(was_freed);
			if (was_freed) {
				p_err_text = "Left operand of 'is' is a previously freed instance.";
				return;
			}

			bool result = false;
			if (object && object->get_script_instance()) {
				Script *script_ptr = object->get_script_instance()->get_script().ptr();
				while (script_ptr) {
					if (script_ptr == script_type) {
						result = true;
						return;
					}
					script_ptr = script_ptr->get_base_script().ptr();
				}
			}

			*dst = result;
			p_ip += OP_SIZE;
			break;
		}

		default:
			_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Invalid opcode argument for OPCODE_TYPE_TEST: ' " + String::num_int64(arg) + " '. This is a bug, please report it!...");
			break;
	}
}

OP_EXEC_IMPLEMENT(OPCODE_SET_KEYED) {} // Can be Validated

OP_EXEC_IMPLEMENT(OPCODE_SET_INDEXED) {} // Only Validated

OP_EXEC_IMPLEMENT(OPCODE_GET_KEYED) {} // Can be Validated

OP_EXEC_IMPLEMENT(OPCODE_GET_INDEXED) {} // Only Validated

OP_EXEC_IMPLEMENT(OPCODE_SET_NAMED) {} // Can be Validated

OP_EXEC_IMPLEMENT(OPCODE_GET_NAMED) {} // Can be Validated

OP_EXEC_IMPLEMENT(OPCODE_SET_MEMBER) {}

OP_EXEC_IMPLEMENT(OPCODE_GET_MEMBER) {}

OP_EXEC_IMPLEMENT(OPCODE_SET_STATIC_VARIABLE) {} // Only for GDScript.

OP_EXEC_IMPLEMENT(OPCODE_GET_STATIC_VARIABLE) {} // Only for GDScript.

OP_EXEC_IMPLEMENT(OPCODE_ASSIGN) {} // Args is AssignArguments

OP_EXEC_IMPLEMENT(OPCODE_CAST) {} // Args is CastArgs

OP_EXEC_IMPLEMENT(OPCODE_CONSTRUCT) {} // Only for basic types! Args is ConstructArguments

OP_EXEC_IMPLEMENT(OPCODE_CALL) {} // Args is CallArguments

OP_EXEC_IMPLEMENT(OPCODE_AWAIT) {}

OP_EXEC_IMPLEMENT(OPCODE_AWAIT_RESUME) {}

OP_EXEC_IMPLEMENT(OPCODE_CREATE_LAMBDA) {}

OP_EXEC_IMPLEMENT(OPCODE_CREATE_SELF_LAMBDA) {}

OP_EXEC_IMPLEMENT(OPCODE_JUMP) {} // Args is JumpArgs

OP_EXEC_IMPLEMENT(OPCODE_RETURN) {} // Args is ReturnArgs

OP_EXEC_IMPLEMENT(OPCODE_ITERATE_BEGIN) {} // Args is IterateArguments

OP_EXEC_IMPLEMENT(OPCODE_ITERATE) {} // Args is IterateArguments

OP_EXEC_IMPLEMENT(OPCODE_STORE_GLOBAL) {}

OP_EXEC_IMPLEMENT(OPCODE_STORE_NAMED_GLOBAL) {}

OP_EXEC_IMPLEMENT(OPCODE_TYPE_ADJUST) {} // Args is AdjustArguments

OP_EXEC_IMPLEMENT(OPCODE_ASSERT) {}

OP_EXEC_IMPLEMENT(OPCODE_BREAKPOINT) {}

OP_EXEC_IMPLEMENT(OPCODE_LINE) {}
