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
	GD_ERR_BREAK((p_info->ip + m_space) > _code_size)

#define GET_VARIANT_PTR(m_v, m_code_ofs)                                                                    \
	Variant *m_v;                                                                                           \
	{                                                                                                       \
		int address = _code_ptr[p_info->ip + 1 + (m_code_ofs)];                                             \
		int address_type = (address & ADDR_TYPE_MASK) >> ADDR_BITS;                                         \
		if (unlikely(address_type < 0 || address_type >= ADDR_TYPE_MAX)) {                                  \
			p_info->err_text = "Bad address type.";                                                         \
			return;                                                                                         \
		}                                                                                                   \
		int address_index = address & ADDR_MASK;                                                            \
		if (unlikely(address_index < 0 || address_index >= p_info->variant_address_limits[address_type])) { \
			if (address_type == ADDR_TYPE_MEMBER && !p_instance) {                                          \
				p_info->err_text = "Cannot access member without instance.";                                \
			} else {                                                                                        \
				p_info->err_text = "Bad address index.";                                                    \
			}                                                                                               \
			return;                                                                                         \
		}                                                                                                   \
		m_v = &p_info->variant_addresses[address_type][address_index];                                      \
		if (unlikely(!m_v))                                                                                 \
			return;                                                                                         \
	}

#else // !DEBUG_ENABLED
#define GD_ERR_BREAK(m_cond)
#define CHECK_SPACE(m_space)

#define GET_VARIANT_PTR(m_v, m_code_ofs)                                                                                                                      \
	Variant *m_v;                                                                                                                                             \
	{                                                                                                                                                         \
		int address = p_code_ptr[p_info->ip + 1 + (m_code_ofs)];                                                                                              \
		m_v = &p_info->variant_addresses[(address & GDScriptFunction::ADDR_TYPE_MASK) >> GDScriptFunction::ADDR_BITS][address & GDScriptFunction::ADDR_MASK]; \
		if (unlikely(!m_v))                                                                                                                                   \
			return;                                                                                                                                           \
	}

#endif // DEBUG_ENABLED

#define OP_EXEC_IMPLEMENT(m_opcode) void GDScriptFunction::_exec_##m_opcode OP_ARGS

#define GET_CALL_ARGUMENT(m_v) \
	int m_v = _code_ptr[p_info->ip + 1];

#define LOAD_INSTRUCTION_ARGS                        \
	int instr_arg_count = _code_ptr[p_info->ip + 1]; \
	for (int i = 0; i < instr_arg_count; i++) {      \
		GET_VARIANT_PTR(v, i + 1);                   \
		instruction_args[i] = v;                     \
	}                                                \
	p_info->ip += 1; // Offset to skip instruction argcount.

#define GET_INSTRUCTION_ARG(m_v, m_idx) \
	Variant *m_v = p_info->instruction_args[m_idx]

Variant GDScriptFunction::call(GDScriptInstance *p_instance, const Variant **p_args, int p_argcount, Callable::CallError &r_err, CallState *p_state) {
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
	while (ip < _code_size) {
		int last_opcode = _code_ptr[ip];
#else
	OPCODE_WHILE(true) {
#endif

		switch (_code_ptr[ip]) {
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
	Variant::Operator op = (Variant::Operator)_code_ptr[p_info->> ip + 4];
	GD_ERR_BREAK(op >= Variant::OP_MAX);

	GET_CALL_ARGUMENT(arg);
	bool is_validated = (arg & GDScriptFunction::IS_VALIDATED) != 0;

	if (is_validated) {
		const int OP_SIZE = 6;
		const int LHS_INDEX = 1;
		const int RHS_INDEX = 2;
		const int DST_INDEX = 3;

		CHECK_SPACE(OP_SIZE);

		int operator_idx = _code_ptr[p_info->ip + 4];
		GD_ERR_BREAK(operator_idx < 0 || operator_idx >= _operator_funcs_count);
		Variant::ValidatedOperatorEvaluator operator_func = _operator_funcs_ptr[operator_idx];

		GET_VARIANT_PTR(a, 0);
		GET_VARIANT_PTR(b, 1);
		GET_VARIANT_PTR(dst, 2);

		operator_func(a, b, dst);

		p_info->ip += OP_SIZE;
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
		uint32_t op_signature = _code_ptr[p_info->ip + 5];
		uint32_t actual_signature = (a->get_type() << 8) | (b->get_type());

#ifdef DEBUG_ENABLED
		if (op == Variant::OP_DIVIDE || op == Variant::OP_MODULE) {
			// Don't optimize division and modulo since there's not check for division by zero with validated calls.
			op_signature = 0xFFFF;
			_code_ptr[p_info->ip + 5] = op_signature;
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
				p_info->err_text = "Invalid operands '" + Variant::get_type_name(a->get_type()) + "' and '" + Variant::get_type_name(b->get_type()) + "' in operator '" + Variant::get_operator_name(op) + "'.";
#endif
				initializer_mutex.unlock();
			} else {
				Variant::Type ret_type = Variant::get_operator_return_type(op, a_type, b_type);
				VariantInternal::initialize(dst, ret_type);
				op_func(a, b, dst);

				// Check again in case another thread already set it.
				if (_code_ptr[p_info->ip + 5] == 0) {
					_code_ptr[p_info->ip + 5] = actual_signature;
					_code_ptr[p_info->ip + 6] = static_cast<int>(ret_type);
					Variant::ValidatedOperatorEvaluator *tmp = reinterpret_cast<Variant::ValidatedOperatorEvaluator *>(&_code_ptr[p_info->ip + 7]);
					*tmp = op_func;
				}
			}
			initializer_mutex.unlock();
		} else if (likely(op_signature == actual_signature)) {
			// If the signature matches, we can use the optimized path.
			Variant::Type ret_type = static_cast<Variant::Type>(_code_ptr[p_info->ip + 6]);
			Variant::ValidatedOperatorEvaluator op_func = *reinterpret_cast<Variant::ValidatedOperatorEvaluator *>(&_code_ptr[p_info->ip + 7]);

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
					p_info->err_text = ret;
					p_info->err_text += " in operator '" + Variant::get_operator_name(op) + "'.";
				} else {
					p_info->err_text = "Invalid operands '" + Variant::get_type_name(a->get_type()) + "' and '" + Variant::get_type_name(b->get_type()) + "' in operator '" + Variant::get_operator_name(op) + "'.";
				}
				return;
			}
			*dst = ret;
#endif
		}
		p_info->ip += OP_SIZE + _pointer_size;
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

			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_POS];
			GD_ERR_BREAK(builtin_type < 0 || builtin_type >= Variant::VARIANT_MAX);

			*dst = value->get_type() == builtin_type;
			p_info->ip += OP_SIZE;
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
			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_POS];
			int native_type_idx = _code_ptr[p_info->ip + NATIVE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			bool result = false;
			if (value->get_type() == Variant::ARRAY) {
				Array *array = VariantInternal::get_array(value);
				result = array->get_typed_builtin() == ((uint32_t)builtin_type) && array->get_typed_class_name() == native_type && array->get_typed_script() == *script_type;
			}

			*dst = result;
			p_info->ip += OP_SIZE;
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
			Variant::Type key_builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_A_POS];
			int key_native_type_idx = _code_ptr[p_info->ip + NATIVE_B_POS];
			GD_ERR_BREAK(key_native_type_idx < 0 || key_native_type_idx >= _global_names_count);
			const StringName key_native_type = _global_names_ptr[key_native_type_idx];

			GET_VARIANT_PTR(value_script_type, SCRIPT_B_POS);
			Variant::Type value_builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_B_POS];
			int value_native_type_idx = _code_ptr[p_info->ip + NATIVE_B_POS];
			GD_ERR_BREAK(value_native_type_idx < 0 || value_native_type_idx >= _global_names_count);
			const StringName value_native_type = _global_names_ptr[value_native_type_idx];

			bool result = false;
			if (value->get_type() == Variant::DICTIONARY) {
				Dictionary *dictionary = VariantInternal::get_dictionary(value);
				result = dictionary->get_typed_key_builtin() == ((uint32_t)key_builtin_type) && dictionary->get_typed_key_class_name() == key_native_type && dictionary->get_typed_key_script() == *key_script_type &&
						dictionary->get_typed_value_builtin() == ((uint32_t)value_builtin_type) && dictionary->get_typed_value_class_name() == value_native_type && dictionary->get_typed_value_script() == *value_script_type;
			}

			*dst = result;
			p_info->ip += OP_SIZE;
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

			int native_type_idx = _code_ptr[p_info->ip + NATIVE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			bool was_freed = false;
			Object *object = value->get_validated_object_with_check(was_freed);
			if (was_freed) {
				p_info->err_text = "Left operand of 'is' is a previously freed instance.";
				return;
			}

			*dst = object && ClassDB::is_parent_class(object->get_class_name(), native_type);
			p_info->ip += OP_SIZE;
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
				p_info->err_text = "Left operand of 'is' is a previously freed instance.";
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
			p_info->ip += OP_SIZE;
			break;
		}

		default:
			_report_invalid_arg_error(arg, _STR(OPCODE_TYPE_TEST));
			break;
	}
}

OP_EXEC_IMPLEMENT(OPCODE_SET) {
	GET_CALL_ARGUMENT(arg)

	switch (arg) {
		case SetGetArgs::KEYED: {
			const int
					OP_SIZE = 4,
					DST_POS = 1,
					IDX_POS = 2,
					VAL_POS = 3;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(index, IDX_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			bool valid;
#ifdef DEBUG_ENABLED
			Variant::VariantSetError err_code;
			dst->set(*index, *value, &valid, &err_code);
#else
			dst->set(*index, *value, &valid);
#endif
#ifdef DEBUG_ENABLED
			if (!valid) {
				if (dst->is_read_only()) {
					p_info->err_text = "Invalid assignment on read-only value (on base: '" + _get_var_type(dst) + "').";
				} else {
					Object *obj = dst->get_validated_object();
					String v = index->operator String();
					bool read_only_property = false;
					if (obj) {
						read_only_property = ClassDB::has_property(obj->get_class_name(), v) && (ClassDB::get_property_setter(obj->get_class_name(), v) == StringName());
					}
					if (read_only_property) {
						p_info->err_text = vformat(R"(Cannot set value into property "%s" (on base "%s") because it is read-only.)", v, _get_var_type(dst));
					} else {
						if (!v.is_empty()) {
							v = "'" + v + "'";
						} else {
							v = "of type '" + _get_var_type(index) + "'";
						}
						p_info->err_text = "Invalid assignment of property or key " + v + " with value of type '" + _get_var_type(value) + "' on a base object of type '" + _get_var_type(dst) + "'.";
						if (err_code == Variant::VariantSetError::SET_INDEXED_ERR) {
							p_info->err_text = "Invalid assignment of index " + v + " (on base: '" + _get_var_type(dst) + "') with value of type '" + _get_var_type(value) + "'.";
						}
					}
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::KEYED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					IDX_POS = 2,
					VAL_POS = 3,
					SETTER_POS = 4;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(index, IDX_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			int index_setter = _code_ptr[p_info->ip + SETTER_POS];
			GD_ERR_BREAK(index_setter < 0 || index_setter >= _keyed_setters_count);
			const Variant::ValidatedKeyedSetter setter = _keyed_setters_ptr[index_setter];

			bool valid;
			setter(dst, index, value, &valid);

#ifdef DEBUG_ENABLED
			if (!valid) {
				if (dst->is_read_only()) {
					p_info->err_text = "Invalid assignment on read-only value (on base: '" + _get_var_type(dst) + "').";
				} else {
					String v = index->operator String();
					if (!v.is_empty()) {
						v = "'" + v + "'";
					} else {
						v = "of type '" + _get_var_type(index) + "'";
					}
					p_info->err_text = "Invalid assignment of property or key " + v + " with value of type '" + _get_var_type(value) + "' on a base object of type '" + _get_var_type(dst) + "'.";
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::INDEXED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					IDX_POS = 2,
					VAL_POS = 3,
					SETTER_POS = 4;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(index, IDX_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			int index_setter = _code_ptr[p_info->ip + SETTER_POS];
			GD_ERR_BREAK(index_setter < 0 || index_setter >= _indexed_setters_count);
			const Variant::ValidatedIndexedSetter setter = _indexed_setters_ptr[index_setter];

			int64_t int_index = *VariantInternal::get_int(index);

			bool oob;
			setter(dst, int_index, value, &oob);

#ifdef DEBUG_ENABLED
			if (oob) {
				if (dst->is_read_only()) {
					p_info->err_text = "Invalid assignment on read-only value (on base: '" + _get_var_type(dst) + "').";
				} else {
					String v = index->operator String();
					if (!v.is_empty()) {
						v = "'" + v + "'";
					} else {
						v = "of type '" + _get_var_type(index) + "'";
					}
					p_info->err_text = "Out of bounds set index " + v + " (on base: '" + _get_var_type(dst) + "')";
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::NAMED: {
			const int
					OP_SIZE = 4,
					DST_POS = 1,
					VAL_POS = 2,
					IDX_POS = 3;
			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			int indexname = _code_ptr[p_info->ip + IDX_POS];

			GD_ERR_BREAK(indexname < 0 || indexname >= _global_names_count);
			const StringName *index = &_global_names_ptr[indexname];

			bool valid;
			dst->set_named(*index, *value, valid);

#ifdef DEBUG_ENABLED
			if (!valid) {
				if (dst->is_read_only()) {
					p_info->err_text = "Invalid assignment on read-only value (on base: '" + _get_var_type(dst) + "').";
				} else {
					Object *obj = dst->get_validated_object();
					bool read_only_property = false;
					if (obj) {
						read_only_property = ClassDB::has_property(obj->get_class_name(), *index) && (ClassDB::get_property_setter(obj->get_class_name(), *index) == StringName());
					}
					if (read_only_property) {
						p_info->err_text = vformat(R"(Cannot set value into property "%s" (on base "%s") because it is read-only.)", String(*index), _get_var_type(dst));
					} else {
						p_info->err_text = "Invalid assignment of property or key '" + String(*index) + "' with value of type '" + _get_var_type(value) + "' on a base object of type '" + _get_var_type(dst) + "'.";
					}
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::NAMED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					DST_POS = 1,
					VAL_POS = 2,
					IDX_POS = 3;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(value, VAL_POS);

			int index_setter = _code_ptr[p_info->ip + IDX_POS];
			GD_ERR_BREAK(index_setter < 0 || index_setter >= _setters_count);
			const Variant::ValidatedSetter setter = _setters_ptr[index_setter];

			setter(dst, value);
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::MEMBER: {
			const int
					OP_SIZE = 4,
					SRC_POS = 1,
					VAL_POS = 2,
					IDX_POS = 3;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(src, SRC_POS);
			int indexname = _code_ptr[p_info->ip + IDX_POS];
			GD_ERR_BREAK(indexname < 0 || indexname >= _global_names_count);
			const StringName *index = &_global_names_ptr[indexname];

			bool valid;
#ifndef DEBUG_ENABLED
			ClassDB::set_property(p_instance->owner, *index, *src, &valid);
#else
			bool ok = ClassDB::set_property(p_instance->owner, *index, *src, &valid);
			if (!ok) {
				p_info->err_text = "Internal error setting property: " + String(*index);
				return;
			} else if (!valid) {
				p_info->err_text = "Error setting property '" + String(*index) + "' with value of type " + Variant::get_type_name(src->get_type()) + ".";
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::STATIC_VARIABLE: {
			const int
					OP_SIZE = 5,
					VAL_POS = 1,
					CLASS_POS = 2,
					IDX_POS = 4;
			CHECK_SPACE(4);

			GET_VARIANT_PTR(value, VAL_POS);

			GET_VARIANT_PTR(_class, CLASS_POS);
			GDScript *gdscript = Object::cast_to<GDScript>(_class->operator Object *());
			GD_ERR_BREAK(!gdscript);

			int index = _code_ptr[p_info->ip + IDX_POS];
			GD_ERR_BREAK(index < 0 || index >= gdscript->static_variables.size());

			gdscript->static_variables.write[index] = *value;

			p_info->ip += OP_SIZE;
			break;
		}

		default:
			_report_invalid_arg_error(arg, _STR(OPCODE_SET));
			break;
	}
}

OP_EXEC_IMPLEMENT(OPCODE_GET) {
	GET_CALL_ARGUMENT(arg)

	switch (arg) {
		case SetGetArgs::KEYED: {
			const int
					OP_SIZE = 4,
					SRC_POS = 1,
					IDX_POS = 2,
					DST_POS = 3;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(src, SRC_POS);
			GET_VARIANT_PTR(index, IDX_POS);
			GET_VARIANT_PTR(dst, DST_POS);

			bool valid;
#ifdef DEBUG_ENABLED
			// Allow better error message in cases where src and dst are the same stack position.
			Variant::VariantGetError err_code;
			Variant ret = src->get(*index, &valid, &err_code);
#else
			*dst = src->get(*index, &valid);

#endif
#ifdef DEBUG_ENABLED
			if (!valid) {
				String v = index->operator String();
				if (!v.is_empty()) {
					v = "'" + v + "'";
				} else {
					v = "of type '" + _get_var_type(index) + "'";
				}
				p_info->err_text = "Invalid access to property or key " + v + " on a base object of type '" + _get_var_type(src) + "'.";
				if (err_code == Variant::VariantGetError::GET_INDEXED_ERR) {
					p_info->err_text = "Invalid access of index " + v + " on a base object of type: '" + _get_var_type(src) + "'.";
				}
				return;
			}
			*dst = ret;
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::KEYED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 5,
					SRC_POS = 1,
					KEY_POS = 2,
					DST_POS = 3,
					GETTER_POS = 5;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(src, SRC_POS);
			GET_VARIANT_PTR(key, KEY_POS);
			GET_VARIANT_PTR(dst, DST_POS);

			int index_getter = _code_ptr[p_info->ip + 4];
			GD_ERR_BREAK(index_getter < 0 || index_getter >= _keyed_getters_count);
			const Variant::ValidatedKeyedGetter getter = _keyed_getters_ptr[index_getter];

			bool valid;
#ifdef DEBUG_ENABLED
			// Allow better error message in cases where src and dst are the same stack position.
			Variant ret;
			getter(src, key, &ret, &valid);
#else
			getter(src, key, dst, &valid);
#endif
#ifdef DEBUG_ENABLED
			if (!valid) {
				String v = key->operator String();
				if (!v.is_empty()) {
					v = "'" + v + "'";
				} else {
					v = "of type '" + _get_var_type(key) + "'";
				}
				p_info->err_text = "Invalid access to property or key " + v + " on a base object of type '" + _get_var_type(src) + "'.";
				return;
			}
			*dst = ret;
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::INDEXED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 5,
					SRC_POS = 1,
					IDX_POS = 2,
					DST_POS = 3,
					GETTER_POS = 5;
			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(src, SRC_POS);
			GET_VARIANT_PTR(index, IDX_POS);
			GET_VARIANT_PTR(dst, DST_POS);

			int index_getter = _code_ptr[p_info->ip + GETTER_POS];
			GD_ERR_BREAK(index_getter < 0 || index_getter >= _indexed_getters_count);
			const Variant::ValidatedIndexedGetter getter = _indexed_getters_ptr[index_getter];

			int64_t int_index = *VariantInternal::get_int(index);

			bool oob;
			getter(src, int_index, dst, &oob);

#ifdef DEBUG_ENABLED
			if (oob) {
				String v = index->operator String();
				if (!v.is_empty()) {
					v = "'" + v + "'";
				} else {
					v = "of type '" + _get_var_type(index) + "'";
				}
				p_info->err_text = "Out of bounds get index " + v + " (on base: '" + _get_var_type(src) + "')";
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case SetGetArgs::NAMED: {
			const int
					OP_SIZE = 5,
					SRC_POS = 1,
					DST_POS = 2,
					IDX_NAME_POS = 4;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(src, SRC_POS);
			GET_VARIANT_PTR(dst, DST_POS);

			int indexname = _code_ptr[p_info->ip + IDX_NAME_POS];

			GD_ERR_BREAK(indexname < 0 || indexname >= _global_names_count);
			const StringName *index = &_global_names_ptr[indexname];

			bool valid;
#ifdef DEBUG_ENABLED
			//allow better error message in cases where src and dst are the same stack position
			Variant ret = src->get_named(*index, valid);

#else
			*dst = src->get_named(*index, valid);
#endif
#ifdef DEBUG_ENABLED
			if (!valid) {
				p_info->err_text = "Invalid access to property or key '" + index->operator String() + "' on a base object of type '" + _get_var_type(src) + "'.";
				return;
			}
			*dst = ret;
#endif
			p_info->ip += OP_SIZE;
			break;
		}

		case SetGetArgs::NAMED | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					SRC_POS = 1,
					DST_POS = 2,
					IDX_NAME_POS = 4;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(src, SRC_POS);
			GET_VARIANT_PTR(dst, DST_POS);

			int index_getter = _code_ptr[p_info->ip + 3];
			GD_ERR_BREAK(index_getter < 0 || index_getter >= _getters_count);
			const Variant::ValidatedGetter getter = _getters_ptr[index_getter];

			getter(src, dst);
			p_info->ip += OP_SIZE;
			break;
		}

		case SetGetArgs::MEMBER: {
			const int
					OP_SIZE = 4,
					DST_POS = 1,
					IDX_NAME_POS = 3;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			int indexname = _code_ptr[p_info->ip + IDX_NAME_POS];
			GD_ERR_BREAK(indexname < 0 || indexname >= _global_names_count);
			const StringName *index = &_global_names_ptr[indexname];
#ifndef DEBUG_ENABLED
			ClassDB::get_property(p_instance->owner, *index, *dst);
#else
			bool ok = ClassDB::get_property(p_instance->owner, *index, *dst);
			if (!ok) {
				p_info->err_text = "Internal error getting property: " + String(*index);
				return;
			}
#endif
			p_info->ip += OP_SIZE;
			break;
		}

		case SetGetArgs::STATIC_VARIABLE: {
			static int
					OP_SIZE = 5,
					TARGET_POS = 1,
					CLASS_POS = 2,
					IDX_POS = 4;

			CHECK_SPACE(OP_SIZE);

			GET_VARIANT_PTR(target, TARGET_POS);

			GET_VARIANT_PTR(_class, CLASS_POS);
			GDScript *gdscript = Object::cast_to<GDScript>(_class->operator Object *());
			GD_ERR_BREAK(!gdscript);

			int index = _code_ptr[p_info->ip + IDX_POS];
			GD_ERR_BREAK(index < 0 || index >= gdscript->static_variables.size());

			*target = gdscript->static_variables[index];

			p_info->ip += OP_SIZE;
			break;
		}

		default: {
			_report_invalid_arg_error(arg, _STR(OPCODE_GET));
			break;
		}
	}
}

OP_EXEC_IMPLEMENT(OPCODE_ASSIGN) {
	GET_CALL_ARGUMENT(arg);

	switch (arg) {
		case AssignArguments::OTHER: {
			const int
					OP_SIZE = 4,
					DST_POS = 1,
					SRC_POS = 2;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

			*dst = *src;

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::NULL_VAL: {
			const int
					OP_SIZE = 3,
					DST_POS = 1;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);

			*dst = Variant();

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::TRUE: {
			const int
					OP_SIZE = 3,
					DST_POS = 1;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);

			*dst = true;

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::FALSE: {
			const int
					OP_SIZE = 3,
					DST_POS = 1;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);

			*dst = Variant();

			p_info->ip += false;
			break;
		}

		case AssignArguments::TYPED_BUILTIN: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					SRC_POS = 2,
					TYPE_POS = 4;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

			Variant::Type var_type = (Variant::Type)_code_ptr[p_info->ip + TYPE_POS];
			GD_ERR_BREAK(var_type < 0 || var_type >= Variant::VARIANT_MAX);

			if (src->get_type() != var_type) {
#ifdef DEBUG_ENABLED
				if (Variant::can_convert_strict(src->get_type(), var_type)) {
#endif // DEBUG_ENABLED
					Callable::CallError ce;
					Variant::construct(var_type, *dst, const_cast<const Variant **>(&src), 1, ce);
				} else {
#ifdef DEBUG_ENABLED
					p_info->err_text = "Trying to assign value of type '" + Variant::get_type_name(src->get_type()) +
							"' to a variable of type '" + Variant::get_type_name(var_type) + "'.";
					return;
				}
			} else {
#endif // DEBUG_ENABLED
				*dst = *src;
			}

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::TYPED_ARRAY: {
			const int
					OP_SIZE = 7,
					DST_POS = 1,
					SRC_POS = 2,
					SCRIPT_TYPE_POS = 3,
					BUILTIN_TYPE_POS = 5,
					NATIVE_TYPE_POS = 6;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

			GET_VARIANT_PTR(script_type, SCRIPT_TYPE_POS);
			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_TYPE_POS];
			int native_type_idx = _code_ptr[p_info->ip + NATIVE_TYPE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			if (src->get_type() != Variant::ARRAY) {
#ifdef DEBUG_ENABLED
				p_info->err_text = vformat(R"(Trying to assign a value of type "%s" to a variable of type "Array[%s]".)",
						_get_var_type(src), _get_element_type(builtin_type, native_type, *script_type));
#endif // DEBUG_ENABLED
				return;
			}

			Array *array = VariantInternal::get_array(src);

			if (array->get_typed_builtin() != ((uint32_t)builtin_type) || array->get_typed_class_name() != native_type || array->get_typed_script() != *script_type) {
#ifdef DEBUG_ENABLED
				p_info->err_text = vformat(R"(Trying to assign an array of type "%s" to a variable of type "Array[%s]".)",
						_get_var_type(src), _get_element_type(builtin_type, native_type, *script_type));
#endif // DEBUG_ENABLED
				return;
			}

			*dst = *src;

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::TYPED_DICTIONARY: {
			const int
					OP_SIZE = 10,
					DST_POS = 1,
					SRC_POS = 2,
					KEY_SCRIPT_TYPE_POS = 3,
					VAL_SCRIPT_TYPE_POS = 4,
					KEY_TYPE_POS = 6,
					KEY_NATIVE_TYPE_POS = 7,
					VAL_TYPE_POS = 8,
					VAL_NATIVE_TYPE_POS = 9;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

			GET_VARIANT_PTR(key_script_type, KEY_SCRIPT_TYPE_POS);
			Variant::Type key_builtin_type = (Variant::Type)_code_ptr[p_info->ip + KEY_SCRIPT_TYPE_POS];
			int key_native_type_idx = _code_ptr[p_info->ip + KEY_NATIVE_TYPE_POS];
			GD_ERR_BREAK(key_native_type_idx < 0 || key_native_type_idx >= _global_names_count);
			const StringName key_native_type = _global_names_ptr[key_native_type_idx];

			GET_VARIANT_PTR(value_script_type, VAL_SCRIPT_TYPE_POS);
			Variant::Type value_builtin_type = (Variant::Type)_code_ptr[p_info->ip + VAL_TYPE_POS];
			int value_native_type_idx = _code_ptr[p_info->ip + VAL_NATIVE_TYPE_POS];
			GD_ERR_BREAK(value_native_type_idx < 0 || value_native_type_idx >= _global_names_count);
			const StringName value_native_type = _global_names_ptr[value_native_type_idx];

			if (src->get_type() != Variant::DICTIONARY) {
#ifdef DEBUG_ENABLED
				p_info->err_text = vformat(R"(Trying to assign a value of type "%s" to a variable of type "Dictionary[%s, %s]".)",
						_get_var_type(src), _get_element_type(key_builtin_type, key_native_type, *key_script_type),
						_get_element_type(value_builtin_type, value_native_type, *value_script_type));
#endif // DEBUG_ENABLED
				return;
			}

			Dictionary *dictionary = VariantInternal::get_dictionary(src);

			if (dictionary->get_typed_key_builtin() != ((uint32_t)key_builtin_type) || dictionary->get_typed_key_class_name() != key_native_type || dictionary->get_typed_key_script() != *key_script_type ||
					dictionary->get_typed_value_builtin() != ((uint32_t)value_builtin_type) || dictionary->get_typed_value_class_name() != value_native_type || dictionary->get_typed_value_script() != *value_script_type) {
#ifdef DEBUG_ENABLED
				p_info->err_text = vformat(R"(Trying to assign a dictionary of type "%s" to a variable of type "Dictionary[%s, %s]".)",
						_get_var_type(src), _get_element_type(key_builtin_type, key_native_type, *key_script_type),
						_get_element_type(value_builtin_type, value_native_type, *value_script_type));
#endif // DEBUG_ENABLED
				return;
			}

			*dst = *src;

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::TYPED_NATIVE: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					SRC_POS = 2,
					TYPE_POS = 3;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

#ifdef DEBUG_ENABLED
			GET_VARIANT_PTR(type, TYPE_POS);
			GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(type->operator Object *());
			GD_ERR_BREAK(!nc);
			if (src->get_type() != Variant::OBJECT && src->get_type() != Variant::NIL) {
				p_info->err_text = "Trying to assign value of type '" + Variant::get_type_name(src->get_type()) +
						"' to a variable of type '" + nc->get_name() + "'.";
				return;
			}

			if (src->get_type() == Variant::OBJECT) {
				bool was_freed = false;
				Object *src_obj = src->get_validated_object_with_check(was_freed);
				if (!src_obj && was_freed) {
					p_info->err_text = "Trying to assign invalid previously freed instance.";
					return;
				}

				if (src_obj && !ClassDB::is_parent_class(src_obj->get_class_name(), nc->get_name())) {
					p_info->err_text = "Trying to assign value of type '" + src_obj->get_class_name() +
							"' to a variable of type '" + nc->get_name() + "'.";
					return;
				}
			}
#endif // DEBUG_ENABLED
			*dst = *src;

			p_info->ip += OP_SIZE;
			break;
		}

		case AssignArguments::TYPED_SCRIPT: {
			const int
					OP_SIZE = 5,
					DST_POS = 1,
					SRC_POS = 2,
					TYPE_POS = 3;

			CHECK_SPACE(OP_SIZE);
			GET_VARIANT_PTR(dst, DST_POS);
			GET_VARIANT_PTR(src, SRC_POS);

#ifdef DEBUG_ENABLED
			GET_VARIANT_PTR(type, TYPE_POS);
			Script *base_type = Object::cast_to<Script>(type->operator Object *());

			GD_ERR_BREAK(!base_type);

			if (src->get_type() != Variant::OBJECT && src->get_type() != Variant::NIL) {
				p_info->err_text = "Trying to assign a non-object value to a variable of type '" + base_type->get_path().get_file() + "'.";
				return;
			}

			if (src->get_type() == Variant::OBJECT) {
				bool was_freed = false;
				Object *val_obj = src->get_validated_object_with_check(was_freed);
				if (!val_obj && was_freed) {
					p_info->err_text = "Trying to assign invalid previously freed instance.";
					return;
				}

				if (val_obj) { // src is not null
					ScriptInstance *scr_inst = val_obj->get_script_instance();
					if (!scr_inst) {
						p_info->err_text = "Trying to assign value of type '" + val_obj->get_class_name() +
								"' to a variable of type '" + base_type->get_path().get_file() + "'.";
						return;
					}

					Script *src_type = scr_inst->get_script().ptr();
					bool valid = false;

					while (src_type) {
						if (src_type == base_type) {
							valid = true;
							break;
						}
						src_type = src_type->get_base_script().ptr();
					}

					if (!valid) {
						p_info->err_text = "Trying to assign value of type '" + val_obj->get_script_instance()->get_script()->get_path().get_file() +
								"' to a variable of type '" + base_type->get_path().get_file() + "'.";
						return;
					}
				}
			}
#endif // DEBUG_ENABLED

			*dst = *src;

			p_info->ip += OP_SIZE;
			break;
		}

		default: {
			_report_invalid_arg_error(arg, _STR(OPCODE_ASSIGN));
			break;
		}
	}
}

OP_EXEC_IMPLEMENT(OPCODE_CAST) {
	GET_CALL_ARGUMENT(arg);

	const int
			OP_SIZE = 5,
			SRC_POS = 1,
			DST_POS = 2,
			TYPE_POS = 4;

	CHECK_SPACE(OP_SIZE);
	GET_VARIANT_PTR(src, SRC_POS);
	GET_VARIANT_PTR(dst, DST_POS);

	switch (arg) {
		case CastArgs::BUILTIN: {
			Variant::Type to_type = (Variant::Type)_code_ptr[p_info->ip + TYPE_POS];
			GD_ERR_BREAK(to_type < 0 || to_type >= Variant::VARIANT_MAX);

#ifdef DEBUG_ENABLED
			if (src->operator Object *() && !src->get_validated_object()) {
				p_info->err_text = "Trying to cast a freed object.";
				return;
			}
#endif

			Callable::CallError err;
			Variant::construct(to_type, *dst, (const Variant **)&src, 1, err);

#ifdef DEBUG_ENABLED
			if (err.error != Callable::CallError::CALL_OK) {
				p_info->err_text = "Invalid cast: could not convert value to '" + Variant::get_type_name(to_type) + "'.";
				return;
			}
#endif
			break;
		}

		case CastArgs::NATIVE: {
			GET_VARIANT_PTR(to_type, 2);
			GDScriptNativeClass *nc = Object::cast_to<GDScriptNativeClass>(to_type->operator Object *());
			GD_ERR_BREAK(!nc);

#ifdef DEBUG_ENABLED
			if (src->operator Object *() && !src->get_validated_object()) {
				p_info->err_text = "Trying to cast a freed object.";
				return;
			}
			if (src->get_type() != Variant::OBJECT && src->get_type() != Variant::NIL) {
				p_info->err_text = "Invalid cast: can't convert a non-object value to an object type.";
				return;
			}
#endif
			Object *src_obj = src->operator Object *();

			if (src_obj && !ClassDB::is_parent_class(src_obj->get_class_name(), nc->get_name())) {
				*dst = Variant(); // invalid cast, assign NULL
			} else {
				*dst = *src;
			}
			break;
		}

		case CastArgs::SCRIPT: {
			GET_VARIANT_PTR(to_type, 2);
			Script *base_type = Object::cast_to<Script>(to_type->operator Object *());

			GD_ERR_BREAK(!base_type);

#ifdef DEBUG_ENABLED
			if (src->operator Object *() && !src->get_validated_object()) {
				p_info->err_text = "Trying to cast a freed object.";
				return;
			}
			if (src->get_type() != Variant::OBJECT && src->get_type() != Variant::NIL) {
				p_info->err_text = "Trying to assign a non-object value to a variable of type '" + base_type->get_path().get_file() + "'.";
				return;
			}
#endif

			bool valid = false;

			if (src->get_type() != Variant::NIL && src->operator Object *() != nullptr) {
				ScriptInstance *scr_inst = src->operator Object *()->get_script_instance();

				if (scr_inst) {
					Script *src_type = src->operator Object *()->get_script_instance()->get_script().ptr();

					while (src_type) {
						if (src_type == base_type) {
							valid = true;
							break;
						}
						src_type = src_type->get_base_script().ptr();
					}
				}
			}

			if (valid) {
				*dst = *src; // Valid cast, copy the source object
			} else {
				*dst = Variant(); // invalid cast, assign NULL
			}
		}

		default: {
			_report_invalid_arg_error(arg, _STR(OPCODE_CAST));
			break;
		}
	}

	p_info->ip += OP_SIZE;
}

OP_EXEC_IMPLEMENT(OPCODE_CONSTRUCT) {
	GET_CALL_ARGUMENT(arg);

	const int
			ARGS_POS = 2;

	int instr_arg_count = _code_ptr[p_info->ip + 2];
	for (int i = 0; i < instr_arg_count; i++) {
		GET_VARIANT_PTR(v, i + 1);
		p_info->instruction_args[i] = v;
	}
	p_info->ip += 1; // Offset to skip instruction argcount.

	p_info->ip += instr_arg_count;

	int argc = _code_ptr[p_info->ip + ARGS_POS];

	switch (arg) {
		case ConstructArguments::OTHER: {
			const int OP_SIZE = 3;
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			Variant::Type t = Variant::Type(_code_ptr[p_info->ip + 2]);

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			Callable::CallError err;
			Variant::construct(t, *dst, (const Variant **)argptrs, argc, err);

#ifdef DEBUG_ENABLED
			if (err.error != Callable::CallError::CALL_OK) {
				p_info->err_text = _get_call_error("'" + Variant::get_type_name(t) + "' constructor", (const Variant **)argptrs, *dst, err);
				return;
			}
#endif
			p_info->ip += OP_SIZE + 1;
			break;
		}

		case ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 3,
					CONSTRUCTOR_POS = 3;
			CHECK_SPACE(OP_SIZE + instr_arg_count);
			int constructor_idx = _code_ptr[p_info->ip + CONSTRUCTOR_POS];
			GD_ERR_BREAK(constructor_idx < 0 || constructor_idx >= _constructors_count);
			Variant::ValidatedConstructor constructor = _constructors_ptr[constructor_idx];

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			constructor(dst, (const Variant **)argptrs);

			p_info->ip += OP_SIZE + 1;
			break;
		}

		case ConstructArguments::ARRAY: {
			const int OP_SIZE = 2;
			CHECK_SPACE(OP_SIZE + instr_arg_count);
			Array array;
			array.resize(argc);

			for (int i = 0; i < argc; i++) {
				array[i] = *(p_info->instruction_args[i]);
			}

			GET_INSTRUCTION_ARG(dst, argc);
			*dst = Variant(); // Clear potential previous typed array.

			*dst = array;

			p_info->ip += OP_SIZE + 1;
			break;
		}

		case ConstructArguments::TYPED_ARRAY: {
			const int
					OP_SIZE = 4,
					BUILTIN_TYPE_POS = 3,
					NATIVE_TYPE_POS = 4;
			CHECK_SPACE(OP_SIZE + instr_arg_count);
			GET_INSTRUCTION_ARG(script_type, argc + 1);
			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_info->ip + BUILTIN_TYPE_POS];
			int native_type_idx = _code_ptr[p_info->ip + NATIVE_TYPE_POS];
			GD_ERR_BREAK(native_type_idx < 0 || native_type_idx >= _global_names_count);
			const StringName native_type = _global_names_ptr[native_type_idx];

			Array array;
			array.resize(argc);
			for (int i = 0; i < argc; i++) {
				array[i] = *(p_info->instruction_args[i]);
			}

			GET_INSTRUCTION_ARG(dst, argc);
			*dst = Variant(); // Clear potential previous typed array.

			*dst = Array(array, builtin_type, native_type, *script_type);

			p_info->ip += OP_SIZE + 1;
			break;
		}

		case ConstructArguments::DICTIONARY: {
			const int OP_SIZE = 3;
			CHECK_SPACE(OP_SIZE + instr_arg_count);
			Dictionary dict;

			for (int i = 0; i < argc; i++) {
				GET_INSTRUCTION_ARG(k, i * 2 + 0);
				GET_INSTRUCTION_ARG(v, i * 2 + 1);
				dict[*k] = *v;
			}

			GET_INSTRUCTION_ARG(dst, argc * 2);

			*dst = Variant(); // Clear potential previous typed dictionary.

			*dst = dict;

			p_info->ip += OP_SIZE;
			break;
		}

		case ConstructArguments::TYPED_DICTIONARY: {
			const int
					OP_SIZE = 7,
					KEY_BUILTIN_TYPE_POS = 3,
					KEY_NATIVE_TYPE_POS = 4,
					VAL_BUILTIN_TYPE_POS = 5,
					VAL_NATIVE_TYPE_POS = 6;
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			GET_INSTRUCTION_ARG(key_script_type, argc * 2 + 1);
			Variant::Type key_builtin_type = (Variant::Type)_code_ptr[p_info->ip + KEY_BUILTIN_TYPE_POS];
			int key_native_type_idx = _code_ptr[p_info->ip + KEY_NATIVE_TYPE_POS];
			GD_ERR_BREAK(key_native_type_idx < 0 || key_native_type_idx >= _global_names_count);
			const StringName key_native_type = _global_names_ptr[key_native_type_idx];

			GET_INSTRUCTION_ARG(value_script_type, argc * 2 + 2);
			Variant::Type value_builtin_type = (Variant::Type)_code_ptr[p_info->ip + VAL_BUILTIN_TYPE_POS];
			int value_native_type_idx = _code_ptr[p_info->ip + VAL_NATIVE_TYPE_POS];
			GD_ERR_BREAK(value_native_type_idx < 0 || value_native_type_idx >= _global_names_count);
			const StringName value_native_type = _global_names_ptr[value_native_type_idx];

			Dictionary dict;

			for (int i = 0; i < argc; i++) {
				GET_INSTRUCTION_ARG(k, i * 2 + 0);
				GET_INSTRUCTION_ARG(v, i * 2 + 1);
				dict[*k] = *v;
			}

			GET_INSTRUCTION_ARG(dst, argc * 2);

			*dst = Variant(); // Clear potential previous typed dictionary.

			*dst = Dictionary(dict, key_builtin_type, key_native_type, *key_script_type, value_builtin_type, value_native_type, *value_script_type);

			p_info->ip += OP_SIZE;
			break;
		}

		default: {
			_report_invalid_arg_error(arg, _STR(OPCODE_CONSTRUCT));
			break;
		}
	}
}

OP_EXEC_IMPLEMENT(OPCODE_CALL) {
	GET_CALL_ARGUMENT(arg);

#define LOAD_ARGS                                    \
	int instr_arg_count = _code_ptr[p_info->ip + 2]; \
	for (int i = 0; i < instr_arg_count; i++) {      \
		GET_VARIANT_PTR(v, i + 1);                   \
		p_info->instruction_args[i] = v;             \
	}                                                \
	p_info->ip += 1; // Offset to skip instruction argcount.

	bool is_validated = arg & ArgumentMask::IS_VALIDATED;

	switch (arg) {
		case CallArguments::ASYNC:
		case CallArguments::RETURN:
		case CallArguments::OTHER: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHODNAME = 3;
			bool call_ret = arg != CallArguments::OTHER;
#ifdef DEBUG_ENABLED
			bool call_async = arg == CallArguments::ASYNC;
#endif
			LOAD_ARGS
			CHECK_SPACE(3 + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			int methodname_idx = _code_ptr[p_info->ip + METHODNAME];
			GD_ERR_BREAK(methodname_idx < 0 || methodname_idx >= _global_names_count);
			const StringName *methodname = &_global_names_ptr[methodname_idx];

			GET_INSTRUCTION_ARG(base, argc);
			Variant **argptrs = p_info->instruction_args;

#ifdef DEBUG_ENABLED
			uint64_t call_time = 0;

			if (GDScriptLanguage::get_singleton()->profiling) {
				call_time = OS::get_singleton()->get_ticks_usec();
			}
			Variant::Type base_type = base->get_type();
			Object *base_obj = base->get_validated_object();
			StringName base_class = base_obj ? base_obj->get_class_name() : StringName();
#endif

			Variant temp_ret;
			Callable::CallError err;
			if (call_ret) {
				GET_INSTRUCTION_ARG(ret, argc + 1);
				base->callp(*methodname, (const Variant **)argptrs, argc, temp_ret, err);
				*ret = temp_ret;
#ifdef DEBUG_ENABLED
				if (ret->get_type() == Variant::NIL) {
					if (base_type == Variant::OBJECT) {
						if (base_obj) {
							MethodBind *method = ClassDB::get_method(base_class, *methodname);
							if (*methodname == CoreStringName(free_) || (method && !method->has_return())) {
								p_info->err_text = R"(Trying to get a return value of a method that returns "void")";
								return;
							}
						}
					} else if (Variant::has_builtin_method(base_type, *methodname) && !Variant::has_builtin_method_return_value(base_type, *methodname)) {
						p_info->err_text = R"(Trying to get a return value of a method that returns "void")";
						return;
					}
				}

				if (!call_async && ret->get_type() == Variant::OBJECT) {
					// Check if getting a function state without await.
					bool was_freed = false;
					Object *obj = ret->get_validated_object_with_check(was_freed);

					if (obj && obj->is_class_ptr(GDScriptFunctionState::get_class_ptr_static())) {
						p_info->err_text = R"(Trying to call an async function without "await".)";
						return;
					}
				}
#endif
			} else {
				base->callp(*methodname, (const Variant **)argptrs, argc, temp_ret, err);
			}
#ifdef DEBUG_ENABLED

			if (GDScriptLanguage::get_singleton()->profiling) {
				uint64_t t_taken = OS::get_singleton()->get_ticks_usec() - call_time;
				if (GDScriptLanguage::get_singleton()->profile_native_calls && _profile_count_as_native(base_obj, *methodname)) {
					_profile_native_call(t_taken, *methodname, base_class);
				}
				p_info->function_call_time += t_taken;
			}

			if (err.error != Callable::CallError::CALL_OK) {
				String methodstr = *methodname;
				String basestr = _get_var_type(base);
				bool is_callable = false;

				if (methodstr == "call") {
					if (argc >= 1 && base->get_type() != Variant::CALLABLE) {
						methodstr = String(*argptrs[0]) + " (via call)";
						if (err.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
							err.argument += 1;
						}
					} else {
						methodstr = base->operator String() + " (Callable)";
						is_callable = true;
					}
				} else if (methodstr == "free") {
					if (err.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
						if (base->is_ref_counted()) {
							p_info->err_text = "Attempted to free a RefCounted object.";
							return;
						} else if (base->get_type() == Variant::OBJECT) {
							p_info->err_text = "Attempted to free a locked object (calling or emitting).";
							return;
						}
					}
				} else if (methodstr == "call_recursive" && basestr == "TreeItem") {
					if (argc >= 1) {
						methodstr = String(*argptrs[0]) + " (via TreeItem.call_recursive)";
						if (err.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
							err.argument += 1;
						}
					}
				}
				p_info->err_text = _get_call_error("function '" + methodstr + (is_callable ? "" : "' in base '" + basestr) + "'", (const Variant **)argptrs, temp_ret, err);
				return;
			}
#endif // DEBUG_ENABLED

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::UTILITY: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;
			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _global_names_count);
			StringName function = _global_names_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			Callable::CallError err;
			Variant::call_utility_function(function, dst, (const Variant **)argptrs, argc, err);

#ifdef DEBUG_ENABLED
			if (err.error != Callable::CallError::CALL_OK) {
				String methodstr = function;
				if (dst->get_type() == Variant::STRING && !dst->operator String().is_empty()) {
					// Call provided error string.
					p_info->err_text = vformat(R"*(Error calling utility function "%s()": %s)*", methodstr, *dst);
				} else {
					p_info->err_text = _get_call_error(vformat(R"*(utility function "%s()")*", methodstr), (const Variant **)argptrs, *dst, err);
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::UTILITY | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _utilities_count);
			Variant::ValidatedUtilityFunction function = _utilities_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			function(dst, (const Variant **)argptrs, argc);

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::GDSCRIPT_UTILITY: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _gds_utilities_count);
			GDScriptUtilityFunctions::FunctionPtr function = _gds_utilities_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			Callable::CallError err;
			function(dst, (const Variant **)argptrs, argc, err);

#ifdef DEBUG_ENABLED
			if (err.error != Callable::CallError::CALL_OK) {
				String methodstr = gds_utilities_names[_code_ptr[ip + 2]];
				if (dst->get_type() == Variant::STRING && !dst->operator String().is_empty()) {
					// Call provided error string.
					p_info->err_text = vformat(R"*(Error calling GDScript utility function "%s()": %s)*", methodstr, *dst);
				} else {
					p_info->err_text = _get_call_error(vformat(R"*(GDScript utility function "%s()")*", methodstr), (const Variant **)argptrs, *dst, err);
				}
				return;
			}
#endif
			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::BUILTIN_TYPE | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GET_INSTRUCTION_ARG(base, argc);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _builtin_methods_count);
			Variant::ValidatedBuiltInMethod method = _builtin_methods_ptr[_code_ptr[p_info->ip + METHOD_POS]];
			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(ret, argc + 1);
			method(base, (const Variant **)argptrs, argc, ret);

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::SELF_BASE: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			int self_fun = _code_ptr[p_info->ip + METHOD_POS];
#ifdef DEBUG_ENABLED
			if (self_fun < 0 || self_fun >= _global_names_count) {
				p_info->err_text = "compiler bug, function name not found";
				return;
			}
#endif
			const StringName *methodname = &_global_names_ptr[self_fun];

			Variant **argptrs = p_info->instruction_args;

			GET_INSTRUCTION_ARG(dst, argc);

			const GDScript *gds = _script;

			HashMap<StringName, GDScriptFunction *>::ConstIterator E;
			while (gds->base.ptr()) {
				gds = gds->base.ptr();
				E = gds->member_functions.find(*methodname);
				if (E) {
					break;
				}
			}

			Callable::CallError err;

			if (E) {
				*dst = E->value->call(p_instance, (const Variant **)argptrs, argc, err);
			} else if (gds->native.ptr()) {
				if (*methodname != GDScriptLanguage::get_singleton()->strings._init) {
					MethodBind *mb = ClassDB::get_method(gds->native->get_name(), *methodname);
					if (!mb) {
						err.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
					} else {
						*dst = mb->call(p_instance->owner, (const Variant **)argptrs, argc, err);
					}
				} else {
					err.error = Callable::CallError::CALL_OK;
				}
			} else {
				if (*methodname != GDScriptLanguage::get_singleton()->strings._init) {
					err.error = Callable::CallError::CALL_ERROR_INVALID_METHOD;
				} else {
					err.error = Callable::CallError::CALL_OK;
				}
			}

			if (err.error != Callable::CallError::CALL_OK) {
				String methodstr = *methodname;
				p_info->err_text = _get_call_error("function '" + methodstr + "'", (const Variant **)argptrs, *dst, err);

				return;
			}

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::METHOD_BIND:
		case CallArguments::METHOD_BIND_RETURN: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;

			bool call_ret = arg == CallArguments::METHOD_BIND_RETURN;
			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);
			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);
			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _methods_count);
			MethodBind *method = _methods_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			GET_INSTRUCTION_ARG(base, argc);

#ifdef DEBUG_ENABLED
			bool freed = false;
			Object *base_obj = base->get_validated_object_with_check(freed);
			if (freed) {
				p_info->err_text = METHOD_CALL_ON_FREED_INSTANCE_ERROR(method);
				return;
			} else if (!base_obj) {
				p_info->err_text = METHOD_CALL_ON_NULL_VALUE_ERROR(method);
				return;
			}
#else
			Object *base_obj = base->operator Object *();
#endif
			Variant **argptrs = p_info->instruction_args;

#ifdef DEBUG_ENABLED
			uint64_t call_time = 0;
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				call_time = OS::get_singleton()->get_ticks_usec();
			}
#endif

			Variant temp_ret;
			Callable::CallError err;
			if (call_ret) {
				GET_INSTRUCTION_ARG(ret, argc + 1);
				temp_ret = method->call(base_obj, (const Variant **)argptrs, argc, err);
				*ret = temp_ret;
			} else {
				temp_ret = method->call(base_obj, (const Variant **)argptrs, argc, err);
			}

#ifdef DEBUG_ENABLED

			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				uint64_t t_taken = OS::get_singleton()->get_ticks_usec() - call_time;
				_profile_native_call(t_taken, method->get_name(), method->get_instance_class());
				p_function_call_time += t_taken;
			}

			if (err.error != Callable::CallError::CALL_OK) {
				String methodstr = method->get_name();
				String basestr = _get_var_type(base);

				if (methodstr == "call") {
					if (argc >= 1) {
						methodstr = String(*argptrs[0]) + " (via call)";
						if (err.error == Callable::CallError::CALL_ERROR_INVALID_ARGUMENT) {
							err.argument += 1;
						}
					}
				} else if (methodstr == "free") {
					if (err.error == Callable::CallError::CALL_ERROR_INVALID_METHOD) {
						if (base->is_ref_counted()) {
							p_info->err_text = "Attempted to free a RefCounted object.";
							return;
						} else if (base->get_type() == Variant::OBJECT) {
							p_info->err_text = "Attempted to free a locked object (calling or emitting).";
							return;
						}
					}
				}
				p_info->err_text = _get_call_error("function '" + methodstr + "' in base '" + basestr + "'", (const Variant **)argptrs, temp_ret, err);
				return;
			}
#endif
			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::BUILTIN_STATIC: {
			const int
					OP_SIZE = 5,
					METHOD_POS = 2,
					METHOD_NAME_POS = 3,
					ARGS_POS = 4;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= Variant::VARIANT_MAX);
			Variant::Type builtin_type = (Variant::Type)_code_ptr[p_info->ip + METHOD_POS];

			int methodname_idx = _code_ptr[p_info->ip + METHOD_NAME_POS];
			GD_ERR_BREAK(methodname_idx < 0 || methodname_idx >= _global_names_count);
			const StringName *methodname = &_global_names_ptr[methodname_idx];

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GET_INSTRUCTION_ARG(ret, argc);

			const Variant **argptrs = const_cast<const Variant **>(p_info->instruction_args);

			Callable::CallError err;
			Variant::call_static(builtin_type, *methodname, argptrs, argc, *ret, err);

#ifdef DEBUG_ENABLED
			if (err.error != Callable::CallError::CALL_OK) {
				p_info->err_text = _get_call_error("static function '" + methodname->operator String() + "' in type '" + Variant::get_type_name(builtin_type) + "'", argptrs, *ret, err);
				return;
			}
#endif

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::NATIVE_STATIC: {
			const int
					OP_SIZE = 4,
					METHOD_POS = 2,
					ARGS_POS = 3;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _methods_count);
			MethodBind *method = _methods_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GET_INSTRUCTION_ARG(ret, argc);

			const Variant **argptrs = const_cast<const Variant **>(p_info->instruction_args);

#ifdef DEBUG_ENABLED
			uint64_t call_time = 0;
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				call_time = OS::get_singleton()->get_ticks_usec();
			}
#endif

			Callable::CallError err;
			*ret = method->call(nullptr, argptrs, argc, err);

#ifdef DEBUG_ENABLED
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				uint64_t t_taken = OS::get_singleton()->get_ticks_usec() - call_time;
				_profile_native_call(t_taken, method->get_name(), method->get_instance_class());
				p_function_call_time += t_taken;
			}
#endif

			if (err.error != Callable::CallError::CALL_OK) {
				p_info->err_text = _get_call_error("static function '" + method->get_name().operator String() + "' in type '" + method->get_instance_class().operator String() + "'", argptrs, *ret, err);
				return;
			}

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::NATIVE_STATIC_RETURN | ArgumentMask::IS_VALIDATED:
		case CallArguments::NATIVE_STATIC_NO_RETURN | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;
			bool is_return = arg == CallArguments::NATIVE_STATIC_RETURN | ArgumentMask::IS_VALIDATED;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _methods_count);
			MethodBind *method = _methods_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			Variant **argptrs = p_info->instruction_args;

#ifdef DEBUG_ENABLED
			uint64_t call_time = 0;
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				call_time = OS::get_singleton()->get_ticks_usec();
			}
#endif
			if (is_return) {
				GET_INSTRUCTION_ARG(ret, argc);
				method->validated_call(nullptr, (const Variant **)argptrs, ret);
			}
#ifdef DEBUG_ENABLED
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				uint64_t t_taken = OS::get_singleton()->get_ticks_usec() - call_time;
				_profile_native_call(t_taken, method->get_name(), method->get_instance_class());
				p_function_call_time += t_taken;
			}
#endif

			p_info->ip += OP_SIZE;
			break;
		}

		case CallArguments::METHOD_BIND_RETURN | ArgumentMask::IS_VALIDATED:
		case CallArguments::METHOD_BIND_NO_RETURN | ArgumentMask::IS_VALIDATED: {
			const int
					OP_SIZE = 4,
					ARGS_POS = 2,
					METHOD_POS = 3;
			bool is_return = arg == CallArguments::METHOD_BIND_RETURN | ArgumentMask::IS_VALIDATED;

			LOAD_ARGS
			CHECK_SPACE(OP_SIZE + instr_arg_count);

			p_info->ip += instr_arg_count;

			int argc = _code_ptr[p_info->ip + ARGS_POS];
			GD_ERR_BREAK(argc < 0);

			GD_ERR_BREAK(_code_ptr[p_info->ip + METHOD_POS] < 0 || _code_ptr[p_info->ip + METHOD_POS] >= _methods_count);
			MethodBind *method = _methods_ptr[_code_ptr[p_info->ip + METHOD_POS]];

			GET_INSTRUCTION_ARG(base, argc);
#ifdef DEBUG_ENABLED
			bool freed = false;
			Object *base_obj = base->get_validated_object_with_check(freed);
			if (freed) {
				p_info->err_text = METHOD_CALL_ON_FREED_INSTANCE_ERROR(method);
				return;
			} else if (!base_obj) {
				p_info->err_text = METHOD_CALL_ON_NULL_VALUE_ERROR(method);
				return;
			}
#else
			Object *base_obj = *VariantInternal::get_object(base);
#endif
			Variant **argptrs = p_info->instruction_args;
#ifdef DEBUG_ENABLED
			uint64_t call_time = 0;
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				call_time = OS::get_singleton()->get_ticks_usec();
			}
#endif

			GET_INSTRUCTION_ARG(ret, argc + 1);
			if (is_return) {
				VariantInternal::initialize(ret, Variant::NIL);
			}
			method->validated_call(base_obj, (const Variant **)argptrs, nullptr);

#ifdef DEBUG_ENABLED
			if (GDScriptLanguage::get_singleton()->profiling && GDScriptLanguage::get_singleton()->profile_native_calls) {
				uint64_t t_taken = OS::get_singleton()->get_ticks_usec() - call_time;
				_profile_native_call(t_taken, method->get_name(), method->get_instance_class());
				p_info->function_call_time += t_taken;
			}
#endif

			p_info->ip += OP_SIZE;
			break;
		}

		default: {
			_report_invalid_arg_error(arg, _STR(OPCODE_CALL));
			break;
		}
	}
}

OP_EXEC_IMPLEMENT(OPCODE_AWAIT) {
	const int
			OP_SIZE = 2,
			OBJ_POS = 0;

	CHECK_SPACE(OP_SIZE);

	// Do the one-shot connect.
	GET_VARIANT_PTR(argobj, OBJ_POS);

	Signal sig;
	bool is_signal = true;

	{
		Variant result = *argobj;

		if (argobj->get_type() == Variant::OBJECT) {
			bool was_freed = false;
			Object *obj = argobj->get_validated_object_with_check(was_freed);

			if (was_freed) {
				p_info->err_text = "Trying to await on a freed object.";
				return;
			}

			// Is this even possible to be null at this point?
			if (obj) {
				if (obj->is_class_ptr(GDScriptFunctionState::get_class_ptr_static())) {
					result = Signal(obj, "completed");
				}
			}
		}

		if (result.get_type() != Variant::SIGNAL) {
			// Not async, return immediately using the target from OPCODE_AWAIT_RESUME.
			GET_VARIANT_PTR(target, 2);
			*target = result;
			p_info->ip += 4; // Skip OPCODE_AWAIT_RESUME and its data.
			is_signal = false;
		} else {
			sig = result;
		}
	}

	if (is_signal) {
		Ref<GDScriptFunctionState> gdfs = memnew(GDScriptFunctionState);
		gdfs->function = this;

		gdfs->state.stack.resize(p_info->alloca_size);

		// First 3 stack addresses are special, so we just skip them here.
		for (int i = 3; i < _stack_size; i++) {
			memnew_placement(&gdfs->state.stack.write[sizeof(Variant) * i], Variant(p_info->stack[i]));
		}
		gdfs->state.stack_size = _stack_size;
		gdfs->state.alloca_size = p_info->alloca_size;
		gdfs->state.ip = p_info->ip + 2;
		gdfs->state.line = p_info->line;
		gdfs->state.script = _script;
		{
			MutexLock lock(GDScriptLanguage::get_singleton()->mutex);
			_script->pending_func_states.add(&gdfs->scripts_list);
			if (p_instance) {
				gdfs->state.instance = p_instance;
				p_instance->pending_func_states.add(&gdfs->instances_list);
			} else {
				gdfs->state.instance = nullptr;
			}
		}
#ifdef DEBUG_ENABLED
		gdfs->state.function_name = name;
		gdfs->state.script_path = _script->get_script_path();
#endif
		gdfs->state.defarg = p_info->defarg;
		gdfs->function = this;

		p_info->retvalue = gdfs;

		Error err = sig.connect(Callable(gdfs.ptr(), "_signal_callback").bind(p_info->retvalue), Object::CONNECT_ONE_SHOT);
		if (err != OK) {
			p_info->err_text = "Error connecting to signal: " + sig.get_name() + " during await.";
			return;
		}

#ifdef DEBUG_ENABLED
		p_info->exit_ok = true;
		p_info->awaited = true;
#endif
	}
}

OP_EXEC_IMPLEMENT(OPCODE_AWAIT_RESUME) {
	const int
			OP_SIZE = 3,
			RES_POS = 1;

	CHECK_SPACE(OP_SIZE);
#ifdef DEBUG_ENABLED
	if (!p_info->state) {
		p_info->err_text = ("Invalid Resume (bug?)");
		return;
	}
#endif
	GET_VARIANT_PTR(result, RES_POS);
	*result = p_info->state->result;
	p_info->ip += 2;
}

OP_EXEC_IMPLEMENT(OPCODE_CREATE_LAMBDA) {}

OP_EXEC_IMPLEMENT(OPCODE_CREATE_SELF_LAMBDA) {}

OP_EXEC_IMPLEMENT(OPCODE_JUMP) {}

OP_EXEC_IMPLEMENT(OPCODE_RETURN) {}

OP_EXEC_IMPLEMENT(OPCODE_ITERATE_BEGIN) {}

OP_EXEC_IMPLEMENT(OPCODE_ITERATE) {}

OP_EXEC_IMPLEMENT(OPCODE_STORE_GLOBAL) {}

OP_EXEC_IMPLEMENT(OPCODE_STORE_NAMED_GLOBAL) {}

OP_EXEC_IMPLEMENT(OPCODE_TYPE_ADJUST) {}

OP_EXEC_IMPLEMENT(OPCODE_ASSERT) {}

OP_EXEC_IMPLEMENT(OPCODE_BREAKPOINT) {}

OP_EXEC_IMPLEMENT(OPCODE_LINE) {}

void _report_invalid_arg_error(int arg, String opcode_name) {
	_err_print_error(FUNCTION_STR, __FILE__, __LINE__, "Invalid opcode argument for" + opcode_name + ": ' " + String::num_int64(arg) + " '. This is a bug, please report it!...");
}
