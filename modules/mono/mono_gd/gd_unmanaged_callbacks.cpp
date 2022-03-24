/*************************************************************************/
/*  gd_unmanaged_callbacks.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "gd_unmanaged_callbacks.h"

MethodBind *godotsharp_method_bind_get_method(const StringName *p_classname, const StringName *p_methodname) {
	return ClassDB::get_method(*p_classname, *p_methodname);
}

godotsharp_class_creation_func godotsharp_get_class_constructor(const StringName *p_classname) {
	ClassDB::ClassInfo *class_info = ClassDB::classes.getptr(*p_classname);
	if (class_info) {
		return class_info->creation_func;
	}
	return nullptr;
}

Object *godotsharp_engine_get_singleton(const String *p_name) {
	return Engine::get_singleton()->get_singleton_object(*p_name);
}

Error godotsharp_stack_info_vector_resize(
		Vector<ScriptLanguage::StackInfo> *p_stack_info_vector, int p_size) {
	return p_stack_info_vector->resize(p_size);
}

void godotsharp_stack_info_vector_destroy(
		Vector<ScriptLanguage::StackInfo> *p_stack_info_vector) {
	p_stack_info_vector->~Vector();
}

void godotsharp_internal_script_debugger_send_error(const String *p_func,
		const String *p_file, int32_t p_line, const String *p_err, const String *p_descr,
		bool p_warning, const Vector<ScriptLanguage::StackInfo> *p_stack_info_vector) {
	EngineDebugger::get_script_debugger()->send_error(*p_func, *p_file, p_line, *p_err, *p_descr,
			true, p_warning ? ERR_HANDLER_WARNING : ERR_HANDLER_ERROR, *p_stack_info_vector);
}

bool godotsharp_internal_script_debugger_is_active() {
	return EngineDebugger::is_active();
}

GCHandleIntPtr godotsharp_internal_object_get_associated_gchandle(Object *p_ptr) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_ptr == nullptr);
#endif

	if (p_ptr->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_ptr->get_script_instance());
		if (cs_instance) {
			if (!cs_instance->is_destructing_script_instance()) {
				return cs_instance->get_gchandle_intptr();
			}
			return { nullptr };
		}
	}

	void *data = CSharpLanguage::get_existing_instance_binding(p_ptr);

	if (data) {
		CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get();
		if (script_binding.inited) {
			MonoGCHandleData &gchandle = script_binding.gchandle;
			return !gchandle.is_released() ? gchandle.get_intptr() : GCHandleIntPtr{ nullptr };
		}
	}

	return { nullptr };
}

void godotsharp_internal_object_disposed(Object *p_ptr, GCHandleIntPtr p_gchandle_to_free) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_ptr == nullptr);
#endif

	if (p_ptr->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_ptr->get_script_instance());
		if (cs_instance) {
			if (!cs_instance->is_destructing_script_instance()) {
				cs_instance->mono_object_disposed(p_gchandle_to_free);
				p_ptr->set_script_instance(nullptr);
			}
			return;
		}
	}

	void *data = CSharpLanguage::get_existing_instance_binding(p_ptr);

	if (data) {
		CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get();
		if (script_binding.inited) {
			MonoGCHandleData &gchandle = script_binding.gchandle;
			if (!gchandle.is_released()) {
				CSharpLanguage::release_script_gchandle(p_gchandle_to_free, gchandle);
				script_binding.inited = false;
			}
		}
	}
}

void godotsharp_internal_refcounted_disposed(Object *p_ptr, GCHandleIntPtr p_gchandle_to_free, bool p_is_finalizer) {
#ifdef DEBUG_ENABLED
	CRASH_COND(p_ptr == nullptr);
	// This is only called with RefCounted derived classes
	CRASH_COND(!Object::cast_to<RefCounted>(p_ptr));
#endif

	RefCounted *rc = static_cast<RefCounted *>(p_ptr);

	if (rc->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(rc->get_script_instance());
		if (cs_instance) {
			if (!cs_instance->is_destructing_script_instance()) {
				bool delete_owner;
				bool remove_script_instance;

				cs_instance->mono_object_disposed_baseref(p_gchandle_to_free, p_is_finalizer,
						delete_owner, remove_script_instance);

				if (delete_owner) {
					memdelete(rc);
				} else if (remove_script_instance) {
					rc->set_script_instance(nullptr);
				}
			}
			return;
		}
	}

	// Unsafe refcount decrement. The managed instance also counts as a reference.
	// See: CSharpLanguage::alloc_instance_binding_data(Object *p_object)
	CSharpLanguage::get_singleton()->pre_unsafe_unreference(rc);
	if (rc->unreference()) {
		memdelete(rc);
	} else {
		void *data = CSharpLanguage::get_existing_instance_binding(rc);

		if (data) {
			CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->get();
			if (script_binding.inited) {
				MonoGCHandleData &gchandle = script_binding.gchandle;
				if (!gchandle.is_released()) {
					CSharpLanguage::release_script_gchandle(p_gchandle_to_free, gchandle);
					script_binding.inited = false;
				}
			}
		}
	}
}

void godotsharp_internal_object_connect_event_signal(Object *p_ptr, const StringName *p_event_signal) {
	CSharpInstance *csharp_instance = CAST_CSHARP_INSTANCE(p_ptr->get_script_instance());
	if (csharp_instance) {
		csharp_instance->connect_event_signal(*p_event_signal);
	}
}

Error godotsharp_internal_signal_awaiter_connect(Object *p_source, StringName *p_signal, Object *p_target, GCHandleIntPtr p_awaiter_handle_ptr) {
	StringName signal = p_signal ? *p_signal : StringName();
	return gd_mono_connect_signal_awaiter(p_source, signal, p_target, p_awaiter_handle_ptr);
}

GCHandleIntPtr godotsharp_internal_unmanaged_get_script_instance_managed(Object *p_unmanaged, bool *r_has_cs_script_instance) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
	CRASH_COND(!r_has_cs_script_instance);
#endif

	if (p_unmanaged->get_script_instance()) {
		CSharpInstance *cs_instance = CAST_CSHARP_INSTANCE(p_unmanaged->get_script_instance());

		if (cs_instance) {
			*r_has_cs_script_instance = true;
			return cs_instance->get_gchandle_intptr();
		}
	}

	*r_has_cs_script_instance = false;
	return { nullptr };
}

GCHandleIntPtr godotsharp_internal_unmanaged_get_instance_binding_managed(Object *p_unmanaged) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
#endif

	void *data = CSharpLanguage::get_instance_binding(p_unmanaged);
	ERR_FAIL_NULL_V(data, { nullptr });
	CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->value();
	ERR_FAIL_COND_V(!script_binding.inited, { nullptr });

	return script_binding.gchandle.get_intptr();
}

GCHandleIntPtr godotsharp_internal_unmanaged_instance_binding_create_managed(Object *p_unmanaged, GCHandleIntPtr p_old_gchandle) {
#ifdef DEBUG_ENABLED
	CRASH_COND(!p_unmanaged);
#endif

	void *data = CSharpLanguage::get_instance_binding(p_unmanaged);
	ERR_FAIL_NULL_V(data, { nullptr });
	CSharpScriptBinding &script_binding = ((Map<Object *, CSharpScriptBinding>::Element *)data)->value();
	ERR_FAIL_COND_V(!script_binding.inited, { nullptr });

	MonoGCHandleData &gchandle = script_binding.gchandle;

	// TODO: Possible data race?
	CRASH_COND(gchandle.get_intptr().value != p_old_gchandle.value);

	CSharpLanguage::get_singleton()->release_script_gchandle(gchandle);

	// Create a new one

#ifdef DEBUG_ENABLED
	CRASH_COND(script_binding.type_name == StringName());
#endif

	bool parent_is_object_class = ClassDB::is_parent_class(p_unmanaged->get_class_name(), script_binding.type_name);
	ERR_FAIL_COND_V_MSG(!parent_is_object_class, { nullptr },
			"Type inherits from native type '" + script_binding.type_name + "', so it can't be instantiated in object of type: '" + p_unmanaged->get_class() + "'.");

	GCHandleIntPtr strong_gchandle =
			GDMonoCache::managed_callbacks.ScriptManagerBridge_CreateManagedForGodotObjectBinding(
					&script_binding.type_name, p_unmanaged);

	ERR_FAIL_NULL_V(strong_gchandle.value, { nullptr });

	gchandle = MonoGCHandleData(strong_gchandle, gdmono::GCHandleType::STRONG_HANDLE);

	// Tie managed to unmanaged
	RefCounted *rc = Object::cast_to<RefCounted>(p_unmanaged);

	if (rc) {
		// Unsafe refcount increment. The managed instance also counts as a reference.
		// This way if the unmanaged world has no references to our owner
		// but the managed instance is alive, the refcount will be 1 instead of 0.
		// See: godot_icall_RefCounted_Dtor(MonoObject *p_obj, Object *p_ptr)
		rc->reference();
		CSharpLanguage::get_singleton()->post_unsafe_reference(rc);
	}

	return gchandle.get_intptr();
}

void godotsharp_internal_tie_native_managed_to_unmanaged(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged, const StringName *p_native_name, bool p_ref_counted) {
	CSharpLanguage::tie_native_managed_to_unmanaged(p_gchandle_intptr, p_unmanaged, p_native_name, p_ref_counted);
}

void godotsharp_internal_tie_user_managed_to_unmanaged(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged, Ref<CSharpScript> *p_script, bool p_ref_counted) {
	CSharpLanguage::tie_user_managed_to_unmanaged(p_gchandle_intptr, p_unmanaged, p_script, p_ref_counted);
}

void godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup(GCHandleIntPtr p_gchandle_intptr, Object *p_unmanaged) {
	CSharpLanguage::tie_managed_to_unmanaged_with_pre_setup(p_gchandle_intptr, p_unmanaged);
}

void godotsharp_internal_new_csharp_script(Ref<CSharpScript> *r_dest) {
	memnew_placement(r_dest, Ref<CSharpScript>(memnew(CSharpScript)));
}

void godotsharp_internal_reload_registered_script(CSharpScript *p_script) {
	CRASH_COND(!p_script);
	CSharpScript::reload_registered_script(Ref<CSharpScript>(p_script));
}

void godotsharp_array_filter_godot_objects_by_native(StringName *p_native_name, const Array *p_input, Array *r_output) {
	memnew_placement(r_output, Array);

	for (int i = 0; i < p_input->size(); ++i) {
		if (ClassDB::is_parent_class(((Object *)(*p_input)[i])->get_class(), *p_native_name)) {
			r_output->push_back(p_input[i]);
		}
	}
}

void godotsharp_array_filter_godot_objects_by_non_native(const Array *p_input, Array *r_output) {
	memnew_placement(r_output, Array);

	for (int i = 0; i < p_input->size(); ++i) {
		CSharpInstance *si = CAST_CSHARP_INSTANCE(((Object *)(*p_input)[i])->get_script_instance());

		if (si != nullptr) {
			r_output->push_back(p_input[i]);
		}
	}
}

void godotsharp_ref_new_from_ref_counted_ptr(Ref<RefCounted> *r_dest, RefCounted *p_ref_counted_ptr) {
	memnew_placement(r_dest, Ref<RefCounted>(p_ref_counted_ptr));
}

void godotsharp_ref_destroy(Ref<RefCounted> *p_instance) {
	p_instance->~Ref();
}

void godotsharp_string_name_new_from_string(StringName *r_dest, const String *p_name) {
	memnew_placement(r_dest, StringName(*p_name));
}

void godotsharp_node_path_new_from_string(NodePath *r_dest, const String *p_name) {
	memnew_placement(r_dest, NodePath(*p_name));
}

void godotsharp_string_name_as_string(String *r_dest, const StringName *p_name) {
	memnew_placement(r_dest, String(p_name->operator String()));
}

void godotsharp_node_path_as_string(String *r_dest, const NodePath *p_np) {
	memnew_placement(r_dest, String(p_np->operator String()));
}

godot_packed_byte_array godotsharp_packed_byte_array_new_mem_copy(const uint8_t *p_src, int32_t p_length) {
	godot_packed_byte_array ret;
	memnew_placement(&ret, PackedByteArray);
	PackedByteArray *array = reinterpret_cast<PackedByteArray *>(&ret);
	array->resize(p_length);
	uint8_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(uint8_t));
	return ret;
}

godot_packed_int32_array godotsharp_packed_int32_array_new_mem_copy(const int32_t *p_src, int32_t p_length) {
	godot_packed_int32_array ret;
	memnew_placement(&ret, PackedInt32Array);
	PackedInt32Array *array = reinterpret_cast<PackedInt32Array *>(&ret);
	array->resize(p_length);
	int32_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int32_t));
	return ret;
}

godot_packed_int64_array godotsharp_packed_int64_array_new_mem_copy(const int64_t *p_src, int32_t p_length) {
	godot_packed_int64_array ret;
	memnew_placement(&ret, PackedInt64Array);
	PackedInt64Array *array = reinterpret_cast<PackedInt64Array *>(&ret);
	array->resize(p_length);
	int64_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int64_t));
	return ret;
}

godot_packed_float32_array godotsharp_packed_float32_array_new_mem_copy(const float *p_src, int32_t p_length) {
	godot_packed_float32_array ret;
	memnew_placement(&ret, PackedFloat32Array);
	PackedFloat32Array *array = reinterpret_cast<PackedFloat32Array *>(&ret);
	array->resize(p_length);
	float *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(float));
	return ret;
}

godot_packed_float64_array godotsharp_packed_float64_array_new_mem_copy(const double *p_src, int32_t p_length) {
	godot_packed_float64_array ret;
	memnew_placement(&ret, PackedFloat64Array);
	PackedFloat64Array *array = reinterpret_cast<PackedFloat64Array *>(&ret);
	array->resize(p_length);
	double *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(double));
	return ret;
}

godot_packed_vector2_array godotsharp_packed_vector2_array_new_mem_copy(const Vector2 *p_src, int32_t p_length) {
	godot_packed_vector2_array ret;
	memnew_placement(&ret, PackedVector2Array);
	PackedVector2Array *array = reinterpret_cast<PackedVector2Array *>(&ret);
	array->resize(p_length);
	Vector2 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector2));
	return ret;
}

godot_packed_vector3_array godotsharp_packed_vector3_array_new_mem_copy(const Vector3 *p_src, int32_t p_length) {
	godot_packed_vector3_array ret;
	memnew_placement(&ret, PackedVector3Array);
	PackedVector3Array *array = reinterpret_cast<PackedVector3Array *>(&ret);
	array->resize(p_length);
	Vector3 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector3));
	return ret;
}

godot_packed_color_array godotsharp_packed_color_array_new_mem_copy(const Color *p_src, int32_t p_length) {
	godot_packed_color_array ret;
	memnew_placement(&ret, PackedColorArray);
	PackedColorArray *array = reinterpret_cast<PackedColorArray *>(&ret);
	array->resize(p_length);
	Color *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Color));
	return ret;
}

void godotsharp_packed_string_array_add(PackedStringArray *r_dest, const String *p_element) {
	r_dest->append(*p_element);
}

void godotsharp_callable_new_with_delegate(GCHandleIntPtr p_delegate_handle, Callable *r_callable) {
	// TODO: Use pooling for ManagedCallable instances.
	CallableCustom *managed_callable = memnew(ManagedCallable(p_delegate_handle));
	memnew_placement(r_callable, Callable(managed_callable));
}

bool godotsharp_callable_get_data_for_marshalling(const Callable *p_callable,
		GCHandleIntPtr *r_delegate_handle, Object **r_object, StringName *r_name) {
	if (p_callable->is_custom()) {
		CallableCustom *custom = p_callable->get_custom();
		CallableCustom::CompareEqualFunc compare_equal_func = custom->get_compare_equal_func();

		if (compare_equal_func == ManagedCallable::compare_equal_func_ptr) {
			ManagedCallable *managed_callable = static_cast<ManagedCallable *>(custom);
			*r_delegate_handle = managed_callable->get_delegate();
			*r_object = nullptr;
			memnew_placement(r_name, StringName());
			return true;
		} else if (compare_equal_func == SignalAwaiterCallable::compare_equal_func_ptr) {
			SignalAwaiterCallable *signal_awaiter_callable = static_cast<SignalAwaiterCallable *>(custom);
			*r_delegate_handle = { nullptr };
			*r_object = ObjectDB::get_instance(signal_awaiter_callable->get_object());
			memnew_placement(r_name, StringName(signal_awaiter_callable->get_signal()));
			return true;
		} else if (compare_equal_func == EventSignalCallable::compare_equal_func_ptr) {
			EventSignalCallable *event_signal_callable = static_cast<EventSignalCallable *>(custom);
			*r_delegate_handle = { nullptr };
			*r_object = ObjectDB::get_instance(event_signal_callable->get_object());
			memnew_placement(r_name, StringName(event_signal_callable->get_signal()));
			return true;
		}

		// Some other CallableCustom. We only support ManagedCallable.
		*r_delegate_handle = { nullptr };
		*r_object = nullptr;
		memnew_placement(r_name, StringName());
		return false;
	} else {
		*r_delegate_handle = { nullptr };
		*r_object = ObjectDB::get_instance(p_callable->get_object_id());
		memnew_placement(r_name, StringName(p_callable->get_method()));
		return true;
	}
}

godot_variant godotsharp_callable_call(godot_callable *p_callable, const godot_variant **p_args, const int32_t p_arg_count, godot_variant_call_error *p_call_error) {
	const Variant **args = reinterpret_cast<const Variant **>(p_args);
	Callable::CallError *call_error = reinterpret_cast<Callable::CallError *>(p_call_error);

	godot_variant ret;
	godot_variant_new_nil(&ret);

	Variant *ret_val = (Variant *)&ret;

	reinterpret_cast<Callable *>(p_callable)->call(args, p_arg_count, *ret_val, *call_error);

	return ret;
}

void godotsharp_callable_call_deferred(godot_callable *p_callable, const godot_variant **p_args, const int32_t p_arg_count) {
	reinterpret_cast<Callable *>(p_callable)->call_deferred(reinterpret_cast<const Variant **>(p_args), p_arg_count);
}

// GDNative functions

// gdnative.h

void godotsharp_method_bind_ptrcall(godot_method_bind *p_method_bind, godot_object *p_instance, const void **p_args, void *p_ret) {
	godot_method_bind_ptrcall(p_method_bind, p_instance, p_args, p_ret);
}

godot_variant godotsharp_method_bind_call(godot_method_bind *p_method_bind, godot_object *p_instance, const godot_variant **p_args, const int32_t p_arg_count, godot_variant_call_error *p_call_error) {
	return godot_method_bind_call(p_method_bind, p_instance, p_args, p_arg_count, p_call_error);
}

// variant.h

void godotsharp_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s) {
	godot_variant_new_string_name(r_dest, p_s);
}

void godotsharp_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np) {
	godot_variant_new_node_path(r_dest, p_np);
}

void godotsharp_variant_new_object(godot_variant *r_dest, const godot_object *p_obj) {
	godot_variant_new_object(r_dest, p_obj);
}

void godotsharp_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d) {
	godot_variant_new_transform2d(r_dest, p_t2d);
}

void godotsharp_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis) {
	godot_variant_new_basis(r_dest, p_basis);
}

void godotsharp_variant_new_transform3d(godot_variant *r_dest, const godot_transform3d *p_trans) {
	godot_variant_new_transform3d(r_dest, p_trans);
}

void godotsharp_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb) {
	godot_variant_new_aabb(r_dest, p_aabb);
}

void godotsharp_variant_new_dictionary(godot_variant *r_dest, const godot_dictionary *p_dict) {
	godot_variant_new_dictionary(r_dest, p_dict);
}

void godotsharp_variant_new_array(godot_variant *r_dest, const godot_array *p_arr) {
	godot_variant_new_array(r_dest, p_arr);
}

void godotsharp_variant_new_packed_byte_array(godot_variant *r_dest, const godot_packed_byte_array *p_pba) {
	godot_variant_new_packed_byte_array(r_dest, p_pba);
}

void godotsharp_variant_new_packed_int32_array(godot_variant *r_dest, const godot_packed_int32_array *p_pia) {
	godot_variant_new_packed_int32_array(r_dest, p_pia);
}

void godotsharp_variant_new_packed_int64_array(godot_variant *r_dest, const godot_packed_int64_array *p_pia) {
	godot_variant_new_packed_int64_array(r_dest, p_pia);
}

void godotsharp_variant_new_packed_float32_array(godot_variant *r_dest, const godot_packed_float32_array *p_pra) {
	godot_variant_new_packed_float32_array(r_dest, p_pra);
}

void godotsharp_variant_new_packed_float64_array(godot_variant *r_dest, const godot_packed_float64_array *p_pra) {
	godot_variant_new_packed_float64_array(r_dest, p_pra);
}

void godotsharp_variant_new_packed_string_array(godot_variant *r_dest, const godot_packed_string_array *p_psa) {
	godot_variant_new_packed_string_array(r_dest, p_psa);
}

void godotsharp_variant_new_packed_vector2_array(godot_variant *r_dest, const godot_packed_vector2_array *p_pv2a) {
	godot_variant_new_packed_vector2_array(r_dest, p_pv2a);
}

void godotsharp_variant_new_packed_vector3_array(godot_variant *r_dest, const godot_packed_vector3_array *p_pv3a) {
	godot_variant_new_packed_vector3_array(r_dest, p_pv3a);
}

void godotsharp_variant_new_packed_color_array(godot_variant *r_dest, const godot_packed_color_array *p_pca) {
	godot_variant_new_packed_color_array(r_dest, p_pca);
}

godot_bool godotsharp_variant_as_bool(const godot_variant *p_self) {
	return godot_variant_as_bool(p_self);
}

godot_int godotsharp_variant_as_int(const godot_variant *p_self) {
	return godot_variant_as_int(p_self);
}

godot_float godotsharp_variant_as_float(const godot_variant *p_self) {
	return godot_variant_as_float(p_self);
}

godot_string godotsharp_variant_as_string(const godot_variant *p_self) {
	return godot_variant_as_string(p_self);
}

godot_vector2 godotsharp_variant_as_vector2(const godot_variant *p_self) {
	return godot_variant_as_vector2(p_self);
}

godot_vector2i godotsharp_variant_as_vector2i(const godot_variant *p_self) {
	return godot_variant_as_vector2i(p_self);
}

godot_rect2 godotsharp_variant_as_rect2(const godot_variant *p_self) {
	return godot_variant_as_rect2(p_self);
}

godot_rect2i godotsharp_variant_as_rect2i(const godot_variant *p_self) {
	return godot_variant_as_rect2i(p_self);
}

godot_vector3 godotsharp_variant_as_vector3(const godot_variant *p_self) {
	return godot_variant_as_vector3(p_self);
}

godot_vector3i godotsharp_variant_as_vector3i(const godot_variant *p_self) {
	return godot_variant_as_vector3i(p_self);
}

godot_transform2d godotsharp_variant_as_transform2d(const godot_variant *p_self) {
	return godot_variant_as_transform2d(p_self);
}

godot_plane godotsharp_variant_as_plane(const godot_variant *p_self) {
	return godot_variant_as_plane(p_self);
}

godot_quaternion godotsharp_variant_as_quaternion(const godot_variant *p_self) {
	return godot_variant_as_quaternion(p_self);
}

godot_aabb godotsharp_variant_as_aabb(const godot_variant *p_self) {
	return godot_variant_as_aabb(p_self);
}

godot_basis godotsharp_variant_as_basis(const godot_variant *p_self) {
	return godot_variant_as_basis(p_self);
}

godot_transform3d godotsharp_variant_as_transform3d(const godot_variant *p_self) {
	return godot_variant_as_transform3d(p_self);
}

godot_color godotsharp_variant_as_color(const godot_variant *p_self) {
	return godot_variant_as_color(p_self);
}

godot_string_name godotsharp_variant_as_string_name(const godot_variant *p_self) {
	return godot_variant_as_string_name(p_self);
}

godot_node_path godotsharp_variant_as_node_path(const godot_variant *p_self) {
	return godot_variant_as_node_path(p_self);
}

godot_rid godotsharp_variant_as_rid(const godot_variant *p_self) {
	return godot_variant_as_rid(p_self);
}

godot_callable godotsharp_variant_as_callable(const godot_variant *p_self) {
	return godot_variant_as_callable(p_self);
}

godot_signal godotsharp_variant_as_signal(const godot_variant *p_self) {
	return godot_variant_as_signal(p_self);
}

godot_dictionary godotsharp_variant_as_dictionary(const godot_variant *p_self) {
	return godot_variant_as_dictionary(p_self);
}

godot_array godotsharp_variant_as_array(const godot_variant *p_self) {
	return godot_variant_as_array(p_self);
}

godot_packed_byte_array godotsharp_variant_as_packed_byte_array(const godot_variant *p_self) {
	return godot_variant_as_packed_byte_array(p_self);
}

godot_packed_int32_array godotsharp_variant_as_packed_int32_array(const godot_variant *p_self) {
	return godot_variant_as_packed_int32_array(p_self);
}

godot_packed_int64_array godotsharp_variant_as_packed_int64_array(const godot_variant *p_self) {
	return godot_variant_as_packed_int64_array(p_self);
}

godot_packed_float32_array godotsharp_variant_as_packed_float32_array(const godot_variant *p_self) {
	return godot_variant_as_packed_float32_array(p_self);
}

godot_packed_float64_array godotsharp_variant_as_packed_float64_array(const godot_variant *p_self) {
	return godot_variant_as_packed_float64_array(p_self);
}

godot_packed_string_array godotsharp_variant_as_packed_string_array(const godot_variant *p_self) {
	return godot_variant_as_packed_string_array(p_self);
}

godot_packed_vector2_array godotsharp_variant_as_packed_vector2_array(const godot_variant *p_self) {
	return godot_variant_as_packed_vector2_array(p_self);
}

godot_packed_vector3_array godotsharp_variant_as_packed_vector3_array(const godot_variant *p_self) {
	return godot_variant_as_packed_vector3_array(p_self);
}

godot_packed_color_array godotsharp_variant_as_packed_color_array(const godot_variant *p_self) {
	return godot_variant_as_packed_color_array(p_self);
}

bool godotsharp_variant_equals(const godot_variant *p_a, const godot_variant *p_b) {
	return *reinterpret_cast<const Variant *>(p_a) == *reinterpret_cast<const Variant *>(p_b);
}

// string.h

void godotsharp_string_new_with_utf16_chars(godot_string *r_dest, const char16_t *p_contents) {
	godot_string_new_with_utf16_chars(r_dest, p_contents);
}

// string_name.h

void godotsharp_string_name_new_copy(godot_string_name *r_dest, const godot_string_name *p_src) {
	godot_string_name_new_copy(r_dest, p_src);
}

// node_path.h

void godotsharp_node_path_new_copy(godot_node_path *r_dest, const godot_node_path *p_src) {
	godot_node_path_new_copy(r_dest, p_src);
}

// array.h

void godotsharp_array_new(godot_array *r_dest) {
	godot_array_new(r_dest);
}

void godotsharp_array_new_copy(godot_array *r_dest, const godot_array *p_src) {
	godot_array_new_copy(r_dest, p_src);
}

godot_variant *godotsharp_array_ptrw(godot_array *p_self) {
	return reinterpret_cast<godot_variant *>(&reinterpret_cast<Array *>(p_self)->operator[](0));
}

// dictionary.h

void godotsharp_dictionary_new(godot_dictionary *r_dest) {
	godot_dictionary_new(r_dest);
}

void godotsharp_dictionary_new_copy(godot_dictionary *r_dest, const godot_dictionary *p_src) {
	godot_dictionary_new_copy(r_dest, p_src);
}

// destroy functions

void godotsharp_packed_byte_array_destroy(godot_packed_byte_array *p_self) {
	godot_packed_byte_array_destroy(p_self);
}

void godotsharp_packed_int32_array_destroy(godot_packed_int32_array *p_self) {
	godot_packed_int32_array_destroy(p_self);
}

void godotsharp_packed_int64_array_destroy(godot_packed_int64_array *p_self) {
	godot_packed_int64_array_destroy(p_self);
}

void godotsharp_packed_float32_array_destroy(godot_packed_float32_array *p_self) {
	godot_packed_float32_array_destroy(p_self);
}

void godotsharp_packed_float64_array_destroy(godot_packed_float64_array *p_self) {
	godot_packed_float64_array_destroy(p_self);
}

void godotsharp_packed_string_array_destroy(godot_packed_string_array *p_self) {
	godot_packed_string_array_destroy(p_self);
}

void godotsharp_packed_vector2_array_destroy(godot_packed_vector2_array *p_self) {
	godot_packed_vector2_array_destroy(p_self);
}

void godotsharp_packed_vector3_array_destroy(godot_packed_vector3_array *p_self) {
	godot_packed_vector3_array_destroy(p_self);
}

void godotsharp_packed_color_array_destroy(godot_packed_color_array *p_self) {
	godot_packed_color_array_destroy(p_self);
}

void godotsharp_variant_destroy(godot_variant *p_self) {
	godot_variant_destroy(p_self);
}

void godotsharp_string_destroy(godot_string *p_self) {
	godot_string_destroy(p_self);
}

void godotsharp_string_name_destroy(godot_string_name *p_self) {
	godot_string_name_destroy(p_self);
}

void godotsharp_node_path_destroy(godot_node_path *p_self) {
	godot_node_path_destroy(p_self);
}

void godotsharp_signal_destroy(godot_signal *p_self) {
	godot_signal_destroy(p_self);
}

void godotsharp_callable_destroy(godot_callable *p_self) {
	godot_callable_destroy(p_self);
}

void godotsharp_array_destroy(godot_array *p_self) {
	godot_array_destroy(p_self);
}

void godotsharp_dictionary_destroy(godot_dictionary *p_self) {
	godot_dictionary_destroy(p_self);
}

// Array

int32_t godotsharp_array_add(Array *p_self, const Variant *p_item) {
	p_self->append(*p_item);
	return p_self->size();
}

void godotsharp_array_duplicate(const Array *p_self, bool p_deep, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->duplicate(p_deep)));
}

int32_t godotsharp_array_index_of(const Array *p_self, const Variant *p_item) {
	return p_self->find(*p_item);
}

void godotsharp_array_insert(Array *p_self, int32_t p_index, const Variant *p_item) {
	p_self->insert(p_index, *p_item);
}

void godotsharp_array_remove_at(Array *p_self, int32_t p_index) {
	p_self->remove_at(p_index);
}

Error godotsharp_array_resize(Array *p_self, int32_t p_new_size) {
	return p_self->resize(p_new_size);
}

void godotsharp_array_shuffle(Array *p_self) {
	p_self->shuffle();
}

void godotsharp_array_to_string(const Array *p_self, String *r_str) {
	*r_str = Variant(*p_self).operator String();
}

// Dictionary

bool godotsharp_dictionary_try_get_value(const Dictionary *p_self, const Variant *p_key, Variant *r_value) {
	const Variant *ret = p_self->getptr(*p_key);
	if (ret == nullptr) {
		memnew_placement(r_value, Variant());
		return false;
	}
	memnew_placement(r_value, Variant(*ret));
	return true;
}

void godotsharp_dictionary_set_value(Dictionary *p_self, const Variant *p_key, const Variant *p_value) {
	p_self->operator[](*p_key) = *p_value;
}

void godotsharp_dictionary_keys(const Dictionary *p_self, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->keys()));
}

void godotsharp_dictionary_values(const Dictionary *p_self, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->values()));
}

int32_t godotsharp_dictionary_count(const Dictionary *p_self) {
	return p_self->size();
}

void godotsharp_dictionary_key_value_pair_at(const Dictionary *p_self, int32_t p_index, Variant *r_key, Variant *r_value) {
	memnew_placement(r_key, Variant(p_self->get_key_at_index(p_index)));
	memnew_placement(r_value, Variant(p_self->get_value_at_index(p_index)));
}

void godotsharp_dictionary_to_string(const Dictionary *p_self, String *r_str) {
	*r_str = Variant(*p_self).operator String();
}

void godotsharp_dictionary_add(Dictionary *p_self, const Variant *p_key, const Variant *p_value) {
	p_self->operator[](*p_key) = *p_value;
}

void godotsharp_dictionary_clear(Dictionary *p_self) {
	p_self->clear();
}

bool godotsharp_dictionary_contains_key(const Dictionary *p_self, const Variant *p_key) {
	return p_self->has(*p_key);
}

void godotsharp_dictionary_duplicate(const Dictionary *p_self, bool p_deep, Dictionary *r_dest) {
	memnew_placement(r_dest, Dictionary(p_self->duplicate(p_deep)));
}

bool godotsharp_dictionary_remove_key(Dictionary *p_self, const Variant *p_key) {
	return p_self->erase(*p_key);
}

void godotsharp_string_md5_buffer(const String *p_self, PackedByteArray *r_md5_buffer) {
	memnew_placement(r_md5_buffer, PackedByteArray(p_self->md5_buffer()));
}

void godotsharp_string_md5_text(const String *p_self, String *r_md5_text) {
	memnew_placement(r_md5_text, String(p_self->md5_text()));
}

int32_t godotsharp_string_rfind(const String *p_self, const String *p_what, int32_t p_from) {
	return p_self->rfind(*p_what, p_from);
}

int32_t godotsharp_string_rfindn(const String *p_self, const String *p_what, int32_t p_from) {
	return p_self->rfindn(*p_what, p_from);
}

void godotsharp_string_sha256_buffer(const String *p_self, PackedByteArray *r_sha256_buffer) {
	memnew_placement(r_sha256_buffer, PackedByteArray(p_self->sha256_buffer()));
}

void godotsharp_string_sha256_text(const String *p_self, String *r_sha256_text) {
	memnew_placement(r_sha256_text, String(p_self->sha256_text()));
}

void godotsharp_string_simplify_path(const String *p_self, String *r_simplified_path) {
	memnew_placement(r_simplified_path, String(p_self->simplify_path()));
}

void godotsharp_node_path_get_as_property_path(const NodePath *p_ptr, NodePath *r_dest) {
	memnew_placement(r_dest, NodePath(p_ptr->get_as_property_path()));
}

void godotsharp_node_path_get_concatenated_subnames(const NodePath *p_self, String *r_subnames) {
	memnew_placement(r_subnames, String(p_self->get_concatenated_subnames()));
}

void godotsharp_node_path_get_name(const NodePath *p_self, uint32_t p_idx, String *r_name) {
	memnew_placement(r_name, String(p_self->get_name(p_idx)));
}

int32_t godotsharp_node_path_get_name_count(const NodePath *p_self) {
	return p_self->get_name_count();
}

void godotsharp_node_path_get_subname(const NodePath *p_self, uint32_t p_idx, String *r_subname) {
	memnew_placement(r_subname, String(p_self->get_subname(p_idx)));
}

int32_t godotsharp_node_path_get_subname_count(const NodePath *p_self) {
	return p_self->get_subname_count();
}

bool godotsharp_node_path_is_absolute(const NodePath *p_self) {
	return p_self->is_absolute();
}

void godotsharp_randomize() {
	Math::randomize();
}

uint32_t godotsharp_randi() {
	return Math::rand();
}

float godotsharp_randf() {
	return Math::randf();
}

int32_t godotsharp_randi_range(int32_t p_from, int32_t p_to) {
	return Math::random(p_from, p_to);
}

double godotsharp_randf_range(double p_from, double p_to) {
	return Math::random(p_from, p_to);
}

double godotsharp_randfn(double p_mean, double p_deviation) {
	return Math::randfn(p_mean, p_deviation);
}

void godotsharp_seed(uint64_t p_seed) {
	Math::seed(p_seed);
}

uint32_t godotsharp_rand_from_seed(uint64_t p_seed, uint64_t *r_new_seed) {
	uint32_t ret = Math::rand_from_seed(&p_seed);
	*r_new_seed = p_seed;
	return ret;
}

void godotsharp_weakref(Object *p_ptr, Ref<RefCounted> *r_weak_ref) {
	if (!p_ptr) {
		return;
	}

	Ref<WeakRef> wref;
	RefCounted *rc = Object::cast_to<RefCounted>(p_ptr);

	if (rc) {
		REF r = rc;
		if (!r.is_valid()) {
			return;
		}

		wref.instantiate();
		wref->set_ref(r);
	} else {
		wref.instantiate();
		wref->set_obj(p_ptr);
	}

	memnew_placement(r_weak_ref, Ref<RefCounted>(wref));
}

void godotsharp_str(const godot_array *p_what, godot_string *r_ret) {
	String &str = *memnew_placement(r_ret, String);
	const Array &what = *reinterpret_cast<const Array *>(p_what);

	for (int i = 0; i < what.size(); i++) {
		String os = what[i].operator String();

		if (i == 0) {
			str = os;
		} else {
			str += os;
		}
	}
}

void godotsharp_print(const godot_string *p_what) {
	print_line(*reinterpret_cast<const String *>(p_what));
}

void godotsharp_printerr(const godot_string *p_what) {
	print_error(*reinterpret_cast<const String *>(p_what));
}

void godotsharp_printt(const godot_string *p_what) {
	print_line(*reinterpret_cast<const String *>(p_what));
}

void godotsharp_prints(const godot_string *p_what) {
	print_line(*reinterpret_cast<const String *>(p_what));
}

void godotsharp_printraw(const godot_string *p_what) {
	OS::get_singleton()->print("%s", reinterpret_cast<const String *>(p_what)->utf8().get_data());
}

void godotsharp_pusherror(const godot_string *p_str) {
	ERR_PRINT(*reinterpret_cast<const String *>(p_str));
}

void godotsharp_pushwarning(const godot_string *p_str) {
	WARN_PRINT(*reinterpret_cast<const String *>(p_str));
}

void godotsharp_var2str(const godot_variant *p_var, godot_string *r_ret) {
	const Variant &var = *reinterpret_cast<const Variant *>(p_var);
	String &vars = *memnew_placement(r_ret, String);
	VariantWriter::write_to_string(var, vars);
}

void godotsharp_str2var(const godot_string *p_str, godot_variant *r_ret) {
	Variant ret;

	VariantParser::StreamString ss;
	ss.s = *reinterpret_cast<const String *>(p_str);

	String errs;
	int line;
	Error err = VariantParser::parse(&ss, ret, errs, line);
	if (err != OK) {
		String err_str = "Parse error at line " + itos(line) + ": " + errs + ".";
		ERR_PRINT(err_str);
		ret = err_str;
	}
	memnew_placement(r_ret, Variant(ret));
}

void godotsharp_var2bytes(const godot_variant *p_var, bool p_full_objects, godot_packed_byte_array *r_bytes) {
	const Variant &var = *reinterpret_cast<const Variant *>(p_var);
	PackedByteArray &bytes = *memnew_placement(r_bytes, PackedByteArray);

	int len;
	Error err = encode_variant(var, nullptr, len, p_full_objects);
	ERR_FAIL_COND_MSG(err != OK, "Unexpected error encoding variable to bytes, likely unserializable type found (Object or RID).");

	bytes.resize(len);
	encode_variant(var, bytes.ptrw(), len, p_full_objects);
}

void godotsharp_bytes2var(const godot_packed_byte_array *p_bytes, bool p_allow_objects, godot_variant *r_ret) {
	const PackedByteArray *bytes = reinterpret_cast<const PackedByteArray *>(p_bytes);
	Variant ret;
	Error err = decode_variant(ret, bytes->ptr(), bytes->size(), nullptr, p_allow_objects);
	if (err != OK) {
		ret = RTR("Not enough bytes for decoding bytes, or invalid format.");
	}
	memnew_placement(r_ret, Variant(ret));
}

int godotsharp_hash(const godot_variant *p_var) {
	return reinterpret_cast<const Variant *>(p_var)->hash();
}

void godotsharp_convert(const godot_variant *p_what, int32_t p_type, godot_variant *r_ret) {
	const Variant *args[1] = { reinterpret_cast<const Variant *>(p_what) };
	Callable::CallError ce;
	Variant ret;
	Variant::construct(Variant::Type(p_type), ret, args, 1, ce);
	if (ce.error != Callable::CallError::CALL_OK) {
		memnew_placement(r_ret, Variant);
		ERR_FAIL_MSG("Unable to convert parameter from '" +
				Variant::get_type_name(reinterpret_cast<const Variant *>(p_what)->get_type()) +
				"' to '" + Variant::get_type_name(Variant::Type(p_type)) + "'.");
	}
	memnew_placement(r_ret, Variant(ret));
}

Object *godotsharp_instance_from_id(uint64_t p_instance_id) {
	return ObjectDB::get_instance(ObjectID(p_instance_id));
}

void godotsharp_object_to_string(Object *p_ptr, godot_string *r_str) {
#ifdef DEBUG_ENABLED
	// Cannot happen in C#; would get an ObjectDisposedException instead.
	CRASH_COND(p_ptr == nullptr);
#endif
	// Can't call 'Object::to_string()' here, as that can end up calling 'ToString' again resulting in an endless circular loop.
	memnew_placement(r_str,
			String("[" + p_ptr->get_class() + ":" + itos(p_ptr->get_instance_id()) + "]"));
}

UnmanagedCallbacks UnmanagedCallbacks::create() {
	UnmanagedCallbacks unmanaged_callbacks;

#define REGISTER_CALLBACK(m_method) unmanaged_callbacks.m_method = ::m_method

	REGISTER_CALLBACK(godotsharp_method_bind_get_method);
	REGISTER_CALLBACK(godotsharp_get_class_constructor);
	REGISTER_CALLBACK(godotsharp_engine_get_singleton);
	REGISTER_CALLBACK(godotsharp_stack_info_vector_resize);
	REGISTER_CALLBACK(godotsharp_stack_info_vector_destroy);
	REGISTER_CALLBACK(godotsharp_internal_script_debugger_send_error);
	REGISTER_CALLBACK(godotsharp_internal_script_debugger_is_active);
	REGISTER_CALLBACK(godotsharp_internal_object_get_associated_gchandle);
	REGISTER_CALLBACK(godotsharp_internal_object_disposed);
	REGISTER_CALLBACK(godotsharp_internal_refcounted_disposed);
	REGISTER_CALLBACK(godotsharp_internal_object_connect_event_signal);
	REGISTER_CALLBACK(godotsharp_internal_signal_awaiter_connect);
	REGISTER_CALLBACK(godotsharp_internal_unmanaged_get_script_instance_managed);
	REGISTER_CALLBACK(godotsharp_internal_unmanaged_get_instance_binding_managed);
	REGISTER_CALLBACK(godotsharp_internal_unmanaged_instance_binding_create_managed);
	REGISTER_CALLBACK(godotsharp_internal_tie_native_managed_to_unmanaged);
	REGISTER_CALLBACK(godotsharp_internal_tie_user_managed_to_unmanaged);
	REGISTER_CALLBACK(godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup);
	REGISTER_CALLBACK(godotsharp_internal_new_csharp_script);
	REGISTER_CALLBACK(godotsharp_internal_reload_registered_script);
	REGISTER_CALLBACK(godotsharp_array_filter_godot_objects_by_native);
	REGISTER_CALLBACK(godotsharp_array_filter_godot_objects_by_non_native);
	REGISTER_CALLBACK(godotsharp_ref_new_from_ref_counted_ptr);
	REGISTER_CALLBACK(godotsharp_ref_destroy);
	REGISTER_CALLBACK(godotsharp_string_name_new_from_string);
	REGISTER_CALLBACK(godotsharp_node_path_new_from_string);
	REGISTER_CALLBACK(godotsharp_string_name_as_string);
	REGISTER_CALLBACK(godotsharp_node_path_as_string);
	REGISTER_CALLBACK(godotsharp_packed_byte_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_int32_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_int64_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_float32_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_float64_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_vector2_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_vector3_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_color_array_new_mem_copy);
	REGISTER_CALLBACK(godotsharp_packed_string_array_add);
	REGISTER_CALLBACK(godotsharp_callable_new_with_delegate);
	REGISTER_CALLBACK(godotsharp_callable_get_data_for_marshalling);
	REGISTER_CALLBACK(godotsharp_callable_call);
	REGISTER_CALLBACK(godotsharp_callable_call_deferred);
	REGISTER_CALLBACK(godotsharp_method_bind_ptrcall);
	REGISTER_CALLBACK(godotsharp_method_bind_call);
	REGISTER_CALLBACK(godotsharp_variant_new_string_name);
	REGISTER_CALLBACK(godotsharp_variant_new_node_path);
	REGISTER_CALLBACK(godotsharp_variant_new_object);
	REGISTER_CALLBACK(godotsharp_variant_new_transform2d);
	REGISTER_CALLBACK(godotsharp_variant_new_basis);
	REGISTER_CALLBACK(godotsharp_variant_new_transform3d);
	REGISTER_CALLBACK(godotsharp_variant_new_aabb);
	REGISTER_CALLBACK(godotsharp_variant_new_dictionary);
	REGISTER_CALLBACK(godotsharp_variant_new_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_byte_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_int32_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_int64_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_float32_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_float64_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_string_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_vector2_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_vector3_array);
	REGISTER_CALLBACK(godotsharp_variant_new_packed_color_array);
	REGISTER_CALLBACK(godotsharp_variant_as_bool);
	REGISTER_CALLBACK(godotsharp_variant_as_int);
	REGISTER_CALLBACK(godotsharp_variant_as_float);
	REGISTER_CALLBACK(godotsharp_variant_as_string);
	REGISTER_CALLBACK(godotsharp_variant_as_vector2);
	REGISTER_CALLBACK(godotsharp_variant_as_vector2i);
	REGISTER_CALLBACK(godotsharp_variant_as_rect2);
	REGISTER_CALLBACK(godotsharp_variant_as_rect2i);
	REGISTER_CALLBACK(godotsharp_variant_as_vector3);
	REGISTER_CALLBACK(godotsharp_variant_as_vector3i);
	REGISTER_CALLBACK(godotsharp_variant_as_transform2d);
	REGISTER_CALLBACK(godotsharp_variant_as_plane);
	REGISTER_CALLBACK(godotsharp_variant_as_quaternion);
	REGISTER_CALLBACK(godotsharp_variant_as_aabb);
	REGISTER_CALLBACK(godotsharp_variant_as_basis);
	REGISTER_CALLBACK(godotsharp_variant_as_transform3d);
	REGISTER_CALLBACK(godotsharp_variant_as_color);
	REGISTER_CALLBACK(godotsharp_variant_as_string_name);
	REGISTER_CALLBACK(godotsharp_variant_as_node_path);
	REGISTER_CALLBACK(godotsharp_variant_as_rid);
	REGISTER_CALLBACK(godotsharp_variant_as_callable);
	REGISTER_CALLBACK(godotsharp_variant_as_signal);
	REGISTER_CALLBACK(godotsharp_variant_as_dictionary);
	REGISTER_CALLBACK(godotsharp_variant_as_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_byte_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_int32_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_int64_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_float32_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_float64_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_string_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_vector2_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_vector3_array);
	REGISTER_CALLBACK(godotsharp_variant_as_packed_color_array);
	REGISTER_CALLBACK(godotsharp_variant_equals);
	REGISTER_CALLBACK(godotsharp_string_new_with_utf16_chars);
	REGISTER_CALLBACK(godotsharp_string_name_new_copy);
	REGISTER_CALLBACK(godotsharp_node_path_new_copy);
	REGISTER_CALLBACK(godotsharp_array_new);
	REGISTER_CALLBACK(godotsharp_array_new_copy);
	REGISTER_CALLBACK(godotsharp_array_ptrw);
	REGISTER_CALLBACK(godotsharp_dictionary_new);
	REGISTER_CALLBACK(godotsharp_dictionary_new_copy);
	REGISTER_CALLBACK(godotsharp_packed_byte_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_int32_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_int64_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_float32_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_float64_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_string_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_vector2_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_vector3_array_destroy);
	REGISTER_CALLBACK(godotsharp_packed_color_array_destroy);
	REGISTER_CALLBACK(godotsharp_variant_destroy);
	REGISTER_CALLBACK(godotsharp_string_destroy);
	REGISTER_CALLBACK(godotsharp_string_name_destroy);
	REGISTER_CALLBACK(godotsharp_node_path_destroy);
	REGISTER_CALLBACK(godotsharp_signal_destroy);
	REGISTER_CALLBACK(godotsharp_callable_destroy);
	REGISTER_CALLBACK(godotsharp_array_destroy);
	REGISTER_CALLBACK(godotsharp_dictionary_destroy);
	REGISTER_CALLBACK(godotsharp_array_add);
	REGISTER_CALLBACK(godotsharp_array_duplicate);
	REGISTER_CALLBACK(godotsharp_array_index_of);
	REGISTER_CALLBACK(godotsharp_array_insert);
	REGISTER_CALLBACK(godotsharp_array_remove_at);
	REGISTER_CALLBACK(godotsharp_array_resize);
	REGISTER_CALLBACK(godotsharp_array_shuffle);
	REGISTER_CALLBACK(godotsharp_array_to_string);
	REGISTER_CALLBACK(godotsharp_dictionary_try_get_value);
	REGISTER_CALLBACK(godotsharp_dictionary_set_value);
	REGISTER_CALLBACK(godotsharp_dictionary_keys);
	REGISTER_CALLBACK(godotsharp_dictionary_values);
	REGISTER_CALLBACK(godotsharp_dictionary_count);
	REGISTER_CALLBACK(godotsharp_dictionary_key_value_pair_at);
	REGISTER_CALLBACK(godotsharp_dictionary_to_string);
	REGISTER_CALLBACK(godotsharp_dictionary_add);
	REGISTER_CALLBACK(godotsharp_dictionary_clear);
	REGISTER_CALLBACK(godotsharp_dictionary_contains_key);
	REGISTER_CALLBACK(godotsharp_dictionary_duplicate);
	REGISTER_CALLBACK(godotsharp_dictionary_remove_key);
	REGISTER_CALLBACK(godotsharp_string_md5_buffer);
	REGISTER_CALLBACK(godotsharp_string_md5_text);
	REGISTER_CALLBACK(godotsharp_string_rfind);
	REGISTER_CALLBACK(godotsharp_string_rfindn);
	REGISTER_CALLBACK(godotsharp_string_sha256_buffer);
	REGISTER_CALLBACK(godotsharp_string_sha256_text);
	REGISTER_CALLBACK(godotsharp_string_simplify_path);
	REGISTER_CALLBACK(godotsharp_node_path_get_as_property_path);
	REGISTER_CALLBACK(godotsharp_node_path_get_concatenated_subnames);
	REGISTER_CALLBACK(godotsharp_node_path_get_name);
	REGISTER_CALLBACK(godotsharp_node_path_get_name_count);
	REGISTER_CALLBACK(godotsharp_node_path_get_subname);
	REGISTER_CALLBACK(godotsharp_node_path_get_subname_count);
	REGISTER_CALLBACK(godotsharp_node_path_is_absolute);
	REGISTER_CALLBACK(godotsharp_randomize);
	REGISTER_CALLBACK(godotsharp_randi);
	REGISTER_CALLBACK(godotsharp_randf);
	REGISTER_CALLBACK(godotsharp_randi_range);
	REGISTER_CALLBACK(godotsharp_randf_range);
	REGISTER_CALLBACK(godotsharp_randfn);
	REGISTER_CALLBACK(godotsharp_seed);
	REGISTER_CALLBACK(godotsharp_rand_from_seed);
	REGISTER_CALLBACK(godotsharp_weakref);
	REGISTER_CALLBACK(godotsharp_str);
	REGISTER_CALLBACK(godotsharp_print);
	REGISTER_CALLBACK(godotsharp_printerr);
	REGISTER_CALLBACK(godotsharp_printt);
	REGISTER_CALLBACK(godotsharp_prints);
	REGISTER_CALLBACK(godotsharp_printraw);
	REGISTER_CALLBACK(godotsharp_pusherror);
	REGISTER_CALLBACK(godotsharp_pushwarning);
	REGISTER_CALLBACK(godotsharp_var2str);
	REGISTER_CALLBACK(godotsharp_str2var);
	REGISTER_CALLBACK(godotsharp_var2bytes);
	REGISTER_CALLBACK(godotsharp_bytes2var);
	REGISTER_CALLBACK(godotsharp_hash);
	REGISTER_CALLBACK(godotsharp_convert);
	REGISTER_CALLBACK(godotsharp_instance_from_id);
	REGISTER_CALLBACK(godotsharp_object_to_string);

#undef REGISTER_CALLBACK

	return unmanaged_callbacks;
}
