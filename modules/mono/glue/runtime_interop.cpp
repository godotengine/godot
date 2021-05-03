/*************************************************************************/
/*  runtime_interop.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/config/engine.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/string/string_name.h"

#include <gdnative/gdnative.h>

#include "modules/mono/managed_callable.h"
#include "modules/mono/signal_awaiter_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define MAYBE_UNUSED
#endif

#ifdef __GNUC__
#define GD_PINVOKE_EXPORT MAYBE_UNUSED __attribute__((visibility("default")))
#elif defined(_WIN32)
#define GD_PINVOKE_EXPORT MAYBE_UNUSED __declspec(dllexport)
#else
#define GD_PINVOKE_EXPORT MAYBE_UNUSED
#endif

typedef Object *(*godotsharp_class_creation_func)();

GD_PINVOKE_EXPORT MethodBind *godotsharp_method_bind_get_method(const StringName *p_classname, const char16_t *p_methodname) {
	return ClassDB::get_method(*p_classname, StringName(String::utf16(p_methodname)));
}

GD_PINVOKE_EXPORT godotsharp_class_creation_func godotsharp_get_class_constructor(const StringName *p_classname) {
	ClassDB::ClassInfo *class_info = ClassDB::classes.getptr(*p_classname);
	if (class_info) {
		return class_info->creation_func;
	}
	return nullptr;
}

GD_PINVOKE_EXPORT Object *godotsharp_invoke_class_constructor(godotsharp_class_creation_func p_creation_func) {
	return p_creation_func();
}

GD_PINVOKE_EXPORT Object *godotsharp_engine_get_singleton(const String *p_name) {
	return Engine::get_singleton()->get_singleton_object(*p_name);
}

GD_PINVOKE_EXPORT void godotsharp_ref_destroy(Ref<RefCounted> *p_instance) {
	p_instance->~Ref();
}

GD_PINVOKE_EXPORT void godotsharp_string_name_new_from_string(StringName *r_dest, const String *p_name) {
	memnew_placement(r_dest, StringName(*p_name));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_new_from_string(NodePath *r_dest, const String *p_name) {
	memnew_placement(r_dest, NodePath(*p_name));
}

GD_PINVOKE_EXPORT void godotsharp_string_name_as_string(String *r_dest, const StringName *p_name) {
	memnew_placement(r_dest, String(p_name->operator String()));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_as_string(String *r_dest, const NodePath *p_np) {
	memnew_placement(r_dest, String(p_np->operator String()));
}

GD_PINVOKE_EXPORT godot_packed_byte_array godotsharp_packed_byte_array_new_mem_copy(const uint8_t *p_src, int32_t p_length) {
	godot_packed_byte_array ret;
	memnew_placement(&ret, PackedByteArray);
	PackedByteArray *array = reinterpret_cast<PackedByteArray *>(&ret);
	array->resize(p_length);
	uint8_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(uint8_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_int32_array godotsharp_packed_int32_array_new_mem_copy(const int32_t *p_src, int32_t p_length) {
	godot_packed_int32_array ret;
	memnew_placement(&ret, PackedInt32Array);
	PackedInt32Array *array = reinterpret_cast<PackedInt32Array *>(&ret);
	array->resize(p_length);
	int32_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int32_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_int64_array godotsharp_packed_int64_array_new_mem_copy(const int64_t *p_src, int32_t p_length) {
	godot_packed_int64_array ret;
	memnew_placement(&ret, PackedInt64Array);
	PackedInt64Array *array = reinterpret_cast<PackedInt64Array *>(&ret);
	array->resize(p_length);
	int64_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int64_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_float32_array godotsharp_packed_float32_array_new_mem_copy(const float *p_src, int32_t p_length) {
	godot_packed_float32_array ret;
	memnew_placement(&ret, PackedFloat32Array);
	PackedFloat32Array *array = reinterpret_cast<PackedFloat32Array *>(&ret);
	array->resize(p_length);
	float *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(float));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_float64_array godotsharp_packed_float64_array_new_mem_copy(const double *p_src, int32_t p_length) {
	godot_packed_float64_array ret;
	memnew_placement(&ret, PackedFloat64Array);
	PackedFloat64Array *array = reinterpret_cast<PackedFloat64Array *>(&ret);
	array->resize(p_length);
	double *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(double));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_vector2_array godotsharp_packed_vector2_array_new_mem_copy(const Vector2 *p_src, int32_t p_length) {
	godot_packed_vector2_array ret;
	memnew_placement(&ret, PackedVector2Array);
	PackedVector2Array *array = reinterpret_cast<PackedVector2Array *>(&ret);
	array->resize(p_length);
	Vector2 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector2));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_vector3_array godotsharp_packed_vector3_array_new_mem_copy(const Vector3 *p_src, int32_t p_length) {
	godot_packed_vector3_array ret;
	memnew_placement(&ret, PackedVector3Array);
	PackedVector3Array *array = reinterpret_cast<PackedVector3Array *>(&ret);
	array->resize(p_length);
	Vector3 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector3));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_color_array godotsharp_packed_color_array_new_mem_copy(const Color *p_src, int32_t p_length) {
	godot_packed_color_array ret;
	memnew_placement(&ret, PackedColorArray);
	PackedColorArray *array = reinterpret_cast<PackedColorArray *>(&ret);
	array->resize(p_length);
	Color *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Color));
	return ret;
}

GD_PINVOKE_EXPORT void godotsharp_packed_string_array_add(PackedStringArray *r_dest, const String *p_element) {
	r_dest->append(*p_element);
}

GD_PINVOKE_EXPORT void godotsharp_callable_new_with_delegate(void *p_delegate_handle, Callable *r_callable) {
	// TODO: Use pooling for ManagedCallable instances.
	CallableCustom *managed_callable = memnew(ManagedCallable(p_delegate_handle));
	*r_callable = Callable(managed_callable);
}

GD_PINVOKE_EXPORT bool godotsharp_callable_get_data_for_marshalling(const Callable *p_callable,
		void **r_delegate_handle, Object **r_object, StringName *r_name) {
	if (p_callable->is_custom()) {
		CallableCustom *custom = p_callable->get_custom();
		CallableCustom::CompareEqualFunc compare_equal_func = custom->get_compare_equal_func();

		if (compare_equal_func == ManagedCallable::compare_equal_func_ptr) {
			ManagedCallable *managed_callable = static_cast<ManagedCallable *>(custom);
			*r_delegate_handle = managed_callable->get_delegate();
			*r_object = nullptr;
			*r_name = StringName();
			return true;
		} else if (compare_equal_func == SignalAwaiterCallable::compare_equal_func_ptr) {
			SignalAwaiterCallable *signal_awaiter_callable = static_cast<SignalAwaiterCallable *>(custom);
			*r_delegate_handle = nullptr;
			*r_object = ObjectDB::get_instance(signal_awaiter_callable->get_object());
			*r_name = signal_awaiter_callable->get_signal();
			return true;
		} else if (compare_equal_func == EventSignalCallable::compare_equal_func_ptr) {
			EventSignalCallable *event_signal_callable = static_cast<EventSignalCallable *>(custom);
			*r_delegate_handle = nullptr;
			*r_object = ObjectDB::get_instance(event_signal_callable->get_object());
			*r_name = event_signal_callable->get_signal();
			return true;
		}

		// Some other CallableCustom. We only support ManagedCallable.
		*r_delegate_handle = nullptr;
		*r_object = nullptr;
		*r_name = StringName();
		return false;
	} else {
		*r_delegate_handle = nullptr;
		*r_object = ObjectDB::get_instance(p_callable->get_object_id());
		*r_name = p_callable->get_method();
		return true;
	}
}

// GDNative functions

// gdnative.h

GD_PINVOKE_EXPORT void godotsharp_method_bind_ptrcall(godot_method_bind *p_method_bind, godot_object *p_instance, const void **p_args, void *p_ret) {
	godot_method_bind_ptrcall(p_method_bind, p_instance, p_args, p_ret);
}

GD_PINVOKE_EXPORT godot_variant godotsharp_method_bind_call(godot_method_bind *p_method_bind, godot_object *p_instance, const godot_variant **p_args, const int p_arg_count, godot_variant_call_error *p_call_error) {
	return godot_method_bind_call(p_method_bind, p_instance, p_args, p_arg_count, p_call_error);
}

// variant.h

GD_PINVOKE_EXPORT void godotsharp_variant_new_string_name(godot_variant *r_dest, const godot_string_name *p_s) {
	godot_variant_new_string_name(r_dest, p_s);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_node_path(godot_variant *r_dest, const godot_node_path *p_np) {
	godot_variant_new_node_path(r_dest, p_np);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_object(godot_variant *r_dest, const godot_object *p_obj) {
	godot_variant_new_object(r_dest, p_obj);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_transform2d(godot_variant *r_dest, const godot_transform2d *p_t2d) {
	godot_variant_new_transform2d(r_dest, p_t2d);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_basis(godot_variant *r_dest, const godot_basis *p_basis) {
	godot_variant_new_basis(r_dest, p_basis);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_transform3d(godot_variant *r_dest, const godot_transform3d *p_trans) {
	godot_variant_new_transform3d(r_dest, p_trans);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_aabb(godot_variant *r_dest, const godot_aabb *p_aabb) {
	godot_variant_new_aabb(r_dest, p_aabb);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_dictionary(godot_variant *r_dest, const godot_dictionary *p_dict) {
	godot_variant_new_dictionary(r_dest, p_dict);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_array(godot_variant *r_dest, const godot_array *p_arr) {
	godot_variant_new_array(r_dest, p_arr);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_byte_array(godot_variant *r_dest, const godot_packed_byte_array *p_pba) {
	godot_variant_new_packed_byte_array(r_dest, p_pba);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_int32_array(godot_variant *r_dest, const godot_packed_int32_array *p_pia) {
	godot_variant_new_packed_int32_array(r_dest, p_pia);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_int64_array(godot_variant *r_dest, const godot_packed_int64_array *p_pia) {
	godot_variant_new_packed_int64_array(r_dest, p_pia);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_float32_array(godot_variant *r_dest, const godot_packed_float32_array *p_pra) {
	godot_variant_new_packed_float32_array(r_dest, p_pra);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_float64_array(godot_variant *r_dest, const godot_packed_float64_array *p_pra) {
	godot_variant_new_packed_float64_array(r_dest, p_pra);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_string_array(godot_variant *r_dest, const godot_packed_string_array *p_psa) {
	godot_variant_new_packed_string_array(r_dest, p_psa);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_vector2_array(godot_variant *r_dest, const godot_packed_vector2_array *p_pv2a) {
	godot_variant_new_packed_vector2_array(r_dest, p_pv2a);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_vector3_array(godot_variant *r_dest, const godot_packed_vector3_array *p_pv3a) {
	godot_variant_new_packed_vector3_array(r_dest, p_pv3a);
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_color_array(godot_variant *r_dest, const godot_packed_color_array *p_pca) {
	godot_variant_new_packed_color_array(r_dest, p_pca);
}

GD_PINVOKE_EXPORT godot_bool godotsharp_variant_as_bool(const godot_variant *p_self) {
	return godot_variant_as_bool(p_self);
}

GD_PINVOKE_EXPORT godot_int godotsharp_variant_as_int(const godot_variant *p_self) {
	return godot_variant_as_int(p_self);
}

GD_PINVOKE_EXPORT godot_float godotsharp_variant_as_float(const godot_variant *p_self) {
	return godot_variant_as_float(p_self);
}

GD_PINVOKE_EXPORT godot_string godotsharp_variant_as_string(const godot_variant *p_self) {
	return godot_variant_as_string(p_self);
}

GD_PINVOKE_EXPORT godot_vector2 godotsharp_variant_as_vector2(const godot_variant *p_self) {
	return godot_variant_as_vector2(p_self);
}

GD_PINVOKE_EXPORT godot_vector2i godotsharp_variant_as_vector2i(const godot_variant *p_self) {
	return godot_variant_as_vector2i(p_self);
}

GD_PINVOKE_EXPORT godot_rect2 godotsharp_variant_as_rect2(const godot_variant *p_self) {
	return godot_variant_as_rect2(p_self);
}

GD_PINVOKE_EXPORT godot_rect2i godotsharp_variant_as_rect2i(const godot_variant *p_self) {
	return godot_variant_as_rect2i(p_self);
}

GD_PINVOKE_EXPORT godot_vector3 godotsharp_variant_as_vector3(const godot_variant *p_self) {
	return godot_variant_as_vector3(p_self);
}

GD_PINVOKE_EXPORT godot_vector3i godotsharp_variant_as_vector3i(const godot_variant *p_self) {
	return godot_variant_as_vector3i(p_self);
}

GD_PINVOKE_EXPORT godot_transform2d godotsharp_variant_as_transform2d(const godot_variant *p_self) {
	return godot_variant_as_transform2d(p_self);
}

GD_PINVOKE_EXPORT godot_plane godotsharp_variant_as_plane(const godot_variant *p_self) {
	return godot_variant_as_plane(p_self);
}

GD_PINVOKE_EXPORT godot_quaternion godotsharp_variant_as_quaternion(const godot_variant *p_self) {
	return godot_variant_as_quaternion(p_self);
}

GD_PINVOKE_EXPORT godot_aabb godotsharp_variant_as_aabb(const godot_variant *p_self) {
	return godot_variant_as_aabb(p_self);
}

GD_PINVOKE_EXPORT godot_basis godotsharp_variant_as_basis(const godot_variant *p_self) {
	return godot_variant_as_basis(p_self);
}

GD_PINVOKE_EXPORT godot_transform3d godotsharp_variant_as_transform3d(const godot_variant *p_self) {
	return godot_variant_as_transform3d(p_self);
}

GD_PINVOKE_EXPORT godot_color godotsharp_variant_as_color(const godot_variant *p_self) {
	return godot_variant_as_color(p_self);
}

GD_PINVOKE_EXPORT godot_string_name godotsharp_variant_as_string_name(const godot_variant *p_self) {
	return godot_variant_as_string_name(p_self);
}

GD_PINVOKE_EXPORT godot_node_path godotsharp_variant_as_node_path(const godot_variant *p_self) {
	return godot_variant_as_node_path(p_self);
}

GD_PINVOKE_EXPORT godot_rid godotsharp_variant_as_rid(const godot_variant *p_self) {
	return godot_variant_as_rid(p_self);
}

GD_PINVOKE_EXPORT godot_callable godotsharp_variant_as_callable(const godot_variant *p_self) {
	return godot_variant_as_callable(p_self);
}

GD_PINVOKE_EXPORT godot_signal godotsharp_variant_as_signal(const godot_variant *p_self) {
	return godot_variant_as_signal(p_self);
}

GD_PINVOKE_EXPORT godot_dictionary godotsharp_variant_as_dictionary(const godot_variant *p_self) {
	return godot_variant_as_dictionary(p_self);
}

GD_PINVOKE_EXPORT godot_array godotsharp_variant_as_array(const godot_variant *p_self) {
	return godot_variant_as_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_byte_array godotsharp_variant_as_packed_byte_array(const godot_variant *p_self) {
	return godot_variant_as_packed_byte_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_int32_array godotsharp_variant_as_packed_int32_array(const godot_variant *p_self) {
	return godot_variant_as_packed_int32_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_int64_array godotsharp_variant_as_packed_int64_array(const godot_variant *p_self) {
	return godot_variant_as_packed_int64_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_float32_array godotsharp_variant_as_packed_float32_array(const godot_variant *p_self) {
	return godot_variant_as_packed_float32_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_float64_array godotsharp_variant_as_packed_float64_array(const godot_variant *p_self) {
	return godot_variant_as_packed_float64_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_string_array godotsharp_variant_as_packed_string_array(const godot_variant *p_self) {
	return godot_variant_as_packed_string_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_vector2_array godotsharp_variant_as_packed_vector2_array(const godot_variant *p_self) {
	return godot_variant_as_packed_vector2_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_vector3_array godotsharp_variant_as_packed_vector3_array(const godot_variant *p_self) {
	return godot_variant_as_packed_vector3_array(p_self);
}

GD_PINVOKE_EXPORT godot_packed_color_array godotsharp_variant_as_packed_color_array(const godot_variant *p_self) {
	return godot_variant_as_packed_color_array(p_self);
}

// string.h

GD_PINVOKE_EXPORT void godotsharp_string_new_with_utf16_chars(godot_string *r_dest, const char16_t *p_contents) {
	godot_string_new_with_utf16_chars(r_dest, p_contents);
}

// string_name.h

GD_PINVOKE_EXPORT void godotsharp_string_name_new_copy(godot_string_name *r_dest, const godot_string_name *p_src) {
	godot_string_name_new_copy(r_dest, p_src);
}

// node_path.h

GD_PINVOKE_EXPORT void godotsharp_node_path_new_copy(godot_node_path *r_dest, const godot_node_path *p_src) {
	godot_node_path_new_copy(r_dest, p_src);
}

// array.h

GD_PINVOKE_EXPORT void godotsharp_array_new_copy(godot_array *r_dest, const godot_array *p_src) {
	godot_array_new_copy(r_dest, p_src);
}

// dictionary.h

GD_PINVOKE_EXPORT void godotsharp_dictionary_new_copy(godot_dictionary *r_dest, const godot_dictionary *p_src) {
	godot_dictionary_new_copy(r_dest, p_src);
}

// destroy functions

GD_PINVOKE_EXPORT void godotsharp_packed_byte_array_destroy(godot_packed_byte_array *p_self) {
	godot_packed_byte_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_int32_array_destroy(godot_packed_int32_array *p_self) {
	godot_packed_int32_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_int64_array_destroy(godot_packed_int64_array *p_self) {
	godot_packed_int64_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_float32_array_destroy(godot_packed_float32_array *p_self) {
	godot_packed_float32_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_float64_array_destroy(godot_packed_float64_array *p_self) {
	godot_packed_float64_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_string_array_destroy(godot_packed_string_array *p_self) {
	godot_packed_string_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_vector2_array_destroy(godot_packed_vector2_array *p_self) {
	godot_packed_vector2_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_vector3_array_destroy(godot_packed_vector3_array *p_self) {
	godot_packed_vector3_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_packed_color_array_destroy(godot_packed_color_array *p_self) {
	godot_packed_color_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_variant_destroy(godot_variant *p_self) {
	godot_variant_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_string_destroy(godot_string *p_self) {
	godot_string_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_string_name_destroy(godot_string_name *p_self) {
	godot_string_name_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_node_path_destroy(godot_node_path *p_self) {
	godot_node_path_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_signal_destroy(godot_signal *p_self) {
	godot_signal_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_callable_destroy(godot_callable *p_self) {
	godot_callable_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_array_destroy(godot_array *p_self) {
	godot_array_destroy(p_self);
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_destroy(godot_dictionary *p_self) {
	godot_dictionary_destroy(p_self);
}

#ifdef __cplusplus
}
#endif

// We need this to prevent the functions from being stripped.
void *godotsharp_pinvoke_funcs[95] = {
	(void *)godotsharp_method_bind_get_method,
	(void *)godotsharp_get_class_constructor,
	(void *)godotsharp_invoke_class_constructor,
	(void *)godotsharp_engine_get_singleton,
	(void *)godotsharp_ref_destroy,
	(void *)godotsharp_string_name_new_from_string,
	(void *)godotsharp_node_path_new_from_string,
	(void *)godotsharp_string_name_as_string,
	(void *)godotsharp_node_path_as_string,
	(void *)godotsharp_packed_byte_array_new_mem_copy,
	(void *)godotsharp_packed_int32_array_new_mem_copy,
	(void *)godotsharp_packed_int64_array_new_mem_copy,
	(void *)godotsharp_packed_float32_array_new_mem_copy,
	(void *)godotsharp_packed_float64_array_new_mem_copy,
	(void *)godotsharp_packed_vector2_array_new_mem_copy,
	(void *)godotsharp_packed_vector3_array_new_mem_copy,
	(void *)godotsharp_packed_color_array_new_mem_copy,
	(void *)godotsharp_packed_string_array_add,
	(void *)godotsharp_callable_new_with_delegate,
	(void *)godotsharp_callable_get_data_for_marshalling,
	(void *)godotsharp_method_bind_ptrcall,
	(void *)godotsharp_method_bind_call,
	(void *)godotsharp_variant_new_string_name,
	(void *)godotsharp_variant_new_node_path,
	(void *)godotsharp_variant_new_object,
	(void *)godotsharp_variant_new_transform2d,
	(void *)godotsharp_variant_new_basis,
	(void *)godotsharp_variant_new_transform3d,
	(void *)godotsharp_variant_new_aabb,
	(void *)godotsharp_variant_new_dictionary,
	(void *)godotsharp_variant_new_array,
	(void *)godotsharp_variant_new_packed_byte_array,
	(void *)godotsharp_variant_new_packed_int32_array,
	(void *)godotsharp_variant_new_packed_int64_array,
	(void *)godotsharp_variant_new_packed_float32_array,
	(void *)godotsharp_variant_new_packed_float64_array,
	(void *)godotsharp_variant_new_packed_string_array,
	(void *)godotsharp_variant_new_packed_vector2_array,
	(void *)godotsharp_variant_new_packed_vector3_array,
	(void *)godotsharp_variant_new_packed_color_array,
	(void *)godotsharp_variant_as_bool,
	(void *)godotsharp_variant_as_int,
	(void *)godotsharp_variant_as_float,
	(void *)godotsharp_variant_as_string,
	(void *)godotsharp_variant_as_vector2,
	(void *)godotsharp_variant_as_vector2i,
	(void *)godotsharp_variant_as_rect2,
	(void *)godotsharp_variant_as_rect2i,
	(void *)godotsharp_variant_as_vector3,
	(void *)godotsharp_variant_as_vector3i,
	(void *)godotsharp_variant_as_transform2d,
	(void *)godotsharp_variant_as_plane,
	(void *)godotsharp_variant_as_quaternion,
	(void *)godotsharp_variant_as_aabb,
	(void *)godotsharp_variant_as_basis,
	(void *)godotsharp_variant_as_transform3d,
	(void *)godotsharp_variant_as_color,
	(void *)godotsharp_variant_as_string_name,
	(void *)godotsharp_variant_as_node_path,
	(void *)godotsharp_variant_as_rid,
	(void *)godotsharp_variant_as_callable,
	(void *)godotsharp_variant_as_signal,
	(void *)godotsharp_variant_as_dictionary,
	(void *)godotsharp_variant_as_array,
	(void *)godotsharp_variant_as_packed_byte_array,
	(void *)godotsharp_variant_as_packed_int32_array,
	(void *)godotsharp_variant_as_packed_int64_array,
	(void *)godotsharp_variant_as_packed_float32_array,
	(void *)godotsharp_variant_as_packed_float64_array,
	(void *)godotsharp_variant_as_packed_string_array,
	(void *)godotsharp_variant_as_packed_vector2_array,
	(void *)godotsharp_variant_as_packed_vector3_array,
	(void *)godotsharp_variant_as_packed_color_array,
	(void *)godotsharp_string_new_with_utf16_chars,
	(void *)godotsharp_string_name_new_copy,
	(void *)godotsharp_node_path_new_copy,
	(void *)godotsharp_array_new_copy,
	(void *)godotsharp_dictionary_new_copy,
	(void *)godotsharp_packed_byte_array_destroy,
	(void *)godotsharp_packed_int32_array_destroy,
	(void *)godotsharp_packed_int64_array_destroy,
	(void *)godotsharp_packed_float32_array_destroy,
	(void *)godotsharp_packed_float64_array_destroy,
	(void *)godotsharp_packed_string_array_destroy,
	(void *)godotsharp_packed_vector2_array_destroy,
	(void *)godotsharp_packed_vector3_array_destroy,
	(void *)godotsharp_packed_color_array_destroy,
	(void *)godotsharp_variant_destroy,
	(void *)godotsharp_string_destroy,
	(void *)godotsharp_string_name_destroy,
	(void *)godotsharp_node_path_destroy,
	(void *)godotsharp_signal_destroy,
	(void *)godotsharp_callable_destroy,
	(void *)godotsharp_array_destroy,
	(void *)godotsharp_dictionary_destroy
};
