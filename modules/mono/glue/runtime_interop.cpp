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

#include "../interop_types.h"

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

// For ArrayPrivate and DictionaryPrivate
static_assert(sizeof(SafeRefCount) == sizeof(uint32_t));

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

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_byte_array_new_mem_copy(const uint8_t *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedByteArray);
	PackedByteArray *array = reinterpret_cast<PackedByteArray *>(&ret);
	array->resize(p_length);
	uint8_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(uint8_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_int32_array_new_mem_copy(const int32_t *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedInt32Array);
	PackedInt32Array *array = reinterpret_cast<PackedInt32Array *>(&ret);
	array->resize(p_length);
	int32_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int32_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_int64_array_new_mem_copy(const int64_t *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedInt64Array);
	PackedInt64Array *array = reinterpret_cast<PackedInt64Array *>(&ret);
	array->resize(p_length);
	int64_t *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(int64_t));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_float32_array_new_mem_copy(const float *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedFloat32Array);
	PackedFloat32Array *array = reinterpret_cast<PackedFloat32Array *>(&ret);
	array->resize(p_length);
	float *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(float));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_float64_array_new_mem_copy(const double *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedFloat64Array);
	PackedFloat64Array *array = reinterpret_cast<PackedFloat64Array *>(&ret);
	array->resize(p_length);
	double *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(double));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_vector2_array_new_mem_copy(const Vector2 *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedVector2Array);
	PackedVector2Array *array = reinterpret_cast<PackedVector2Array *>(&ret);
	array->resize(p_length);
	Vector2 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector2));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_vector3_array_new_mem_copy(const Vector3 *p_src, int32_t p_length) {
	godot_packed_array ret;
	memnew_placement(&ret, PackedVector3Array);
	PackedVector3Array *array = reinterpret_cast<PackedVector3Array *>(&ret);
	array->resize(p_length);
	Vector3 *dst = array->ptrw();
	memcpy(dst, p_src, p_length * sizeof(Vector3));
	return ret;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_packed_color_array_new_mem_copy(const Color *p_src, int32_t p_length) {
	godot_packed_array ret;
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
	memnew_placement(r_callable, Callable(managed_callable));
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
			memnew_placement(r_name, StringName());
			return true;
		} else if (compare_equal_func == SignalAwaiterCallable::compare_equal_func_ptr) {
			SignalAwaiterCallable *signal_awaiter_callable = static_cast<SignalAwaiterCallable *>(custom);
			*r_delegate_handle = nullptr;
			*r_object = ObjectDB::get_instance(signal_awaiter_callable->get_object());
			memnew_placement(r_name, StringName(signal_awaiter_callable->get_signal()));
			return true;
		} else if (compare_equal_func == EventSignalCallable::compare_equal_func_ptr) {
			EventSignalCallable *event_signal_callable = static_cast<EventSignalCallable *>(custom);
			*r_delegate_handle = nullptr;
			*r_object = ObjectDB::get_instance(event_signal_callable->get_object());
			memnew_placement(r_name, StringName(event_signal_callable->get_signal()));
			return true;
		}

		// Some other CallableCustom. We only support ManagedCallable.
		*r_delegate_handle = nullptr;
		*r_object = nullptr;
		memnew_placement(r_name, StringName());
		return false;
	} else {
		*r_delegate_handle = nullptr;
		*r_object = ObjectDB::get_instance(p_callable->get_object_id());
		memnew_placement(r_name, StringName(p_callable->get_method()));
		return true;
	}
}

// GDNative functions

// gdnative.h

GD_PINVOKE_EXPORT void godotsharp_method_bind_ptrcall(MethodBind *p_method_bind, Object *p_instance, const void **p_args, void *p_ret) {
	p_method_bind->ptrcall(p_instance, p_args, p_ret);
}

GD_PINVOKE_EXPORT godot_variant godotsharp_method_bind_call(MethodBind *p_method_bind, Object *p_instance, const godot_variant **p_args, const int32_t p_arg_count, Callable::CallError *p_call_error) {
	godot_variant ret;
	memnew_placement(&ret, Variant());

	Variant *ret_val = (Variant *)&ret;

	*ret_val = p_method_bind->call(p_instance, (const Variant **)p_args, p_arg_count, *p_call_error);

	return ret;
}

// variant.h

GD_PINVOKE_EXPORT void godotsharp_variant_new_string_name(godot_variant *r_dest, const StringName *p_s) {
	memnew_placement(r_dest, Variant(*p_s));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_node_path(godot_variant *r_dest, const NodePath *p_np) {
	memnew_placement(r_dest, Variant(*p_np));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_object(godot_variant *r_dest, const Object *p_obj) {
	memnew_placement(r_dest, Variant(p_obj));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_transform2d(godot_variant *r_dest, const Transform2D *p_t2d) {
	memnew_placement(r_dest, Variant(*p_t2d));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_vector4(godot_variant *r_dest, const Vector4 *p_vec4) {
	memnew_placement(r_dest, Variant(*p_vec4));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_vector4i(godot_variant *r_dest, const Vector4i *p_vec4i) {
	memnew_placement(r_dest, Variant(*p_vec4i));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_basis(godot_variant *r_dest, const Basis *p_basis) {
	memnew_placement(r_dest, Variant(*p_basis));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_transform3d(godot_variant *r_dest, const Transform3D *p_trans) {
	memnew_placement(r_dest, Variant(*p_trans));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_projection(godot_variant *r_dest, const Projection *p_proj) {
	memnew_placement(r_dest, Variant(*p_proj));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_aabb(godot_variant *r_dest, const AABB *p_aabb) {
	memnew_placement(r_dest, Variant(*p_aabb));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_dictionary(godot_variant *r_dest, const Dictionary *p_dict) {
	memnew_placement(r_dest, Variant(*p_dict));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_array(godot_variant *r_dest, const Array *p_arr) {
	memnew_placement(r_dest, Variant(*p_arr));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_byte_array(godot_variant *r_dest, const PackedByteArray *p_pba) {
	memnew_placement(r_dest, Variant(*p_pba));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_int32_array(godot_variant *r_dest, const PackedInt32Array *p_pia) {
	memnew_placement(r_dest, Variant(*p_pia));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_int64_array(godot_variant *r_dest, const PackedInt64Array *p_pia) {
	memnew_placement(r_dest, Variant(*p_pia));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_float32_array(godot_variant *r_dest, const PackedFloat32Array *p_pra) {
	memnew_placement(r_dest, Variant(*p_pra));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_float64_array(godot_variant *r_dest, const PackedFloat64Array *p_pra) {
	memnew_placement(r_dest, Variant(*p_pra));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_string_array(godot_variant *r_dest, const PackedStringArray *p_psa) {
	memnew_placement(r_dest, Variant(*p_psa));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_vector2_array(godot_variant *r_dest, const PackedVector2Array *p_pv2a) {
	memnew_placement(r_dest, Variant(*p_pv2a));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_vector3_array(godot_variant *r_dest, const PackedVector3Array *p_pv3a) {
	memnew_placement(r_dest, Variant(*p_pv3a));
}

GD_PINVOKE_EXPORT void godotsharp_variant_new_packed_color_array(godot_variant *r_dest, const PackedColorArray *p_pca) {
	memnew_placement(r_dest, Variant(*p_pca));
}

GD_PINVOKE_EXPORT bool godotsharp_variant_as_bool(const Variant *p_self) {
	return p_self->operator bool();
}

GD_PINVOKE_EXPORT int64_t godotsharp_variant_as_int(const Variant *p_self) {
	return p_self->operator int64_t();
}

GD_PINVOKE_EXPORT double godotsharp_variant_as_float(const Variant *p_self) {
	return p_self->operator double();
}

GD_PINVOKE_EXPORT godot_string godotsharp_variant_as_string(const Variant *p_self) {
	godot_string raw_dest;
	String *dest = (String *)&raw_dest;
	memnew_placement(dest, String(p_self->operator String()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector2 godotsharp_variant_as_vector2(const Variant *p_self) {
	godot_vector2 raw_dest;
	Vector2 *dest = (Vector2 *)&raw_dest;
	memnew_placement(dest, Vector2(p_self->operator Vector2()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector2i godotsharp_variant_as_vector2i(const Variant *p_self) {
	godot_vector2i raw_dest;
	Vector2i *dest = (Vector2i *)&raw_dest;
	memnew_placement(dest, Vector2i(p_self->operator Vector2i()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_rect2 godotsharp_variant_as_rect2(const Variant *p_self) {
	godot_rect2 raw_dest;
	Rect2 *dest = (Rect2 *)&raw_dest;
	memnew_placement(dest, Rect2(p_self->operator Rect2()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_rect2i godotsharp_variant_as_rect2i(const Variant *p_self) {
	godot_rect2i raw_dest;
	Rect2i *dest = (Rect2i *)&raw_dest;
	memnew_placement(dest, Rect2i(p_self->operator Rect2i()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector3 godotsharp_variant_as_vector3(const Variant *p_self) {
	godot_vector3 raw_dest;
	Vector3 *dest = (Vector3 *)&raw_dest;
	memnew_placement(dest, Vector3(p_self->operator Vector3()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector3i godotsharp_variant_as_vector3i(const Variant *p_self) {
	godot_vector3i raw_dest;
	Vector3i *dest = (Vector3i *)&raw_dest;
	memnew_placement(dest, Vector3i(p_self->operator Vector3i()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_transform2d godotsharp_variant_as_transform2d(const Variant *p_self) {
	godot_transform2d raw_dest;
	Transform2D *dest = (Transform2D *)&raw_dest;
	memnew_placement(dest, Transform2D(p_self->operator Transform2D()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector4 godotsharp_variant_as_vector4(const Variant *p_self) {
	godot_vector4 raw_dest;
	Vector4 *dest = (Vector4 *)&raw_dest;
	memnew_placement(dest, Vector4(p_self->operator Vector4()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_vector4i godotsharp_variant_as_vector4i(const Variant *p_self) {
	godot_vector4i raw_dest;
	Vector4i *dest = (Vector4i *)&raw_dest;
	memnew_placement(dest, Vector4i(p_self->operator Vector4i()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_plane godotsharp_variant_as_plane(const Variant *p_self) {
	godot_plane raw_dest;
	Plane *dest = (Plane *)&raw_dest;
	memnew_placement(dest, Plane(p_self->operator Plane()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_quaternion godotsharp_variant_as_quaternion(const Variant *p_self) {
	godot_quaternion raw_dest;
	Quaternion *dest = (Quaternion *)&raw_dest;
	memnew_placement(dest, Quaternion(p_self->operator Quaternion()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_aabb godotsharp_variant_as_aabb(const Variant *p_self) {
	godot_aabb raw_dest;
	AABB *dest = (AABB *)&raw_dest;
	memnew_placement(dest, AABB(p_self->operator ::AABB()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_basis godotsharp_variant_as_basis(const Variant *p_self) {
	godot_basis raw_dest;
	Basis *dest = (Basis *)&raw_dest;
	memnew_placement(dest, Basis(p_self->operator Basis()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_transform3d godotsharp_variant_as_transform3d(const Variant *p_self) {
	godot_transform3d raw_dest;
	Transform3D *dest = (Transform3D *)&raw_dest;
	memnew_placement(dest, Transform3D(p_self->operator Transform3D()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_projection godotsharp_variant_as_projection(const Variant *p_self) {
	godot_projection raw_dest;
	Projection *dest = (Projection *)&raw_dest;
	memnew_placement(dest, Projection(p_self->operator Projection()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_color godotsharp_variant_as_color(const Variant *p_self) {
	godot_color raw_dest;
	Color *dest = (Color *)&raw_dest;
	memnew_placement(dest, Color(p_self->operator Color()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_string_name godotsharp_variant_as_string_name(const Variant *p_self) {
	godot_string_name raw_dest;
	StringName *dest = (StringName *)&raw_dest;
	memnew_placement(dest, StringName(p_self->operator StringName()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_node_path godotsharp_variant_as_node_path(const Variant *p_self) {
	godot_node_path raw_dest;
	NodePath *dest = (NodePath *)&raw_dest;
	memnew_placement(dest, NodePath(p_self->operator NodePath()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_rid godotsharp_variant_as_rid(const Variant *p_self) {
	godot_rid raw_dest;
	RID *dest = (RID *)&raw_dest;
	memnew_placement(dest, RID(p_self->operator ::RID()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_callable godotsharp_variant_as_callable(const Variant *p_self) {
	godot_callable raw_dest;
	Callable *dest = (Callable *)&raw_dest;
	memnew_placement(dest, Callable(p_self->operator Callable()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_signal godotsharp_variant_as_signal(const Variant *p_self) {
	godot_signal raw_dest;
	Signal *dest = (Signal *)&raw_dest;
	memnew_placement(dest, Signal(p_self->operator Signal()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_dictionary godotsharp_variant_as_dictionary(const Variant *p_self) {
	godot_dictionary raw_dest;
	Dictionary *dest = (Dictionary *)&raw_dest;
	memnew_placement(dest, Dictionary(p_self->operator Dictionary()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_array godotsharp_variant_as_array(const Variant *p_self) {
	godot_array raw_dest;
	Array *dest = (Array *)&raw_dest;
	memnew_placement(dest, Array(p_self->operator Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_byte_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedByteArray *dest = (PackedByteArray *)&raw_dest;
	memnew_placement(dest, PackedByteArray(p_self->operator PackedByteArray()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_int32_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedInt32Array *dest = (PackedInt32Array *)&raw_dest;
	memnew_placement(dest, PackedInt32Array(p_self->operator PackedInt32Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_int64_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedInt64Array *dest = (PackedInt64Array *)&raw_dest;
	memnew_placement(dest, PackedInt64Array(p_self->operator PackedInt64Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_float32_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedFloat32Array *dest = (PackedFloat32Array *)&raw_dest;
	memnew_placement(dest, PackedFloat32Array(p_self->operator PackedFloat32Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_float64_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedFloat64Array *dest = (PackedFloat64Array *)&raw_dest;
	memnew_placement(dest, PackedFloat64Array(p_self->operator PackedFloat64Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_string_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedStringArray *dest = (PackedStringArray *)&raw_dest;
	memnew_placement(dest, PackedStringArray(p_self->operator PackedStringArray()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_vector2_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedVector2Array *dest = (PackedVector2Array *)&raw_dest;
	memnew_placement(dest, PackedVector2Array(p_self->operator PackedVector2Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_vector3_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedVector3Array *dest = (PackedVector3Array *)&raw_dest;
	memnew_placement(dest, PackedVector3Array(p_self->operator PackedVector3Array()));
	return raw_dest;
}

GD_PINVOKE_EXPORT godot_packed_array godotsharp_variant_as_packed_color_array(const Variant *p_self) {
	godot_packed_array raw_dest;
	PackedColorArray *dest = (PackedColorArray *)&raw_dest;
	memnew_placement(dest, PackedColorArray(p_self->operator PackedColorArray()));
	return raw_dest;
}

GD_PINVOKE_EXPORT bool godotsharp_variant_equals(const godot_variant *p_a, const godot_variant *p_b) {
	return *reinterpret_cast<const Variant *>(p_a) == *reinterpret_cast<const Variant *>(p_b);
}

// string.h

GD_PINVOKE_EXPORT void godotsharp_string_new_with_utf16_chars(String *r_dest, const char16_t *p_contents) {
	memnew_placement(r_dest, String());
	r_dest->parse_utf16(p_contents);
}

// string_name.h

GD_PINVOKE_EXPORT void godotsharp_string_name_new_copy(StringName *r_dest, const StringName *p_src) {
	memnew_placement(r_dest, StringName(*p_src));
}

// node_path.h

GD_PINVOKE_EXPORT void godotsharp_node_path_new_copy(NodePath *r_dest, const NodePath *p_src) {
	memnew_placement(r_dest, NodePath(*p_src));
}

// array.h

GD_PINVOKE_EXPORT void godotsharp_array_new(Array *r_dest) {
	memnew_placement(r_dest, Array);
}

GD_PINVOKE_EXPORT void godotsharp_array_new_copy(Array *r_dest, const Array *p_src) {
	memnew_placement(r_dest, Array(*p_src));
}

GD_PINVOKE_EXPORT godot_variant *godotsharp_array_ptrw(godot_array *p_self) {
	return reinterpret_cast<godot_variant *>(&reinterpret_cast<Array *>(p_self)->operator[](0));
}

// dictionary.h

GD_PINVOKE_EXPORT void godotsharp_dictionary_new(Dictionary *r_dest) {
	memnew_placement(r_dest, Dictionary);
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_new_copy(Dictionary *r_dest, const Dictionary *p_src) {
	memnew_placement(r_dest, Dictionary(*p_src));
}

// destroy functions

GD_PINVOKE_EXPORT void godotsharp_packed_byte_array_destroy(PackedByteArray *p_self) {
	p_self->~PackedByteArray();
}

GD_PINVOKE_EXPORT void godotsharp_packed_int32_array_destroy(PackedInt32Array *p_self) {
	p_self->~PackedInt32Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_int64_array_destroy(PackedInt64Array *p_self) {
	p_self->~PackedInt64Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_float32_array_destroy(PackedFloat32Array *p_self) {
	p_self->~PackedFloat32Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_float64_array_destroy(PackedFloat64Array *p_self) {
	p_self->~PackedFloat64Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_string_array_destroy(PackedStringArray *p_self) {
	p_self->~PackedStringArray();
}

GD_PINVOKE_EXPORT void godotsharp_packed_vector2_array_destroy(PackedVector2Array *p_self) {
	p_self->~PackedVector2Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_vector3_array_destroy(PackedVector3Array *p_self) {
	p_self->~PackedVector3Array();
}

GD_PINVOKE_EXPORT void godotsharp_packed_color_array_destroy(PackedColorArray *p_self) {
	p_self->~PackedColorArray();
}

GD_PINVOKE_EXPORT void godotsharp_variant_destroy(Variant *p_self) {
	p_self->~Variant();
}

GD_PINVOKE_EXPORT void godotsharp_string_destroy(String *p_self) {
	p_self->~String();
}

GD_PINVOKE_EXPORT void godotsharp_string_name_destroy(StringName *p_self) {
	p_self->~StringName();
}

GD_PINVOKE_EXPORT void godotsharp_node_path_destroy(NodePath *p_self) {
	p_self->~NodePath();
}

GD_PINVOKE_EXPORT void godotsharp_signal_destroy(Signal *p_self) {
	p_self->~Signal();
}

GD_PINVOKE_EXPORT void godotsharp_callable_destroy(Callable *p_self) {
	p_self->~Callable();
}

GD_PINVOKE_EXPORT void godotsharp_array_destroy(Array *p_self) {
	p_self->~Array();
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_destroy(Dictionary *p_self) {
	p_self->~Dictionary();
}

// Array

GD_PINVOKE_EXPORT int32_t godotsharp_array_add(Array *p_self, const Variant *p_item) {
	p_self->append(*p_item);
	return p_self->size();
}

GD_PINVOKE_EXPORT void godotsharp_array_duplicate(const Array *p_self, bool p_deep, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->duplicate(p_deep)));
}

GD_PINVOKE_EXPORT int32_t godotsharp_array_index_of(const Array *p_self, const Variant *p_item) {
	return p_self->find(*p_item);
}

GD_PINVOKE_EXPORT void godotsharp_array_insert(Array *p_self, int32_t p_index, const Variant *p_item) {
	p_self->insert(p_index, *p_item);
}

GD_PINVOKE_EXPORT void godotsharp_array_remove_at(Array *p_self, int32_t p_index) {
	p_self->remove_at(p_index);
}

GD_PINVOKE_EXPORT int32_t godotsharp_array_resize(Array *p_self, int32_t p_new_size) {
	return (int32_t)p_self->resize(p_new_size);
}

GD_PINVOKE_EXPORT void godotsharp_array_shuffle(Array *p_self) {
	p_self->shuffle();
}

// Dictionary

GD_PINVOKE_EXPORT bool godotsharp_dictionary_try_get_value(const Dictionary *p_self, const Variant *p_key, Variant *r_value) {
	const Variant *ret = p_self->getptr(*p_key);
	if (ret == nullptr) {
		memnew_placement(r_value, Variant());
		return false;
	}
	memnew_placement(r_value, Variant(*ret));
	return true;
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_set_value(Dictionary *p_self, const Variant *p_key, const Variant *p_value) {
	p_self->operator[](*p_key) = *p_value;
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_keys(const Dictionary *p_self, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->keys()));
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_values(const Dictionary *p_self, Array *r_dest) {
	memnew_placement(r_dest, Array(p_self->values()));
}

GD_PINVOKE_EXPORT int32_t godotsharp_dictionary_count(const Dictionary *p_self) {
	return p_self->size();
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_key_value_pair_at(const Dictionary *p_self, int32_t p_index, Variant *r_key, Variant *r_value) {
	memnew_placement(r_key, Variant(p_self->get_key_at_index(p_index)));
	memnew_placement(r_value, Variant(p_self->get_value_at_index(p_index)));
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_add(Dictionary *p_self, const Variant *p_key, const Variant *p_value) {
	p_self->operator[](*p_key) = *p_value;
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_clear(Dictionary *p_self) {
	p_self->clear();
}

GD_PINVOKE_EXPORT bool godotsharp_dictionary_contains_key(const Dictionary *p_self, const Variant *p_key) {
	return p_self->has(*p_key);
}

GD_PINVOKE_EXPORT void godotsharp_dictionary_duplicate(const Dictionary *p_self, bool p_deep, Dictionary *r_dest) {
	memnew_placement(r_dest, Dictionary(p_self->duplicate(p_deep)));
}

GD_PINVOKE_EXPORT bool godotsharp_dictionary_remove_key(Dictionary *p_self, const Variant *p_key) {
	return p_self->erase(*p_key);
}

GD_PINVOKE_EXPORT void godotsharp_string_md5_buffer(const String *p_self, PackedByteArray *r_md5_buffer) {
	memnew_placement(r_md5_buffer, PackedByteArray(p_self->md5_buffer()));
}

GD_PINVOKE_EXPORT void godotsharp_string_md5_text(const String *p_self, String *r_md5_text) {
	memnew_placement(r_md5_text, String(p_self->md5_text()));
}

GD_PINVOKE_EXPORT int32_t godotsharp_string_rfind(const String *p_self, const String *p_what, int32_t p_from) {
	return p_self->rfind(*p_what, p_from);
}

GD_PINVOKE_EXPORT int32_t godotsharp_string_rfindn(const String *p_self, const String *p_what, int32_t p_from) {
	return p_self->rfindn(*p_what, p_from);
}

GD_PINVOKE_EXPORT void godotsharp_string_sha256_buffer(const String *p_self, PackedByteArray *r_sha256_buffer) {
	memnew_placement(r_sha256_buffer, PackedByteArray(p_self->sha256_buffer()));
}

GD_PINVOKE_EXPORT void godotsharp_string_sha256_text(const String *p_self, String *r_sha256_text) {
	memnew_placement(r_sha256_text, String(p_self->sha256_text()));
}

GD_PINVOKE_EXPORT void godotsharp_string_simplify_path(const String *p_self, String *r_simplified_path) {
	memnew_placement(r_simplified_path, String(p_self->simplify_path()));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_get_as_property_path(const NodePath *p_ptr, NodePath *r_dest) {
	memnew_placement(r_dest, NodePath(p_ptr->get_as_property_path()));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_get_concatenated_names(const NodePath *p_self, String *r_subnames) {
	memnew_placement(r_subnames, String(p_self->get_concatenated_names()));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_get_concatenated_subnames(const NodePath *p_self, String *r_subnames) {
	memnew_placement(r_subnames, String(p_self->get_concatenated_subnames()));
}

GD_PINVOKE_EXPORT void godotsharp_node_path_get_name(const NodePath *p_self, uint32_t p_idx, String *r_name) {
	memnew_placement(r_name, String(p_self->get_name(p_idx)));
}

GD_PINVOKE_EXPORT int32_t godotsharp_node_path_get_name_count(const NodePath *p_self) {
	return p_self->get_name_count();
}

GD_PINVOKE_EXPORT void godotsharp_node_path_get_subname(const NodePath *p_self, uint32_t p_idx, String *r_subname) {
	memnew_placement(r_subname, String(p_self->get_subname(p_idx)));
}

GD_PINVOKE_EXPORT int32_t godotsharp_node_path_get_subname_count(const NodePath *p_self) {
	return p_self->get_subname_count();
}

GD_PINVOKE_EXPORT bool godotsharp_node_path_is_absolute(const NodePath *p_self) {
	return p_self->is_absolute();
}

#ifdef __cplusplus
}
#endif

// We need this to prevent the functions from being stripped.
void *godotsharp_pinvoke_funcs[138] = {
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
	(void *)godotsharp_variant_new_vector4,
	(void *)godotsharp_variant_new_vector4i,
	(void *)godotsharp_variant_new_basis,
	(void *)godotsharp_variant_new_transform3d,
	(void *)godotsharp_variant_new_projection,
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
	(void *)godotsharp_variant_as_vector4,
	(void *)godotsharp_variant_as_vector4i,
	(void *)godotsharp_variant_as_plane,
	(void *)godotsharp_variant_as_quaternion,
	(void *)godotsharp_variant_as_aabb,
	(void *)godotsharp_variant_as_basis,
	(void *)godotsharp_variant_as_transform3d,
	(void *)godotsharp_variant_as_projection,
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
	(void *)godotsharp_variant_equals,
	(void *)godotsharp_string_new_with_utf16_chars,
	(void *)godotsharp_string_name_new_copy,
	(void *)godotsharp_node_path_new_copy,
	(void *)godotsharp_array_new,
	(void *)godotsharp_array_new_copy,
	(void *)godotsharp_array_ptrw,
	(void *)godotsharp_dictionary_new,
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
	(void *)godotsharp_dictionary_destroy,
	(void *)godotsharp_array_add,
	(void *)godotsharp_array_duplicate,
	(void *)godotsharp_array_index_of,
	(void *)godotsharp_array_insert,
	(void *)godotsharp_array_remove_at,
	(void *)godotsharp_array_resize,
	(void *)godotsharp_array_shuffle,
	(void *)godotsharp_dictionary_try_get_value,
	(void *)godotsharp_dictionary_set_value,
	(void *)godotsharp_dictionary_keys,
	(void *)godotsharp_dictionary_values,
	(void *)godotsharp_dictionary_count,
	(void *)godotsharp_dictionary_key_value_pair_at,
	(void *)godotsharp_dictionary_add,
	(void *)godotsharp_dictionary_clear,
	(void *)godotsharp_dictionary_contains_key,
	(void *)godotsharp_dictionary_duplicate,
	(void *)godotsharp_dictionary_remove_key,
	(void *)godotsharp_string_md5_buffer,
	(void *)godotsharp_string_md5_text,
	(void *)godotsharp_string_rfind,
	(void *)godotsharp_string_rfindn,
	(void *)godotsharp_string_sha256_buffer,
	(void *)godotsharp_string_sha256_text,
	(void *)godotsharp_string_simplify_path,
	(void *)godotsharp_node_path_get_as_property_path,
	(void *)godotsharp_node_path_get_concatenated_names,
	(void *)godotsharp_node_path_get_concatenated_subnames,
	(void *)godotsharp_node_path_get_name,
	(void *)godotsharp_node_path_get_name_count,
	(void *)godotsharp_node_path_get_subname,
	(void *)godotsharp_node_path_get_subname_count,
	(void *)godotsharp_node_path_is_absolute
};
