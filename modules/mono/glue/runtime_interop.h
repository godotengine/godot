/*************************************************************************/
/*  runtime_interop.h                                                    */
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

#ifndef RUNTIME_INTEROP_H
#define RUNTIME_INTEROP_H

#include "core/config/engine.h"
#include "core/debugger/engine_debugger.h"
#include "core/debugger/script_debugger.h"
#include "core/io/marshalls.h"
#include "core/object/class_db.h"
#include "core/object/method_bind.h"
#include "core/os/os.h"
#include "core/string/string_name.h"

#include <gdnative/gdnative.h>

#include "modules/mono/managed_callable.h"
#include "modules/mono/mono_gd/gd_mono_cache.h"
#include "modules/mono/signal_awaiter_utils.h"

#ifdef WIN32
#define GD_CLR_STDCALL __stdcall
#else
#define GD_CLR_STDCALL
#endif

// For ArrayPrivate and DictionaryPrivate
static_assert(sizeof(SafeRefCount) == sizeof(uint32_t));

typedef Object *(*godotsharp_class_creation_func)();

// The order of the fields defined in UnmanagedCallbacks must match the order
// of the defined methods in GodotSharp/Core/NativeInterop/NativeFuncs.cs
struct UnmanagedCallbacks {
	using Func_godotsharp_method_bind_get_method = MethodBind *(GD_CLR_STDCALL *)(const StringName *, const StringName *);
	using Func_godotsharp_get_class_constructor = godotsharp_class_creation_func(GD_CLR_STDCALL *)(const StringName *);
	using Func_godotsharp_engine_get_singleton = Object *(GD_CLR_STDCALL *)(const String *);
	using Func_godotsharp_stack_info_vector_resize = Error(GD_CLR_STDCALL *)(Vector<ScriptLanguage::StackInfo> *, int);
	using Func_godotsharp_stack_info_vector_destroy = void(GD_CLR_STDCALL *)(Vector<ScriptLanguage::StackInfo> *);
	using Func_godotsharp_internal_script_debugger_send_error = void(GD_CLR_STDCALL *)(const String *, const String *, int32_t, const String *, const String *, bool, const Vector<ScriptLanguage::StackInfo> *);
	using Func_godotsharp_internal_script_debugger_is_active = bool(GD_CLR_STDCALL *)();
	using Func_godotsharp_internal_object_get_associated_gchandle = GCHandleIntPtr(GD_CLR_STDCALL *)(Object *);
	using Func_godotsharp_internal_object_disposed = void(GD_CLR_STDCALL *)(Object *, GCHandleIntPtr);
	using Func_godotsharp_internal_refcounted_disposed = void(GD_CLR_STDCALL *)(Object *, GCHandleIntPtr, bool);
	using Func_godotsharp_internal_object_connect_event_signal = void(GD_CLR_STDCALL *)(Object *, const StringName *);
	using Func_godotsharp_internal_signal_awaiter_connect = Error(GD_CLR_STDCALL *)(Object *, StringName *, Object *, GCHandleIntPtr);
	using Func_godotsharp_internal_unmanaged_get_script_instance_managed = GCHandleIntPtr(GD_CLR_STDCALL *)(Object *, bool *);
	using Func_godotsharp_internal_unmanaged_get_instance_binding_managed = GCHandleIntPtr(GD_CLR_STDCALL *)(Object *);
	using Func_godotsharp_internal_unmanaged_instance_binding_create_managed = GCHandleIntPtr(GD_CLR_STDCALL *)(Object *, GCHandleIntPtr);
	using Func_godotsharp_internal_tie_native_managed_to_unmanaged = void(GD_CLR_STDCALL *)(GCHandleIntPtr, Object *, const StringName *, bool);
	using Func_godotsharp_internal_tie_user_managed_to_unmanaged = void(GD_CLR_STDCALL *)(GCHandleIntPtr, Object *, Ref<CSharpScript> *, bool);
	using Func_godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup = void(GD_CLR_STDCALL *)(GCHandleIntPtr, Object *);
	using Func_godotsharp_internal_new_csharp_script = void(GD_CLR_STDCALL *)(Ref<CSharpScript> *);
	using Func_godotsharp_internal_reload_registered_script = void(GD_CLR_STDCALL *)(CSharpScript *);
	using Func_godotsharp_array_filter_godot_objects_by_native = void(GD_CLR_STDCALL *)(StringName *, const Array *, Array *);
	using Func_godotsharp_array_filter_godot_objects_by_non_native = void(GD_CLR_STDCALL *)(const Array *, Array *);
	using Func_godotsharp_ref_new_from_ref_counted_ptr = void(GD_CLR_STDCALL *)(Ref<RefCounted> *, RefCounted *);
	using Func_godotsharp_ref_destroy = void(GD_CLR_STDCALL *)(Ref<RefCounted> *);
	using Func_godotsharp_string_name_new_from_string = void(GD_CLR_STDCALL *)(StringName *, const String *);
	using Func_godotsharp_node_path_new_from_string = void(GD_CLR_STDCALL *)(NodePath *, const String *);
	using Func_godotsharp_string_name_as_string = void(GD_CLR_STDCALL *)(String *, const StringName *);
	using Func_godotsharp_node_path_as_string = void(GD_CLR_STDCALL *)(String *, const NodePath *);
	using Func_godotsharp_packed_byte_array_new_mem_copy = godot_packed_byte_array(GD_CLR_STDCALL *)(const uint8_t *, int32_t);
	using Func_godotsharp_packed_int32_array_new_mem_copy = godot_packed_int32_array(GD_CLR_STDCALL *)(const int32_t *, int32_t);
	using Func_godotsharp_packed_int64_array_new_mem_copy = godot_packed_int64_array(GD_CLR_STDCALL *)(const int64_t *, int32_t);
	using Func_godotsharp_packed_float32_array_new_mem_copy = godot_packed_float32_array(GD_CLR_STDCALL *)(const float *, int32_t);
	using Func_godotsharp_packed_float64_array_new_mem_copy = godot_packed_float64_array(GD_CLR_STDCALL *)(const double *, int32_t);
	using Func_godotsharp_packed_vector2_array_new_mem_copy = godot_packed_vector2_array(GD_CLR_STDCALL *)(const Vector2 *, int32_t);
	using Func_godotsharp_packed_vector3_array_new_mem_copy = godot_packed_vector3_array(GD_CLR_STDCALL *)(const Vector3 *, int32_t);
	using Func_godotsharp_packed_color_array_new_mem_copy = godot_packed_color_array(GD_CLR_STDCALL *)(const Color *, int32_t);
	using Func_godotsharp_packed_string_array_add = void(GD_CLR_STDCALL *)(PackedStringArray *, const String *);
	using Func_godotsharp_callable_new_with_delegate = void(GD_CLR_STDCALL *)(GCHandleIntPtr, Callable *);
	using Func_godotsharp_callable_get_data_for_marshalling = bool(GD_CLR_STDCALL *)(const Callable *, GCHandleIntPtr *, Object **, StringName *);
	using Func_godotsharp_callable_call = godot_variant(GD_CLR_STDCALL *)(godot_callable *, const godot_variant **, const int32_t, godot_variant_call_error *);
	using Func_godotsharp_callable_call_deferred = void(GD_CLR_STDCALL *)(godot_callable *, const godot_variant **, const int32_t);
	using Func_godotsharp_method_bind_ptrcall = void(GD_CLR_STDCALL *)(godot_method_bind *, godot_object *, const void **, void *);
	using Func_godotsharp_method_bind_call = godot_variant(GD_CLR_STDCALL *)(godot_method_bind *, godot_object *, const godot_variant **, const int32_t, godot_variant_call_error *);
	using Func_godotsharp_variant_new_string_name = void(GD_CLR_STDCALL *)(godot_variant *, const godot_string_name *);
	using Func_godotsharp_variant_new_node_path = void(GD_CLR_STDCALL *)(godot_variant *, const godot_node_path *);
	using Func_godotsharp_variant_new_object = void(GD_CLR_STDCALL *)(godot_variant *, const godot_object *);
	using Func_godotsharp_variant_new_transform2d = void(GD_CLR_STDCALL *)(godot_variant *, const godot_transform2d *);
	using Func_godotsharp_variant_new_basis = void(GD_CLR_STDCALL *)(godot_variant *, const godot_basis *);
	using Func_godotsharp_variant_new_transform3d = void(GD_CLR_STDCALL *)(godot_variant *, const godot_transform3d *);
	using Func_godotsharp_variant_new_aabb = void(GD_CLR_STDCALL *)(godot_variant *, const godot_aabb *);
	using Func_godotsharp_variant_new_dictionary = void(GD_CLR_STDCALL *)(godot_variant *, const godot_dictionary *);
	using Func_godotsharp_variant_new_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_array *);
	using Func_godotsharp_variant_new_packed_byte_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_byte_array *);
	using Func_godotsharp_variant_new_packed_int32_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_int32_array *);
	using Func_godotsharp_variant_new_packed_int64_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_int64_array *);
	using Func_godotsharp_variant_new_packed_float32_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_float32_array *);
	using Func_godotsharp_variant_new_packed_float64_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_float64_array *);
	using Func_godotsharp_variant_new_packed_string_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_string_array *);
	using Func_godotsharp_variant_new_packed_vector2_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_vector2_array *);
	using Func_godotsharp_variant_new_packed_vector3_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_vector3_array *);
	using Func_godotsharp_variant_new_packed_color_array = void(GD_CLR_STDCALL *)(godot_variant *, const godot_packed_color_array *);
	using Func_godotsharp_variant_as_bool = godot_bool(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_int = godot_int(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_float = godot_float(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_string = godot_string(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_vector2 = godot_vector2(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_vector2i = godot_vector2i(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_rect2 = godot_rect2(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_rect2i = godot_rect2i(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_vector3 = godot_vector3(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_vector3i = godot_vector3i(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_transform2d = godot_transform2d(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_plane = godot_plane(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_quaternion = godot_quaternion(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_aabb = godot_aabb(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_basis = godot_basis(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_transform3d = godot_transform3d(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_color = godot_color(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_string_name = godot_string_name(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_node_path = godot_node_path(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_rid = godot_rid(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_callable = godot_callable(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_signal = godot_signal(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_dictionary = godot_dictionary(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_array = godot_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_byte_array = godot_packed_byte_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_int32_array = godot_packed_int32_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_int64_array = godot_packed_int64_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_float32_array = godot_packed_float32_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_float64_array = godot_packed_float64_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_string_array = godot_packed_string_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_vector2_array = godot_packed_vector2_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_vector3_array = godot_packed_vector3_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_as_packed_color_array = godot_packed_color_array(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_variant_equals = bool(GD_CLR_STDCALL *)(const godot_variant *, const godot_variant *);
	using Func_godotsharp_string_new_with_utf16_chars = void(GD_CLR_STDCALL *)(godot_string *, const char16_t *);
	using Func_godotsharp_string_name_new_copy = void(GD_CLR_STDCALL *)(godot_string_name *, const godot_string_name *);
	using Func_godotsharp_node_path_new_copy = void(GD_CLR_STDCALL *)(godot_node_path *, const godot_node_path *);
	using Func_godotsharp_array_new = void(GD_CLR_STDCALL *)(godot_array *);
	using Func_godotsharp_array_new_copy = void(GD_CLR_STDCALL *)(godot_array *, const godot_array *);
	using Func_godotsharp_array_ptrw = godot_variant *(GD_CLR_STDCALL *)(godot_array *);
	using Func_godotsharp_dictionary_new = void(GD_CLR_STDCALL *)(godot_dictionary *);
	using Func_godotsharp_dictionary_new_copy = void(GD_CLR_STDCALL *)(godot_dictionary *, const godot_dictionary *);
	using Func_godotsharp_packed_byte_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_byte_array *);
	using Func_godotsharp_packed_int32_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_int32_array *);
	using Func_godotsharp_packed_int64_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_int64_array *);
	using Func_godotsharp_packed_float32_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_float32_array *);
	using Func_godotsharp_packed_float64_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_float64_array *);
	using Func_godotsharp_packed_string_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_string_array *);
	using Func_godotsharp_packed_vector2_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_vector2_array *);
	using Func_godotsharp_packed_vector3_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_vector3_array *);
	using Func_godotsharp_packed_color_array_destroy = void(GD_CLR_STDCALL *)(godot_packed_color_array *);
	using Func_godotsharp_variant_destroy = void(GD_CLR_STDCALL *)(godot_variant *);
	using Func_godotsharp_string_destroy = void(GD_CLR_STDCALL *)(godot_string *);
	using Func_godotsharp_string_name_destroy = void(GD_CLR_STDCALL *)(godot_string_name *);
	using Func_godotsharp_node_path_destroy = void(GD_CLR_STDCALL *)(godot_node_path *);
	using Func_godotsharp_signal_destroy = void(GD_CLR_STDCALL *)(godot_signal *);
	using Func_godotsharp_callable_destroy = void(GD_CLR_STDCALL *)(godot_callable *);
	using Func_godotsharp_array_destroy = void(GD_CLR_STDCALL *)(godot_array *);
	using Func_godotsharp_dictionary_destroy = void(GD_CLR_STDCALL *)(godot_dictionary *);
	using Func_godotsharp_array_add = int32_t(GD_CLR_STDCALL *)(Array *, const Variant *);
	using Func_godotsharp_array_duplicate = void(GD_CLR_STDCALL *)(const Array *, bool, Array *);
	using Func_godotsharp_array_index_of = int32_t(GD_CLR_STDCALL *)(const Array *, const Variant *);
	using Func_godotsharp_array_insert = void(GD_CLR_STDCALL *)(Array *, int32_t, const Variant *);
	using Func_godotsharp_array_remove_at = void(GD_CLR_STDCALL *)(Array *, int32_t);
	using Func_godotsharp_array_resize = Error(GD_CLR_STDCALL *)(Array *, int32_t);
	using Func_godotsharp_array_shuffle = void(GD_CLR_STDCALL *)(Array *);
	using Func_godotsharp_array_to_string = void(GD_CLR_STDCALL *)(const Array *, String *);
	using Func_godotsharp_dictionary_try_get_value = bool(GD_CLR_STDCALL *)(const Dictionary *, const Variant *, Variant *);
	using Func_godotsharp_dictionary_set_value = void(GD_CLR_STDCALL *)(Dictionary *, const Variant *, const Variant *);
	using Func_godotsharp_dictionary_keys = void(GD_CLR_STDCALL *)(const Dictionary *, Array *);
	using Func_godotsharp_dictionary_values = void(GD_CLR_STDCALL *)(const Dictionary *, Array *);
	using Func_godotsharp_dictionary_count = int32_t(GD_CLR_STDCALL *)(const Dictionary *);
	using Func_godotsharp_dictionary_key_value_pair_at = void(GD_CLR_STDCALL *)(const Dictionary *, int32_t, Variant *, Variant *);
	using Func_godotsharp_dictionary_to_string = void(GD_CLR_STDCALL *)(const Dictionary *, String *);
	using Func_godotsharp_dictionary_add = void(GD_CLR_STDCALL *)(Dictionary *, const Variant *, const Variant *);
	using Func_godotsharp_dictionary_clear = void(GD_CLR_STDCALL *)(Dictionary *);
	using Func_godotsharp_dictionary_contains_key = bool(GD_CLR_STDCALL *)(const Dictionary *, const Variant *);
	using Func_godotsharp_dictionary_duplicate = void(GD_CLR_STDCALL *)(const Dictionary *, bool, Dictionary *);
	using Func_godotsharp_dictionary_remove_key = bool(GD_CLR_STDCALL *)(Dictionary *, const Variant *);
	using Func_godotsharp_string_md5_buffer = void(GD_CLR_STDCALL *)(const String *, PackedByteArray *);
	using Func_godotsharp_string_md5_text = void(GD_CLR_STDCALL *)(const String *, String *);
	using Func_godotsharp_string_rfind = int32_t(GD_CLR_STDCALL *)(const String *, const String *, int32_t);
	using Func_godotsharp_string_rfindn = int32_t(GD_CLR_STDCALL *)(const String *, const String *, int32_t);
	using Func_godotsharp_string_sha256_buffer = void(GD_CLR_STDCALL *)(const String *, PackedByteArray *);
	using Func_godotsharp_string_sha256_text = void(GD_CLR_STDCALL *)(const String *, String *);
	using Func_godotsharp_string_simplify_path = void(GD_CLR_STDCALL *)(const String *, String *);
	using Func_godotsharp_node_path_get_as_property_path = void(GD_CLR_STDCALL *)(const NodePath *, NodePath *);
	using Func_godotsharp_node_path_get_concatenated_subnames = void(GD_CLR_STDCALL *)(const NodePath *, String *);
	using Func_godotsharp_node_path_get_name = void(GD_CLR_STDCALL *)(const NodePath *, uint32_t, String *);
	using Func_godotsharp_node_path_get_name_count = int32_t(GD_CLR_STDCALL *)(const NodePath *);
	using Func_godotsharp_node_path_get_subname = void(GD_CLR_STDCALL *)(const NodePath *, uint32_t, String *);
	using Func_godotsharp_node_path_get_subname_count = int32_t(GD_CLR_STDCALL *)(const NodePath *);
	using Func_godotsharp_node_path_is_absolute = bool(GD_CLR_STDCALL *)(const NodePath *);
	using Func_godotsharp_randomize = void(GD_CLR_STDCALL *)();
	using Func_godotsharp_randi = uint32_t(GD_CLR_STDCALL *)();
	using Func_godotsharp_randf = float(GD_CLR_STDCALL *)();
	using Func_godotsharp_randi_range = int32_t(GD_CLR_STDCALL *)(int32_t, int32_t);
	using Func_godotsharp_randf_range = double(GD_CLR_STDCALL *)(double, double);
	using Func_godotsharp_randfn = double(GD_CLR_STDCALL *)(double, double);
	using Func_godotsharp_seed = void(GD_CLR_STDCALL *)(uint64_t);
	using Func_godotsharp_rand_from_seed = uint32_t(GD_CLR_STDCALL *)(uint64_t, uint64_t *);
	using Func_godotsharp_weakref = void(GD_CLR_STDCALL *)(Object *, Ref<RefCounted> *);
	using Func_godotsharp_str = void(GD_CLR_STDCALL *)(const godot_array *, godot_string *);
	using Func_godotsharp_print = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_printerr = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_printt = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_prints = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_printraw = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_pusherror = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_pushwarning = void(GD_CLR_STDCALL *)(const godot_string *);
	using Func_godotsharp_var2str = void(GD_CLR_STDCALL *)(const godot_variant *, godot_string *);
	using Func_godotsharp_str2var = void(GD_CLR_STDCALL *)(const godot_string *, godot_variant *);
	using Func_godotsharp_var2bytes = void(GD_CLR_STDCALL *)(const godot_variant *, bool, godot_packed_byte_array *);
	using Func_godotsharp_bytes2var = void(GD_CLR_STDCALL *)(const godot_packed_byte_array *, bool, godot_variant *);
	using Func_godotsharp_hash = int(GD_CLR_STDCALL *)(const godot_variant *);
	using Func_godotsharp_convert = void(GD_CLR_STDCALL *)(const godot_variant *, int32_t, godot_variant *);
	using Func_godotsharp_instance_from_id = Object *(GD_CLR_STDCALL *)(uint64_t);
	using Func_godotsharp_object_to_string = void(GD_CLR_STDCALL *)(Object *, godot_string *);

	Func_godotsharp_method_bind_get_method godotsharp_method_bind_get_method;
	Func_godotsharp_get_class_constructor godotsharp_get_class_constructor;
	Func_godotsharp_engine_get_singleton godotsharp_engine_get_singleton;
	Func_godotsharp_stack_info_vector_resize godotsharp_stack_info_vector_resize;
	Func_godotsharp_stack_info_vector_destroy godotsharp_stack_info_vector_destroy;
	Func_godotsharp_internal_script_debugger_send_error godotsharp_internal_script_debugger_send_error;
	Func_godotsharp_internal_script_debugger_is_active godotsharp_internal_script_debugger_is_active;
	Func_godotsharp_internal_object_get_associated_gchandle godotsharp_internal_object_get_associated_gchandle;
	Func_godotsharp_internal_object_disposed godotsharp_internal_object_disposed;
	Func_godotsharp_internal_refcounted_disposed godotsharp_internal_refcounted_disposed;
	Func_godotsharp_internal_object_connect_event_signal godotsharp_internal_object_connect_event_signal;
	Func_godotsharp_internal_signal_awaiter_connect godotsharp_internal_signal_awaiter_connect;
	Func_godotsharp_internal_unmanaged_get_script_instance_managed godotsharp_internal_unmanaged_get_script_instance_managed;
	Func_godotsharp_internal_unmanaged_get_instance_binding_managed godotsharp_internal_unmanaged_get_instance_binding_managed;
	Func_godotsharp_internal_unmanaged_instance_binding_create_managed godotsharp_internal_unmanaged_instance_binding_create_managed;
	Func_godotsharp_internal_tie_native_managed_to_unmanaged godotsharp_internal_tie_native_managed_to_unmanaged;
	Func_godotsharp_internal_tie_user_managed_to_unmanaged godotsharp_internal_tie_user_managed_to_unmanaged;
	Func_godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup;
	Func_godotsharp_internal_new_csharp_script godotsharp_internal_new_csharp_script;
	Func_godotsharp_internal_reload_registered_script godotsharp_internal_reload_registered_script;
	Func_godotsharp_array_filter_godot_objects_by_native godotsharp_array_filter_godot_objects_by_native;
	Func_godotsharp_array_filter_godot_objects_by_non_native godotsharp_array_filter_godot_objects_by_non_native;
	Func_godotsharp_ref_new_from_ref_counted_ptr godotsharp_ref_new_from_ref_counted_ptr;
	Func_godotsharp_ref_destroy godotsharp_ref_destroy;
	Func_godotsharp_string_name_new_from_string godotsharp_string_name_new_from_string;
	Func_godotsharp_node_path_new_from_string godotsharp_node_path_new_from_string;
	Func_godotsharp_string_name_as_string godotsharp_string_name_as_string;
	Func_godotsharp_node_path_as_string godotsharp_node_path_as_string;
	Func_godotsharp_packed_byte_array_new_mem_copy godotsharp_packed_byte_array_new_mem_copy;
	Func_godotsharp_packed_int32_array_new_mem_copy godotsharp_packed_int32_array_new_mem_copy;
	Func_godotsharp_packed_int64_array_new_mem_copy godotsharp_packed_int64_array_new_mem_copy;
	Func_godotsharp_packed_float32_array_new_mem_copy godotsharp_packed_float32_array_new_mem_copy;
	Func_godotsharp_packed_float64_array_new_mem_copy godotsharp_packed_float64_array_new_mem_copy;
	Func_godotsharp_packed_vector2_array_new_mem_copy godotsharp_packed_vector2_array_new_mem_copy;
	Func_godotsharp_packed_vector3_array_new_mem_copy godotsharp_packed_vector3_array_new_mem_copy;
	Func_godotsharp_packed_color_array_new_mem_copy godotsharp_packed_color_array_new_mem_copy;
	Func_godotsharp_packed_string_array_add godotsharp_packed_string_array_add;
	Func_godotsharp_callable_new_with_delegate godotsharp_callable_new_with_delegate;
	Func_godotsharp_callable_get_data_for_marshalling godotsharp_callable_get_data_for_marshalling;
	Func_godotsharp_callable_call godotsharp_callable_call;
	Func_godotsharp_callable_call_deferred godotsharp_callable_call_deferred;
	Func_godotsharp_method_bind_ptrcall godotsharp_method_bind_ptrcall;
	Func_godotsharp_method_bind_call godotsharp_method_bind_call;
	Func_godotsharp_variant_new_string_name godotsharp_variant_new_string_name;
	Func_godotsharp_variant_new_node_path godotsharp_variant_new_node_path;
	Func_godotsharp_variant_new_object godotsharp_variant_new_object;
	Func_godotsharp_variant_new_transform2d godotsharp_variant_new_transform2d;
	Func_godotsharp_variant_new_basis godotsharp_variant_new_basis;
	Func_godotsharp_variant_new_transform3d godotsharp_variant_new_transform3d;
	Func_godotsharp_variant_new_aabb godotsharp_variant_new_aabb;
	Func_godotsharp_variant_new_dictionary godotsharp_variant_new_dictionary;
	Func_godotsharp_variant_new_array godotsharp_variant_new_array;
	Func_godotsharp_variant_new_packed_byte_array godotsharp_variant_new_packed_byte_array;
	Func_godotsharp_variant_new_packed_int32_array godotsharp_variant_new_packed_int32_array;
	Func_godotsharp_variant_new_packed_int64_array godotsharp_variant_new_packed_int64_array;
	Func_godotsharp_variant_new_packed_float32_array godotsharp_variant_new_packed_float32_array;
	Func_godotsharp_variant_new_packed_float64_array godotsharp_variant_new_packed_float64_array;
	Func_godotsharp_variant_new_packed_string_array godotsharp_variant_new_packed_string_array;
	Func_godotsharp_variant_new_packed_vector2_array godotsharp_variant_new_packed_vector2_array;
	Func_godotsharp_variant_new_packed_vector3_array godotsharp_variant_new_packed_vector3_array;
	Func_godotsharp_variant_new_packed_color_array godotsharp_variant_new_packed_color_array;
	Func_godotsharp_variant_as_bool godotsharp_variant_as_bool;
	Func_godotsharp_variant_as_int godotsharp_variant_as_int;
	Func_godotsharp_variant_as_float godotsharp_variant_as_float;
	Func_godotsharp_variant_as_string godotsharp_variant_as_string;
	Func_godotsharp_variant_as_vector2 godotsharp_variant_as_vector2;
	Func_godotsharp_variant_as_vector2i godotsharp_variant_as_vector2i;
	Func_godotsharp_variant_as_rect2 godotsharp_variant_as_rect2;
	Func_godotsharp_variant_as_rect2i godotsharp_variant_as_rect2i;
	Func_godotsharp_variant_as_vector3 godotsharp_variant_as_vector3;
	Func_godotsharp_variant_as_vector3i godotsharp_variant_as_vector3i;
	Func_godotsharp_variant_as_transform2d godotsharp_variant_as_transform2d;
	Func_godotsharp_variant_as_plane godotsharp_variant_as_plane;
	Func_godotsharp_variant_as_quaternion godotsharp_variant_as_quaternion;
	Func_godotsharp_variant_as_aabb godotsharp_variant_as_aabb;
	Func_godotsharp_variant_as_basis godotsharp_variant_as_basis;
	Func_godotsharp_variant_as_transform3d godotsharp_variant_as_transform3d;
	Func_godotsharp_variant_as_color godotsharp_variant_as_color;
	Func_godotsharp_variant_as_string_name godotsharp_variant_as_string_name;
	Func_godotsharp_variant_as_node_path godotsharp_variant_as_node_path;
	Func_godotsharp_variant_as_rid godotsharp_variant_as_rid;
	Func_godotsharp_variant_as_callable godotsharp_variant_as_callable;
	Func_godotsharp_variant_as_signal godotsharp_variant_as_signal;
	Func_godotsharp_variant_as_dictionary godotsharp_variant_as_dictionary;
	Func_godotsharp_variant_as_array godotsharp_variant_as_array;
	Func_godotsharp_variant_as_packed_byte_array godotsharp_variant_as_packed_byte_array;
	Func_godotsharp_variant_as_packed_int32_array godotsharp_variant_as_packed_int32_array;
	Func_godotsharp_variant_as_packed_int64_array godotsharp_variant_as_packed_int64_array;
	Func_godotsharp_variant_as_packed_float32_array godotsharp_variant_as_packed_float32_array;
	Func_godotsharp_variant_as_packed_float64_array godotsharp_variant_as_packed_float64_array;
	Func_godotsharp_variant_as_packed_string_array godotsharp_variant_as_packed_string_array;
	Func_godotsharp_variant_as_packed_vector2_array godotsharp_variant_as_packed_vector2_array;
	Func_godotsharp_variant_as_packed_vector3_array godotsharp_variant_as_packed_vector3_array;
	Func_godotsharp_variant_as_packed_color_array godotsharp_variant_as_packed_color_array;
	Func_godotsharp_variant_equals godotsharp_variant_equals;
	Func_godotsharp_string_new_with_utf16_chars godotsharp_string_new_with_utf16_chars;
	Func_godotsharp_string_name_new_copy godotsharp_string_name_new_copy;
	Func_godotsharp_node_path_new_copy godotsharp_node_path_new_copy;
	Func_godotsharp_array_new godotsharp_array_new;
	Func_godotsharp_array_new_copy godotsharp_array_new_copy;
	Func_godotsharp_array_ptrw godotsharp_array_ptrw;
	Func_godotsharp_dictionary_new godotsharp_dictionary_new;
	Func_godotsharp_dictionary_new_copy godotsharp_dictionary_new_copy;
	Func_godotsharp_packed_byte_array_destroy godotsharp_packed_byte_array_destroy;
	Func_godotsharp_packed_int32_array_destroy godotsharp_packed_int32_array_destroy;
	Func_godotsharp_packed_int64_array_destroy godotsharp_packed_int64_array_destroy;
	Func_godotsharp_packed_float32_array_destroy godotsharp_packed_float32_array_destroy;
	Func_godotsharp_packed_float64_array_destroy godotsharp_packed_float64_array_destroy;
	Func_godotsharp_packed_string_array_destroy godotsharp_packed_string_array_destroy;
	Func_godotsharp_packed_vector2_array_destroy godotsharp_packed_vector2_array_destroy;
	Func_godotsharp_packed_vector3_array_destroy godotsharp_packed_vector3_array_destroy;
	Func_godotsharp_packed_color_array_destroy godotsharp_packed_color_array_destroy;
	Func_godotsharp_variant_destroy godotsharp_variant_destroy;
	Func_godotsharp_string_destroy godotsharp_string_destroy;
	Func_godotsharp_string_name_destroy godotsharp_string_name_destroy;
	Func_godotsharp_node_path_destroy godotsharp_node_path_destroy;
	Func_godotsharp_signal_destroy godotsharp_signal_destroy;
	Func_godotsharp_callable_destroy godotsharp_callable_destroy;
	Func_godotsharp_array_destroy godotsharp_array_destroy;
	Func_godotsharp_dictionary_destroy godotsharp_dictionary_destroy;
	Func_godotsharp_array_add godotsharp_array_add;
	Func_godotsharp_array_duplicate godotsharp_array_duplicate;
	Func_godotsharp_array_index_of godotsharp_array_index_of;
	Func_godotsharp_array_insert godotsharp_array_insert;
	Func_godotsharp_array_remove_at godotsharp_array_remove_at;
	Func_godotsharp_array_resize godotsharp_array_resize;
	Func_godotsharp_array_shuffle godotsharp_array_shuffle;
	Func_godotsharp_array_to_string godotsharp_array_to_string;
	Func_godotsharp_dictionary_try_get_value godotsharp_dictionary_try_get_value;
	Func_godotsharp_dictionary_set_value godotsharp_dictionary_set_value;
	Func_godotsharp_dictionary_keys godotsharp_dictionary_keys;
	Func_godotsharp_dictionary_values godotsharp_dictionary_values;
	Func_godotsharp_dictionary_count godotsharp_dictionary_count;
	Func_godotsharp_dictionary_key_value_pair_at godotsharp_dictionary_key_value_pair_at;
	Func_godotsharp_dictionary_to_string godotsharp_dictionary_to_string;
	Func_godotsharp_dictionary_add godotsharp_dictionary_add;
	Func_godotsharp_dictionary_clear godotsharp_dictionary_clear;
	Func_godotsharp_dictionary_contains_key godotsharp_dictionary_contains_key;
	Func_godotsharp_dictionary_duplicate godotsharp_dictionary_duplicate;
	Func_godotsharp_dictionary_remove_key godotsharp_dictionary_remove_key;
	Func_godotsharp_string_md5_buffer godotsharp_string_md5_buffer;
	Func_godotsharp_string_md5_text godotsharp_string_md5_text;
	Func_godotsharp_string_rfind godotsharp_string_rfind;
	Func_godotsharp_string_rfindn godotsharp_string_rfindn;
	Func_godotsharp_string_sha256_buffer godotsharp_string_sha256_buffer;
	Func_godotsharp_string_sha256_text godotsharp_string_sha256_text;
	Func_godotsharp_string_simplify_path godotsharp_string_simplify_path;
	Func_godotsharp_node_path_get_as_property_path godotsharp_node_path_get_as_property_path;
	Func_godotsharp_node_path_get_concatenated_subnames godotsharp_node_path_get_concatenated_subnames;
	Func_godotsharp_node_path_get_name godotsharp_node_path_get_name;
	Func_godotsharp_node_path_get_name_count godotsharp_node_path_get_name_count;
	Func_godotsharp_node_path_get_subname godotsharp_node_path_get_subname;
	Func_godotsharp_node_path_get_subname_count godotsharp_node_path_get_subname_count;
	Func_godotsharp_node_path_is_absolute godotsharp_node_path_is_absolute;
	Func_godotsharp_randomize godotsharp_randomize;
	Func_godotsharp_randi godotsharp_randi;
	Func_godotsharp_randf godotsharp_randf;
	Func_godotsharp_randi_range godotsharp_randi_range;
	Func_godotsharp_randf_range godotsharp_randf_range;
	Func_godotsharp_randfn godotsharp_randfn;
	Func_godotsharp_seed godotsharp_seed;
	Func_godotsharp_rand_from_seed godotsharp_rand_from_seed;
	Func_godotsharp_weakref godotsharp_weakref;
	Func_godotsharp_str godotsharp_str;
	Func_godotsharp_print godotsharp_print;
	Func_godotsharp_printerr godotsharp_printerr;
	Func_godotsharp_printt godotsharp_printt;
	Func_godotsharp_prints godotsharp_prints;
	Func_godotsharp_printraw godotsharp_printraw;
	Func_godotsharp_pusherror godotsharp_pusherror;
	Func_godotsharp_pushwarning godotsharp_pushwarning;
	Func_godotsharp_var2str godotsharp_var2str;
	Func_godotsharp_str2var godotsharp_str2var;
	Func_godotsharp_var2bytes godotsharp_var2bytes;
	Func_godotsharp_bytes2var godotsharp_bytes2var;
	Func_godotsharp_hash godotsharp_hash;
	Func_godotsharp_convert godotsharp_convert;
	Func_godotsharp_instance_from_id godotsharp_instance_from_id;
	Func_godotsharp_object_to_string godotsharp_object_to_string;

	static UnmanagedCallbacks create();
};

#endif // RUNTIME_INTEROP_H
