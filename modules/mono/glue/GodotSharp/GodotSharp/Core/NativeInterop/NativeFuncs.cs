using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    /*
     * TODO:
     * P/Invoke pins by-ref parameters in case the managed memory is moved.
     * That's not needed here since we use "ref structs" which are stack-only.
     * Unfortunately, the runtime is not smart enough and pins them anyway.
     * We want to avoid pinning, so we must wrap these DllImports in methods
     * that reinterpret refs and pointers with CustomUnsafe.AsPointer/AsRef.
     * I wish such unnecessary boilerplate wasn't needed...
     */

    [SuppressMessage("Interoperability", "CA1401",
        MessageId = "P/Invokes should not be visible" /* Meh. What are you gonna do about it? */)]
    public static unsafe partial class NativeFuncs
    {
        private const string GodotDllName = "__Internal";

        // Custom functions

        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_method_bind_get_method(in godot_string_name p_classname,
            in godot_string_name p_methodname);

        [DllImport(GodotDllName)]
        public static extern delegate* unmanaged<IntPtr> godotsharp_get_class_constructor(
            in godot_string_name p_classname);

        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_engine_get_singleton(in godot_string p_name);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_object_disposed(IntPtr ptr);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_refcounted_disposed(IntPtr ptr, godot_bool isFinalizer);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_object_connect_event_signal(IntPtr obj,
            in godot_string_name eventSignal);

        [DllImport(GodotDllName)]
        internal static extern Error godotsharp_internal_signal_awaiter_connect(IntPtr source,
            in godot_string_name signal,
            IntPtr target, IntPtr awaiterHandlePtr);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_tie_native_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, in godot_string_name nativeName, godot_bool refCounted);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_tie_user_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, IntPtr scriptPtr, godot_bool refCounted);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup(
            IntPtr gcHandleIntPtr, IntPtr unmanaged);

        [DllImport(GodotDllName)]
        internal static extern IntPtr godotsharp_internal_unmanaged_get_script_instance_managed(IntPtr p_unmanaged,
            out godot_bool r_has_cs_script_instance);

        [DllImport(GodotDllName)]
        internal static extern IntPtr godotsharp_internal_unmanaged_get_instance_binding_managed(IntPtr p_unmanaged);

        [DllImport(GodotDllName)]
        internal static extern IntPtr godotsharp_internal_unmanaged_instance_binding_create_managed(IntPtr p_unmanaged,
            IntPtr oldGCHandlePtr);

        [DllImport(GodotDllName)]
        internal static extern IntPtr godotsharp_internal_new_csharp_script();

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_array_filter_godot_objects_by_native(in godot_string_name p_native_name,
            in godot_array p_input, out godot_array r_output);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_array_filter_godot_objects_by_non_native(in godot_array p_input,
            out godot_array r_output);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_ref_destroy(ref godot_ref p_instance);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_new_from_string(out godot_string_name r_dest,
            in godot_string p_name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_new_from_string(out godot_node_path r_dest,
            in godot_string p_name);

        [DllImport(GodotDllName)]
        public static extern void
            godotsharp_string_name_as_string(out godot_string r_dest, in godot_string_name p_name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_as_string(out godot_string r_dest, in godot_node_path p_np);

        [DllImport(GodotDllName)]
        public static extern godot_packed_byte_array godotsharp_packed_byte_array_new_mem_copy(byte* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int32_array godotsharp_packed_int32_array_new_mem_copy(int* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int64_array godotsharp_packed_int64_array_new_mem_copy(long* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float32_array godotsharp_packed_float32_array_new_mem_copy(float* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float64_array godotsharp_packed_float64_array_new_mem_copy(double* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector2_array godotsharp_packed_vector2_array_new_mem_copy(Vector2* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector3_array godotsharp_packed_vector3_array_new_mem_copy(Vector3* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_color_array godotsharp_packed_color_array_new_mem_copy(Color* p_src,
            int p_length);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_string_array_add(ref godot_packed_string_array r_dest,
            in godot_string p_element);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_callable_new_with_delegate(IntPtr p_delegate_handle,
            out godot_callable r_callable);

        [DllImport(GodotDllName)]
        internal static extern godot_bool godotsharp_callable_get_data_for_marshalling(in godot_callable p_callable,
            out IntPtr r_delegate_handle, out IntPtr r_object, out godot_string_name r_name);

        [DllImport(GodotDllName)]
        internal static extern godot_variant godotsharp_callable_call(in godot_callable p_callable,
            godot_variant** p_args, int p_arg_count, out godot_variant_call_error p_call_error);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_callable_call_deferred(in godot_callable p_callable,
            godot_variant** p_args, int p_arg_count);

        // GDNative functions

        // gdnative.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_method_bind_ptrcall(IntPtr p_method_bind, IntPtr p_instance, void** p_args,
            void* p_ret);

        [DllImport(GodotDllName)]
        public static extern godot_variant godotsharp_method_bind_call(IntPtr p_method_bind, IntPtr p_instance,
            godot_variant** p_args, int p_arg_count, out godot_variant_call_error p_call_error);

        // variant.h

        [DllImport(GodotDllName)]
        public static extern void
            godotsharp_variant_new_string_name(out godot_variant r_dest, in godot_string_name p_s);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_node_path(out godot_variant r_dest, in godot_node_path p_np);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_object(out godot_variant r_dest, IntPtr p_obj);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_transform2d(out godot_variant r_dest, in Transform2D p_t2d);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_basis(out godot_variant r_dest, in Basis p_basis);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_transform3d(out godot_variant r_dest, in Transform3D p_trans);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_aabb(out godot_variant r_dest, in AABB p_aabb);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_dictionary(out godot_variant r_dest,
            in godot_dictionary p_dict);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_array(out godot_variant r_dest, in godot_array p_arr);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_byte_array(out godot_variant r_dest,
            in godot_packed_byte_array p_pba);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_int32_array(out godot_variant r_dest,
            in godot_packed_int32_array p_pia);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_int64_array(out godot_variant r_dest,
            in godot_packed_int64_array p_pia);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_float32_array(out godot_variant r_dest,
            in godot_packed_float32_array p_pra);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_float64_array(out godot_variant r_dest,
            in godot_packed_float64_array p_pra);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_string_array(out godot_variant r_dest,
            in godot_packed_string_array p_psa);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_vector2_array(out godot_variant r_dest,
            in godot_packed_vector2_array p_pv2a);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_vector3_array(out godot_variant r_dest,
            in godot_packed_vector3_array p_pv3a);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_color_array(out godot_variant r_dest,
            in godot_packed_color_array p_pca);

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_variant_as_bool(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Int64 godotsharp_variant_as_int(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern double godotsharp_variant_as_float(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_string godotsharp_variant_as_string(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Vector2 godotsharp_variant_as_vector2(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Vector2i godotsharp_variant_as_vector2i(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Rect2 godotsharp_variant_as_rect2(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Rect2i godotsharp_variant_as_rect2i(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Vector3 godotsharp_variant_as_vector3(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Vector3i godotsharp_variant_as_vector3i(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Transform2D godotsharp_variant_as_transform2d(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Plane godotsharp_variant_as_plane(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Quaternion godotsharp_variant_as_quaternion(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern AABB godotsharp_variant_as_aabb(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Basis godotsharp_variant_as_basis(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Transform3D godotsharp_variant_as_transform3d(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern Color godotsharp_variant_as_color(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_string_name godotsharp_variant_as_string_name(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_node_path godotsharp_variant_as_node_path(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern RID godotsharp_variant_as_rid(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_callable godotsharp_variant_as_callable(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_signal godotsharp_variant_as_signal(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_dictionary godotsharp_variant_as_dictionary(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_array godotsharp_variant_as_array(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_byte_array godotsharp_variant_as_packed_byte_array(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int32_array godotsharp_variant_as_packed_int32_array(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int64_array godotsharp_variant_as_packed_int64_array(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float32_array godotsharp_variant_as_packed_float32_array(
            in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float64_array godotsharp_variant_as_packed_float64_array(
            in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_string_array godotsharp_variant_as_packed_string_array(
            in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector2_array godotsharp_variant_as_packed_vector2_array(
            in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector3_array godotsharp_variant_as_packed_vector3_array(
            in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_color_array godotsharp_variant_as_packed_color_array(in godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_variant_equals(in godot_variant p_a, in godot_variant p_b);

        // string.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_new_with_utf16_chars(out godot_string r_dest, char* p_contents);

        // string_name.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_new_copy(out godot_string_name r_dest,
            in godot_string_name p_src);

        // node_path.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_new_copy(out godot_node_path r_dest, in godot_node_path p_src);

        // array.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_new(out godot_array r_dest);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_new_copy(out godot_array r_dest, in godot_array p_src);

        [DllImport(GodotDllName)]
        public static extern godot_variant* godotsharp_array_ptrw(ref godot_array p_self);

        // dictionary.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_new(out godot_dictionary r_dest);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_new_copy(out godot_dictionary r_dest,
            in godot_dictionary p_src);

        // destroy functions

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_byte_array_destroy(ref godot_packed_byte_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_int32_array_destroy(ref godot_packed_int32_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_int64_array_destroy(ref godot_packed_int64_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_float32_array_destroy(ref godot_packed_float32_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_float64_array_destroy(ref godot_packed_float64_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_string_array_destroy(ref godot_packed_string_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_vector2_array_destroy(ref godot_packed_vector2_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_vector3_array_destroy(ref godot_packed_vector3_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_color_array_destroy(ref godot_packed_color_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_destroy(ref godot_variant p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_destroy(ref godot_string p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_destroy(ref godot_string_name p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_destroy(ref godot_node_path p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_signal_destroy(ref godot_signal p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_callable_destroy(ref godot_callable p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_destroy(ref godot_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_destroy(ref godot_dictionary p_self);

        // Array

        [DllImport(GodotDllName)]
        public static extern int godotsharp_array_add(ref godot_array p_self, in godot_variant p_item);

        [DllImport(GodotDllName)]
        public static extern void
            godotsharp_array_duplicate(ref godot_array p_self, godot_bool p_deep, out godot_array r_dest);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_array_index_of(ref godot_array p_self, in godot_variant p_item);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_insert(ref godot_array p_self, int p_index, in godot_variant p_item);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_remove_at(ref godot_array p_self, int p_index);

        [DllImport(GodotDllName)]
        public static extern Error godotsharp_array_resize(ref godot_array p_self, int p_new_size);

        [DllImport(GodotDllName)]
        public static extern Error godotsharp_array_shuffle(ref godot_array p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_to_string(ref godot_array p_self, out godot_string r_str);

        // Dictionary

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_dictionary_try_get_value(ref godot_dictionary p_self,
            in godot_variant p_key,
            out godot_variant r_value);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_set_value(ref godot_dictionary p_self, in godot_variant p_key,
            in godot_variant p_value);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_keys(ref godot_dictionary p_self, out godot_array r_dest);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_values(ref godot_dictionary p_self, out godot_array r_dest);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_dictionary_count(ref godot_dictionary p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_key_value_pair_at(ref godot_dictionary p_self, int p_index,
            out godot_variant r_key, out godot_variant r_value);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_add(ref godot_dictionary p_self, in godot_variant p_key,
            in godot_variant p_value);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_clear(ref godot_dictionary p_self);

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_dictionary_contains_key(ref godot_dictionary p_self,
            in godot_variant p_key);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_duplicate(ref godot_dictionary p_self, godot_bool p_deep,
            out godot_dictionary r_dest);

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_dictionary_remove_key(ref godot_dictionary p_self,
            in godot_variant p_key);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_to_string(ref godot_dictionary p_self, out godot_string r_str);

        // StringExtensions

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_md5_buffer(in godot_string p_self,
            out godot_packed_byte_array r_md5_buffer);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_md5_text(in godot_string p_self, out godot_string r_md5_text);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_string_rfind(in godot_string p_self, in godot_string p_what, int p_from);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_string_rfindn(in godot_string p_self, in godot_string p_what, int p_from);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_sha256_buffer(in godot_string p_self,
            out godot_packed_byte_array r_sha256_buffer);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_sha256_text(in godot_string p_self,
            out godot_string r_sha256_text);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_simplify_path(in godot_string p_self,
            out godot_string r_simplified_path);

        // NodePath

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_get_as_property_path(in godot_node_path p_self,
            ref godot_node_path r_dest);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_get_concatenated_subnames(in godot_node_path p_self,
            out godot_string r_subnames);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_get_name(in godot_node_path p_self, int p_idx,
            out godot_string r_name);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_node_path_get_name_count(in godot_node_path p_self);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_get_subname(in godot_node_path p_self, int p_idx,
            out godot_string r_subname);

        [DllImport(GodotDllName)]
        public static extern int godotsharp_node_path_get_subname_count(in godot_node_path p_self);

        [DllImport(GodotDllName)]
        public static extern godot_bool godotsharp_node_path_is_absolute(in godot_node_path p_self);

        // GD, etc

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_bytes2var(in godot_packed_byte_array p_bytes,
            godot_bool p_allow_objects,
            out godot_variant r_ret);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_convert(in godot_variant p_what, Variant.Type p_type,
            out godot_variant r_ret);

        [DllImport(GodotDllName)]
        internal static extern int godotsharp_hash(in godot_variant p_var);

        [DllImport(GodotDllName)]
        internal static extern IntPtr godotsharp_instance_from_id(ulong p_instance_id);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_print(in godot_string p_what);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_printerr(in godot_string p_what);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_printraw(in godot_string p_what);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_prints(in godot_string p_what);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_printt(in godot_string p_what);

        [DllImport(GodotDllName)]
        internal static extern float godotsharp_randf();

        [DllImport(GodotDllName)]
        internal static extern uint godotsharp_randi();

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_randomize();

        [DllImport(GodotDllName)]
        internal static extern double godotsharp_randf_range(double from, double to);

        [DllImport(GodotDllName)]
        internal static extern double godotsharp_randfn(double mean, double deviation);

        [DllImport(GodotDllName)]
        internal static extern int godotsharp_randi_range(int from, int to);

        [DllImport(GodotDllName)]
        internal static extern uint godotsharp_rand_from_seed(ulong seed, out ulong newSeed);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_seed(ulong seed);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_weakref(IntPtr p_obj, out godot_ref r_weak_ref);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_str(in godot_array p_what, out godot_string r_ret);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_str2var(in godot_string p_str, out godot_variant r_ret);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_var2bytes(in godot_variant p_what, godot_bool p_full_objects,
            out godot_packed_byte_array r_bytes);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_var2str(in godot_variant p_var, out godot_string r_ret);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_pusherror(in godot_string p_str);

        [DllImport(GodotDllName)]
        internal static extern void godotsharp_pushwarning(in godot_string p_str);

        // Object

        [DllImport(GodotDllName)]
        public static extern void godotsharp_object_to_string(IntPtr ptr, out godot_string r_str);
    }
}
