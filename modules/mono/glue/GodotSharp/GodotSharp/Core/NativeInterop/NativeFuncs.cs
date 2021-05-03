using System;
using System.Runtime.InteropServices;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
#if !NET
    // This improves P/Invoke performance.
    // The attribute is not available with .NET Core and it's not needed there.
    [System.Security.SuppressUnmanagedCodeSecurity]
#endif
    internal static unsafe partial class NativeFuncs
    {
        private const string GodotDllName = "__Internal";

        // Custom functions

        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_method_bind_get_method(ref godot_string_name p_classname, char* p_methodname);

#if NET
        [DllImport(GodotDllName)]
        public static extern delegate* unmanaged<IntPtr> godotsharp_get_class_constructor(ref godot_string_name p_classname);
#else
        // Workaround until we switch to .NET 5/6
        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_get_class_constructor(ref godot_string_name p_classname);

        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_invoke_class_constructor(IntPtr p_creation_func);
#endif

        [DllImport(GodotDllName)]
        public static extern IntPtr godotsharp_engine_get_singleton(godot_string* p_name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_ref_destroy(ref godot_ref p_instance);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_new_from_string(godot_string_name* dest, godot_string* name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_new_from_string(godot_node_path* dest, godot_string* name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_as_string(godot_string* r_dest, godot_string_name* p_name);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_as_string(godot_string* r_dest, godot_node_path* p_np);

        [DllImport(GodotDllName)]
        public static extern godot_packed_byte_array godotsharp_packed_byte_array_new_mem_copy(byte* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int32_array godotsharp_packed_int32_array_new_mem_copy(int* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int64_array godotsharp_packed_int64_array_new_mem_copy(long* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float32_array godotsharp_packed_float32_array_new_mem_copy(float* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float64_array godotsharp_packed_float64_array_new_mem_copy(double* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector2_array godotsharp_packed_vector2_array_new_mem_copy(Vector2* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector3_array godotsharp_packed_vector3_array_new_mem_copy(Vector3* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern godot_packed_color_array godotsharp_packed_color_array_new_mem_copy(Color* p_src, int p_length);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_packed_string_array_add(godot_packed_string_array* r_dest, godot_string* p_element);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_callable_new_with_delegate(IntPtr p_delegate_handle, godot_callable* r_callable);

        [DllImport(GodotDllName)]
        public static extern bool godotsharp_callable_get_data_for_marshalling(godot_callable* p_callable, IntPtr* r_delegate_handle, IntPtr* r_object, godot_string_name* r_name);

        // GDNative functions

        // gdnative.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_method_bind_ptrcall(IntPtr p_method_bind, IntPtr p_instance, void** p_args, void* p_ret);

        [DllImport(GodotDllName)]
        public static extern godot_variant godotsharp_method_bind_call(IntPtr p_method_bind, IntPtr p_instance, godot_variant** p_args, int p_arg_count, godot_variant_call_error* p_call_error);

        // variant.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_string_name(godot_variant* r_dest, godot_string_name* p_s);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_node_path(godot_variant* r_dest, godot_node_path* p_np);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_object(godot_variant* r_dest, IntPtr p_obj);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_transform2d(godot_variant* r_dest, Transform2D* p_t2d);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_basis(godot_variant* r_dest, Basis* p_basis);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_transform3d(godot_variant* r_dest, Transform3D* p_trans);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_aabb(godot_variant* r_dest, AABB* p_aabb);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_dictionary(godot_variant* r_dest, godot_dictionary* p_dict);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_array(godot_variant* r_dest, godot_array* p_arr);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_byte_array(godot_variant* r_dest, godot_packed_byte_array* p_pba);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_int32_array(godot_variant* r_dest, godot_packed_int32_array* p_pia);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_int64_array(godot_variant* r_dest, godot_packed_int64_array* p_pia);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_float32_array(godot_variant* r_dest, godot_packed_float32_array* p_pra);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_float64_array(godot_variant* r_dest, godot_packed_float64_array* p_pra);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_string_array(godot_variant* r_dest, godot_packed_string_array* p_psa);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_vector2_array(godot_variant* r_dest, godot_packed_vector2_array* p_pv2a);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_vector3_array(godot_variant* r_dest, godot_packed_vector3_array* p_pv3a);

        [DllImport(GodotDllName)]
        public static extern void godotsharp_variant_new_packed_color_array(godot_variant* r_dest, godot_packed_color_array* p_pca);

        [DllImport(GodotDllName)]
        public static extern bool godotsharp_variant_as_bool(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Int64 godotsharp_variant_as_int(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern double godotsharp_variant_as_float(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_string godotsharp_variant_as_string(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Vector2 godotsharp_variant_as_vector2(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Vector2i godotsharp_variant_as_vector2i(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Rect2 godotsharp_variant_as_rect2(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Rect2i godotsharp_variant_as_rect2i(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Vector3 godotsharp_variant_as_vector3(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Vector3i godotsharp_variant_as_vector3i(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Transform2D godotsharp_variant_as_transform2d(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Plane godotsharp_variant_as_plane(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Quaternion godotsharp_variant_as_quaternion(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern AABB godotsharp_variant_as_aabb(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Basis godotsharp_variant_as_basis(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Transform3D godotsharp_variant_as_transform3d(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern Color godotsharp_variant_as_color(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_string_name godotsharp_variant_as_string_name(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_node_path godotsharp_variant_as_node_path(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern RID godotsharp_variant_as_rid(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_callable godotsharp_variant_as_callable(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_signal godotsharp_variant_as_signal(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_dictionary godotsharp_variant_as_dictionary(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_array godotsharp_variant_as_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_byte_array godotsharp_variant_as_packed_byte_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int32_array godotsharp_variant_as_packed_int32_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_int64_array godotsharp_variant_as_packed_int64_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float32_array godotsharp_variant_as_packed_float32_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_float64_array godotsharp_variant_as_packed_float64_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_string_array godotsharp_variant_as_packed_string_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector2_array godotsharp_variant_as_packed_vector2_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_vector3_array godotsharp_variant_as_packed_vector3_array(godot_variant* p_self);

        [DllImport(GodotDllName)]
        public static extern godot_packed_color_array godotsharp_variant_as_packed_color_array(godot_variant* p_self);

        // string.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_new_with_utf16_chars(godot_string* r_dest, char* p_contents);

        // string_name.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_string_name_new_copy(godot_string_name* r_dest, godot_string_name* p_src);

        // node_path.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_node_path_new_copy(godot_node_path* r_dest, godot_node_path* p_src);

        // array.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_array_new_copy(godot_array* r_dest, godot_array* p_src);

        // dictionary.h

        [DllImport(GodotDllName)]
        public static extern void godotsharp_dictionary_new_copy(godot_dictionary* r_dest, godot_dictionary* p_src);

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
    }
}
