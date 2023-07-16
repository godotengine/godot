using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Godot.SourceGenerators.Internal;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    /*
     * IMPORTANT:
     * The order of the methods defined in NativeFuncs must match the order
     * in the array defined at the bottom of 'glue/runtime_interop.cpp'.
     */

    /// <summary>
    /// Collection of native functions utilized within <see cref="Godot.NativeInterop"/>.
    /// </summary>
    [GenerateUnmanagedCallbacks(typeof(UnmanagedCallbacks))]
    public static unsafe partial class NativeFuncs
    {
        private static bool initialized = false;

        /// <summary>
        /// Initializes the provided unmanaged callbacks.
        /// </summary>
        /// <param name="unmanagedCallbacks">A pointer of callbacks to initialize.</param>
        /// <param name="unmanagedCallbacksSize">The amount of callbacks passed.</param>
        /// <exception cref="InvalidOperationException"></exception>
        /// <exception cref="ArgumentException"></exception>
        // ReSharper disable once ParameterOnlyUsedForPreconditionCheck.Global
        public static void Initialize(IntPtr unmanagedCallbacks, int unmanagedCallbacksSize)
        {
            if (initialized)
                throw new InvalidOperationException("Already initialized.");
            initialized = true;

            if (unmanagedCallbacksSize != sizeof(UnmanagedCallbacks))
                throw new ArgumentException("Unmanaged callbacks size mismatch.", nameof(unmanagedCallbacksSize));

            _unmanagedCallbacks = Unsafe.AsRef<UnmanagedCallbacks>((void*)unmanagedCallbacks);
        }

        private partial struct UnmanagedCallbacks
        {
        }

        // Custom functions

        internal static partial godot_bool godotsharp_dotnet_module_is_initialized();

        /// <summary>
        /// Retrieves a bound method.
        /// </summary>
        /// <param name="p_classname">The name of the class.</param>
        /// <param name="p_methodname">The name of the method.</param>
        /// <returns>A pointer to the bound method.</returns>
        public static partial IntPtr godotsharp_method_bind_get_method(in godot_string_name p_classname,
            in godot_string_name p_methodname);

        /// <summary>
        /// Retrieves a class constructor.
        /// </summary>
        /// <param name="p_classname">The name of the class.</param>
        /// <returns>A pointer delegate to the class constructor.</returns>
        public static partial delegate* unmanaged<IntPtr> godotsharp_get_class_constructor(
            in godot_string_name p_classname);

        /// <summary>
        /// Retrieves an engine singleton.
        /// </summary>
        /// <param name="p_name">The name of the singleton.</param>
        /// <returns>A pointer to the engine singleton.</returns>
        public static partial IntPtr godotsharp_engine_get_singleton(in godot_string p_name);


        internal static partial Error godotsharp_stack_info_vector_resize(
            ref DebuggingUtils.godot_stack_info_vector p_stack_info_vector, int p_size);

        internal static partial void godotsharp_stack_info_vector_destroy(
            ref DebuggingUtils.godot_stack_info_vector p_stack_info_vector);

        internal static partial void godotsharp_internal_editor_file_system_update_file(in godot_string p_script_path);

        internal static partial void godotsharp_internal_script_debugger_send_error(in godot_string p_func,
            in godot_string p_file, int p_line, in godot_string p_err, in godot_string p_descr,
            godot_bool p_warning, in DebuggingUtils.godot_stack_info_vector p_stack_info_vector);

        internal static partial godot_bool godotsharp_internal_script_debugger_is_active();

        internal static partial IntPtr godotsharp_internal_object_get_associated_gchandle(IntPtr ptr);

        internal static partial void godotsharp_internal_object_disposed(IntPtr ptr, IntPtr gcHandleToFree);

        internal static partial void godotsharp_internal_refcounted_disposed(IntPtr ptr, IntPtr gcHandleToFree,
            godot_bool isFinalizer);

        internal static partial Error godotsharp_internal_signal_awaiter_connect(IntPtr source,
            in godot_string_name signal,
            IntPtr target, IntPtr awaiterHandlePtr);

        internal static partial void godotsharp_internal_tie_native_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, in godot_string_name nativeName, godot_bool refCounted);

        internal static partial void godotsharp_internal_tie_user_managed_to_unmanaged(IntPtr gcHandleIntPtr,
            IntPtr unmanaged, godot_ref* scriptPtr, godot_bool refCounted);

        internal static partial void godotsharp_internal_tie_managed_to_unmanaged_with_pre_setup(
            IntPtr gcHandleIntPtr, IntPtr unmanaged);

        internal static partial IntPtr godotsharp_internal_unmanaged_get_script_instance_managed(IntPtr p_unmanaged,
            out godot_bool r_has_cs_script_instance);

        internal static partial IntPtr godotsharp_internal_unmanaged_get_instance_binding_managed(IntPtr p_unmanaged);

        internal static partial IntPtr godotsharp_internal_unmanaged_instance_binding_create_managed(IntPtr p_unmanaged,
            IntPtr oldGCHandlePtr);

        internal static partial void godotsharp_internal_new_csharp_script(godot_ref* r_dest);

        internal static partial godot_bool godotsharp_internal_script_load(in godot_string p_path, godot_ref* r_dest);

        internal static partial void godotsharp_internal_reload_registered_script(IntPtr scriptPtr);

        internal static partial void godotsharp_array_filter_godot_objects_by_native(in godot_string_name p_native_name,
            in godot_array p_input, out godot_array r_output);

        internal static partial void godotsharp_array_filter_godot_objects_by_non_native(in godot_array p_input,
            out godot_array r_output);

        /// <summary>
        /// Natively initializes an <see cref="IntPtr"/> as a new <see cref="godot_ref"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_ref"/>.</param>
        /// <param name="p_ref_counted_ptr">The <see cref="IntPtr"/> to initialize from.</param>
        public static partial void godotsharp_ref_new_from_ref_counted_ptr(out godot_ref r_dest,
            IntPtr p_ref_counted_ptr);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_ref"/> instance.
        /// </summary>
        /// <param name="p_instance">The <see cref="godot_ref"/> instance to destroy.</param>
        public static partial void godotsharp_ref_destroy(ref godot_ref p_instance);

        /// <summary>
        /// Natively initializes a <see cref="godot_string"/> as a new <see cref="godot_string_name"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_ref"/>.</param>
        /// <param name="p_name">The <see cref="godot_string"/> to initialize from.</param>
        public static partial void godotsharp_string_name_new_from_string(out godot_string_name r_dest,
            in godot_string p_name);

        /// <summary>
        /// Natively initializes a <see cref="godot_string"/> as a new <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_node_path"/>.</param>
        /// <param name="p_name">The <see cref="godot_string"/> to initialize from.</param>
        public static partial void godotsharp_node_path_new_from_string(out godot_node_path r_dest,
            in godot_string p_name);

        /// <summary>
        /// Natively initializes a <see cref="godot_string_name"/> as a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_string"/>.</param>
        /// <param name="p_name">The <see cref="godot_string_name"/> to initialize from.</param>
        public static partial void
            godotsharp_string_name_as_string(out godot_string r_dest, in godot_string_name p_name);

        /// <summary>
        /// Natively initializes a <see cref="godot_node_path"/> as a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_string"/>.</param>
        /// <param name="p_np">The <see cref="godot_node_path"/> to initialize from.</param>
        public static partial void godotsharp_node_path_as_string(out godot_string r_dest, in godot_node_path p_np);

        /// <summary>
        /// Natively initializes a <see langword="byte"/> pointer as a <see cref="godot_packed_byte_array"/>.
        /// </summary>
        /// <param name="p_src">The <see langword="byte"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_byte_array"/> initialized from this <see langword="byte"/> pointer.</returns>
        public static partial godot_packed_byte_array godotsharp_packed_byte_array_new_mem_copy(byte* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes an <see langword="int"/> pointer as a <see cref="godot_packed_int32_array"/>.
        /// </summary>
        /// <param name="p_src">The <see langword="int"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_int32_array"/> initialized from this <see langword="int"/> pointer.</returns>
        public static partial godot_packed_int32_array godotsharp_packed_int32_array_new_mem_copy(int* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see langword="long"/> pointer as a <see cref="godot_packed_int64_array"/>.
        /// </summary>
        /// <param name="p_src">The <see langword="long"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_int64_array"/> initialized from this <see langword="long"/> pointer.</returns>
        public static partial godot_packed_int64_array godotsharp_packed_int64_array_new_mem_copy(long* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see langword="float"/> pointer as a <see cref="godot_packed_float32_array"/>.
        /// </summary>
        /// <param name="p_src">The <see langword="float"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_float32_array"/> initialized from this <see langword="float"/> pointer.</returns>
        public static partial godot_packed_float32_array godotsharp_packed_float32_array_new_mem_copy(float* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see langword="double"/> pointer as a <see cref="godot_packed_float64_array"/>.
        /// </summary>
        /// <param name="p_src">The <see langword="double"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_float64_array"/> initialized from this <see langword="double"/> pointer.</returns>
        public static partial godot_packed_float64_array godotsharp_packed_float64_array_new_mem_copy(double* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see cref="Vector2"/> pointer as a <see cref="godot_packed_vector2_array"/>.
        /// </summary>
        /// <param name="p_src">The <see cref="Vector2"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_vector2_array"/> initialized from this <see cref="Vector2"/> pointer.</returns>
        public static partial godot_packed_vector2_array godotsharp_packed_vector2_array_new_mem_copy(Vector2* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see cref="Vector3"/> pointer as a <see cref="godot_packed_vector3_array"/>.
        /// </summary>
        /// <param name="p_src">The <see cref="Vector3"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_vector3_array"/> initialized from this <see cref="Vector3"/> pointer.</returns>
        public static partial godot_packed_vector3_array godotsharp_packed_vector3_array_new_mem_copy(Vector3* p_src,
            int p_length);

        /// <summary>
        /// Natively initializes a <see cref="Color"/> pointer as a <see cref="godot_packed_color_array"/>.
        /// </summary>
        /// <param name="p_src">The <see cref="Color"/> pointer to initialize from.</param>
        /// <param name="p_length">The container size of this pointer.</param>
        /// <returns>A <see cref="godot_packed_color_array"/> initialized from this <see cref="Color"/> pointer.</returns>
        public static partial godot_packed_color_array godotsharp_packed_color_array_new_mem_copy(Color* p_src,
            int p_length);

        /// <summary>
        /// Natively appends a <see cref="godot_string"/> to a referenced <see cref="godot_packed_string_array"/>.
        /// </summary>
        /// <param name="r_dest">The <see cref="godot_packed_string_array"/> reference to append onto.</param>
        /// <param name="p_element">The <see cref="godot_string"/> to be appended.</param>
        public static partial void godotsharp_packed_string_array_add(ref godot_packed_string_array r_dest,
            in godot_string p_element);

        /// <summary>
        /// Natively initializes a <see cref="Callable"/> as a new <see cref="godot_callable"/>.
        /// </summary>
        /// <param name="p_delegate_handle">A pointer to the callable's GCHandle.</param>
        /// <param name="p_trampoline">A pointer to the callable's Trampoline delegate.</param>
        /// <param name="p_object">A pointer to the callable's target object.</param>
        /// <param name="r_callable">The initialized <see cref="godot_callable"/>.</param>
        public static partial void godotsharp_callable_new_with_delegate(IntPtr p_delegate_handle, IntPtr p_trampoline,
            IntPtr p_object, out godot_callable r_callable);

        internal static partial godot_bool godotsharp_callable_get_data_for_marshalling(in godot_callable p_callable,
            out IntPtr r_delegate_handle, out IntPtr r_trampoline, out IntPtr r_object, out godot_string_name r_name);

        internal static partial godot_variant godotsharp_callable_call(in godot_callable p_callable,
            godot_variant** p_args, int p_arg_count, out godot_variant_call_error p_call_error);

        internal static partial void godotsharp_callable_call_deferred(in godot_callable p_callable,
            godot_variant** p_args, int p_arg_count);

        internal static partial Color godotsharp_color_from_ok_hsl(float p_h, float p_s, float p_l, float p_alpha);

        // GDNative functions

        // gdnative.h

        /// <summary>
        /// Natively calls to a bound method pointer.
        /// </summary>
        /// <param name="p_method_bind">A pointer to the bound method pointer to call.</param>
        /// <param name="p_instance">A pointer to the instance that will make the call.</param>
        /// <param name="p_args">A collection of arguments.</param>
        /// <param name="p_ret">A pointer to the return value.</param>
        public static partial void godotsharp_method_bind_ptrcall(IntPtr p_method_bind, IntPtr p_instance, void** p_args,
            void* p_ret);

        /// <summary>
        /// Natively calls to a bound method.
        /// </summary>
        /// <param name="p_method_bind">A pointer to the bound method to call.</param>
        /// <param name="p_instance">A pointer to the instance that will make the call.</param>
        /// <param name="p_args">A collection of arguments.</param>
        /// <param name="p_arg_count">The argument collection size.</param>
        /// <param name="p_call_error">The resulting <see cref="godot_variant_call_error"/> from this call.</param>
        /// <returns>The bound method's return value.</returns>
        public static partial godot_variant godotsharp_method_bind_call(IntPtr p_method_bind, IntPtr p_instance,
            godot_variant** p_args, int p_arg_count, out godot_variant_call_error p_call_error);

        // variant.h

        /// <summary>
        /// Natively initializes a <see cref="godot_string_name"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_s">The <see cref="godot_string_name"/> to initialize from.</param>
        public static partial void
            godotsharp_variant_new_string_name(out godot_variant r_dest, in godot_string_name p_s);

        /// <summary>
        /// Natively initializes a <see cref="godot_variant"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_src">The <see cref="godot_variant"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_copy(out godot_variant r_dest, in godot_variant p_src);

        /// <summary>
        /// Natively initializes a <see cref="godot_node_path"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_np">The <see cref="godot_node_path"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_node_path(out godot_variant r_dest, in godot_node_path p_np);

        /// <summary>
        /// Natively initializes a <see cref="IntPtr"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_obj">The <see cref="IntPtr"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_object(out godot_variant r_dest, IntPtr p_obj);

        /// <summary>
        /// Natively initializes a <see cref="Transform2D"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_t2d">The <see cref="Transform2D"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_transform2d(out godot_variant r_dest, in Transform2D p_t2d);

        /// <summary>
        /// Natively initializes a <see cref="Basis"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_basis">The <see cref="Basis"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_basis(out godot_variant r_dest, in Basis p_basis);

        /// <summary>
        /// Natively initializes a <see cref="Transform3D"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_trans">The <see cref="Transform3D"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_transform3d(out godot_variant r_dest, in Transform3D p_trans);

        /// <summary>
        /// Natively initializes a <see cref="Projection"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_proj">The <see cref="Projection"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_projection(out godot_variant r_dest, in Projection p_proj);

        /// <summary>
        /// Natively initializes a <see cref="Aabb"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_aabb">The <see cref="Aabb"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_aabb(out godot_variant r_dest, in Aabb p_aabb);

        /// <summary>
        /// Natively initializes a <see cref="godot_dictionary"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_dict">The <see cref="godot_dictionary"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_dictionary(out godot_variant r_dest,
            in godot_dictionary p_dict);

        /// <summary>
        /// Natively initializes a <see cref="godot_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_arr">The <see cref="godot_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_array(out godot_variant r_dest, in godot_array p_arr);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_byte_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pba">The <see cref="godot_packed_byte_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_byte_array(out godot_variant r_dest,
            in godot_packed_byte_array p_pba);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_int32_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pia">The <see cref="godot_packed_int32_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_int32_array(out godot_variant r_dest,
            in godot_packed_int32_array p_pia);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_int64_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pia">The <see cref="godot_packed_int64_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_int64_array(out godot_variant r_dest,
            in godot_packed_int64_array p_pia);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_float32_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pra">The <see cref="godot_packed_float32_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_float32_array(out godot_variant r_dest,
            in godot_packed_float32_array p_pra);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_float64_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pra">The <see cref="godot_packed_float64_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_float64_array(out godot_variant r_dest,
            in godot_packed_float64_array p_pra);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_string_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_psa">The <see cref="godot_packed_string_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_string_array(out godot_variant r_dest,
            in godot_packed_string_array p_psa);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_vector2_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pv2a">The <see cref="godot_packed_vector2_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_vector2_array(out godot_variant r_dest,
            in godot_packed_vector2_array p_pv2a);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_vector3_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pv3a">The <see cref="godot_packed_vector3_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_vector3_array(out godot_variant r_dest,
            in godot_packed_vector3_array p_pv3a);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_color_array"/> as a new <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_variant"/>.</param>
        /// <param name="p_pca">The <see cref="godot_packed_color_array"/> to initialize from.</param>
        public static partial void godotsharp_variant_new_packed_color_array(out godot_variant r_dest,
            in godot_packed_color_array p_pca);

        /// <summary>
        /// Natively initializes a <see cref="godot_bool"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_bool"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_bool godotsharp_variant_as_bool(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see langword="long"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see langword="long"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Int64 godotsharp_variant_as_int(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see langword="double"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see langword="double"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial double godotsharp_variant_as_float(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_string"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_string"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_string godotsharp_variant_as_string(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector2"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector2"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector2 godotsharp_variant_as_vector2(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector2I"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector2I"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector2I godotsharp_variant_as_vector2i(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Rect2"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Rect2"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Rect2 godotsharp_variant_as_rect2(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Rect2I"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Rect2I"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Rect2I godotsharp_variant_as_rect2i(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector3"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector3"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector3 godotsharp_variant_as_vector3(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector3I"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector3I"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector3I godotsharp_variant_as_vector3i(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Transform2D"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Transform2D"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Transform2D godotsharp_variant_as_transform2d(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector4"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector4"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector4 godotsharp_variant_as_vector4(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Vector4I"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Vector4I"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Vector4I godotsharp_variant_as_vector4i(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Plane"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Plane"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Plane godotsharp_variant_as_plane(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Quaternion"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Quaternion"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Quaternion godotsharp_variant_as_quaternion(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Aabb"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Aabb"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Aabb godotsharp_variant_as_aabb(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Basis"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Basis"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Basis godotsharp_variant_as_basis(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Transform3D"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Transform3D"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Transform3D godotsharp_variant_as_transform3d(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Projection"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Projection"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Projection godotsharp_variant_as_projection(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Color"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Color"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Color godotsharp_variant_as_color(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_string_name"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_string_name"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_string_name godotsharp_variant_as_string_name(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_node_path"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_node_path"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_node_path godotsharp_variant_as_node_path(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="Rid"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="Rid"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial Rid godotsharp_variant_as_rid(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_callable"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_callable"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_callable godotsharp_variant_as_callable(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_signal"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_signal"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_signal godotsharp_variant_as_signal(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_dictionary"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_dictionary"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_dictionary godotsharp_variant_as_dictionary(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_array godotsharp_variant_as_array(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_byte_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_byte_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_byte_array godotsharp_variant_as_packed_byte_array(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_int32_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_int32_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_int32_array godotsharp_variant_as_packed_int32_array(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_int64_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_int64_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_int64_array godotsharp_variant_as_packed_int64_array(in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_float32_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_float32_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_float32_array godotsharp_variant_as_packed_float32_array(
            in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_float64_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_float64_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_float64_array godotsharp_variant_as_packed_float64_array(
            in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_string_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_string_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_string_array godotsharp_variant_as_packed_string_array(
            in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_vector2_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_vector2_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_vector2_array godotsharp_variant_as_packed_vector2_array(
            in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_vector3_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_vector3_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_vector3_array godotsharp_variant_as_packed_vector3_array(
            in godot_variant p_self);

        /// <summary>
        /// Natively initializes a <see cref="godot_packed_color_array"/> as a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> to initialize from.</param>
        /// <returns>A <see cref="godot_packed_color_array"/> initialized from this <see cref="godot_variant"/>.</returns>
        public static partial godot_packed_color_array godotsharp_variant_as_packed_color_array(in godot_variant p_self);

        /// <summary>
        /// Natively evaluates if the <see cref="godot_variant"/> instances are exactly equal.
        /// </summary>
        /// <param name="p_a">The left <see cref="godot_variant"/>.</param>
        /// <param name="p_b">The right <see cref="godot_variant"/>.</param>
        /// <returns><see cref="godot_bool.True"/> if these <see cref="godot_variant"/> are
        /// exactly equal; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_variant_equals(in godot_variant p_a, in godot_variant p_b);

        // string.h

        /// <summary>
        /// Natively initializes a <see langword="char"/> pointer as a new <see cref="godot_string"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_string"/>.</param>
        /// <param name="p_contents">The <see langword="char"/> pointer to initialize from.</param>
        public static partial void godotsharp_string_new_with_utf16_chars(out godot_string r_dest, char* p_contents);

        // string_name.h

        /// <summary>
        /// Natively initializes a <see cref="godot_string_name"/> as a new <see cref="godot_string_name"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_string"/>.</param>
        /// <param name="p_src">The <see cref="godot_string_name"/> pointer to initialize from.</param>
        public static partial void godotsharp_string_name_new_copy(out godot_string_name r_dest,
            in godot_string_name p_src);

        // node_path.h

        /// <summary>
        /// Natively initializes a <see cref="godot_node_path"/> as a new <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_node_path"/>.</param>
        /// <param name="p_src">The <see cref="godot_node_path"/> pointer to initialize from.</param>
        public static partial void godotsharp_node_path_new_copy(out godot_node_path r_dest, in godot_node_path p_src);

        // array.h

        /// <summary>
        /// Natively initializes a new <see cref="godot_array"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_array"/>.</param>
        public static partial void godotsharp_array_new(out godot_array r_dest);

        /// <summary>
        /// Natively initializes a <see cref="godot_array"/> as a new <see cref="godot_array"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_array"/>.</param>
        /// <param name="p_src">The <see cref="godot_array"/> pointer to initialize from.</param>
        public static partial void godotsharp_array_new_copy(out godot_array r_dest, in godot_array p_src);

        /// <summary>
        /// Natively retrieves a <see cref="godot_variant"/> pointer collection from the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to retrieve the collection pointer from.</param>
        /// <returns>A <see cref="godot_variant"/> pointer collection extracted from <paramref name="p_self"/>.</returns>
        public static partial godot_variant* godotsharp_array_ptrw(ref godot_array p_self);

        // dictionary.h

        /// <summary>
        /// Natively initializes a new <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_dictionary"/>.</param>
        public static partial void godotsharp_dictionary_new(out godot_dictionary r_dest);

        /// <summary>
        /// Natively initializes a <see cref="godot_dictionary"/> as a new <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="r_dest">The initialized <see cref="godot_dictionary"/>.</param>
        /// <param name="p_src">The <see cref="godot_dictionary"/> pointer to initialize from.</param>
        public static partial void godotsharp_dictionary_new_copy(out godot_dictionary r_dest,
            in godot_dictionary p_src);

        // destroy functions

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_byte_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_byte_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_byte_array_destroy(ref godot_packed_byte_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_int32_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_int32_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_int32_array_destroy(ref godot_packed_int32_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_int64_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_int64_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_int64_array_destroy(ref godot_packed_int64_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_float32_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_float32_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_float32_array_destroy(ref godot_packed_float32_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_float64_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_float64_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_float64_array_destroy(ref godot_packed_float64_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_string_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_string_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_string_array_destroy(ref godot_packed_string_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_vector2_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_vector2_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_vector2_array_destroy(ref godot_packed_vector2_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_vector3_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_vector3_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_vector3_array_destroy(ref godot_packed_vector3_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_packed_color_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_packed_color_array"/> reference to destroy.</param>
        public static partial void godotsharp_packed_color_array_destroy(ref godot_packed_color_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_variant"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_variant"/> reference to destroy.</param>
        public static partial void godotsharp_variant_destroy(ref godot_variant p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_string"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string"/> reference to destroy.</param>
        public static partial void godotsharp_string_destroy(ref godot_string p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_string_name"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string_name"/> reference to destroy.</param>
        public static partial void godotsharp_string_name_destroy(ref godot_string_name p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_node_path"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> reference to destroy.</param>
        public static partial void godotsharp_node_path_destroy(ref godot_node_path p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_signal"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_signal"/> reference to destroy.</param>
        public static partial void godotsharp_signal_destroy(ref godot_signal p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_callable"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_callable"/> reference to destroy.</param>
        public static partial void godotsharp_callable_destroy(ref godot_callable p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_array"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> reference to destroy.</param>
        public static partial void godotsharp_array_destroy(ref godot_array p_self);

        /// <summary>
        /// Natively destroys the provided <see cref="godot_dictionary"/> reference.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> reference to destroy.</param>
        public static partial void godotsharp_dictionary_destroy(ref godot_dictionary p_self);

        // Array

        /// <summary>
        /// Natively adds a <see cref="godot_variant"/> to the end of the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to append onto.</param>
        /// <param name="p_item">The <see cref="godot_variant"/> to add.</param>
        public static partial int godotsharp_array_add(ref godot_array p_self, in godot_variant p_item);

        /// <summary>
        /// Natively adds a <see cref="godot_array"/> to the end of the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to append onto.</param>
        /// <param name="p_collection">The <see cref="godot_variant"/> to add.</param>
        public static partial int godotsharp_array_add_range(ref godot_array p_self, in godot_array p_collection);

        /// <summary>
        /// Natively finds the index of a <see cref="godot_variant"/> in the provided <see cref="godot_array"/>
        /// using a binary search.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="p_index">The starting index to search.</param>
        /// <param name="p_count">The amount of items to search.</param>
        /// <param name="p_value">The <see cref="godot_variant"/> to find the index of.</param>
        /// <returns>The index of the <see cref="godot_variant"/> in the <see cref="godot_array"/>.</returns>
        public static partial int godotsharp_array_binary_search(ref godot_array p_self, int p_index, int p_count, in godot_variant p_value);

        /// <summary>
        /// Natively duplicates a provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to duplicate.</param>
        /// <param name="p_deep">Determines if a deep copy should be performed.</param>
        /// <param name="r_dest">The newly duplicated <see cref="godot_array"/>.</param>
        public static partial void
            godotsharp_array_duplicate(ref godot_array p_self, godot_bool p_deep, out godot_array r_dest);

        /// <summary>
        /// Natively fills a provided <see cref="godot_array"/> with a given <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to be filled.</param>
        /// <param name="p_value">The <see cref="godot_variant"/> to fill the <see cref="godot_array"/> with.</param>
        public static partial void godotsharp_array_fill(ref godot_array p_self, in godot_variant p_value);

        /// <summary>
        /// Natively finds the index of a <see cref="godot_variant"/> in the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="p_item">The <see cref="godot_variant"/> to find the index of.</param>
        /// <param name="p_index">The starting index to search.</param>
        /// <returns>The index of the <see cref="godot_variant"/> in the <see cref="godot_array"/>.</returns>
        public static partial int godotsharp_array_index_of(ref godot_array p_self, in godot_variant p_item, int p_index = 0);

        /// <summary>
        /// Natively inserts a <see cref="godot_variant"/> to the provided <see cref="godot_array"/> at the specified index.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to insert into.</param>
        /// <param name="p_index">The <see cref="godot_variant"/> to be inserted.</param>
        /// <param name="p_item">The index to insert at.</param>
        public static partial void godotsharp_array_insert(ref godot_array p_self, int p_index, in godot_variant p_item);

        /// <summary>
        /// Natively finds the last index of a <see cref="godot_variant"/> in the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="p_item">The <see cref="godot_variant"/> to find the last index of.</param>
        /// <param name="p_index">The starting index to search.</param>
        /// <returns>The index of the <see cref="godot_variant"/> in the <see cref="godot_array"/>.</returns>
        public static partial int godotsharp_array_last_index_of(ref godot_array p_self, in godot_variant p_item, int p_index);

        /// <summary>
        /// Natively makes the provided <see cref="godot_array"/> read-only.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to make read-only.</param>
        public static partial void godotsharp_array_make_read_only(ref godot_array p_self);

        /// <summary>
        /// Natively retrieves the maximum <see cref="godot_variant"/> value from the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="r_value">The parsed, max value <see cref="godot_variant"/>.</param>
        public static partial void godotsharp_array_max(ref godot_array p_self, out godot_variant r_value);

        /// <summary>
        /// Natively retrieves the minimum <see cref="godot_variant"/> value from the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="r_value">The parsed, min value <see cref="godot_variant"/>.</param>
        public static partial void godotsharp_array_min(ref godot_array p_self, out godot_variant r_value);

        /// <summary>
        /// Natively retrieves a random <see cref="godot_variant"/> value from the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to parse.</param>
        /// <param name="r_value">The parsed, random value <see cref="godot_variant"/>.</param>
        public static partial void godotsharp_array_pick_random(ref godot_array p_self, out godot_variant r_value);

        /// <summary>
        /// Natively evaluates if the <see cref="godot_array"/> instances are recursively equal.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to equate.</param>
        /// <param name="p_other">The <see cref="godot_array"/> to compare with.</param>
        /// <returns><see cref="godot_bool.True"/> if these <see cref="godot_array"/> are
        /// recursively equal; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_array_recursive_equal(ref godot_array p_self, in godot_array p_other);

        /// <summary>
        /// Natively removes an item from the provided <see cref="godot_array"/> at the specified index.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to remove from.</param>
        /// <param name="p_index">The index to remove at.</param>
        public static partial void godotsharp_array_remove_at(ref godot_array p_self, int p_index);

        /// <summary>
        /// Natively resizes the provided <see cref="godot_array"/> to the specified size.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to resize.</param>
        /// <param name="p_new_size">The new size of the <see cref="godot_array"/>.</param>
        /// <returns>The <see cref="Error"/> status of this operation.</returns>
        public static partial Error godotsharp_array_resize(ref godot_array p_self, int p_new_size);

        /// <summary>
        /// Natively reverses the order of items in the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to invert.</param>
        public static partial void godotsharp_array_reverse(ref godot_array p_self);

        /// <summary>
        /// Natively shuffles the order of items in the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to randomize.</param>
        public static partial void godotsharp_array_shuffle(ref godot_array p_self);

        /// <summary>
        /// Natively slices the provided <see cref="godot_array"/> into a new <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to slice.</param>
        /// <param name="p_start">The zero-based starting index.</param>
        /// <param name="p_end">The zero-based ending index.</param>
        /// <param name="p_step">The relative index between elements.</param>
        /// <param name="p_deep">Determines if a deep copy should be performed.</param>
        /// <param name="r_dest">The resulting <see cref="godot_array"/> from the slice.</param>
        public static partial void godotsharp_array_slice(ref godot_array p_self, int p_start, int p_end,
            int p_step, godot_bool p_deep, out godot_array r_dest);

        /// <summary>
        /// Natively sorts the order of items in the provided <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to organize.</param>
        public static partial void godotsharp_array_sort(ref godot_array p_self);

        /// <summary>
        /// Natively converts the specified <see cref="godot_array"/> to a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_array"/> to convert.</param>
        /// <param name="r_str">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_array_to_string(ref godot_array p_self, out godot_string r_str);

        // Dictionary

        /// <summary>
        /// Natively tries to retrieve a value from the provided <see cref="godot_dictionary"/>
        /// at the specified <paramref name="p_key"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <param name="p_key">The <see cref="godot_variant"/> key to search with.</param>
        /// <param name="r_value">The parsed <see cref="godot_variant"/> value.</param>
        /// <returns><see cref="godot_bool.True"/> if the <paramref name="r_value"/> was
        /// successfully retrieved; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_dictionary_try_get_value(ref godot_dictionary p_self,
            in godot_variant p_key,
            out godot_variant r_value);

        /// <summary>
        /// Natively sets a value in the provided <see cref="godot_dictionary"/> at the
        /// specified <paramref name="p_key"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to use.</param>
        /// <param name="p_key">The <see cref="godot_variant"/> key to insert at.</param>
        /// <param name="p_value">The <see cref="godot_variant"/> value to store.</param>
        public static partial void godotsharp_dictionary_set_value(ref godot_dictionary p_self, in godot_variant p_key,
            in godot_variant p_value);

        /// <summary>
        /// Natively retrieves a <see cref="godot_array"/> of keys from the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <param name="r_dest">An extracted <see cref="godot_array"/> of keys.</param>
        public static partial void godotsharp_dictionary_keys(ref godot_dictionary p_self, out godot_array r_dest);

        /// <summary>
        /// Natively retrieves a <see cref="godot_array"/> of values from the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <param name="r_dest">An extracted <see cref="godot_array"/> of values.</param>
        public static partial void godotsharp_dictionary_values(ref godot_dictionary p_self, out godot_array r_dest);

        /// <summary>
        /// Natively retrieves the number of items in the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <returns>The number of items.</returns>
        public static partial int godotsharp_dictionary_count(ref godot_dictionary p_self);

        /// <summary>
        /// Natively retrieves a key/value pair from the provided <see cref="godot_dictionary"/> at the specified index.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <param name="p_index">The index to search at.</param>
        /// <param name="r_key">The extracted <see cref="godot_variant"/> key.</param>
        /// <param name="r_value">The extracted <see cref="godot_variant"/> value.</param>
        public static partial void godotsharp_dictionary_key_value_pair_at(ref godot_dictionary p_self, int p_index,
            out godot_variant r_key, out godot_variant r_value);

        /// <summary>
        /// Natively adds a new key/value pair to the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to insert into.</param>
        /// <param name="p_key">The <see cref="godot_variant"/> key to insert.</param>
        /// <param name="p_value">The <see cref="godot_variant"/> value to insert.</param>
        public static partial void godotsharp_dictionary_add(ref godot_dictionary p_self, in godot_variant p_key,
            in godot_variant p_value);

        /// <summary>
        /// Natively clears all items from the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to expunge.</param>
        public static partial void godotsharp_dictionary_clear(ref godot_dictionary p_self);

        /// <summary>
        /// Natively determines if the provided <see cref="godot_dictionary"/> contains a specified <paramref name="p_key"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to parse.</param>
        /// <param name="p_key">The <see cref="godot_variant"/> key to search for.</param>
        /// <returns><see cref="godot_bool.True"/> if <paramref name="p_key"/> was found in the
        /// provided <see cref="godot_dictionary"/>; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_dictionary_contains_key(ref godot_dictionary p_self,
            in godot_variant p_key);

        /// <summary>
        /// Natively duplicates a provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to duplicate.</param>
        /// <param name="p_deep">Determines if a deep copy should be performed.</param>
        /// <param name="r_dest">The newly duplicated <see cref="godot_dictionary"/>.</param>
        public static partial void godotsharp_dictionary_duplicate(ref godot_dictionary p_self, godot_bool p_deep,
            out godot_dictionary r_dest);

        /// <summary>
        /// Natively merges a <see cref="godot_dictionary"/> with the provided <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to merge into.</param>
        /// <param name="p_dictionary">The <see cref="godot_dictionary"/> to be merged.</param>
        /// <param name="p_overwrite">Determines if duplicate keys should be overwritten.</param>
        public static partial void godotsharp_dictionary_merge(ref godot_dictionary p_self, in godot_dictionary p_dictionary, godot_bool p_overwrite);

        /// <summary>
        /// Natively evaluates if the <see cref="godot_dictionary"/> instances are recursively equal.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to equate.</param>
        /// <param name="p_other">The <see cref="godot_dictionary"/> to compare with.</param>
        /// <returns><see cref="godot_bool.True"/> if these <see cref="godot_dictionary"/> are
        /// recursively equal; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_dictionary_recursive_equal(ref godot_dictionary p_self, in godot_dictionary p_other);

        /// <summary>
        /// Natively removes an item from the provided <see cref="godot_dictionary"/> at a specified <paramref name="p_key"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to remove from.</param>
        /// <param name="p_key">The <see cref="godot_variant"/> key to remove.</param>
        /// <returns></returns>
        public static partial godot_bool godotsharp_dictionary_remove_key(ref godot_dictionary p_self,
            in godot_variant p_key);

        /// <summary>
        /// Natively makes the provided <see cref="godot_dictionary"/> read-only.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to make read-only.</param>
        public static partial void godotsharp_dictionary_make_read_only(ref godot_dictionary p_self);

        /// <summary>
        /// Natively converts the specified <see cref="godot_dictionary"/> to a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_dictionary"/> to convert.</param>
        /// <param name="r_str">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_dictionary_to_string(ref godot_dictionary p_self, out godot_string r_str);

        // StringExtensions

        /// <summary>
        /// Natively simplifies a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string"/> to simplify.</param>
        /// <param name="r_simplified_path">The simplified <see cref="godot_string"/>.</param>
        public static partial void godotsharp_string_simplify_path(in godot_string p_self,
            out godot_string r_simplified_path);

        /// <summary>
        /// Natively converts a <see cref="godot_string"/> to camel case.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string"/> to convert.</param>
        /// <param name="r_camel_case">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_string_to_camel_case(in godot_string p_self,
            out godot_string r_camel_case);

        /// <summary>
        /// Natively converts a <see cref="godot_string"/> to pascal case.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string"/> to convert.</param>
        /// <param name="r_pascal_case">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_string_to_pascal_case(in godot_string p_self,
            out godot_string r_pascal_case);

        /// <summary>
        /// Natively converts a <see cref="godot_string"/> to snake case.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_string"/> to convert.</param>
        /// <param name="r_snake_case">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_string_to_snake_case(in godot_string p_self,
            out godot_string r_snake_case);

        // NodePath

        /// <summary>
        /// Natively converts a <see cref="godot_node_path"/> to a property path.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to convert.</param>
        /// <param name="r_dest">The converted <see cref="godot_node_path"/>.</param>
        public static partial void godotsharp_node_path_get_as_property_path(in godot_node_path p_self,
            ref godot_node_path r_dest);

        /// <summary>
        /// Natively converts a <see cref="godot_node_path"/> to a concatenated <see cref="godot_string"/> of names.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to convert.</param>
        /// <param name="r_names">The concatenated <see cref="godot_string"/>.</param>
        public static partial void godotsharp_node_path_get_concatenated_names(in godot_node_path p_self,
            out godot_string r_names);

        /// <summary>
        /// Natively converts a <see cref="godot_node_path"/> to a concatenated <see cref="godot_string"/> of subnames.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to convert.</param>
        /// <param name="r_subnames">The concatenated <see cref="godot_string"/>.</param>
        public static partial void godotsharp_node_path_get_concatenated_subnames(in godot_node_path p_self,
            out godot_string r_subnames);

        /// <summary>
        /// Natively converts a <see cref="godot_node_path"/> to a <see cref="godot_string"/> name.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to convert.</param>
        /// <param name="p_idx">The index to search.</param>
        /// <param name="r_name">The <see cref="godot_string"/> name.</param>
        public static partial void godotsharp_node_path_get_name(in godot_node_path p_self, int p_idx,
            out godot_string r_name);

        /// <summary>
        /// Natively retrieves the amount of names making up a <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to parse.</param>
        /// <returns>The amount of names in this <see cref="godot_node_path"/>.</returns>
        public static partial int godotsharp_node_path_get_name_count(in godot_node_path p_self);

        /// <summary>
        /// Natively converts a <see cref="godot_node_path"/> to a <see cref="godot_string"/> subname.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to convert.</param>
        /// <param name="p_idx">The index to search.</param>
        /// <param name="r_subname">The <see cref="godot_string"/> r_subname.</param>
        public static partial void godotsharp_node_path_get_subname(in godot_node_path p_self, int p_idx,
            out godot_string r_subname);

        /// <summary>
        /// Natively retrieves the amount of subnames making up a <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to parse.</param>
        /// <returns>The amount of subnames in this <see cref="godot_node_path"/>.</returns>
        public static partial int godotsharp_node_path_get_subname_count(in godot_node_path p_self);

        /// <summary>
        /// Natively determines if a <see cref="godot_node_path"/> is absolute.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to parse.</param>
        /// <returns><see cref="godot_bool.True"/> if this <see cref="godot_node_path"/>
        /// is absolute; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_node_path_is_absolute(in godot_node_path p_self);

        /// <summary>
        /// Natively evaluates if the <see cref="godot_node_path"/> instances are exactly equal.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to equate.</param>
        /// <param name="p_other">The <see cref="godot_node_path"/> to compare with.</param>
        /// <returns><see cref="godot_bool.True"/> if these <see cref="godot_node_path"/> are
        /// exactly equal; otherwise, <see cref="godot_bool.False"/>.</returns>
        public static partial godot_bool godotsharp_node_path_equals(in godot_node_path p_self, in godot_node_path p_other);

        /// <summary>
        /// Natively retrieves the hash of a <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="p_self">The <see cref="godot_node_path"/> to parse.</param>
        /// <returns>A hash code for this <see cref="godot_node_path"/>.</returns>
        public static partial int godotsharp_node_path_hash(in godot_node_path p_self);

        // GD, etc

        internal static partial void godotsharp_bytes_to_var(in godot_packed_byte_array p_bytes,
            godot_bool p_allow_objects,
            out godot_variant r_ret);

        internal static partial void godotsharp_convert(in godot_variant p_what, int p_type,
            out godot_variant r_ret);

        internal static partial int godotsharp_hash(in godot_variant p_var);

        internal static partial IntPtr godotsharp_instance_from_id(ulong p_instance_id);

        internal static partial void godotsharp_print(in godot_string p_what);

        /// <summary>
        /// Natively prints a <see cref="godot_string"/> using rich text.
        /// </summary>
        /// <param name="p_what">The <see cref="godot_string"/> to print.</param>
        public static partial void godotsharp_print_rich(in godot_string p_what);

        internal static partial void godotsharp_printerr(in godot_string p_what);

        internal static partial void godotsharp_printraw(in godot_string p_what);

        internal static partial void godotsharp_prints(in godot_string p_what);

        internal static partial void godotsharp_printt(in godot_string p_what);

        internal static partial float godotsharp_randf();

        internal static partial uint godotsharp_randi();

        internal static partial void godotsharp_randomize();

        internal static partial double godotsharp_randf_range(double from, double to);

        internal static partial double godotsharp_randfn(double mean, double deviation);

        internal static partial int godotsharp_randi_range(int from, int to);

        internal static partial uint godotsharp_rand_from_seed(ulong seed, out ulong newSeed);

        internal static partial void godotsharp_seed(ulong seed);

        internal static partial void godotsharp_weakref(IntPtr p_obj, out godot_ref r_weak_ref);

        internal static partial void godotsharp_str_to_var(in godot_string p_str, out godot_variant r_ret);

        internal static partial void godotsharp_var_to_bytes(in godot_variant p_what, godot_bool p_full_objects,
            out godot_packed_byte_array r_bytes);

        internal static partial void godotsharp_var_to_str(in godot_variant p_var, out godot_string r_ret);

        internal static partial void godotsharp_pusherror(in godot_string p_str);

        internal static partial void godotsharp_pushwarning(in godot_string p_str);

        // Object

        /// <summary>
        /// Natively converts an object <see cref="IntPtr"/> to a <see cref="godot_string"/>.
        /// </summary>
        /// <param name="ptr">The object <see cref="IntPtr"/> to convert from.</param>
        /// <param name="r_str">The converted <see cref="godot_string"/>.</param>
        public static partial void godotsharp_object_to_string(IntPtr ptr, out godot_string r_str);
    }
}
