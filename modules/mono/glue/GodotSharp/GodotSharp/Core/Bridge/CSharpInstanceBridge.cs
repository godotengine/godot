using System;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot.Bridge
{
    internal static class CSharpInstanceBridge
    {
        private static unsafe void Call(IntPtr godotObjectGCHandle, godot_string_name* method,
            godot_variant** args, int argCount, godot_variant_call_error* ref_callError, godot_variant* r_ret)
        {
            // Performance is not critical here as this will be replaced with source generators.
            var godotObject = (Object)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

            if (godotObject == null)
            {
                *r_ret = default;
                (*ref_callError).error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INSTANCE_IS_NULL;
                return;
            }

            using godot_string dest = default;
            NativeFuncs.godotsharp_string_name_as_string(&dest, method);
            string methodStr = Marshaling.mono_string_from_godot(dest);

            bool methodInvoked = godotObject.InternalGodotScriptCall(methodStr, args, argCount, out godot_variant outRet);

            if (!methodInvoked)
            {
                *r_ret = default;
                // This is important, as it tells Object::call that no method was called.
                // Otherwise, it would prevent Object::call from calling native methods.
                (*ref_callError).error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD;
                return;
            }

            *r_ret = outRet;
        }

        private static unsafe bool Set(IntPtr godotObjectGCHandle, godot_string_name* name, godot_variant* value)
        {
            // Performance is not critical here as this will be replaced with source generators.
            var godotObject = (Object)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

            if (godotObject == null)
                throw new InvalidOperationException();

            var nameManaged = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(name));

            if (godotObject.InternalGodotScriptSetFieldOrPropViaReflection(nameManaged.ToString(), value))
                return true;

            object valueManaged = Marshaling.variant_to_mono_object(value);

            return godotObject._Set(nameManaged, valueManaged);
        }

        private static unsafe bool Get(IntPtr godotObjectGCHandle, godot_string_name* name, godot_variant* r_retValue)
        {
            // Performance is not critical here as this will be replaced with source generators.
            var godotObject = (Object)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

            if (godotObject == null)
                throw new InvalidOperationException();

            var nameManaged = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(name));

            if (godotObject.InternalGodotScriptGetFieldOrPropViaReflection(nameManaged.ToString(),
                out godot_variant outRet))
            {
                *r_retValue = outRet;
                return true;
            }

            object ret = godotObject._Get(nameManaged);

            if (ret == null)
            {
                *r_retValue = default;
                return false;
            }

            *r_retValue = Marshaling.mono_object_to_variant(ret);
            return true;
        }

        private static void CallDispose(IntPtr godotObjectGCHandle, bool okIfNull)
        {
            var godotObject = (Object)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

            if (okIfNull)
                godotObject?.Dispose();
            else
                godotObject!.Dispose();
        }

        private static unsafe void CallToString(IntPtr godotObjectGCHandle, godot_string* r_res, bool* r_valid)
        {
            var self = (Object)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

            if (self == null)
            {
                *r_res = default;
                *r_valid = false;
                return;
            }

            var resultStr = self.ToString();

            if (resultStr == null)
            {
                *r_res = default;
                *r_valid = false;
                return;
            }

            *r_res = Marshaling.mono_string_to_godot(resultStr);
            *r_valid = true;
        }
    }
}
