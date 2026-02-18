using Godot.NativeInterop;
using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Bridge
{
    internal static class CSharpInstanceBridge
    {
        [UnmanagedCallersOnly]
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe godot_bool Call(
            IntPtr godotObjectGCHandle,
            godot_string_name* method,
            godot_variant** args,
            int argCount,
            godot_variant_call_error* refCallError,
            godot_variant* ret)
        {
            var handle = GCHandle.FromIntPtr(godotObjectGCHandle);
            var godotObject = Unsafe.As<GodotObject>(handle.Target);

            if (godotObject is null)
            {
                *ret = default;
                (*refCallError).Error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INSTANCE_IS_NULL;
                return godot_bool.False;
            }

            godot_variant retValue;
            var executedMethodSuccessfully = Invoke(godotObject, in method, in args, argCount, in refCallError, out retValue);
            if (executedMethodSuccessfully)
            {
                *ret = retValue;
                return godot_bool.True;
            }

            return godot_bool.False;
        }

        private static unsafe bool Invoke(GodotObject godotObject, scoped in godot_string_name* method, scoped in godot_variant** args, int argCount, scoped in godot_variant_call_error* refCallError, scoped out godot_variant retValue)
        {
            try
            {
                ref readonly var scriptMethod = ref godotObject.TryGetGodotClassMethod(in *method, argCount);
                if (!Unsafe.IsNullRef(in scriptMethod))
                {
                    var argsStruct = new NativeVariantPtrArgs(args, argCount);
                    retValue = scriptMethod(godotObject, in argsStruct);

                    return true;

                }
                else
                {
                    ref var callError = ref *refCallError;
                    callError.Error = godot_variant_call_error_error.GODOT_CALL_ERROR_CALL_ERROR_INVALID_METHOD;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }

            retValue = default;
            return false;
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Set(IntPtr godotObjectGCHandle, godot_string_name* name, godot_variant* value)
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (godotObject == null)
                    throw new InvalidOperationException();

                if (godotObject.SetGodotClassPropertyValue(CustomUnsafe.AsRef(name), CustomUnsafe.AsRef(value)))
                {
                    return godot_bool.True;
                }

                if (!godotObject.HasGodotClassMethod(GodotObject.MethodName._Set.NativeValue.DangerousSelfRef))
                {
                    return godot_bool.False;
                }

                var nameManaged = StringName.CreateTakingOwnershipOfDisposableValue(
                    NativeFuncs.godotsharp_string_name_new_copy(CustomUnsafe.AsRef(name)));

                Variant valueManaged = Variant.CreateCopyingBorrowed(*value);

                return godotObject._Set(nameManaged, valueManaged).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool Get(IntPtr godotObjectGCHandle, godot_string_name* name,
            godot_variant* outRet)
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (godotObject == null)
                    throw new InvalidOperationException();

                // Properties
                if (godotObject.GetGodotClassPropertyValue(CustomUnsafe.AsRef(name), out godot_variant outRetValue))
                {
                    *outRet = outRetValue;
                    return godot_bool.True;
                }

                // Signals
                if (godotObject.HasGodotClassSignal(CustomUnsafe.AsRef(name)))
                {
                    godot_signal signal = new godot_signal(NativeFuncs.godotsharp_string_name_new_copy(*name), godotObject.GetInstanceId());
                    *outRet = VariantUtils.CreateFromSignalTakingOwnershipOfDisposableValue(signal);
                    return godot_bool.True;
                }

                // Methods
                if (godotObject.HasGodotClassMethod(CustomUnsafe.AsRef(name)))
                {
                    godot_callable method = new godot_callable(NativeFuncs.godotsharp_string_name_new_copy(*name), godotObject.GetInstanceId());
                    *outRet = VariantUtils.CreateFromCallableTakingOwnershipOfDisposableValue(method);
                    return godot_bool.True;
                }

                if (!godotObject.HasGodotClassMethod(GodotObject.MethodName._Get.NativeValue.DangerousSelfRef))
                {
                    return godot_bool.False;
                }

                var nameManaged = StringName.CreateTakingOwnershipOfDisposableValue(
                    NativeFuncs.godotsharp_string_name_new_copy(CustomUnsafe.AsRef(name)));

                Variant ret = godotObject._Get(nameManaged);

                if (ret.VariantType == Variant.Type.Nil)
                {
                    *outRet = default;
                    return godot_bool.False;
                }

                *outRet = ret.CopyNativeVariant();
                return godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRet = default;
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static void CallDispose(IntPtr godotObjectGCHandle, godot_bool okIfNull)
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (okIfNull.ToBool())
                    godotObject?.Dispose();
                else
                    godotObject!.Dispose();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void CallToString(IntPtr godotObjectGCHandle, godot_string* outRes, godot_bool* outValid)
        {
            try
            {
                var self = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (self == null)
                {
                    *outRes = default;
                    *outValid = godot_bool.False;
                    return;
                }

                var resultStr = self.ToString();

                if (resultStr == null)
                {
                    *outRes = default;
                    *outValid = godot_bool.False;
                    return;
                }

                *outRes = Marshaling.ConvertStringToNative(resultStr);
                *outValid = godot_bool.True;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRes = default;
                *outValid = godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool HasMethodUnknownParams(IntPtr godotObjectGCHandle, godot_string_name* method)
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (godotObject == null)
                    return godot_bool.False;

                return godotObject.HasGodotClassMethod(CustomUnsafe.AsRef(method)).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void SerializeState(
            IntPtr godotObjectGCHandle,
            godot_dictionary* propertiesState,
            godot_dictionary* signalEventsState
        )
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (godotObject == null)
                    return;

                // Call OnBeforeSerialize

                // ReSharper disable once SuspiciousTypeConversion.Global
                if (godotObject is ISerializationListener serializationListener)
                    serializationListener.OnBeforeSerialize();

                // Save instance state

                using var info = GodotSerializationInfo.CreateCopyingBorrowed(
                    *propertiesState, *signalEventsState);

                godotObject.SaveGodotObjectData(info);
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void DeserializeState(
            IntPtr godotObjectGCHandle,
            godot_dictionary* propertiesState,
            godot_dictionary* signalEventsState
        )
        {
            try
            {
                var godotObject = (GodotObject)GCHandle.FromIntPtr(godotObjectGCHandle).Target;

                if (godotObject == null)
                    return;

                // Restore instance state

                using var info = GodotSerializationInfo.CreateCopyingBorrowed(
                    *propertiesState, *signalEventsState);

                godotObject.RestoreGodotObjectData(info);

                // Call OnAfterDeserialize

                // ReSharper disable once SuspiciousTypeConversion.Global
                if (godotObject is ISerializationListener serializationListener)
                    serializationListener.OnAfterDeserialize();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
            }
        }
    }
}
