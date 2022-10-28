#nullable enable

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot.NativeInterop;

namespace Godot
{
    internal static class DelegateUtils
    {
        [UnmanagedCallersOnly]
        internal static godot_bool DelegateEquals(IntPtr delegateGCHandleA, IntPtr delegateGCHandleB)
        {
            try
            {
                var @delegateA = (Delegate?)GCHandle.FromIntPtr(delegateGCHandleA).Target;
                var @delegateB = (Delegate?)GCHandle.FromIntPtr(delegateGCHandleB).Target;
                return (@delegateA! == @delegateB!).ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe void InvokeWithVariantArgs(IntPtr delegateGCHandle, void* trampoline,
            godot_variant** args, int argc, godot_variant* outRet)
        {
            try
            {
                if (trampoline == null)
                {
                    throw new ArgumentNullException(nameof(trampoline),
                        "Cannot dynamically invoke delegate because the trampoline is null.");
                }

                var @delegate = (Delegate)GCHandle.FromIntPtr(delegateGCHandle).Target!;
                var trampolineFn = (delegate* managed<object, NativeVariantPtrArgs, out godot_variant, void>)trampoline;

                trampolineFn(@delegate, new NativeVariantPtrArgs(args, argc), out godot_variant ret);

                *outRet = ret;
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *outRet = default;
            }
        }

        // TODO: Check if we should be using BindingFlags.DeclaredOnly (would give better reflection performance).

        private enum TargetKind : uint
        {
            Static,
            GodotObject,
            CompilerGenerated
        }

        internal static bool TrySerializeDelegate(Delegate @delegate, Collections.Array serializedData)
        {
            if (@delegate is null)
            {
                return false;
            }

            if (@delegate is MulticastDelegate multicastDelegate)
            {
                bool someDelegatesSerialized = false;

                Delegate[] invocationList = multicastDelegate.GetInvocationList();

                if (invocationList.Length > 1)
                {
                    var multiCastData = new Collections.Array();

                    foreach (Delegate oneDelegate in invocationList)
                        someDelegatesSerialized |= TrySerializeDelegate(oneDelegate, multiCastData);

                    if (!someDelegatesSerialized)
                        return false;

                    serializedData.Add(multiCastData);
                    return true;
                }
            }

            if (TrySerializeSingleDelegate(@delegate, out byte[]? buffer))
            {
                serializedData.Add((Span<byte>)buffer);
                return true;
            }

            return false;
        }

        private static bool TrySerializeSingleDelegate(Delegate @delegate, [MaybeNullWhen(false)] out byte[] buffer)
        {
            buffer = null;

            object? target = @delegate.Target;

            switch (target)
            {
                case null:
                {
                    using (var stream = new MemoryStream())
                    using (var writer = new BinaryWriter(stream))
                    {
                        writer.Write((ulong)TargetKind.Static);

                        SerializeType(writer, @delegate.GetType());

                        if (!TrySerializeMethodInfo(writer, @delegate.Method))
                            return false;

                        buffer = stream.ToArray();
                        return true;
                    }
                }
                // ReSharper disable once RedundantNameQualifier
                case Godot.Object godotObject:
                {
                    using (var stream = new MemoryStream())
                    using (var writer = new BinaryWriter(stream))
                    {
                        writer.Write((ulong)TargetKind.GodotObject);
                        // ReSharper disable once RedundantCast
                        writer.Write((ulong)godotObject.GetInstanceId());

                        SerializeType(writer, @delegate.GetType());

                        if (!TrySerializeMethodInfo(writer, @delegate.Method))
                            return false;

                        buffer = stream.ToArray();
                        return true;
                    }
                }
                default:
                {
                    Type targetType = target.GetType();

                    if (targetType.IsDefined(typeof(CompilerGeneratedAttribute), true))
                    {
                        // Compiler generated. Probably a closure. Try to serialize it.

                        using (var stream = new MemoryStream())
                        using (var writer = new BinaryWriter(stream))
                        {
                            writer.Write((ulong)TargetKind.CompilerGenerated);
                            SerializeType(writer, targetType);

                            SerializeType(writer, @delegate.GetType());

                            if (!TrySerializeMethodInfo(writer, @delegate.Method))
                                return false;

                            FieldInfo[] fields = targetType.GetFields(BindingFlags.Instance | BindingFlags.Public);

                            writer.Write(fields.Length);

                            foreach (FieldInfo field in fields)
                            {
                                Type fieldType = field.GetType();

                                Variant.Type variantType = GD.TypeToVariantType(fieldType);

                                if (variantType == Variant.Type.Nil)
                                    return false;

                                static byte[] VarToBytes(in godot_variant var)
                                {
                                    NativeFuncs.godotsharp_var_to_bytes(var, false.ToGodotBool(), out var varBytes);
                                    using (varBytes)
                                        return Marshaling.ConvertNativePackedByteArrayToSystemArray(varBytes);
                                }

                                writer.Write(field.Name);

                                var fieldValue = field.GetValue(target);
                                using var fieldValueVariant = Marshaling.ConvertManagedObjectToVariant(fieldValue);
                                byte[] valueBuffer = VarToBytes(fieldValueVariant);
                                writer.Write(valueBuffer.Length);
                                writer.Write(valueBuffer);
                            }

                            buffer = stream.ToArray();
                            return true;
                        }
                    }

                    return false;
                }
            }
        }

        private static bool TrySerializeMethodInfo(BinaryWriter writer, MethodInfo methodInfo)
        {
            SerializeType(writer, methodInfo.DeclaringType);

            writer.Write(methodInfo.Name);

            int flags = 0;

            if (methodInfo.IsPublic)
                flags |= (int)BindingFlags.Public;
            else
                flags |= (int)BindingFlags.NonPublic;

            if (methodInfo.IsStatic)
                flags |= (int)BindingFlags.Static;
            else
                flags |= (int)BindingFlags.Instance;

            writer.Write(flags);

            Type returnType = methodInfo.ReturnType;
            bool hasReturn = methodInfo.ReturnType != typeof(void);

            writer.Write(hasReturn);
            if (hasReturn)
                SerializeType(writer, returnType);

            ParameterInfo[] parameters = methodInfo.GetParameters();

            writer.Write(parameters.Length);

            if (parameters.Length > 0)
            {
                for (int i = 0; i < parameters.Length; i++)
                    SerializeType(writer, parameters[i].ParameterType);
            }

            return true;
        }

        private static void SerializeType(BinaryWriter writer, Type? type)
        {
            if (type == null)
            {
                int genericArgumentsCount = -1;
                writer.Write(genericArgumentsCount);
            }
            else if (type.IsGenericType)
            {
                Type genericTypeDef = type.GetGenericTypeDefinition();
                Type[] genericArgs = type.GetGenericArguments();

                int genericArgumentsCount = genericArgs.Length;
                writer.Write(genericArgumentsCount);

                writer.Write(genericTypeDef.Assembly.GetName().Name ?? "");
                writer.Write(genericTypeDef.FullName ?? genericTypeDef.ToString());

                for (int i = 0; i < genericArgs.Length; i++)
                    SerializeType(writer, genericArgs[i]);
            }
            else
            {
                int genericArgumentsCount = 0;
                writer.Write(genericArgumentsCount);

                writer.Write(type.Assembly.GetName().Name ?? "");
                writer.Write(type.FullName ?? type.ToString());
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool TrySerializeDelegateWithGCHandle(IntPtr delegateGCHandle,
            godot_array* nSerializedData)
        {
            try
            {
                var serializedData = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    NativeFuncs.godotsharp_array_new_copy(*nSerializedData));

                var @delegate = (Delegate)GCHandle.FromIntPtr(delegateGCHandle).Target!;

                return TrySerializeDelegate(@delegate, serializedData)
                    .ToGodotBool();
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                return godot_bool.False;
            }
        }

        [UnmanagedCallersOnly]
        internal static unsafe godot_bool TryDeserializeDelegateWithGCHandle(godot_array* nSerializedData,
            IntPtr* delegateGCHandle)
        {
            try
            {
                var serializedData = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    NativeFuncs.godotsharp_array_new_copy(*nSerializedData));

                if (TryDeserializeDelegate(serializedData, out Delegate? @delegate))
                {
                    *delegateGCHandle = GCHandle.ToIntPtr(CustomGCHandle.AllocStrong(@delegate));
                    return godot_bool.True;
                }
                else
                {
                    *delegateGCHandle = IntPtr.Zero;
                    return godot_bool.False;
                }
            }
            catch (Exception e)
            {
                ExceptionUtils.LogException(e);
                *delegateGCHandle = default;
                return godot_bool.False;
            }
        }

        internal static bool TryDeserializeDelegate(Collections.Array serializedData,
            [MaybeNullWhen(false)] out Delegate @delegate)
        {
            @delegate = null;

            if (serializedData.Count == 1)
            {
                var elem = serializedData[0].Obj;

                if (elem == null)
                    return false;

                if (elem is Collections.Array multiCastData)
                    return TryDeserializeDelegate(multiCastData, out @delegate);

                return TryDeserializeSingleDelegate((byte[])elem, out @delegate);
            }

            var delegates = new List<Delegate>(serializedData.Count);

            foreach (Variant variantElem in serializedData)
            {
                var elem = variantElem.Obj;

                if (elem == null)
                    continue;

                if (elem is Collections.Array multiCastData)
                {
                    if (TryDeserializeDelegate(multiCastData, out Delegate? oneDelegate))
                        delegates.Add(oneDelegate);
                }
                else
                {
                    if (TryDeserializeSingleDelegate((byte[])elem, out Delegate? oneDelegate))
                        delegates.Add(oneDelegate);
                }
            }

            if (delegates.Count <= 0)
                return false;

            @delegate = delegates.Count == 1 ? delegates[0] : Delegate.Combine(delegates.ToArray())!;
            return true;
        }

        private static bool TryDeserializeSingleDelegate(byte[] buffer, [MaybeNullWhen(false)] out Delegate @delegate)
        {
            @delegate = null;

            using (var stream = new MemoryStream(buffer, writable: false))
            using (var reader = new BinaryReader(stream))
            {
                var targetKind = (TargetKind)reader.ReadUInt64();

                switch (targetKind)
                {
                    case TargetKind.Static:
                    {
                        Type? delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo? methodInfo))
                            return false;

                        @delegate = Delegate.CreateDelegate(delegateType, null, methodInfo, throwOnBindFailure: false);

                        if (@delegate == null)
                            return false;

                        return true;
                    }
                    case TargetKind.GodotObject:
                    {
                        ulong objectId = reader.ReadUInt64();
                        // ReSharper disable once RedundantNameQualifier
                        Godot.Object godotObject = GD.InstanceFromId(objectId);
                        if (godotObject == null)
                            return false;

                        Type? delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo? methodInfo))
                            return false;

                        @delegate = Delegate.CreateDelegate(delegateType, godotObject, methodInfo,
                            throwOnBindFailure: false);

                        if (@delegate == null)
                            return false;

                        return true;
                    }
                    case TargetKind.CompilerGenerated:
                    {
                        Type? targetType = DeserializeType(reader);
                        if (targetType == null)
                            return false;

                        Type? delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo? methodInfo))
                            return false;

                        int fieldCount = reader.ReadInt32();

                        object recreatedTarget = Activator.CreateInstance(targetType)!;

                        for (int i = 0; i < fieldCount; i++)
                        {
                            string name = reader.ReadString();
                            int valueBufferLength = reader.ReadInt32();
                            byte[] valueBuffer = reader.ReadBytes(valueBufferLength);

                            FieldInfo? fieldInfo = targetType.GetField(name,
                                BindingFlags.Instance | BindingFlags.Public);
                            fieldInfo?.SetValue(recreatedTarget, GD.BytesToVar(valueBuffer));
                        }

                        @delegate = Delegate.CreateDelegate(delegateType, recreatedTarget, methodInfo,
                            throwOnBindFailure: false);

                        if (@delegate == null)
                            return false;

                        return true;
                    }
                    default:
                        return false;
                }
            }
        }

        private static bool TryDeserializeMethodInfo(BinaryReader reader,
            [MaybeNullWhen(false)] out MethodInfo methodInfo)
        {
            methodInfo = null;

            Type? declaringType = DeserializeType(reader);

            if (declaringType == null)
                return false;

            string methodName = reader.ReadString();

            int flags = reader.ReadInt32();

            bool hasReturn = reader.ReadBoolean();
            Type? returnType = hasReturn ? DeserializeType(reader) : typeof(void);

            int parametersCount = reader.ReadInt32();

            if (parametersCount > 0)
            {
                var parameterTypes = new Type[parametersCount];

                for (int i = 0; i < parametersCount; i++)
                {
                    Type? parameterType = DeserializeType(reader);
                    if (parameterType == null)
                        return false;
                    parameterTypes[i] = parameterType;
                }

                methodInfo = declaringType.GetMethod(methodName, (BindingFlags)flags, null, parameterTypes, null);
                return methodInfo != null && methodInfo.ReturnType == returnType;
            }

            methodInfo = declaringType.GetMethod(methodName, (BindingFlags)flags);
            return methodInfo != null && methodInfo.ReturnType == returnType;
        }

        private static Type? DeserializeType(BinaryReader reader)
        {
            int genericArgumentsCount = reader.ReadInt32();

            if (genericArgumentsCount == -1)
                return null;

            string assemblyName = reader.ReadString();

            if (assemblyName.Length == 0)
            {
                GD.PushError($"Missing assembly name of type when attempting to deserialize delegate");
                return null;
            }

            string typeFullName = reader.ReadString();
            var type = ReflectionUtils.FindTypeInLoadedAssemblies(assemblyName, typeFullName);

            if (type == null)
                return null; // Type not found

            if (genericArgumentsCount != 0)
            {
                var genericArgumentTypes = new Type[genericArgumentsCount];

                for (int i = 0; i < genericArgumentsCount; i++)
                {
                    Type? genericArgumentType = DeserializeType(reader);
                    if (genericArgumentType == null)
                        return null;
                    genericArgumentTypes[i] = genericArgumentType;
                }

                type = type.MakeGenericType(genericArgumentTypes);
            }

            return type;
        }
    }
}
