using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace Godot
{
    internal static class DelegateUtils
    {
        private enum TargetKind : uint
        {
            Static,
            GodotObject,
            CompilerGenerated
        }

        internal static bool TrySerializeDelegate(Delegate @delegate, Collections.Array serializedData)
        {
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

            if (TrySerializeSingleDelegate(@delegate, out byte[] buffer))
            {
                serializedData.Add(buffer);
                return true;
            }

            return false;
        }

        private static bool TrySerializeSingleDelegate(Delegate @delegate, out byte[] buffer)
        {
            buffer = null;

            object target = @delegate.Target;

            switch (target)
            {
                case null:
                {
                    using (var stream = new MemoryStream())
                    using (var writer = new BinaryWriter(stream))
                    {
                        writer.Write((ulong) TargetKind.Static);

                        SerializeType(writer, @delegate.GetType());

                        if (!TrySerializeMethodInfo(writer, @delegate.Method))
                            return false;

                        buffer = stream.ToArray();
                        return true;
                    }
                }
                case Godot.Object godotObject:
                {
                    using (var stream = new MemoryStream())
                    using (var writer = new BinaryWriter(stream))
                    {
                        writer.Write((ulong) TargetKind.GodotObject);
                        writer.Write((ulong) godotObject.GetInstanceId());

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

                    if (targetType.GetCustomAttribute(typeof(CompilerGeneratedAttribute), true) != null)
                    {
                        // Compiler generated. Probably a closure. Try to serialize it.

                        using (var stream = new MemoryStream())
                        using (var writer = new BinaryWriter(stream))
                        {
                            writer.Write((ulong) TargetKind.CompilerGenerated);
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

                                writer.Write(field.Name);
                                byte[] valueBuffer = GD.Var2Bytes(field.GetValue(target));
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
            if (methodInfo == null)
                return false;

            SerializeType(writer, methodInfo.DeclaringType);

            writer.Write(methodInfo.Name);

            int flags = 0;

            if (methodInfo.IsPublic)
                flags |= (int) BindingFlags.Public;
            else
                flags |= (int) BindingFlags.NonPublic;

            if (methodInfo.IsStatic)
                flags |= (int) BindingFlags.Static;
            else
                flags |= (int) BindingFlags.Instance;

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

        private static void SerializeType(BinaryWriter writer, Type type)
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

                string assemblyQualifiedName = genericTypeDef.AssemblyQualifiedName;
                Debug.Assert(assemblyQualifiedName != null);
                writer.Write(assemblyQualifiedName);

                for (int i = 0; i < genericArgs.Length; i++)
                    SerializeType(writer, genericArgs[i]);
            }
            else
            {
                int genericArgumentsCount = 0;
                writer.Write(genericArgumentsCount);

                string assemblyQualifiedName = type.AssemblyQualifiedName;
                Debug.Assert(assemblyQualifiedName != null);
                writer.Write(assemblyQualifiedName);
            }
        }

        private static bool TryDeserializeDelegate(Collections.Array serializedData, out Delegate @delegate)
        {
            if (serializedData.Count == 1)
            {
                object elem = serializedData[0];

                if (elem is Collections.Array multiCastData)
                    return TryDeserializeDelegate(multiCastData, out @delegate);

                return TryDeserializeSingleDelegate((byte[])elem, out @delegate);
            }

            @delegate = null;

            var delegates = new List<Delegate>(serializedData.Count);

            foreach (object elem in serializedData)
            {
                if (elem is Collections.Array multiCastData)
                {
                    if (TryDeserializeDelegate(multiCastData, out Delegate oneDelegate))
                        delegates.Add(oneDelegate);
                }
                else
                {
                    if (TryDeserializeSingleDelegate((byte[]) elem, out Delegate oneDelegate))
                        delegates.Add(oneDelegate);
                }
            }

            if (delegates.Count <= 0)
                return false;

            @delegate = delegates.Count == 1 ? delegates[0] : Delegate.Combine(delegates.ToArray());
            return true;
        }

        private static bool TryDeserializeSingleDelegate(byte[] buffer, out Delegate @delegate)
        {
            @delegate = null;

            using (var stream = new MemoryStream(buffer, writable: false))
            using (var reader = new BinaryReader(stream))
            {
                var targetKind = (TargetKind) reader.ReadUInt64();

                switch (targetKind)
                {
                    case TargetKind.Static:
                    {
                        Type delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo methodInfo))
                            return false;

                        @delegate = Delegate.CreateDelegate(delegateType, null, methodInfo);
                        return true;
                    }
                    case TargetKind.GodotObject:
                    {
                        ulong objectId = reader.ReadUInt64();
                        Godot.Object godotObject = GD.InstanceFromId(objectId);
                        if (godotObject == null)
                            return false;

                        Type delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo methodInfo))
                            return false;

                        @delegate = Delegate.CreateDelegate(delegateType, godotObject, methodInfo);
                        return true;
                    }
                    case TargetKind.CompilerGenerated:
                    {
                        Type targetType = DeserializeType(reader);
                        if (targetType == null)
                            return false;

                        Type delegateType = DeserializeType(reader);
                        if (delegateType == null)
                            return false;

                        if (!TryDeserializeMethodInfo(reader, out MethodInfo methodInfo))
                            return false;

                        int fieldCount = reader.ReadInt32();

                        object recreatedTarget = Activator.CreateInstance(targetType);

                        for (int i = 0; i < fieldCount; i++)
                        {
                            string name = reader.ReadString();
                            int valueBufferLength = reader.ReadInt32();
                            byte[] valueBuffer = reader.ReadBytes(valueBufferLength);

                            FieldInfo fieldInfo = targetType.GetField(name, BindingFlags.Instance | BindingFlags.Public);
                            fieldInfo?.SetValue(recreatedTarget, GD.Bytes2Var(valueBuffer));
                        }

                        @delegate = Delegate.CreateDelegate(delegateType, recreatedTarget, methodInfo);
                        return true;
                    }
                    default:
                        return false;
                }
            }
        }

        private static bool TryDeserializeMethodInfo(BinaryReader reader, out MethodInfo methodInfo)
        {
            methodInfo = null;

            Type declaringType = DeserializeType(reader);

            string methodName = reader.ReadString();

            int flags = reader.ReadInt32();

            bool hasReturn = reader.ReadBoolean();
            Type returnType = hasReturn ? DeserializeType(reader) : typeof(void);

            int parametersCount = reader.ReadInt32();

            if (parametersCount > 0)
            {
                var parameterTypes = new Type[parametersCount];

                for (int i = 0; i < parametersCount; i++)
                {
                    Type parameterType = DeserializeType(reader);
                    if (parameterType == null)
                        return false;
                    parameterTypes[i] = parameterType;
                }

                methodInfo = declaringType.GetMethod(methodName, (BindingFlags) flags, null, parameterTypes, null);
                return methodInfo != null && methodInfo.ReturnType == returnType;
            }

            methodInfo = declaringType.GetMethod(methodName, (BindingFlags) flags);
            return methodInfo != null && methodInfo.ReturnType == returnType;
        }

        private static Type DeserializeType(BinaryReader reader)
        {
            int genericArgumentsCount = reader.ReadInt32();

            if (genericArgumentsCount == -1)
                return null;

            string assemblyQualifiedName = reader.ReadString();
            var type = Type.GetType(assemblyQualifiedName);

            if (type == null)
                return null; // Type not found

            if (genericArgumentsCount != 0)
            {
                var genericArgumentTypes = new Type[genericArgumentsCount];

                for (int i = 0; i < genericArgumentsCount; i++)
                {
                    Type genericArgumentType = DeserializeType(reader);
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
