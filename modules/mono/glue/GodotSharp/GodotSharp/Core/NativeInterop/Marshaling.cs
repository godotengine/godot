using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

// ReSharper disable InconsistentNaming

// We want to use full name qualifiers here even if redundant for clarity
// ReSharper disable RedundantNameQualifier

#nullable enable

// TODO: Consider removing support for IEnumerable

namespace Godot.NativeInterop
{
    public static class Marshaling
    {
        internal static Variant.Type ConvertManagedTypeToVariantType(Type type, out bool r_nil_is_variant)
        {
            r_nil_is_variant = false;

            switch (Type.GetTypeCode(type))
            {
                case TypeCode.Boolean:
                    return Variant.Type.Bool;
                case TypeCode.Char:
                    return Variant.Type.Int;
                case TypeCode.SByte:
                    return Variant.Type.Int;
                case TypeCode.Int16:
                    return Variant.Type.Int;
                case TypeCode.Int32:
                    return Variant.Type.Int;
                case TypeCode.Int64:
                    return Variant.Type.Int;
                case TypeCode.Byte:
                    return Variant.Type.Int;
                case TypeCode.UInt16:
                    return Variant.Type.Int;
                case TypeCode.UInt32:
                    return Variant.Type.Int;
                case TypeCode.UInt64:
                    return Variant.Type.Int;
                case TypeCode.Single:
                    return Variant.Type.Float;
                case TypeCode.Double:
                    return Variant.Type.Float;
                case TypeCode.String:
                    return Variant.Type.String;
                default:
                {
                    if (type == typeof(Vector2))
                        return Variant.Type.Vector2;

                    if (type == typeof(Vector2i))
                        return Variant.Type.Vector2i;

                    if (type == typeof(Rect2))
                        return Variant.Type.Rect2;

                    if (type == typeof(Rect2i))
                        return Variant.Type.Rect2i;

                    if (type == typeof(Transform2D))
                        return Variant.Type.Transform2d;

                    if (type == typeof(Vector3))
                        return Variant.Type.Vector3;

                    if (type == typeof(Vector3i))
                        return Variant.Type.Vector3i;

                    if (type == typeof(Vector4))
                        return Variant.Type.Vector4;

                    if (type == typeof(Vector4i))
                        return Variant.Type.Vector4i;

                    if (type == typeof(Basis))
                        return Variant.Type.Basis;

                    if (type == typeof(Quaternion))
                        return Variant.Type.Quaternion;

                    if (type == typeof(Transform3D))
                        return Variant.Type.Transform3d;

                    if (type == typeof(Projection))
                        return Variant.Type.Projection;

                    if (type == typeof(AABB))
                        return Variant.Type.Aabb;

                    if (type == typeof(Color))
                        return Variant.Type.Color;

                    if (type == typeof(Plane))
                        return Variant.Type.Plane;

                    if (type == typeof(Callable))
                        return Variant.Type.Callable;

                    if (type == typeof(SignalInfo))
                        return Variant.Type.Signal;

                    if (type.IsEnum)
                        return Variant.Type.Int;

                    if (type.IsArray || type.IsSZArray)
                    {
                        if (type == typeof(byte[]))
                            return Variant.Type.PackedByteArray;

                        if (type == typeof(Int32[]))
                            return Variant.Type.PackedInt32Array;

                        if (type == typeof(Int64[]))
                            return Variant.Type.PackedInt64Array;

                        if (type == typeof(float[]))
                            return Variant.Type.PackedFloat32Array;

                        if (type == typeof(double[]))
                            return Variant.Type.PackedFloat64Array;

                        if (type == typeof(string[]))
                            return Variant.Type.PackedStringArray;

                        if (type == typeof(Vector2[]))
                            return Variant.Type.PackedVector2Array;

                        if (type == typeof(Vector3[]))
                            return Variant.Type.PackedVector3Array;

                        if (type == typeof(Color[]))
                            return Variant.Type.PackedColorArray;

                        if (type == typeof(StringName[]))
                            return Variant.Type.Array;

                        if (type == typeof(NodePath[]))
                            return Variant.Type.Array;

                        if (type == typeof(RID[]))
                            return Variant.Type.Array;

                        if (typeof(Godot.Object[]).IsAssignableFrom(type))
                            return Variant.Type.Array;

                        if (type == typeof(object[]))
                            return Variant.Type.Array;
                    }
                    else if (type.IsGenericType)
                    {
                        var genericTypeDefinition = type.GetGenericTypeDefinition();

                        if (genericTypeDefinition == typeof(Collections.Dictionary<,>))
                            return Variant.Type.Dictionary;

                        if (genericTypeDefinition == typeof(Collections.Array<>))
                            return Variant.Type.Array;

                        if (genericTypeDefinition == typeof(System.Collections.Generic.Dictionary<,>))
                            return Variant.Type.Dictionary;

                        if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
                            return Variant.Type.Array;

                        if (genericTypeDefinition == typeof(IDictionary<,>))
                            return Variant.Type.Dictionary;

                        if (genericTypeDefinition == typeof(ICollection<>) ||
                            genericTypeDefinition == typeof(IEnumerable<>))
                            return Variant.Type.Array;

                        if (typeof(Godot.Object).IsAssignableFrom(type))
                            return Variant.Type.Object;
                    }
                    else if (type == typeof(object))
                    {
                        r_nil_is_variant = true;
                        return Variant.Type.Nil;
                    }
                    else
                    {
                        if (typeof(Godot.Object).IsAssignableFrom(type))
                            return Variant.Type.Object;

                        if (typeof(StringName) == type)
                            return Variant.Type.StringName;

                        if (typeof(NodePath) == type)
                            return Variant.Type.NodePath;

                        if (typeof(RID) == type)
                            return Variant.Type.Rid;

                        if (typeof(Collections.Dictionary) == type || typeof(System.Collections.IDictionary) == type)
                            return Variant.Type.Dictionary;

                        if (typeof(Collections.Array) == type ||
                            typeof(System.Collections.ICollection) == type ||
                            typeof(System.Collections.IEnumerable) == type)
                        {
                            return Variant.Type.Array;
                        }
                    }

                    break;
                }
            }

            // Unknown
            return Variant.Type.Nil;
        }

        /* TODO: Reflection and type checking each time is slow. This will be replaced with source generators. */

        public static godot_variant ConvertManagedObjectToVariant(object? p_obj)
        {
            if (p_obj == null)
                return new godot_variant();

            switch (p_obj)
            {
                case bool @bool:
                    return VariantUtils.CreateFromBool(@bool);
                case char @char:
                    return VariantUtils.CreateFromInt(@char);
                case sbyte @int8:
                    return VariantUtils.CreateFromInt(@int8);
                case Int16 @int16:
                    return VariantUtils.CreateFromInt(@int16);
                case Int32 @int32:
                    return VariantUtils.CreateFromInt(@int32);
                case Int64 @int64:
                    return VariantUtils.CreateFromInt(@int64);
                case byte @uint8:
                    return VariantUtils.CreateFromInt(@uint8);
                case UInt16 @uint16:
                    return VariantUtils.CreateFromInt(@uint16);
                case UInt32 @uint32:
                    return VariantUtils.CreateFromInt(@uint32);
                case UInt64 @uint64:
                    return VariantUtils.CreateFromInt(@uint64);
                case float @float:
                    return VariantUtils.CreateFromFloat(@float);
                case double @double:
                    return VariantUtils.CreateFromFloat(@double);
                case Vector2 @vector2:
                    return VariantUtils.CreateFromVector2(@vector2);
                case Vector2i @vector2i:
                    return VariantUtils.CreateFromVector2i(@vector2i);
                case Rect2 @rect2:
                    return VariantUtils.CreateFromRect2(@rect2);
                case Rect2i @rect2i:
                    return VariantUtils.CreateFromRect2i(@rect2i);
                case Transform2D @transform2D:
                    return VariantUtils.CreateFromTransform2D(@transform2D);
                case Vector3 @vector3:
                    return VariantUtils.CreateFromVector3(@vector3);
                case Vector3i @vector3i:
                    return VariantUtils.CreateFromVector3i(@vector3i);
                case Vector4 @vector4:
                    return VariantUtils.CreateFromVector4(@vector4);
                case Vector4i @vector4i:
                    return VariantUtils.CreateFromVector4i(@vector4i);
                case Basis @basis:
                    return VariantUtils.CreateFromBasis(@basis);
                case Quaternion @quaternion:
                    return VariantUtils.CreateFromQuaternion(@quaternion);
                case Transform3D @transform3d:
                    return VariantUtils.CreateFromTransform3D(@transform3d);
                case Projection @projection:
                    return VariantUtils.CreateFromProjection(@projection);
                case AABB @aabb:
                    return VariantUtils.CreateFromAABB(@aabb);
                case Color @color:
                    return VariantUtils.CreateFromColor(@color);
                case Plane @plane:
                    return VariantUtils.CreateFromPlane(@plane);
                case Callable @callable:
                    return VariantUtils.CreateFromCallable(@callable);
                case SignalInfo @signalInfo:
                    return VariantUtils.CreateFromSignalInfo(@signalInfo);
                case Enum @enum:
                    return VariantUtils.CreateFromInt(Convert.ToInt64(@enum));
                case string @string:
                    return VariantUtils.CreateFromString(@string);
                case byte[] byteArray:
                    return VariantUtils.CreateFromPackedByteArray(byteArray);
                case Int32[] int32Array:
                    return VariantUtils.CreateFromPackedInt32Array(int32Array);
                case Int64[] int64Array:
                    return VariantUtils.CreateFromPackedInt64Array(int64Array);
                case float[] floatArray:
                    return VariantUtils.CreateFromPackedFloat32Array(floatArray);
                case double[] doubleArray:
                    return VariantUtils.CreateFromPackedFloat64Array(doubleArray);
                case string[] stringArray:
                    return VariantUtils.CreateFromPackedStringArray(stringArray);
                case Vector2[] vector2Array:
                    return VariantUtils.CreateFromPackedVector2Array(vector2Array);
                case Vector3[] vector3Array:
                    return VariantUtils.CreateFromPackedVector3Array(vector3Array);
                case Color[] colorArray:
                    return VariantUtils.CreateFromPackedColorArray(colorArray);
                case StringName[] stringNameArray:
                    return VariantUtils.CreateFromSystemArrayOfSupportedType(stringNameArray);
                case NodePath[] nodePathArray:
                    return VariantUtils.CreateFromSystemArrayOfSupportedType(nodePathArray);
                case RID[] ridArray:
                    return VariantUtils.CreateFromSystemArrayOfSupportedType(ridArray);
                case Godot.Object[] godotObjectArray:
                    return VariantUtils.CreateFromSystemArrayOfGodotObject(godotObjectArray);
                case object[] objectArray: // Last one to avoid catching others like string[] and Godot.Object[]
                {
                    // The pattern match for `object[]` catches arrays on any reference type,
                    // so we need to check the actual type to make sure it's truly `object[]`.
                    if (objectArray.GetType() == typeof(object[]))
                        return VariantUtils.CreateFromSystemArrayOfVariant(objectArray);

                    GD.PushError("Attempted to convert a managed array of unmarshallable element type to Variant.");
                    return new godot_variant();
                }
                case Godot.Object godotObject:
                    return VariantUtils.CreateFromGodotObject(godotObject);
                case StringName stringName:
                    return VariantUtils.CreateFromStringName(stringName);
                case NodePath nodePath:
                    return VariantUtils.CreateFromNodePath(nodePath);
                case RID rid:
                    return VariantUtils.CreateFromRID(rid);
                case Collections.Dictionary godotDictionary:
                    return VariantUtils.CreateFromDictionary(godotDictionary);
                case Collections.Array godotArray:
                    return VariantUtils.CreateFromArray(godotArray);
                case Collections.IGenericGodotDictionary genericGodotDictionary:
                {
                    var godotDict = genericGodotDictionary.UnderlyingDictionary;
                    if (godotDict == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromDictionary(godotDict);
                }
                case Collections.IGenericGodotArray genericGodotArray:
                {
                    var godotArray = genericGodotArray.UnderlyingArray;
                    if (godotArray == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromArray(godotArray);
                }
                default:
                {
                    var type = p_obj.GetType();

                    if (type.IsGenericType)
                    {
                        var genericTypeDefinition = type.GetGenericTypeDefinition();

                        if (genericTypeDefinition == typeof(System.Collections.Generic.Dictionary<,>))
                        {
                            // TODO: Validate key and value types are compatible with Variant
                            var godotDict = new Collections.Dictionary();

                            foreach (KeyValuePair<object, object> entry in (IDictionary)p_obj)
                                godotDict.Add(entry.Key, entry.Value);

                            return VariantUtils.CreateFromDictionary(godotDict);
                        }

                        if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
                        {
                            // TODO: Validate element type is compatible with Variant
                            using var nativeGodotArray = ConvertICollectionToNativeGodotArray((ICollection)p_obj);
                            return VariantUtils.CreateFromArray(nativeGodotArray);
                        }
                    }

                    break;
                }
            }

            GD.PushError("Attempted to convert an unmarshallable managed type to Variant. Name: '" +
                         p_obj.GetType().FullName + ".");
            return new godot_variant();
        }

        public static object? ConvertVariantToManagedObjectOfType(in godot_variant p_var, Type type)
        {
            // This function is only needed to set the value of properties. Fields have their own implementation, set_value_from_variant.
            switch (Type.GetTypeCode(type))
            {
                case TypeCode.Boolean:
                    return VariantUtils.ConvertToBool(p_var);
                case TypeCode.Char:
                    return VariantUtils.ConvertToChar(p_var);
                case TypeCode.SByte:
                    return VariantUtils.ConvertToInt8(p_var);
                case TypeCode.Int16:
                    return VariantUtils.ConvertToInt16(p_var);
                case TypeCode.Int32:
                    return VariantUtils.ConvertToInt32(p_var);
                case TypeCode.Int64:
                    return VariantUtils.ConvertToInt64(p_var);
                case TypeCode.Byte:
                    return VariantUtils.ConvertToUInt8(p_var);
                case TypeCode.UInt16:
                    return VariantUtils.ConvertToUInt16(p_var);
                case TypeCode.UInt32:
                    return VariantUtils.ConvertToUInt32(p_var);
                case TypeCode.UInt64:
                    return VariantUtils.ConvertToUInt64(p_var);
                case TypeCode.Single:
                    return VariantUtils.ConvertToFloat32(p_var);
                case TypeCode.Double:
                    return VariantUtils.ConvertToFloat64(p_var);
                case TypeCode.String:
                    return VariantUtils.ConvertToStringObject(p_var);
                default:
                {
                    if (type == typeof(Vector2))
                        return VariantUtils.ConvertToVector2(p_var);

                    if (type == typeof(Vector2i))
                        return VariantUtils.ConvertToVector2i(p_var);

                    if (type == typeof(Rect2))
                        return VariantUtils.ConvertToRect2(p_var);

                    if (type == typeof(Rect2i))
                        return VariantUtils.ConvertToRect2i(p_var);

                    if (type == typeof(Transform2D))
                        return VariantUtils.ConvertToTransform2D(p_var);

                    if (type == typeof(Vector3))
                        return VariantUtils.ConvertToVector3(p_var);

                    if (type == typeof(Vector3i))
                        return VariantUtils.ConvertToVector3i(p_var);

                    if (type == typeof(Vector4))
                        return VariantUtils.ConvertToVector4(p_var);

                    if (type == typeof(Vector4i))
                        return VariantUtils.ConvertToVector4i(p_var);

                    if (type == typeof(Basis))
                        return VariantUtils.ConvertToBasis(p_var);

                    if (type == typeof(Quaternion))
                        return VariantUtils.ConvertToQuaternion(p_var);

                    if (type == typeof(Transform3D))
                        return VariantUtils.ConvertToTransform3D(p_var);

                    if (type == typeof(Projection))
                        return VariantUtils.ConvertToProjection(p_var);

                    if (type == typeof(AABB))
                        return VariantUtils.ConvertToAABB(p_var);

                    if (type == typeof(Color))
                        return VariantUtils.ConvertToColor(p_var);

                    if (type == typeof(Plane))
                        return VariantUtils.ConvertToPlane(p_var);

                    if (type == typeof(Callable))
                        return VariantUtils.ConvertToCallableManaged(p_var);

                    if (type == typeof(SignalInfo))
                        return VariantUtils.ConvertToSignalInfo(p_var);

                    if (type.IsEnum)
                    {
                        var enumUnderlyingType = type.GetEnumUnderlyingType();
                        switch (Type.GetTypeCode(enumUnderlyingType))
                        {
                            case TypeCode.SByte:
                                return VariantUtils.ConvertToInt8(p_var);
                            case TypeCode.Int16:
                                return VariantUtils.ConvertToInt16(p_var);
                            case TypeCode.Int32:
                                return VariantUtils.ConvertToInt32(p_var);
                            case TypeCode.Int64:
                                return VariantUtils.ConvertToInt64(p_var);
                            case TypeCode.Byte:
                                return VariantUtils.ConvertToUInt8(p_var);
                            case TypeCode.UInt16:
                                return VariantUtils.ConvertToUInt16(p_var);
                            case TypeCode.UInt32:
                                return VariantUtils.ConvertToUInt32(p_var);
                            case TypeCode.UInt64:
                                return VariantUtils.ConvertToUInt64(p_var);
                            default:
                            {
                                GD.PushError(
                                    "Attempted to convert Variant to enum value of unsupported underlying type. Name: " +
                                    type.FullName + " : " + enumUnderlyingType.FullName + ".");
                                return null;
                            }
                        }
                    }

                    if (type.IsArray || type.IsSZArray)
                        return ConvertVariantToSystemArrayOfType(p_var, type);
                    else if (type.IsGenericType)
                        return ConvertVariantToManagedObjectOfGenericType(p_var, type);
                    else if (type == typeof(object))
                        return ConvertVariantToManagedObject(p_var);
                    if (ConvertVariantToManagedObjectOfClass(p_var, type, out object? res))
                        return res;

                    break;
                }
            }

            GD.PushError("Attempted to convert Variant to unsupported type. Name: " +
                         type.FullName + ".");
            return null;
        }

        private static object? ConvertVariantToSystemArrayOfType(in godot_variant p_var, Type type)
        {
            if (type == typeof(byte[]))
                return VariantUtils.ConvertAsPackedByteArrayToSystemArray(p_var);

            if (type == typeof(Int32[]))
                return VariantUtils.ConvertAsPackedInt32ArrayToSystemArray(p_var);

            if (type == typeof(Int64[]))
                return VariantUtils.ConvertAsPackedInt64ArrayToSystemArray(p_var);

            if (type == typeof(float[]))
                return VariantUtils.ConvertAsPackedFloat32ArrayToSystemArray(p_var);

            if (type == typeof(double[]))
                return VariantUtils.ConvertAsPackedFloat64ArrayToSystemArray(p_var);

            if (type == typeof(string[]))
                return VariantUtils.ConvertAsPackedStringArrayToSystemArray(p_var);

            if (type == typeof(Vector2[]))
                return VariantUtils.ConvertAsPackedVector2ArrayToSystemArray(p_var);

            if (type == typeof(Vector3[]))
                return VariantUtils.ConvertAsPackedVector3ArrayToSystemArray(p_var);

            if (type == typeof(Color[]))
                return VariantUtils.ConvertAsPackedColorArrayToSystemArray(p_var);

            if (type == typeof(StringName[]))
                return VariantUtils.ConvertToSystemArrayOfSupportedType<StringName>(p_var);

            if (type == typeof(NodePath[]))
                return VariantUtils.ConvertToSystemArrayOfSupportedType<NodePath>(p_var);

            if (type == typeof(RID[]))
                return VariantUtils.ConvertToSystemArrayOfSupportedType<RID>(p_var);

            if (typeof(Godot.Object[]).IsAssignableFrom(type))
                return VariantUtils.ConvertToSystemArrayOfGodotObject(p_var, type);

            if (type == typeof(object[]))
                return VariantUtils.ConvertToSystemArrayOfVariant(p_var);

            GD.PushError("Attempted to convert Variant to array of unsupported element type. Name: " +
                         type.GetElementType()!.FullName + ".");
            return null;
        }

        private static bool ConvertVariantToManagedObjectOfClass(in godot_variant p_var, Type type,
            out object? res)
        {
            if (typeof(Godot.Object).IsAssignableFrom(type))
            {
                if (p_var.Type == Variant.Type.Nil)
                {
                    res = null;
                    return true;
                }

                if (p_var.Type != Variant.Type.Object)
                {
                    GD.PushError("Invalid cast when marshaling Godot.Object type." +
                                 $" Variant type is `{p_var.Type}`; expected `{p_var.Object}`.");
                    res = null;
                    return true;
                }

                var godotObjectPtr = VariantUtils.ConvertToGodotObjectPtr(p_var);

                if (godotObjectPtr == IntPtr.Zero)
                {
                    res = null;
                    return true;
                }

                var godotObject = InteropUtils.UnmanagedGetManaged(godotObjectPtr);

                if (!type.IsInstanceOfType(godotObject))
                {
                    GD.PushError("Invalid cast when marshaling Godot.Object type." +
                                 $" `{godotObject.GetType()}` is not assignable to `{type.FullName}`.");
                    res = null;
                    return false;
                }

                res = godotObject;
                return true;
            }

            if (typeof(StringName) == type)
            {
                res = VariantUtils.ConvertToStringNameObject(p_var);
                return true;
            }

            if (typeof(NodePath) == type)
            {
                res = VariantUtils.ConvertToNodePathObject(p_var);
                return true;
            }

            if (typeof(RID) == type)
            {
                res = VariantUtils.ConvertToRID(p_var);
                return true;
            }

            if (typeof(Collections.Dictionary) == type || typeof(System.Collections.IDictionary) == type)
            {
                res = VariantUtils.ConvertToDictionaryObject(p_var);
                return true;
            }

            if (typeof(Collections.Array) == type ||
                typeof(System.Collections.ICollection) == type ||
                typeof(System.Collections.IEnumerable) == type)
            {
                res = VariantUtils.ConvertToArrayObject(p_var);
                return true;
            }

            res = null;
            return false;
        }

        private static object? ConvertVariantToManagedObjectOfGenericType(in godot_variant p_var, Type type)
        {
            static object ConvertVariantToGenericGodotCollectionsDictionary(in godot_variant p_var, Type fullType)
            {
                var underlyingDict = VariantUtils.ConvertToDictionaryObject(p_var);
                return Activator.CreateInstance(fullType,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[] { underlyingDict }, null)!;
            }

            static object ConvertVariantToGenericGodotCollectionsArray(in godot_variant p_var, Type fullType)
            {
                var underlyingArray = VariantUtils.ConvertToArrayObject(p_var);
                return Activator.CreateInstance(fullType,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[] { underlyingArray }, null)!;
            }

            var genericTypeDefinition = type.GetGenericTypeDefinition();

            if (genericTypeDefinition == typeof(Collections.Dictionary<,>))
                return ConvertVariantToGenericGodotCollectionsDictionary(p_var, type);

            if (genericTypeDefinition == typeof(Collections.Array<>))
                return ConvertVariantToGenericGodotCollectionsArray(p_var, type);

            if (genericTypeDefinition == typeof(System.Collections.Generic.Dictionary<,>))
            {
                using var godotDictionary = VariantUtils.ConvertToDictionaryObject(p_var);

                var dictionary = (System.Collections.IDictionary)Activator.CreateInstance(type,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[]
                    {
                        /* capacity: */ godotDictionary.Count
                    }, null)!;

                foreach (KeyValuePair<object, object> pair in godotDictionary)
                    dictionary.Add(pair.Key, pair.Value);

                return dictionary;
            }

            if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
            {
                using var godotArray = VariantUtils.ConvertToArrayObject(p_var);

                var list = (System.Collections.IList)Activator.CreateInstance(type,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[]
                    {
                        /* capacity: */ godotArray.Count
                    }, null)!;

                foreach (object elem in godotArray)
                    list.Add(elem);

                return list;
            }

            if (genericTypeDefinition == typeof(IDictionary<,>))
            {
                var genericArgs = type.GetGenericArguments();
                var keyType = genericArgs[0];
                var valueType = genericArgs[1];
                var genericGodotDictionaryType = typeof(Collections.Dictionary<,>)
                    .MakeGenericType(keyType, valueType);

                return ConvertVariantToGenericGodotCollectionsDictionary(p_var, genericGodotDictionaryType);
            }

            if (genericTypeDefinition == typeof(ICollection<>) || genericTypeDefinition == typeof(IEnumerable<>))
            {
                var elementType = type.GetGenericArguments()[0];
                var genericGodotArrayType = typeof(Collections.Array<>)
                    .MakeGenericType(elementType);

                return ConvertVariantToGenericGodotCollectionsArray(p_var, genericGodotArrayType);
            }

            if (typeof(Godot.Object).IsAssignableFrom(type))
            {
                var godotObject = VariantUtils.ConvertToGodotObject(p_var);

                if (!type.IsInstanceOfType(godotObject))
                {
                    GD.PushError("Invalid cast when marshaling Godot.Object type." +
                                 $" `{godotObject.GetType()}` is not assignable to `{type.FullName}`.");
                    return null;
                }

                return godotObject;
            }

            return null;
        }

        public static unsafe object? ConvertVariantToManagedObject(in godot_variant p_var)
        {
            switch (p_var.Type)
            {
                case Variant.Type.Bool:
                    return p_var.Bool.ToBool();
                case Variant.Type.Int:
                    return p_var.Int;
                case Variant.Type.Float:
                {
#if REAL_T_IS_DOUBLE
                    return p_var.Float;
#else
                    return (float)p_var.Float;
#endif
                }
                case Variant.Type.String:
                    return ConvertStringToManaged(p_var.String);
                case Variant.Type.Vector2:
                    return p_var.Vector2;
                case Variant.Type.Vector2i:
                    return p_var.Vector2i;
                case Variant.Type.Rect2:
                    return p_var.Rect2;
                case Variant.Type.Rect2i:
                    return p_var.Rect2i;
                case Variant.Type.Vector3:
                    return p_var.Vector3;
                case Variant.Type.Vector3i:
                    return p_var.Vector3i;
                case Variant.Type.Transform2d:
                    return *p_var.Transform2D;
                case Variant.Type.Vector4:
                    return *p_var.Vector4;
                case Variant.Type.Vector4i:
                    return *p_var.Vector4i;
                case Variant.Type.Plane:
                    return p_var.Plane;
                case Variant.Type.Quaternion:
                    return p_var.Quaternion;
                case Variant.Type.Aabb:
                    return *p_var.AABB;
                case Variant.Type.Basis:
                    return *p_var.Basis;
                case Variant.Type.Transform3d:
                    return *p_var.Transform3D;
                case Variant.Type.Projection:
                    return *p_var.Projection;
                case Variant.Type.Color:
                    return p_var.Color;
                case Variant.Type.StringName:
                {
                    // The Variant owns the value, so we need to make a copy
                    return StringName.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_string_name_new_copy(p_var.StringName));
                }
                case Variant.Type.NodePath:
                {
                    // The Variant owns the value, so we need to make a copy
                    return NodePath.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_node_path_new_copy(p_var.NodePath));
                }
                case Variant.Type.Rid:
                    return p_var.RID;
                case Variant.Type.Object:
                    return InteropUtils.UnmanagedGetManaged(p_var.Object);
                case Variant.Type.Callable:
                    return ConvertCallableToManaged(p_var.Callable);
                case Variant.Type.Signal:
                    return ConvertSignalToManaged(p_var.Signal);
                case Variant.Type.Dictionary:
                {
                    // The Variant owns the value, so we need to make a copy
                    return Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_dictionary_new_copy(p_var.Dictionary));
                }
                case Variant.Type.Array:
                {
                    // The Variant owns the value, so we need to make a copy
                    return Collections.Array.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_array_new_copy(p_var.Array));
                }
                case Variant.Type.PackedByteArray:
                    return VariantUtils.ConvertAsPackedByteArrayToSystemArray(p_var);
                case Variant.Type.PackedInt32Array:
                    return VariantUtils.ConvertAsPackedInt32ArrayToSystemArray(p_var);
                case Variant.Type.PackedInt64Array:
                    return VariantUtils.ConvertAsPackedInt64ArrayToSystemArray(p_var);
                case Variant.Type.PackedFloat32Array:
                    return VariantUtils.ConvertAsPackedFloat32ArrayToSystemArray(p_var);
                case Variant.Type.PackedFloat64Array:
                    return VariantUtils.ConvertAsPackedFloat64ArrayToSystemArray(p_var);
                case Variant.Type.PackedStringArray:
                    return VariantUtils.ConvertAsPackedStringArrayToSystemArray(p_var);
                case Variant.Type.PackedVector2Array:
                    return VariantUtils.ConvertAsPackedVector2ArrayToSystemArray(p_var);
                case Variant.Type.PackedVector3Array:
                    return VariantUtils.ConvertAsPackedVector3ArrayToSystemArray(p_var);
                case Variant.Type.PackedColorArray:
                    return VariantUtils.ConvertAsPackedColorArrayToSystemArray(p_var);
                default:
                    return null;
            }
        }

        // String

        public static unsafe godot_string ConvertStringToNative(string? p_mono_string)
        {
            if (p_mono_string == null)
                return new godot_string();

            fixed (char* methodChars = p_mono_string)
            {
                NativeFuncs.godotsharp_string_new_with_utf16_chars(out godot_string dest, methodChars);
                return dest;
            }
        }

        public static unsafe string ConvertStringToManaged(in godot_string p_string)
        {
            if (p_string.Buffer == IntPtr.Zero)
                return string.Empty;

            const int sizeOfChar32 = 4;
            byte* bytes = (byte*)p_string.Buffer;
            int size = p_string.Size;
            if (size == 0)
                return string.Empty;
            size -= 1; // zero at the end
            int sizeInBytes = size * sizeOfChar32;
            return System.Text.Encoding.UTF32.GetString(bytes, sizeInBytes);
        }

        // Callable

        public static godot_callable ConvertCallableToNative(in Callable p_managed_callable)
        {
            if (p_managed_callable.Delegate != null)
            {
                var gcHandle = CustomGCHandle.AllocStrong(p_managed_callable.Delegate);
                NativeFuncs.godotsharp_callable_new_with_delegate(
                    GCHandle.ToIntPtr(gcHandle), out godot_callable callable);
                return callable;
            }
            else
            {
                godot_string_name method;

                if (p_managed_callable.Method != null && !p_managed_callable.Method.IsEmpty)
                {
                    var src = (godot_string_name)p_managed_callable.Method.NativeValue;
                    method = NativeFuncs.godotsharp_string_name_new_copy(src);
                }
                else
                {
                    method = default;
                }

                return new godot_callable(method /* Takes ownership of disposable */,
                    p_managed_callable.Target.GetInstanceId());
            }
        }

        public static Callable ConvertCallableToManaged(in godot_callable p_callable)
        {
            if (NativeFuncs.godotsharp_callable_get_data_for_marshalling(p_callable,
                    out IntPtr delegateGCHandle, out IntPtr godotObject,
                    out godot_string_name name).ToBool())
            {
                if (delegateGCHandle != IntPtr.Zero)
                {
                    return new Callable((Delegate?)GCHandle.FromIntPtr(delegateGCHandle).Target);
                }
                else
                {
                    return new Callable(
                        InteropUtils.UnmanagedGetManaged(godotObject),
                        StringName.CreateTakingOwnershipOfDisposableValue(name));
                }
            }

            // Some other unsupported callable
            return new Callable();
        }

        // SignalInfo

        public static godot_signal ConvertSignalToNative(in SignalInfo p_managed_signal)
        {
            ulong ownerId = p_managed_signal.Owner.GetInstanceId();
            godot_string_name name;

            if (p_managed_signal.Name != null && !p_managed_signal.Name.IsEmpty)
            {
                var src = (godot_string_name)p_managed_signal.Name.NativeValue;
                name = NativeFuncs.godotsharp_string_name_new_copy(src);
            }
            else
            {
                name = default;
            }

            return new godot_signal(name, ownerId);
        }

        public static SignalInfo ConvertSignalToManaged(in godot_signal p_signal)
        {
            var owner = GD.InstanceFromId(p_signal.ObjectId);
            var name = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(p_signal.Name));
            return new SignalInfo(owner, name);
        }

        // Array

        public static object[] ConvertNativeGodotArrayToSystemArray(in godot_array p_array)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            var ret = new object[length];

            array.CopyTo(ret, 0); // ConvertVariantToManagedObject handled by Collections.Array

            return ret;
        }

        internal static T[] ConvertNativeGodotArrayToSystemArrayOfType<T>(in godot_array p_array)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            var ret = new T[length];

            // ConvertVariantToManagedObjectOfType handled by Collections.Array.CopyToGeneric<T>
            array.CopyToGeneric(ret, 0);

            return ret;
        }

        internal static T[] ConvertNativeGodotArrayToSystemArrayOfGodotObjectType<T>(in godot_array p_array)
            where T : Godot.Object
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            var ret = new T[length];

            // ConvertVariantToManagedObjectOfType handled by Collections.Array.CopyToGeneric<T>
            array.CopyToGeneric(ret, 0);

            return ret;
        }

        internal static Godot.Object[] ConvertNativeGodotArrayToSystemArrayOfGodotObjectType(in godot_array p_array,
            Type type)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            var ret = (Godot.Object[])Activator.CreateInstance(type, length)!;

            // ConvertVariantToManagedObjectOfType handled by Collections.Array.CopyToGeneric<T>
            array.CopyToGeneric(ret, 0, type.GetElementType());

            return ret;
        }

        public static godot_array ConvertSystemArrayToNativeGodotArray(object[] p_array)
        {
            int length = p_array.Length;

            if (length == 0)
                return NativeFuncs.godotsharp_array_new();

            using var array = new Collections.Array();
            array.Resize(length);

            for (int i = 0; i < length; i++)
                array[i] = p_array[i];

            var src = (godot_array)array.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        public static godot_array ConvertSystemArrayToNativeGodotArray<T>(T[] p_array)
        {
            int length = p_array.Length;

            if (length == 0)
                return NativeFuncs.godotsharp_array_new();

            using var array = new Collections.Array();
            array.Resize(length);

            for (int i = 0; i < length; i++)
                array[i] = p_array[i];

            var src = (godot_array)array.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        public static godot_array ConvertICollectionToNativeGodotArray(ICollection p_array)
        {
            int length = p_array.Count;

            if (length == 0)
                return NativeFuncs.godotsharp_array_new();

            using var array = new Collections.Array();
            array.Resize(length);

            int i = 0;
            foreach (var elem in p_array)
            {
                array[i] = elem;
                i++;
            }

            var src = (godot_array)array.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        public static godot_array ConvertGenericICollectionToNativeGodotArray<T>(ICollection<T> p_array)
        {
            int length = p_array.Count;

            if (length == 0)
                return NativeFuncs.godotsharp_array_new();

            var array = new Collections.Array<T>();
            using var underlyingArray = (Collections.Array)array;
            array.Resize(length);

            int i = 0;
            foreach (var elem in p_array)
            {
                array[i] = elem;
                i++;
            }

            var src = (godot_array)underlyingArray.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        public static godot_array ConvertIEnumerableToNativeGodotArray(IEnumerable p_array)
        {
            using var array = new Collections.Array();

            foreach (var elem in p_array)
                array.Add(elem);

            var src = (godot_array)array.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        public static godot_array ConvertGenericIEnumerableToNativeGodotArray<T>(IEnumerable<T> p_array)
        {
            var array = new Collections.Array<T>();
            using var underlyingArray = (Collections.Array)array;

            foreach (var elem in p_array)
                array.Add(elem);

            var src = (godot_array)underlyingArray.NativeValue;
            return NativeFuncs.godotsharp_array_new_copy(src);
        }

        // PackedByteArray

        public static unsafe byte[] ConvertNativePackedByteArrayToSystemArray(in godot_packed_byte_array p_array)
        {
            byte* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<byte>();
            var array = new byte[size];
            fixed (byte* dest = array)
                Buffer.MemoryCopy(buffer, dest, size, size);
            return array;
        }

        public static unsafe godot_packed_byte_array ConvertSystemArrayToNativePackedByteArray(Span<byte> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_byte_array();
            fixed (byte* src = p_array)
                return NativeFuncs.godotsharp_packed_byte_array_new_mem_copy(src, p_array.Length);
        }

        // PackedInt32Array

        public static unsafe int[] ConvertNativePackedInt32ArrayToSystemArray(godot_packed_int32_array p_array)
        {
            int* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<int>();
            int sizeInBytes = size * sizeof(int);
            var array = new int[size];
            fixed (int* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_int32_array ConvertSystemArrayToNativePackedInt32Array(Span<int> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_int32_array();
            fixed (int* src = p_array)
                return NativeFuncs.godotsharp_packed_int32_array_new_mem_copy(src, p_array.Length);
        }

        // PackedInt64Array

        public static unsafe long[] ConvertNativePackedInt64ArrayToSystemArray(godot_packed_int64_array p_array)
        {
            long* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<long>();
            int sizeInBytes = size * sizeof(long);
            var array = new long[size];
            fixed (long* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_int64_array ConvertSystemArrayToNativePackedInt64Array(Span<long> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_int64_array();
            fixed (long* src = p_array)
                return NativeFuncs.godotsharp_packed_int64_array_new_mem_copy(src, p_array.Length);
        }

        // PackedFloat32Array

        public static unsafe float[] ConvertNativePackedFloat32ArrayToSystemArray(godot_packed_float32_array p_array)
        {
            float* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<float>();
            int sizeInBytes = size * sizeof(float);
            var array = new float[size];
            fixed (float* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_float32_array ConvertSystemArrayToNativePackedFloat32Array(
            Span<float> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_float32_array();
            fixed (float* src = p_array)
                return NativeFuncs.godotsharp_packed_float32_array_new_mem_copy(src, p_array.Length);
        }

        // PackedFloat64Array

        public static unsafe double[] ConvertNativePackedFloat64ArrayToSystemArray(godot_packed_float64_array p_array)
        {
            double* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<double>();
            int sizeInBytes = size * sizeof(double);
            var array = new double[size];
            fixed (double* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_float64_array ConvertSystemArrayToNativePackedFloat64Array(
            Span<double> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_float64_array();
            fixed (double* src = p_array)
                return NativeFuncs.godotsharp_packed_float64_array_new_mem_copy(src, p_array.Length);
        }

        // PackedStringArray

        public static unsafe string[] ConvertNativePackedStringArrayToSystemArray(godot_packed_string_array p_array)
        {
            godot_string* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<string>();
            var array = new string[size];
            for (int i = 0; i < size; i++)
                array[i] = ConvertStringToManaged(buffer[i]);
            return array;
        }

        public static godot_packed_string_array ConvertSystemArrayToNativePackedStringArray(Span<string> p_array)
        {
            godot_packed_string_array dest = new godot_packed_string_array();

            if (p_array.IsEmpty)
                return dest;

            /* TODO: Replace godotsharp_packed_string_array_add with a single internal call to
             get the write address. We can't use `dest._ptr` directly for writing due to COW. */

            for (int i = 0; i < p_array.Length; i++)
            {
                using godot_string godotStrElem = ConvertStringToNative(p_array[i]);
                NativeFuncs.godotsharp_packed_string_array_add(ref dest, godotStrElem);
            }

            return dest;
        }

        // PackedVector2Array

        public static unsafe Vector2[] ConvertNativePackedVector2ArrayToSystemArray(godot_packed_vector2_array p_array)
        {
            Vector2* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<Vector2>();
            int sizeInBytes = size * sizeof(Vector2);
            var array = new Vector2[size];
            fixed (Vector2* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_vector2_array ConvertSystemArrayToNativePackedVector2Array(
            Span<Vector2> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_vector2_array();
            fixed (Vector2* src = p_array)
                return NativeFuncs.godotsharp_packed_vector2_array_new_mem_copy(src, p_array.Length);
        }

        // PackedVector3Array

        public static unsafe Vector3[] ConvertNativePackedVector3ArrayToSystemArray(godot_packed_vector3_array p_array)
        {
            Vector3* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<Vector3>();
            int sizeInBytes = size * sizeof(Vector3);
            var array = new Vector3[size];
            fixed (Vector3* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_vector3_array ConvertSystemArrayToNativePackedVector3Array(
            Span<Vector3> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_vector3_array();
            fixed (Vector3* src = p_array)
                return NativeFuncs.godotsharp_packed_vector3_array_new_mem_copy(src, p_array.Length);
        }

        // PackedColorArray

        public static unsafe Color[] ConvertNativePackedColorArrayToSystemArray(godot_packed_color_array p_array)
        {
            Color* buffer = p_array.Buffer;
            int size = p_array.Size;
            if (size == 0)
                return Array.Empty<Color>();
            int sizeInBytes = size * sizeof(Color);
            var array = new Color[size];
            fixed (Color* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_color_array ConvertSystemArrayToNativePackedColorArray(Span<Color> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_color_array();
            fixed (Color* src = p_array)
                return NativeFuncs.godotsharp_packed_color_array_new_mem_copy(src, p_array.Length);
        }
    }
}
