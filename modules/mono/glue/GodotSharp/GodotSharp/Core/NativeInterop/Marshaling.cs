using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Runtime.InteropServices;

// ReSharper disable InconsistentNaming

// We want to use full name qualifiers here even if redundant for clarity
// ReSharper disable RedundantNameQualifier

#nullable enable

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

                    if (type == typeof(Basis))
                        return Variant.Type.Basis;

                    if (type == typeof(Quaternion))
                        return Variant.Type.Quaternion;

                    if (type == typeof(Transform3D))
                        return Variant.Type.Transform3d;

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
                            return Variant.Type.RawArray;

                        if (type == typeof(Int32[]))
                            return Variant.Type.Int32Array;

                        if (type == typeof(Int64[]))
                            return Variant.Type.Int64Array;

                        if (type == typeof(float[]))
                            return Variant.Type.Float32Array;

                        if (type == typeof(double[]))
                            return Variant.Type.Float64Array;

                        if (type == typeof(string[]))
                            return Variant.Type.StringArray;

                        if (type == typeof(Vector2[]))
                            return Variant.Type.Vector2Array;

                        if (type == typeof(Vector3[]))
                            return Variant.Type.Vector3Array;

                        if (type == typeof(Color[]))
                            return Variant.Type.ColorArray;

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
                case Basis @basis:
                    return VariantUtils.CreateFromBasis(@basis);
                case Quaternion @quaternion:
                    return VariantUtils.CreateFromQuaternion(@quaternion);
                case Transform3D @transform3d:
                    return VariantUtils.CreateFromTransform3D(@transform3d);
                case AABB @aabb:
                    return VariantUtils.CreateFromAABB(@aabb);
                case Color @color:
                    return VariantUtils.CreateFromColor(@color);
                case Plane @plane:
                    return VariantUtils.CreateFromPlane(@plane);
                case Callable @callable:
                    return VariantUtils.CreateFromCallableTakingOwnershipOfDisposableValue(
                        ConvertCallableToNative(ref @callable));
                case SignalInfo @signalInfo:
                    return VariantUtils.CreateFromSignalTakingOwnershipOfDisposableValue(
                        ConvertSignalToNative(ref @signalInfo));
                case Enum @enum:
                    return VariantUtils.CreateFromInt(Convert.ToInt64(@enum));
                case string @string:
                {
                    return VariantUtils.CreateFromStringTakingOwnershipOfDisposableValue(
                        ConvertStringToNative(@string));
                }
                case byte[] byteArray:
                {
                    using godot_packed_byte_array array = ConvertSystemArrayToNativePackedByteArray(byteArray);
                    return VariantUtils.CreateFromPackedByteArray(array);
                }
                case Int32[] int32Array:
                {
                    using godot_packed_int32_array array = ConvertSystemArrayToNativePackedInt32Array(int32Array);
                    return VariantUtils.CreateFromPackedInt32Array(array);
                }
                case Int64[] int64Array:
                {
                    using godot_packed_int64_array array = ConvertSystemArrayToNativePackedInt64Array(int64Array);
                    return VariantUtils.CreateFromPackedInt64Array(array);
                }
                case float[] floatArray:
                {
                    using godot_packed_float32_array array = ConvertSystemArrayToNativePackedFloat32Array(floatArray);
                    return VariantUtils.CreateFromPackedFloat32Array(array);
                }
                case double[] doubleArray:
                {
                    using godot_packed_float64_array array = ConvertSystemArrayToNativePackedFloat64Array(doubleArray);
                    return VariantUtils.CreateFromPackedFloat64Array(array);
                }
                case string[] stringArray:
                {
                    using godot_packed_string_array array = ConvertSystemArrayToNativePackedStringArray(stringArray);
                    return VariantUtils.CreateFromPackedStringArray(array);
                }
                case Vector2[] vector2Array:
                {
                    using godot_packed_vector2_array array = ConvertSystemArrayToNativePackedVector2Array(vector2Array);
                    return VariantUtils.CreateFromPackedVector2Array(array);
                }
                case Vector3[] vector3Array:
                {
                    using godot_packed_vector3_array array = ConvertSystemArrayToNativePackedVector3Array(vector3Array);
                    return VariantUtils.CreateFromPackedVector3Array(array);
                }
                case Color[] colorArray:
                {
                    using godot_packed_color_array array = ConvertSystemArrayToNativePackedColorArray(colorArray);
                    return VariantUtils.CreateFromPackedColorArray(array);
                }
                case Godot.Object[] godotObjectArray:
                {
                    // ReSharper disable once CoVariantArrayConversion
                    using godot_array array = ConvertSystemArrayToNativeGodotArray(godotObjectArray);
                    return VariantUtils.CreateFromArray(array);
                }
                case object[] objectArray: // Last one to avoid catching others like string[] and Godot.Object[]
                {
                    // The pattern match for `object[]` catches arrays on any reference type,
                    // so we need to check the actual type to make sure it's truly `object[]`.
                    if (objectArray.GetType() == typeof(object[]))
                    {
                        using godot_array array = ConvertSystemArrayToNativeGodotArray(objectArray);
                        return VariantUtils.CreateFromArray(array);
                    }

                    GD.PushError("Attempted to convert a managed array of unmarshallable element type to Variant.");
                    return new godot_variant();
                }
                case Godot.Object godotObject:
                    return VariantUtils.CreateFromGodotObject(godotObject.NativeInstance);
                case StringName stringName:
                    return VariantUtils.CreateFromStringName(stringName.NativeValue.DangerousSelfRef);
                case NodePath nodePath:
                    return VariantUtils.CreateFromNodePath((godot_node_path)nodePath.NativeValue);
                case RID rid:
                    return VariantUtils.CreateFromRID(rid);
                case Collections.Dictionary godotDictionary:
                    return VariantUtils.CreateFromDictionary((godot_dictionary)godotDictionary.NativeValue);
                case Collections.Array godotArray:
                    return VariantUtils.CreateFromArray((godot_array)godotArray.NativeValue);
                case Collections.IGenericGodotDictionary genericGodotDictionary:
                {
                    var godotDict = genericGodotDictionary.UnderlyingDictionary;
                    if (godotDict == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromDictionary((godot_dictionary)godotDict.NativeValue);
                }
                case Collections.IGenericGodotArray genericGodotArray:
                {
                    var godotArray = genericGodotArray.UnderlyingArray;
                    if (godotArray == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromArray((godot_array)godotArray.NativeValue);
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

                            return VariantUtils.CreateFromDictionary((godot_dictionary)godotDict.NativeValue);
                        }

                        if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
                        {
                            // TODO: Validate element type is compatible with Variant
                            using var nativeGodotArray = ConvertIListToNativeGodotArray((IList)p_obj);
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

        private static string? ConvertVariantToManagedString(in godot_variant p_var)
        {
            switch (p_var.Type)
            {
                case Variant.Type.Nil:
                    return null; // Otherwise, Variant -> String would return the string "Null"
                case Variant.Type.String:
                {
                    // We avoid the internal call if the stored type is the same we want.
                    return ConvertStringToManaged(p_var.String);
                }
                default:
                {
                    using godot_string godotString = NativeFuncs.godotsharp_variant_as_string(p_var);
                    return ConvertStringToManaged(godotString);
                }
            }
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
                    return ConvertVariantToManagedString(p_var);
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

                    if (type == typeof(Basis))
                        return VariantUtils.ConvertToBasis(p_var);

                    if (type == typeof(Quaternion))
                        return VariantUtils.ConvertToQuaternion(p_var);

                    if (type == typeof(Transform3D))
                        return VariantUtils.ConvertToTransform3D(p_var);

                    if (type == typeof(AABB))
                        return VariantUtils.ConvertToAABB(p_var);

                    if (type == typeof(Color))
                        return VariantUtils.ConvertToColor(p_var);

                    if (type == typeof(Plane))
                        return VariantUtils.ConvertToPlane(p_var);

                    if (type == typeof(Callable))
                    {
                        using godot_callable callable = NativeFuncs.godotsharp_variant_as_callable(p_var);
                        return ConvertCallableToManaged(in callable);
                    }

                    if (type == typeof(SignalInfo))
                    {
                        using godot_signal signal = NativeFuncs.godotsharp_variant_as_signal(p_var);
                        return ConvertSignalToManaged(in signal);
                    }

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
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
                return ConvertNativePackedByteArrayToSystemArray(packedArray);
            }

            if (type == typeof(Int32[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
                return ConvertNativePackedInt32ArrayToSystemArray(packedArray);
            }

            if (type == typeof(Int64[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
                return ConvertNativePackedInt64ArrayToSystemArray(packedArray);
            }

            if (type == typeof(float[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
                return ConvertNativePackedFloat32ArrayToSystemArray(packedArray);
            }

            if (type == typeof(double[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
                return ConvertNativePackedFloat64ArrayToSystemArray(packedArray);
            }

            if (type == typeof(string[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
                return ConvertNativePackedStringArrayToSystemArray(packedArray);
            }

            if (type == typeof(Vector2[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
                return ConvertNativePackedVector2ArrayToSystemArray(packedArray);
            }

            if (type == typeof(Vector3[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
                return ConvertNativePackedVector3ArrayToSystemArray(packedArray);
            }

            if (type == typeof(Color[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
                return ConvertNativePackedColorArrayToSystemArray(packedArray);
            }

            if (typeof(Godot.Object[]).IsAssignableFrom(type))
            {
                using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
                return ConvertNativeGodotArrayToSystemArrayOfType(godotArray, type);
            }

            if (type == typeof(object[]))
            {
                using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
                return ConvertNativeGodotArrayToSystemArray(godotArray);
            }

            GD.PushError("Attempted to convert Variant to array of unsupported element type. Name: " +
                         type.GetElementType()!.FullName + ".");
            return null;
        }

        private static bool ConvertVariantToManagedObjectOfClass(in godot_variant p_var, Type type,
            out object? res)
        {
            if (typeof(Godot.Object).IsAssignableFrom(type))
            {
                res = InteropUtils.UnmanagedGetManaged(VariantUtils.ConvertToGodotObject(p_var));
                return true;
            }

            if (typeof(StringName) == type)
            {
                res = StringName.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToStringName(p_var));
                return true;
            }

            if (typeof(NodePath) == type)
            {
                res = NodePath.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToNodePath(p_var));
                return true;
            }

            if (typeof(RID) == type)
            {
                res = VariantUtils.ConvertToRID(p_var);
                return true;
            }

            if (typeof(Collections.Dictionary) == type || typeof(System.Collections.IDictionary) == type)
            {
                res = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToDictionary(p_var));
                return true;
            }

            if (typeof(Collections.Array) == type ||
                typeof(System.Collections.ICollection) == type ||
                typeof(System.Collections.IEnumerable) == type)
            {
                res = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToArray(p_var));
                return true;
            }

            res = null;
            return false;
        }

        private static object? ConvertVariantToManagedObjectOfGenericType(in godot_variant p_var, Type type)
        {
            static object ConvertVariantToGenericGodotCollectionsDictionary(in godot_variant p_var, Type fullType)
            {
                var underlyingDict = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToDictionary(p_var));
                return Activator.CreateInstance(fullType,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[] { underlyingDict }, null)!;
            }

            static object ConvertVariantToGenericGodotCollectionsArray(in godot_variant p_var, Type fullType)
            {
                var underlyingArray = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToArray(p_var));
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
                using var godotDictionary = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToDictionary(p_var));

                var dictionary = (System.Collections.IDictionary)Activator.CreateInstance(type,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[]
                    {
                        /* capacity: */ godotDictionary.Count
                    }, null)!;

                foreach (System.Collections.DictionaryEntry pair in godotDictionary)
                    dictionary.Add(pair.Key, pair.Value);

                return dictionary;
            }

            if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
            {
                using var godotArray = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToArray(p_var));

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
                case Variant.Type.RawArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
                    return ConvertNativePackedByteArrayToSystemArray(packedArray);
                }
                case Variant.Type.Int32Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
                    return ConvertNativePackedInt32ArrayToSystemArray(packedArray);
                }
                case Variant.Type.Int64Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
                    return ConvertNativePackedInt64ArrayToSystemArray(packedArray);
                }
                case Variant.Type.Float32Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
                    return ConvertNativePackedFloat32ArrayToSystemArray(packedArray);
                }
                case Variant.Type.Float64Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
                    return ConvertNativePackedFloat64ArrayToSystemArray(packedArray);
                }
                case Variant.Type.StringArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
                    return ConvertNativePackedStringArrayToSystemArray(packedArray);
                }
                case Variant.Type.Vector2Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
                    return ConvertNativePackedVector2ArrayToSystemArray(packedArray);
                }
                case Variant.Type.Vector3Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
                    return ConvertNativePackedVector3ArrayToSystemArray(packedArray);
                }
                case Variant.Type.ColorArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
                    return ConvertNativePackedColorArrayToSystemArray(packedArray);
                }
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

        public static godot_callable ConvertCallableToNative(ref Callable p_managed_callable)
        {
            if (p_managed_callable.Delegate != null)
            {
                NativeFuncs.godotsharp_callable_new_with_delegate(
                    GCHandle.ToIntPtr(GCHandle.Alloc(p_managed_callable.Delegate)),
                    out godot_callable callable);
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

        public static godot_signal ConvertSignalToNative(ref SignalInfo p_managed_signal)
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

        private static object ConvertNativeGodotArrayToSystemArrayOfType(in godot_array p_array, Type type)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            object ret = Activator.CreateInstance(type, length)!;

            // ConvertVariantToManagedObject handled by Collections.Array
            // ConvertVariantToManagedObjectOfType is not needed because target element types are Godot.Object (or derived)
            array.CopyTo((object[])ret, 0);

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

        public static godot_array ConvertIListToNativeGodotArray(IList p_array)
        {
            int length = p_array.Count;

            if (length == 0)
                return NativeFuncs.godotsharp_array_new();

            using var array = new Collections.Array();
            array.Resize(length);

            for (int i = 0; i < length; i++)
                array[i] = p_array[i];

            var src = (godot_array)array.NativeValue;
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
