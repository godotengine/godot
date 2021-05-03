using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Runtime.InteropServices;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    // We want to use full name qualifiers here even if redundant for clarity
    [SuppressMessage("ReSharper", "RedundantNameQualifier")]
    internal static class Marshaling
    {
        public static unsafe void SetFieldValue(FieldInfo fieldInfo, object obj, godot_variant* value)
        {
            var valueObj = variant_to_mono_object_of_type(value, fieldInfo.FieldType);
            fieldInfo.SetValue(obj, valueObj);
        }

        public static Variant.Type managed_to_variant_type(Type type, ref bool r_nil_is_variant)
        {
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
                        if (type == typeof(Byte[]))
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

                        if (genericTypeDefinition == typeof(ICollection<>) || genericTypeDefinition == typeof(IEnumerable<>))
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

            r_nil_is_variant = false;

            // Unknown
            return Variant.Type.Nil;
        }

        public static bool try_get_array_element_type(Type p_array_type, out Type r_elem_type)
        {
            if (p_array_type.IsArray || p_array_type.IsSZArray)
            {
                r_elem_type = p_array_type.GetElementType();
                return true;
            }
            else if (p_array_type.IsGenericType)
            {
                var genericTypeDefinition = p_array_type.GetGenericTypeDefinition();

                if (typeof(Collections.Array) == genericTypeDefinition ||
                    typeof(System.Collections.Generic.List<>) == genericTypeDefinition ||
                    typeof(System.Collections.ICollection) == genericTypeDefinition ||
                    typeof(System.Collections.IEnumerable) == genericTypeDefinition)
                {
                    r_elem_type = p_array_type.GetGenericArguments()[0];
                    return true;
                }
            }

            r_elem_type = null;
            return false;
        }

        /* TODO: Reflection and type checking each time is slow. This will be replaced with source generators. */

        public static godot_variant mono_object_to_variant(object p_obj)
        {
            return mono_object_to_variant_impl(p_obj);
        }

        public static godot_variant mono_object_to_variant_no_err(object p_obj)
        {
            return mono_object_to_variant_impl(p_obj);
        }

        // TODO: Only called from C++. Remove once no longer needed.
        private static unsafe void mono_object_to_variant_out(object p_obj, bool p_fail_with_err, godot_variant* r_ret)
            => *r_ret = mono_object_to_variant_impl(p_obj, p_fail_with_err);

        private static unsafe godot_variant mono_object_to_variant_impl(object p_obj, bool p_fail_with_err = true)
        {
            if (p_obj == null)
                return new godot_variant();

            switch (p_obj)
            {
                case bool @bool:
                    return VariantUtils.CreateFromBool(@bool);
                case char @char:
                    return VariantUtils.CreateFromInt(@char);
                case SByte @int8:
                    return VariantUtils.CreateFromInt(@int8);
                case Int16 @int16:
                    return VariantUtils.CreateFromInt(@int16);
                case Int32 @int32:
                    return VariantUtils.CreateFromInt(@int32);
                case Int64 @int64:
                    return VariantUtils.CreateFromInt(@int64);
                case Byte @uint8:
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
                        mono_string_to_godot(@string));
                }
                case Byte[] byteArray:
                {
                    using godot_packed_byte_array array = mono_array_to_PackedByteArray(byteArray);
                    return VariantUtils.CreateFromPackedByteArray(&array);
                }
                case Int32[] int32Array:
                {
                    using godot_packed_int32_array array = mono_array_to_PackedInt32Array(int32Array);
                    return VariantUtils.CreateFromPackedInt32Array(&array);
                }
                case Int64[] int64Array:
                {
                    using godot_packed_int64_array array = mono_array_to_PackedInt64Array(int64Array);
                    return VariantUtils.CreateFromPackedInt64Array(&array);
                }
                case float[] floatArray:
                {
                    using godot_packed_float32_array array = mono_array_to_PackedFloat32Array(floatArray);
                    return VariantUtils.CreateFromPackedFloat32Array(&array);
                }
                case double[] doubleArray:
                {
                    using godot_packed_float64_array array = mono_array_to_PackedFloat64Array(doubleArray);
                    return VariantUtils.CreateFromPackedFloat64Array(&array);
                }
                case string[] stringArray:
                {
                    using godot_packed_string_array array = mono_array_to_PackedStringArray(stringArray);
                    return VariantUtils.CreateFromPackedStringArray(&array);
                }
                case Vector2[] vector2Array:
                {
                    using godot_packed_vector2_array array = mono_array_to_PackedVector2Array(vector2Array);
                    return VariantUtils.CreateFromPackedVector2Array(&array);
                }
                case Vector3[] vector3Array:
                {
                    using godot_packed_vector3_array array = mono_array_to_PackedVector3Array(vector3Array);
                    return VariantUtils.CreateFromPackedVector3Array(&array);
                }
                case Color[] colorArray:
                {
                    using godot_packed_color_array array = mono_array_to_PackedColorArray(colorArray);
                    return VariantUtils.CreateFromPackedColorArray(&array);
                }
                case Godot.Object[] godotObjectArray:
                {
                    // ReSharper disable once CoVariantArrayConversion
                    using godot_array array = mono_array_to_Array(godotObjectArray);
                    return VariantUtils.CreateFromArray(&array);
                }
                case object[] objectArray: // Last one to avoid catching others like string[] and Godot.Object[]
                {
                    // The pattern match for `object[]` catches arrays on any reference type,
                    // so we need to check the actual type to make sure it's truly `object[]`.
                    if (objectArray.GetType() == typeof(object[]))
                    {
                        using godot_array array = mono_array_to_Array(objectArray);
                        return VariantUtils.CreateFromArray(&array);
                    }

                    if (p_fail_with_err)
                    {
                        GD.PushError("Attempted to convert a managed array of unmarshallable element type to Variant.");
                        return new godot_variant();
                    }
                    else
                    {
                        return new godot_variant();
                    }
                }
                case Godot.Object godotObject:
                    return VariantUtils.CreateFromGodotObject(godotObject.NativeInstance);
                case StringName stringName:
                    return VariantUtils.CreateFromStringName(ref stringName.NativeValue);
                case NodePath nodePath:
                    return VariantUtils.CreateFromNodePath(ref nodePath.NativeValue);
                case RID rid:
                    return VariantUtils.CreateFromRID(rid);
                case Collections.Dictionary godotDictionary:
                    return VariantUtils.CreateFromDictionary(godotDictionary.NativeValue);
                case Collections.Array godotArray:
                    return VariantUtils.CreateFromArray(godotArray.NativeValue);
                case Collections.IGenericGodotDictionary genericGodotDictionary:
                {
                    var godotDict = genericGodotDictionary.UnderlyingDictionary;
                    if (godotDict == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromDictionary(godotDict.NativeValue);
                }
                case Collections.IGenericGodotArray genericGodotArray:
                {
                    var godotArray = genericGodotArray.UnderlyingArray;
                    if (godotArray == null)
                        return new godot_variant();
                    return VariantUtils.CreateFromArray(godotArray.NativeValue);
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
#if NET
                            Collections.IGenericGodotDictionary genericGodotDictionary = IDictionaryToGenericGodotDictionary((dynamic)p_obj);
#else
                            var genericArguments = type.GetGenericArguments();

                            // With .NET Standard we need a package reference for Microsoft.CSharp in order to
                            // use dynamic, so we have this workaround for now until we switch to .NET 5/6.
                            var method = typeof(Marshaling).GetMethod(nameof(IDictionaryToGenericGodotDictionary),
                                    BindingFlags.NonPublic | BindingFlags.Static | BindingFlags.DeclaredOnly)!
                                .MakeGenericMethod(genericArguments[0], genericArguments[1]);

                            var genericGodotDictionary = (Collections.IGenericGodotDictionary)method
                                .Invoke(null, new[] {p_obj});
#endif

                            var godotDict = genericGodotDictionary.UnderlyingDictionary;
                            if (godotDict == null)
                                return new godot_variant();
                            return VariantUtils.CreateFromDictionary(godotDict.NativeValue);
                        }

                        if (genericTypeDefinition == typeof(System.Collections.Generic.List<>))
                        {
                            // TODO: Validate element type is compatible with Variant
#if NET
                            var nativeGodotArray = mono_array_to_Array(System.Runtime.InteropServices.CollectionsMarshal.AsSpan((dynamic)p_obj));
#else
                            // With .NET Standard we need a package reference for Microsoft.CSharp in order to
                            // use dynamic, so we have this workaround for now until we switch to .NET 5/6.
                            // Also CollectionsMarshal.AsSpan is not available with .NET Standard.

                            var collection = (System.Collections.ICollection)p_obj;
                            var array = new object[collection.Count];
                            collection.CopyTo(array, 0);
                            var nativeGodotArray = mono_array_to_Array(array);
#endif
                            return VariantUtils.CreateFromArray(&nativeGodotArray);
                        }
                    }

                    break;
                }
            }

            if (p_fail_with_err)
            {
                GD.PushError("Attempted to convert an unmarshallable managed type to Variant. Name: '" +
                             p_obj.GetType().FullName + ".");
                return new godot_variant();
            }
            else
            {
                return new godot_variant();
            }
        }

        private static Collections.Dictionary<TKey, TValue> IDictionaryToGenericGodotDictionary<TKey, TValue>
            (IDictionary<TKey, TValue> dictionary) => new(dictionary);

        public static unsafe string variant_to_mono_string(godot_variant* p_var)
        {
            switch ((*p_var)._type)
            {
                case Variant.Type.Nil:
                    return null; // Otherwise, Variant -> String would return the string "Null"
                case Variant.Type.String:
                {
                    // We avoid the internal call if the stored type is the same we want.
                    return mono_string_from_godot(&(*p_var)._data._m_string);
                }
                default:
                {
                    using godot_string godotString = NativeFuncs.godotsharp_variant_as_string(p_var);
                    return mono_string_from_godot(&godotString);
                }
            }
        }

        public static unsafe object variant_to_mono_object_of_type(godot_variant* p_var, Type type)
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
                    return variant_to_mono_string(p_var);
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
                        return ConvertCallableToManaged(&callable);
                    }

                    if (type == typeof(SignalInfo))
                    {
                        using godot_signal signal = NativeFuncs.godotsharp_variant_as_signal(p_var);
                        return ConvertSignalToManaged(&signal);
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
                                GD.PushError("Attempted to convert Variant to enum value of unsupported underlying type. Name: " +
                                             type.FullName + " : " + enumUnderlyingType.FullName + ".");
                                return null;
                            }
                        }
                    }

                    if (type.IsArray || type.IsSZArray)
                        return variant_to_mono_array_of_type(p_var, type);
                    else if (type.IsGenericType)
                        return variant_to_mono_object_of_genericinst(p_var, type);
                    else if (type == typeof(object))
                        return variant_to_mono_object(p_var);
                    if (variant_to_mono_object_of_class(p_var, type, out object res))
                        return res;

                    break;
                }
            }

            GD.PushError("Attempted to convert Variant to unsupported type. Name: " +
                         type.FullName + ".");
            return null;
        }

        private static unsafe object variant_to_mono_array_of_type(godot_variant* p_var, Type type)
        {
            if (type == typeof(Byte[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
                return PackedByteArray_to_mono_array(&packedArray);
            }

            if (type == typeof(Int32[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
                return PackedInt32Array_to_mono_array(&packedArray);
            }

            if (type == typeof(Int64[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
                return PackedInt64Array_to_mono_array(&packedArray);
            }

            if (type == typeof(float[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
                return PackedFloat32Array_to_mono_array(&packedArray);
            }

            if (type == typeof(double[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
                return PackedFloat64Array_to_mono_array(&packedArray);
            }

            if (type == typeof(string[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
                return PackedStringArray_to_mono_array(&packedArray);
            }

            if (type == typeof(Vector2[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
                return PackedVector2Array_to_mono_array(&packedArray);
            }

            if (type == typeof(Vector3[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
                return PackedVector3Array_to_mono_array(&packedArray);
            }

            if (type == typeof(Color[]))
            {
                using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
                return PackedColorArray_to_mono_array(&packedArray);
            }

            if (typeof(Godot.Object[]).IsAssignableFrom(type))
            {
                using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
                return Array_to_mono_array_of_type(&godotArray, type);
            }

            if (type == typeof(object[]))
            {
                using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
                return Array_to_mono_array(&godotArray);
            }

            GD.PushError("Attempted to convert Variant to array of unsupported element type. Name: " +
                         type.GetElementType()!.FullName + ".");
            return null;
        }

        private static unsafe bool variant_to_mono_object_of_class(godot_variant* p_var, Type type, out object res)
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

        private static unsafe object variant_to_mono_object_of_genericinst(godot_variant* p_var, Type type)
        {
            static object variant_to_generic_godot_collections_dictionary(godot_variant* p_var, Type fullType)
            {
                var underlyingDict = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToDictionary(p_var));
                return Activator.CreateInstance(fullType,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[] {underlyingDict}, null);
            }

            static object variant_to_generic_godot_collections_array(godot_variant* p_var, Type fullType)
            {
                var underlyingArray = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToArray(p_var));
                return Activator.CreateInstance(fullType,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[] {underlyingArray}, null);
            }

            var genericTypeDefinition = type.GetGenericTypeDefinition();

            if (genericTypeDefinition == typeof(Collections.Dictionary<,>))
                return variant_to_generic_godot_collections_dictionary(p_var, type);

            if (genericTypeDefinition == typeof(Collections.Array<>))
                return variant_to_generic_godot_collections_array(p_var, type);

            if (genericTypeDefinition == typeof(System.Collections.Generic.Dictionary<,>))
            {
                using var godotDictionary = Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                    VariantUtils.ConvertToDictionary(p_var));

                var dictionary = (System.Collections.IDictionary)Activator.CreateInstance(type,
                    BindingFlags.Public | BindingFlags.Instance, null,
                    args: new object[]
                    {
                        /* capacity: */ godotDictionary.Count
                    }, null);

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
                    }, null);

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

                return variant_to_generic_godot_collections_dictionary(p_var, genericGodotDictionaryType);
            }

            if (genericTypeDefinition == typeof(ICollection<>) || genericTypeDefinition == typeof(IEnumerable<>))
            {
                var elementType = type.GetGenericArguments()[0];
                var genericGodotArrayType = typeof(Collections.Array<>)
                    .MakeGenericType(elementType);

                return variant_to_generic_godot_collections_array(p_var, genericGodotArrayType);
            }

            return null;
        }

        public static unsafe object variant_to_mono_object(godot_variant* p_var)
        {
            switch ((*p_var)._type)
            {
                case Variant.Type.Bool:
                    return (bool)(*p_var)._data._bool;
                case Variant.Type.Int:
                    return (*p_var)._data._int;
                case Variant.Type.Float:
                {
#if REAL_T_IS_DOUBLE
                    return (*p_var)._data._float;
#else
                    return (float)(*p_var)._data._float;
#endif
                }
                case Variant.Type.String:
                    return mono_string_from_godot(&(*p_var)._data._m_string);
                case Variant.Type.Vector2:
                    return (*p_var)._data._m_vector2;
                case Variant.Type.Vector2i:
                    return (*p_var)._data._m_vector2i;
                case Variant.Type.Rect2:
                    return (*p_var)._data._m_rect2;
                case Variant.Type.Rect2i:
                    return (*p_var)._data._m_rect2i;
                case Variant.Type.Vector3:
                    return (*p_var)._data._m_vector3;
                case Variant.Type.Vector3i:
                    return (*p_var)._data._m_vector3i;
                case Variant.Type.Transform2d:
                    return *(*p_var)._data._transform2d;
                case Variant.Type.Plane:
                    return (*p_var)._data._m_plane;
                case Variant.Type.Quaternion:
                    return (*p_var)._data._m_quaternion;
                case Variant.Type.Aabb:
                    return *(*p_var)._data._aabb;
                case Variant.Type.Basis:
                    return *(*p_var)._data._basis;
                case Variant.Type.Transform3d:
                    return *(*p_var)._data._transform3d;
                case Variant.Type.Color:
                    return (*p_var)._data._m_color;
                case Variant.Type.StringName:
                {
                    // The Variant owns the value, so we need to make a copy
                    return StringName.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_string_name_new_copy(&(*p_var)._data._m_string_name));
                }
                case Variant.Type.NodePath:
                {
                    // The Variant owns the value, so we need to make a copy
                    return NodePath.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_node_path_new_copy(&(*p_var)._data._m_node_path));
                }
                case Variant.Type.Rid:
                    return (*p_var)._data._m_rid;
                case Variant.Type.Object:
                    return InteropUtils.UnmanagedGetManaged((*p_var)._data._m_obj_data.obj);
                case Variant.Type.Callable:
                    return ConvertCallableToManaged(&(*p_var)._data._m_callable);
                case Variant.Type.Signal:
                    return ConvertSignalToManaged(&(*p_var)._data._m_signal);
                case Variant.Type.Dictionary:
                {
                    // The Variant owns the value, so we need to make a copy
                    return Collections.Dictionary.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_dictionary_new_copy(&(*p_var)._data._m_dictionary));
                }
                case Variant.Type.Array:
                {
                    // The Variant owns the value, so we need to make a copy
                    return Collections.Array.CreateTakingOwnershipOfDisposableValue(
                        NativeFuncs.godotsharp_array_new_copy(&(*p_var)._data._m_array));
                }
                case Variant.Type.RawArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
                    return PackedByteArray_to_mono_array(&packedArray);
                }
                case Variant.Type.Int32Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
                    return PackedInt32Array_to_mono_array(&packedArray);
                }
                case Variant.Type.Int64Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
                    return PackedInt64Array_to_mono_array(&packedArray);
                }
                case Variant.Type.Float32Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
                    return PackedFloat32Array_to_mono_array(&packedArray);
                }
                case Variant.Type.Float64Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
                    return PackedFloat64Array_to_mono_array(&packedArray);
                }
                case Variant.Type.StringArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
                    return PackedStringArray_to_mono_array(&packedArray);
                }
                case Variant.Type.Vector2Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
                    return PackedVector2Array_to_mono_array(&packedArray);
                }
                case Variant.Type.Vector3Array:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
                    return PackedVector3Array_to_mono_array(&packedArray);
                }
                case Variant.Type.ColorArray:
                {
                    using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
                    return PackedColorArray_to_mono_array(&packedArray);
                }
                default:
                    return null;
            }
        }

        // String

        public static unsafe godot_string mono_string_to_godot(string p_mono_string)
        {
            if (p_mono_string == null)
                return new godot_string();

            fixed (char* methodChars = p_mono_string)
            {
                godot_string dest;
                NativeFuncs.godotsharp_string_new_with_utf16_chars(&dest, methodChars);
                return dest;
            }
        }

        public static unsafe string mono_string_from_godot(godot_string* p_string)
        {
            if ((*p_string)._ptr == IntPtr.Zero)
                return string.Empty;

            const int sizeOfChar32 = 4;
            byte* bytes = (byte*)(*p_string)._ptr;
            int size = *((int*)(*p_string)._ptr - 1);
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
                unsafe
                {
                    godot_callable callable;
                    NativeFuncs.godotsharp_callable_new_with_delegate(
                        GCHandle.ToIntPtr(GCHandle.Alloc(p_managed_callable.Delegate)), &callable);
                    return callable;
                }
            }
            else
            {
                unsafe
                {
                    godot_string_name method;

                    if (p_managed_callable.Method != null && !p_managed_callable.Method.IsEmpty)
                    {
                        godot_string_name src = p_managed_callable.Method.NativeValue;
                        method = NativeFuncs.godotsharp_string_name_new_copy(&src);
                    }
                    else
                    {
                        method = default;
                    }

                    return new godot_callable
                    {
                        _method = method, // Takes ownership of disposable
                        _objectId = p_managed_callable.Target.GetInstanceId()
                    };
                }
            }
        }

        public static unsafe Callable ConvertCallableToManaged(godot_callable* p_callable)
        {
            IntPtr delegateGCHandle;
            IntPtr godotObject;
            godot_string_name name;

            if (NativeFuncs.godotsharp_callable_get_data_for_marshalling(
                p_callable, &delegateGCHandle, &godotObject, &name))
            {
                if (delegateGCHandle != IntPtr.Zero)
                {
                    return new Callable((Delegate)GCHandle.FromIntPtr(delegateGCHandle).Target);
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
            unsafe
            {
                godot_string_name name;

                if (p_managed_signal.Name != null && !p_managed_signal.Name.IsEmpty)
                {
                    godot_string_name src = p_managed_signal.Name.NativeValue;
                    name = NativeFuncs.godotsharp_string_name_new_copy(&src);
                }
                else
                {
                    name = default;
                }

                return new godot_signal()
                {
                    _name = name,
                    _objectId = ownerId
                };
            }
        }

        public static unsafe SignalInfo ConvertSignalToManaged(godot_signal* p_signal)
        {
            var owner = GD.InstanceFromId((*p_signal)._objectId);
            var name = StringName.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_string_name_new_copy(&(*p_signal)._name));
            return new SignalInfo(owner, name);
        }

        // Array

        public static unsafe object[] Array_to_mono_array(godot_array* p_array)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            var ret = new object[length];

            array.CopyTo(ret, 0); // variant_to_mono_object handled by Collections.Array

            return ret;
        }

        public static unsafe object Array_to_mono_array_of_type(godot_array* p_array, Type type)
        {
            var array = Collections.Array.CreateTakingOwnershipOfDisposableValue(
                NativeFuncs.godotsharp_array_new_copy(p_array));

            int length = array.Count;
            object ret = Activator.CreateInstance(type, length);

            array.CopyTo((object[])ret, 0); // variant_to_mono_object handled by Collections.Array

            return ret;
        }

        public static godot_array mono_array_to_Array(Span<object> p_array)
        {
            if (p_array.IsEmpty)
            {
                godot_array ret;
                Collections.Array.godot_icall_Array_Ctor(out ret);
                return ret;
            }

            using var array = new Collections.Array();
            array.Resize(p_array.Length);

            for (int i = 0; i < p_array.Length; i++)
                array[i] = p_array[i];

            godot_array src = array.NativeValue;
            unsafe
            {
                return NativeFuncs.godotsharp_array_new_copy(&src);
            }
        }

        // PackedByteArray

        public static unsafe byte[] PackedByteArray_to_mono_array(godot_packed_byte_array* p_array)
        {
            byte* buffer = (byte*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            var array = new byte[size];
            fixed (byte* dest = array)
                Buffer.MemoryCopy(buffer, dest, size, size);
            return array;
        }

        public static unsafe godot_packed_byte_array mono_array_to_PackedByteArray(Span<byte> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_byte_array();
            fixed (byte* src = p_array)
                return NativeFuncs.godotsharp_packed_byte_array_new_mem_copy(src, p_array.Length);
        }

        // PackedInt32Array

        public static unsafe int[] PackedInt32Array_to_mono_array(godot_packed_int32_array* p_array)
        {
            int* buffer = (int*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(int);
            var array = new int[size];
            fixed (int* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_int32_array mono_array_to_PackedInt32Array(Span<int> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_int32_array();
            fixed (int* src = p_array)
                return NativeFuncs.godotsharp_packed_int32_array_new_mem_copy(src, p_array.Length);
        }

        // PackedInt64Array

        public static unsafe long[] PackedInt64Array_to_mono_array(godot_packed_int64_array* p_array)
        {
            long* buffer = (long*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(long);
            var array = new long[size];
            fixed (long* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_int64_array mono_array_to_PackedInt64Array(Span<long> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_int64_array();
            fixed (long* src = p_array)
                return NativeFuncs.godotsharp_packed_int64_array_new_mem_copy(src, p_array.Length);
        }

        // PackedFloat32Array

        public static unsafe float[] PackedFloat32Array_to_mono_array(godot_packed_float32_array* p_array)
        {
            float* buffer = (float*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(float);
            var array = new float[size];
            fixed (float* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_float32_array mono_array_to_PackedFloat32Array(Span<float> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_float32_array();
            fixed (float* src = p_array)
                return NativeFuncs.godotsharp_packed_float32_array_new_mem_copy(src, p_array.Length);
        }

        // PackedFloat64Array

        public static unsafe double[] PackedFloat64Array_to_mono_array(godot_packed_float64_array* p_array)
        {
            double* buffer = (double*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(double);
            var array = new double[size];
            fixed (double* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_float64_array mono_array_to_PackedFloat64Array(Span<double> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_float64_array();
            fixed (double* src = p_array)
                return NativeFuncs.godotsharp_packed_float64_array_new_mem_copy(src, p_array.Length);
        }

        // PackedStringArray

        public static unsafe string[] PackedStringArray_to_mono_array(godot_packed_string_array* p_array)
        {
            godot_string* buffer = (godot_string*)(*p_array)._ptr;
            if (buffer == null)
                return new string[] { };
            int size = *((int*)(*p_array)._ptr - 1);
            var array = new string[size];
            for (int i = 0; i < size; i++)
                array[i] = mono_string_from_godot(&buffer[i]);
            return array;
        }

        public static unsafe godot_packed_string_array mono_array_to_PackedStringArray(Span<string> p_array)
        {
            godot_packed_string_array dest = new godot_packed_string_array();

            if (p_array.IsEmpty)
                return dest;

            /* TODO: Replace godotsharp_packed_string_array_add with a single internal call to
             get the write address. We can't use `dest._ptr` directly for writing due to COW. */

            for (int i = 0; i < p_array.Length; i++)
            {
                using godot_string godotStrElem = mono_string_to_godot(p_array[i]);
                NativeFuncs.godotsharp_packed_string_array_add(&dest, &godotStrElem);
            }

            return dest;
        }

        // PackedVector2Array

        public static unsafe Vector2[] PackedVector2Array_to_mono_array(godot_packed_vector2_array* p_array)
        {
            Vector2* buffer = (Vector2*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(Vector2);
            var array = new Vector2[size];
            fixed (Vector2* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_vector2_array mono_array_to_PackedVector2Array(Span<Vector2> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_vector2_array();
            fixed (Vector2* src = p_array)
                return NativeFuncs.godotsharp_packed_vector2_array_new_mem_copy(src, p_array.Length);
        }

        // PackedVector3Array

        public static unsafe Vector3[] PackedVector3Array_to_mono_array(godot_packed_vector3_array* p_array)
        {
            Vector3* buffer = (Vector3*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(Vector3);
            var array = new Vector3[size];
            fixed (Vector3* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_vector3_array mono_array_to_PackedVector3Array(Span<Vector3> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_vector3_array();
            fixed (Vector3* src = p_array)
                return NativeFuncs.godotsharp_packed_vector3_array_new_mem_copy(src, p_array.Length);
        }

        // PackedColorArray

        public static unsafe Color[] PackedColorArray_to_mono_array(godot_packed_color_array* p_array)
        {
            Color* buffer = (Color*)(*p_array)._ptr;
            int size = *((int*)(*p_array)._ptr - 1);
            int sizeInBytes = size * sizeof(Color);
            var array = new Color[size];
            fixed (Color* dest = array)
                Buffer.MemoryCopy(buffer, dest, sizeInBytes, sizeInBytes);
            return array;
        }

        public static unsafe godot_packed_color_array mono_array_to_PackedColorArray(Span<Color> p_array)
        {
            if (p_array.IsEmpty)
                return new godot_packed_color_array();
            fixed (Color* src = p_array)
                return NativeFuncs.godotsharp_packed_color_array_new_mem_copy(src, p_array.Length);
        }
    }
}
