using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace Godot.NativeInterop;

// TODO: Change VariantConversionCallbacks<T>. Store the callback in a static field for quick repeated access, instead of checking every time.
internal static unsafe class VariantConversionCallbacks
{
    internal static System.Collections.Generic.Dictionary<Type, (IntPtr ToVariant, IntPtr FromVariant)>
        GenericConversionCallbacks = new();

    [SuppressMessage("ReSharper", "RedundantNameQualifier")]
    internal static delegate*<in T, godot_variant> GetToVariantCallback<T>()
    {
        static godot_variant FromBool(in bool @bool) =>
            VariantUtils.CreateFromBool(@bool);

        static godot_variant FromChar(in char @char) =>
            VariantUtils.CreateFromInt(@char);

        static godot_variant FromInt8(in sbyte @int8) =>
            VariantUtils.CreateFromInt(@int8);

        static godot_variant FromInt16(in short @int16) =>
            VariantUtils.CreateFromInt(@int16);

        static godot_variant FromInt32(in int @int32) =>
            VariantUtils.CreateFromInt(@int32);

        static godot_variant FromInt64(in long @int64) =>
            VariantUtils.CreateFromInt(@int64);

        static godot_variant FromUInt8(in byte @uint8) =>
            VariantUtils.CreateFromInt(@uint8);

        static godot_variant FromUInt16(in ushort @uint16) =>
            VariantUtils.CreateFromInt(@uint16);

        static godot_variant FromUInt32(in uint @uint32) =>
            VariantUtils.CreateFromInt(@uint32);

        static godot_variant FromUInt64(in ulong @uint64) =>
            VariantUtils.CreateFromInt(@uint64);

        static godot_variant FromFloat(in float @float) =>
            VariantUtils.CreateFromFloat(@float);

        static godot_variant FromDouble(in double @double) =>
            VariantUtils.CreateFromFloat(@double);

        static godot_variant FromVector2(in Vector2 @vector2) =>
            VariantUtils.CreateFromVector2(@vector2);

        static godot_variant FromVector2I(in Vector2i vector2I) =>
            VariantUtils.CreateFromVector2i(vector2I);

        static godot_variant FromRect2(in Rect2 @rect2) =>
            VariantUtils.CreateFromRect2(@rect2);

        static godot_variant FromRect2I(in Rect2i rect2I) =>
            VariantUtils.CreateFromRect2i(rect2I);

        static godot_variant FromTransform2D(in Transform2D @transform2D) =>
            VariantUtils.CreateFromTransform2D(@transform2D);

        static godot_variant FromVector3(in Vector3 @vector3) =>
            VariantUtils.CreateFromVector3(@vector3);

        static godot_variant FromVector3I(in Vector3i vector3I) =>
            VariantUtils.CreateFromVector3i(vector3I);

        static godot_variant FromBasis(in Basis @basis) =>
            VariantUtils.CreateFromBasis(@basis);

        static godot_variant FromQuaternion(in Quaternion @quaternion) =>
            VariantUtils.CreateFromQuaternion(@quaternion);

        static godot_variant FromTransform3D(in Transform3D @transform3d) =>
            VariantUtils.CreateFromTransform3D(@transform3d);

        static godot_variant FromVector4(in Vector4 @vector4) =>
            VariantUtils.CreateFromVector4(@vector4);

        static godot_variant FromVector4I(in Vector4i vector4I) =>
            VariantUtils.CreateFromVector4i(vector4I);

        static godot_variant FromAabb(in AABB @aabb) =>
            VariantUtils.CreateFromAABB(@aabb);

        static godot_variant FromColor(in Color @color) =>
            VariantUtils.CreateFromColor(@color);

        static godot_variant FromPlane(in Plane @plane) =>
            VariantUtils.CreateFromPlane(@plane);

        static godot_variant FromCallable(in Callable @callable) =>
            VariantUtils.CreateFromCallable(@callable);

        static godot_variant FromSignalInfo(in SignalInfo @signalInfo) =>
            VariantUtils.CreateFromSignalInfo(@signalInfo);

        static godot_variant FromString(in string @string) =>
            VariantUtils.CreateFromString(@string);

        static godot_variant FromByteArray(in byte[] byteArray) =>
            VariantUtils.CreateFromPackedByteArray(byteArray);

        static godot_variant FromInt32Array(in int[] int32Array) =>
            VariantUtils.CreateFromPackedInt32Array(int32Array);

        static godot_variant FromInt64Array(in long[] int64Array) =>
            VariantUtils.CreateFromPackedInt64Array(int64Array);

        static godot_variant FromFloatArray(in float[] floatArray) =>
            VariantUtils.CreateFromPackedFloat32Array(floatArray);

        static godot_variant FromDoubleArray(in double[] doubleArray) =>
            VariantUtils.CreateFromPackedFloat64Array(doubleArray);

        static godot_variant FromStringArray(in string[] stringArray) =>
            VariantUtils.CreateFromPackedStringArray(stringArray);

        static godot_variant FromVector2Array(in Vector2[] vector2Array) =>
            VariantUtils.CreateFromPackedVector2Array(vector2Array);

        static godot_variant FromVector3Array(in Vector3[] vector3Array) =>
            VariantUtils.CreateFromPackedVector3Array(vector3Array);

        static godot_variant FromColorArray(in Color[] colorArray) =>
            VariantUtils.CreateFromPackedColorArray(colorArray);

        static godot_variant FromStringNameArray(in StringName[] stringNameArray) =>
            VariantUtils.CreateFromSystemArrayOfStringName(stringNameArray);

        static godot_variant FromNodePathArray(in NodePath[] nodePathArray) =>
            VariantUtils.CreateFromSystemArrayOfNodePath(nodePathArray);

        static godot_variant FromRidArray(in RID[] ridArray) =>
            VariantUtils.CreateFromSystemArrayOfRID(ridArray);

        static godot_variant FromGodotObject(in Godot.Object godotObject) =>
            VariantUtils.CreateFromGodotObject(godotObject);

        static godot_variant FromStringName(in StringName stringName) =>
            VariantUtils.CreateFromStringName(stringName);

        static godot_variant FromNodePath(in NodePath nodePath) =>
            VariantUtils.CreateFromNodePath(nodePath);

        static godot_variant FromRid(in RID rid) =>
            VariantUtils.CreateFromRID(rid);

        static godot_variant FromGodotDictionary(in Collections.Dictionary godotDictionary) =>
            VariantUtils.CreateFromDictionary(godotDictionary);

        static godot_variant FromGodotArray(in Collections.Array godotArray) =>
            VariantUtils.CreateFromArray(godotArray);

        static godot_variant FromVariant(in Variant variant) =>
            NativeFuncs.godotsharp_variant_new_copy((godot_variant)variant.NativeVar);

        var typeOfT = typeof(T);

        if (typeOfT == typeof(bool))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in bool, godot_variant>)
                &FromBool;
        }

        if (typeOfT == typeof(char))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in char, godot_variant>)
                &FromChar;
        }

        if (typeOfT == typeof(sbyte))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in sbyte, godot_variant>)
                &FromInt8;
        }

        if (typeOfT == typeof(short))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in short, godot_variant>)
                &FromInt16;
        }

        if (typeOfT == typeof(int))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in int, godot_variant>)
                &FromInt32;
        }

        if (typeOfT == typeof(long))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in long, godot_variant>)
                &FromInt64;
        }

        if (typeOfT == typeof(byte))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in byte, godot_variant>)
                &FromUInt8;
        }

        if (typeOfT == typeof(ushort))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in ushort, godot_variant>)
                &FromUInt16;
        }

        if (typeOfT == typeof(uint))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in uint, godot_variant>)
                &FromUInt32;
        }

        if (typeOfT == typeof(ulong))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in ulong, godot_variant>)
                &FromUInt64;
        }

        if (typeOfT == typeof(float))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in float, godot_variant>)
                &FromFloat;
        }

        if (typeOfT == typeof(double))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in double, godot_variant>)
                &FromDouble;
        }

        if (typeOfT == typeof(Vector2))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector2, godot_variant>)
                &FromVector2;
        }

        if (typeOfT == typeof(Vector2i))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector2i, godot_variant>)
                &FromVector2I;
        }

        if (typeOfT == typeof(Rect2))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Rect2, godot_variant>)
                &FromRect2;
        }

        if (typeOfT == typeof(Rect2i))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Rect2i, godot_variant>)
                &FromRect2I;
        }

        if (typeOfT == typeof(Transform2D))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Transform2D, godot_variant>)
                &FromTransform2D;
        }

        if (typeOfT == typeof(Vector3))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector3, godot_variant>)
                &FromVector3;
        }

        if (typeOfT == typeof(Vector3i))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector3i, godot_variant>)
                &FromVector3I;
        }

        if (typeOfT == typeof(Basis))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Basis, godot_variant>)
                &FromBasis;
        }

        if (typeOfT == typeof(Quaternion))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Quaternion, godot_variant>)
                &FromQuaternion;
        }

        if (typeOfT == typeof(Transform3D))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Transform3D, godot_variant>)
                &FromTransform3D;
        }

        if (typeOfT == typeof(Vector4))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector4, godot_variant>)
                &FromVector4;
        }

        if (typeOfT == typeof(Vector4i))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector4i, godot_variant>)
                &FromVector4I;
        }

        if (typeOfT == typeof(AABB))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in AABB, godot_variant>)
                &FromAabb;
        }

        if (typeOfT == typeof(Color))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Color, godot_variant>)
                &FromColor;
        }

        if (typeOfT == typeof(Plane))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Plane, godot_variant>)
                &FromPlane;
        }

        if (typeOfT == typeof(Callable))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Callable, godot_variant>)
                &FromCallable;
        }

        if (typeOfT == typeof(SignalInfo))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in SignalInfo, godot_variant>)
                &FromSignalInfo;
        }

        if (typeOfT.IsEnum)
        {
            var enumUnderlyingType = typeOfT.GetEnumUnderlyingType();

            switch (Type.GetTypeCode(enumUnderlyingType))
            {
                case TypeCode.SByte:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in sbyte, godot_variant>)
                        &FromInt8;
                }
                case TypeCode.Int16:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in short, godot_variant>)
                        &FromInt16;
                }
                case TypeCode.Int32:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in int, godot_variant>)
                        &FromInt32;
                }
                case TypeCode.Int64:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in long, godot_variant>)
                        &FromInt64;
                }
                case TypeCode.Byte:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in byte, godot_variant>)
                        &FromUInt8;
                }
                case TypeCode.UInt16:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in ushort, godot_variant>)
                        &FromUInt16;
                }
                case TypeCode.UInt32:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in uint, godot_variant>)
                        &FromUInt32;
                }
                case TypeCode.UInt64:
                {
                    return (delegate*<in T, godot_variant>)(delegate*<in ulong, godot_variant>)
                        &FromUInt64;
                }
                default:
                    return null;
            }
        }

        if (typeOfT == typeof(string))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in string, godot_variant>)
                &FromString;
        }

        if (typeOfT == typeof(byte[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in byte[], godot_variant>)
                &FromByteArray;
        }

        if (typeOfT == typeof(int[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in int[], godot_variant>)
                &FromInt32Array;
        }

        if (typeOfT == typeof(long[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in long[], godot_variant>)
                &FromInt64Array;
        }

        if (typeOfT == typeof(float[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in float[], godot_variant>)
                &FromFloatArray;
        }

        if (typeOfT == typeof(double[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in double[], godot_variant>)
                &FromDoubleArray;
        }

        if (typeOfT == typeof(string[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in string[], godot_variant>)
                &FromStringArray;
        }

        if (typeOfT == typeof(Vector2[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector2[], godot_variant>)
                &FromVector2Array;
        }

        if (typeOfT == typeof(Vector3[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Vector3[], godot_variant>)
                &FromVector3Array;
        }

        if (typeOfT == typeof(Color[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Color[], godot_variant>)
                &FromColorArray;
        }

        if (typeOfT == typeof(StringName[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in StringName[], godot_variant>)
                &FromStringNameArray;
        }

        if (typeOfT == typeof(NodePath[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in NodePath[], godot_variant>)
                &FromNodePathArray;
        }

        if (typeOfT == typeof(RID[]))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in RID[], godot_variant>)
                &FromRidArray;
        }

        if (typeof(Godot.Object).IsAssignableFrom(typeOfT))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Godot.Object, godot_variant>)
                &FromGodotObject;
        }

        if (typeOfT == typeof(StringName))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in StringName, godot_variant>)
                &FromStringName;
        }

        if (typeOfT == typeof(NodePath))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in NodePath, godot_variant>)
                &FromNodePath;
        }

        if (typeOfT == typeof(RID))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in RID, godot_variant>)
                &FromRid;
        }

        if (typeOfT == typeof(Godot.Collections.Dictionary))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Godot.Collections.Dictionary, godot_variant>)
                &FromGodotDictionary;
        }

        if (typeOfT == typeof(Godot.Collections.Array))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Godot.Collections.Array, godot_variant>)
                &FromGodotArray;
        }

        if (typeOfT == typeof(Variant))
        {
            return (delegate*<in T, godot_variant>)(delegate*<in Variant, godot_variant>)
                &FromVariant;
        }

        // TODO:
        //   IsGenericType and GetGenericTypeDefinition don't work in NativeAOT's reflection-free mode.
        //   We could make the Godot collections implement an interface and use IsAssignableFrom instead.
        //   Or we could just skip the check and always look for a conversion callback for the type.
        if (typeOfT.IsGenericType)
        {
            var genericTypeDef = typeOfT.GetGenericTypeDefinition();

            if (genericTypeDef == typeof(Godot.Collections.Dictionary<,>) ||
                genericTypeDef == typeof(Godot.Collections.Array<>))
            {
                RuntimeHelpers.RunClassConstructor(typeOfT.TypeHandle);

                if (GenericConversionCallbacks.TryGetValue(typeOfT, out var genericConversion))
                {
                    return (delegate*<in T, godot_variant>)genericConversion.ToVariant;
                }
            }
        }

        return null;
    }

    [SuppressMessage("ReSharper", "RedundantNameQualifier")]
    internal static delegate*<in godot_variant, T> GetToManagedCallback<T>()
    {
        static bool ToBool(in godot_variant variant) =>
            VariantUtils.ConvertToBool(variant);

        static char ToChar(in godot_variant variant) =>
            VariantUtils.ConvertToChar(variant);

        static sbyte ToInt8(in godot_variant variant) =>
            VariantUtils.ConvertToInt8(variant);

        static short ToInt16(in godot_variant variant) =>
            VariantUtils.ConvertToInt16(variant);

        static int ToInt32(in godot_variant variant) =>
            VariantUtils.ConvertToInt32(variant);

        static long ToInt64(in godot_variant variant) =>
            VariantUtils.ConvertToInt64(variant);

        static byte ToUInt8(in godot_variant variant) =>
            VariantUtils.ConvertToUInt8(variant);

        static ushort ToUInt16(in godot_variant variant) =>
            VariantUtils.ConvertToUInt16(variant);

        static uint ToUInt32(in godot_variant variant) =>
            VariantUtils.ConvertToUInt32(variant);

        static ulong ToUInt64(in godot_variant variant) =>
            VariantUtils.ConvertToUInt64(variant);

        static float ToFloat(in godot_variant variant) =>
            VariantUtils.ConvertToFloat32(variant);

        static double ToDouble(in godot_variant variant) =>
            VariantUtils.ConvertToFloat64(variant);

        static Vector2 ToVector2(in godot_variant variant) =>
            VariantUtils.ConvertToVector2(variant);

        static Vector2i ToVector2I(in godot_variant variant) =>
            VariantUtils.ConvertToVector2i(variant);

        static Rect2 ToRect2(in godot_variant variant) =>
            VariantUtils.ConvertToRect2(variant);

        static Rect2i ToRect2I(in godot_variant variant) =>
            VariantUtils.ConvertToRect2i(variant);

        static Transform2D ToTransform2D(in godot_variant variant) =>
            VariantUtils.ConvertToTransform2D(variant);

        static Vector3 ToVector3(in godot_variant variant) =>
            VariantUtils.ConvertToVector3(variant);

        static Vector3i ToVector3I(in godot_variant variant) =>
            VariantUtils.ConvertToVector3i(variant);

        static Basis ToBasis(in godot_variant variant) =>
            VariantUtils.ConvertToBasis(variant);

        static Quaternion ToQuaternion(in godot_variant variant) =>
            VariantUtils.ConvertToQuaternion(variant);

        static Transform3D ToTransform3D(in godot_variant variant) =>
            VariantUtils.ConvertToTransform3D(variant);

        static Vector4 ToVector4(in godot_variant variant) =>
            VariantUtils.ConvertToVector4(variant);

        static Vector4i ToVector4I(in godot_variant variant) =>
            VariantUtils.ConvertToVector4i(variant);

        static AABB ToAabb(in godot_variant variant) =>
            VariantUtils.ConvertToAABB(variant);

        static Color ToColor(in godot_variant variant) =>
            VariantUtils.ConvertToColor(variant);

        static Plane ToPlane(in godot_variant variant) =>
            VariantUtils.ConvertToPlane(variant);

        static Callable ToCallable(in godot_variant variant) =>
            VariantUtils.ConvertToCallableManaged(variant);

        static SignalInfo ToSignalInfo(in godot_variant variant) =>
            VariantUtils.ConvertToSignalInfo(variant);

        static string ToString(in godot_variant variant) =>
            VariantUtils.ConvertToStringObject(variant);

        static byte[] ToByteArray(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedByteArrayToSystemArray(variant);

        static int[] ToInt32Array(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedInt32ArrayToSystemArray(variant);

        static long[] ToInt64Array(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedInt64ArrayToSystemArray(variant);

        static float[] ToFloatArray(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedFloat32ArrayToSystemArray(variant);

        static double[] ToDoubleArray(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedFloat64ArrayToSystemArray(variant);

        static string[] ToStringArray(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedStringArrayToSystemArray(variant);

        static Vector2[] ToVector2Array(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedVector2ArrayToSystemArray(variant);

        static Vector3[] ToVector3Array(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedVector3ArrayToSystemArray(variant);

        static Color[] ToColorArray(in godot_variant variant) =>
            VariantUtils.ConvertAsPackedColorArrayToSystemArray(variant);

        static StringName[] ToStringNameArray(in godot_variant variant) =>
            VariantUtils.ConvertToSystemArrayOfStringName(variant);

        static NodePath[] ToNodePathArray(in godot_variant variant) =>
            VariantUtils.ConvertToSystemArrayOfNodePath(variant);

        static RID[] ToRidArray(in godot_variant variant) =>
            VariantUtils.ConvertToSystemArrayOfRID(variant);

        static Godot.Object ToGodotObject(in godot_variant variant) =>
            VariantUtils.ConvertToGodotObject(variant);

        static StringName ToStringName(in godot_variant variant) =>
            VariantUtils.ConvertToStringNameObject(variant);

        static NodePath ToNodePath(in godot_variant variant) =>
            VariantUtils.ConvertToNodePathObject(variant);

        static RID ToRid(in godot_variant variant) =>
            VariantUtils.ConvertToRID(variant);

        static Collections.Dictionary ToGodotDictionary(in godot_variant variant) =>
            VariantUtils.ConvertToDictionaryObject(variant);

        static Collections.Array ToGodotArray(in godot_variant variant) =>
            VariantUtils.ConvertToArrayObject(variant);

        static Variant ToVariant(in godot_variant variant) =>
            Variant.CreateCopyingBorrowed(variant);

        var typeOfT = typeof(T);

        // ReSharper disable RedundantCast
        // Rider is being stupid here. These casts are definitely needed. We get build errors without them.

        if (typeOfT == typeof(bool))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, bool>)
                &ToBool;
        }

        if (typeOfT == typeof(char))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, char>)
                &ToChar;
        }

        if (typeOfT == typeof(sbyte))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, sbyte>)
                &ToInt8;
        }

        if (typeOfT == typeof(short))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, short>)
                &ToInt16;
        }

        if (typeOfT == typeof(int))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, int>)
                &ToInt32;
        }

        if (typeOfT == typeof(long))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, long>)
                &ToInt64;
        }

        if (typeOfT == typeof(byte))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, byte>)
                &ToUInt8;
        }

        if (typeOfT == typeof(ushort))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, ushort>)
                &ToUInt16;
        }

        if (typeOfT == typeof(uint))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, uint>)
                &ToUInt32;
        }

        if (typeOfT == typeof(ulong))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, ulong>)
                &ToUInt64;
        }

        if (typeOfT == typeof(float))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, float>)
                &ToFloat;
        }

        if (typeOfT == typeof(double))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, double>)
                &ToDouble;
        }

        if (typeOfT == typeof(Vector2))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector2>)
                &ToVector2;
        }

        if (typeOfT == typeof(Vector2i))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector2i>)
                &ToVector2I;
        }

        if (typeOfT == typeof(Rect2))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Rect2>)
                &ToRect2;
        }

        if (typeOfT == typeof(Rect2i))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Rect2i>)
                &ToRect2I;
        }

        if (typeOfT == typeof(Transform2D))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Transform2D>)
                &ToTransform2D;
        }

        if (typeOfT == typeof(Vector3))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector3>)
                &ToVector3;
        }

        if (typeOfT == typeof(Vector3i))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector3i>)
                &ToVector3I;
        }

        if (typeOfT == typeof(Basis))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Basis>)
                &ToBasis;
        }

        if (typeOfT == typeof(Quaternion))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Quaternion>)
                &ToQuaternion;
        }

        if (typeOfT == typeof(Transform3D))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Transform3D>)
                &ToTransform3D;
        }

        if (typeOfT == typeof(Vector4))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector4>)
                &ToVector4;
        }

        if (typeOfT == typeof(Vector4i))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector4i>)
                &ToVector4I;
        }

        if (typeOfT == typeof(AABB))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, AABB>)
                &ToAabb;
        }

        if (typeOfT == typeof(Color))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Color>)
                &ToColor;
        }

        if (typeOfT == typeof(Plane))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Plane>)
                &ToPlane;
        }

        if (typeOfT == typeof(Callable))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Callable>)
                &ToCallable;
        }

        if (typeOfT == typeof(SignalInfo))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, SignalInfo>)
                &ToSignalInfo;
        }

        if (typeOfT.IsEnum)
        {
            var enumUnderlyingType = typeOfT.GetEnumUnderlyingType();

            switch (Type.GetTypeCode(enumUnderlyingType))
            {
                case TypeCode.SByte:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, sbyte>)
                        &ToInt8;
                }
                case TypeCode.Int16:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, short>)
                        &ToInt16;
                }
                case TypeCode.Int32:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, int>)
                        &ToInt32;
                }
                case TypeCode.Int64:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, long>)
                        &ToInt64;
                }
                case TypeCode.Byte:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, byte>)
                        &ToUInt8;
                }
                case TypeCode.UInt16:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, ushort>)
                        &ToUInt16;
                }
                case TypeCode.UInt32:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, uint>)
                        &ToUInt32;
                }
                case TypeCode.UInt64:
                {
                    return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, ulong>)
                        &ToUInt64;
                }
                default:
                    return null;
            }
        }

        if (typeOfT == typeof(string))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, string>)
                &ToString;
        }

        if (typeOfT == typeof(byte[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, byte[]>)
                &ToByteArray;
        }

        if (typeOfT == typeof(int[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, int[]>)
                &ToInt32Array;
        }

        if (typeOfT == typeof(long[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, long[]>)
                &ToInt64Array;
        }

        if (typeOfT == typeof(float[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, float[]>)
                &ToFloatArray;
        }

        if (typeOfT == typeof(double[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, double[]>)
                &ToDoubleArray;
        }

        if (typeOfT == typeof(string[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, string[]>)
                &ToStringArray;
        }

        if (typeOfT == typeof(Vector2[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector2[]>)
                &ToVector2Array;
        }

        if (typeOfT == typeof(Vector3[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Vector3[]>)
                &ToVector3Array;
        }

        if (typeOfT == typeof(Color[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Color[]>)
                &ToColorArray;
        }

        if (typeOfT == typeof(StringName[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, StringName[]>)
                &ToStringNameArray;
        }

        if (typeOfT == typeof(NodePath[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, NodePath[]>)
                &ToNodePathArray;
        }

        if (typeOfT == typeof(RID[]))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, RID[]>)
                &ToRidArray;
        }

        if (typeof(Godot.Object).IsAssignableFrom(typeOfT))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Godot.Object>)
                &ToGodotObject;
        }

        if (typeOfT == typeof(StringName))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, StringName>)
                &ToStringName;
        }

        if (typeOfT == typeof(NodePath))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, NodePath>)
                &ToNodePath;
        }

        if (typeOfT == typeof(RID))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, RID>)
                &ToRid;
        }

        if (typeOfT == typeof(Godot.Collections.Dictionary))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Godot.Collections.Dictionary>)
                &ToGodotDictionary;
        }

        if (typeOfT == typeof(Godot.Collections.Array))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Godot.Collections.Array>)
                &ToGodotArray;
        }

        if (typeOfT == typeof(Variant))
        {
            return (delegate*<in godot_variant, T>)(delegate*<in godot_variant, Variant>)
                &ToVariant;
        }

        // TODO:
        //   IsGenericType and GetGenericTypeDefinition don't work in NativeAOT's reflection-free mode.
        //   We could make the Godot collections implement an interface and use IsAssignableFrom instead.
        //   Or we could just skip the check and always look for a conversion callback for the type.
        if (typeOfT.IsGenericType)
        {
            var genericTypeDef = typeOfT.GetGenericTypeDefinition();

            if (genericTypeDef == typeof(Godot.Collections.Dictionary<,>) ||
                genericTypeDef == typeof(Godot.Collections.Array<>))
            {
                RuntimeHelpers.RunClassConstructor(typeOfT.TypeHandle);

                if (GenericConversionCallbacks.TryGetValue(typeOfT, out var genericConversion))
                {
                    return (delegate*<in godot_variant, T>)genericConversion.FromVariant;
                }
            }
        }

        // ReSharper restore RedundantCast

        return null;
    }
}
