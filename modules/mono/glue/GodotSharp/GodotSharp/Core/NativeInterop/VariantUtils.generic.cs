using System;
using System.Runtime.CompilerServices;

namespace Godot.NativeInterop;

#nullable enable

public partial class VariantUtils
{
    private static Exception UnsupportedType<T>() => new InvalidOperationException(
        $"The type is not supported for conversion to/from Variant: '{typeof(T).FullName}'");
    private delegate T ConvertToDelegate<[MustBeVariant] T>(in godot_variant variant);
    private delegate godot_variant CreateFromDelegate<[MustBeVariant] T>(in T from);

    internal static class GenericConversion<T>
    {
        public static unsafe godot_variant ToVariant(in T from) =>
            ToVariantCb != null ? ToVariantCb(from) : throw UnsupportedType<T>();

        public static unsafe T FromVariant(in godot_variant variant) =>
            FromVariantCb != null ? FromVariantCb(variant) : throw UnsupportedType<T>();

        // ReSharper disable once StaticMemberInGenericType
        internal static unsafe delegate*<in T, godot_variant> ToVariantCb;

        // ReSharper disable once StaticMemberInGenericType
        internal static unsafe delegate*<in godot_variant, T> FromVariantCb;

        static GenericConversion()
        {
            RuntimeHelpers.RunClassConstructor(typeof(T).TypeHandle);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static godot_variant CreateFrom<[MustBeVariant] T>(in T from) => CreateFromLookup<T>.CreateFrom(from);

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static T ConvertTo<[MustBeVariant] T>(in godot_variant variant) => ConvertToLookup<T>.ConvertTo(variant);

    // We use a generic static class to lookup the conversion method once during its initialization.
    // The .NET runtime does the heavy lifting of creating an instance per type which makes this implementation
    // behave similar to C++ templates: a specific implementation for each type without any overhead at runtime.
    private static class ConvertToLookup<[MustBeVariant] T>
    {
        private static readonly ConvertToDelegate<T> _converter = DetermineConvertToDelegate<T>();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static T ConvertTo(in godot_variant variant) => _converter(variant);
    }

    // We use a generic static class to lookup the conversion method once during its initialization.
    // The .NET runtime does the heavy lifting of creating an instance per type which makes this implementation
    // behave similar to C++ templates: a specific implementation for each type without any overhead at runtime.
    private static class CreateFromLookup<[MustBeVariant] T>
    {
        private static readonly CreateFromDelegate<T> _converter = DetermineCreateFromDelegate<T>();

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static godot_variant CreateFrom(in T from) => _converter(from);
    }

    private static CreateFromDelegate<T> DetermineCreateFromDelegate<T>()
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static TTo UnsafeAs<TTo>(in T f) => Unsafe.As<T, TTo>(ref Unsafe.AsRef(f));

        if (typeof(T) == typeof(bool))
            return (in T from) => CreateFromBool(UnsafeAs<bool>(from));

        if (typeof(T) == typeof(char))
            return (in T from) => CreateFromInt(UnsafeAs<char>(from));

        if (typeof(T) == typeof(sbyte))
            return (in T from) => CreateFromInt(UnsafeAs<sbyte>(from));

        if (typeof(T) == typeof(short))
            return (in T from) => CreateFromInt(UnsafeAs<short>(from));

        if (typeof(T) == typeof(int))
            return (in T from) => CreateFromInt(UnsafeAs<int>(from));

        if (typeof(T) == typeof(long))
            return (in T from) => CreateFromInt(UnsafeAs<long>(from));

        if (typeof(T) == typeof(byte))
            return (in T from) => CreateFromInt(UnsafeAs<byte>(from));

        if (typeof(T) == typeof(ushort))
            return (in T from) => CreateFromInt(UnsafeAs<ushort>(from));

        if (typeof(T) == typeof(uint))
            return (in T from) => CreateFromInt(UnsafeAs<uint>(from));

        if (typeof(T) == typeof(ulong))
            return (in T from) => CreateFromInt(UnsafeAs<ulong>(from));

        if (typeof(T) == typeof(float))
            return (in T from) => CreateFromFloat(UnsafeAs<float>(from));

        if (typeof(T) == typeof(double))
            return (in T from) => CreateFromFloat(UnsafeAs<double>(from));

        if (typeof(T) == typeof(Vector2))
            return (in T from) => CreateFromVector2(UnsafeAs<Vector2>(from));

        if (typeof(T) == typeof(Vector2I))
            return (in T from) => CreateFromVector2I(UnsafeAs<Vector2I>(from));

        if (typeof(T) == typeof(Rect2))
            return (in T from) => CreateFromRect2(UnsafeAs<Rect2>(from));

        if (typeof(T) == typeof(Rect2I))
            return (in T from) => CreateFromRect2I(UnsafeAs<Rect2I>(from));

        if (typeof(T) == typeof(Transform2D))
            return (in T from) => CreateFromTransform2D(UnsafeAs<Transform2D>(from));

        if (typeof(T) == typeof(Projection))
            return (in T from) => CreateFromProjection(UnsafeAs<Projection>(from));

        if (typeof(T) == typeof(Vector3))
            return (in T from) => CreateFromVector3(UnsafeAs<Vector3>(from));

        if (typeof(T) == typeof(Vector3I))
            return (in T from) => CreateFromVector3I(UnsafeAs<Vector3I>(from));

        if (typeof(T) == typeof(Basis))
            return (in T from) => CreateFromBasis(UnsafeAs<Basis>(from));

        if (typeof(T) == typeof(Quaternion))
            return (in T from) => CreateFromQuaternion(UnsafeAs<Quaternion>(from));

        if (typeof(T) == typeof(Transform3D))
            return (in T from) => CreateFromTransform3D(UnsafeAs<Transform3D>(from));

        if (typeof(T) == typeof(Vector4))
            return (in T from) => CreateFromVector4(UnsafeAs<Vector4>(from));

        if (typeof(T) == typeof(Vector4I))
            return (in T from) => CreateFromVector4I(UnsafeAs<Vector4I>(from));

        if (typeof(T) == typeof(Aabb))
            return (in T from) => CreateFromAabb(UnsafeAs<Aabb>(from));

        if (typeof(T) == typeof(Color))
            return (in T from) => CreateFromColor(UnsafeAs<Color>(from));

        if (typeof(T) == typeof(Plane))
            return (in T from) => CreateFromPlane(UnsafeAs<Plane>(from));

        if (typeof(T) == typeof(Callable))
            return (in T from) => CreateFromCallable(UnsafeAs<Callable>(from));

        if (typeof(T) == typeof(Signal))
            return (in T from) => CreateFromSignal(UnsafeAs<Signal>(from));

        if (typeof(T) == typeof(string))
            return (in T from) => CreateFromString(UnsafeAs<string>(from));

        if (typeof(T) == typeof(byte[]))
            return (in T from) => CreateFromPackedByteArray(UnsafeAs<byte[]>(from));

        if (typeof(T) == typeof(int[]))
            return (in T from) => CreateFromPackedInt32Array(UnsafeAs<int[]>(from));

        if (typeof(T) == typeof(long[]))
            return (in T from) => CreateFromPackedInt64Array(UnsafeAs<long[]>(from));

        if (typeof(T) == typeof(float[]))
            return (in T from) => CreateFromPackedFloat32Array(UnsafeAs<float[]>(from));

        if (typeof(T) == typeof(double[]))
            return (in T from) => CreateFromPackedFloat64Array(UnsafeAs<double[]>(from));

        if (typeof(T) == typeof(string[]))
            return (in T from) => CreateFromPackedStringArray(UnsafeAs<string[]>(from));

        if (typeof(T) == typeof(Vector2[]))
            return (in T from) => CreateFromPackedVector2Array(UnsafeAs<Vector2[]>(from));

        if (typeof(T) == typeof(Vector3[]))
            return (in T from) => CreateFromPackedVector3Array(UnsafeAs<Vector3[]>(from));

        if (typeof(T) == typeof(Vector4[]))
            return CreateFromPackedVector4Array(UnsafeAs<Vector4[]>(from));

        if (typeof(T) == typeof(Color[]))
            return (in T from) => CreateFromPackedColorArray(UnsafeAs<Color[]>(from));

        if (typeof(T) == typeof(StringName[]))
            return (in T from) => CreateFromSystemArrayOfStringName(UnsafeAs<StringName[]>(from));

        if (typeof(T) == typeof(NodePath[]))
            return (in T from) => CreateFromSystemArrayOfNodePath(UnsafeAs<NodePath[]>(from));

        if (typeof(T) == typeof(Rid[]))
            return (in T from) => CreateFromSystemArrayOfRid(UnsafeAs<Rid[]>(from));

        if (typeof(T) == typeof(StringName))
            return (in T from) => CreateFromStringName(UnsafeAs<StringName>(from));

        if (typeof(T) == typeof(NodePath))
            return (in T from) => CreateFromNodePath(UnsafeAs<NodePath>(from));

        if (typeof(T) == typeof(Rid))
            return (in T from) => CreateFromRid(UnsafeAs<Rid>(from));

        if (typeof(T) == typeof(Godot.Collections.Dictionary))
            return (in T from) => CreateFromDictionary(UnsafeAs<Godot.Collections.Dictionary>(from));

        if (typeof(T) == typeof(Godot.Collections.Array))
            return (in T from) => CreateFromArray(UnsafeAs<Godot.Collections.Array>(from));

        if (typeof(T) == typeof(Variant))
            return (in T from) => NativeFuncs.godotsharp_variant_new_copy((godot_variant)UnsafeAs<Variant>(from).NativeVar);

        // More complex checks here at the end, to avoid screwing the simple ones in case they're not optimized away.

        // `typeof(X).IsAssignableFrom(typeof(T))` is optimized away

        if (typeof(GodotObject).IsAssignableFrom(typeof(T)))
            return (in T from) => CreateFromGodotObject(UnsafeAs<GodotObject>(from));

        // `typeof(T).IsValueType` is optimized away
        // `typeof(T).IsEnum` is NOT optimized away: https://github.com/dotnet/runtime/issues/67113
        // Fortunately, `typeof(System.Enum).IsAssignableFrom(typeof(T))` does the job!

        if (typeof(T).IsValueType && typeof(Enum).IsAssignableFrom(typeof(T)))
        {
            // `Type.GetTypeCode(typeof(T).GetEnumUnderlyingType())` is not optimized away.
            // Fortunately, `Unsafe.SizeOf<T>()` works and is optimized away.
            // We don't need to know whether it's signed or unsigned.

            if (Unsafe.SizeOf<T>() == 1)
                return (in T from) => CreateFromInt(UnsafeAs<sbyte>(from));

            if (Unsafe.SizeOf<T>() == 2)
                return (in T from) => CreateFromInt(UnsafeAs<short>(from));

            if (Unsafe.SizeOf<T>() == 4)
                return (in T from) => CreateFromInt(UnsafeAs<int>(from));

            if (Unsafe.SizeOf<T>() == 8)
                return (in T from) => CreateFromInt(UnsafeAs<long>(from));

            throw UnsupportedType<T>();
        }

        return (in T from) => GenericConversion<T>.ToVariant(from);

    }

    private static ConvertToDelegate<T> DetermineConvertToDelegate<T>()
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        static T UnsafeAsT<TFrom>(TFrom f) => Unsafe.As<TFrom, T>(ref Unsafe.AsRef(f));

        if (typeof(T) == typeof(bool))
            return (in godot_variant variant) => UnsafeAsT(ConvertToBool(variant));

        if (typeof(T) == typeof(char))
            return (in godot_variant variant) => UnsafeAsT(ConvertToChar(variant));

        if (typeof(T) == typeof(sbyte))
            return (in godot_variant variant) => UnsafeAsT(ConvertToInt8(variant));

        if (typeof(T) == typeof(short))
            return (in godot_variant variant) => UnsafeAsT(ConvertToInt16(variant));

        if (typeof(T) == typeof(int))
            return (in godot_variant variant) => UnsafeAsT(ConvertToInt32(variant));

        if (typeof(T) == typeof(long))
            return (in godot_variant variant) => UnsafeAsT(ConvertToInt64(variant));

        if (typeof(T) == typeof(byte))
            return (in godot_variant variant) => UnsafeAsT(ConvertToUInt8(variant));

        if (typeof(T) == typeof(ushort))
            return (in godot_variant variant) => UnsafeAsT(ConvertToUInt16(variant));

        if (typeof(T) == typeof(uint))
            return (in godot_variant variant) => UnsafeAsT(ConvertToUInt32(variant));

        if (typeof(T) == typeof(ulong))
            return (in godot_variant variant) => UnsafeAsT(ConvertToUInt64(variant));

        if (typeof(T) == typeof(float))
            return (in godot_variant variant) => UnsafeAsT(ConvertToFloat32(variant));

        if (typeof(T) == typeof(double))
            return (in godot_variant variant) => UnsafeAsT(ConvertToFloat64(variant));

        if (typeof(T) == typeof(Vector2))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector2(variant));

        if (typeof(T) == typeof(Vector2I))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector2I(variant));

        if (typeof(T) == typeof(Rect2))
            return (in godot_variant variant) => UnsafeAsT(ConvertToRect2(variant));

        if (typeof(T) == typeof(Rect2I))
            return (in godot_variant variant) => UnsafeAsT(ConvertToRect2I(variant));

        if (typeof(T) == typeof(Transform2D))
            return (in godot_variant variant) => UnsafeAsT(ConvertToTransform2D(variant));

        if (typeof(T) == typeof(Vector3))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector3(variant));

        if (typeof(T) == typeof(Vector3I))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector3I(variant));

        if (typeof(T) == typeof(Basis))
            return (in godot_variant variant) => UnsafeAsT(ConvertToBasis(variant));

        if (typeof(T) == typeof(Quaternion))
            return (in godot_variant variant) => UnsafeAsT(ConvertToQuaternion(variant));

        if (typeof(T) == typeof(Transform3D))
            return (in godot_variant variant) => UnsafeAsT(ConvertToTransform3D(variant));

        if (typeof(T) == typeof(Projection))
            return (in godot_variant variant) => UnsafeAsT(ConvertToProjection(variant));

        if (typeof(T) == typeof(Vector4))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector4(variant));

        if (typeof(T) == typeof(Vector4I))
            return (in godot_variant variant) => UnsafeAsT(ConvertToVector4I(variant));

        if (typeof(T) == typeof(Aabb))
            return (in godot_variant variant) => UnsafeAsT(ConvertToAabb(variant));

        if (typeof(T) == typeof(Color))
            return (in godot_variant variant) => UnsafeAsT(ConvertToColor(variant));

        if (typeof(T) == typeof(Plane))
            return (in godot_variant variant) => UnsafeAsT(ConvertToPlane(variant));

        if (typeof(T) == typeof(Callable))
            return (in godot_variant variant) => UnsafeAsT(ConvertToCallable(variant));

        if (typeof(T) == typeof(Signal))
            return (in godot_variant variant) => UnsafeAsT(ConvertToSignal(variant));

        if (typeof(T) == typeof(string))
            return (in godot_variant variant) => UnsafeAsT(ConvertToString(variant));

        if (typeof(T) == typeof(byte[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedByteArrayToSystemArray(variant));

        if (typeof(T) == typeof(int[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedInt32ArrayToSystemArray(variant));

        if (typeof(T) == typeof(long[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedInt64ArrayToSystemArray(variant));

        if (typeof(T) == typeof(float[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedFloat32ArrayToSystemArray(variant));

        if (typeof(T) == typeof(double[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedFloat64ArrayToSystemArray(variant));

        if (typeof(T) == typeof(string[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedStringArrayToSystemArray(variant));

        if (typeof(T) == typeof(Vector2[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedVector2ArrayToSystemArray(variant));

        if (typeof(T) == typeof(Vector3[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedVector3ArrayToSystemArray(variant));

        if (typeof(T) == typeof(Vector4[]))
            return UnsafeAsT(ConvertAsPackedVector4ArrayToSystemArray(variant));

        if (typeof(T) == typeof(Color[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertAsPackedColorArrayToSystemArray(variant));

        if (typeof(T) == typeof(StringName[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertToSystemArrayOfStringName(variant));

        if (typeof(T) == typeof(NodePath[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertToSystemArrayOfNodePath(variant));

        if (typeof(T) == typeof(Rid[]))
            return (in godot_variant variant) => UnsafeAsT(ConvertToSystemArrayOfRid(variant));

        if (typeof(T) == typeof(StringName))
            return (in godot_variant variant) => UnsafeAsT(ConvertToStringName(variant));

        if (typeof(T) == typeof(NodePath))
            return (in godot_variant variant) => UnsafeAsT(ConvertToNodePath(variant));

        if (typeof(T) == typeof(Rid))
            return (in godot_variant variant) => UnsafeAsT(ConvertToRid(variant));

        if (typeof(T) == typeof(Godot.Collections.Dictionary))
            return (in godot_variant variant) => UnsafeAsT(ConvertToDictionary(variant));

        if (typeof(T) == typeof(Godot.Collections.Array))
            return (in godot_variant variant) => UnsafeAsT(ConvertToArray(variant));

        if (typeof(T) == typeof(Variant))
            return (in godot_variant variant) => UnsafeAsT(Variant.CreateCopyingBorrowed(variant));
        // More complex checks here at the end, to avoid screwing the simple ones in case they're not optimized away.

        // `typeof(X).IsAssignableFrom(typeof(T))` is optimized away

        if (typeof(GodotObject).IsAssignableFrom(typeof(T)))
            return (in godot_variant variant) => (T)(object)ConvertToGodotObject(variant);

        // `typeof(T).IsValueType` is optimized away
        // `typeof(T).IsEnum` is NOT optimized away: https://github.com/dotnet/runtime/issues/67113
        // Fortunately, `typeof(System.Enum).IsAssignableFrom(typeof(T))` does the job!

        if (typeof(T).IsValueType && typeof(Enum).IsAssignableFrom(typeof(T)))
        {
            // `Type.GetTypeCode(typeof(T).GetEnumUnderlyingType())` is not optimized away.
            // Fortunately, `Unsafe.SizeOf<T>()` works and is optimized away.
            // We don't need to know whether it's signed or unsigned.

            if (Unsafe.SizeOf<T>() == 1)
                return (in godot_variant variant) => UnsafeAsT(ConvertToInt8(variant));

            if (Unsafe.SizeOf<T>() == 2)
                return (in godot_variant variant) => UnsafeAsT(ConvertToInt16(variant));

            if (Unsafe.SizeOf<T>() == 4)
                return (in godot_variant variant) => UnsafeAsT(ConvertToInt32(variant));

            if (Unsafe.SizeOf<T>() == 8)
                return (in godot_variant variant) => UnsafeAsT(ConvertToInt64(variant));

            throw UnsupportedType<T>();
        }

        return (in godot_variant variant) => GenericConversion<T>.FromVariant(variant);
    }
}
