using System;
using System.Runtime.CompilerServices;
using Godot.Collections;

// ReSharper disable InconsistentNaming

#nullable enable

namespace Godot.NativeInterop
{
    /// <summary>
    /// Collection of conversion callbacks used for marshaling by callables
    /// and generic Godot collections.
    /// </summary>
    public static partial class VariantUtils
    {
        /// <summary>
        /// Converts an <see cref="Rid"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Rid"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Rid"/>.</returns>
        public static godot_variant CreateFromRid(Rid from)
            => new() { Type = Variant.Type.Rid, Rid = from };

        /// <summary>
        /// Converts a <see langword="bool"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="bool"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="bool"/>.</returns>
        public static godot_variant CreateFromBool(bool from)
            => new() { Type = Variant.Type.Bool, Bool = from.ToGodotBool() };

        /// <summary>
        /// Converts a <see langword="long"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="long"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="long"/>.</returns>
        public static godot_variant CreateFromInt(long from)
            => new() { Type = Variant.Type.Int, Int = from };

        /// <summary>
        /// Converts a <see langword="ulong"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="ulong"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="ulong"/>.</returns>
        public static godot_variant CreateFromInt(ulong from)
            => new() { Type = Variant.Type.Int, Int = (long)from };

        /// <summary>
        /// Converts a <see langword="double"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="double"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="double"/>.</returns>
        public static godot_variant CreateFromFloat(double from)
            => new() { Type = Variant.Type.Float, Float = from };

        /// <summary>
        /// Converts a <see cref="Vector2"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector2"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector2"/>.</returns>
        public static godot_variant CreateFromVector2(Vector2 from)
            => new() { Type = Variant.Type.Vector2, Vector2 = from };

        /// <summary>
        /// Converts a <see cref="Vector2I"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector2I"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector2I"/>.</returns>
        public static godot_variant CreateFromVector2I(Vector2I from)
            => new() { Type = Variant.Type.Vector2I, Vector2I = from };

        /// <summary>
        /// Converts a <see cref="Vector3"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector3"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector3"/>.</returns>
        public static godot_variant CreateFromVector3(Vector3 from)
            => new() { Type = Variant.Type.Vector3, Vector3 = from };

        /// <summary>
        /// Converts a <see cref="Vector3I"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector3I"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector3I"/>.</returns>
        public static godot_variant CreateFromVector3I(Vector3I from)
            => new() { Type = Variant.Type.Vector3I, Vector3I = from };

        /// <summary>
        /// Converts a <see cref="Vector4"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector4"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector4"/>.</returns>
        public static godot_variant CreateFromVector4(Vector4 from)
            => new() { Type = Variant.Type.Vector4, Vector4 = from };

        /// <summary>
        /// Converts a <see cref="Vector4I"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector4I"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector4I"/>.</returns>
        public static godot_variant CreateFromVector4I(Vector4I from)
            => new() { Type = Variant.Type.Vector4I, Vector4I = from };

        /// <summary>
        /// Converts a <see cref="Rect2"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Rect2"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Rect2"/>.</returns>
        public static godot_variant CreateFromRect2(Rect2 from)
            => new() { Type = Variant.Type.Rect2, Rect2 = from };

        /// <summary>
        /// Converts a <see cref="Rect2I"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Rect2I"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Rect2I"/>.</returns>
        public static godot_variant CreateFromRect2I(Rect2I from)
            => new() { Type = Variant.Type.Rect2I, Rect2I = from };

        /// <summary>
        /// Converts a <see cref="Quaternion"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Quaternion"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Quaternion"/>.</returns>
        public static godot_variant CreateFromQuaternion(Quaternion from)
            => new() { Type = Variant.Type.Quaternion, Quaternion = from };

        /// <summary>
        /// Converts a <see cref="Color"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Color"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Color"/>.</returns>
        public static godot_variant CreateFromColor(Color from)
            => new() { Type = Variant.Type.Color, Color = from };

        /// <summary>
        /// Converts a <see cref="Plane"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Plane"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Plane"/>.</returns>
        public static godot_variant CreateFromPlane(Plane from)
            => new() { Type = Variant.Type.Plane, Plane = from };

        /// <summary>
        /// Converts a <see cref="Transform2D"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Transform2D"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Transform2D"/>.</returns>
        public static godot_variant CreateFromTransform2D(Transform2D from)
        {
            NativeFuncs.godotsharp_variant_new_transform2d(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="Basis"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Basis"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Basis"/>.</returns>
        public static godot_variant CreateFromBasis(Basis from)
        {
            NativeFuncs.godotsharp_variant_new_basis(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="Transform3D"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Transform3D"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Transform3D"/>.</returns>
        public static godot_variant CreateFromTransform3D(Transform3D from)
        {
            NativeFuncs.godotsharp_variant_new_transform3d(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="Projection"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Projection"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Projection"/>.</returns>
        public static godot_variant CreateFromProjection(Projection from)
        {
            NativeFuncs.godotsharp_variant_new_projection(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts an <see cref="Aabb"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Aabb"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Aabb"/>.</returns>
        public static godot_variant CreateFromAabb(Aabb from)
        {
            NativeFuncs.godotsharp_variant_new_aabb(out godot_variant ret, from);
            return ret;
        }

        // Explicit name to make it very clear
        /// <summary>
        /// Converts a <see cref="godot_callable"/> to a <see cref="godot_variant"/>, taking ownership of the disposable value in the process.
        /// </summary>
        /// <param name="from">The <see cref="godot_callable"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_callable"/>.</returns>
        public static godot_variant CreateFromCallableTakingOwnershipOfDisposableValue(godot_callable from)
            => new() { Type = Variant.Type.Callable, Callable = from };

        /// <summary>
        /// Converts a <see cref="Callable"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Callable"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Callable"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromCallable(Callable from)
            => CreateFromCallableTakingOwnershipOfDisposableValue(
                Marshaling.ConvertCallableToNative(from));

        // Explicit name to make it very clear
        /// <summary>
        /// Converts a <see cref="godot_signal"/> to a <see cref="godot_variant"/>, taking ownership of the disposable value in the process.
        /// </summary>
        /// <param name="from">The <see cref="godot_signal"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_signal"/>.</returns>
        public static godot_variant CreateFromSignalTakingOwnershipOfDisposableValue(godot_signal from)
            => new() { Type = Variant.Type.Signal, Signal = from };

        /// <summary>
        /// Converts a <see cref="Signal"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Signal"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Signal"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromSignal(Signal from)
            => CreateFromSignalTakingOwnershipOfDisposableValue(
                Marshaling.ConvertSignalToNative(from));

        // Explicit name to make it very clear
        /// <summary>
        /// Converts a <see cref="godot_string"/> to a <see cref="godot_variant"/>, taking ownership of the disposable value in the process.
        /// </summary>
        /// <param name="from">The <see cref="godot_string"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_string"/>.</returns>
        public static godot_variant CreateFromStringTakingOwnershipOfDisposableValue(godot_string from)
            => new() { Type = Variant.Type.String, String = from };

        /// <summary>
        /// Converts a <see langword="string"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="string"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="string"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromString(string? from)
            => CreateFromStringTakingOwnershipOfDisposableValue(Marshaling.ConvertStringToNative(from));

        /// <summary>
        /// Converts a <see cref="godot_packed_byte_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_byte_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_byte_array"/>.</returns>
        public static godot_variant CreateFromPackedByteArray(in godot_packed_byte_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_byte_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_int32_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_int32_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_int32_array"/>.</returns>
        public static godot_variant CreateFromPackedInt32Array(in godot_packed_int32_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_int32_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_int64_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_int64_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_int64_array"/>.</returns>
        public static godot_variant CreateFromPackedInt64Array(in godot_packed_int64_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_int64_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_float32_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_float32_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_float32_array"/>.</returns>
        public static godot_variant CreateFromPackedFloat32Array(in godot_packed_float32_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_float32_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_float64_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_float64_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_float64_array"/>.</returns>
        public static godot_variant CreateFromPackedFloat64Array(in godot_packed_float64_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_float64_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_string_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_string_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_string_array"/>.</returns>
        public static godot_variant CreateFromPackedStringArray(in godot_packed_string_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_string_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_vector2_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_vector2_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_vector2_array"/>.</returns>
        public static godot_variant CreateFromPackedVector2Array(in godot_packed_vector2_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_vector2_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_vector3_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_vector3_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_vector3_array"/>.</returns>
        public static godot_variant CreateFromPackedVector3Array(in godot_packed_vector3_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_vector3_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="godot_packed_color_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_packed_color_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_packed_color_array"/>.</returns>
        public static godot_variant CreateFromPackedColorArray(in godot_packed_color_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_color_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see langword="byte"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="byte"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="byte"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedByteArray(Span<byte> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedByteArray(from);
            return CreateFromPackedByteArray(nativePackedArray);
        }

        /// <summary>
        /// Converts an <see langword="int"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="int"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="int"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedInt32Array(Span<int> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedInt32Array(from);
            return CreateFromPackedInt32Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see langword="long"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="long"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="long"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedInt64Array(Span<long> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedInt64Array(from);
            return CreateFromPackedInt64Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see langword="float"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="float"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="float"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedFloat32Array(Span<float> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedFloat32Array(from);
            return CreateFromPackedFloat32Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see langword="double"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="double"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="double"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedFloat64Array(Span<double> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedFloat64Array(from);
            return CreateFromPackedFloat64Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see langword="string"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see langword="string"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see langword="string"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedStringArray(Span<string> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedStringArray(from);
            return CreateFromPackedStringArray(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see cref="Vector2"/> span to a <see cref="godot_variant"/> span.
        /// </summary>
        /// <param name="from">The <see cref="Vector2"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector2"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedVector2Array(Span<Vector2> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedVector2Array(from);
            return CreateFromPackedVector2Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see cref="Vector3"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Vector3"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Vector3"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedVector3Array(Span<Vector3> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedVector3Array(from);
            return CreateFromPackedVector3Array(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see cref="Color"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Color"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Color"/> span.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedColorArray(Span<Color> from)
        {
            using var nativePackedArray = Marshaling.ConvertSystemArrayToNativePackedColorArray(from);
            return CreateFromPackedColorArray(nativePackedArray);
        }

        /// <summary>
        /// Converts a <see cref="StringName"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="StringName"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="StringName"/> span.</returns>
        public static godot_variant CreateFromSystemArrayOfStringName(Span<StringName> from)
            => CreateFromArray(new Collections.Array(from));

        /// <summary>
        /// Converts a <see cref="NodePath"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="NodePath"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="NodePath"/> span.</returns>
        public static godot_variant CreateFromSystemArrayOfNodePath(Span<NodePath> from)
            => CreateFromArray(new Collections.Array(from));

        /// <summary>
        /// Converts an <see cref="Rid"/> span to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Rid"/> span to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Rid"/> span.</returns>
        public static godot_variant CreateFromSystemArrayOfRid(Span<Rid> from)
            => CreateFromArray(new Collections.Array(from));

        /// <summary>
        /// Converts a <see cref="GodotObject"/>[] to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="GodotObject"/>[] to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="GodotObject"/>[].</returns>
        // ReSharper disable once RedundantNameQualifier
        public static godot_variant CreateFromSystemArrayOfGodotObject(GodotObject[]? from)
        {
            if (from == null)
                return default; // Nil
            using var fromGodot = new Collections.Array(from);
            return CreateFromArray((godot_array)fromGodot.NativeValue);
        }

        /// <summary>
        /// Converts a <see cref="godot_array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_array"/>.</returns>
        public static godot_variant CreateFromArray(godot_array from)
        {
            NativeFuncs.godotsharp_variant_new_array(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts an <see cref="Collections.Array"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Collections.Array"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Collections.Array"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromArray(Collections.Array? from)
            => from != null ? CreateFromArray((godot_array)from.NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="Array{T}"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Array{T}"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Array{T}"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromArray<[MustBeVariant] T>(Array<T>? from)
            => from != null ? CreateFromArray((godot_array)((Collections.Array)from).NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="godot_dictionary"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_dictionary"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_dictionary"/>.</returns>
        public static godot_variant CreateFromDictionary(godot_dictionary from)
        {
            NativeFuncs.godotsharp_variant_new_dictionary(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="Dictionary"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Dictionary"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Dictionary"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromDictionary(Dictionary? from)
            => from != null ? CreateFromDictionary((godot_dictionary)from.NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="Dictionary{TKey, TValue}"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="Dictionary{TKey, TValue}"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="Dictionary{TKey, TValue}"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromDictionary<[MustBeVariant] TKey, [MustBeVariant] TValue>(Dictionary<TKey, TValue>? from)
            => from != null ? CreateFromDictionary((godot_dictionary)((Dictionary)from).NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="godot_string_name"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_string_name"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_string_name"/>.</returns>
        public static godot_variant CreateFromStringName(godot_string_name from)
        {
            NativeFuncs.godotsharp_variant_new_string_name(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="StringName"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="StringName"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="StringName"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromStringName(StringName? from)
            => from != null ? CreateFromStringName((godot_string_name)from.NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="godot_node_path"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="godot_node_path"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="godot_node_path"/>.</returns>
        public static godot_variant CreateFromNodePath(godot_node_path from)
        {
            NativeFuncs.godotsharp_variant_new_node_path(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="NodePath"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="NodePath"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="NodePath"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromNodePath(NodePath? from)
            => from != null ? CreateFromNodePath((godot_node_path)from.NativeValue) : default;

        /// <summary>
        /// Converts a <see cref="IntPtr"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="IntPtr"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="IntPtr"/>.</returns>
        public static godot_variant CreateFromGodotObjectPtr(IntPtr from)
        {
            if (from == IntPtr.Zero)
                return new godot_variant();
            NativeFuncs.godotsharp_variant_new_object(out godot_variant ret, from);
            return ret;
        }

        /// <summary>
        /// Converts a <see cref="GodotObject"/> to a <see cref="godot_variant"/>.
        /// </summary>
        /// <param name="from">The <see cref="GodotObject"/> to convert.</param>
        /// <returns>A <see cref="godot_variant"/> representation of this <see cref="GodotObject"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once RedundantNameQualifier
        public static godot_variant CreateFromGodotObject(GodotObject? from)
            => from != null ? CreateFromGodotObjectPtr(GodotObject.GetPtr(from)) : default;

        // We avoid the internal call if the stored type is the same we want.

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="bool"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="bool"/> representation of this <see cref="godot_variant"/>.</returns>
        public static bool ConvertToBool(in godot_variant p_var)
            => p_var.Type == Variant.Type.Bool ?
                p_var.Bool.ToBool() :
                NativeFuncs.godotsharp_variant_as_bool(p_var).ToBool();

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="char"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="char"/> representation of this <see cref="godot_variant"/>.</returns>
        public static char ConvertToChar(in godot_variant p_var)
            => (char)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see langword="sbyte"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see langword="sbyte"/> representation of this <see cref="godot_variant"/>.</returns>
        public static sbyte ConvertToInt8(in godot_variant p_var)
            => (sbyte)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="short"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="short"/> representation of this <see cref="godot_variant"/>.</returns>
        public static short ConvertToInt16(in godot_variant p_var)
            => (short)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see langword="int"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see langword="int"/> representation of this <see cref="godot_variant"/>.</returns>
        public static int ConvertToInt32(in godot_variant p_var)
            => (int)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="long"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="long"/> representation of this <see cref="godot_variant"/>.</returns>
        public static long ConvertToInt64(in godot_variant p_var)
            => p_var.Type == Variant.Type.Int ? p_var.Int : NativeFuncs.godotsharp_variant_as_int(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="byte"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="byte"/> representation of this <see cref="godot_variant"/>.</returns>
        public static byte ConvertToUInt8(in godot_variant p_var)
            => (byte)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="ushort"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="ushort"/> representation of this <see cref="godot_variant"/>.</returns>
        public static ushort ConvertToUInt16(in godot_variant p_var)
            => (ushort)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="uint"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="uint"/> representation of this <see cref="godot_variant"/>.</returns>
        public static uint ConvertToUInt32(in godot_variant p_var)
            => (uint)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="ulong"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="ulong"/> representation of this <see cref="godot_variant"/>.</returns>
        public static ulong ConvertToUInt64(in godot_variant p_var)
            => (ulong)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="float"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="float"/> representation of this <see cref="godot_variant"/>.</returns>
        public static float ConvertToFloat32(in godot_variant p_var)
            => (float)(p_var.Type == Variant.Type.Float ?
                p_var.Float :
                NativeFuncs.godotsharp_variant_as_float(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="double"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="double"/> representation of this <see cref="godot_variant"/>.</returns>
        public static double ConvertToFloat64(in godot_variant p_var)
            => p_var.Type == Variant.Type.Float ?
                p_var.Float :
                NativeFuncs.godotsharp_variant_as_float(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector2"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector2"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Vector2 ConvertToVector2(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector2 ?
                p_var.Vector2 :
                NativeFuncs.godotsharp_variant_as_vector2(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector2I"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector2I"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Vector2I ConvertToVector2I(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector2I ?
                p_var.Vector2I :
                NativeFuncs.godotsharp_variant_as_vector2i(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Rect2"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Rect2"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Rect2 ConvertToRect2(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rect2 ?
                p_var.Rect2 :
                NativeFuncs.godotsharp_variant_as_rect2(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Rect2I"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Rect2I"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Rect2I ConvertToRect2I(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rect2I ?
                p_var.Rect2I :
                NativeFuncs.godotsharp_variant_as_rect2i(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Transform2D"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Transform2D"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Transform2D ConvertToTransform2D(in godot_variant p_var)
            => p_var.Type == Variant.Type.Transform2D ?
                *p_var.Transform2D :
                NativeFuncs.godotsharp_variant_as_transform2d(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector3"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector3"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Vector3 ConvertToVector3(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector3 ?
                p_var.Vector3 :
                NativeFuncs.godotsharp_variant_as_vector3(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector3I"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector3I"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Vector3I ConvertToVector3I(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector3I ?
                p_var.Vector3I :
                NativeFuncs.godotsharp_variant_as_vector3i(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector4"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector4"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Vector4 ConvertToVector4(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector4 ?
                p_var.Vector4 :
                NativeFuncs.godotsharp_variant_as_vector4(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector4I"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector4I"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Vector4I ConvertToVector4I(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector4I ?
                p_var.Vector4I :
                NativeFuncs.godotsharp_variant_as_vector4i(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Basis"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Basis"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Basis ConvertToBasis(in godot_variant p_var)
            => p_var.Type == Variant.Type.Basis ?
                *p_var.Basis :
                NativeFuncs.godotsharp_variant_as_basis(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Quaternion"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Quaternion"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Quaternion ConvertToQuaternion(in godot_variant p_var)
            => p_var.Type == Variant.Type.Quaternion ?
                p_var.Quaternion :
                NativeFuncs.godotsharp_variant_as_quaternion(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Transform3D"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Transform3D"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Transform3D ConvertToTransform3D(in godot_variant p_var)
            => p_var.Type == Variant.Type.Transform3D ?
                *p_var.Transform3D :
                NativeFuncs.godotsharp_variant_as_transform3d(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Projection"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Projection"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Projection ConvertToProjection(in godot_variant p_var)
            => p_var.Type == Variant.Type.Projection ?
                *p_var.Projection :
                NativeFuncs.godotsharp_variant_as_projection(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Aabb"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Aabb"/> representation of this <see cref="godot_variant"/>.</returns>
        public static unsafe Aabb ConvertToAabb(in godot_variant p_var)
            => p_var.Type == Variant.Type.Aabb ?
                *p_var.Aabb :
                NativeFuncs.godotsharp_variant_as_aabb(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Color"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Color"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Color ConvertToColor(in godot_variant p_var)
            => p_var.Type == Variant.Type.Color ?
                p_var.Color :
                NativeFuncs.godotsharp_variant_as_color(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Plane"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Plane"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Plane ConvertToPlane(in godot_variant p_var)
            => p_var.Type == Variant.Type.Plane ?
                p_var.Plane :
                NativeFuncs.godotsharp_variant_as_plane(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Rid"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Rid"/> representation of this <see cref="godot_variant"/>.</returns>
        public static Rid ConvertToRid(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rid ?
                p_var.Rid :
                NativeFuncs.godotsharp_variant_as_rid(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="IntPtr"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="IntPtr"/> representation of this <see cref="godot_variant"/>.</returns>
        public static IntPtr ConvertToGodotObjectPtr(in godot_variant p_var)
            => p_var.Type == Variant.Type.Object ? p_var.Object : IntPtr.Zero;

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="GodotObject"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="GodotObject"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once RedundantNameQualifier
        public static GodotObject ConvertToGodotObject(in godot_variant p_var)
            => InteropUtils.UnmanagedGetManaged(ConvertToGodotObjectPtr(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="string"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="string"/> representation of this <see cref="godot_variant"/>.</returns>
        public static string ConvertToString(in godot_variant p_var)
        {
            switch (p_var.Type)
            {
                case Variant.Type.Nil:
                    return ""; // Otherwise, Variant -> String would return the string "Null"
                case Variant.Type.String:
                {
                    // We avoid the internal call if the stored type is the same we want.
                    return Marshaling.ConvertStringToManaged(p_var.String);
                }
                default:
                {
                    using godot_string godotString = NativeFuncs.godotsharp_variant_as_string(p_var);
                    return Marshaling.ConvertStringToManaged(godotString);
                }
            }
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_string_name"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_string_name"/> representation of this <see cref="godot_variant"/>.</returns>
        public static godot_string_name ConvertToNativeStringName(in godot_variant p_var)
            => p_var.Type == Variant.Type.StringName ?
                NativeFuncs.godotsharp_string_name_new_copy(p_var.StringName) :
                NativeFuncs.godotsharp_variant_as_string_name(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="StringName"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="StringName"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static StringName ConvertToStringName(in godot_variant p_var)
            => StringName.CreateTakingOwnershipOfDisposableValue(ConvertToNativeStringName(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_node_path"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_node_path"/> representation of this <see cref="godot_variant"/>.</returns>
        public static godot_node_path ConvertToNativeNodePath(in godot_variant p_var)
            => p_var.Type == Variant.Type.NodePath ?
                NativeFuncs.godotsharp_node_path_new_copy(p_var.NodePath) :
                NativeFuncs.godotsharp_variant_as_node_path(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="NodePath"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="NodePath"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NodePath ConvertToNodePath(in godot_variant p_var)
            => NodePath.CreateTakingOwnershipOfDisposableValue(ConvertToNativeNodePath(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_callable"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_callable"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_callable ConvertToNativeCallable(in godot_variant p_var)
            => NativeFuncs.godotsharp_variant_as_callable(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Callable"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Callable"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Callable ConvertToCallable(in godot_variant p_var)
            => Marshaling.ConvertCallableToManaged(ConvertToNativeCallable(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_signal"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_signal"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_signal ConvertToNativeSignal(in godot_variant p_var)
            => NativeFuncs.godotsharp_variant_as_signal(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Signal"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Signal"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Signal ConvertToSignal(in godot_variant p_var)
            => Marshaling.ConvertSignalToManaged(ConvertToNativeSignal(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_array"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_array"/> representation of this <see cref="godot_variant"/>.</returns>
        public static godot_array ConvertToNativeArray(in godot_variant p_var)
            => p_var.Type == Variant.Type.Array ?
                NativeFuncs.godotsharp_array_new_copy(p_var.Array) :
                NativeFuncs.godotsharp_variant_as_array(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see cref="Collections.Array"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see cref="Collections.Array"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Collections.Array ConvertToArray(in godot_variant p_var)
            => Collections.Array.CreateTakingOwnershipOfDisposableValue(ConvertToNativeArray(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see cref="Array{T}"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> ConvertToArray<[MustBeVariant] T>(in godot_variant p_var)
            => Array<T>.CreateTakingOwnershipOfDisposableValue(ConvertToNativeArray(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="godot_dictionary"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="godot_dictionary"/> representation of this <see cref="godot_variant"/>.</returns>
        public static godot_dictionary ConvertToNativeDictionary(in godot_variant p_var)
            => p_var.Type == Variant.Type.Dictionary ?
                NativeFuncs.godotsharp_dictionary_new_copy(p_var.Dictionary) :
                NativeFuncs.godotsharp_variant_as_dictionary(p_var);

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Dictionary"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary ConvertToDictionary(in godot_variant p_var)
            => Dictionary.CreateTakingOwnershipOfDisposableValue(ConvertToNativeDictionary(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Dictionary{TKey, TValue}"/> representation of this <see cref="godot_variant"/>.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<TKey, TValue> ConvertToDictionary<[MustBeVariant] TKey, [MustBeVariant] TValue>(in godot_variant p_var)
            => Dictionary<TKey, TValue>.CreateTakingOwnershipOfDisposableValue(ConvertToNativeDictionary(p_var));

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="byte"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="byte"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static byte[] ConvertAsPackedByteArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
            return Marshaling.ConvertNativePackedByteArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see langword="int"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see langword="int"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static int[] ConvertAsPackedInt32ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
            return Marshaling.ConvertNativePackedInt32ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="long"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="long"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static long[] ConvertAsPackedInt64ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
            return Marshaling.ConvertNativePackedInt64ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="float"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="float"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static float[] ConvertAsPackedFloat32ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
            return Marshaling.ConvertNativePackedFloat32ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="double"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="double"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static double[] ConvertAsPackedFloat64ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
            return Marshaling.ConvertNativePackedFloat64ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see langword="string"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see langword="string"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static string[] ConvertAsPackedStringArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
            return Marshaling.ConvertNativePackedStringArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector2"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector2"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static Vector2[] ConvertAsPackedVector2ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
            return Marshaling.ConvertNativePackedVector2ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Vector3"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Vector3"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static Vector3[] ConvertAsPackedVector3ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
            return Marshaling.ConvertNativePackedVector3ArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="Color"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="Color"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static Color[] ConvertAsPackedColorArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
            return Marshaling.ConvertNativePackedColorArrayToSystemArray(packedArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="StringName"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="StringName"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static StringName[] ConvertToSystemArrayOfStringName(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfStringName(godotArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to a <see cref="NodePath"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>A <see cref="NodePath"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static NodePath[] ConvertToSystemArrayOfNodePath(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfNodePath(godotArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see cref="Rid"/>[].
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see cref="Rid"/>[] representation of this <see cref="godot_variant"/>.</returns>
        public static Rid[] ConvertToSystemArrayOfRid(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfRid(godotArray);
        }

        /// <summary>
        /// Converts a <see cref="godot_variant"/> to an <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="p_var">The <see cref="godot_variant"/> to convert.</param>
        /// <returns>An <see cref="Array{T}"/> representation of this <see cref="godot_variant"/>.</returns>
        public static T[] ConvertToSystemArrayOfGodotObject<T>(in godot_variant p_var)
            // ReSharper disable once RedundantNameQualifier
            where T : GodotObject
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfGodotObjectType<T>(godotArray);
        }
    }
}
