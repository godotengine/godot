using System;
using System.Runtime.CompilerServices;
using Godot.Collections;

// ReSharper disable InconsistentNaming

#nullable enable

namespace Godot.NativeInterop
{
    public static partial class VariantUtils
    {
        public static godot_variant CreateFromRID(RID from)
            => new() { Type = Variant.Type.Rid, RID = from };

        public static godot_variant CreateFromBool(bool from)
            => new() { Type = Variant.Type.Bool, Bool = from.ToGodotBool() };

        public static godot_variant CreateFromInt(long from)
            => new() { Type = Variant.Type.Int, Int = from };

        public static godot_variant CreateFromInt(ulong from)
            => new() { Type = Variant.Type.Int, Int = (long)from };

        public static godot_variant CreateFromFloat(double from)
            => new() { Type = Variant.Type.Float, Float = from };

        public static godot_variant CreateFromVector2(Vector2 from)
            => new() { Type = Variant.Type.Vector2, Vector2 = from };

        public static godot_variant CreateFromVector2i(Vector2i from)
            => new() { Type = Variant.Type.Vector2i, Vector2i = from };

        public static godot_variant CreateFromVector3(Vector3 from)
            => new() { Type = Variant.Type.Vector3, Vector3 = from };

        public static godot_variant CreateFromVector3i(Vector3i from)
            => new() { Type = Variant.Type.Vector3i, Vector3i = from };

        public static godot_variant CreateFromVector4(Vector4 from)
            => new() { Type = Variant.Type.Vector4, Vector4 = from };

        public static godot_variant CreateFromVector4i(Vector4i from)
            => new() { Type = Variant.Type.Vector4i, Vector4i = from };

        public static godot_variant CreateFromRect2(Rect2 from)
            => new() { Type = Variant.Type.Rect2, Rect2 = from };

        public static godot_variant CreateFromRect2i(Rect2i from)
            => new() { Type = Variant.Type.Rect2i, Rect2i = from };

        public static godot_variant CreateFromQuaternion(Quaternion from)
            => new() { Type = Variant.Type.Quaternion, Quaternion = from };

        public static godot_variant CreateFromColor(Color from)
            => new() { Type = Variant.Type.Color, Color = from };

        public static godot_variant CreateFromPlane(Plane from)
            => new() { Type = Variant.Type.Plane, Plane = from };

        public static godot_variant CreateFromTransform2D(Transform2D from)
        {
            NativeFuncs.godotsharp_variant_new_transform2d(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromBasis(Basis from)
        {
            NativeFuncs.godotsharp_variant_new_basis(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromTransform3D(Transform3D from)
        {
            NativeFuncs.godotsharp_variant_new_transform3d(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromProjection(Projection from)
        {
            NativeFuncs.godotsharp_variant_new_projection(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromAABB(AABB from)
        {
            NativeFuncs.godotsharp_variant_new_aabb(out godot_variant ret, from);
            return ret;
        }

        // Explicit name to make it very clear
        public static godot_variant CreateFromCallableTakingOwnershipOfDisposableValue(godot_callable from)
            => new() { Type = Variant.Type.Callable, Callable = from };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromCallable(Callable from)
            => CreateFromCallableTakingOwnershipOfDisposableValue(
                Marshaling.ConvertCallableToNative(from));

        // Explicit name to make it very clear
        public static godot_variant CreateFromSignalTakingOwnershipOfDisposableValue(godot_signal from)
            => new() { Type = Variant.Type.Signal, Signal = from };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromSignal(Signal from)
            => CreateFromSignalTakingOwnershipOfDisposableValue(
                Marshaling.ConvertSignalToNative(from));

        // Explicit name to make it very clear
        public static godot_variant CreateFromStringTakingOwnershipOfDisposableValue(godot_string from)
            => new() { Type = Variant.Type.String, String = from };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromString(string? from)
            => CreateFromStringTakingOwnershipOfDisposableValue(Marshaling.ConvertStringToNative(from));

        public static godot_variant CreateFromPackedByteArray(in godot_packed_byte_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_byte_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedInt32Array(in godot_packed_int32_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_int32_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedInt64Array(in godot_packed_int64_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_int64_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedFloat32Array(in godot_packed_float32_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_float32_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedFloat64Array(in godot_packed_float64_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_float64_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedStringArray(in godot_packed_string_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_string_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedVector2Array(in godot_packed_vector2_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_vector2_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedVector3Array(in godot_packed_vector3_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_vector3_array(out godot_variant ret, from);
            return ret;
        }

        public static godot_variant CreateFromPackedColorArray(in godot_packed_color_array from)
        {
            NativeFuncs.godotsharp_variant_new_packed_color_array(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedByteArray(Span<byte> from)
            => CreateFromPackedByteArray(Marshaling.ConvertSystemArrayToNativePackedByteArray(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedInt32Array(Span<int> from)
            => CreateFromPackedInt32Array(Marshaling.ConvertSystemArrayToNativePackedInt32Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedInt64Array(Span<long> from)
            => CreateFromPackedInt64Array(Marshaling.ConvertSystemArrayToNativePackedInt64Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedFloat32Array(Span<float> from)
            => CreateFromPackedFloat32Array(Marshaling.ConvertSystemArrayToNativePackedFloat32Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedFloat64Array(Span<double> from)
            => CreateFromPackedFloat64Array(Marshaling.ConvertSystemArrayToNativePackedFloat64Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedStringArray(Span<string> from)
            => CreateFromPackedStringArray(Marshaling.ConvertSystemArrayToNativePackedStringArray(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedVector2Array(Span<Vector2> from)
            => CreateFromPackedVector2Array(Marshaling.ConvertSystemArrayToNativePackedVector2Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedVector3Array(Span<Vector3> from)
            => CreateFromPackedVector3Array(Marshaling.ConvertSystemArrayToNativePackedVector3Array(from));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromPackedColorArray(Span<Color> from)
            => CreateFromPackedColorArray(Marshaling.ConvertSystemArrayToNativePackedColorArray(from));

        public static godot_variant CreateFromSystemArrayOfStringName(Span<StringName> from)
            => CreateFromArray(new Collections.Array(from));

        public static godot_variant CreateFromSystemArrayOfNodePath(Span<NodePath> from)
            => CreateFromArray(new Collections.Array(from));

        public static godot_variant CreateFromSystemArrayOfRID(Span<RID> from)
            => CreateFromArray(new Collections.Array(from));

        // ReSharper disable once RedundantNameQualifier
        public static godot_variant CreateFromSystemArrayOfGodotObject(Godot.Object[]? from)
        {
            if (from == null)
                return default; // Nil
            using var fromGodot = new Collections.Array(from);
            return CreateFromArray((godot_array)fromGodot.NativeValue);
        }

        public static godot_variant CreateFromArray(godot_array from)
        {
            NativeFuncs.godotsharp_variant_new_array(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromArray(Collections.Array? from)
            => from != null ? CreateFromArray((godot_array)from.NativeValue) : default;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromArray<T>(Array<T>? from)
            => from != null ? CreateFromArray((godot_array)((Collections.Array)from).NativeValue) : default;

        public static godot_variant CreateFromDictionary(godot_dictionary from)
        {
            NativeFuncs.godotsharp_variant_new_dictionary(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromDictionary(Dictionary? from)
            => from != null ? CreateFromDictionary((godot_dictionary)from.NativeValue) : default;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromDictionary<TKey, TValue>(Dictionary<TKey, TValue>? from)
            => from != null ? CreateFromDictionary((godot_dictionary)((Dictionary)from).NativeValue) : default;

        public static godot_variant CreateFromStringName(godot_string_name from)
        {
            NativeFuncs.godotsharp_variant_new_string_name(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromStringName(StringName? from)
            => from != null ? CreateFromStringName((godot_string_name)from.NativeValue) : default;

        public static godot_variant CreateFromNodePath(godot_node_path from)
        {
            NativeFuncs.godotsharp_variant_new_node_path(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_variant CreateFromNodePath(NodePath? from)
            => from != null ? CreateFromNodePath((godot_node_path)from.NativeValue) : default;

        public static godot_variant CreateFromGodotObjectPtr(IntPtr from)
        {
            if (from == IntPtr.Zero)
                return new godot_variant();
            NativeFuncs.godotsharp_variant_new_object(out godot_variant ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once RedundantNameQualifier
        public static godot_variant CreateFromGodotObject(Godot.Object? from)
            => from != null ? CreateFromGodotObjectPtr(Object.GetPtr(from)) : default;

        // We avoid the internal call if the stored type is the same we want.

        public static bool ConvertToBool(in godot_variant p_var)
            => p_var.Type == Variant.Type.Bool ?
                p_var.Bool.ToBool() :
                NativeFuncs.godotsharp_variant_as_bool(p_var).ToBool();

        public static char ConvertToChar(in godot_variant p_var)
            => (char)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static sbyte ConvertToInt8(in godot_variant p_var)
            => (sbyte)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static short ConvertToInt16(in godot_variant p_var)
            => (short)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static int ConvertToInt32(in godot_variant p_var)
            => (int)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static long ConvertToInt64(in godot_variant p_var)
            => p_var.Type == Variant.Type.Int ? p_var.Int : NativeFuncs.godotsharp_variant_as_int(p_var);

        public static byte ConvertToUInt8(in godot_variant p_var)
            => (byte)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static ushort ConvertToUInt16(in godot_variant p_var)
            => (ushort)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static uint ConvertToUInt32(in godot_variant p_var)
            => (uint)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static ulong ConvertToUInt64(in godot_variant p_var)
            => (ulong)(p_var.Type == Variant.Type.Int ?
                p_var.Int :
                NativeFuncs.godotsharp_variant_as_int(p_var));

        public static float ConvertToFloat32(in godot_variant p_var)
            => (float)(p_var.Type == Variant.Type.Float ?
                p_var.Float :
                NativeFuncs.godotsharp_variant_as_float(p_var));

        public static double ConvertToFloat64(in godot_variant p_var)
            => p_var.Type == Variant.Type.Float ?
                p_var.Float :
                NativeFuncs.godotsharp_variant_as_float(p_var);

        public static Vector2 ConvertToVector2(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector2 ?
                p_var.Vector2 :
                NativeFuncs.godotsharp_variant_as_vector2(p_var);

        public static Vector2i ConvertToVector2i(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector2i ?
                p_var.Vector2i :
                NativeFuncs.godotsharp_variant_as_vector2i(p_var);

        public static Rect2 ConvertToRect2(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rect2 ?
                p_var.Rect2 :
                NativeFuncs.godotsharp_variant_as_rect2(p_var);

        public static Rect2i ConvertToRect2i(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rect2i ?
                p_var.Rect2i :
                NativeFuncs.godotsharp_variant_as_rect2i(p_var);

        public static unsafe Transform2D ConvertToTransform2D(in godot_variant p_var)
            => p_var.Type == Variant.Type.Transform2d ?
                *p_var.Transform2D :
                NativeFuncs.godotsharp_variant_as_transform2d(p_var);

        public static Vector3 ConvertToVector3(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector3 ?
                p_var.Vector3 :
                NativeFuncs.godotsharp_variant_as_vector3(p_var);

        public static Vector3i ConvertToVector3i(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector3i ?
                p_var.Vector3i :
                NativeFuncs.godotsharp_variant_as_vector3i(p_var);

        public static unsafe Vector4 ConvertToVector4(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector4 ?
                p_var.Vector4 :
                NativeFuncs.godotsharp_variant_as_vector4(p_var);

        public static unsafe Vector4i ConvertToVector4i(in godot_variant p_var)
            => p_var.Type == Variant.Type.Vector4i ?
                p_var.Vector4i :
                NativeFuncs.godotsharp_variant_as_vector4i(p_var);

        public static unsafe Basis ConvertToBasis(in godot_variant p_var)
            => p_var.Type == Variant.Type.Basis ?
                *p_var.Basis :
                NativeFuncs.godotsharp_variant_as_basis(p_var);

        public static Quaternion ConvertToQuaternion(in godot_variant p_var)
            => p_var.Type == Variant.Type.Quaternion ?
                p_var.Quaternion :
                NativeFuncs.godotsharp_variant_as_quaternion(p_var);

        public static unsafe Transform3D ConvertToTransform3D(in godot_variant p_var)
            => p_var.Type == Variant.Type.Transform3d ?
                *p_var.Transform3D :
                NativeFuncs.godotsharp_variant_as_transform3d(p_var);

        public static unsafe Projection ConvertToProjection(in godot_variant p_var)
            => p_var.Type == Variant.Type.Projection ?
                *p_var.Projection :
                NativeFuncs.godotsharp_variant_as_projection(p_var);

        public static unsafe AABB ConvertToAABB(in godot_variant p_var)
            => p_var.Type == Variant.Type.Aabb ?
                *p_var.AABB :
                NativeFuncs.godotsharp_variant_as_aabb(p_var);

        public static Color ConvertToColor(in godot_variant p_var)
            => p_var.Type == Variant.Type.Color ?
                p_var.Color :
                NativeFuncs.godotsharp_variant_as_color(p_var);

        public static Plane ConvertToPlane(in godot_variant p_var)
            => p_var.Type == Variant.Type.Plane ?
                p_var.Plane :
                NativeFuncs.godotsharp_variant_as_plane(p_var);

        public static RID ConvertToRID(in godot_variant p_var)
            => p_var.Type == Variant.Type.Rid ?
                p_var.RID :
                NativeFuncs.godotsharp_variant_as_rid(p_var);

        public static IntPtr ConvertToGodotObjectPtr(in godot_variant p_var)
            => p_var.Type == Variant.Type.Object ? p_var.Object : IntPtr.Zero;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        // ReSharper disable once RedundantNameQualifier
        public static Godot.Object ConvertToGodotObject(in godot_variant p_var)
            => InteropUtils.UnmanagedGetManaged(ConvertToGodotObjectPtr(p_var));

        public static string ConvertToStringObject(in godot_variant p_var)
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

        public static godot_string_name ConvertToStringName(in godot_variant p_var)
            => p_var.Type == Variant.Type.StringName ?
                NativeFuncs.godotsharp_string_name_new_copy(p_var.StringName) :
                NativeFuncs.godotsharp_variant_as_string_name(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static StringName ConvertToStringNameObject(in godot_variant p_var)
            => StringName.CreateTakingOwnershipOfDisposableValue(ConvertToStringName(p_var));

        public static godot_node_path ConvertToNodePath(in godot_variant p_var)
            => p_var.Type == Variant.Type.NodePath ?
                NativeFuncs.godotsharp_node_path_new_copy(p_var.NodePath) :
                NativeFuncs.godotsharp_variant_as_node_path(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NodePath ConvertToNodePathObject(in godot_variant p_var)
            => NodePath.CreateTakingOwnershipOfDisposableValue(ConvertToNodePath(p_var));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_callable ConvertToCallable(in godot_variant p_var)
            => NativeFuncs.godotsharp_variant_as_callable(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Callable ConvertToCallableManaged(in godot_variant p_var)
            => Marshaling.ConvertCallableToManaged(ConvertToCallable(p_var));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static godot_signal ConvertToSignal(in godot_variant p_var)
            => NativeFuncs.godotsharp_variant_as_signal(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Signal ConvertToSignalManaged(in godot_variant p_var)
            => Marshaling.ConvertSignalToManaged(ConvertToSignal(p_var));

        public static godot_array ConvertToArray(in godot_variant p_var)
            => p_var.Type == Variant.Type.Array ?
                NativeFuncs.godotsharp_array_new_copy(p_var.Array) :
                NativeFuncs.godotsharp_variant_as_array(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Collections.Array ConvertToArrayObject(in godot_variant p_var)
            => Collections.Array.CreateTakingOwnershipOfDisposableValue(ConvertToArray(p_var));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Array<T> ConvertToArrayObject<T>(in godot_variant p_var)
            => Array<T>.CreateTakingOwnershipOfDisposableValue(ConvertToArray(p_var));

        public static godot_dictionary ConvertToDictionary(in godot_variant p_var)
            => p_var.Type == Variant.Type.Dictionary ?
                NativeFuncs.godotsharp_dictionary_new_copy(p_var.Dictionary) :
                NativeFuncs.godotsharp_variant_as_dictionary(p_var);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary ConvertToDictionaryObject(in godot_variant p_var)
            => Dictionary.CreateTakingOwnershipOfDisposableValue(ConvertToDictionary(p_var));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Dictionary<TKey, TValue> ConvertToDictionaryObject<TKey, TValue>(in godot_variant p_var)
            => Dictionary<TKey, TValue>.CreateTakingOwnershipOfDisposableValue(ConvertToDictionary(p_var));

        public static byte[] ConvertAsPackedByteArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_byte_array(p_var);
            return Marshaling.ConvertNativePackedByteArrayToSystemArray(packedArray);
        }

        public static int[] ConvertAsPackedInt32ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int32_array(p_var);
            return Marshaling.ConvertNativePackedInt32ArrayToSystemArray(packedArray);
        }

        public static long[] ConvertAsPackedInt64ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_int64_array(p_var);
            return Marshaling.ConvertNativePackedInt64ArrayToSystemArray(packedArray);
        }

        public static float[] ConvertAsPackedFloat32ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float32_array(p_var);
            return Marshaling.ConvertNativePackedFloat32ArrayToSystemArray(packedArray);
        }

        public static double[] ConvertAsPackedFloat64ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_float64_array(p_var);
            return Marshaling.ConvertNativePackedFloat64ArrayToSystemArray(packedArray);
        }

        public static string[] ConvertAsPackedStringArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_string_array(p_var);
            return Marshaling.ConvertNativePackedStringArrayToSystemArray(packedArray);
        }

        public static Vector2[] ConvertAsPackedVector2ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector2_array(p_var);
            return Marshaling.ConvertNativePackedVector2ArrayToSystemArray(packedArray);
        }

        public static Vector3[] ConvertAsPackedVector3ArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_vector3_array(p_var);
            return Marshaling.ConvertNativePackedVector3ArrayToSystemArray(packedArray);
        }

        public static Color[] ConvertAsPackedColorArrayToSystemArray(in godot_variant p_var)
        {
            using var packedArray = NativeFuncs.godotsharp_variant_as_packed_color_array(p_var);
            return Marshaling.ConvertNativePackedColorArrayToSystemArray(packedArray);
        }

        public static StringName[] ConvertToSystemArrayOfStringName(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfStringName(godotArray);
        }

        public static NodePath[] ConvertToSystemArrayOfNodePath(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfNodePath(godotArray);
        }

        public static RID[] ConvertToSystemArrayOfRID(in godot_variant p_var)
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfRID(godotArray);
        }

        public static T[] ConvertToSystemArrayOfGodotObject<T>(in godot_variant p_var)
            // ReSharper disable once RedundantNameQualifier
            where T : Godot.Object
        {
            using var godotArray = NativeFuncs.godotsharp_variant_as_array(p_var);
            return Marshaling.ConvertNativeGodotArrayToSystemArrayOfGodotObjectType<T>(godotArray);
        }
    }
}
