using System;
using System.Runtime.CompilerServices;

// ReSharper disable InconsistentNaming

namespace Godot.NativeInterop
{
    internal static class VariantUtils
    {
        public static godot_variant CreateFromRID(RID from)
            => new() {_type = Variant.Type.Rid, _data = {_m_rid = from}};

        public static godot_variant CreateFromBool(bool from)
            => new() {_type = Variant.Type.Bool, _data = {_bool = from}};

        public static godot_variant CreateFromInt(long from)
            => new() {_type = Variant.Type.Int, _data = {_int = from}};

        public static godot_variant CreateFromInt(ulong from)
            => new() {_type = Variant.Type.Int, _data = {_int = (long)from}};

        public static godot_variant CreateFromFloat(double from)
            => new() {_type = Variant.Type.Float, _data = {_float = from}};

        public static godot_variant CreateFromVector2(Vector2 from)
            => new() {_type = Variant.Type.Vector2, _data = {_m_vector2 = from}};

        public static godot_variant CreateFromVector2i(Vector2i from)
            => new() {_type = Variant.Type.Vector2i, _data = {_m_vector2i = from}};

        public static godot_variant CreateFromVector3(Vector3 from)
            => new() {_type = Variant.Type.Vector3, _data = {_m_vector3 = from}};

        public static godot_variant CreateFromVector3i(Vector3i from)
            => new() {_type = Variant.Type.Vector3i, _data = {_m_vector3i = from}};

        public static godot_variant CreateFromRect2(Rect2 from)
            => new() {_type = Variant.Type.Rect2, _data = {_m_rect2 = from}};

        public static godot_variant CreateFromRect2i(Rect2i from)
            => new() {_type = Variant.Type.Rect2i, _data = {_m_rect2i = from}};

        public static godot_variant CreateFromQuaternion(Quaternion from)
            => new() {_type = Variant.Type.Quaternion, _data = {_m_quaternion = from}};

        public static godot_variant CreateFromColor(Color from)
            => new() {_type = Variant.Type.Color, _data = {_m_color = from}};

        public static godot_variant CreateFromPlane(Plane from)
            => new() {_type = Variant.Type.Plane, _data = {_m_plane = from}};

        public static unsafe godot_variant CreateFromTransform2D(Transform2D from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_transform2d(&ret, &from);
            return ret;
        }

        public static unsafe godot_variant CreateFromBasis(Basis from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_basis(&ret, &from);
            return ret;
        }

        public static unsafe godot_variant CreateFromTransform3D(Transform3D from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_transform3d(&ret, &from);
            return ret;
        }

        public static unsafe godot_variant CreateFromAABB(AABB from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_aabb(&ret, &from);
            return ret;
        }

        // Explicit name to make it very clear
        public static godot_variant CreateFromCallableTakingOwnershipOfDisposableValue(godot_callable from)
            => new() {_type = Variant.Type.Callable, _data = {_m_callable = from}};

        // Explicit name to make it very clear
        public static godot_variant CreateFromSignalTakingOwnershipOfDisposableValue(godot_signal from)
            => new() {_type = Variant.Type.Signal, _data = {_m_signal = from}};

        // Explicit name to make it very clear
        public static godot_variant CreateFromStringTakingOwnershipOfDisposableValue(godot_string from)
            => new() {_type = Variant.Type.String, _data = {_m_string = from}};

        public static unsafe godot_variant CreateFromPackedByteArray(godot_packed_byte_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_byte_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedInt32Array(godot_packed_int32_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_int32_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedInt64Array(godot_packed_int64_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_int64_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedFloat32Array(godot_packed_float32_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_float32_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedFloat64Array(godot_packed_float64_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_float64_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedStringArray(godot_packed_string_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_string_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedVector2Array(godot_packed_vector2_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_vector2_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedVector3Array(godot_packed_vector3_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_vector3_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromPackedColorArray(godot_packed_color_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_packed_color_array(&ret, from);
            return ret;
        }

        public static unsafe godot_variant CreateFromArray(godot_array* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_array(&ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe godot_variant CreateFromArray(godot_array from)
            => CreateFromArray(&from);

        public static unsafe godot_variant CreateFromDictionary(godot_dictionary* from)
        {
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_dictionary(&ret, from);
            return ret;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static unsafe godot_variant CreateFromDictionary(godot_dictionary from)
            => CreateFromDictionary(&from);

        public static unsafe godot_variant CreateFromStringName(ref godot_string_name arg1)
        {
            godot_variant ret;
            godot_string_name src = arg1;
            NativeFuncs.godotsharp_variant_new_string_name(&ret, &src);
            return ret;
        }

        public static unsafe godot_variant CreateFromNodePath(ref godot_node_path arg1)
        {
            godot_variant ret;
            godot_node_path src = arg1;
            NativeFuncs.godotsharp_variant_new_node_path(&ret, &src);
            return ret;
        }

        public static unsafe godot_variant CreateFromGodotObject(IntPtr from)
        {
            if (from == IntPtr.Zero)
                return new godot_variant();
            godot_variant ret;
            NativeFuncs.godotsharp_variant_new_object(&ret, from);
            return ret;
        }

        // We avoid the internal call if the stored type is the same we want.

        public static unsafe bool ConvertToBool(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Bool ? (*p_var)._data._bool : NativeFuncs.godotsharp_variant_as_bool(p_var);

        public static unsafe char ConvertToChar(godot_variant* p_var)
            => (char)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe sbyte ConvertToInt8(godot_variant* p_var)
            => (sbyte)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe Int16 ConvertToInt16(godot_variant* p_var)
            => (Int16)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe Int32 ConvertToInt32(godot_variant* p_var)
            => (Int32)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe Int64 ConvertToInt64(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var);

        public static unsafe byte ConvertToUInt8(godot_variant* p_var)
            => (byte)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe UInt16 ConvertToUInt16(godot_variant* p_var)
            => (UInt16)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe UInt32 ConvertToUInt32(godot_variant* p_var)
            => (UInt32)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe UInt64 ConvertToUInt64(godot_variant* p_var)
            => (UInt64)((*p_var)._type == Variant.Type.Int ? (*p_var)._data._int : NativeFuncs.godotsharp_variant_as_int(p_var));

        public static unsafe float ConvertToFloat32(godot_variant* p_var)
            => (float)((*p_var)._type == Variant.Type.Float ? (*p_var)._data._float : NativeFuncs.godotsharp_variant_as_float(p_var));

        public static unsafe double ConvertToFloat64(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Float ? (*p_var)._data._float : NativeFuncs.godotsharp_variant_as_float(p_var);

        public static unsafe Vector2 ConvertToVector2(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Vector2 ? (*p_var)._data._m_vector2 : NativeFuncs.godotsharp_variant_as_vector2(p_var);

        public static unsafe Vector2i ConvertToVector2i(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Vector2i ? (*p_var)._data._m_vector2i : NativeFuncs.godotsharp_variant_as_vector2i(p_var);

        public static unsafe Rect2 ConvertToRect2(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Rect2 ? (*p_var)._data._m_rect2 : NativeFuncs.godotsharp_variant_as_rect2(p_var);

        public static unsafe Rect2i ConvertToRect2i(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Rect2i ? (*p_var)._data._m_rect2i : NativeFuncs.godotsharp_variant_as_rect2i(p_var);

        public static unsafe Transform2D ConvertToTransform2D(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Transform2d ? *(*p_var)._data._transform2d : NativeFuncs.godotsharp_variant_as_transform2d(p_var);

        public static unsafe Vector3 ConvertToVector3(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Vector3 ? (*p_var)._data._m_vector3 : NativeFuncs.godotsharp_variant_as_vector3(p_var);

        public static unsafe Vector3i ConvertToVector3i(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Vector3i ? (*p_var)._data._m_vector3i : NativeFuncs.godotsharp_variant_as_vector3i(p_var);

        public static unsafe Basis ConvertToBasis(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Basis ? *(*p_var)._data._basis : NativeFuncs.godotsharp_variant_as_basis(p_var);

        public static unsafe Quaternion ConvertToQuaternion(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Quaternion ? (*p_var)._data._m_quaternion : NativeFuncs.godotsharp_variant_as_quaternion(p_var);

        public static unsafe Transform3D ConvertToTransform3D(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Transform3d ? *(*p_var)._data._transform3d : NativeFuncs.godotsharp_variant_as_transform3d(p_var);

        public static unsafe AABB ConvertToAABB(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Aabb ? *(*p_var)._data._aabb : NativeFuncs.godotsharp_variant_as_aabb(p_var);

        public static unsafe Color ConvertToColor(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Color ? (*p_var)._data._m_color : NativeFuncs.godotsharp_variant_as_color(p_var);

        public static unsafe Plane ConvertToPlane(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Plane ? (*p_var)._data._m_plane : NativeFuncs.godotsharp_variant_as_plane(p_var);

        public static unsafe IntPtr ConvertToGodotObject(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Object ? (*p_var)._data._m_obj_data.obj : IntPtr.Zero;

        public static unsafe RID ConvertToRID(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Rid ? (*p_var)._data._m_rid : NativeFuncs.godotsharp_variant_as_rid(p_var);

        public static unsafe godot_string_name ConvertToStringName(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.StringName ?
                NativeFuncs.godotsharp_string_name_new_copy(&(*p_var)._data._m_string_name) :
                NativeFuncs.godotsharp_variant_as_string_name(p_var);

        public static unsafe godot_node_path ConvertToNodePath(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.NodePath ?
                NativeFuncs.godotsharp_node_path_new_copy(&(*p_var)._data._m_node_path) :
                NativeFuncs.godotsharp_variant_as_node_path(p_var);

        public static unsafe godot_array ConvertToArray(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Array ?
                NativeFuncs.godotsharp_array_new_copy(&(*p_var)._data._m_array) :
                NativeFuncs.godotsharp_variant_as_array(p_var);

        public static unsafe godot_dictionary ConvertToDictionary(godot_variant* p_var)
            => (*p_var)._type == Variant.Type.Dictionary ?
                NativeFuncs.godotsharp_dictionary_new_copy(&(*p_var)._data._m_dictionary) :
                NativeFuncs.godotsharp_variant_as_dictionary(p_var);
    }
}
