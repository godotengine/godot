using System.Diagnostics.CodeAnalysis;

namespace Godot.SourceGenerators
{
    [SuppressMessage("ReSharper", "InconsistentNaming")]
    public enum MarshalType
    {
        Boolean,
        Char,
        SByte,
        Int16,
        Int32,
        Int64,
        Byte,
        UInt16,
        UInt32,
        UInt64,
        Single,
        Double,
        String,

        // Godot structs
        Vector2,
        Vector2i,
        Rect2,
        Rect2i,
        Transform2D,
        Vector3,
        Vector3i,
        Basis,
        Quaternion,
        Transform3D,
        Vector4,
        Vector4i,
        Projection,
        AABB,
        Color,
        Plane,
        Callable,
        Signal,

        // Enums
        Enum,

        // Arrays
        ByteArray,
        Int32Array,
        Int64Array,
        Float32Array,
        Float64Array,
        StringArray,
        Vector2Array,
        Vector3Array,
        ColorArray,
        GodotObjectOrDerivedArray,
        SystemArrayOfStringName,
        SystemArrayOfNodePath,
        SystemArrayOfRID,

        // Variant
        Variant,

        // Classes
        GodotObjectOrDerived,
        StringName,
        NodePath,
        RID,
        GodotDictionary,
        GodotArray,
        GodotGenericDictionary,
        GodotGenericArray,
    }
}
