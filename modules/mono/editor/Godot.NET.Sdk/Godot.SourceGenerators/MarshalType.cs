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
        Vector2I,
        Rect2,
        Rect2I,
        Transform2D,
        Vector3,
        Vector3I,
        Basis,
        Quaternion,
        Transform3D,
        Vector4,
        Vector4I,
        Projection,
        Aabb,
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
        SystemArrayOfRid,

        // Variant
        Variant,

        // Classes
        GodotObjectOrDerived,
        StringName,
        NodePath,
        Rid,
        GodotDictionary,
        GodotArray,
        GodotGenericDictionary,
        GodotGenericArray,
    }
}
