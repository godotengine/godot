/*
This is specifically designed to handle Variant serialization/deserialization
*/
using System;
using System.Text;

public static class VariantHandling
{
    // Godot Variant Type IDs
    public enum VariantType : int
    {
        Nil = 0,
        Bool = 1,
        Int = 2,
        Float = 3,
        String = 4,
        Vector2 = 5,
        Vector2i = 6,
        Rect2 = 7,
        Rect2i = 8,
        Vector3 = 9,
        Vector3i = 10,
        Transform2D = 11,
        Vector4 = 12,
        Vector4i = 13,
        Plane = 14,
        Quaternion = 15,
        Aabb = 16,
        Basis = 17,
        Transform3D = 18,
        Projection = 19,
        Color = 20,
        StringName = 21,
        NodePath = 22,
        Rid = 23,
        Object = 24,
        Callable = 25,
        Signal = 26,
        Dictionary = 27,
        Array = 28,
        PackedByteArray = 29,
        PackedInt32Array = 30,
        PackedInt64Array = 31,
        PackedFloat32Array = 32,
        PackedFloat64Array = 33,
        PackedStringArray = 34,
        PackedVector2Array = 35,
        PackedVector3Array = 36,
        PackedColorArray = 37
    }

    // Flag indicating 64-bit encoding (Required for Int/Float in Godot 4)
    private const int ENCODE_FLAG_64 = 1 << 16;

    public static void Encode(int offset, object value)
    {
        int dataOffset = offset + 4; //the data is written after the type header
    
        if (value == null)
        {
            Helpers.WriteInt32(offset, (int)VariantType.Nil);
            return;
        }
    
        switch (value)
        {
            case bool b:
                Helpers.WriteInt32(offset, (int)VariantType.Bool);
                Helpers.WriteInt32(dataOffset, b ? 1 : 0);
                break;
    
            case int i:
                Helpers.WriteInt32(offset, (int)VariantType.Int | ENCODE_FLAG_64);
                Helpers.WriteInt64(dataOffset, (long)i);
                break;
    
            case long l:
                Helpers.WriteInt32(offset, (int)VariantType.Int | ENCODE_FLAG_64);
                Helpers.WriteInt64(dataOffset, l);
                break;
    
            case float f:
                Helpers.WriteInt32(offset, (int)VariantType.Float | ENCODE_FLAG_64);
                Helpers.WriteDouble(dataOffset, (double)f);
                break;
    
            case double d:
                Helpers.WriteInt32(offset, (int)VariantType.Float | ENCODE_FLAG_64);
                Helpers.WriteDouble(dataOffset, d);
                break;
    
            case string s:
                Helpers.WriteInt32(offset, (int)VariantType.String);
                Helpers.WriteString(dataOffset, s);
                break;
    
            case Vector2 v2:
                Helpers.WriteInt32(offset, (int)VariantType.Vector2);
                Helpers.WriteVector2(dataOffset, v2);
                break;
    
            case Vector2i v2i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector2i);
                Helpers.WriteVector2i(dataOffset, v2i);
                break;
    
            case Rect2 r2:
                Helpers.WriteInt32(offset, (int)VariantType.Rect2);
                Helpers.WriteRect2(dataOffset, r2);
                break;
    
            case Rect2i r2i:
                Helpers.WriteInt32(offset, (int)VariantType.Rect2i);
                Helpers.WriteRect2i(dataOffset, r2i);
                break;
    
            case Vector3 v3:
                Helpers.WriteInt32(offset, (int)VariantType.Vector3);
                Helpers.WriteVector3(dataOffset, v3);
                break;
    
            case Vector3i v3i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector3i);
                Helpers.WriteVector3i(dataOffset, v3i);
                break;
    
            case Transform2D t2d:
                Helpers.WriteInt32(offset, (int)VariantType.Transform2D);
                Helpers.WriteTransform2D(dataOffset, t2d);
                break;
    
            case Vector4 v4:
                Helpers.WriteInt32(offset, (int)VariantType.Vector4);
                Helpers.WriteVector4(dataOffset, v4);
                break;
    
            case Vector4i v4i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector4i);
                Helpers.WriteVector4i(dataOffset, v4i);
                break;
    
            case Plane plane:
                Helpers.WriteInt32(offset, (int)VariantType.Plane);
                Helpers.WritePlane(dataOffset, plane);
                break;
    
            case Quaternion q:
                Helpers.WriteInt32(offset, (int)VariantType.Quaternion);
                Helpers.WriteQuaternion(dataOffset, q);
                break;
    
            case AABB aabb:
                Helpers.WriteInt32(offset, (int)VariantType.Aabb);
                Helpers.WriteAABB(dataOffset, aabb);
                break;
    
            case Basis basis:
                Helpers.WriteInt32(offset, (int)VariantType.Basis);
                Helpers.WriteBasis(dataOffset, basis);
                break;
    
            case Transform3D t3d:
                Helpers.WriteInt32(offset, (int)VariantType.Transform3D);
                Helpers.WriteTransform3D(dataOffset, t3d);
                break;
    
            case Projection proj:
                Helpers.WriteInt32(offset, (int)VariantType.Projection);
                Helpers.WriteProjection(dataOffset, proj);
                break;
    
            case Color color:
                Helpers.WriteInt32(offset, (int)VariantType.Color);
                Helpers.WriteColor(dataOffset, color);
                break;
    
            case ulong rid:
                Helpers.WriteInt32(offset, (int)VariantType.Rid);
                Helpers.WriteRID(dataOffset, rid);
                break;
    
            case byte[] pba:
                Helpers.WriteInt32(offset, (int)VariantType.PackedByteArray);
                Helpers.WritePackedByteArray(dataOffset, pba);
                break;
    
            case int[] p32:
                Helpers.WriteInt32(offset, (int)VariantType.PackedInt32Array);
                Helpers.WritePackedInt32Array(dataOffset, p32);
                break;
    
            case long[] p64:
                Helpers.WriteInt32(offset, (int)VariantType.PackedInt64Array);
                Helpers.WritePackedInt64Array(dataOffset, p64);
                break;
    
            case float[] pf32:
                Helpers.WriteInt32(offset, (int)VariantType.PackedFloat32Array);
                Helpers.WritePackedFloat32Array(dataOffset, pf32);
                break;
    
            case double[] pf64:
                Helpers.WriteInt32(offset, (int)VariantType.PackedFloat64Array);
                Helpers.WritePackedFloat64Array(dataOffset, pf64);
                break;
    
            case string[] ps:
                Helpers.WriteInt32(offset, (int)VariantType.PackedStringArray);
                Helpers.WritePackedStringArray(dataOffset, ps);
                break;
    
            case Vector2[] pv2:
                Helpers.WriteInt32(offset, (int)VariantType.PackedVector2Array);
                Helpers.WritePackedVector2Array(dataOffset, pv2);
                break;
    
            case Vector3[] pv3:
                Helpers.WriteInt32(offset, (int)VariantType.PackedVector3Array);
                Helpers.WritePackedVector3Array(dataOffset, pv3);
                break;
    
            case Color[] pc:
                Helpers.WriteInt32(offset, (int)VariantType.PackedColorArray);
                Helpers.WritePackedColorArray(dataOffset, pc);
                break;
    
            default:
                throw new NotSupportedException($"Type {value.GetType()} not implemented for Godot 4 Variant.");
        }
    }
    
    public static object Decode(int offset)
    {
        int header = Helpers.ReadInt32(offset);
        int typeId = header & 0xFFFF;
        bool is64 = (header & ENCODE_FLAG_64) != 0;
        int dataOffset = offset + 4;
    
        return (VariantType)typeId switch
        {
            VariantType.Nil => null,
            
            VariantType.Bool => Helpers.ReadInt32(dataOffset) != 0,
            
            VariantType.Int => is64 ? Helpers.ReadInt64(dataOffset) : Helpers.ReadInt32(dataOffset),
            
            VariantType.Float => is64 ? Helpers.ReadDouble(dataOffset) : Helpers.ReadFloat(dataOffset),
            
            VariantType.String => Helpers.ReadString(dataOffset),
    
            VariantType.Vector2 => Helpers.ReadVector2(dataOffset),
            
            VariantType.Vector2i => Helpers.ReadVector2i(dataOffset),
            
            VariantType.Rect2 => Helpers.ReadRect2(dataOffset),
            
            VariantType.Rect2i => Helpers.ReadRect2i(dataOffset),
            
            VariantType.Vector3 => Helpers.ReadVector3(dataOffset),
            
            VariantType.Vector3i => Helpers.ReadVector3i(dataOffset),
            
            VariantType.Transform2D => Helpers.ReadTransform2D(dataOffset),
            
            VariantType.Vector4 => Helpers.ReadVector4(dataOffset),
            
            VariantType.Vector4i => Helpers.ReadVector4i(dataOffset),
            
            VariantType.Plane => Helpers.ReadPlane(dataOffset),
            
            VariantType.Quaternion => Helpers.ReadQuaternion(dataOffset),
            
            VariantType.Aabb => Helpers.ReadAABB(dataOffset),
            
            VariantType.Basis => Helpers.ReadBasis(dataOffset),
            
            VariantType.Transform3D => Helpers.ReadTransform3D(dataOffset),
            
            VariantType.Projection => Helpers.ReadProjection(dataOffset),
            
            VariantType.Color => Helpers.ReadColor(dataOffset),
    
            VariantType.StringName => Helpers.ReadStringName(dataOffset),
            
            VariantType.NodePath => Helpers.ReadNodePath(dataOffset),
            
            VariantType.Rid => Helpers.ReadRID(dataOffset),
    
            VariantType.Dictionary => Helpers.ReadDictionary(dataOffset),
            
            VariantType.Array => Helpers.ReadArray(dataOffset),
    
            VariantType.PackedByteArray => Helpers.ReadPackedByteArray(dataOffset),
            
            VariantType.PackedInt32Array => Helpers.ReadPackedInt32Array(dataOffset),
            
            VariantType.PackedInt64Array => Helpers.ReadPackedInt64Array(dataOffset),
            
            VariantType.PackedFloat32Array => Helpers.ReadPackedFloat32Array(dataOffset),
            
            VariantType.PackedFloat64Array => Helpers.ReadPackedFloat64Array(dataOffset),
            
            VariantType.PackedStringArray => Helpers.ReadPackedStringArray(dataOffset),
            
            VariantType.PackedVector2Array => Helpers.ReadPackedVector2Array(dataOffset),
            
            VariantType.PackedVector3Array => Helpers.ReadPackedVector3Array(dataOffset),
            
            VariantType.PackedColorArray => Helpers.ReadPackedColorArray(dataOffset),
    
            _ => null
        };
    }
}
