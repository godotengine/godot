/*
This is specifically designed to handle Variant serialization/deserialization
*/
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

public struct CustomCallable
{
    public ulong TargetId;
    public string Method;
    public CustomCallable(ulong id, string m) { TargetId = id; Method = m; }
}

public struct CustomSignal
{
    public ulong TargetId;
    public string Name;
    public CustomSignal(ulong id, string n) { TargetId = id; Name = n; }
}

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

    // Flag indicating 64-bit encoding
    private const int ENCODE_FLAG_64 = 1 << 16;

    public static int Encode(int offset, object value)
    {
        // This is the offset where the data starts
        int dataOffset = offset + 4;

        if (value == null)
        {
            Helpers.WriteInt32(offset, (int)VariantType.Nil);
            return 4;
        }

        // Notes the number of bytes written per case
        int bytesWritten;

        switch (value)
        {
            case bool b:
                Helpers.WriteInt32(offset, (int)VariantType.Bool);
                Helpers.WriteInt32(dataOffset, b ? 1 : 0);
                bytesWritten = 8;
                break;

            case int i:
                Helpers.WriteInt32(offset, (int)VariantType.Int | ENCODE_FLAG_64);
                Helpers.WriteInt64(dataOffset, i);
                bytesWritten = 12;
                break;

            case long l:
                Helpers.WriteInt32(offset, (int)VariantType.Int | ENCODE_FLAG_64);
                Helpers.WriteInt64(dataOffset, l);
                bytesWritten = 12;
                break;

            case float f:
                Helpers.WriteInt32(offset, (int)VariantType.Float | ENCODE_FLAG_64);
                Helpers.WriteDouble(dataOffset, f);
                bytesWritten = 12;
                break;

            case double d:
                Helpers.WriteInt32(offset, (int)VariantType.Float | ENCODE_FLAG_64);
                Helpers.WriteDouble(dataOffset, d);
                bytesWritten = 12;
                break;

            case string s:
                {
                    // Its Variable so we have to get the current number of bytes of the string in question
                    byte[] utf8 = Encoding.UTF8.GetBytes(s ?? "");
                    int byteLen = utf8.Length;
                    // Padding to 4‑byte boundary (after the 4‑byte length field) - this specifically helps with Dict and Array Problems
                    int paddedLen = (byteLen + 3) & ~3;

                    // Type
                    Helpers.WriteInt32(offset, (int)VariantType.String);
                    // Length
                    Helpers.WriteInt32(dataOffset, byteLen);
                    // Actual String Data
                    Interop.BulkWrite(utf8, dataOffset + 4, byteLen);
                    // Padding zeros
                    for (int p = byteLen; p < paddedLen; p++)
                        Helpers.WriteByte(dataOffset + 4 + p, 0);

                    bytesWritten = 8 + paddedLen;   // header(4) + length(4) + padded data
                    break;
                }

            case Vector2 v2:
                Helpers.WriteInt32(offset, (int)VariantType.Vector2);
                Helpers.WriteVector2(dataOffset, v2);
                bytesWritten = 12;
                break;

            case Vector2i v2i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector2i);
                Helpers.WriteVector2i(dataOffset, v2i);
                bytesWritten = 12;
                break;

            case Rect2 r2:
                Helpers.WriteInt32(offset, (int)VariantType.Rect2);
                Helpers.WriteRect2(dataOffset, r2);
                bytesWritten = 20;
                break;

            case Rect2i r2i:
                Helpers.WriteInt32(offset, (int)VariantType.Rect2i);
                Helpers.WriteRect2i(dataOffset, r2i);
                bytesWritten = 20;
                break;

            case Vector3 v3:
                Helpers.WriteInt32(offset, (int)VariantType.Vector3);
                Helpers.WriteVector3(dataOffset, v3);
                bytesWritten = 16;
                break;

            case Vector3i v3i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector3i);
                Helpers.WriteVector3i(dataOffset, v3i);
                bytesWritten = 16;
                break;

            case Transform2D t2d:
                Helpers.WriteInt32(offset, (int)VariantType.Transform2D);
                Helpers.WriteTransform2D(dataOffset, t2d);
                bytesWritten = 28;
                break;

            case Vector4 v4:
                Helpers.WriteInt32(offset, (int)VariantType.Vector4);
                Helpers.WriteVector4(dataOffset, v4);
                bytesWritten = 20;
                break;

            case Vector4i v4i:
                Helpers.WriteInt32(offset, (int)VariantType.Vector4i);
                Helpers.WriteVector4i(dataOffset, v4i);
                bytesWritten = 20;
                break;

            case Plane plane:
                Helpers.WriteInt32(offset, (int)VariantType.Plane);
                Helpers.WritePlane(dataOffset, plane);
                bytesWritten = 20;
                break;

            case Quaternion q:
                Helpers.WriteInt32(offset, (int)VariantType.Quaternion);
                Helpers.WriteQuaternion(dataOffset, q);
                bytesWritten = 20;
                break;

            case AABB aabb:
                Helpers.WriteInt32(offset, (int)VariantType.Aabb);
                Helpers.WriteAABB(dataOffset, aabb);
                bytesWritten = 28;
                break;

            case Basis basis:
                Helpers.WriteInt32(offset, (int)VariantType.Basis);
                Helpers.WriteBasis(dataOffset, basis);
                bytesWritten = 40;
                break;

            case Transform3D t3d:
                Helpers.WriteInt32(offset, (int)VariantType.Transform3D);
                Helpers.WriteTransform3D(dataOffset, t3d);
                bytesWritten = 52;
                break;

            case Projection proj:
                Helpers.WriteInt32(offset, (int)VariantType.Projection);
                Helpers.WriteProjection(dataOffset, proj);
                bytesWritten = 68;
                break;

            case Color color:
                Helpers.WriteInt32(offset, (int)VariantType.Color);
                Helpers.WriteColor(dataOffset, color);
                bytesWritten = 20;
                break;

            case ulong rid:
                Helpers.WriteInt32(offset, (int)VariantType.Rid);
                Helpers.WriteRID(dataOffset, rid);
                bytesWritten = 12;
                break;

            case byte[] pba:
                Helpers.WriteInt32(offset, (int)VariantType.PackedByteArray);
                Helpers.WritePackedByteArray(dataOffset, pba);
                bytesWritten = 8 + (pba?.Length ?? 0);
                break;

            case int[] p32:
                Helpers.WriteInt32(offset, (int)VariantType.PackedInt32Array);
                Helpers.WritePackedInt32Array(dataOffset, p32);
                bytesWritten = 8 + ((p32?.Length ?? 0) * 4);
                break;

            case long[] p64:
                Helpers.WriteInt32(offset, (int)VariantType.PackedInt64Array);
                Helpers.WritePackedInt64Array(dataOffset, p64);
                bytesWritten = 8 + ((p64?.Length ?? 0) * 8);
                break;

            case float[] pf32:
                Helpers.WriteInt32(offset, (int)VariantType.PackedFloat32Array);
                Helpers.WritePackedFloat32Array(dataOffset, pf32);
                bytesWritten = 8 + ((pf32?.Length ?? 0) * 4);
                break;

            case double[] pf64:
                Helpers.WriteInt32(offset, (int)VariantType.PackedFloat64Array);
                Helpers.WritePackedFloat64Array(dataOffset, pf64);
                bytesWritten = 8 + ((pf64?.Length ?? 0) * 8);
                break;

            case string[] ps:
                Helpers.WriteInt32(offset, (int)VariantType.PackedStringArray);
                Helpers.WritePackedStringArray(dataOffset, ps);
                bytesWritten = 8;
                if (ps != null)
                {
                    for (int i = 0; i < ps.Length; i++)
                    {
                        bytesWritten += 4 + Encoding.UTF8.GetByteCount(ps[i] ?? "");
                    }
                }
                break;

            case Vector2[] pv2:
                Helpers.WriteInt32(offset, (int)VariantType.PackedVector2Array);
                Helpers.WritePackedVector2Array(dataOffset, pv2);
                bytesWritten = 8 + ((pv2?.Length ?? 0) * 8);
                break;

            case Vector3[] pv3:
                Helpers.WriteInt32(offset, (int)VariantType.PackedVector3Array);
                Helpers.WritePackedVector3Array(dataOffset, pv3);
                bytesWritten = 8 + ((pv3?.Length ?? 0) * 12);
                break;

            case Color[] pc:
                Helpers.WriteInt32(offset, (int)VariantType.PackedColorArray);
                Helpers.WritePackedColorArray(dataOffset, pc);
                bytesWritten = 8 + ((pc?.Length ?? 0) * 16);
                break;

            case IDictionary dict:
                {
                    Helpers.WriteInt32(offset, (int)VariantType.Dictionary);

                    int pos = dataOffset;
                    Helpers.WriteInt32(pos, dict.Count);
                    pos += 4;

                    foreach (DictionaryEntry entry in dict)
                    {
                        pos += Encode(pos, entry.Key);
                        pos += Encode(pos, entry.Value);
                    }

                    bytesWritten = pos - offset;
                    break;
                }

            case System.Array arr:
                {
                    Helpers.WriteInt32(offset, (int)VariantType.Array);

                    int pos = dataOffset;
                    Helpers.WriteInt32(pos, arr.Length);
                    pos += 4;

                    foreach (var item in arr)
                    {
                        pos += Encode(pos, item);
                    }

                    bytesWritten = pos - offset;
                    break;
                }

            case CustomCallable c:
            {
                Helpers.WriteInt32(offset, (int)VariantType.Callable);
                var cBuf = new VariantBuffer(dataOffset);
                cBuf.WriteCallable(c);
                bytesWritten = cBuf.Cursor - offset;
                break;
            }

            case CustomSignal s:
            {
                Helpers.WriteInt32(offset, (int)VariantType.Signal);
                var sBuf = new VariantBuffer(dataOffset);
                sBuf.WriteSignal(s);
                bytesWritten = sBuf.Cursor - offset;
                break;
            }

            default:
                throw new NotSupportedException($"Type {value.GetType()} not implemented for Godot 4 Variant.");
        }

        return bytesWritten;
    }

    public static object Decode(int offset) => DecodeInternal(offset, out _);

    public static object DecodeInternal(int offset, out int bytesRead)
    {
        int header = Helpers.ReadInt32(offset);
        int typeId = header & 0xFFFF;
        bool is64 = (header & ENCODE_FLAG_64) != 0;
        int dataOffset = offset + 4;

        switch ((VariantType)typeId)
        {
            case VariantType.Nil:
                bytesRead = 4;
                return null;

            case VariantType.Bool:
                bytesRead = 8;
                return Helpers.ReadInt32(dataOffset) != 0;

            case VariantType.Int:
                bytesRead = is64 ? 12 : 8;
                return is64 ? Helpers.ReadInt64(dataOffset) : Helpers.ReadInt32(dataOffset);

            case VariantType.Float:
                bytesRead = is64 ? 12 : 8;
                return is64 ? Helpers.ReadDouble(dataOffset) : Helpers.ReadFloat(dataOffset);

            case VariantType.String:
            {
                int len = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, len);
                return Helpers.ReadString(dataOffset);
            }

            case VariantType.Vector2:
                bytesRead = 12;
                return Helpers.ReadVector2(dataOffset);

            case VariantType.Vector2i:
                bytesRead = 12;
                return Helpers.ReadVector2i(dataOffset);

            case VariantType.Rect2:
                bytesRead = 20;
                return Helpers.ReadRect2(dataOffset);

            case VariantType.Rect2i:
                bytesRead = 20;
                return Helpers.ReadRect2i(dataOffset);

            case VariantType.Vector3:
                bytesRead = 16;
                return Helpers.ReadVector3(dataOffset);

            case VariantType.Vector3i:
                bytesRead = 16;
                return Helpers.ReadVector3i(dataOffset);

            case VariantType.Transform2D:
                bytesRead = 28;
                return Helpers.ReadTransform2D(dataOffset);

            case VariantType.Vector4:
                bytesRead = 20;
                return Helpers.ReadVector4(dataOffset);

            case VariantType.Vector4i:
                bytesRead = 20;
                return Helpers.ReadVector4i(dataOffset);

            case VariantType.Plane:
                bytesRead = 20;
                return Helpers.ReadPlane(dataOffset);

            case VariantType.Quaternion:
                bytesRead = 20;
                return Helpers.ReadQuaternion(dataOffset);

            case VariantType.Aabb:
                bytesRead = 28;
                return Helpers.ReadAABB(dataOffset);

            case VariantType.Basis:
                bytesRead = 40;
                return Helpers.ReadBasis(dataOffset);

            case VariantType.Transform3D:
                bytesRead = 52;
                return Helpers.ReadTransform3D(dataOffset);

            case VariantType.Projection:
                bytesRead = 68;
                return Helpers.ReadProjection(dataOffset);

            case VariantType.Color:
                bytesRead = 20;
                return Helpers.ReadColor(dataOffset);

            case VariantType.StringName:
            {
                int len = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, len);
                return Helpers.ReadStringName(dataOffset);
            }

            case VariantType.NodePath:
            {
                int len = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, len);
                return Helpers.ReadNodePath(dataOffset);
            }

            case VariantType.Rid:
                bytesRead = 12;
                return Helpers.ReadRID(dataOffset);

            case VariantType.PackedByteArray:
            {
                int len = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, len);
                return Helpers.ReadPackedByteArray(dataOffset);
            }

            case VariantType.PackedInt32Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 4;
                return Helpers.ReadPackedInt32Array(dataOffset);
            }

            case VariantType.PackedInt64Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 8;
                return Helpers.ReadPackedInt64Array(dataOffset);
            }

            case VariantType.PackedFloat32Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 4;
                return Helpers.ReadPackedFloat32Array(dataOffset);
            }

            case VariantType.PackedFloat64Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 8;
                return Helpers.ReadPackedFloat64Array(dataOffset);
            }

            case VariantType.PackedStringArray:
            {
                string[] arr = Helpers.ReadPackedStringArray(dataOffset);
                int size = 8;
                for (int i = 0; i < arr.Length; i++)
                {
                    size += 4 + Encoding.UTF8.GetByteCount(arr[i] ?? "");
                }
                bytesRead = size;
                return arr;
            }

            case VariantType.PackedVector2Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 8;
                return Helpers.ReadPackedVector2Array(dataOffset);
            }

            case VariantType.PackedVector3Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 12;
                return Helpers.ReadPackedVector3Array(dataOffset);
            }

            case VariantType.PackedColorArray:
            {
                int count = Helpers.ReadInt32(dataOffset);
                bytesRead = 8 + Math.Max(0, count) * 16;
                return Helpers.ReadPackedColorArray(dataOffset);
            }

            case VariantType.Dictionary:
            {
                int count = Helpers.ReadInt32(dataOffset);
                int pos = dataOffset + 4;
                var dict = new Dictionary<object, object>(count);

                for (int i = 0; i < count; i++)
                {
                    object key = DecodeInternal(pos, out int keyBytes);
                    pos += keyBytes;

                    object val = DecodeInternal(pos, out int valBytes);
                    pos += valBytes;

                    dict[key] = val;
                }

                bytesRead = pos - offset;
                return dict;
            }

            case VariantType.Array:
            {
                int count = Helpers.ReadInt32(dataOffset);
                int pos = dataOffset + 4;
                object[] arr = new object[count];

                for (int i = 0; i < count; i++)
                {
                    arr[i] = DecodeInternal(pos, out int itemBytes);
                    pos += itemBytes;
                }

                bytesRead = pos - offset;
                return arr;
            }

            case VariantType.Callable:
            {
                ulong id = Helpers.ReadUInt64(dataOffset);
                string method = Helpers.ReadString(dataOffset + 8);
                int len = Helpers.ReadInt32(dataOffset + 8);
                bytesRead = 16 + Math.Max(0, len);
                return new CustomCallable(id, method);
            }

            case VariantType.Signal:
            {
                ulong id = Helpers.ReadUInt64(dataOffset);
                string name = Helpers.ReadString(dataOffset + 8);
                int len = Helpers.ReadInt32(dataOffset + 8);
                bytesRead = 16 + Math.Max(0, len);
                return new CustomSignal(id, name);
            }

            default:
                bytesRead = 4;
                return null;
        }
    }
}
