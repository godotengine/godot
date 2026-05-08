// Mirrors bridge_helpers.h
using System;
using System.Text;
using System.Threading;

public static class Helpers
{

    // responsible for sending commands to the command offset
    public static void SendCommand(int cmd)
    {
        Interop.AtomicWriteInt32(Commands.CMD_OFFSET, cmd);
        Interop.AtomicWriteInt32(Commands.STATUS_OFFSET, Commands.STATUS_PENDING);
    }

    // used to reset the command offset to 0 and status to 0 at the start of the program
    public static void ResetCommandBuffer()
    {
        WriteByte(Commands.CMD_OFFSET, 0);
        //makes sure to reset the next byte too
        WriteByte(Commands.CMD_OFFSET + 1, 0);
        WriteByte(Commands.STATUS_OFFSET, 0);
    }

    // busy-wait loop
    public static void WaitForCompletion()
    {
        while (Interop.AtomicReadInt32(Commands.STATUS_OFFSET) != Commands.STATUS_DONE)
        {
            Thread.SpinWait(1);
        }
    }

    // Offsets are the addresses to which we push bytes or pull bytes out
    public static void WriteByte(int offset, byte value)
        => Interop.BulkWrite(new[] { value }, offset, 1);

    public static byte ReadByte(int offset)
    {
        var bytes = Interop.BulkRead(offset, 1);
        return bytes.Length > 0 ? bytes[0] : (byte)0;
    }

    // int32 ops
    public static void WriteInt32(int offset, int value)
        => Interop.BulkWrite(BitConverter.GetBytes(value), offset, 4);

    public static int ReadInt32(int offset)
    {
        var bytes = Interop.BulkRead(offset, 4);
        return bytes.Length >= 4 ? BitConverter.ToInt32(bytes, 0) : 0;
    }

    // int64 operations
    public static void WriteInt64(int offset, long value)
        => Interop.BulkWrite(BitConverter.GetBytes(value), offset, 8);

    public static long ReadInt64(int offset)
    {
        var bytes = Interop.BulkRead(offset, 8);
        return bytes.Length >= 8 ? BitConverter.ToInt64(bytes, 0) : 0;
    }

    public static void WriteUInt64(int offset, ulong value)
        => Interop.BulkWrite(BitConverter.GetBytes(value), offset, 8);

    public static ulong ReadUInt64(int offset)
    {
        var bytes = Interop.BulkRead(offset, 8);
        return bytes.Length >= 8 ? BitConverter.ToUInt64(bytes, 0) : 0;
    }

    public static void WriteFloat(int offset, float value)
        => Interop.BulkWrite(BitConverter.GetBytes(value), offset, 4);

    public static float ReadFloat(int offset)
    {
        var bytes = Interop.BulkRead(offset, 4);
        return bytes.Length >= 4 ? BitConverter.ToSingle(bytes, 0) : 0.0f;
    }

    public static void WriteString(int offset, string value, int maxBytes = 256)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value ?? "");
        if (bytes.Length > maxBytes)
        {
            Array.Resize(ref bytes, maxBytes);
        }

        WriteInt32(offset, bytes.Length);
        Interop.BulkWrite(bytes, offset + 4, bytes.Length);
    }



    // string helpers
    public static string ReadString(int offset)
    {
        int len = ReadInt32(offset);
        if (len <= 0 || len > 4096)
        {
            return string.Empty;
        }

        var bytes = Interop.BulkRead(offset + 4, len);
        return Encoding.UTF8.GetString(bytes);
    }

    public static void WriteStringName(int offset, string value) => WriteString(offset, value);
    public static string ReadStringName(int offset) => ReadString(offset);

    public static void WriteNodePath(int offset, string value) => WriteString(offset, value);
    public static string ReadNodePath(int offset) => ReadString(offset);

    // double ops
    public static void WriteDouble(int offset, double value)
        => Interop.BulkWrite(BitConverter.GetBytes(value), offset, 8);

    public static double ReadDouble(int offset)
    {
        var bytes = Interop.BulkRead(offset, 8);
        return BitConverter.ToDouble(bytes, 0);
    }

    // RID ops
    public static void WriteRID(int offset, ulong value) => WriteUInt64(offset, value);
    public static ulong ReadRID(int offset) => ReadUInt64(offset);

    // vector types ops
    public static void WriteVector2(int offset, Vector2 value)
    {
        WriteFloat(offset, value.X);
        WriteFloat(offset + 4, value.Y);
    }

    public static Vector2 ReadVector2(int offset)
    {
        float x = ReadFloat(offset);
        float y = ReadFloat(offset + 4);
        return new Vector2(x, y);
    }

    public static void WriteVector2i(int offset, Vector2i value)
    {
        WriteInt32(offset, value.X);
        WriteInt32(offset + 4, value.Y);
    }

    public static Vector2i ReadVector2i(int offset)
    {
        int x = ReadInt32(offset);
        int y = ReadInt32(offset + 4);
        return new Vector2i(x, y);
    }

    public static void WriteVector3(int offset, Vector3 value)
    {
        WriteFloat(offset, value.X);
        WriteFloat(offset + 4, value.Y);
        WriteFloat(offset + 8, value.Z);
    }

    public static Vector3 ReadVector3(int offset)
    {
        float x = ReadFloat(offset);
        float y = ReadFloat(offset + 4);
        float z = ReadFloat(offset + 8);
        return new Vector3(x, y, z);
    }

    public static void WriteVector3i(int offset, Vector3i value)
    {
        WriteInt32(offset, value.X);
        WriteInt32(offset + 4, value.Y);
        WriteInt32(offset + 8, value.Z);
    }

    public static Vector3i ReadVector3i(int offset)
    {
        int x = ReadInt32(offset);
        int y = ReadInt32(offset + 4);
        int z = ReadInt32(offset + 8);
        return new Vector3i(x, y, z);
    }

    public static void WriteVector4(int offset, Vector4 value)
    {
        WriteFloat(offset, value.X);
        WriteFloat(offset + 4, value.Y);
        WriteFloat(offset + 8, value.Z);
        WriteFloat(offset + 12, value.W);
    }

    public static Vector4 ReadVector4(int offset)
    {
        float x = ReadFloat(offset);
        float y = ReadFloat(offset + 4);
        float z = ReadFloat(offset + 8);
        float w = ReadFloat(offset + 12);
        return new Vector4(x, y, z, w);
    }

    public static void WriteVector4i(int offset, Vector4i value)
    {
        WriteInt32(offset, value.X);
        WriteInt32(offset + 4, value.Y);
        WriteInt32(offset + 8, value.Z);
        WriteInt32(offset + 12, value.W);
    }

    public static Vector4i ReadVector4i(int offset)
    {
        int x = ReadInt32(offset);
        int y = ReadInt32(offset + 4);
        int z = ReadInt32(offset + 8);
        int w = ReadInt32(offset + 12);
        return new Vector4i(x, y, z, w);
    }

    // color ops
    public static void WriteColor(int offset, Color value)
    {
        WriteFloat(offset, value.R);
        WriteFloat(offset + 4, value.G);
        WriteFloat(offset + 8, value.B);
        WriteFloat(offset + 12, value.A);
    }

    public static Color ReadColor(int offset)
    {
        float r = ReadFloat(offset);
        float g = ReadFloat(offset + 4);
        float b = ReadFloat(offset + 8);
        float a = ReadFloat(offset + 12);
        return new Color(r, g, b, a);
    }

    // rect ops
    public static void WriteRect2(int offset, Rect2 value)
    {
        WriteVector2(offset, value.Position);
        WriteVector2(offset + 8, value.Size);
    }

    public static Rect2 ReadRect2(int offset)
    {
        Vector2 pos = ReadVector2(offset);
        Vector2 size = ReadVector2(offset + 8);
        return new Rect2(pos, size);
    }

    public static void WriteRect2i(int offset, Rect2i value)
    {
        WriteVector2i(offset, value.Position);
        WriteVector2i(offset + 8, value.Size);
    }

    public static Rect2i ReadRect2i(int offset)
    {
        Vector2i pos = ReadVector2i(offset);
        Vector2i size = ReadVector2i(offset + 8);
        return new Rect2i(pos, size);
    }

    // Transform2D ops
    public static void WriteTransform2D(int offset, Transform2D value)
    {
        WriteVector2(offset, value.X);
        WriteVector2(offset + 8, value.Y);
        WriteVector2(offset + 16, value.Origin);
    }

    public static Transform2D ReadTransform2D(int offset)
    {
        Vector2 xAxis = ReadVector2(offset);
        Vector2 yAxis = ReadVector2(offset + 8);
        Vector2 origin = ReadVector2(offset + 16);
        return new Transform2D(xAxis, yAxis, origin);
    }

    // Transform3D ops
    public static void WriteTransform3D(int offset, Transform3D value)
    {
        WriteBasis(offset, value.Basis);
        WriteVector3(offset + 36, value.Origin);
    }

    public static Transform3D ReadTransform3D(int offset)
    {
        Basis basis = ReadBasis(offset);
        Vector3 origin = ReadVector3(offset + 36);
        return new Transform3D(basis, origin);
    }

    // basis ops
    public static void WriteBasis(int offset, Basis value)
    {
        WriteVector3(offset, value.Column0);
        WriteVector3(offset + 12, value.Column1);
        WriteVector3(offset + 24, value.Column2);
    }

    public static Basis ReadBasis(int offset)
    {
        Vector3 row0 = ReadVector3(offset);
        Vector3 row1 = ReadVector3(offset + 12);
        Vector3 row2 = ReadVector3(offset + 24);
        return new Basis(row0, row1, row2);
    }

    // Quaternion ops
    public static void WriteQuaternion(int offset, Quaternion value)
    {
        WriteFloat(offset, value.W);
        WriteFloat(offset + 4, value.X);
        WriteFloat(offset + 8, value.Y);
        WriteFloat(offset + 12, value.Z);
    }

    public static Quaternion ReadQuaternion(int offset)
    {
        float w = ReadFloat(offset);
        float x = ReadFloat(offset + 4);
        float y = ReadFloat(offset + 8);
        float z = ReadFloat(offset + 12);
        return new Quaternion(w, x, y, z);
    }

    // AABB ops
    public static void WriteAABB(int offset, AABB value)
    {
        WriteVector3(offset, value.Position);
        WriteVector3(offset + 12, value.Size);
    }

    public static AABB ReadAABB(int offset)
    {
        Vector3 pos = ReadVector3(offset);
        Vector3 size = ReadVector3(offset + 12);
        return new AABB(pos, size);
    }

    // plane ops
    public static void WritePlane(int offset, Plane value)
    {
        WriteFloat(offset, value.Normal.X);
        WriteFloat(offset + 4, value.Normal.Y);
        WriteFloat(offset + 8, value.Normal.Z);
        WriteFloat(offset + 12, value.D);
    }

    public static Plane ReadPlane(int offset)
    {
        float nx = ReadFloat(offset);
        float ny = ReadFloat(offset + 4);
        float nz = ReadFloat(offset + 8);
        float d = ReadFloat(offset + 12);
        return new Plane(new Vector3(nx, ny, nz), d);
    }

    // projection ops
    public static void WriteProjection(int offset, Projection value)
    {
        WriteVector4(offset, value.X);
        WriteVector4(offset + 16, value.Y);
        WriteVector4(offset + 32, value.Z);
        WriteVector4(offset + 48, value.W);
    }

    public static Projection ReadProjection(int offset)
    {
        Vector4 x = ReadVector4(offset);
        Vector4 y = ReadVector4(offset + 16);
        Vector4 z = ReadVector4(offset + 32);
        Vector4 w = ReadVector4(offset + 48);
        return new Projection(x, y, z, w);
    }

    // direct helpers for types that are not blob-based here
    public static object ReadCallable(int offset) => null;
    public static void WriteCallable(int offset, object value) { }

    public static object ReadSignal(int offset) => null;
    public static void WriteSignal(int offset, object value) { }

    public static object ReadDictionary(int offset) => null;
    public static void WriteDictionary(int offset, object value) { }

    public static object ReadArray(int offset) => null;
    public static void WriteArray(int offset, object value) { }

    // packed byte array
    public static void WritePackedByteArray(int offset, byte[] values)
    {
        values ??= Array.Empty<byte>();
        WriteInt32(offset, values.Length);

        if (values.Length > 0)
        {
            Interop.BulkWrite(values, offset + 4, values.Length);
        }
    }

    public static byte[] ReadPackedByteArray(int offset)
    {
        int len = ReadInt32(offset);
        if (len <= 0)
        {
            return Array.Empty<byte>();
        }

        return Interop.BulkRead(offset + 4, len);
    }

    // packed int32 array
    public static void WritePackedInt32Array(int offset, int[] values)
    {
        values ??= Array.Empty<int>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteInt32(offset + 4 + (i * 4), values[i]);
        }
    }

    public static int[] ReadPackedInt32Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<int>();
        }

        int[] result = new int[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadInt32(offset + 4 + (i * 4));
        }

        return result;
    }

    // packed int64 array
    public static void WritePackedInt64Array(int offset, long[] values)
    {
        values ??= Array.Empty<long>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteInt64(offset + 4 + (i * 8), values[i]);
        }
    }

    public static long[] ReadPackedInt64Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<long>();
        }

        long[] result = new long[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadInt64(offset + 4 + (i * 8));
        }

        return result;
    }

    // packed float32 array
    public static void WritePackedFloat32Array(int offset, float[] values)
    {
        values ??= Array.Empty<float>();
        WriteInt32(offset, values.Length);

        if (values.Length == 0)
        {
            return;
        }

        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        Interop.BulkWrite(bytes, offset + 4, bytes.Length);
    }

    public static float[] ReadPackedFloat32Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<float>();
        }

        byte[] bytes = Interop.BulkRead(offset + 4, count * sizeof(float));
        float[] result = new float[count];
        Buffer.BlockCopy(bytes, 0, result, 0, Math.Min(bytes.Length, count * sizeof(float)));
        return result;
    }

    // packed float64 array
    public static void WritePackedFloat64Array(int offset, double[] values)
    {
        values ??= Array.Empty<double>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteDouble(offset + 4 + (i * 8), values[i]);
        }
    }

    public static double[] ReadPackedFloat64Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<double>();
        }

        double[] result = new double[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadDouble(offset + 4 + (i * 8));
        }

        return result;
    }

    // packed string array
    public static void WritePackedStringArray(int offset, string[] values)
    {
        values ??= Array.Empty<string>();
        WriteInt32(offset, values.Length);

        int pos = offset + 4;
        for (int i = 0; i < values.Length; i++)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(values[i] ?? "");
            WriteInt32(pos, bytes.Length);
            pos += 4;

            if (bytes.Length > 0)
            {
                Interop.BulkWrite(bytes, pos, bytes.Length);
                pos += bytes.Length;
            }
        }
    }

    public static string[] ReadPackedStringArray(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<string>();
        }

        string[] result = new string[count];
        int pos = offset + 4;

        for (int i = 0; i < count; i++)
        {
            int len = ReadInt32(pos);
            pos += 4;

            if (len <= 0)
            {
                result[i] = string.Empty;
                continue;
            }

            byte[] bytes = Interop.BulkRead(pos, len);
            result[i] = Encoding.UTF8.GetString(bytes);
            pos += len;
        }

        return result;
    }

    // packed Vector2 array
    public static void WritePackedVector2Array(int offset, Vector2[] values)
    {
        values ??= Array.Empty<Vector2>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteVector2(offset + 4 + (i * 8), values[i]);
        }
    }

    public static Vector2[] ReadPackedVector2Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<Vector2>();
        }

        Vector2[] result = new Vector2[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadVector2(offset + 4 + (i * 8));
        }

        return result;
    }

    // packed Vector3 array
    public static void WritePackedVector3Array(int offset, Vector3[] values)
    {
        values ??= Array.Empty<Vector3>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteVector3(offset + 4 + (i * 12), values[i]);
        }
    }

    public static Vector3[] ReadPackedVector3Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<Vector3>();
        }

        Vector3[] result = new Vector3[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadVector3(offset + 4 + (i * 12));
        }

        return result;
    }

    // packed Color array
    public static void WritePackedColorArray(int offset, Color[] values)
    {
        values ??= Array.Empty<Color>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteColor(offset + 4 + (i * 16), values[i]);
        }
    }

    public static Color[] ReadPackedColorArray(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<Color>();
        }

        Color[] result = new Color[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadColor(offset + 4 + (i * 16));
        }

        return result;
    }

    // packed Vector4 array
    public static void WritePackedVector4Array(int offset, Vector4[] values)
    {
        values ??= Array.Empty<Vector4>();
        WriteInt32(offset, values.Length);

        for (int i = 0; i < values.Length; i++)
        {
            WriteVector4(offset + 4 + (i * 16), values[i]);
        }
    }

    public static Vector4[] ReadPackedVector4Array(int offset)
    {
        int count = ReadInt32(offset);
        if (count <= 0)
        {
            return Array.Empty<Vector4>();
        }

        Vector4[] result = new Vector4[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = ReadVector4(offset + 4 + (i * 16));
        }

        return result;
    }

    // compatibility alias
    public static void WriteFloatArray(int offset, float[] values)
        => WritePackedFloat32Array(offset, values);

        // Variant ops
        public static void WriteVariant(int offset, object value)
        {
            VariantHandling.Encode(offset, value);
        }

        public static object ReadVariant(int offset)
        {
            return VariantHandling.Decode(offset);
        }
}
