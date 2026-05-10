using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

public static class Tests
{
    // Offsets – must match C++ test exactly
    private const int OFF_INT32            = 0;
    private const int OFF_INT64            = 4;
    private const int OFF_FLOAT            = 12;
    private const int OFF_DOUBLE           = 16;
    private const int OFF_BOOL             = 24;
    private const int OFF_STRING           = 28;
    private const int OFF_STRINGNAME       = 48;
    private const int OFF_NODEPATH         = 60;
    private const int OFF_RID              = 76;
    private const int OFF_VECTOR2          = 84;
    private const int OFF_VECTOR2I         = 92;
    private const int OFF_RECT2            = 100;
    private const int OFF_RECT2I           = 116;
    private const int OFF_VECTOR3          = 132;
    private const int OFF_VECTOR3I         = 144;
    private const int OFF_VECTOR4          = 156;
    private const int OFF_VECTOR4I         = 172;
    private const int OFF_PLANE            = 188;
    private const int OFF_QUATERNION       = 204;
    private const int OFF_AABB             = 220;
    private const int OFF_BASIS            = 244;
    private const int OFF_TRANSFORM3D      = 280;
    private const int OFF_PROJECTION       = 328;
    private const int OFF_COLOR            = 392;
    private const int OFF_PACKED_BYTE      = 408;
    private const int OFF_PACKED_INT32     = 415;
    private const int OFF_PACKED_INT64     = 431;
    private const int OFF_PACKED_FLOAT32   = 451;
    private const int OFF_PACKED_FLOAT64   = 463;
    private const int OFF_PACKED_STRING    = 483;
    private const int OFF_PACKED_VECTOR2   = 501;
    private const int OFF_PACKED_VECTOR3   = 521;
    private const int OFF_PACKED_COLOR     = 537;
    private const int OFF_DICTIONARY       = 600;
    private const int OFF_ARRAY            = 720;//moved from 700 to 720 to avoid overwriting dictionary. This was interfering with Godot's encode/decode

    public const int CMD_OFFSET            = 1_000_000;
    public const int STATUS_OFFSET         = 1_000_004;
    public const int RESULT_OFFSET         = 1_000_008;
    public const int STATUS_PENDING        = 1;
    public const int STATUS_DONE           = 2;
    public const int CMD_RUN_VARIANT_TESTS = 1;

    public static async Task Run()
    {
        Helpers.ResetCommandBuffer();
        WriteAllData();
        Helpers.SendCommand(CMD_RUN_VARIANT_TESTS);
        Helpers.WaitForCompletion();
        byte result = Helpers.ReadByte(RESULT_OFFSET);
        Console.WriteLine(result == 1 ? "bridge_test: PASSED" : "bridge_test: FAILED");
    }

    private static void WriteAllData()
    {
        // Scalars
        Helpers.WriteInt32(OFF_INT32, 42);
        Helpers.WriteInt64(OFF_INT64, 12345678901234L);
        Helpers.WriteFloat(OFF_FLOAT, 3.14f);
        Helpers.WriteDouble(OFF_DOUBLE, 2.718281828);
        Helpers.WriteInt32(OFF_BOOL, 1);   // true

        // Strings
        Helpers.WriteString(OFF_STRING, "Hello, WASM!");
        Helpers.WriteString(OFF_STRINGNAME, "TestName");
        Helpers.WriteString(OFF_NODEPATH, "/root/Player");

        // RID (as ulong)
        Helpers.WriteUInt64(OFF_RID, 0xDEADBEEF);

        // Vectors & math types
        Helpers.WriteVector2(OFF_VECTOR2, new Vector2(1.5f, 2.5f));
        Helpers.WriteVector2i(OFF_VECTOR2I, new Vector2i(10, -20));
        Helpers.WriteRect2(OFF_RECT2, new Rect2(new Vector2(1, 2), new Vector2(3, 4)));
        Helpers.WriteRect2i(OFF_RECT2I, new Rect2i(new Vector2i(5, 6), new Vector2i(7, 8)));
        Helpers.WriteVector3(OFF_VECTOR3, new Vector3(1, 2, 3));
        Helpers.WriteVector3i(OFF_VECTOR3I, new Vector3i(4, 5, 6));
        Helpers.WriteVector4(OFF_VECTOR4, new Vector4(1, 2, 3, 4));
        Helpers.WriteVector4i(OFF_VECTOR4I, new Vector4i(5, 6, 7, 8));
        Helpers.WritePlane(OFF_PLANE, new Plane(new Vector3(0, 1, 0), 2.5f));

        // Quaternion – our wire order is w,x,y,z
        Helpers.WriteQuaternion(OFF_QUATERNION, new Quaternion(0, 1, 0, 0)); // w=0, x=1, y=0, z=0

        Helpers.WriteAABB(OFF_AABB, new AABB(new Vector3(1, 2, 3), new Vector3(4, 5, 6)));
        Helpers.WriteBasis(OFF_BASIS, new Basis(new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)));
        Helpers.WriteTransform3D(OFF_TRANSFORM3D,
            new Transform3D(new Basis(new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)),
                            new Vector3(7, 8, 9)));
        Helpers.WriteProjection(OFF_PROJECTION,
            new Projection(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 0, 0, 1)));
        Helpers.WriteColor(OFF_COLOR, new Color(0.1f, 0.2f, 0.3f, 0.4f));

        // Packed arrays
        Helpers.WritePackedByteArray(OFF_PACKED_BYTE, new byte[] { 1, 2, 3 });
        Helpers.WritePackedInt32Array(OFF_PACKED_INT32, new int[] { 100, 200, 300 });
        Helpers.WritePackedInt64Array(OFF_PACKED_INT64, new long[] { 1000, 2000 });
        Helpers.WritePackedFloat32Array(OFF_PACKED_FLOAT32, new float[] { 1.1f, 2.2f });
        Helpers.WritePackedFloat64Array(OFF_PACKED_FLOAT64, new double[] { 3.3, 4.4 });
        Helpers.WritePackedStringArray(OFF_PACKED_STRING, new string[] { "one", "two" });
        Helpers.WritePackedVector2Array(OFF_PACKED_VECTOR2, new Vector2[] { new Vector2(1, 2), new Vector2(3, 4) });
        Helpers.WritePackedVector3Array(OFF_PACKED_VECTOR3, new Vector3[] { new Vector3(5, 6, 7) });
        Helpers.WritePackedColorArray(OFF_PACKED_COLOR, new Color[] { new Color(1, 0, 0, 1), new Color(0, 1, 0, 1) });

        // Dictionary
        var dict = new Dictionary<object, object>
        {
            { "id", 7L },
            { "name", "alpha" },
            { "items", new object[] { 8L, 9L } }
        };
        // Dictionary
        Helpers.WriteDictionary(OFF_DICTIONARY, dict);

        // Array
        var arr = new object[] { 1L, "two", new Vector3(3, 4, 5) };

        // Array
        Helpers.WriteArray(OFF_ARRAY, arr);          // replaces Helpers.WriteArray
    }
}
