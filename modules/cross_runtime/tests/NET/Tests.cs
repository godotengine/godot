using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

public static partial class Tests
{
    // Offsets — must match C++ test exactly.
    public const int CMD_OFFSET = 0x5000;
    public const int STATUS_OFFSET = 0x5004;
    public const int RESULT_OFFSET = 0x5008;

    // Data section begins at 0x5016 (20496).
    public const int CMD_DATA = 0x5016;

    // Status and command values — must match C++.
    public const int STATUS_PENDING = 0;
    public const int STATUS_DONE = 1;
    public const int CMD_RUN_VARIANT_TESTS = 1;

    // All data offsets relative to CMD_DATA.
    private const int OFF_INT32 = CMD_DATA + 0;
    private const int OFF_INT64 = CMD_DATA + 4;
    private const int OFF_FLOAT = CMD_DATA + 12;
    private const int OFF_DOUBLE = CMD_DATA + 16;
    private const int OFF_BOOL = CMD_DATA + 24;
    private const int OFF_STRING = CMD_DATA + 28;
    private const int OFF_STRINGNAME = CMD_DATA + 48;
    private const int OFF_NODEPATH = CMD_DATA + 60;
    private const int OFF_RID = CMD_DATA + 76;
    private const int OFF_VECTOR2 = CMD_DATA + 84;
    private const int OFF_VECTOR2I = CMD_DATA + 92;
    private const int OFF_RECT2 = CMD_DATA + 100;
    private const int OFF_RECT2I = CMD_DATA + 116;
    private const int OFF_VECTOR3 = CMD_DATA + 132;
    private const int OFF_VECTOR3I = CMD_DATA + 144;
    private const int OFF_VECTOR4 = CMD_DATA + 156;
    private const int OFF_VECTOR4I = CMD_DATA + 172;
    private const int OFF_PLANE = CMD_DATA + 188;
    private const int OFF_QUATERNION = CMD_DATA + 204;
    private const int OFF_AABB = CMD_DATA + 220;
    private const int OFF_BASIS = CMD_DATA + 244;
    private const int OFF_TRANSFORM3D = CMD_DATA + 280;
    private const int OFF_PROJECTION = CMD_DATA + 328;
    private const int OFF_COLOR = CMD_DATA + 392;
    private const int OFF_PACKED_BYTE = CMD_DATA + 408;
    private const int OFF_PACKED_INT32 = CMD_DATA + 415;
    private const int OFF_PACKED_INT64 = CMD_DATA + 431;
    private const int OFF_PACKED_FLOAT32 = CMD_DATA + 451;
    private const int OFF_PACKED_FLOAT64 = CMD_DATA + 463;
    private const int OFF_PACKED_STRING = CMD_DATA + 483;
    private const int OFF_PACKED_VECTOR2 = CMD_DATA + 501;
    private const int OFF_PACKED_VECTOR3 = CMD_DATA + 521;
    private const int OFF_PACKED_COLOR = CMD_DATA + 537;
    private const int OFF_DICTIONARY = CMD_DATA + 600;
    private const int OFF_ARRAY = CMD_DATA + 720;
    private const int OFF_SIGNAL = CMD_DATA + 900;

    public static async Task Run()
    {

        WriteInitialData();
        Helpers.SendCommand(CMD_RUN_VARIANT_TESTS);
        Helpers.WaitForCompletion();

        byte result = Helpers.ReadByte(RESULT_OFFSET);
        bool cppPassed = (result == 1);
        Console.WriteLine(cppPassed ? "C++ test: PASSED" : "C++ test: FAILED");

        if (cppPassed)
        {
            bool verifyPassed = ReadAndVerifyWrittenBackData();
            Console.WriteLine(verifyPassed ? "C# verification: PASSED" : "C# verification: FAILED");
            Console.WriteLine(verifyPassed ? "bridge_test: PASSED" : "bridge_test: FAILED");
        }
        else
        {
            Console.WriteLine("bridge_test: FAILED (C++ failed)");
        }
    }

    private static void WriteInitialData()
    {
        // Scalars.
        Helpers.WriteInt32(OFF_INT32, 42);
        Helpers.WriteInt64(OFF_INT64, 12345678901234L);
        Helpers.WriteFloat(OFF_FLOAT, 3.14f);
        Helpers.WriteDouble(OFF_DOUBLE, 2.718281828);
        Helpers.WriteInt32(OFF_BOOL, 1);

        // Strings.
        Helpers.WriteString(OFF_STRING, "Hello, WASM!");
        Helpers.WriteString(OFF_STRINGNAME, "TestName");
        Helpers.WriteString(OFF_NODEPATH, "/root/Player");

        // RID.
        Helpers.WriteUInt64(OFF_RID, 0xDEADBEEF);

        // Vectors and math types.
        Helpers.WriteVector2(OFF_VECTOR2, new Vector2(1.5f, 2.5f));
        Helpers.WriteVector2i(OFF_VECTOR2I, new Vector2i(10, -20));
        Helpers.WriteRect2(OFF_RECT2, new Rect2(new Vector2(1, 2), new Vector2(3, 4)));
        Helpers.WriteRect2i(OFF_RECT2I, new Rect2i(new Vector2i(5, 6), new Vector2i(7, 8)));
        Helpers.WriteVector3(OFF_VECTOR3, new Vector3(1, 2, 3));
        Helpers.WriteVector3i(OFF_VECTOR3I, new Vector3i(4, 5, 6));
        Helpers.WriteVector4(OFF_VECTOR4, new Vector4(1, 2, 3, 4));
        Helpers.WriteVector4i(OFF_VECTOR4I, new Vector4i(5, 6, 7, 8));
        Helpers.WritePlane(OFF_PLANE, new Plane(new Vector3(0, 1, 0), 2.5f));
        Helpers.WriteQuaternion(OFF_QUATERNION, new Quaternion(1, 0, 0, 0));
        Helpers.WriteAABB(OFF_AABB, new AABB(new Vector3(1, 2, 3), new Vector3(4, 5, 6)));
        Helpers.WriteBasis(OFF_BASIS, new Basis(
            new Vector3(1, 0, 0),
            new Vector3(0, 1, 0),
            new Vector3(0, 0, 1)));
        Helpers.WriteTransform3D(
            OFF_TRANSFORM3D,
            new Transform3D(
                new Basis(
                    new Vector3(1, 0, 0),
                    new Vector3(0, 1, 0),
                    new Vector3(0, 0, 1)),
                new Vector3(7, 8, 9)));
        Helpers.WriteProjection(
            OFF_PROJECTION,
            new Projection(
                new Vector4(1, 0, 0, 0),
                new Vector4(0, 1, 0, 0),
                new Vector4(0, 0, 1, 0),
                new Vector4(0, 0, 0, 1)));
        Helpers.WriteColor(OFF_COLOR, new Color(0.1f, 0.2f, 0.3f, 0.4f));

        // Packed arrays.
        Helpers.WritePackedByteArray(OFF_PACKED_BYTE, new byte[] { 1, 2, 3 });
        Helpers.WritePackedInt32Array(OFF_PACKED_INT32, new int[] { 100, 200, 300 });
        Helpers.WritePackedInt64Array(OFF_PACKED_INT64, new long[] { 1000, 2000 });
        Helpers.WritePackedFloat32Array(OFF_PACKED_FLOAT32, new float[] { 1.1f, 2.2f });
        Helpers.WritePackedFloat64Array(OFF_PACKED_FLOAT64, new double[] { 3.3, 4.4 });
        Helpers.WritePackedStringArray(OFF_PACKED_STRING, new string[] { "one", "two" });
        Helpers.WritePackedVector2Array(OFF_PACKED_VECTOR2, new Vector2[] { new Vector2(1, 2), new Vector2(3, 4) });
        Helpers.WritePackedVector3Array(OFF_PACKED_VECTOR3, new Vector3[] { new Vector3(5, 6, 7) });
        Helpers.WritePackedColorArray(OFF_PACKED_COLOR, new Color[] { new Color(1, 0, 0, 1), new Color(0, 1, 0, 1) });

        // Dictionary.
        var dict = new Dictionary<object, object>
        {
            { "id", 7L },
            { "name", "alpha" },
            { "items", new object[] { 8L, 9L } }
        };
        Helpers.WriteDictionary(OFF_DICTIONARY, dict);

        // Array.
        var arr = new object[] { 1L, "two", new Vector3(3, 4, 5) };
        Helpers.WriteArray(OFF_ARRAY, arr);

        // Signal - Uses WriteVariant Directly
        Helpers.WriteVariant(OFF_SIGNAL, new CustomSignal(67890, "test_signal"));
    }

    private static bool ReadAndVerifyWrittenBackData()
    {
        bool ok = true;

        // Scalars.
        int i32 = Helpers.ReadInt32(OFF_INT32);
        if (i32 == -99) Console.WriteLine("[C#] read_int32 passed");
        else { Console.WriteLine($"[C#] int32 mismatch: expected -99, got {i32}"); ok = false; }

        long i64 = Helpers.ReadInt64(OFF_INT64);
        if (i64 == -98765432109876L) Console.WriteLine("[C#] read_int64 passed");
        else { Console.WriteLine($"[C#] int64 mismatch: expected -98765432109876, got {i64}"); ok = false; }

        float f = Helpers.ReadFloat(OFF_FLOAT);
        if (f == -2.71f) Console.WriteLine("[C#] read_float passed");
        else { Console.WriteLine($"[C#] float mismatch: expected -2.71, got {f}"); ok = false; }

        double d = Helpers.ReadDouble(OFF_DOUBLE);
        if (d == -1.41421356237) Console.WriteLine("[C#] read_double passed");
        else { Console.WriteLine($"[C#] double mismatch: expected -1.41421356237, got {d}"); ok = false; }

        bool b = Helpers.ReadInt32(OFF_BOOL) != 0;
        if (!b) Console.WriteLine("[C#] bool passed");
        else { Console.WriteLine("[C#] bool mismatch: expected false"); ok = false; }

        // Strings.
        string str = Helpers.ReadString(OFF_STRING);
        if (str == "WriteOK!") Console.WriteLine("[C#] Read String passed");
        else { Console.WriteLine($"[C#] String mismatch: expected 'WriteOK!', got '{str}'"); ok = false; }

        string sn = Helpers.ReadStringName(OFF_STRINGNAME);
        if (sn == "WriteSN") Console.WriteLine("[C#] Read stringname passed");
        else { Console.WriteLine($"[C#] StringName mismatch: expected 'WriteSN', got '{sn}'"); ok = false; }

        string np = Helpers.ReadNodePath(OFF_NODEPATH);
        if (np == "/write/path") Console.WriteLine("[C#] Read NodePath passed");
        else { Console.WriteLine($"[C#] NodePath mismatch: expected '/write/path', got '{np}'"); ok = false; }

        // RID.
        ulong rid = Helpers.ReadRID(OFF_RID);
        if (rid == 0xBEEFCAFEUL) Console.WriteLine("[C#] Read RID passed");
        else { Console.WriteLine($"[C#] RID mismatch: expected 0xBEEFCAFE, got 0x{rid:X}"); ok = false; }

        // Vectors and math.
        Vector2 v2 = Helpers.ReadVector2(OFF_VECTOR2);
        if (v2.X == 10.5f && v2.Y == -5.5f) Console.WriteLine("[C#] Read Vector2 passed");
        else { Console.WriteLine($"[C#] Vector2 mismatch: expected (10.5, -5.5), got ({v2.X}, {v2.Y})"); ok = false; }

        Vector2i v2i = Helpers.ReadVector2i(OFF_VECTOR2I);
        if (v2i.X == -100 && v2i.Y == 200) Console.WriteLine("[C#] Read Vector2i passed");
        else { Console.WriteLine($"[C#] Vector2i mismatch: expected (-100, 200), got ({v2i.X}, {v2i.Y})"); ok = false; }

        Rect2 r2 = Helpers.ReadRect2(OFF_RECT2);
        if (r2.Position.X == 9f && r2.Position.Y == 8f && r2.Size.X == 7f && r2.Size.Y == 6f)
            Console.WriteLine("[C#] Read Rect2 passed");
        else { Console.WriteLine("[C#] Rect2 mismatch"); ok = false; }

        Rect2i r2i = Helpers.ReadRect2i(OFF_RECT2I);
        if (r2i.Position.X == 15 && r2i.Position.Y == 16 && r2i.Size.X == 17 && r2i.Size.Y == 18)
            Console.WriteLine("[C#] Read Rect2i passed");
        else { Console.WriteLine("[C#] Rect2i mismatch"); ok = false; }

        Vector3 v3 = Helpers.ReadVector3(OFF_VECTOR3);
        if (v3.X == 10f && v3.Y == 20f && v3.Z == 30f)
            Console.WriteLine("[C#] Read Vector3 passed");
        else { Console.WriteLine("[C#] Vector3 mismatch"); ok = false; }

        Vector3i v3i = Helpers.ReadVector3i(OFF_VECTOR3I);
        if (v3i.X == 40 && v3i.Y == 50 && v3i.Z == 60)
            Console.WriteLine("[C#] Read Vector3i passed");
        else { Console.WriteLine("[C#] Vector3i mismatch"); ok = false; }

        Vector4 v4 = Helpers.ReadVector4(OFF_VECTOR4);
        if (v4.X == 11f && v4.Y == 22f && v4.Z == 33f && v4.W == 44f)
            Console.WriteLine("[C#] Read Vector4 passed");
        else { Console.WriteLine("[C#] Vector4 mismatch"); ok = false; }

        Vector4i v4i = Helpers.ReadVector4i(OFF_VECTOR4I);
        if (v4i.X == 55 && v4i.Y == 66 && v4i.Z == 77 && v4i.W == 88)
            Console.WriteLine("[C#] Read Vector4i passed");
        else { Console.WriteLine("[C#] Vector4i mismatch"); ok = false; }

        Plane plane = Helpers.ReadPlane(OFF_PLANE);
        if (plane.Normal.X == 1f && plane.Normal.Y == 0f && plane.Normal.Z == 0f && plane.D == 5f)
            Console.WriteLine("[C#] Read Plane passed");
        else { Console.WriteLine("[C#] Plane mismatch"); ok = false; }

        Quaternion q = Helpers.ReadQuaternion(OFF_QUATERNION);
        if (q.X == 0f && q.Y == 0.707f && q.Z == 0f && q.W == 0.707f)
            Console.WriteLine("[C#] Read Quaternion passed");
        else { Console.WriteLine("[C#] Quaternion mismatch"); ok = false; }

        AABB aabb = Helpers.ReadAABB(OFF_AABB);
        if (aabb.Position.X == 0f && aabb.Position.Y == 0f && aabb.Position.Z == 0f &&
            aabb.Size.X == 10f && aabb.Size.Y == 10f && aabb.Size.Z == 10f)
            Console.WriteLine("[C#] Read AABB passed");
        else { Console.WriteLine("[C#] AABB mismatch"); ok = false; }

        Basis basis = Helpers.ReadBasis(OFF_BASIS);
        if (basis.Column0.X == 0f && basis.Column0.Y == 0f && basis.Column0.Z == 1f &&
            basis.Column1.X == 0f && basis.Column1.Y == 1f && basis.Column1.Z == 0f &&
            basis.Column2.X == -1f && basis.Column2.Y == 0f && basis.Column2.Z == 0f)
        {
            Console.WriteLine("[C#] Read Basis passed");
        }
        else
        {
            Console.WriteLine(
                $"[C#] Basis mismatch: got cols " +
                $"({basis.Column0.X},{basis.Column0.Y},{basis.Column0.Z}), " +
                $"({basis.Column1.X},{basis.Column1.Y},{basis.Column1.Z}), " +
                $"({basis.Column2.X},{basis.Column2.Y},{basis.Column2.Z})");
            ok = false;
        }

        // Transform3D (contains its own Basis + origin).
        Transform3D t3d = Helpers.ReadTransform3D(OFF_TRANSFORM3D);
        Basis t3dBasis = t3d.Basis;

        // C++ writes: Transform3D(Basis(Vector3(-1,0,0), Vector3(0,-1,0), Vector3(0,0,-1)), Vector3(-7,-8,-9))
        if (t3dBasis.Column0.X == -1f && t3dBasis.Column0.Y == 0f && t3dBasis.Column0.Z == 0f &&
            t3dBasis.Column1.X == 0f && t3dBasis.Column1.Y == -1f && t3dBasis.Column1.Z == 0f &&
            t3dBasis.Column2.X == 0f && t3dBasis.Column2.Y == 0f && t3dBasis.Column2.Z == -1f &&
            t3d.Origin.X == -7f && t3d.Origin.Y == -8f && t3d.Origin.Z == -9f)
        {
            Console.WriteLine("[C#] Read Transform3D passed");
        }
        else
        {
            Console.WriteLine(
                $"[C#] Transform3D mismatch: got basis cols " +
                $"({t3dBasis.Column0.X},{t3dBasis.Column0.Y},{t3dBasis.Column0.Z}), " +
                $"({t3dBasis.Column1.X},{t3dBasis.Column1.Y},{t3dBasis.Column1.Z}), " +
                $"({t3dBasis.Column2.X},{t3dBasis.Column2.Y},{t3dBasis.Column2.Z}) " +
                $"origin ({t3d.Origin.X},{t3d.Origin.Y},{t3d.Origin.Z})");
            ok = false;
        }

        Projection proj = Helpers.ReadProjection(OFF_PROJECTION);
        if (proj.X.X == 2f && proj.X.Y == 0f && proj.X.Z == 0f && proj.X.W == 0f &&
            proj.Y.X == 0f && proj.Y.Y == 2f && proj.Y.Z == 0f && proj.Y.W == 0f &&
            proj.Z.X == 0f && proj.Z.Y == 0f && proj.Z.Z == 2f && proj.Z.W == 0f &&
            proj.W.X == 0f && proj.W.Y == 0f && proj.W.Z == 0f && proj.W.W == 2f)
            Console.WriteLine("[C#] Read Projection passed");
        else { Console.WriteLine("[C#] Projection mismatch"); ok = false; }

        Color color = Helpers.ReadColor(OFF_COLOR);
        if (color.R == 0.9f && color.G == 0.8f && color.B == 0.7f && color.A == 0.6f)
            Console.WriteLine("[C#] Read Color passed");
        else { Console.WriteLine("[C#] Color mismatch"); ok = false; }

        // Packed arrays.
        byte[] pba = Helpers.ReadPackedByteArray(OFF_PACKED_BYTE);
        if (pba.Length == 3 && pba[0] == 10 && pba[1] == 20 && pba[2] == 30)
            Console.WriteLine("[C#] Read PackedByteArray passed");
        else { Console.WriteLine("[C#] PackedByteArray mismatch"); ok = false; }

        int[] p32 = Helpers.ReadPackedInt32Array(OFF_PACKED_INT32);
        if (p32.Length == 3 && p32[0] == 400 && p32[1] == 500 && p32[2] == 600)
            Console.WriteLine("[C#] Read PackedInt32Array passed");
        else { Console.WriteLine("[C#] PackedInt32Array mismatch"); ok = false; }

        long[] p64 = Helpers.ReadPackedInt64Array(OFF_PACKED_INT64);
        if (p64.Length == 2 && p64[0] == -1000 && p64[1] == -2000)
            Console.WriteLine("[C#] Read PackedInt64Array passed");
        else { Console.WriteLine("[C#] PackedInt64Array mismatch"); ok = false; }

        float[] pf32 = Helpers.ReadPackedFloat32Array(OFF_PACKED_FLOAT32);
        if (pf32.Length == 2 && pf32[0] == -1.1f && pf32[1] == -2.2f)
            Console.WriteLine("[C#] Read PackedFloat32Array passed");
        else { Console.WriteLine("[C#] PackedFloat32Array mismatch"); ok = false; }

        double[] pf64 = Helpers.ReadPackedFloat64Array(OFF_PACKED_FLOAT64);
        if (pf64.Length == 2 && pf64[0] == -3.3 && pf64[1] == -4.4)
            Console.WriteLine("[C#] Read PackedFloat64Array passed");
        else { Console.WriteLine("[C#] PackedFloat64Array mismatch"); ok = false; }

        string[] ps = Helpers.ReadPackedStringArray(OFF_PACKED_STRING);
        if (ps.Length == 2 && ps[0] == "abc" && ps[1] == "xyz")
            Console.WriteLine("[C#] Read PackedStringArray passed");
        else { Console.WriteLine("[C#] PackedStringArray mismatch"); ok = false; }

        Vector2[] pv2 = Helpers.ReadPackedVector2Array(OFF_PACKED_VECTOR2);
        if (pv2.Length == 2 &&
            pv2[0].X == 10f && pv2[0].Y == 20f &&
            pv2[1].X == 30f && pv2[1].Y == 40f)
            Console.WriteLine("[C#] Read PackedVector2Array passed");
        else { Console.WriteLine("[C#] PackedVector2Array mismatch"); ok = false; }

        Vector3[] pv3 = Helpers.ReadPackedVector3Array(OFF_PACKED_VECTOR3);
        if (pv3.Length == 1 &&
            pv3[0].X == 50f && pv3[0].Y == 60f && pv3[0].Z == 70f)
            Console.WriteLine("[C#] Read PackedVector3Array passed");
        else { Console.WriteLine("[C#] PackedVector3Array mismatch"); ok = false; }

        Color[] pc = Helpers.ReadPackedColorArray(OFF_PACKED_COLOR);
        if (pc.Length == 2 &&
            pc[0].R == 0f && pc[0].G == 0f && pc[0].B == 1f && pc[0].A == 1f &&
            pc[1].R == 1f && pc[1].G == 1f && pc[1].B == 0f && pc[1].A == 1f)
            Console.WriteLine("[C#] Read PackedColorArray passed");
        else { Console.WriteLine("[C#] PackedColorArray mismatch"); ok = false; }

        // Array.
        object arrObj = Helpers.ReadArray(OFF_ARRAY);
        if (arrObj is System.Collections.IList list)
        {
            if (list.Count == 3 &&
                list[0] is Vector3 v0 &&
                v0.X == 30f && v0.Y == 40f && v0.Z == 50f &&
                list[1] is string s1 && s1 == "write" &&
                list[2] is long l2 && l2 == 42)
            {
                Console.WriteLine("[C#] Read Array passed");
            }
            else
            {
                Console.WriteLine("[C#] Array content mismatch");
                ok = false;
            }
        }
        else
        {
            Console.WriteLine("[C#] Array not a list");
            ok = false;
        }

        // Dictionary.
        IDictionary dictObj = (IDictionary)Helpers.ReadDictionary(OFF_DICTIONARY);
        if (dictObj is IDictionary dict)
        {
            if (dict.Count == 3 &&
                dict.Contains("x") && dict["x"] is long xVal && xVal == 10 &&
                dict.Contains("y") && dict["y"] is long yVal && yVal == 20 &&
                dict.Contains("z") && dict["z"] is Vector3 zVal &&
                zVal.X == 1f && zVal.Y == 2f && zVal.Z == 3f)
            {
                Console.WriteLine("[C#] Read Dictionary passed");
            }
            else
            {
                Console.WriteLine("[C#] Dictionary content mismatch");
                ok = false;
            }
        }
        else
        {
            Console.WriteLine("[C#] Dictionary not a dictionary");
            ok = false;
        }

        // Signal.
        CustomSignal signal = (CustomSignal)Helpers.ReadSignal(OFF_SIGNAL);

        if (signal.TargetId == 99999UL && string.Equals(signal.Name, "write_signal", StringComparison.Ordinal))
        {
            Console.WriteLine("[C#] Read Signal passed");
        }
        else
        {
            Console.WriteLine($"[C#] Signal mismatch: expected (99999, 'write_signal'), got ({signal.TargetId}, '{signal.Name}')");
            ok = false;
        }

        return ok;
    }
}