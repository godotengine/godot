using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

public static partial class Tests
{
	// Offsets – must match C++ test exactly
	// Base of the shared command area
	public const int CMD_OFFSET = 0x5000;  // 20480
	public const int STATUS_OFFSET = 0x5004;  // 20484
	public const int RESULT_OFFSET = 0x5008;  // 20488

	// Data section begins at 0x5016 (20496)
	public const int CMD_DATA = 0x5016;

	// Status and command values (must match C++)
	public const int STATUS_PENDING = 0;
	public const int STATUS_DONE = 1;
	public const int CMD_RUN_VARIANT_TESTS = 1;

	// All data offsets relative to CMD_DATA
	private const int OFF_INT32 = CMD_DATA + 0;    // 0x5016
	private const int OFF_INT64 = CMD_DATA + 4;    // 0x501A
	private const int OFF_FLOAT = CMD_DATA + 12;   // 0x5022
	private const int OFF_DOUBLE = CMD_DATA + 16;   // 0x5026
	private const int OFF_BOOL = CMD_DATA + 24;   // 0x502E
	private const int OFF_STRING = CMD_DATA + 28;   // 0x5032
	private const int OFF_STRINGNAME = CMD_DATA + 48;   // 0x5046
	private const int OFF_NODEPATH = CMD_DATA + 60;   // 0x5052
	private const int OFF_RID = CMD_DATA + 76;   // 0x5062
	private const int OFF_VECTOR2 = CMD_DATA + 84;   // 0x506A
	private const int OFF_VECTOR2I = CMD_DATA + 92;   // 0x5072
	private const int OFF_RECT2 = CMD_DATA + 100;  // 0x507A
	private const int OFF_RECT2I = CMD_DATA + 116;  // 0x5090
	private const int OFF_VECTOR3 = CMD_DATA + 132;  // 0x509A
	private const int OFF_VECTOR3I = CMD_DATA + 144;  // 0x50A6
	private const int OFF_VECTOR4 = CMD_DATA + 156;  // 0x50B2
	private const int OFF_VECTOR4I = CMD_DATA + 172;  // 0x50C2
	private const int OFF_PLANE = CMD_DATA + 188;  // 0x50D2
	private const int OFF_QUATERNION = CMD_DATA + 204;  // 0x50E2
	private const int OFF_AABB = CMD_DATA + 220;  // 0x50F2
	private const int OFF_BASIS = CMD_DATA + 244;  // 0x510A
	private const int OFF_TRANSFORM3D = CMD_DATA + 280;  // 0x512E
	private const int OFF_PROJECTION = CMD_DATA + 328;  // 0x515E
	private const int OFF_COLOR = CMD_DATA + 392;  // 0x519E
	private const int OFF_PACKED_BYTE = CMD_DATA + 408;  // 0x51AE
	private const int OFF_PACKED_INT32 = CMD_DATA + 415;  // 0x51B5
	private const int OFF_PACKED_INT64 = CMD_DATA + 431;  // 0x51C5
	private const int OFF_PACKED_FLOAT32 = CMD_DATA + 451;  // 0x51D9
	private const int OFF_PACKED_FLOAT64 = CMD_DATA + 463;  // 0x51E5
	private const int OFF_PACKED_STRING = CMD_DATA + 483;  // 0x51F9
	private const int OFF_PACKED_VECTOR2 = CMD_DATA + 501;  // 0x520B
	private const int OFF_PACKED_VECTOR3 = CMD_DATA + 521;  // 0x521F
	private const int OFF_PACKED_COLOR = CMD_DATA + 537;  // 0x522F
	private const int OFF_DICTIONARY = CMD_DATA + 600;  // 0x526E
	private const int OFF_ARRAY = CMD_DATA + 720;  // 0x52E6




	public static async Task Run()
	{
		//Console.WriteLine("Started in C#");
		Helpers.ResetCommandBuffer();
		WriteAllData(); // Writes all the data first
						//Console.WriteLine($"The print here is : {Helpers.ReadInt32(OFF_INT32)}"); // confirms the first one is correct

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

		Helpers.WriteQuaternion(OFF_QUATERNION, new Quaternion(1, 0, 0, 0));

		Helpers.WriteAABB(OFF_AABB, new AABB(new Vector3(1, 2, 3), new Vector3(4, 5, 6)));

		Helpers.WriteBasis(OFF_BASIS, new Basis(new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)));

		Helpers.WriteTransform3D(OFF_TRANSFORM3D, new Transform3D(new Basis(new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)), new Vector3(7, 8, 9)));

		Helpers.WriteProjection(OFF_PROJECTION, new Projection(new Vector4(1, 0, 0, 0), new Vector4(0, 1, 0, 0), new Vector4(0, 0, 1, 0), new Vector4(0, 0, 0, 1)));

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

		Helpers.WriteDictionary(OFF_DICTIONARY, dict);

		// Array
		var arr = new object[] { 1L, "two", new Vector3(3, 4, 5) };


		Helpers.WriteArray(OFF_ARRAY, arr);
	}
}
