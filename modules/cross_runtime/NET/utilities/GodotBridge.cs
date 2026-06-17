using System;
using System.Runtime.InteropServices.JavaScript;

/// <summary>
/// Primary interop layer between the .NET WebAssembly runtime and the Godot WebAssembly runtime.
/// CallGodot provides a generic invocation path
/// CallGodotPacked provides a specialized bulk-memory path for large data transfers such as PackedArray types./// Both methods ultimately forward to the JavaScript bridge
/// (globalThis.__callGodot), which dispatches the request to exported functions inside the Godot WebAssembly module.
/// </summary>

namespace Godot
{
	public static partial class GodotBridge
	{
		[JSImport("globalThis.__callGodot")]
		[return: JSMarshalAs<JSType.Any>]
		public static partial object CallGodot(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPackedTest(
			string fn,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked(
			string fn,
			[JSMarshalAs<JSType.Number>] double id,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked1(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked2(
				   string fn,
				   [JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
				   [JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0,
				   [JSMarshalAs<JSType.MemoryView>] Span<byte> buffer1
			   );

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked3(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer1,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer2
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked4(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer1,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer2,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer3
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked5(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer1,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer2,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer3,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer4
		);

		[JSImport("globalThis.__callGodot")]
		public static partial void CallGodotPacked6(
			string fn,
			[JSMarshalAs<JSType.Array<JSType.Any>>] object[] args,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer0,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer1,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer2,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer3,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer4,
			[JSMarshalAs<JSType.MemoryView>] Span<byte> buffer5
		);


	}
}
