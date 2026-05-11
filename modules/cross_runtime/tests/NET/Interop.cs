using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices.JavaScript;

public static partial class Interop
{
	[JSImport("bulkRead", "interop")]
	public static partial byte[] BulkRead(int srcOffset, int length);

	[JSImport("bulkWrite", "interop")]
	public static partial void BulkWrite(byte[] srcArray, int destOffset, int length);

	[JSImport("atomicWriteInt32", "interop")]
	public static partial void AtomicWriteInt32(int offset, int value);

	[JSImport("atomicReadInt32", "interop")]
	public static partial int AtomicReadInt32(int offset);

	[JSExport]
	public static async Task InitInterop()
	{
		await JSHost.ImportAsync("interop", "/cs/interop.js");
	}

	[JSExport]
	public static async Task RunGame()
	{
		Helpers.ResetCommandBuffer();
		await Tests.Run();
	}
}
