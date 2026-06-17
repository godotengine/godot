using System;
using System.Collections.Generic;
using System.Text;
using Godot;

#nullable enable

// GD.cs — shim matching Godot's global GD namespace for .NET WASM.
// Provides the subset of GD.* functions used by game scripts so they
// compile unchanged against this runtime. Unimplemented methods that
// depend on Variant (Load, VarToStr, etc.) are commented out until
// the Variant layer is wired up.
public static class GD
{
	// ── Print / Error ──────────────────────────────────────────────────────
	// All overloads funnel into the two string primitivies below.
	// PrintRich forwards to Console as-is; BBCode tags are not rendered.

	public static void Print(string what) => Console.WriteLine(what);
	public static void Print(params object[] what) => Print(AppendPrintParams(what));

	public static void PrintRich(string what) => Console.WriteLine(what);
	public static void PrintRich(params object[] what) => PrintRich(AppendPrintParams(what));

	public static void PrintErr(string what) => Console.Error.WriteLine(what);
	public static void PrintErr(params object[] what) => PrintErr(AppendPrintParams(what));

	// PrintRaw omits the trailing newline, matching GD.print_raw behavior.
	public static void PrintRaw(string what) => Console.Write(what);
	public static void PrintRaw(params object[] what) => PrintRaw(AppendPrintParams(what));

	// PrintS / PrintT join arguments with a space or tab respectively.
	public static void PrintS(params object[] what) => Print(AppendPrintParams(' ', what));
	public static void PrintT(params object[] what) => Print(AppendPrintParams('\t', what));

	public static void PushError(string message) => Console.Error.WriteLine($"[ERROR] {message}");
	public static void PushError(params object[] what) => PushError(AppendPrintParams(what));

	public static void PushWarning(string message) => Console.WriteLine($"[WARNING] {message}");
	public static void PushWarning(params object[] what) => PushWarning(AppendPrintParams(what));


	// ── Random ─────────────────────────────────────────────────────────────

	private static Random _random = new Random();

	public static float Randf() => (float)_random.NextDouble();

	// Box-Muller transform — produces a normally distributed sample.
	public static double Randfn(double mean, double deviation)
	{
		double u1 = 1.0 - _random.NextDouble();
		double u2 = 1.0 - _random.NextDouble();
		double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
		return mean + deviation * z;
	}

	// Full 32-bit range via NextBytes — avoids the int31 ceiling of Next().
	public static uint Randi()
	{
		byte[] buf = new byte[4];
		_random.NextBytes(buf);
		return BitConverter.ToUInt32(buf, 0);
	}

	// Re-seeds with a time-dependent value, matching GD.randomize().
	public static void Randomize() => _random = new Random();

	public static double RandRange(double from, double to) =>
		from + _random.NextDouble() * (to - from);

	// Next(min, max) is exclusive of max; +1 matches Godot's inclusive upper bound.
	public static int RandRange(int from, int to) => _random.Next(from, to + 1);

	// PCG32 minimal implementation matching Godot's internal seed-based RNG.
	public static uint RandFromSeed(ref long seed)
	{
		long old = seed;
		seed = old * 6364136223846793005L + 1442695040888963407L;
		uint shifted = (uint)(((old >> 18) ^ old) >> 27);
		int rot = (int)(old >> 59);
		return (shifted >> rot) | (shifted << ((-rot) & 31));
	}

	// System.Random takes an int seed; casting truncates but keeps state consistent.
	public static void Seed(long seed) => _random = new Random((int)seed);



	// ── Helpers ────────────────────────────────────────────────────────────

	private static string AppendPrintParams(object[] parameters)
	{
		if (parameters == null) return "null";
		var sb = new StringBuilder();
		foreach (var p in parameters)
			sb.Append(p?.ToString() ?? "null");
		return sb.ToString();
	}

	private static string AppendPrintParams(char separator, object[] parameters)
	{
		if (parameters == null) return "null";
		var sb = new StringBuilder();
		for (int i = 0; i < parameters.Length; i++)
		{
			if (i != 0) sb.Append(separator);
			sb.Append(parameters[i]?.ToString() ?? "null");
		}
		return sb.ToString();
	}
}
