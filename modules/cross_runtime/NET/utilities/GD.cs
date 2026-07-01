using System;
using System.Collections.Generic;
using System.Text;


#nullable enable

// GD.cs — shim matching Godot's global GD namespace for .NET WASM.
// Provides the subset of GD.* functions used by game scripts so they
// compile unchanged against this runtime. Unimplemented methods that
// depend on Variant (Load, VarToStr, etc.) are commented out until
// the Variant layer is wired up.
namespace Godot
{
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

		/// <summary>
		/// Returns a <see cref="IEnumerable{T}"/> that iterates from
		/// <paramref name="start"/> (inclusive) to <paramref name="end"/> (exclusive)
		/// in steps of <c>1</c>.
		/// </summary>
		/// <param name="start">The first index.</param>
		/// <param name="end">The last index.</param>
		public static IEnumerable<int> Range(int start, int end)
		{
			return Range(start, end, 1);
		}

		/// <summary>
		/// Returns a <see cref="IEnumerable{T}"/> that iterates from
		/// <paramref name="start"/> (inclusive) to <paramref name="end"/> (exclusive)
		/// in steps of <paramref name="step"/>.
		/// The argument <paramref name="step"/> can be negative, but not <c>0</c>.
		/// </summary>
		/// <exception cref="ArgumentException">
		/// <paramref name="step"/> is 0.
		/// </exception>
		/// <param name="start">The first index.</param>
		/// <param name="end">The last index.</param>
		/// <param name="step">The amount by which to increment the index on each iteration.</param>
		public static IEnumerable<int> Range(int start, int end, int step)
		{
			if (step == 0)
				throw new ArgumentException("step cannot be 0.", nameof(step));

			if (end < start && step > 0)
				yield break;

			if (end > start && step < 0)
				yield break;

			if (step > 0)
			{
				for (int i = start; i < end; i += step)
					yield return i;
			}
			else
			{
				for (int i = start; i > end; i += step)
					yield return i;
			}
		}

		/// <summary>
		/// Loads a resource from the filesystem located at <paramref name="path"/>.
		/// The resource is loaded on the method call (unless it's referenced already
		/// elsewhere, e.g. in another script or in the scene), which might cause slight delay,
		/// especially when loading scenes. To avoid unnecessary delays when loading something
		/// multiple times, either store the resource in a variable.
		///
		/// Note: Resource paths can be obtained by right-clicking on a resource in the FileSystem
		/// dock and choosing "Copy Path" or by dragging the file from the FileSystem dock into the script.
		///
		/// Important: The path must be absolute, a local path will just return <see langword="null"/>.
		/// This method is a simplified version of <see cref="ResourceLoader.Load"/>, which can be used
		/// for more advanced scenarios.
		/// </summary>
		/// <example>
		/// <code>
		/// // Load a scene called main located in the root of the project directory and cache it in a variable.
		/// var main = GD.Load("res://main.tscn"); // main will contain a PackedScene resource.
		/// </code>
		/// </example>
		/// <param name="path">Path of the <see cref="Resource"/> to load.</param>
		/// <returns>The loaded <see cref="Resource"/>.</returns>
		public static Resource Load(string path)
		{
			return ResourceLoader.Load(path);
		}

		/// <summary>
		/// Loads a resource from the filesystem located at <paramref name="path"/>.
		/// The resource is loaded on the method call (unless it's referenced already
		/// elsewhere, e.g. in another script or in the scene), which might cause slight delay,
		/// especially when loading scenes. To avoid unnecessary delays when loading something
		/// multiple times, either store the resource in a variable.
		///
		/// Note: Resource paths can be obtained by right-clicking on a resource in the FileSystem
		/// dock and choosing "Copy Path" or by dragging the file from the FileSystem dock into the script.
		///
		/// Important: The path must be absolute, a local path will just return <see langword="null"/>.
		/// This method is a simplified version of <see cref="ResourceLoader.Load"/>, which can be used
		/// for more advanced scenarios.
		/// </summary>
		/// <example>
		/// <code>
		/// // Load a scene called main located in the root of the project directory and cache it in a variable.
		/// var main = GD.Load&lt;PackedScene&gt;("res://main.tscn"); // main will contain a PackedScene resource.
		/// </code>
		/// </example>
		/// <param name="path">Path of the <see cref="Resource"/> to load.</param>
		/// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Resource"/>.</typeparam>
		public static T Load<T>(string path) where T : class
		{
			return ResourceLoader.Load<T>(path);
		}



		private static string AppendPrintParams(object[] parameters)
		{
			if (parameters == null)
			{
				return "null";
			}

			var sb = new StringBuilder();
			for (int i = 0; i < parameters.Length; i++)
			{
				sb.Append(parameters[i]?.ToString() ?? "null");
			}
			return sb.ToString();
		}

		private static string AppendPrintParams(char separator, object[] parameters)
		{
			if (parameters == null)
			{
				return "null";
			}

			var sb = new StringBuilder();
			for (int i = 0; i < parameters.Length; i++)
			{
				if (i != 0)
					sb.Append(separator);
				sb.Append(parameters[i]?.ToString() ?? "null");
			}
			return sb.ToString();
		}
	}
}
