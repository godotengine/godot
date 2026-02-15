using System;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Bridge
{
    internal static class ScriptMethodCache<T>
        where T : GodotObject
    {

        [StructLayout(LayoutKind.Sequential, Pack = 8)]
        private struct MethodSlot
        {
            public IntPtr Key; // 8 Byte
            public int ArgCount; // 4 Byte
            public ScriptMethod<GodotObject> Method; // 8 Byte (reference)
            // The CPU pads here often up to 24 or 32 bytes, 
            // which fits perfectly in a 64 bytes cache line.
        }

        private const int Tries = 8;
        private const int SizeLimit = 8192;

        private static MethodSlot[] _table;
        private static int _mask;
        private static int _shift;
        private static bool _useMixer;

        public static void Initialize((IntPtr ptr, int argc, ScriptMethod<GodotObject> method)[] methods)
        {
            int capacity = methods.Length;
            int maxProbes = int.MaxValue;
            double averageProbes = Tries;

            // calculate optimal size (2^n) for optimal masking
            bool requestMoreSize = true;
            int size = 1;
            int potenz = 0;
            while (size < capacity * 2)
            {
                size <<= 1;
                potenz++;
                _shift = 64 - potenz;
            }

            while (requestMoreSize)
            {
                requestMoreSize = false;

                _mask = size - 1;
                _table = new MethodSlot[size];

                foreach (var method in methods)
                {
                    int currentMaxProbes = 0;
                    int slot = GetSlot(method.ptr);
                    while (_table[slot].Key != IntPtr.Zero)
                    {
                        currentMaxProbes++;
                        slot = (slot + 1) & _mask;
                    }

                    if (requestMoreSize)
                    {
                        break;
                    }

                    _table[slot] = new MethodSlot
                    {
                        Method = (ScriptMethod<GodotObject>)(object)method.method,
                        ArgCount = method.argc,
                        Key = method.ptr
                    };
                }

                if (!requestMoreSize)
                {
                    (averageProbes, maxProbes) = CalculateDiagnostics();

                    if (maxProbes >= 3 ||
                        averageProbes >= 2)
                    {
                        if (size < SizeLimit)
                        {
                            requestMoreSize = true;
                        }
                    }
                }
                if (requestMoreSize)
                {
                    Console.WriteLine("Requesting more capacity...");
                    if (!_useMixer)
                    {
                        _useMixer = true;
                        Console.WriteLine("Mix before enlarging size");
                    }
                    else
                    {
                        Console.WriteLine("More size");
                        _useMixer = false;
                        size <<= 1;
                        potenz++;
                        _shift = 64 - potenz;
                    }

                    continue;
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static bool TryGet(in IntPtr namePtr, in int argCount, out ScriptMethod<T> method)
        {
            int slot = GetSlot(namePtr);

            int i;
            for (i = 0; i < Tries; i++)
            {
                int curr = (slot + i) & _mask;
                IntPtr key = _table[curr].Key;

                if (key == namePtr &&
                    _table[curr].ArgCount == argCount)
                {
                    method = (ScriptMethod<T>)(object)_table[curr].Method;
                    return true;
                }

                if (key == IntPtr.Zero)
                {
                    break;
                }
            }
            if (i == Tries)
            {
                Console.WriteLine("LIMIT REACHED - NOT FOUND");
            }

            method = default;
            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ref readonly ScriptMethod<GodotObject> TryGetFast(in IntPtr namePtr, int argCount)
        {
            int slot = GetSlot(namePtr);

            int i;
            for (i = 0; i < Tries; i++)
            {
                int curr = (slot + i) & _mask;
                IntPtr key = _table[curr].Key;

                if (key == namePtr &&
                    _table[curr].ArgCount == argCount)
                {
                    return ref _table[curr].Method;
                }

                if (key == IntPtr.Zero)
                {
                    break;
                }
            }
            if (i == Tries)
            {
                Console.WriteLine("LIMIT REACHED - NOT FOUND");
            }

            return ref Unsafe.NullRef<ScriptMethod<GodotObject>>();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int GetSlot(nint namePtr)
        {
            ulong x = (ulong)namePtr.ToInt64();
            if (_useMixer)
            {
                // SplitMix64 - extremly fast and cracks open pointer patterns
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9uL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebuL;
                x ^= x >> 31;
            }
            else
            {
                x >>= 3;
            }

            // Fibonacci distributes it perfectly over the length of the array
            return (int)((x * 11400714819323198485uL) >> _shift);
        }

        public static (double AverageProbes, int MaxProbes) CalculateDiagnostics()
        {
            long totalProbes = 0;
            int maxProbes = 0;
            int count = 0;

            for (int i = 0; i < _table.Length; i++)
            {
                if (_table[i].Key != IntPtr.Zero)
                {
                    count++;
                    int idealSlot = GetSlot(_table[i].Key);

                    // calculate distance
                    int distance = (i - idealSlot + _table.Length) & _mask;
                    distance += 1;

                    totalProbes += distance;
                    if (distance > maxProbes) maxProbes = distance;
                }
            }

            var averageProbes = (double)totalProbes / count;
            Console.WriteLine($"Size: {_table.Length}, Items: {count}, Avg Probes: {averageProbes:F2}, Max: {maxProbes}");

            return (averageProbes, maxProbes);
        }
    }
}
