using Godot.NativeInterop;
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace Godot.Bridge
{
    public static class ScriptMethodCache<T>
        where T : GodotObject
    {
        private const int Tries = 16;

        private static IntPtr[] _keys;
        private static int[] _argCounts;
        private static ScriptMethod<GodotObject>[] _methods;
        private static int _mask;

        public static void Initialize((IntPtr ptr, int argc, ScriptMethod<GodotObject> method)[] methods)
        {
            int capacity = methods.Length;

            // calculate optimal size (2^n) for optimal masking
            int size = 1;
            while (size < capacity * 2) size <<= 1;

            _mask = size - 1;
            _keys = new IntPtr[size];
            _argCounts = new int[size];
            _methods = new ScriptMethod<GodotObject>[size];

            foreach (var method in methods)
            {
                int slot = GetSlot(method.ptr);
                while (_keys[slot] != IntPtr.Zero)
                {
                    slot = (slot + 1) & _mask;
                }

                _keys[slot] = method.ptr;
                _argCounts[slot] = method.argc;
                _methods[slot] = (ScriptMethod<GodotObject>)(object)method.method;
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
                IntPtr key = _keys[curr];

                if (key == namePtr &&
                    _argCounts[curr] == argCount)
                {
                    method = (ScriptMethod<T>)(object)_methods[curr];
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
                IntPtr key = _keys[curr];

                if (key == namePtr &&
                    _argCounts[curr] == argCount)
                {
                    return ref _methods[curr];
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int GetSlot(nint namePtr)
        {
            //return (int)((long)namePtr >> 3) & _mask;
            // Fibonacci Hashing
            int slot = (int)(((ulong)namePtr.ToInt64() * 11400714819323198485uL) >> 32) & _mask;

            return slot;
        }

        public static void PrintDiagnostics()
        {
            //var table = _state.CurrentTable;
            long totalProbes = 0;
            int maxProbes = 0;
            int count = 0;

            for (int i = 0; i < _keys.Length; i++)
            {
                if (_keys[i] != IntPtr.Zero)
                {
                    count++;
                    int idealSlot = GetSlot(_keys[i]);

                    // Distanz berechnen (Modular Arithmetic)
                    int distance = (i - idealSlot + _keys.Length) & _mask;
                    distance += 1; // 1-basiert für Probes

                    totalProbes += distance;
                    if (distance > maxProbes) maxProbes = distance;
                }
            }

            Console.WriteLine($"Items: {count}, Avg Probes: {(double)totalProbes / count:F2}, Max: {maxProbes}");
        }
    }
}
