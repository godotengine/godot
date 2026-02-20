using Godot.NativeInterop;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;

namespace Godot.Bridge
{
    //public class ScriptMethodCache<T> : ScriptCache<T, ScriptMethod<GodotObject>>
    //    where T : GodotObject
    //{

    //}

    public delegate godot_variant PropertyAccessMethod<T>(T godotObject, scoped in godot_variant value)
        where T : GodotObject;

    public delegate void SignalScriptMethod<T>(T godotObject, scoped in NativeVariantPtrArgs args)
        where T : GodotObject;

    //public class ScriptPropertyCache<T> : ScriptCache<T, ScriptMethod<GodotObject>>
    //    where T : GodotObject
    //{

    //}

#pragma warning disable CA1000 // Do not declare static members on generic types

    public class ScriptCache<T, TMethod>
        where T : GodotObject
    {
        private const int MaxProbes = 4;
        private const double MaxAverageProbes = 1.5;
        private const int SizeLimit = 8192;

        private static IntPtr[] _keys;
        private static byte[] _argCounts;
        private static TMethod[] _methods;

        private static int _mask;
        private static int _shift;
        private static bool _useMixer;
        private static int _finalMaxProbes;

        private static List<string> _mostUsedMethods = [
            GodotObject.MethodName.Notification,
            Node.MethodName._Process,
            Node.MethodName._PhysicsProcess,
            CanvasItem.MethodName._Draw,
            Node.MethodName._Input,
            Node.MethodName._UnhandledInput,
            Control.MethodName._GuiInput,
            Node.MethodName._Ready,
            Node.MethodName._EnterTree,
            Node.MethodName._ExitTree,
        ];

        public static void Initialize((StringName NamePtr, int ArgCount, TMethod Method)[] methods)
        {
            int capacity = methods.Length;
            int maxProbes = int.MaxValue;
            double averageProbes = MaxProbes;

            var sortedMethods = SortMethodsOnExpectedCallCountForImprovedPerformance(methods);

            // calculate optimal size (2^n) for optimal masking
            bool requestMoreSize = true;
            int size = 1;
            int power = 0;
            while (size < capacity * 2)
            {
                size <<= 1;
                power++;
                _shift = 64 - power;
            }

            while (requestMoreSize)
            {
                requestMoreSize = false;

                _mask = size - 1;
                _keys = new IntPtr[size];
                _argCounts = new byte[size];
                _methods = new TMethod[size];

                foreach (var method in methods)
                {
                    int currentMaxProbes = 0;
                    int slot = GetSlot(method.NamePtr.NativeValue._data);
                    while (_keys[slot] != IntPtr.Zero)
                    {
                        currentMaxProbes++;
                        slot = (slot + 1) & _mask;
                        if (currentMaxProbes > MaxProbes &&
                            CanRequestMoreSize(size))
                        {
                            requestMoreSize = true;
                            break;
                        }
                    }

                    if (requestMoreSize)
                    {
                        break;
                    }

                    _keys[slot] = method.NamePtr.NativeValue._data;
                    _argCounts[slot] = (byte)method.ArgCount;
                    _methods[slot] = (TMethod)(object)method.Method;
                }

                if (!requestMoreSize)
                {
                    // check if amount of probes is still reasonable fast
                    (averageProbes, maxProbes, _) = CalculateDiagnostics();

                    if (maxProbes > MaxProbes ||
                        averageProbes > MaxAverageProbes)
                    {
                        if (CanRequestMoreSize(size))
                        {
                            requestMoreSize = true;
                        }
                    }
                }

                if (requestMoreSize)
                {
                    // requesting more capacity
                    if (!_useMixer)
                    {
                        // mix before enlarging size
                        _useMixer = true;
                    }
                    else
                    {
                        // increase size
                        _useMixer = false;
                        size <<= 1;
                        power++;
                        _shift = 64 - power;
                    }

                    continue;
                }
            }

            var (finalAverageProbes, finalMaxProbes, FinalCount) = CalculateDiagnostics();
            //GD.Print($"Final: Size: {_keys.Length}, Items: {FinalCount}, Avg Probes: {finalAverageProbes:F2}, Max: {finalMaxProbes}\n");
            _finalMaxProbes = finalMaxProbes; // this is the real worst count of probes
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool CanRequestMoreSize(int size)
        {
            return size < SizeLimit;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public static ref readonly TMethod GetOrNullRef(scoped in IntPtr namePtr, int argCount)
        {
            int slot = GetSlot(namePtr);

            int i;
            for (i = 0; i < _finalMaxProbes; i++)
            {
                int currentSlot = (slot + i) & _mask;
                IntPtr key = _keys[currentSlot];

                if (key == namePtr &&
                    _argCounts[currentSlot] == argCount)
                {
                    return ref _methods[currentSlot];
                }

                if (key == IntPtr.Zero)
                {
                    break;
                }
            }

            return ref Unsafe.NullRef<TMethod>();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static int GetSlot(nint namePtr)
        {
            ulong x = (ulong)namePtr.ToInt64();
            if (_useMixer)
            {
                // SplitMix64 - extremely fast and cracks open pointer patterns
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

        public static (double AverageProbes, int MaxProbes, int Count) CalculateDiagnostics()
        {
            var totalProbes = 0L;
            var maxProbes = 0;
            var count = 0;

            for (int i = 0; i < _keys.Length; i++)
            {
                var searchKey = _keys[i];
                if (searchKey != IntPtr.Zero)
                {
                    count++;

                    var idealSlot = GetSlot(_keys[i]);
                    var searchArgCount = _argCounts[i];
                    int probesNeeded = 1;

                    int currentSlot = idealSlot;
                    while (true)
                    {
                        if (_keys[currentSlot] == searchKey &&
                            _argCounts[currentSlot] == searchArgCount)
                        {
                            break;
                        }

                        currentSlot = (currentSlot + 1) & _mask;
                        probesNeeded++;

                        // safety-check if table is corrupt somehow
                        if (probesNeeded > _keys.Length)
                        {
                            break;
                        }
                    }

                    // calculate distance
                    totalProbes += probesNeeded;
                    if (probesNeeded > maxProbes)
                    {
                        maxProbes = probesNeeded;
                    }
                }
            }

            var averageProbes = count == 0 ? 0 : (double)totalProbes / count;

            //GD.Print($"Size: {_keys.Length}, Items: {count}, Avg Probes: {averageProbes:F2}, Max: {maxProbes}");

            return (averageProbes, maxProbes, count);
        }

        private static (StringName NamePtr, int ArgCount, TMethod Method)[] SortMethodsOnExpectedCallCountForImprovedPerformance((StringName NamePtr, int ArgCount, TMethod Method)[] methods)
        {
            return methods
                .Select(method => new
                {
                    method,
                    Name = StringHelpers.ConvertStringNameToString((godot_string_name)method.NamePtr.NativeValue)
                })
                .OrderBy(x =>
                {
                    var index = _mostUsedMethods.IndexOf(x.Name);
                    if (index > -1)
                    {
                        return index;
                    }

                    return 999;
                })
                .Select(x => x.method)
                .ToArray();
        }

        private class StringHelpers
        {
            internal static string ConvertStringNameToString(in godot_string_name name)
            {
                godot_string godotString;
                NativeFuncs.godotsharp_string_name_as_string(out godotString, in name);

                using (godotString)
                {
                    var managedString = Marshaling.ConvertStringToManaged(godotString);

                    return managedString;
                }
            }
        }
    }

#pragma warning restore CA1000 // Do not declare static members on generic types
}
