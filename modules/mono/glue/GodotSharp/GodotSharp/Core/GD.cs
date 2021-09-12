#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Collections.Generic;
using Godot.NativeInterop;

// TODO: Add comments describing what this class does. It is not obvious.

namespace Godot
{
    public static partial class GD
    {
        public static unsafe object Bytes2Var(byte[] bytes, bool allowObjects = false)
        {
            using var varBytes = Marshaling.mono_array_to_PackedByteArray(bytes);
            using godot_variant ret = default;
            NativeFuncs.godotsharp_bytes2var(&varBytes, allowObjects, &ret);
            return Marshaling.variant_to_mono_object(&ret);
        }

        public static unsafe object Convert(object what, Variant.Type type)
        {
            using var whatVariant = Marshaling.mono_object_to_variant(what);
            using godot_variant ret = default;
            NativeFuncs.godotsharp_convert(&whatVariant, type, &ret);
            return Marshaling.variant_to_mono_object(&ret);
        }

        public static real_t Db2Linear(real_t db)
        {
            return (real_t)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        private static string[] GetPrintParams(object[] parameters)
        {
            if (parameters == null)
            {
                return new[] { "null" };
            }

            return Array.ConvertAll(parameters, x => x?.ToString() ?? "null");
        }

        public static unsafe int Hash(object var)
        {
            using var variant = Marshaling.mono_object_to_variant(var);
            return NativeFuncs.godotsharp_hash(&variant);
        }

        public static Object InstanceFromId(ulong instanceId)
        {
            return InteropUtils.UnmanagedGetManaged(NativeFuncs.godotsharp_instance_from_id(instanceId));
        }

        public static real_t Linear2Db(real_t linear)
        {
            return (real_t)(Math.Log(linear) * 8.6858896380650365530225783783321);
        }

        public static Resource Load(string path)
        {
            return ResourceLoader.Load(path);
        }

        public static T Load<T>(string path) where T : class
        {
            return ResourceLoader.Load<T>(path);
        }

        public static unsafe void PushError(string message)
        {
            using var godotStr = Marshaling.mono_string_to_godot(message);
            NativeFuncs.godotsharp_pusherror(&godotStr);
        }

        public static unsafe void PushWarning(string message)
        {
            using var godotStr = Marshaling.mono_string_to_godot(message);
            NativeFuncs.godotsharp_pushwarning(&godotStr);
        }

        public static unsafe void Print(params object[] what)
        {
            string str = string.Concat(GetPrintParams(what));
            using var godotStr = Marshaling.mono_string_to_godot(str);
            NativeFuncs.godotsharp_print(&godotStr);
        }

        public static void PrintStack()
        {
            Print(System.Environment.StackTrace);
        }

        public static unsafe void PrintErr(params object[] what)
        {
            string str = string.Concat(GetPrintParams(what));
            using var godotStr = Marshaling.mono_string_to_godot(str);
            NativeFuncs.godotsharp_printerr(&godotStr);
        }

        public static unsafe void PrintRaw(params object[] what)
        {
            string str = string.Concat(GetPrintParams(what));
            using var godotStr = Marshaling.mono_string_to_godot(str);
            NativeFuncs.godotsharp_printraw(&godotStr);
        }

        public static unsafe void PrintS(params object[] what)
        {
            string str = string.Join(' ', GetPrintParams(what));
            using var godotStr = Marshaling.mono_string_to_godot(str);
            NativeFuncs.godotsharp_prints(&godotStr);
        }

        public static unsafe void PrintT(params object[] what)
        {
            string str = string.Join('\t', GetPrintParams(what));
            using var godotStr = Marshaling.mono_string_to_godot(str);
            NativeFuncs.godotsharp_printt(&godotStr);
        }

        public static float Randf()
        {
            return NativeFuncs.godotsharp_randf();
        }

        public static uint Randi()
        {
            return NativeFuncs.godotsharp_randi();
        }

        public static void Randomize()
        {
            NativeFuncs.godotsharp_randomize();
        }

        public static double RandRange(double from, double to)
        {
            return NativeFuncs.godotsharp_randf_range(from, to);
        }

        public static int RandRange(int from, int to)
        {
            return NativeFuncs.godotsharp_randi_range(from, to);
        }

        public static uint RandFromSeed(ref ulong seed)
        {
            return NativeFuncs.godotsharp_rand_from_seed(seed, out seed);
        }

        public static IEnumerable<int> Range(int end)
        {
            return Range(0, end, 1);
        }

        public static IEnumerable<int> Range(int start, int end)
        {
            return Range(start, end, 1);
        }

        public static IEnumerable<int> Range(int start, int end, int step)
        {
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

        public static void Seed(ulong seed)
        {
            NativeFuncs.godotsharp_seed(seed);
        }

        public static unsafe string Str(params object[] what)
        {
            using var whatGodotArray = Marshaling.mono_array_to_Array(what);
            using godot_string ret = default;
            NativeFuncs.godotsharp_str(&whatGodotArray, &ret);
            return Marshaling.mono_string_from_godot(&ret);
        }

        public static unsafe object Str2Var(string str)
        {
            using var godotStr = Marshaling.mono_string_to_godot(str);
            using godot_variant ret = default;
            NativeFuncs.godotsharp_str2var(&godotStr, &ret);
            return Marshaling.variant_to_mono_object(&ret);
        }

        public static unsafe byte[] Var2Bytes(object var, bool fullObjects = false)
        {
            using var variant = Marshaling.mono_object_to_variant(var);
            using godot_packed_byte_array varBytes = default;
            NativeFuncs.godotsharp_var2bytes(&variant, fullObjects, &varBytes);
            using (varBytes)
                return Marshaling.PackedByteArray_to_mono_array(&varBytes);
        }

        public static unsafe string Var2Str(object var)
        {
            using var variant = Marshaling.mono_object_to_variant(var);
            using godot_string ret = default;
            NativeFuncs.godotsharp_var2str(&variant, &ret);
            return Marshaling.mono_string_from_godot(&ret);
        }

        public static Variant.Type TypeToVariantType(Type type)
        {
            return Marshaling.managed_to_variant_type(type, out bool _);
        }
    }
}
