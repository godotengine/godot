#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

// TODO: Add comments describing what this class does. It is not obvious.

namespace Godot
{
    public static partial class GD
    {
        public static object Bytes2Var(byte[] bytes, bool allowObjects = false)
        {
            return godot_icall_GD_bytes2var(bytes, allowObjects);
        }

        public static object Convert(object what, Variant.Type type)
        {
            return godot_icall_GD_convert(what, type);
        }

        public static real_t Db2Linear(real_t db)
        {
            return (real_t)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        private static object[] GetPrintParams(object[] parameters)
        {
            if (parameters == null)
            {
                return new[] { "null" };
            }

            return Array.ConvertAll(parameters, x => x?.ToString() ?? "null");
        }

        public static int Hash(object var)
        {
            return godot_icall_GD_hash(var);
        }

        public static Object InstanceFromId(ulong instanceId)
        {
            return godot_icall_GD_instance_from_id(instanceId);
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

        public static void PushError(string message)
        {
            godot_icall_GD_pusherror(message);
        }

        public static void PushWarning(string message)
        {
            godot_icall_GD_pushwarning(message);
        }

        public static void Print(params object[] what)
        {
            godot_icall_GD_print(GetPrintParams(what));
        }

        public static void PrintStack()
        {
            Print(System.Environment.StackTrace);
        }

        public static void PrintErr(params object[] what)
        {
            godot_icall_GD_printerr(GetPrintParams(what));
        }

        public static void PrintRaw(params object[] what)
        {
            godot_icall_GD_printraw(GetPrintParams(what));
        }

        public static void PrintS(params object[] what)
        {
            godot_icall_GD_prints(GetPrintParams(what));
        }

        public static void PrintT(params object[] what)
        {
            godot_icall_GD_printt(GetPrintParams(what));
        }

        public static float Randf()
        {
            return godot_icall_GD_randf();
        }

        public static uint Randi()
        {
            return godot_icall_GD_randi();
        }

        public static void Randomize()
        {
            godot_icall_GD_randomize();
        }

        public static double RandRange(double from, double to)
        {
            return godot_icall_GD_randf_range(from, to);
        }

        public static int RandRange(int from, int to)
        {
            return godot_icall_GD_randi_range(from, to);
        }

        public static uint RandFromSeed(ref ulong seed)
        {
            return godot_icall_GD_rand_seed(seed, out seed);
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
            godot_icall_GD_seed(seed);
        }

        public static string Str(params object[] what)
        {
            return godot_icall_GD_str(what);
        }

        public static object Str2Var(string str)
        {
            return godot_icall_GD_str2var(str);
        }

        public static bool TypeExists(StringName type)
        {
            return godot_icall_GD_type_exists(StringName.GetPtr(type));
        }

        public static byte[] Var2Bytes(object var, bool fullObjects = false)
        {
            return godot_icall_GD_var2bytes(var, fullObjects);
        }

        public static string Var2Str(object var)
        {
            return godot_icall_GD_var2str(var);
        }

        public static Variant.Type TypeToVariantType(Type type)
        {
            return godot_icall_TypeToVariantType(type);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_GD_bytes2var(byte[] bytes, bool allowObjects);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_GD_convert(object what, Variant.Type type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_GD_hash(object var);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static Object godot_icall_GD_instance_from_id(ulong instanceId);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_print(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_printerr(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_printraw(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_prints(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_printt(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static float godot_icall_GD_randf();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static uint godot_icall_GD_randi();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_randomize();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static double godot_icall_GD_randf_range(double from, double to);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_GD_randi_range(int from, int to);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static uint godot_icall_GD_rand_seed(ulong seed, out ulong newSeed);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_seed(ulong seed);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_GD_str(object[] what);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_GD_str2var(string str);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_GD_type_exists(IntPtr type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static byte[] godot_icall_GD_var2bytes(object what, bool fullObjects);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static string godot_icall_GD_var2str(object var);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_pusherror(string type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_GD_pushwarning(string type);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Variant.Type godot_icall_TypeToVariantType(Type type);
    }
}
