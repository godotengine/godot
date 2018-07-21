using System;
#if REAL_T_IS_DOUBLE
using real_t = System.Double;
#else
using real_t = System.Single;
#endif

// TODO: Add comments describing what this class does. It is not obvious. 

namespace Godot
{
    public static partial class GD
    {
        public static object Bytes2Var(byte[] bytes)
        {
            return NativeCalls.godot_icall_Godot_bytes2var(bytes);
        }

        public static object Convert(object what, int type)
        {
            return NativeCalls.godot_icall_Godot_convert(what, type);
        }

        public static real_t Db2Linear(real_t db)
        {
            return (real_t)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        public static real_t DecTime(real_t value, real_t amount, real_t step)
        {
            real_t sgn = Mathf.Sign(value);
            real_t val = Mathf.Abs(value);
            val -= amount * step;
            if (val < 0)
                val = 0;
            return val * sgn;
        }

        public static FuncRef FuncRef(Object instance, string funcname)
        {
            var ret = new FuncRef();
            ret.SetInstance(instance);
            ret.SetFunction(funcname);
            return ret;
        }

        public static int Hash(object var)
        {
            return NativeCalls.godot_icall_Godot_hash(var);
        }

        public static Object InstanceFromId(int instanceId)
        {
            return NativeCalls.godot_icall_Godot_instance_from_id(instanceId);
        }

        public static real_t Linear2Db(real_t linear)
        {
            return (real_t)(Math.Log(linear) * 8.6858896380650365530225783783321);
        }

        public static Resource Load(string path)
        {
            return ResourceLoader.Load(path);
        }

        public static void Print(params object[] what)
        {
            NativeCalls.godot_icall_Godot_print(what);
        }

        public static void PrintStack()
        {
            Print(System.Environment.StackTrace);
        }

        public static void PrintErr(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printerr(what);
        }

        public static void PrintRaw(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printraw(what);
        }

        public static void PrintS(params object[] what)
        {
            NativeCalls.godot_icall_Godot_prints(what);
        }

        public static void PrintT(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printt(what);
        }

        public static int[] Range(int length)
        {
            var ret = new int[length];

            for (int i = 0; i < length; i++)
            {
                ret[i] = i;
            }

            return ret;
        }

        public static int[] Range(int from, int to)
        {
            if (to < from)
                return new int[0];

            var ret = new int[to - from];

            for (int i = from; i < to; i++)
            {
                ret[i - from] = i;
            }

            return ret;
        }

        public static int[] Range(int from, int to, int increment)
        {
            if (to < from && increment > 0)
                return new int[0];
            if (to > from && increment < 0)
                return new int[0];

            // Calculate count
            int count;

            if (increment > 0)
                count = (to - from - 1) / increment + 1;
            else
                count = (from - to - 1) / -increment + 1;

            var ret = new int[count];

            if (increment > 0)
            {
                int idx = 0;
                for (int i = from; i < to; i += increment)
                {
                    ret[idx++] = i;
                }
            }
            else
            {
                int idx = 0;
                for (int i = from; i > to; i += increment)
                {
                    ret[idx++] = i;
                }
            }

            return ret;
        }

        public static void Seed(int seed)
        {
            NativeCalls.godot_icall_Godot_seed(seed);
        }

        public static string Str(params object[] what)
        {
            return NativeCalls.godot_icall_Godot_str(what);
        }

        public static object Str2Var(string str)
        {
            return NativeCalls.godot_icall_Godot_str2var(str);
        }

        public static bool TypeExists(string type)
        {
            return NativeCalls.godot_icall_Godot_type_exists(type);
        }

        public static byte[] Var2Bytes(object var)
        {
            return NativeCalls.godot_icall_Godot_var2bytes(var);
        }

        public static string Var2Str(object var)
        {
            return NativeCalls.godot_icall_Godot_var2str(var);
        }

        public static WeakRef WeakRef(Object obj)
        {
            return NativeCalls.godot_icall_Godot_weakref(Object.GetPtr(obj));
        }
    }
}
