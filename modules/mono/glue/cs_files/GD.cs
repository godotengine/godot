using System;

namespace Godot
{
    public static class GD
    {
        /*{GodotGlobalConstants}*/

        public static object Bytes2Var(byte[] bytes)
        {
            return NativeCalls.godot_icall_Godot_bytes2var(bytes);
        }

        public static object Convert(object what, int type)
        {
            return NativeCalls.godot_icall_Godot_convert(what, type);
        }

        public static float Db2Linear(float db)
        {
            return (float)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        public static float Dectime(float value, float amount, float step)
        {
            float sgn = value < 0 ? -1.0f : 1.0f;
            float val = Mathf.Abs(value);
            val -= amount * step;
            if (val < 0.0f)
                val = 0.0f;
            return val * sgn;
        }

        public static FuncRef Funcref(Object instance, string funcname)
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

        public static double Linear2Db(double linear)
        {
            return Math.Log(linear) * 8.6858896380650365530225783783321;
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

        public static void Printerr(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printerr(what);
        }

        public static void Printraw(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printraw(what);
        }

        public static void Prints(params object[] what)
        {
            NativeCalls.godot_icall_Godot_prints(what);
        }

        public static void Printt(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printt(what);
        }

        public static int[] Range(int length)
        {
            int[] ret = new int[length];

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

            int[] ret = new int[to - from];

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
            int count = 0;

            if (increment > 0)
                count = ((to - from - 1) / increment) + 1;
            else
                count = ((from - to - 1) / -increment) + 1;

            int[] ret = new int[count];

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

        public static WeakRef Weakref(Object obj)
        {
            return NativeCalls.godot_icall_Godot_weakref(Object.GetPtr(obj));
        }
    }
}
