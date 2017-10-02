using System;

namespace Godot
{
    public static class GD
    {
        /*{GodotGlobalConstants}*/

        public static object bytes2var(byte[] bytes)
        {
            return NativeCalls.godot_icall_Godot_bytes2var(bytes);
        }

        public static object convert(object what, int type)
        {
            return NativeCalls.godot_icall_Godot_convert(what, type);
        }

        public static float db2linear(float db)
        {
            return (float)Math.Exp(db * 0.11512925464970228420089957273422);
        }

        public static float dectime(float value, float amount, float step)
        {
            float sgn = value < 0 ? -1.0f : 1.0f;
            float val = Mathf.abs(value);
            val -= amount * step;
            if (val < 0.0f)
                val = 0.0f;
            return val * sgn;
        }

        public static FuncRef funcref(Object instance, string funcname)
        {
            var ret = new FuncRef();
            ret.SetInstance(instance);
            ret.SetFunction(funcname);
            return ret;
        }

        public static int hash(object var)
        {
            return NativeCalls.godot_icall_Godot_hash(var);
        }

        public static Object instance_from_id(int instance_id)
        {
            return NativeCalls.godot_icall_Godot_instance_from_id(instance_id);
        }

        public static double linear2db(double linear)
        {
            return Math.Log(linear) * 8.6858896380650365530225783783321;
        }

        public static Resource load(string path)
        {
            return ResourceLoader.Load(path);
        }

        public static void print(params object[] what)
        {
            NativeCalls.godot_icall_Godot_print(what);
        }

        public static void print_stack()
        {
            print(System.Environment.StackTrace);
        }

        public static void printerr(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printerr(what);
        }

        public static void printraw(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printraw(what);
        }

        public static void prints(params object[] what)
        {
            NativeCalls.godot_icall_Godot_prints(what);
        }

        public static void printt(params object[] what)
        {
            NativeCalls.godot_icall_Godot_printt(what);
        }

        public static int[] range(int length)
        {
            int[] ret = new int[length];

            for (int i = 0; i < length; i++)
            {
                ret[i] = i;
            }

            return ret;
        }

        public static int[] range(int from, int to)
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

        public static int[] range(int from, int to, int increment)
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

        public static void seed(int seed)
        {
            NativeCalls.godot_icall_Godot_seed(seed);
        }

        public static string str(params object[] what)
        {
            return NativeCalls.godot_icall_Godot_str(what);
        }

        public static object str2var(string str)
        {
            return NativeCalls.godot_icall_Godot_str2var(str);
        }

        public static bool type_exists(string type)
        {
            return NativeCalls.godot_icall_Godot_type_exists(type);
        }

        public static byte[] var2bytes(object var)
        {
            return NativeCalls.godot_icall_Godot_var2bytes(var);
        }

        public static string var2str(object var)
        {
            return NativeCalls.godot_icall_Godot_var2str(var);
        }

        public static WeakRef weakref(Object obj)
        {
            return NativeCalls.godot_icall_Godot_weakref(Object.GetPtr(obj));
        }
    }
}
