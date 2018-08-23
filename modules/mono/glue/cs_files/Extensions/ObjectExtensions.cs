using System;

namespace Godot
{
    public partial class Object
    {
        public static bool IsInstanceValid(Object instance)
        {
            return instance != null && instance.NativeInstance != IntPtr.Zero;
        }

        public static WeakRef WeakRef(Object obj)
        {
            return NativeCalls.godot_icall_Godot_weakref(Object.GetPtr(obj));
        }
    }
}
