using System;
using System.Runtime.CompilerServices;

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
            return godot_icall_Object_weakref(Object.GetPtr(obj));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static WeakRef godot_icall_Object_weakref(IntPtr obj);
    }
}
