using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object
    {
        public static bool IsInstanceValid(Object instance) =>
            instance != null && instance.NativeInstance != IntPtr.Zero;

        public static WeakRef WeakRef(Object obj) => godot_icall_Object_weakref(GetPtr(obj));

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern WeakRef godot_icall_Object_weakref(IntPtr obj);
    }
}
