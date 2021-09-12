using System;
using Godot.NativeInterop;

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
            if (!IsInstanceValid(obj))
                return null;

            using godot_ref weakRef = default;

            unsafe
            {
                NativeFuncs.godotsharp_weakref(GetPtr(obj), &weakRef);
            }

            if (weakRef.IsNull)
                return null;

            return (WeakRef)InteropUtils.UnmanagedGetManaged(weakRef._reference);
        }
    }
}
