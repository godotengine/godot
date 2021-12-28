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

            NativeFuncs.godotsharp_weakref(GetPtr(obj), out godot_ref weakRef);
            using (weakRef)
            {
                if (weakRef.IsNull)
                    return null;

                return (WeakRef)InteropUtils.UnmanagedGetManaged(weakRef.Reference);
            }
        }
    }
}
