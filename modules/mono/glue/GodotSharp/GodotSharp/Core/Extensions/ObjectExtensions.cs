using System;
using System.Runtime.CompilerServices;

namespace Godot
{
    public partial class Object
    {
        /// <summary>
        /// Returns whether <paramref name="instance"/> is a valid object
        /// (e.g. has not been deleted from memory).
        /// </summary>
        /// <param name="instance">The instance to check.</param>
        /// <returns>If the instance is a valid object.</returns>
        public static bool IsInstanceValid(Object instance)
        {
            return instance != null && instance.NativeInstance != IntPtr.Zero;
        }

        /// <summary>
        /// Returns a weak reference to an object, or <see langword="null"/>
        /// if the argument is invalid.
        /// A weak reference to an object is not enough to keep the object alive:
        /// when the only remaining references to a referent are weak references,
        /// garbage collection is free to destroy the referent and reuse its memory
        /// for something else. However, until the object is actually destroyed the
        /// weak reference may return the object even if there are no strong references
        /// to it.
        /// </summary>
        /// <param name="obj">The object.</param>
        /// <returns>
        /// The <see cref="WeakRef"/> reference to the object or <see langword="null"/>.
        /// </returns>
        public static WeakRef WeakRef(Object obj)
        {
            return godot_icall_Object_weakref(GetPtr(obj));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern WeakRef godot_icall_Object_weakref(IntPtr obj);
    }
}
