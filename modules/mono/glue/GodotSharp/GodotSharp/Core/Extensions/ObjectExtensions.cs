using System;
using Godot.NativeInterop;

namespace Godot
{
    public partial class Object
    {
        /// <summary>
        /// Returns the <see cref="Object"/> that corresponds to <paramref name="instanceId"/>.
        /// All Objects have a unique instance ID. See also <see cref="GetInstanceId"/>.
        /// </summary>
        /// <example>
        /// <code>
        /// public partial class MyNode : Node
        /// {
        ///     public string Foo { get; set; } = "bar";
        ///
        ///     public override void _Ready()
        ///     {
        ///         ulong id = GetInstanceId();
        ///         var inst = (MyNode)InstanceFromId(Id);
        ///         GD.Print(inst.Foo); // Prints bar
        ///     }
        /// }
        /// </code>
        /// </example>
        /// <param name="instanceId">Instance ID of the Object to retrieve.</param>
        /// <returns>The <see cref="Object"/> instance.</returns>
        public static Object InstanceFromId(ulong instanceId)
        {
            return InteropUtils.UnmanagedGetManaged(NativeFuncs.godotsharp_instance_from_id(instanceId));
        }

        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="Object"/> that corresponds
        /// to <paramref name="id"/> is a valid object (e.g. has not been deleted from
        /// memory). All Objects have a unique instance ID.
        /// </summary>
        /// <param name="id">The Object ID to check.</param>
        /// <returns>If the instance with the given ID is a valid object.</returns>
        public static bool IsInstanceIdValid(ulong id)
        {
            return IsInstanceValid(InstanceFromId(id));
        }

        /// <summary>
        /// Returns <see langword="true"/> if <paramref name="instance"/> is a
        /// valid <see cref="Object"/> (e.g. has not been deleted from memory).
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
