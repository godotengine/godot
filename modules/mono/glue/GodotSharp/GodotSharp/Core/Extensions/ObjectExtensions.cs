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

        /// <summary>
        /// <para>Connects a <paramref name="signal"/> to a <paramref name="method"/> on a <paramref name="target"/> object. Pass optional <paramref name="binds"/> to the call as an <see cref="Godot.Collections.Array"/> of parameters. These parameters will be passed to the method after any parameter used in the call to <see cref="Godot.Object.EmitSignal"/>. Use <paramref name="flags"/> to set deferred or one-shot connections. See <see cref="Godot.Object.ConnectFlags"/> constants.</para>
        /// <para>A <paramref name="signal"/> can only be connected once to a <paramref name="method"/>. It will throw an error if already connected, unless the signal was connected with <see cref="ConnectFlags.ReferenceCounted"/>. To avoid this, first, use <see cref="Godot.Object.IsConnected"/> to check for existing connections.</para>
        /// <para>If the <paramref name="target"/> is destroyed in the game's lifecycle, the connection will be lost.</para>
        /// </summary>
        /// <param name="signal">The name of the signal to connect.</param>
        /// <param name="target">This gets passed into a new <see cref="Callable"/>.</param>
        /// <param name="method">This gets passed into a new <see cref="Callable"/>.</param>
        /// <param name="binds">These parameters will be passed to the method after any parameter used in the call to <see cref="EmitSignal"/>. If the parameter is null, then the default value is `new Godot.Collections.Array {}`</param>
        /// <param name="flags">A value made of the <see cref="ConnectFlags"/> enum.</param>
        /// <returns>Either <see cref="Error.Ok"/> if successful, or <see cref="Error.InvalidParameter"/> if the connect fails due to an invalid parameter.</returns>
        public Error Connect(StringName signal, Object target, StringName method, Godot.Collections.Array binds = null, uint flags = (uint)0)
            => Connect(signal, new Callable(target, method), binds, flags);
    }
}
