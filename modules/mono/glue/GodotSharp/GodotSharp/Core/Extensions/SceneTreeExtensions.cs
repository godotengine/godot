using System;
using System.Runtime.CompilerServices;
using Godot.Collections;
using Godot.NativeInterop;

namespace Godot
{
    public partial class SceneTree
    {
        public Array<T> GetNodesInGroup<T>(StringName group) where T : class
        {
            godot_array array;
            godot_icall_SceneTree_get_nodes_in_group_Generic(GetPtr(this), ref group.NativeValue, typeof(T), out array);
            return Array<T>.CreateTakingOwnershipOfDisposableValue(array);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_SceneTree_get_nodes_in_group_Generic(IntPtr obj, ref godot_string_name group, Type elemType, out godot_array dest);
    }
}
