using System;
using System.Runtime.CompilerServices;
using Godot.Collections;

namespace Godot
{
    public partial class SceneTree
    {
        public Array<T> GetNodesInGroup<T>(StringName group) where T : class
        {
            return new Array<T>(godot_icall_SceneTree_get_nodes_in_group_Generic(Object.GetPtr(this), StringName.GetPtr(group), typeof(T)));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_SceneTree_get_nodes_in_group_Generic(IntPtr obj, IntPtr group, Type elemType);
    }
}
