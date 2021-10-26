using System;
using System.Runtime.CompilerServices;
using Godot.Collections;

namespace Godot
{
    public partial class SceneTree
    {
        /// <summary>
        /// Returns a list of all nodes assigned to the given <paramref name="group"/>.
        /// </summary>
        /// <typeparam name="T">The type to cast to. Should be a descendant of <see cref="Node"/>.</typeparam>
        public Array<T> GetNodesInGroup<T>(StringName group) where T : class
        {
            return new Array<T>(godot_icall_SceneTree_get_nodes_in_group_Generic(GetPtr(this), StringName.GetPtr(group), typeof(T)));
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_SceneTree_get_nodes_in_group_Generic(IntPtr obj, IntPtr group, Type elemType);
    }
}
