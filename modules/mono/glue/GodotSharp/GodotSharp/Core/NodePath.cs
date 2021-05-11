using System;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot
{
    public sealed class NodePath : IDisposable
    {
        internal godot_node_path NativeValue;

        ~NodePath()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            // Always dispose `NativeValue` even if disposing is true
            NativeValue.Dispose();
        }

        private NodePath(godot_node_path nativeValueToOwn)
        {
            NativeValue = nativeValueToOwn;
        }

        // Explicit name to make it very clear
        internal static NodePath CreateTakingOwnershipOfDisposableValue(godot_node_path nativeValueToOwn)
            => new NodePath(nativeValueToOwn);

        public NodePath()
        {
        }

        public NodePath(string path)
        {
            if (!string.IsNullOrEmpty(path))
                NativeValue = NativeFuncs.godotsharp_node_path_new_from_string(path);
        }

        public static implicit operator NodePath(string from) => new NodePath(from);

        public static implicit operator string(NodePath from) => from.ToString();

        public override unsafe string ToString()
        {
            if (IsEmpty)
                return string.Empty;

            godot_string dest;
            godot_node_path src = NativeValue;
            NativeFuncs.godotsharp_node_path_as_string(&dest, &src);
            using (dest)
                return Marshaling.mono_string_from_godot(&dest);
        }

        public NodePath GetAsPropertyPath()
        {
            godot_node_path propertyPath = default;
            godot_icall_NodePath_get_as_property_path(ref NativeValue, ref propertyPath);
            return CreateTakingOwnershipOfDisposableValue(propertyPath);
        }

        public string GetConcatenatedSubNames()
        {
            return godot_icall_NodePath_get_concatenated_subnames(ref NativeValue);
        }

        public string GetName(int idx)
        {
            return godot_icall_NodePath_get_name(ref NativeValue, idx);
        }

        public int GetNameCount()
        {
            return godot_icall_NodePath_get_name_count(ref NativeValue);
        }

        public string GetSubName(int idx)
        {
            return godot_icall_NodePath_get_subname(ref NativeValue, idx);
        }

        public int GetSubNameCount()
        {
            return godot_icall_NodePath_get_subname_count(ref NativeValue);
        }

        public bool IsAbsolute()
        {
            return godot_icall_NodePath_is_absolute(ref NativeValue);
        }

        public bool IsEmpty => godot_node_path.IsEmpty(in NativeValue);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void godot_icall_NodePath_get_as_property_path(ref godot_node_path ptr, ref godot_node_path dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_concatenated_subnames(ref godot_node_path ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_name(ref godot_node_path ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern int godot_icall_NodePath_get_name_count(ref godot_node_path ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string godot_icall_NodePath_get_subname(ref godot_node_path ptr, int arg1);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern int godot_icall_NodePath_get_subname_count(ref godot_node_path ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool godot_icall_NodePath_is_absolute(ref godot_node_path ptr);
    }
}
