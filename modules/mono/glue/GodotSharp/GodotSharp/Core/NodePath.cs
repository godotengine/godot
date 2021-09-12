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
            NativeFuncs.godotsharp_node_path_get_as_property_path(ref NativeValue, ref propertyPath);
            return CreateTakingOwnershipOfDisposableValue(propertyPath);
        }

        public unsafe string GetConcatenatedSubNames()
        {
            using godot_string subNames = default;
            NativeFuncs.godotsharp_node_path_get_concatenated_subnames(ref NativeValue, &subNames);
            return Marshaling.mono_string_from_godot(&subNames);
        }

        public unsafe string GetName(int idx)
        {
            using godot_string name = default;
            NativeFuncs.godotsharp_node_path_get_name(ref NativeValue, idx, &name);
            return Marshaling.mono_string_from_godot(&name);
        }

        public int GetNameCount()
        {
            return NativeFuncs.godotsharp_node_path_get_name_count(ref NativeValue);
        }

        public unsafe string GetSubName(int idx)
        {
            using godot_string subName = default;
            NativeFuncs.godotsharp_node_path_get_subname(ref NativeValue, idx, &subName);
            return Marshaling.mono_string_from_godot(&subName);
        }

        public int GetSubNameCount()
        {
            return NativeFuncs.godotsharp_node_path_get_subname_count(ref NativeValue);
        }

        public bool IsAbsolute()
        {
            return NativeFuncs.godotsharp_node_path_is_absolute(ref NativeValue);
        }

        public bool IsEmpty => godot_node_path.IsEmpty(in NativeValue);
    }
}
