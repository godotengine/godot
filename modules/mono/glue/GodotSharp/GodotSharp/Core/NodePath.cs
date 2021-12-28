using System;
using Godot.NativeInterop;

namespace Godot
{
    public sealed class NodePath : IDisposable
    {
        internal godot_node_path.movable NativeValue;

        private WeakReference<IDisposable> _weakReferenceToSelf;

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
            NativeValue.DangerousSelfRef.Dispose();
            DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
        }

        private NodePath(godot_node_path nativeValueToOwn)
        {
            NativeValue = (godot_node_path.movable)nativeValueToOwn;
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
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
            {
                NativeValue = (godot_node_path.movable)NativeFuncs.godotsharp_node_path_new_from_string(path);
                _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
            }
        }

        public static implicit operator NodePath(string from) => new NodePath(from);

        public static implicit operator string(NodePath from) => from?.ToString();

        public override string ToString()
        {
            if (IsEmpty)
                return string.Empty;

            var src = (godot_node_path)NativeValue;
            NativeFuncs.godotsharp_node_path_as_string(out godot_string dest, src);
            using (dest)
                return Marshaling.ConvertStringToManaged(dest);
        }

        public NodePath GetAsPropertyPath()
        {
            godot_node_path propertyPath = default;
            var self = (godot_node_path)NativeValue;
            NativeFuncs.godotsharp_node_path_get_as_property_path(self, ref propertyPath);
            return CreateTakingOwnershipOfDisposableValue(propertyPath);
        }

        public string GetConcatenatedSubNames()
        {
            var self = (godot_node_path)NativeValue;
            NativeFuncs.godotsharp_node_path_get_concatenated_subnames(self, out godot_string subNames);
            using (subNames)
                return Marshaling.ConvertStringToManaged(subNames);
        }

        public string GetName(int idx)
        {
            var self = (godot_node_path)NativeValue;
            NativeFuncs.godotsharp_node_path_get_name(self, idx, out godot_string name);
            using (name)
                return Marshaling.ConvertStringToManaged(name);
        }

        public int GetNameCount()
        {
            var self = (godot_node_path)NativeValue;
            return NativeFuncs.godotsharp_node_path_get_name_count(self);
        }

        public string GetSubName(int idx)
        {
            var self = (godot_node_path)NativeValue;
            NativeFuncs.godotsharp_node_path_get_subname(self, idx, out godot_string subName);
            using (subName)
                return Marshaling.ConvertStringToManaged(subName);
        }

        public int GetSubNameCount()
        {
            var self = (godot_node_path)NativeValue;
            return NativeFuncs.godotsharp_node_path_get_subname_count(self);
        }

        public bool IsAbsolute()
        {
            var self = (godot_node_path)NativeValue;
            return NativeFuncs.godotsharp_node_path_is_absolute(self).ToBool();
        }

        public bool IsEmpty => NativeValue.DangerousSelfRef.IsEmpty;
    }
}
