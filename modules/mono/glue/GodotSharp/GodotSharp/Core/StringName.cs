using System;
using Godot.NativeInterop;

namespace Godot
{
    public sealed class StringName : IDisposable
    {
        internal godot_string_name.movable NativeValue;

        private WeakReference<IDisposable> _weakReferenceToSelf;

        ~StringName()
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

        private StringName(godot_string_name nativeValueToOwn)
        {
            NativeValue = (godot_string_name.movable)nativeValueToOwn;
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        // Explicit name to make it very clear
        internal static StringName CreateTakingOwnershipOfDisposableValue(godot_string_name nativeValueToOwn)
            => new StringName(nativeValueToOwn);

        public StringName()
        {
        }

        public StringName(string name)
        {
            if (!string.IsNullOrEmpty(name))
            {
                NativeValue = (godot_string_name.movable)NativeFuncs.godotsharp_string_name_new_from_string(name);
                _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
            }
        }

        public static implicit operator StringName(string from) => new StringName(from);

        public static implicit operator string(StringName from) => from?.ToString();

        public override string ToString()
        {
            if (IsEmpty)
                return string.Empty;

            var src = (godot_string_name)NativeValue;
            NativeFuncs.godotsharp_string_name_as_string(out godot_string dest, src);
            using (dest)
                return Marshaling.ConvertStringToManaged(dest);
        }

        public bool IsEmpty => NativeValue.DangerousSelfRef.IsEmpty;

        public static bool operator ==(StringName left, StringName right)
        {
            if (left is null)
                return right is null;
            return left.Equals(right);
        }

        public static bool operator !=(StringName left, StringName right)
        {
            return !(left == right);
        }

        public bool Equals(StringName other)
        {
            if (other is null)
                return false;
            return NativeValue.DangerousSelfRef == other.NativeValue.DangerousSelfRef;
        }

        public static bool operator ==(StringName left, in godot_string_name right)
        {
            if (left is null)
                return right.IsEmpty;
            return left.Equals(right);
        }

        public static bool operator !=(StringName left, in godot_string_name right)
        {
            return !(left == right);
        }

        public static bool operator ==(in godot_string_name left, StringName right)
        {
            return right == left;
        }

        public static bool operator !=(in godot_string_name left, StringName right)
        {
            return !(right == left);
        }

        public bool Equals(in godot_string_name other)
        {
            return NativeValue.DangerousSelfRef == other;
        }

        public override bool Equals(object obj)
        {
            return ReferenceEquals(this, obj) || (obj is StringName other && Equals(other));
        }

        public override int GetHashCode()
        {
            return NativeValue.GetHashCode();
        }
    }
}
