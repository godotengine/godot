using System;
using Godot.NativeInterop;

namespace Godot
{
    public sealed class StringName : IDisposable
    {
        internal godot_string_name.movable NativeValue;

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
        }

        private StringName(godot_string_name nativeValueToOwn)
        {
            NativeValue = (godot_string_name.movable)nativeValueToOwn;
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
                NativeValue = (godot_string_name.movable)NativeFuncs.godotsharp_string_name_new_from_string(name);
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
    }
}
