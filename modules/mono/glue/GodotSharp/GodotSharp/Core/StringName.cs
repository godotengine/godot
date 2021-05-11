using System;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot
{
    public sealed class StringName : IDisposable
    {
        internal godot_string_name NativeValue;

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
            NativeValue.Dispose();
        }

        private StringName(godot_string_name nativeValueToOwn)
        {
            NativeValue = nativeValueToOwn;
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
                NativeValue = NativeFuncs.godotsharp_string_name_new_from_string(name);
        }

        public static implicit operator StringName(string from) => new StringName(from);

        public static implicit operator string(StringName from) => from.ToString();

        public override unsafe string ToString()
        {
            if (IsEmpty)
                return string.Empty;

            godot_string dest;
            godot_string_name src = NativeValue;
            NativeFuncs.godotsharp_string_name_as_string(&dest, &src);
            using (dest)
                return Marshaling.mono_string_from_godot(&dest);
        }

        public bool IsEmpty => godot_string_name.IsEmpty(in NativeValue);
    }
}
