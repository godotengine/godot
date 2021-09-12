using Godot.NativeInterop;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace GodotTools.Internals
{
    public static class Globals
    {
        public static float EditorScale => internal_EditorScale();

        public static unsafe object GlobalDef(string setting, object defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            using godot_variant defaultValueIn = Marshaling.mono_object_to_variant(defaultValue);
            internal_GlobalDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        public static unsafe object EditorDef(string setting, object defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            using godot_variant defaultValueIn = Marshaling.mono_object_to_variant(defaultValue);
            internal_EditorDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        public static unsafe object EditorShortcut(string setting)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            internal_EditorShortcut(settingIn, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        [SuppressMessage("ReSharper", "InconsistentNaming")]
        public static string TTR(this string text)
        {
            using godot_string textIn = Marshaling.mono_string_to_godot(text);
            internal_TTR(textIn, out godot_string dest);
            using (dest)
                return Marshaling.mono_string_from_godot(dest);
        }

        // Internal Calls

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern float internal_EditorScale();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_GlobalDef(in godot_string setting, in godot_variant defaultValue,
            bool restartIfChanged, out godot_variant result);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorDef(in godot_string setting, in godot_variant defaultValue,
            bool restartIfChanged, out godot_variant result);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorShortcut(in godot_string setting, out godot_variant result);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_TTR(in godot_string text, out godot_string dest);
    }
}
