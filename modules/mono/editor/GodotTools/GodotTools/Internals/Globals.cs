using Godot.NativeInterop;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace GodotTools.Internals
{
    public static class Globals
    {
        public static float EditorScale => Internal.godot_icall_Globals_EditorScale();

        public static unsafe object GlobalDef(string setting, object defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            using godot_variant defaultValueIn = Marshaling.mono_object_to_variant(defaultValue);
            Internal.godot_icall_Globals_GlobalDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        public static unsafe object EditorDef(string setting, object defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            using godot_variant defaultValueIn = Marshaling.mono_object_to_variant(defaultValue);
            Internal.godot_icall_Globals_EditorDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        public static unsafe object EditorShortcut(string setting)
        {
            using godot_string settingIn = Marshaling.mono_string_to_godot(setting);
            Internal.godot_icall_Globals_EditorShortcut(settingIn, out godot_variant result);
            using (result)
                return Marshaling.variant_to_mono_object(&result);
        }

        [SuppressMessage("ReSharper", "InconsistentNaming")]
        public static string TTR(this string text)
        {
            using godot_string textIn = Marshaling.mono_string_to_godot(text);
            Internal.godot_icall_Globals_TTR(textIn, out godot_string dest);
            using (dest)
                return Marshaling.mono_string_from_godot(dest);
        }
    }
}
