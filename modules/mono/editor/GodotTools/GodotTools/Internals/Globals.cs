using Godot;
using Godot.NativeInterop;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace GodotTools.Internals
{
    public static class Globals
    {
        public static float EditorScale => Internal.godot_icall_Globals_EditorScale();

        // ReSharper disable once UnusedMethodReturnValue.Global
        public static Variant GlobalDef(string setting, Variant defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_variant defaultValueIn = defaultValue.CopyNativeVariant();
            Internal.godot_icall_Globals_GlobalDef(settingIn, defaultValueIn, restartIfChanged,
                out godot_variant result);
            return Variant.CreateTakingOwnershipOfDisposableValue(result);
        }

        // ReSharper disable once UnusedMethodReturnValue.Global
        public static Variant EditorDef(string setting, Variant defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_variant defaultValueIn = defaultValue.CopyNativeVariant();
            Internal.godot_icall_Globals_EditorDef(settingIn, defaultValueIn, restartIfChanged,
                out godot_variant result);
            return Variant.CreateTakingOwnershipOfDisposableValue(result);
        }

        public static Shortcut EditorDefShortcut(string setting, string name, Key keycode = Key.None, bool physical = false)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_string nameIn = Marshaling.ConvertStringToNative(name);
            Internal.godot_icall_Globals_EditorDefShortcut(settingIn, nameIn, keycode, physical.ToGodotBool(), out godot_variant result);
            return (Shortcut)Variant.CreateTakingOwnershipOfDisposableValue(result);
        }

        public static Shortcut EditorGetShortcut(string setting)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            Internal.godot_icall_Globals_EditorGetShortcut(settingIn, out godot_variant result);
            return (Shortcut)Variant.CreateTakingOwnershipOfDisposableValue(result);
        }

        public static void EditorShortcutOverride(string setting, string feature, Key keycode = Key.None, bool physical = false)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_string featureIn = Marshaling.ConvertStringToNative(feature);
            Internal.godot_icall_Globals_EditorShortcutOverride(settingIn, featureIn, keycode, physical.ToGodotBool());
        }

        [SuppressMessage("ReSharper", "InconsistentNaming")]
        public static string TTR(this string text)
        {
            using godot_string textIn = Marshaling.ConvertStringToNative(text);
            Internal.godot_icall_Globals_TTR(textIn, out godot_string dest);
            using (dest)
                return Marshaling.ConvertStringToManaged(dest);
        }
    }
}
