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

        public static Variant EditorShortcut(string setting)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            Internal.godot_icall_Globals_EditorShortcut(settingIn, out godot_variant result);
            return Variant.CreateTakingOwnershipOfDisposableValue(result);
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
