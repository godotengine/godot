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
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_variant defaultValueIn = Marshaling.ConvertManagedObjectToVariant(defaultValue);
            Internal.godot_icall_Globals_GlobalDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.ConvertVariantToManagedObject(result);
        }

        public static unsafe object EditorDef(string setting, object defaultValue, bool restartIfChanged = false)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            using godot_variant defaultValueIn = Marshaling.ConvertManagedObjectToVariant(defaultValue);
            Internal.godot_icall_Globals_EditorDef(settingIn, defaultValueIn, restartIfChanged, out godot_variant result);
            using (result)
                return Marshaling.ConvertVariantToManagedObject(result);
        }

        public static object EditorShortcut(string setting)
        {
            using godot_string settingIn = Marshaling.ConvertStringToNative(setting);
            Internal.godot_icall_Globals_EditorShortcut(settingIn, out godot_variant result);
            using (result)
                return Marshaling.ConvertVariantToManagedObject(result);
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
