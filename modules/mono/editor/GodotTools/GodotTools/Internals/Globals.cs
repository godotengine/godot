using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;

namespace GodotTools.Internals
{
    public static class Globals
    {
        public static float EditorScale => internal_EditorScale();

        public static object GlobalDef(string setting, object defaultValue, bool restartIfChanged = false) =>
            internal_GlobalDef(setting, defaultValue, restartIfChanged);

        public static object EditorDef(string setting, object defaultValue, bool restartIfChanged = false) =>
            internal_EditorDef(setting, defaultValue, restartIfChanged);

        public static object EditorShortcut(string setting) =>
            internal_EditorShortcut(setting);

        [SuppressMessage("ReSharper", "InconsistentNaming")]
        public static string TTR(this string text) => internal_TTR(text);

        // Internal Calls

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern float internal_EditorScale();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern object internal_GlobalDef(string setting, object defaultValue, bool restartIfChanged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern object internal_EditorDef(string setting, object defaultValue, bool restartIfChanged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern object internal_EditorShortcut(string setting);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_TTR(string text);
    }
}
