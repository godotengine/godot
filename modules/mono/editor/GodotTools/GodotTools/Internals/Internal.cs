using System;
using System.Runtime.CompilerServices;
using Godot;
using Godot.NativeInterop;
using GodotTools.IdeMessaging.Requests;

namespace GodotTools.Internals
{
    public static class Internal
    {
        public const string CSharpLanguageType = "CSharpScript";
        public const string CSharpLanguageExtension = ".cs";

        public static string FullExportTemplatesDir
        {
            get
            {
                internal_FullExportTemplatesDir(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static string SimplifyGodotPath(this string path) => Godot.StringExtensions.SimplifyPath(path);

        public static bool IsMacOSAppBundleInstalled(string bundleId)
        {
            using godot_string bundleIdIn = Marshaling.mono_string_to_godot(bundleId);
            return internal_IsMacOSAppBundleInstalled(bundleIdIn);
        }

        public static bool GodotIs32Bits() => internal_GodotIs32Bits();

        public static bool GodotIsRealTDouble() => internal_GodotIsRealTDouble();

        public static void GodotMainIteration() => internal_GodotMainIteration();

        public static bool IsAssembliesReloadingNeeded() => internal_IsAssembliesReloadingNeeded();

        public static void ReloadAssemblies(bool softReload) => internal_ReloadAssemblies(softReload);

        public static void EditorDebuggerNodeReloadScripts() => internal_EditorDebuggerNodeReloadScripts();

        public static bool ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus = true) =>
            internal_ScriptEditorEdit(resource.NativeInstance, line, col, grabFocus);

        public static void EditorNodeShowScriptScreen() => internal_EditorNodeShowScriptScreen();

        public static string MonoWindowsInstallRoot
        {
            get
            {
                internal_MonoWindowsInstallRoot(out godot_string dest);
                using (dest)
                    return Marshaling.mono_string_from_godot(dest);
            }
        }

        public static void EditorRunPlay() => internal_EditorRunPlay();

        public static void EditorRunStop() => internal_EditorRunStop();

        public static void ScriptEditorDebugger_ReloadScripts() => internal_ScriptEditorDebugger_ReloadScripts();

        public static unsafe string[] CodeCompletionRequest(CodeCompletionRequest.CompletionKind kind,
            string scriptFile)
        {
            using godot_string scriptFileIn = Marshaling.mono_string_to_godot(scriptFile);
            internal_CodeCompletionRequest((int)kind, scriptFileIn, out godot_packed_string_array res);
            using (res)
                return Marshaling.PackedStringArray_to_mono_array(&res);
        }

        #region Internal

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_FullExportTemplatesDir(out godot_string dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsMacOSAppBundleInstalled(in godot_string bundleId);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_GodotIs32Bits();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_GodotIsRealTDouble();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_GodotMainIteration();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsAssembliesReloadingNeeded();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ReloadAssemblies(bool softReload);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorDebuggerNodeReloadScripts();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_ScriptEditorEdit(IntPtr resource, int line, int col, bool grabFocus);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorNodeShowScriptScreen();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_MonoWindowsInstallRoot(out godot_string dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorRunPlay();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorRunStop();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ScriptEditorDebugger_ReloadScripts();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_CodeCompletionRequest(int kind, in godot_string scriptFile,
            out godot_packed_string_array res);

        #endregion
    }
}
