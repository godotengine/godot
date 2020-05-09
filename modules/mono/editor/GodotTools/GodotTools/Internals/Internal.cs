using System;
using System.Runtime.CompilerServices;
using Godot;
using Godot.Collections;
using GodotTools.IdeMessaging.Requests;

namespace GodotTools.Internals
{
    public static class Internal
    {
        public const string CSharpLanguageType = "CSharpScript";
        public const string CSharpLanguageExtension = ".cs";

        public static string UpdateApiAssembliesFromPrebuilt(string config) =>
            internal_UpdateApiAssembliesFromPrebuilt(config);

        public static string FullTemplatesDir =>
            internal_FullTemplatesDir();

        public static string SimplifyGodotPath(this string path) => internal_SimplifyGodotPath(path);

        public static bool IsOsxAppBundleInstalled(string bundleId) => internal_IsOsxAppBundleInstalled(bundleId);

        public static bool GodotIs32Bits() => internal_GodotIs32Bits();

        public static bool GodotIsRealTDouble() => internal_GodotIsRealTDouble();

        public static void GodotMainIteration() => internal_GodotMainIteration();

        public static ulong GetCoreApiHash() => internal_GetCoreApiHash();

        public static ulong GetEditorApiHash() => internal_GetEditorApiHash();

        public static bool IsAssembliesReloadingNeeded() => internal_IsAssembliesReloadingNeeded();

        public static void ReloadAssemblies(bool softReload) => internal_ReloadAssemblies(softReload);

        public static void EditorDebuggerNodeReloadScripts() => internal_EditorDebuggerNodeReloadScripts();

        public static bool ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus = true) =>
            internal_ScriptEditorEdit(resource, line, col, grabFocus);

        public static void EditorNodeShowScriptScreen() => internal_EditorNodeShowScriptScreen();

        public static Dictionary<string, object> GetScriptsMetadataOrNothing() =>
            internal_GetScriptsMetadataOrNothing(typeof(Dictionary<string, object>));

        public static string MonoWindowsInstallRoot => internal_MonoWindowsInstallRoot();

        public static void EditorRunPlay() => internal_EditorRunPlay();

        public static void EditorRunStop() => internal_EditorRunStop();

        public static void ScriptEditorDebugger_ReloadScripts() => internal_ScriptEditorDebugger_ReloadScripts();

        public static string[] CodeCompletionRequest(CodeCompletionRequest.CompletionKind kind, string scriptFile) =>
            internal_CodeCompletionRequest((int)kind, scriptFile);

        #region Internal

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_UpdateApiAssembliesFromPrebuilt(string config);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_FullTemplatesDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_SimplifyGodotPath(this string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsOsxAppBundleInstalled(string bundleId);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_GodotIs32Bits();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_GodotIsRealTDouble();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_GodotMainIteration();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern ulong internal_GetCoreApiHash();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern ulong internal_GetEditorApiHash();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsAssembliesReloadingNeeded();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ReloadAssemblies(bool softReload);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorDebuggerNodeReloadScripts();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorNodeShowScriptScreen();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Dictionary<string, object> internal_GetScriptsMetadataOrNothing(Type dictType);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_MonoWindowsInstallRoot();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorRunPlay();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorRunStop();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_ScriptEditorDebugger_ReloadScripts();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string[] internal_CodeCompletionRequest(int kind, string scriptFile);

        #endregion
    }
}
