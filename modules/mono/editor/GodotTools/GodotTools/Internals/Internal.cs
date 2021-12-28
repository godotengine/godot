using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Godot;
using Godot.NativeInterop;
using GodotTools.IdeMessaging.Requests;

namespace GodotTools.Internals
{
    internal static class Internal
    {
        public const string CSharpLanguageType = "CSharpScript";
        public const string CSharpLanguageExtension = ".cs";

        public static string FullTemplatesDir
        {
            get
            {
                godot_icall_Internal_FullTemplatesDir(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        public static string SimplifyGodotPath(this string path) => Godot.StringExtensions.SimplifyPath(path);

        public static bool IsOsxAppBundleInstalled(string bundleId)
        {
            using godot_string bundleIdIn = Marshaling.ConvertStringToNative(bundleId);
            return godot_icall_Internal_IsOsxAppBundleInstalled(bundleIdIn);
        }

        public static bool GodotIs32Bits() => godot_icall_Internal_GodotIs32Bits();

        public static bool GodotIsRealTDouble() => godot_icall_Internal_GodotIsRealTDouble();

        public static void GodotMainIteration() => godot_icall_Internal_GodotMainIteration();

        public static bool IsAssembliesReloadingNeeded() => godot_icall_Internal_IsAssembliesReloadingNeeded();

        public static void ReloadAssemblies(bool softReload) => godot_icall_Internal_ReloadAssemblies(softReload);

        public static void EditorDebuggerNodeReloadScripts() => godot_icall_Internal_EditorDebuggerNodeReloadScripts();

        public static bool ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus = true) =>
            godot_icall_Internal_ScriptEditorEdit(resource.NativeInstance, line, col, grabFocus);

        public static void EditorNodeShowScriptScreen() => godot_icall_Internal_EditorNodeShowScriptScreen();

        public static string MonoWindowsInstallRoot
        {
            get
            {
                godot_icall_Internal_MonoWindowsInstallRoot(out godot_string dest);
                using (dest)
                    return Marshaling.ConvertStringToManaged(dest);
            }
        }

        public static void EditorRunPlay() => godot_icall_Internal_EditorRunPlay();

        public static void EditorRunStop() => godot_icall_Internal_EditorRunStop();

        public static void ScriptEditorDebugger_ReloadScripts() =>
            godot_icall_Internal_ScriptEditorDebugger_ReloadScripts();

        public static unsafe string[] CodeCompletionRequest(CodeCompletionRequest.CompletionKind kind,
            string scriptFile)
        {
            using godot_string scriptFileIn = Marshaling.ConvertStringToNative(scriptFile);
            godot_icall_Internal_CodeCompletionRequest((int)kind, scriptFileIn, out godot_packed_string_array res);
            using (res)
                return Marshaling.ConvertNativePackedStringArrayToSystemArray(res);
        }

        #region Internal

        private const string GodotDllName = "__Internal";

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_ResMetadataDir(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_ResTempAssembliesBaseDir(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_MonoUserDir(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_BuildLogsDirs(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_ProjectSlnPath(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_ProjectCsProjPath(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_GodotSharpDirs_DataEditorToolsDir(out godot_string r_dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_EditorProgress_Create(in godot_string task, in godot_string label,
            int amount, bool canCancel);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_EditorProgress_Dispose(in godot_string task);

        [DllImport(GodotDllName)]
        public static extern bool godot_icall_EditorProgress_Step(in godot_string task, in godot_string state, int step,
            bool forceRefresh);

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_FullTemplatesDir(out godot_string dest);

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_SimplifyGodotPath(in godot_string path, out godot_string dest);

        [DllImport(GodotDllName)]
        private static extern bool godot_icall_Internal_IsOsxAppBundleInstalled(in godot_string bundleId);

        [DllImport(GodotDllName)]
        private static extern bool godot_icall_Internal_GodotIs32Bits();

        [DllImport(GodotDllName)]
        private static extern bool godot_icall_Internal_GodotIsRealTDouble();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_GodotMainIteration();

        [DllImport(GodotDllName)]
        private static extern bool godot_icall_Internal_IsAssembliesReloadingNeeded();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_ReloadAssemblies(bool softReload);

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_EditorDebuggerNodeReloadScripts();

        [DllImport(GodotDllName)]
        private static extern bool godot_icall_Internal_ScriptEditorEdit(IntPtr resource, int line, int col,
            bool grabFocus);

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_EditorNodeShowScriptScreen();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_MonoWindowsInstallRoot(out godot_string dest);

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_EditorRunPlay();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_EditorRunStop();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_ScriptEditorDebugger_ReloadScripts();

        [DllImport(GodotDllName)]
        private static extern void godot_icall_Internal_CodeCompletionRequest(int kind, in godot_string scriptFile,
            out godot_packed_string_array res);

        [DllImport(GodotDllName)]
        public static extern float godot_icall_Globals_EditorScale();

        [DllImport(GodotDllName)]
        public static extern void godot_icall_Globals_GlobalDef(in godot_string setting, in godot_variant defaultValue,
            bool restartIfChanged, out godot_variant result);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_Globals_EditorDef(in godot_string setting, in godot_variant defaultValue,
            bool restartIfChanged, out godot_variant result);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_Globals_EditorShortcut(in godot_string setting, out godot_variant result);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_Globals_TTR(in godot_string text, out godot_string dest);

        [DllImport(GodotDllName)]
        public static extern void godot_icall_Utils_OS_GetPlatformName(out godot_string dest);

        [DllImport(GodotDllName)]
        public static extern bool godot_icall_Utils_OS_UnixFileHasExecutableAccess(in godot_string filePath);

        #endregion
    }
}
