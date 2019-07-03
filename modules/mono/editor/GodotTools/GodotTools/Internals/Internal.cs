using System;
using System.Runtime.CompilerServices;
using Godot;
using Godot.Collections;

namespace GodotTools.Internals
{
    public static class Internal
    {
        public const string CSharpLanguageType = "CSharpScript";
        public const string CSharpLanguageExtension = "cs";

        public static float EditorScale => internal_EditorScale();

        public static object GlobalDef(string setting, object defaultValue, bool restartIfChanged = false) =>
            internal_GlobalDef(setting, defaultValue, restartIfChanged);

        public static object EditorDef(string setting, object defaultValue, bool restartIfChanged = false) =>
            internal_EditorDef(setting, defaultValue, restartIfChanged);

        public static string FullTemplatesDir =>
            internal_FullTemplatesDir();

        public static string SimplifyGodotPath(this string path) => internal_SimplifyGodotPath(path);

        public static bool IsOsxAppBundleInstalled(string bundleId) => internal_IsOsxAppBundleInstalled(bundleId);

        public static bool MetadataIsApiAssemblyInvalidated(ApiAssemblyType apiType) =>
            internal_MetadataIsApiAssemblyInvalidated(apiType);

        public static void MetadataSetApiAssemblyInvalidated(ApiAssemblyType apiType, bool invalidated) =>
            internal_MetadataSetApiAssemblyInvalidated(apiType, invalidated);

        public static bool IsMessageQueueFlushing() => internal_IsMessageQueueFlushing();

        public static bool GodotIs32Bits() => internal_GodotIs32Bits();

        public static bool GodotIsRealTDouble() => internal_GodotIsRealTDouble();

        public static void GodotMainIteration() => internal_GodotMainIteration();

        public static ulong GetCoreApiHash() => internal_GetCoreApiHash();

        public static ulong GetEditorApiHash() => internal_GetEditorApiHash();

        public static bool IsAssembliesReloadingNeeded() => internal_IsAssembliesReloadingNeeded();

        public static void ReloadAssemblies(bool softReload) => internal_ReloadAssemblies(softReload);

        public static void ScriptEditorDebuggerReloadScripts() => internal_ScriptEditorDebuggerReloadScripts();

        public static bool ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus = true) =>
            internal_ScriptEditorEdit(resource, line, col, grabFocus);

        public static void EditorNodeShowScriptScreen() => internal_EditorNodeShowScriptScreen();

        public static Dictionary<string, object> GetScriptsMetadataOrNothing() =>
            internal_GetScriptsMetadataOrNothing(typeof(Dictionary<string, object>));

        public static string MonoWindowsInstallRoot => internal_MonoWindowsInstallRoot();

        // Internal Calls

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern float internal_EditorScale();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern object internal_GlobalDef(string setting, object defaultValue, bool restartIfChanged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern object internal_EditorDef(string setting, object defaultValue, bool restartIfChanged);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_FullTemplatesDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_SimplifyGodotPath(this string path);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsOsxAppBundleInstalled(string bundleId);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_MetadataIsApiAssemblyInvalidated(ApiAssemblyType apiType);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_MetadataSetApiAssemblyInvalidated(ApiAssemblyType apiType, bool invalidated);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_IsMessageQueueFlushing();

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
        private static extern void internal_ScriptEditorDebuggerReloadScripts();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern bool internal_ScriptEditorEdit(Resource resource, int line, int col, bool grabFocus);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern void internal_EditorNodeShowScriptScreen();

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Dictionary<string, object> internal_GetScriptsMetadataOrNothing(Type dictType);

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern string internal_MonoWindowsInstallRoot();
    }
}
