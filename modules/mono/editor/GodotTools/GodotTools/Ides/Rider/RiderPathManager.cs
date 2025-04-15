using System;
using System.IO;
using System.Linq;
using Godot;
using GodotTools.Internals;
using JetBrains.Rider.PathLocator;

namespace GodotTools.Ides.Rider
{
    public static class RiderPathManager
    {
        internal const string EditorPathSettingName = "dotnet/editor/editor_path_optional";

        private static readonly RiderPathLocator RiderPathLocator;
        private static readonly RiderFileOpener RiderFileOpener;

        static RiderPathManager()
        {
            var riderLocatorEnvironment = new RiderLocatorEnvironment();
            RiderPathLocator = new RiderPathLocator(riderLocatorEnvironment);
            RiderFileOpener = new RiderFileOpener(riderLocatorEnvironment);
        }

        private static string? GetRiderPathFromSettings()
        {
            var editorSettings = EditorInterface.Singleton.GetEditorSettings();
            if (editorSettings.HasSetting(EditorPathSettingName))
            {
                return (string)editorSettings.GetSetting(EditorPathSettingName);
            }

            return null;
        }

        public static void InitializeIfNeeded(ExternalEditorId editor)
        {
            var editorSettings = EditorInterface.Singleton.GetEditorSettings();
            if (!editorSettings.HasSetting(EditorPathSettingName))
            {
                Globals.EditorDef(EditorPathSettingName, "");
                editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
                {
                    ["type"] = (int)Variant.Type.String,
                    ["name"] = EditorPathSettingName,
                    ["hint"] = (int)PropertyHint.File,
                    ["hint_string"] = ""
                });
            }

            var editorPath = (string)editorSettings.GetSetting(EditorPathSettingName);
            if (File.Exists(editorPath) && IsMatch(editor, editorPath))
            {
                Globals.EditorDef(EditorPathSettingName, editorPath);
                return;
            }

            var paths = RiderPathLocator.GetAllRiderPaths().Where(info => IsMatch(editor, info.Path)).ToArray();
            if (paths.Length == 0)
            {
                return;
            }

            string newPath = paths.Last().Path;
            Globals.EditorDef(EditorPathSettingName, newPath);
            editorSettings.SetSetting(EditorPathSettingName, newPath);
        }

        private static bool IsMatch(ExternalEditorId editorId, string path)
        {
            if (path.IndexOfAny(Path.GetInvalidPathChars()) != -1)
            {
                return false;
            }

            var fileInfo = new FileInfo(path);
            var name = editorId == ExternalEditorId.Fleet ? "fleet" : "rider";
            return fileInfo.Name.StartsWith(name, StringComparison.OrdinalIgnoreCase);
        }

        private static string? CheckAndUpdatePath(ExternalEditorId editorId, string? idePath)
        {
            if (File.Exists(idePath))
            {
                return idePath;
            }

            var allInfos = RiderPathLocator.GetAllRiderPaths();
            if (allInfos.Length == 0)
            {
                return null;
            }

            // RiderPathLocator includes Rider and Fleet locations.
            var matchingIde = allInfos.LastOrDefault(info => IsMatch(editorId, info.Path));
            var newPath = matchingIde.Path;
            if (string.IsNullOrEmpty(newPath))
            {
                return null;
            }

            var editorSettings = EditorInterface.Singleton.GetEditorSettings();
            editorSettings.SetSetting(EditorPathSettingName, newPath);
            Globals.EditorDef(EditorPathSettingName, newPath);
            return newPath;
        }

        public static void OpenFile(ExternalEditorId editorId, string slnPath, string scriptPath, int line, int column)
        {
            var pathFromSettings = GetRiderPathFromSettings();
            var path = CheckAndUpdatePath(editorId, pathFromSettings);
            if (string.IsNullOrEmpty(path))
            {
                GD.PushError($"Error when trying to run code editor: JetBrains Rider or Fleet. Could not find path to the editor.");
                return;
            }

            RiderFileOpener.OpenFile(path, slnPath, scriptPath, line, column);
        }
    }
}
