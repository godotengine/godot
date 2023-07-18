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
        private static readonly RiderPathLocator RiderPathLocator;
        private static readonly RiderFileOpener RiderFileOpener;

        static RiderPathManager()
        {
            var riderLocatorEnvironment = new RiderLocatorEnvironment();
            RiderPathLocator = new RiderPathLocator(riderLocatorEnvironment);
            RiderFileOpener = new RiderFileOpener(riderLocatorEnvironment);
        }

        public static readonly string EditorPathSettingName = "dotnet/editor/editor_path_optional";

        private static string GetRiderPathFromSettings()
        {
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            if (editorSettings.HasSetting(EditorPathSettingName))
                return (string)editorSettings.GetSetting(EditorPathSettingName);
            return null;
        }

        public static void Initialize()
        {
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var editor = editorSettings.GetSetting(GodotSharpEditor.Settings.ExternalEditor).As<ExternalEditorId>();
            if (editor == ExternalEditorId.Rider)
            {
                if (!editorSettings.HasSetting(EditorPathSettingName))
                {
                    Globals.EditorDef(EditorPathSettingName, "Optional");
                    editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
                    {
                        ["type"] = (int)Variant.Type.String,
                        ["name"] = EditorPathSettingName,
                        ["hint"] = (int)PropertyHint.File,
                        ["hint_string"] = ""
                    });
                }

                var riderPath = (string)editorSettings.GetSetting(EditorPathSettingName);
                if (File.Exists(riderPath))
                {
                    Globals.EditorDef(EditorPathSettingName, riderPath);
                    return;
                }

                var paths = RiderPathLocator.GetAllRiderPaths();

                if (!paths.Any())
                    return;

                string newPath = paths.Last().Path;
                Globals.EditorDef(EditorPathSettingName, newPath);
                editorSettings.SetSetting(EditorPathSettingName, newPath);
            }
        }

        public static bool IsRider(string path)
        {
            if (string.IsNullOrEmpty(path))
                return false;

            if (path.IndexOfAny(Path.GetInvalidPathChars()) != -1)
                return false;

            var fileInfo = new FileInfo(path);
            string filename = fileInfo.Name.ToLowerInvariant();
            return filename.StartsWith("rider", StringComparison.Ordinal);
        }

        private static string CheckAndUpdatePath(string riderPath)
        {
            if (File.Exists(riderPath))
            {
                return riderPath;
            }

            var allInfos = RiderPathLocator.GetAllRiderPaths();
            if (allInfos.Length == 0)
                return null;
            var riderInfos = allInfos.Where(info => IsRider(info.Path)).ToArray();
            string newPath = riderInfos.Length > 0
                ? riderInfos[riderInfos.Length - 1].Path
                : allInfos[allInfos.Length - 1].Path;
            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            editorSettings.SetSetting(EditorPathSettingName, newPath);
            Globals.EditorDef(EditorPathSettingName, newPath);
            return newPath;
        }

        public static void OpenFile(string slnPath, string scriptPath, int line, int column)
        {
            string pathFromSettings = GetRiderPathFromSettings();
            string path = CheckAndUpdatePath(pathFromSettings);
            RiderFileOpener.OpenFile(path, slnPath, scriptPath, line, column);
        }
    }
}
