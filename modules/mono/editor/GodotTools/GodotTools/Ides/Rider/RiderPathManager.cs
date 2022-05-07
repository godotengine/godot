using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Godot;
using GodotTools.Internals;

namespace GodotTools.Ides.Rider
{
    public static class RiderPathManager
    {
        public static readonly string EditorPathSettingName = "mono/editor/editor_path_optional";

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
            var editor = (ExternalEditorId)editorSettings.GetSetting("mono/editor/external_editor");
            if (editor == ExternalEditorId.Rider)
            {
                if (!editorSettings.HasSetting(EditorPathSettingName))
                {
                    Globals.EditorDef(EditorPathSettingName, "Optional");
                    editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
                    {
                        ["type"] = Variant.Type.String,
                        ["name"] = EditorPathSettingName,
                        ["hint"] = PropertyHint.File,
                        ["hint_string"] = ""
                    });
                }

                var riderPath = (string)editorSettings.GetSetting(EditorPathSettingName);
                if (IsRiderAndExists(riderPath))
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

        public static bool IsExternalEditorSetToRider(EditorSettings editorSettings)
        {
            return editorSettings.HasSetting(EditorPathSettingName) &&
                IsRider((string)editorSettings.GetSetting(EditorPathSettingName));
        }

        public static bool IsRider(string path)
        {
            if (string.IsNullOrEmpty(path))
                return false;

            var fileInfo = new FileInfo(path);
            string filename = fileInfo.Name.ToLowerInvariant();
            return filename.StartsWith("rider", StringComparison.Ordinal);
        }

        private static string CheckAndUpdatePath(string riderPath)
        {
            if (IsRiderAndExists(riderPath))
            {
                return riderPath;
            }

            var editorSettings = GodotSharpEditor.Instance.GetEditorInterface().GetEditorSettings();
            var paths = RiderPathLocator.GetAllRiderPaths();

            if (!paths.Any())
                return null;

            string newPath = paths.Last().Path;
            editorSettings.SetSetting(EditorPathSettingName, newPath);
            Globals.EditorDef(EditorPathSettingName, newPath);
            return newPath;
        }

        private static bool IsRiderAndExists(string riderPath)
        {
            return !string.IsNullOrEmpty(riderPath) && IsRider(riderPath) && new FileInfo(riderPath).Exists;
        }

        public static void OpenFile(string slnPath, string scriptPath, int line)
        {
            string pathFromSettings = GetRiderPathFromSettings();
            string path = CheckAndUpdatePath(pathFromSettings);

            var args = new List<string>();
            args.Add(slnPath);
            if (line >= 0)
            {
                args.Add("--line");
                args.Add((line + 1).ToString()); // https://github.com/JetBrains/godot-support/issues/61
            }
            args.Add(scriptPath);
            try
            {
                Utils.OS.RunProcess(path, args);
            }
            catch (Exception e)
            {
                GD.PushError($"Error when trying to run code editor: JetBrains Rider. Exception message: '{e.Message}'");
            }
        }
    }
}
