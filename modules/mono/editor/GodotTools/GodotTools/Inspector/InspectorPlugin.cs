using System;
using System.Collections.Generic;
using Godot;
using GodotTools.Build;
using GodotTools.Utils;

namespace GodotTools.Inspector
{
    public partial class InspectorPlugin : EditorInspectorPlugin
    {
        public override bool _CanHandle(GodotObject godotObject)
        {
            if (godotObject == null)
            {
                return false;
            }

            foreach (var script in EnumerateScripts(godotObject))
            {
                if (script is CSharpScript)
                {
                    return true;
                }
            }
            return false;
        }

        public override void _ParseBegin(GodotObject godotObject)
        {
            foreach (var script in EnumerateScripts(godotObject))
            {
                if (script is not CSharpScript)
                    continue;

                string scriptPath = script.ResourcePath;

                if (string.IsNullOrEmpty(scriptPath))
                {
                    // Generic types used empty paths in older versions of Godot
                    // so we assume your project is out of sync.
                    AddCustomControl(new InspectorOutOfSyncWarning());
                    break;
                }

                if (scriptPath.StartsWith("csharp://", StringComparison.Ordinal))
                {
                    var scriptPathSpan = scriptPath.AsSpan("csharp://".Length);
                    int colonIdx = scriptPathSpan.IndexOf(':');
                    if (colonIdx >= 0)
                    {
                        string basePath = $"res://{scriptPathSpan[..colonIdx]}";
                        string globalBasePath = ProjectSettings.GlobalizePath(basePath);
                        if (!File.Exists(globalBasePath))
                            continue;
                        scriptPath = basePath;
                    }
                    else
                    {
                        continue;
                    }
                }

                if (File.GetLastWriteTime(scriptPath) > BuildManager.LastValidBuildDateTime)
                {
                    AddCustomControl(new InspectorOutOfSyncWarning());
                    break;
                }
            }
        }

        private static IEnumerable<Script> EnumerateScripts(GodotObject godotObject)
        {
            var script = godotObject.GetScript().As<Script>();
            while (script != null)
            {
                yield return script;
                script = script.GetBaseScript();
            }
        }
    }
}
