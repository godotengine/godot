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
                if (script is not CSharpScript) continue;

                if (File.GetLastWriteTime(script.ResourcePath) > BuildManager.LastValidBuildDateTime)
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
