using Godot;
using System;
using GodotTools.ProjectEditor;

namespace GodotTools
{
    public static class CsProjOperations
    {
        public static string GenerateGameProject(string dir, string name, string additionalDefines)
        {
            try
            {
                return ProjectGenerator.GenAndSaveGameProject(dir, name, additionalDefines);
            }
            catch (Exception e)
            {
                GD.PushError(e.ToString());
                return string.Empty;
            }
        }
    }
}
