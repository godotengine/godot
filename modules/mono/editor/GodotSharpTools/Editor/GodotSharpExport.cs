using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

namespace GodotSharpTools.Editor
{
    public static class GodotSharpExport
    {
        public static void _ExportBegin(string[] features, bool debug, string path, int flags)
        {
            var featureSet = new HashSet<string>(features);

            if (PlatformHasTemplateDir(featureSet))
            {
                string templateDirName = "data.mono";

                if (featureSet.Contains("Windows"))
                {
                    templateDirName += ".windows";
                    templateDirName += featureSet.Contains("64") ? ".64" : ".32";
                }
                else if (featureSet.Contains("X11"))
                {
                    templateDirName += ".x11";
                    templateDirName += featureSet.Contains("64") ? ".64" : ".32";
                }
                else
                {
                    throw new NotSupportedException("Target platform not supported");
                }

                templateDirName += debug ? ".debug" : ".release";

                string templateDirPath = Path.Combine(GetTemplatesDir(), templateDirName);

                if (!Directory.Exists(templateDirPath))
                    throw new FileNotFoundException("Data template directory not found");

                string outputDir = new FileInfo(path).Directory.FullName;

                string outputDataDir = Path.Combine(outputDir, GetDataDirName());

                if (Directory.Exists(outputDataDir))
                    Directory.Delete(outputDataDir, recursive: true); // Clean first

                Directory.CreateDirectory(outputDataDir);

                foreach (string dir in Directory.GetDirectories(templateDirPath, "*", SearchOption.AllDirectories))
                {
                    Directory.CreateDirectory(Path.Combine(outputDataDir, dir.Substring(templateDirPath.Length + 1)));
                }

                foreach (string file in Directory.GetFiles(templateDirPath, "*", SearchOption.AllDirectories))
                {
                    File.Copy(file, Path.Combine(outputDataDir, file.Substring(templateDirPath.Length + 1)));
                }
            }
        }

        public static bool PlatformHasTemplateDir(HashSet<string> featureSet)
        {
            // OSX export templates are contained in a zip, so we place
            // our custom template inside it and let Godot do the rest.
            return !featureSet.Contains("OSX");
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        extern static string GetTemplatesDir();

        [MethodImpl(MethodImplOptions.InternalCall)]
        extern static string GetDataDirName();
    }
}
