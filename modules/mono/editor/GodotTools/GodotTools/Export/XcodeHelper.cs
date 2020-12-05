using System;
using System.IO;

namespace GodotTools.Export
{
    public static class XcodeHelper
    {
        private static string _XcodePath = null;

        public static string XcodePath
        {
            get
            {
                if (_XcodePath == null)
                {
                    _XcodePath = FindXcode();

                    if (_XcodePath == null)
                        throw new Exception("Could not find Xcode");
                }

                return _XcodePath;
            }
        }

        private static string FindSelectedXcode()
        {
            var outputWrapper = new Godot.Collections.Array();

            int exitCode = Godot.OS.Execute("xcode-select", new string[] { "--print-path" }, blocking: true, output: outputWrapper);

            if (exitCode == 0)
            {
                string output = (string)outputWrapper[0];
                return output.Trim();
            }

            Console.Error.WriteLine($"'xcode-select --print-path' exited with code: {exitCode}");

            return null;
        }

        public static string FindXcode()
        {
            string selectedXcode = FindSelectedXcode();
            if (selectedXcode != null)
            {
                if (Directory.Exists(Path.Combine(selectedXcode, "Contents", "Developer")))
                    return selectedXcode;

                // The path already pointed to Contents/Developer
                var dirInfo = new DirectoryInfo(selectedXcode);
                if (dirInfo.Name != "Developer" || dirInfo.Parent.Name != "Contents")
                {
                    Console.WriteLine(Path.GetDirectoryName(selectedXcode));
                    Console.WriteLine(System.IO.Directory.GetParent(selectedXcode).Name);
                    Console.Error.WriteLine("Unrecognized path for selected Xcode");
                }
                else
                {
                    return System.IO.Path.GetFullPath($"{selectedXcode}/../..");
                }
            }
            else
            {
                Console.Error.WriteLine("Could not find the selected Xcode; trying with a hint path");
            }

            const string XcodeHintPath = "/Applications/Xcode.app";

            if (Directory.Exists(XcodeHintPath))
            {
                if (Directory.Exists(Path.Combine(XcodeHintPath, "Contents", "Developer")))
                    return XcodeHintPath;

                Console.Error.WriteLine($"Found Xcode at '{XcodeHintPath}' but it's missing the 'Contents/Developer' sub-directory");
            }

            return null;
        }

        public static string FindXcodeTool(string toolName)
        {
            string XcodeDefaultToolchain = Path.Combine(XcodePath, "Contents", "Developer", "Toolchains", "XcodeDefault.xctoolchain");

            string path = Path.Combine(XcodeDefaultToolchain, "usr", "bin", toolName);
            if (File.Exists(path))
                return path;

            throw new FileNotFoundException($"Cannot find Xcode tool: {toolName}");
        }
    }
}
