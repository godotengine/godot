using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;

namespace GodotSharpTools.Editor
{
    public class MonoDevelopInstance
    {
        private Process process;
        private string solutionFile;

        public void Execute(string[] files)
        {
            bool newWindow = process == null || process.HasExited;

            List<string> args = new List<string>();

            args.Add("--ipc-tcp");

            if (newWindow)
                args.Add("\"" + Path.GetFullPath(solutionFile) + "\"");

            foreach (var file in files)
            {
                int semicolonIndex = file.IndexOf(';');

                string filePath = semicolonIndex < 0 ? file : file.Substring(0, semicolonIndex);
                string cursor = semicolonIndex < 0 ? string.Empty : file.Substring(semicolonIndex);

                args.Add("\"" + Path.GetFullPath(filePath.NormalizePath()) + cursor + "\"");
            }

            if (newWindow)
            {
                ProcessStartInfo startInfo = new ProcessStartInfo(MonoDevelopFile, string.Join(" ", args));
                process = Process.Start(startInfo);
            }
            else
            {
                Process.Start(MonoDevelopFile, string.Join(" ", args));
            }
        }

        public MonoDevelopInstance(string solutionFile)
        {
            this.solutionFile = solutionFile;
        }

        private static string MonoDevelopFile
        {
            get
            {
                return "monodevelop";
            }
        }
    }
}
