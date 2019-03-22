using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;

namespace GodotSharpTools.Editor
{
    public class MonoDevelopInstance
    {
        public enum EditorId
        {
            MonoDevelop = 0,
            VisualStudioForMac = 1
        }

        readonly string solutionFile;
        readonly EditorId editorId;

        Process process;

        public void Execute(string[] files)
        {
            bool newWindow = process == null || process.HasExited;

            List<string> args = new List<string>();

            string command;

            if (Utils.OS.IsOSX())
            {
                string bundleId = codeEditorBundleIds[editorId];

                if (IsApplicationBundleInstalled(bundleId))
                {
                    command = "open";

                    args.Add("-b");
                    args.Add(bundleId);

                    // The 'open' process must wait until the application finishes
                    if (newWindow)
                        args.Add("--wait-apps");

                    args.Add("--args");
                }
                else
                {
                    command = codeEditorPaths[editorId];
                }
            }
            else
            {
                command = codeEditorPaths[editorId];
            }

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
                process = Process.Start(new ProcessStartInfo()
                {
                    FileName = command,
                    Arguments = string.Join(" ", args),
                    UseShellExecute = false
                });
            }
            else
            {
                Process.Start(new ProcessStartInfo()
                {
                    FileName = command,
                    Arguments = string.Join(" ", args),
                    UseShellExecute = false
                });
            }
        }

        public MonoDevelopInstance(string solutionFile, EditorId editorId)
        {
            if (editorId == EditorId.VisualStudioForMac && !Utils.OS.IsOSX())
                throw new InvalidOperationException($"{nameof(EditorId.VisualStudioForMac)} not supported on this platform");

            this.solutionFile = solutionFile;
            this.editorId = editorId;
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private extern static bool IsApplicationBundleInstalled(string bundleId);

        static readonly IReadOnlyDictionary<EditorId, string> codeEditorPaths;
        static readonly IReadOnlyDictionary<EditorId, string> codeEditorBundleIds;

        static MonoDevelopInstance()
        {
            if (Utils.OS.IsOSX())
            {
                codeEditorPaths = new Dictionary<EditorId, string>
                {
                    // Rely on PATH
                    { EditorId.MonoDevelop, "monodevelop" },
                    { EditorId.VisualStudioForMac, "VisualStudio" }
                };
                codeEditorBundleIds = new Dictionary<EditorId, string>
                {
                    // TODO EditorId.MonoDevelop
                    { EditorId.VisualStudioForMac, "com.microsoft.visual-studio" }
                };
            }
            else if (Utils.OS.IsWindows())
            {
                codeEditorPaths = new Dictionary<EditorId, string>
                {
                    // XamarinStudio is no longer a thing, and the latest version is quite old
                    // MonoDevelop is available from source only on Windows. The recommendation
                    // is to use Visual Studio instead. Since there are no official builds, we
                    // will rely on custom MonoDevelop builds being added to PATH.
                    { EditorId.MonoDevelop, "MonoDevelop.exe" }
                };
            }
            else if (Utils.OS.IsUnix())
            {
                codeEditorPaths = new Dictionary<EditorId, string>
                {
                    // Rely on PATH
                    { EditorId.MonoDevelop, "monodevelop" }
                };
            }
        }
    }
}
