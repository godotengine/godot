using GodotTools.Core;
using System;
using System.IO;
using System.Collections.Generic;
using System.Diagnostics;
using GodotTools.Internals;

namespace GodotTools
{
    public class MonoDevelopInstance
    {
        public enum EditorId
        {
            MonoDevelop = 0,
            VisualStudioForMac = 1
        }

        private readonly string solutionFile;
        private readonly EditorId editorId;

        private Process process;

        public void Execute(params string[] files)
        {
            bool newWindow = process == null || process.HasExited;

            var args = new List<string>();

            string command;

            if (Utils.OS.IsOSX())
            {
                string bundleId = CodeEditorBundleIds[editorId];

                if (Internal.IsOsxAppBundleInstalled(bundleId))
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
                    command = CodeEditorPaths[editorId];
                }
            }
            else
            {
                command = CodeEditorPaths[editorId];
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
                process = Process.Start(new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = string.Join(" ", args),
                    UseShellExecute = false
                });
            }
            else
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = command,
                    Arguments = string.Join(" ", args),
                    UseShellExecute = false
                })?.Dispose();
            }
        }

        public MonoDevelopInstance(string solutionFile, EditorId editorId)
        {
            if (editorId == EditorId.VisualStudioForMac && !Utils.OS.IsOSX())
                throw new InvalidOperationException($"{nameof(EditorId.VisualStudioForMac)} not supported on this platform");

            this.solutionFile = solutionFile;
            this.editorId = editorId;
        }

        private static readonly IReadOnlyDictionary<EditorId, string> CodeEditorPaths;
        private static readonly IReadOnlyDictionary<EditorId, string> CodeEditorBundleIds;

        static MonoDevelopInstance()
        {
            if (Utils.OS.IsOSX())
            {
                CodeEditorPaths = new Dictionary<EditorId, string>
                {
                    // Rely on PATH
                    {EditorId.MonoDevelop, "monodevelop"},
                    {EditorId.VisualStudioForMac, "VisualStudio"}
                };
                CodeEditorBundleIds = new Dictionary<EditorId, string>
                {
                    // TODO EditorId.MonoDevelop
                    {EditorId.VisualStudioForMac, "com.microsoft.visual-studio"}
                };
            }
            else if (Utils.OS.IsWindows())
            {
                CodeEditorPaths = new Dictionary<EditorId, string>
                {
                    // XamarinStudio is no longer a thing, and the latest version is quite old
                    // MonoDevelop is available from source only on Windows. The recommendation
                    // is to use Visual Studio instead. Since there are no official builds, we
                    // will rely on custom MonoDevelop builds being added to PATH.
                    {EditorId.MonoDevelop, "MonoDevelop.exe"}
                };
            }
            else if (Utils.OS.IsUnix())
            {
                CodeEditorPaths = new Dictionary<EditorId, string>
                {
                    // Rely on PATH
                    {EditorId.MonoDevelop, "monodevelop"}
                };
            }
        }
    }
}
