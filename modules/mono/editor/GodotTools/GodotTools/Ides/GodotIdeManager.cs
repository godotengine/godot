using System;
using System.IO;
using Godot;
using GodotTools.IdeConnection;
using GodotTools.Internals;

namespace GodotTools.Ides
{
    public class GodotIdeManager : Node, ISerializationListener
    {
        public GodotIdeServer GodotIdeServer { get; private set; }

        private MonoDevelop.Instance monoDevelInstance;
        private MonoDevelop.Instance vsForMacInstance;

        private GodotIdeServer GetRunningServer()
        {
            if (GodotIdeServer != null && !GodotIdeServer.IsDisposed)
                return GodotIdeServer;
            StartServer();
            return GodotIdeServer;
        }

        public override void _Ready()
        {
            StartServer();
        }

        public void OnBeforeSerialize()
        {
            GodotIdeServer?.Dispose();
        }

        public void OnAfterDeserialize()
        {
            StartServer();
        }

        private ILogger logger;

        protected ILogger Logger
        {
            get => logger ?? (logger = new ConsoleLogger());
            set => logger = value;
        }

        private void StartServer()
        {
            GodotIdeServer?.Dispose();
            GodotIdeServer = new GodotIdeServer(LaunchIde,
                OS.GetExecutablePath(),
                ProjectSettings.GlobalizePath(GodotSharpDirs.ResMetadataDir));

            GodotIdeServer.Logger = Logger;

            GodotIdeServer.StartServer();
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            GodotIdeServer?.Dispose();
        }

        private void LaunchIde()
        {
            var editor = (ExternalEditorId) GodotSharpEditor.Instance.GetEditorInterface()
                .GetEditorSettings().GetSetting("mono/editor/external_editor");

            switch (editor)
            {
                case ExternalEditorId.None:
                case ExternalEditorId.VisualStudio:
                case ExternalEditorId.VsCode:
                    throw new NotSupportedException();
                case ExternalEditorId.VisualStudioForMac:
                    goto case ExternalEditorId.MonoDevelop;
                case ExternalEditorId.MonoDevelop:
                {
                    MonoDevelop.Instance GetMonoDevelopInstance(string solutionPath)
                    {
                        if (Utils.OS.IsOSX() && editor == ExternalEditorId.VisualStudioForMac)
                        {
                            vsForMacInstance = vsForMacInstance ??
                                               new MonoDevelop.Instance(solutionPath, MonoDevelop.EditorId.VisualStudioForMac);
                            return vsForMacInstance;
                        }

                        monoDevelInstance = monoDevelInstance ??
                                            new MonoDevelop.Instance(solutionPath, MonoDevelop.EditorId.MonoDevelop);
                        return monoDevelInstance;
                    }

                    try
                    {
                        var instance = GetMonoDevelopInstance(GodotSharpDirs.ProjectSlnPath);

                        if (!instance.IsRunning)
                            instance.Execute();
                    }
                    catch (FileNotFoundException)
                    {
                        string editorName = editor == ExternalEditorId.VisualStudioForMac ? "Visual Studio" : "MonoDevelop";
                        GD.PushError($"Cannot find code editor: {editorName}");
                    }

                    break;
                }

                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        private void WriteMessage(string id, params string[] arguments)
        {
            GetRunningServer().WriteMessage(new Message(id, arguments));
        }

        public void SendOpenFile(string file)
        {
            WriteMessage("OpenFile", file);
        }

        public void SendOpenFile(string file, int line)
        {
            WriteMessage("OpenFile", file, line.ToString());
        }

        public void SendOpenFile(string file, int line, int column)
        {
            WriteMessage("OpenFile", file, line.ToString(), column.ToString());
        }

        private class GodotLogger : ILogger
        {
            public void LogDebug(string message)
            {
                if (OS.IsStdoutVerbose())
                    Console.WriteLine(message);
            }

            public void LogInfo(string message)
            {
                if (OS.IsStdoutVerbose())
                    Console.WriteLine(message);
            }

            public void LogWarning(string message)
            {
                GD.PushWarning(message);
            }

            public void LogError(string message)
            {
                GD.PushError(message);
            }

            public void LogError(string message, Exception e)
            {
                GD.PushError(message + "\n" + e);
            }
        }
    }
}
