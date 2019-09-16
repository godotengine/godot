using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GodotTools.IdeConnection;
using GodotTools.Internals;
using GodotTools.Utils;
using Directory = System.IO.Directory;
using File = System.IO.File;
using Thread = System.Threading.Thread;

namespace GodotTools.Ides
{
    public class GodotIdeServer : GodotIdeBase
    {
        private readonly TcpListener listener;
        private readonly FileStream metaFile;
        private readonly Action launchIdeAction;
        private readonly NotifyAwaiter<bool> clientConnectedAwaiter = new NotifyAwaiter<bool>();

        private async Task<bool> AwaitClientConnected()
        {
            return await clientConnectedAwaiter.Reset();
        }

        public GodotIdeServer(Action launchIdeAction, string editorExecutablePath, string projectMetadataDir)
            : base(projectMetadataDir)
        {
            messageHandlers = InitializeMessageHandlers();

            this.launchIdeAction = launchIdeAction;

            // Make sure the directory exists
            Directory.CreateDirectory(projectMetadataDir);

            // The Godot editor's file system thread can keep the file open for writing, so we are forced to allow write sharing...
            const FileShare metaFileShare = FileShare.ReadWrite;

            metaFile = File.Open(MetaFilePath, FileMode.Create, FileAccess.Write, metaFileShare);

            listener = new TcpListener(new IPEndPoint(IPAddress.Loopback, port: 0));
            listener.Start();

            int port = ((IPEndPoint) listener.Server.LocalEndPoint).Port;
            using (var metaFileWriter = new StreamWriter(metaFile, Encoding.UTF8))
            {
                metaFileWriter.WriteLine(port);
                metaFileWriter.WriteLine(editorExecutablePath);
            }

            StartServer();
        }

        public void StartServer()
        {
            var serverThread = new Thread(RunServerThread) {Name = "Godot Ide Connection Server"};
            serverThread.Start();
        }

        private void RunServerThread()
        {
            SynchronizationContext.SetSynchronizationContext(Godot.Dispatcher.SynchronizationContext);

            try
            {
                while (!IsDisposed)
                {
                    TcpClient tcpClient = listener.AcceptTcpClient();

                    Logger.LogInfo("Connection open with Ide Client");

                    lock (ConnectionLock)
                    {
                        Connection = new GodotIdeConnectionServer(tcpClient, HandleMessage);
                        Connection.Logger = Logger;
                    }

                    Connected += () => clientConnectedAwaiter.SetResult(true);

                    Connection.Start();
                }
            }
            catch (Exception e)
            {
                if (!IsDisposed && !(e is SocketException se && se.SocketErrorCode == SocketError.Interrupted))
                    throw;
            }
        }

        public async void WriteMessage(Message message)
        {
            async Task LaunchIde()
            {
                if (IsConnected)
                    return;

                launchIdeAction();
                await Task.WhenAny(Task.Delay(10000), AwaitClientConnected());
            }

            await LaunchIde();

            if (!IsConnected)
            {
                Logger.LogError("Cannot write message: Godot Ide Server not connected");
                return;
            }

            Connection.WriteMessage(message);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            if (disposing)
            {
                listener?.Stop();

                metaFile?.Dispose();

                File.Delete(MetaFilePath);
            }
        }

        protected virtual bool HandleMessage(Message message)
        {
            if (messageHandlers.TryGetValue(message.Id, out var action))
            {
                action(message.Arguments);
                return true;
            }

            return false;
        }

        private readonly Dictionary<string, Action<string[]>> messageHandlers;

        private Dictionary<string, Action<string[]>> InitializeMessageHandlers()
        {
            return new Dictionary<string, Action<string[]>>
            {
                ["Play"] = args =>
                {
                    switch (args.Length)
                    {
                        case 0:
                            Play();
                            return;
                        case 2:
                            Play(debuggerHost: args[0], debuggerPort: int.Parse(args[1]));
                            return;
                        default:
                            throw new ArgumentException();
                    }
                },
                ["ReloadScripts"] = args => ReloadScripts()
            };
        }

        private void DispatchToMainThread(Action action)
        {
            var d = new SendOrPostCallback(state => action());
            Godot.Dispatcher.SynchronizationContext.Post(d, null);
        }

        private void Play()
        {
            DispatchToMainThread(() =>
            {
                CurrentPlayRequest = new PlayRequest();
                Internal.EditorRunPlay();
                CurrentPlayRequest = null;
            });
        }

        private void Play(string debuggerHost, int debuggerPort)
        {
            DispatchToMainThread(() =>
            {
                CurrentPlayRequest = new PlayRequest(debuggerHost, debuggerPort);
                Internal.EditorRunPlay();
                CurrentPlayRequest = null;
            });
        }

        private void ReloadScripts()
        {
            DispatchToMainThread(Internal.ScriptEditorDebugger_ReloadScripts);
        }

        public PlayRequest? CurrentPlayRequest { get; private set; }

        public struct PlayRequest
        {
            public bool HasDebugger { get; }
            public string DebuggerHost { get; }
            public int DebuggerPort { get; }

            public PlayRequest(string debuggerHost, int debuggerPort)
            {
                HasDebugger = true;
                DebuggerHost = debuggerHost;
                DebuggerPort = debuggerPort;
            }
        }
    }
}
