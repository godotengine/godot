using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;

namespace GodotTools.IdeConnection
{
    public abstract class GodotIdeClient : GodotIdeBase
    {
        protected GodotIdeMetadata GodotIdeMetadata;

        private readonly FileSystemWatcher fsWatcher;

        protected GodotIdeClient(string projectMetadataDir) : base(projectMetadataDir)
        {
            messageHandlers = InitializeMessageHandlers();

            // FileSystemWatcher requires an existing directory
            if (!File.Exists(projectMetadataDir))
                Directory.CreateDirectory(projectMetadataDir);

            fsWatcher = new FileSystemWatcher(projectMetadataDir, MetaFileName);
        }

        private void OnMetaFileChanged(object sender, FileSystemEventArgs e)
        {
            if (IsDisposed)
                return;

            lock (ConnectionLock)
            {
                if (IsDisposed)
                    return;

                if (!File.Exists(MetaFilePath))
                    return;

                var metadata = ReadMetadataFile();

                if (metadata != null && metadata != GodotIdeMetadata)
                {
                    GodotIdeMetadata = metadata.Value;
                    ConnectToServer();
                }
            }
        }

        private void OnMetaFileDeleted(object sender, FileSystemEventArgs e)
        {
            if (IsDisposed)
                return;

            if (IsConnected)
                DisposeConnection();

            // The file may have been re-created

            lock (ConnectionLock)
            {
                if (IsDisposed)
                    return;

                if (IsConnected || !File.Exists(MetaFilePath))
                    return;

                var metadata = ReadMetadataFile();

                if (metadata != null)
                {
                    GodotIdeMetadata = metadata.Value;
                    ConnectToServer();
                }
            }
        }

        private GodotIdeMetadata? ReadMetadataFile()
        {
            using (var reader = File.OpenText(MetaFilePath))
            {
                string portStr = reader.ReadLine();

                if (portStr == null)
                    return null;

                string editorExecutablePath = reader.ReadLine();

                if (editorExecutablePath == null)
                    return null;

                if (!int.TryParse(portStr, out int port))
                    return null;

                return new GodotIdeMetadata(port, editorExecutablePath);
            }
        }

        private void ConnectToServer()
        {
            var tcpClient = new TcpClient();

            Connection = new GodotIdeConnectionClient(tcpClient, HandleMessage);
            Connection.Logger = Logger;

            try
            {
                Logger.LogInfo("Connecting to Godot Ide Server");
                
                tcpClient.Connect(IPAddress.Loopback, GodotIdeMetadata.Port);

                Logger.LogInfo("Connection open with Godot Ide Server");

                var clientThread = new Thread(Connection.Start)
                {
                    IsBackground = true,
                    Name = "Godot Ide Connection Client"
                };
                clientThread.Start();
            }
            catch (SocketException e)
            {
                if (e.SocketErrorCode == SocketError.ConnectionRefused)
                    Logger.LogError("The connection to the Godot Ide Server was refused");
                else
                    throw;
            }
        }

        public void Start()
        {
            Logger.LogInfo("Starting Godot Ide Client");
            
            fsWatcher.Changed += OnMetaFileChanged;
            fsWatcher.Deleted += OnMetaFileDeleted;
            fsWatcher.EnableRaisingEvents = true;

            lock (ConnectionLock)
            {
                if (IsDisposed)
                    return;

                if (!File.Exists(MetaFilePath))
                {
                    Logger.LogInfo("There is no Godot Ide Server running");
                    return;
                }

                var metadata = ReadMetadataFile();

                if (metadata != null)
                {
                    GodotIdeMetadata = metadata.Value;
                    ConnectToServer();
                }
                else
                {
                    Logger.LogError("Failed to read Godot Ide metadata file");
                }
            }
        }

        public bool WriteMessage(Message message)
        {
            return Connection.WriteMessage(message);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            if (disposing)
            {
                fsWatcher?.Dispose();
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
                ["OpenFile"] = args =>
                {
                    switch (args.Length)
                    {
                        case 1:
                            OpenFile(file: args[0]);
                            return;
                        case 2:
                            OpenFile(file: args[0], line: int.Parse(args[1]));
                            return;
                        case 3:
                            OpenFile(file: args[0], line: int.Parse(args[1]), column: int.Parse(args[2]));
                            return;
                        default:
                            throw new ArgumentException();
                    }
                }
            };
        }

        protected abstract void OpenFile(string file);
        protected abstract void OpenFile(string file, int line);
        protected abstract void OpenFile(string file, int line, int column);
    }
}
