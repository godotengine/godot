using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using GodotTools.IdeMessaging;
using GodotTools.IdeMessaging.Requests;
using GodotTools.IdeMessaging.Utils;
using GodotTools.Internals;
using GodotTools.Utils;
using Newtonsoft.Json;
using Directory = System.IO.Directory;
using File = System.IO.File;

namespace GodotTools.Ides
{
    public sealed class MessagingServer : IDisposable
    {
        private readonly ILogger logger;

        private readonly FileStream metaFile;
        private string MetaFilePath { get; }

        private readonly SemaphoreSlim peersSem = new SemaphoreSlim(1);

        private readonly TcpListener listener;

        private readonly Dictionary<string, Queue<NotifyAwaiter<bool>>> clientConnectedAwaiters = new Dictionary<string, Queue<NotifyAwaiter<bool>>>();
        private readonly Dictionary<string, Queue<NotifyAwaiter<bool>>> clientDisconnectedAwaiters = new Dictionary<string, Queue<NotifyAwaiter<bool>>>();

        public async Task<bool> AwaitClientConnected(string identity)
        {
            if (!clientConnectedAwaiters.TryGetValue(identity, out var queue))
            {
                queue = new Queue<NotifyAwaiter<bool>>();
                clientConnectedAwaiters.Add(identity, queue);
            }

            var awaiter = new NotifyAwaiter<bool>();
            queue.Enqueue(awaiter);
            return await awaiter;
        }

        public async Task<bool> AwaitClientDisconnected(string identity)
        {
            if (!clientDisconnectedAwaiters.TryGetValue(identity, out var queue))
            {
                queue = new Queue<NotifyAwaiter<bool>>();
                clientDisconnectedAwaiters.Add(identity, queue);
            }

            var awaiter = new NotifyAwaiter<bool>();
            queue.Enqueue(awaiter);
            return await awaiter;
        }

        public bool IsDisposed { get; private set; }

        public bool IsAnyConnected(string identity) => string.IsNullOrEmpty(identity) ?
            Peers.Count > 0 :
            Peers.Any(c => c.RemoteIdentity == identity);

        private List<Peer> Peers { get; } = new List<Peer>();

        ~MessagingServer()
        {
            Dispose(disposing: false);
        }

        public async void Dispose()
        {
            if (IsDisposed)
                return;

            using (await peersSem.UseAsync())
            {
                if (IsDisposed) // lock may not be fair
                    return;
                IsDisposed = true;
            }

            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (disposing)
            {
                foreach (var connection in Peers)
                    connection.Dispose();
                Peers.Clear();
                listener?.Stop();

                metaFile?.Dispose();

                File.Delete(MetaFilePath);
            }
        }

        public MessagingServer(string editorExecutablePath, string projectMetadataDir, ILogger logger)
        {
            this.logger = logger;

            MetaFilePath = Path.Combine(projectMetadataDir, GodotIdeMetadata.DefaultFileName);

            // Make sure the directory exists
            Directory.CreateDirectory(projectMetadataDir);

            // The Godot editor's file system thread can keep the file open for writing, so we are forced to allow write sharing...
            const FileShare metaFileShare = FileShare.ReadWrite;

            metaFile = File.Open(MetaFilePath, FileMode.Create, FileAccess.Write, metaFileShare);

            listener = new TcpListener(new IPEndPoint(IPAddress.Loopback, port: 0));
            listener.Start();

            int port = ((IPEndPoint)listener.Server.LocalEndPoint).Port;
            using (var metaFileWriter = new StreamWriter(metaFile, Encoding.UTF8))
            {
                metaFileWriter.WriteLine(port);
                metaFileWriter.WriteLine(editorExecutablePath);
            }
        }

        private async Task AcceptClient(TcpClient tcpClient)
        {
            logger.LogDebug("Accept client...");

            using (var peer = new Peer(tcpClient, new ServerHandshake(), new ServerMessageHandler(), logger))
            {
                // ReSharper disable AccessToDisposedClosure
                peer.Connected += () =>
                {
                    logger.LogInfo("Connection open with Ide Client");

                    if (clientConnectedAwaiters.TryGetValue(peer.RemoteIdentity, out var queue))
                    {
                        while (queue.Count > 0)
                            queue.Dequeue().SetResult(true);
                        clientConnectedAwaiters.Remove(peer.RemoteIdentity);
                    }
                };

                peer.Disconnected += () =>
                {
                    if (clientDisconnectedAwaiters.TryGetValue(peer.RemoteIdentity, out var queue))
                    {
                        while (queue.Count > 0)
                            queue.Dequeue().SetResult(true);
                        clientDisconnectedAwaiters.Remove(peer.RemoteIdentity);
                    }
                };
                // ReSharper restore AccessToDisposedClosure

                try
                {
                    if (!await peer.DoHandshake("server"))
                    {
                        logger.LogError("Handshake failed");
                        return;
                    }
                }
                catch (Exception e)
                {
                    logger.LogError("Handshake failed with unhandled exception: ", e);
                    return;
                }

                using (await peersSem.UseAsync())
                    Peers.Add(peer);

                try
                {
                    await peer.Process();
                }
                finally
                {
                    using (await peersSem.UseAsync())
                        Peers.Remove(peer);
                }
            }
        }

        public async Task Listen()
        {
            try
            {
                while (!IsDisposed)
                    _ = AcceptClient(await listener.AcceptTcpClientAsync());
            }
            catch (Exception e)
            {
                if (!IsDisposed && !(e is SocketException se && se.SocketErrorCode == SocketError.Interrupted))
                    throw;
            }
        }

        public async void BroadcastRequest<TResponse>(string identity, Request request)
            where TResponse : Response, new()
        {
            using (await peersSem.UseAsync())
            {
                if (!IsAnyConnected(identity))
                {
                    logger.LogError("Cannot write request. No client connected to the Godot Ide Server.");
                    return;
                }

                var selectedConnections = string.IsNullOrEmpty(identity) ?
                    Peers :
                    Peers.Where(c => c.RemoteIdentity == identity);

                string body = JsonConvert.SerializeObject(request);

                foreach (var connection in selectedConnections)
                    _ = connection.SendRequest<TResponse>(request.Id, body);
            }
        }

        private class ServerHandshake : IHandshake
        {
            private static readonly string ServerHandshakeBase = $"{Peer.ServerHandshakeName},Version={Peer.ProtocolVersionMajor}.{Peer.ProtocolVersionMinor}.{Peer.ProtocolVersionRevision}";
            private static readonly string ClientHandshakePattern = $@"{Regex.Escape(Peer.ClientHandshakeName)},Version=([0-9]+)\.([0-9]+)\.([0-9]+),([_a-zA-Z][_a-zA-Z0-9]{{0,63}})";

            public string GetHandshakeLine(string identity) => $"{ServerHandshakeBase},{identity}";

            public bool IsValidPeerHandshake(string handshake, out string identity, ILogger logger)
            {
                identity = null;

                var match = Regex.Match(handshake, ClientHandshakePattern);

                if (!match.Success)
                    return false;

                if (!uint.TryParse(match.Groups[1].Value, out uint clientMajor) || Peer.ProtocolVersionMajor != clientMajor)
                {
                    logger.LogDebug("Incompatible major version: " + match.Groups[1].Value);
                    return false;
                }

                // ReSharper disable once ConditionIsAlwaysTrueOrFalse
                if (!uint.TryParse(match.Groups[2].Value, out uint clientMinor) || Peer.ProtocolVersionMinor > clientMinor)
                {
                    logger.LogDebug("Incompatible minor version: " + match.Groups[2].Value);
                    return false;
                }

                if (!uint.TryParse(match.Groups[3].Value, out uint _)) // Revision
                {
                    logger.LogDebug("Incompatible revision build: " + match.Groups[3].Value);
                    return false;
                }

                identity = match.Groups[4].Value;

                return true;
            }
        }

        private class ServerMessageHandler : IMessageHandler
        {
            private static void DispatchToMainThread(Action action)
            {
                var d = new SendOrPostCallback(state => action());
                Godot.Dispatcher.SynchronizationContext.Post(d, null);
            }

            private readonly Dictionary<string, Peer.RequestHandler> requestHandlers = InitializeRequestHandlers();

            public async Task<MessageContent> HandleRequest(Peer peer, string id, MessageContent content, ILogger logger)
            {
                if (!requestHandlers.TryGetValue(id, out var handler))
                {
                    logger.LogError($"Received unknown request: {id}");
                    return new MessageContent(MessageStatus.RequestNotSupported, "null");
                }

                try
                {
                    var response = await handler(peer, content);
                    return new MessageContent(response.Status, JsonConvert.SerializeObject(response));
                }
                catch (JsonException)
                {
                    logger.LogError($"Received request with invalid body: {id}");
                    return new MessageContent(MessageStatus.InvalidRequestBody, "null");
                }
            }

            private static Dictionary<string, Peer.RequestHandler> InitializeRequestHandlers()
            {
                return new Dictionary<string, Peer.RequestHandler>
                {
                    [PlayRequest.Id] = async (peer, content) =>
                    {
                        _ = JsonConvert.DeserializeObject<PlayRequest>(content.Body);
                        return await HandlePlay();
                    },
                    [DebugPlayRequest.Id] = async (peer, content) =>
                    {
                        var request = JsonConvert.DeserializeObject<DebugPlayRequest>(content.Body);
                        return await HandleDebugPlay(request);
                    },
                    [StopPlayRequest.Id] = async (peer, content) =>
                    {
                        var request = JsonConvert.DeserializeObject<StopPlayRequest>(content.Body);
                        return await HandleStopPlay(request);
                    },
                    [ReloadScriptsRequest.Id] = async (peer, content) =>
                    {
                        _ = JsonConvert.DeserializeObject<ReloadScriptsRequest>(content.Body);
                        return await HandleReloadScripts();
                    },
                    [CodeCompletionRequest.Id] = async (peer, content) =>
                    {
                        var request = JsonConvert.DeserializeObject<CodeCompletionRequest>(content.Body);
                        return await HandleCodeCompletionRequest(request);
                    }
                };
            }

            private static Task<Response> HandlePlay()
            {
                DispatchToMainThread(() =>
                {
                    // TODO: Add BuildBeforePlaying flag to PlayRequest

                    // Run the game
                    Internal.EditorRunPlay();
                });
                return Task.FromResult<Response>(new PlayResponse());
            }

            private static Task<Response> HandleDebugPlay(DebugPlayRequest request)
            {
                DispatchToMainThread(() =>
                {
                    // Tell the build callback whether the editor already built the solution or not
                    GodotSharpEditor.Instance.SkipBuildBeforePlaying = !(request.BuildBeforePlaying ?? true);

                    // Pass the debugger agent settings to the player via an environment variables
                    // TODO: It would be better if this was an argument in EditorRunPlay instead
                    Environment.SetEnvironmentVariable("GODOT_MONO_DEBUGGER_AGENT",
                        "--debugger-agent=transport=dt_socket" +
                        $",address={request.DebuggerHost}:{request.DebuggerPort}" +
                        ",server=n");

                    // Run the game
                    Internal.EditorRunPlay();

                    // Restore normal settings
                    Environment.SetEnvironmentVariable("GODOT_MONO_DEBUGGER_AGENT", "");
                    GodotSharpEditor.Instance.SkipBuildBeforePlaying = false;
                });
                return Task.FromResult<Response>(new DebugPlayResponse());
            }

            private static Task<Response> HandleStopPlay(StopPlayRequest request)
            {
                DispatchToMainThread(Internal.EditorRunStop);
                return Task.FromResult<Response>(new StopPlayResponse());
            }

            private static Task<Response> HandleReloadScripts()
            {
                DispatchToMainThread(Internal.ScriptEditorDebugger_ReloadScripts);
                return Task.FromResult<Response>(new ReloadScriptsResponse());
            }

            private static async Task<Response> HandleCodeCompletionRequest(CodeCompletionRequest request)
            {
                // This is needed if the "resource path" part of the path is case insensitive.
                // However, it doesn't fix resource loading if the rest of the path is also case insensitive.
                string scriptFileLocalized = FsPathUtils.LocalizePathWithCaseChecked(request.ScriptFile);

                var response = new CodeCompletionResponse {Kind = request.Kind, ScriptFile = request.ScriptFile};
                response.Suggestions = await Task.Run(() =>
                    Internal.CodeCompletionRequest(response.Kind, scriptFileLocalized ?? request.ScriptFile));
                return response;
            }
        }
    }
}
