using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using GodotTools.IdeMessaging.Requests;
using GodotTools.IdeMessaging.Utils;

namespace GodotTools.IdeMessaging
{
    public sealed class Peer : IDisposable
    {
        /// <summary>
        /// Major version.
        /// There is no forward nor backward compatibility between different major versions.
        /// Connection is refused if client and server have different major versions.
        /// </summary>
        public static readonly int ProtocolVersionMajor = Assembly.GetAssembly(typeof(Peer)).GetName().Version.Major;

        /// <summary>
        /// Minor version, which clients must be backward compatible with.
        /// Connection is refused if the client's minor version is lower than the server's.
        /// </summary>
        public static readonly int ProtocolVersionMinor = Assembly.GetAssembly(typeof(Peer)).GetName().Version.Minor;

        /// <summary>
        /// Revision, which doesn't affect compatibility.
        /// </summary>
        public static readonly int ProtocolVersionRevision = Assembly.GetAssembly(typeof(Peer)).GetName().Version.Revision;

        public const string ClientHandshakeName = "GodotIdeClient";
        public const string ServerHandshakeName = "GodotIdeServer";

        private const int ClientWriteTimeout = 8000;

        public delegate Task<Response> RequestHandler(Peer peer, MessageContent content);

        private readonly TcpClient tcpClient;

        private readonly TextReader clientReader;
        private readonly TextWriter clientWriter;

        private readonly SemaphoreSlim writeSem = new SemaphoreSlim(1);

        private string remoteIdentity = string.Empty;
        public string RemoteIdentity => remoteIdentity;

        public event Action Connected;
        public event Action Disconnected;

        private ILogger Logger { get; }

        public bool IsDisposed { get; private set; }

        public bool IsTcpClientConnected => tcpClient.Client != null && tcpClient.Client.Connected;

        private bool IsConnected { get; set; }

        private readonly IHandshake handshake;
        private readonly IMessageHandler messageHandler;

        private readonly Dictionary<string, Queue<ResponseAwaiter>> requestAwaiterQueues = new Dictionary<string, Queue<ResponseAwaiter>>();
        private readonly SemaphoreSlim requestsSem = new SemaphoreSlim(1);

        public Peer(TcpClient tcpClient, IHandshake handshake, IMessageHandler messageHandler, ILogger logger)
        {
            this.tcpClient = tcpClient;
            this.handshake = handshake;
            this.messageHandler = messageHandler;

            Logger = logger;

            NetworkStream clientStream = tcpClient.GetStream();
            clientStream.WriteTimeout = ClientWriteTimeout;

            clientReader = new StreamReader(clientStream, Encoding.UTF8);
            clientWriter = new StreamWriter(clientStream, Encoding.UTF8) {NewLine = "\n"};
        }

        public async Task Process()
        {
            try
            {
                var decoder = new MessageDecoder();

                string messageLine;
                while ((messageLine = await ReadLine()) != null)
                {
                    var state = decoder.Decode(messageLine, out var msg);

                    if (state == MessageDecoder.State.Decoding)
                        continue; // Not finished decoding yet

                    if (state == MessageDecoder.State.Errored)
                    {
                        Logger.LogError($"Received message line with invalid format: {messageLine}");
                        continue;
                    }

                    Logger.LogDebug($"Received message: {msg}");

                    try
                    {
                        if (msg.Kind == MessageKind.Request)
                        {
                            var responseContent = await messageHandler.HandleRequest(this, msg.Id, msg.Content, Logger);
                            await WriteMessage(new Message(MessageKind.Response, msg.Id, responseContent));
                        }
                        else if (msg.Kind == MessageKind.Response)
                        {
                            ResponseAwaiter responseAwaiter;

                            using (await requestsSem.UseAsync())
                            {
                                if (!requestAwaiterQueues.TryGetValue(msg.Id, out var queue) || queue.Count <= 0)
                                {
                                    Logger.LogError($"Received unexpected response: {msg.Id}");
                                    return;
                                }

                                responseAwaiter = queue.Dequeue();
                            }

                            responseAwaiter.SetResult(msg.Content);
                        }
                        else
                        {
                            throw new IndexOutOfRangeException($"Invalid message kind {msg.Kind}");
                        }
                    }
                    catch (Exception e)
                    {
                        Logger.LogError($"Message handler for '{msg}' failed with exception", e);
                    }
                }
            }
            catch (Exception e)
            {
                if (!IsDisposed || !(e is SocketException || e.InnerException is SocketException))
                {
                    Logger.LogError("Unhandled exception in the peer loop", e);
                }
            }
        }

        public async Task<bool> DoHandshake(string identity)
        {
            if (!await WriteLine(handshake.GetHandshakeLine(identity)))
            {
                Logger.LogError("Could not write handshake");
                return false;
            }

            var readHandshakeTask = ReadLine();

            if (await Task.WhenAny(readHandshakeTask, Task.Delay(8000)) != readHandshakeTask)
            {
                Logger.LogError("Timeout waiting for the client handshake");
                return false;
            }

            string peerHandshake = await readHandshakeTask;

            if (handshake == null || !handshake.IsValidPeerHandshake(peerHandshake, out remoteIdentity, Logger))
            {
                Logger.LogError("Received invalid handshake: " + peerHandshake);
                return false;
            }

            IsConnected = true;
            Connected?.Invoke();

            Logger.LogInfo("Peer connection started");

            return true;
        }

        private async Task<string> ReadLine()
        {
            try
            {
                return await clientReader.ReadLineAsync();
            }
            catch (Exception e)
            {
                if (IsDisposed)
                {
                    var se = e as SocketException ?? e.InnerException as SocketException;
                    if (se != null && se.SocketErrorCode == SocketError.Interrupted)
                        return null;
                }

                throw;
            }
        }

        private Task<bool> WriteMessage(Message message)
        {
            Logger.LogDebug($"Sending message: {message}");
            int bodyLineCount = message.Content.Body.Count(c => c == '\n');

            bodyLineCount += 1; // Extra line break at the end

            var builder = new StringBuilder();

            builder.AppendLine(message.Kind.ToString());
            builder.AppendLine(message.Id);
            builder.AppendLine(message.Content.Status.ToString());
            builder.AppendLine(bodyLineCount.ToString());
            builder.AppendLine(message.Content.Body);

            return WriteLine(builder.ToString());
        }

        public async Task<TResponse> SendRequest<TResponse>(string id, string body)
            where TResponse : Response, new()
        {
            ResponseAwaiter responseAwaiter;

            using (await requestsSem.UseAsync())
            {
                bool written = await WriteMessage(new Message(MessageKind.Request, id, new MessageContent(body)));

                if (!written)
                    return null;

                if (!requestAwaiterQueues.TryGetValue(id, out var queue))
                {
                    queue = new Queue<ResponseAwaiter>();
                    requestAwaiterQueues.Add(id, queue);
                }

                responseAwaiter = new ResponseAwaiter<TResponse>();
                queue.Enqueue(responseAwaiter);
            }

            return (TResponse)await responseAwaiter;
        }

        private async Task<bool> WriteLine(string text)
        {
            if (clientWriter == null || IsDisposed || !IsTcpClientConnected)
                return false;

            using (await writeSem.UseAsync())
            {
                try
                {
                    await clientWriter.WriteLineAsync(text);
                    await clientWriter.FlushAsync();
                }
                catch (Exception e)
                {
                    if (!IsDisposed)
                    {
                        var se = e as SocketException ?? e.InnerException as SocketException;
                        if (se != null && se.SocketErrorCode == SocketError.Shutdown)
                            Logger.LogInfo("Client disconnected ungracefully");
                        else
                            Logger.LogError("Exception thrown when trying to write to client", e);

                        Dispose();
                    }
                }
            }

            return true;
        }

        // ReSharper disable once UnusedMember.Global
        public void ShutdownSocketSend()
        {
            tcpClient.Client.Shutdown(SocketShutdown.Send);
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;

            IsDisposed = true;

            if (IsTcpClientConnected)
            {
                if (IsConnected)
                    Disconnected?.Invoke();
            }

            clientReader?.Dispose();
            clientWriter?.Dispose();
            ((IDisposable)tcpClient)?.Dispose();
        }
    }
}
