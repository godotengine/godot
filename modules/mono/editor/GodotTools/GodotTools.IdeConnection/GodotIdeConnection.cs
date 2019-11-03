using System;
using System.Diagnostics;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace GodotTools.IdeConnection
{
    public abstract class GodotIdeConnection : IDisposable
    {
        protected const string Version = "1.0";

        protected static readonly string ClientHandshake = $"Godot Ide Client Version {Version}";
        protected static readonly string ServerHandshake = $"Godot Ide Server Version {Version}";

        private const int ClientWriteTimeout = 8000;
        private readonly TcpClient tcpClient;

        private TextReader clientReader;
        private TextWriter clientWriter;

        private readonly object writeLock = new object();

        private readonly Func<Message, bool> messageHandler;

        public event Action Connected;

        private ILogger logger;

        public ILogger Logger
        {
            get => logger ?? (logger = new ConsoleLogger());
            set => logger = value;
        }

        public bool IsDisposed { get; private set; } = false;

        public bool IsConnected => tcpClient.Client != null && tcpClient.Client.Connected;

        protected GodotIdeConnection(TcpClient tcpClient, Func<Message, bool> messageHandler)
        {
            this.tcpClient = tcpClient;
            this.messageHandler = messageHandler;
        }

        public void Start()
        {
            try
            {
                if (!StartConnection())
                    return;

                string messageLine;
                while ((messageLine = ReadLine()) != null)
                {
                    if (!MessageParser.TryParse(messageLine, out Message msg))
                    {
                        Logger.LogError($"Received message with invalid format: {messageLine}");
                        continue;
                    }

                    Logger.LogDebug($"Received message: {msg}");

                    if (msg.Id == "close")
                    {
                        Logger.LogInfo("Closing connection");
                        return;
                    }

                    try
                    {
                        try
                        {
                            Debug.Assert(messageHandler != null);

                            if (!messageHandler(msg))
                                Logger.LogError($"Received unknown message: {msg}");
                        }
                        catch (Exception e)
                        {
                            Logger.LogError($"Message handler for '{msg}' failed with exception", e);
                        }
                    }
                    catch (Exception e)
                    {
                        Logger.LogError($"Exception thrown from message handler. Message: {msg}", e);
                    }
                }
            }
            catch (Exception e)
            {
                Logger.LogError($"Unhandled exception in the Godot Ide Connection thread", e);
            }
            finally
            {
                Dispose();
            }
        }

        private bool StartConnection()
        {
            NetworkStream clientStream = tcpClient.GetStream();

            clientReader = new StreamReader(clientStream, Encoding.UTF8);

            lock (writeLock)
                clientWriter = new StreamWriter(clientStream, Encoding.UTF8);

            clientStream.WriteTimeout = ClientWriteTimeout;

            if (!WriteHandshake())
            {
                Logger.LogError("Could not write handshake");
                return false;
            }

            if (!IsValidResponseHandshake(ReadLine()))
            {
                Logger.LogError("Received invalid handshake");
                return false;
            }

            Connected?.Invoke();

            Logger.LogInfo("Godot Ide connection started");

            return true;
        }

        private string ReadLine()
        {
            try
            {
                return clientReader?.ReadLine();
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

        public bool WriteMessage(Message message)
        {
            Logger.LogDebug($"Sending message {message}");
            
            var messageComposer = new MessageComposer();

            messageComposer.AddArgument(message.Id);
            foreach (string argument in message.Arguments)
                messageComposer.AddArgument(argument);

            return WriteLine(messageComposer.ToString());
        }

        protected bool WriteLine(string text)
        {
            if (clientWriter == null || IsDisposed || !IsConnected)
                return false;

            lock (writeLock)
            {
                try
                {
                    clientWriter.WriteLine(text);
                    clientWriter.Flush();
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

        protected abstract bool WriteHandshake();
        protected abstract bool IsValidResponseHandshake(string handshakeLine);

        public void Dispose()
        {
            if (IsDisposed)
                return;

            IsDisposed = true;

            clientReader?.Dispose();
            clientWriter?.Dispose();
            ((IDisposable) tcpClient)?.Dispose();
        }
    }
}
