using System;
using Path = System.IO.Path;

namespace GodotTools.IdeConnection
{
    public class GodotIdeBase : IDisposable
    {
        private ILogger logger;

        public ILogger Logger
        {
            get => logger ?? (logger = new ConsoleLogger());
            set => logger = value;
        }

        private readonly string projectMetadataDir;

        protected const string MetaFileName = "ide_server_meta.txt";
        protected string MetaFilePath => Path.Combine(projectMetadataDir, MetaFileName);

        private GodotIdeConnection connection;
        protected readonly object ConnectionLock = new object();

        public bool IsDisposed { get; private set; } = false;

        public bool IsConnected => connection != null && !connection.IsDisposed && connection.IsConnected;

        public event Action Connected
        {
            add
            {
                if (connection != null && !connection.IsDisposed)
                    connection.Connected += value;
            }
            remove
            {
                if (connection != null && !connection.IsDisposed)
                    connection.Connected -= value;
            }
        }

        protected GodotIdeConnection Connection
        {
            get => connection;
            set
            {
                connection?.Dispose();
                connection = value;
            }
        }

        protected GodotIdeBase(string projectMetadataDir)
        {
            this.projectMetadataDir = projectMetadataDir;
        }

        protected void DisposeConnection()
        {
            lock (ConnectionLock)
            {
                connection?.Dispose();
            }
        }

        ~GodotIdeBase()
        {
            Dispose(disposing: false);
        }

        public void Dispose()
        {
            if (IsDisposed)
                return;

            lock (ConnectionLock)
            {
                if (IsDisposed) // lock may not be fair
                    return;
                IsDisposed = true;
            }

            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                connection?.Dispose();
            }
        }
    }
}
