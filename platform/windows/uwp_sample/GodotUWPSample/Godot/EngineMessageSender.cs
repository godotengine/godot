// EngineMessageSender.cs
// Host -> engine half of the JSON message bus. Invokes GDScript handlers
// (registered via the engine host singleton's register_handler) through
// GodotEngineHost, which marshals every call onto the engine thread.
//
//   Post(method, argsJson)      fire-and-forget
//   Call(method, argsJson)      synchronous, returns the handler's JSON result
//   CallAsync(method, argsJson) awaitable variant

using System;

namespace Godot.Uwp.Embedding
{
    /// <summary>
    /// Sends messages from the host into the embedded Godot engine. All calls
    /// are marshalled onto the engine thread by the GodotEngineHost passed to
    /// the constructor. argsJson is a JSON document (typically an array).
    /// </summary>
    internal sealed class EngineMessageSender : IDisposable
    {
        private readonly GodotEngineHost _host;
        private bool _isDisposed;

        public EngineMessageSender(GodotEngineHost host)
        {
            if (host == null)
            {
                throw new ArgumentNullException("host");
            }
            _host = host;
        }

        /// <summary>Fire-and-forget: invokes a registered GDScript handler on the engine thread.</summary>
        public void Post(string method, string argsJson)
        {
            if (string.IsNullOrEmpty(method))
            {
                throw new ArgumentException("method must be non-empty.", "method");
            }
            _host.Post(() =>
            {
                try
                {
                    GodotEngineHost.CallEngineRaw(method, argsJson);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine("[EngineMessageSender] Post '" + method + "' failed: " + ex.Message);
                }
            });
        }

        /// <summary>
        /// Invokes a registered GDScript handler on the engine thread and blocks
        /// for its JSON return value.
        /// </summary>
        public string Call(string method, string argsJson)
        {
            if (string.IsNullOrEmpty(method))
            {
                throw new ArgumentException("method must be non-empty.", "method");
            }
            return _host.Invoke(() => GodotEngineHost.CallEngineRaw(method, argsJson));
        }

        /// <summary>Awaitable variant of <see cref="Call"/>.</summary>
        public System.Threading.Tasks.Task<string> CallAsync(string method, string argsJson)
        {
            if (string.IsNullOrEmpty(method))
            {
                throw new ArgumentException("method must be non-empty.", "method");
            }
            return _host.InvokeAsync(() => GodotEngineHost.CallEngineRaw(method, argsJson));
        }

        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }
            _isDisposed = true;
        }
    }
}
