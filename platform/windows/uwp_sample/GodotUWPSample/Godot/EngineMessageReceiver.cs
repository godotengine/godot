// EngineMessageReceiver.cs
// Engine -> host half of the JSON message bus. Registers the native
// host-message callback and surfaces each incoming (method, argsJson) message.
//
// Two delivery modes:
//   * Asynchronous: OnMessage is raised on the SynchronizationContext captured
//     at construction (normally the UI thread), so handlers can touch UI.
//   * Synchronous: a handler registered with RegisterSyncHandler answers inline
//     on the ENGINE thread and its return value becomes the result of the
//     GDScript send_to_host() call. Sync handlers must be fast, must not touch
//     UI, and must never block on the UI thread (deadlock).

using System;
using System.Collections.Generic;
using System.Threading;

namespace Godot.Uwp.Embedding
{
    /// <summary>Payload for an engine -> host message.</summary>
    internal sealed class EngineMessageEventArgs : EventArgs
    {
        /// <summary>The method name sent from GDScript via send_to_host().</summary>
        public string Method;

        /// <summary>JSON-encoded array of arguments. Always non-null; "[]" if empty.</summary>
        public string ArgsJson;

        /// <summary>When the message was received (UTC).</summary>
        public DateTime Timestamp;

        /// <summary>Parses ArgsJson as a JSON array of strings; null on error.</summary>
        public string[] GetArgsAsStringArray()
        {
            try
            {
                var arr = Windows.Data.Json.JsonArray.Parse(ArgsJson);
                var result = new string[arr.Count];
                for (int i = 0; i < arr.Count; i++)
                {
                    result[i] = arr[i].ValueType == Windows.Data.Json.JsonValueType.String
                        ? arr[i].GetString()
                        : arr[i].Stringify();
                }
                return result;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[EngineMessageReceiver] Parse error: " + ex.Message);
                return null;
            }
        }
    }

    /// <summary>
    /// Receives messages from the embedded Godot engine (sent via the engine's
    /// host singleton, e.g. UWPHost.send_to_host(method, args)).
    /// </summary>
    internal sealed class EngineMessageReceiver : IDisposable
    {
        private readonly SynchronizationContext _synchronizationContext;
        private readonly Dictionary<string, Func<string, string>> _syncHandlers =
            new Dictionary<string, Func<string, string>>();
        private bool _isInitialized;
        private bool _isDisposed;

        /// <summary>Construct on the UI thread so OnMessage is raised there.</summary>
        public EngineMessageReceiver()
        {
            _synchronizationContext = SynchronizationContext.Current;
        }

        /// <summary>Raised (async, on the captured context) for every incoming message.</summary>
        public event EventHandler<EngineMessageEventArgs> OnMessage;

        /// <summary>
        /// Registers a SYNCHRONOUS handler for a method: GDScript's send_to_host
        /// blocks until it returns, and the return value (a JSON document)
        /// becomes send_to_host's result.
        ///
        /// Runs on the ENGINE thread mid-frame — keep it fast, don't touch UI,
        /// and never block on the UI thread.
        /// </summary>
        public void RegisterSyncHandler(string method, Func<string, string> handler)
        {
            lock (_syncHandlers)
            {
                _syncHandlers[method] = handler;
            }
        }

        public void UnregisterSyncHandler(string method)
        {
            lock (_syncHandlers)
            {
                _syncHandlers.Remove(method);
            }
        }

        /// <summary>
        /// Registers the native host-message callback. Call BEFORE
        /// GodotEngineHost.Start so messages emitted during script _ready are
        /// not dropped.
        /// </summary>
        public bool Initialize()
        {
            if (_isInitialized)
            {
                return true;
            }
            try
            {
                GodotEngineHost.SetHostMessageHandler(HandleHostMessage);
                _isInitialized = true;
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[EngineMessageReceiver] Initialization failed: " + ex.Message);
                return false;
            }
        }

        public void Shutdown()
        {
            if (!_isInitialized)
            {
                return;
            }
            try { GodotEngineHost.SetHostMessageHandler(null); }
            catch (Exception ex) { System.Diagnostics.Debug.WriteLine("[EngineMessageReceiver] Shutdown error: " + ex.Message); }
            _isInitialized = false;
        }

        // Invoked on the engine thread by the native bridge.
        private string HandleHostMessage(string method, string argsJson)
        {
            try
            {
                // Synchronous path first: GDScript blocks on send_to_host until
                // this returns, so the value is delivered inline.
                Func<string, string> syncHandler = null;
                lock (_syncHandlers)
                {
                    _syncHandlers.TryGetValue(method, out syncHandler);
                }
                if (syncHandler != null)
                {
                    try
                    {
                        return syncHandler(argsJson);
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine(
                            "[EngineMessageReceiver] Sync handler '" + method + "' threw: " + ex.Message);
                        return null;
                    }
                }

                var args = new EngineMessageEventArgs
                {
                    Method = method,
                    ArgsJson = argsJson,
                    Timestamp = DateTime.UtcNow,
                };

                EventHandler<EngineMessageEventArgs> handler = OnMessage;
                if (handler != null)
                {
                    // Hand off to the captured context; never run user code on
                    // the engine thread. Replies (if any) flow back via the sender.
                    if (_synchronizationContext != null)
                    {
                        _synchronizationContext.Post(_ => handler(this, args), null);
                    }
                    else
                    {
                        handler(this, args);
                    }
                }
                return null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine("[EngineMessageReceiver] Error handling message: " + ex.Message);
                return null;
            }
        }

        public void Dispose()
        {
            if (_isDisposed)
            {
                return;
            }
            Shutdown();
            _isDisposed = true;
        }
    }
}
