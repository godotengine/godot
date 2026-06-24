// GodotEngineHost.cs
// Runs the in-process Godot engine's ENTIRE lifecycle — setup, start, the
// per-frame iteration loop, and shutdown — on a single DEDICATED thread,
// never the UWP UI thread.
//
// Port of the proven WinUI3 embedding host (Godot.WinUI3.Embedding) to UWP:
//  - CoreDispatcher (not DispatcherQueue) for the synchronous UI-thread hop
//    the engine needs to bind its swap chain to the SwapChainPanel.
//  - No HWND anywhere: the engine renders into a composition swap chain
//    bound to the panel, and all input is injected through this class.
//  - Conservative C#/marshalling for .NET Native compatibility.

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Windows.UI.Core;

namespace Godot.Uwp.Embedding
{
    internal enum EngineState
    {
        Stopped,
        Starting,
        Running,
        Stopping,
    }

    internal sealed class GodotEngineHost : IDisposable
    {
        // --- Configuration (set before Start) ------------------------------

        /// <summary>Godot project: a directory (run with --path) or .pck (--main-pack).</summary>
        public string ProjectPath { get; set; }

        /// <summary>Extra args appended after the required embedded/d3d12 args.</summary>
        public string[] ExtraArgs { get; set; }

        public int FrameIntervalMs { get; set; }

        // --- Engine thread + work queue ------------------------------------

        private Thread _engineThread;
        private readonly ConcurrentQueue<Action> _work = new ConcurrentQueue<Action>();
        private volatile bool _stopRequested;
        private volatile bool _paused;
        private volatile int _state = (int)EngineState.Stopped;

        private IntPtr _panelUnknown; // IUnknown* of the SwapChainPanel, owned until handed to engine.
        private int _initWidthPx, _initHeightPx;
        private float _initScaleX = 1.0f, _initScaleY = 1.0f;

        private CoreDispatcher _uiDispatcher;

        private bool _disposed;

        // Pinned delegates — must outlive the native registrations.
        private static GodotNative.LogCallback _logCallbackPin;
        private static GodotNative.UiDispatchFunc _uiDispatchPin;
        private static GodotEngineHost _current;

        public EngineState State { get { return (EngineState)_state; } }
        public bool IsRunning { get { return State == EngineState.Running; } }

        /// <summary>Raised on the engine thread when the engine stops (true = engine-requested quit).</summary>
        public event Action<bool> Stopped;

        /// <summary>Raised on the engine thread for every Godot log line.</summary>
        public event Action<string, GodotLogLevel> Log;

        public GodotEngineHost()
        {
            FrameIntervalMs = 16;
        }

        // -------------------------------------------------------------------
        // Lifecycle
        // -------------------------------------------------------------------

        /// <summary>
        /// Spawns the engine thread and brings the engine up there. Returns
        /// immediately. Must be called on the UI thread (captures the
        /// CoreDispatcher and the panel's IUnknown).
        /// </summary>
        /// <param name="panelUnknown">
        /// IUnknown* for the SwapChainPanel (Marshal.GetIUnknownForObject).
        /// Ownership transfers to this host (released after engine AddRefs).
        /// </param>
        public void Start(IntPtr panelUnknown, int initialWidthPx, int initialHeightPx,
            float initialScaleX, float initialScaleY, CoreDispatcher uiDispatcher)
        {
            if (_disposed)
            {
                throw new ObjectDisposedException("GodotEngineHost");
            }
            if (State != EngineState.Stopped)
            {
                Debug.WriteLine("[GodotEngineHost] Start() ignored — engine not stopped.");
                return;
            }
            if (string.IsNullOrEmpty(ProjectPath))
            {
                throw new InvalidOperationException("ProjectPath must be set before Start().");
            }

            _panelUnknown = panelUnknown;
            _initWidthPx = initialWidthPx;
            _initHeightPx = initialHeightPx;
            _initScaleX = initialScaleX;
            _initScaleY = initialScaleY;
            _uiDispatcher = uiDispatcher;

            _stopRequested = false;
            _state = (int)EngineState.Starting;
            _current = this;

            OpenLogFile();

            _engineThread = new Thread(EngineThreadProc);
            _engineThread.Name = "GodotEngine";
            _engineThread.IsBackground = true;
            _engineThread.Start();
        }

        /// <summary>Pauses the iteration loop (call from OnSuspending).</summary>
        public void Pause()
        {
            _paused = true;
            WriteLogLine("Host", "Engine paused (app suspending).");
        }

        /// <summary>Resumes the iteration loop (call from Resuming).</summary>
        public void Resume()
        {
            _paused = false;
            WriteLogLine("Host", "Engine resumed.");
        }

        /// <summary>Requests shutdown and blocks until the engine thread exits.</summary>
        public void Stop()
        {
            Thread thread = _engineThread;
            if (thread == null || State == EngineState.Stopped)
            {
                return;
            }
            if (Thread.CurrentThread == thread)
            {
                _stopRequested = true;
                return;
            }
            _state = (int)EngineState.Stopping;
            _stopRequested = true;
            thread.Join();
            _engineThread = null;
        }

        // -------------------------------------------------------------------
        // Engine thread
        // -------------------------------------------------------------------

        private void EngineThreadProc()
        {
            bool wantsQuit = false;
            bool setUp = false;
            try
            {
                _logCallbackPin = OnNativeLog;
                GodotNative.godot_uwp_set_log_callback(_logCallbackPin);

                _uiDispatchPin = OnNativeUiDispatch;
                GodotNative.godot_uwp_set_ui_dispatcher(_uiDispatchPin);

                // Hand the panel to the engine (it AddRefs internally).
                GodotNative.godot_uwp_set_swap_chain_panel(_panelUnknown);
                if (_panelUnknown != IntPtr.Zero)
                {
                    Marshal.Release(_panelUnknown);
                    _panelUnknown = IntPtr.Zero;
                }

                // Seed initial size/scale BEFORE setup so the display server
                // creates the swap chain at the right dimensions.
                GodotNative.godot_uwp_set_composition_scale(_initScaleX, _initScaleY);
                GodotNative.godot_uwp_notify_panel_resize(_initWidthPx, _initHeightPx);

                // AppContainer cannot write to the real %APPDATA%. Godot
                // resolves user:// via the APPDATA env var, so point it at
                // the package's writable LocalState before engine setup.
                string localState = Windows.Storage.ApplicationData.Current.LocalFolder.Path;
                Environment.SetEnvironmentVariable("APPDATA", localState);
                Environment.SetEnvironmentVariable("LOCALAPPDATA", localState);

                // The D3D12 Agility SDK loader probes ".\x86_64" then ".\"
                // relative to the process CWD (System32 under UWP activation).
                // Point the CWD at the package root where D3D12Core.dll lives.
                try
                {
                    System.IO.Directory.SetCurrentDirectory(
                        Windows.ApplicationModel.Package.Current.InstalledLocation.Path);
                }
                catch (Exception ex)
                {
                    WriteLogLine("Host", "SetCurrentDirectory failed: " + ex.Message);
                }

                bool isPck = ProjectPath.EndsWith(".pck", StringComparison.OrdinalIgnoreCase);
                var args = new System.Collections.Generic.List<string>
                {
                    "godot",
                    "--display-driver", "embedded",
                    "--rendering-driver", "d3d12",
                    isPck ? "--main-pack" : "--path", ProjectPath,
                    "--verbose",
                };
                if (ExtraArgs != null)
                {
                    args.AddRange(ExtraArgs);
                }

                WriteLogLine("Host", "EngineSetup args: " + string.Join(" ", args));
                if (!GodotNative.EngineSetup(args.ToArray()))
                {
                    WriteLogLine("Host", "EngineSetup FAILED.");
                    return;
                }
                setUp = true;

                WriteLogLine("Host", "EngineStart...");
                if (GodotNative.godot_uwp_engine_start() == 0)
                {
                    WriteLogLine("Host", "EngineStart FAILED.");
                    return;
                }
                WriteLogLine("Host", "Engine running.");
                _state = (int)EngineState.Running;

                var sw = Stopwatch.StartNew();
                while (!_stopRequested)
                {
                    // Suspended UWP apps must stop presenting or PLM terminates
                    // the process. Idle the loop while paused.
                    if (_paused)
                    {
                        Thread.Sleep(50);
                        continue;
                    }

                    DrainWork();
                    if (_stopRequested)
                    {
                        break;
                    }

                    if (GodotNative.godot_uwp_engine_iteration() != 0)
                    {
                        wantsQuit = true;
                        break;
                    }

                    long remaining = FrameIntervalMs - sw.ElapsedMilliseconds;
                    if (remaining > 0)
                    {
                        Thread.Sleep((int)remaining);
                    }
                    sw.Restart();
                }
            }
            catch (Exception ex)
            {
                WriteLogLine("Host", "Engine thread crashed: " + ex);
            }
            finally
            {
                DrainWork();
                if (setUp)
                {
                    try { GodotNative.godot_uwp_engine_shutdown(); }
                    catch (Exception ex) { WriteLogLine("Host", "Shutdown threw: " + ex); }
                }
                GodotNative.godot_uwp_set_log_callback(null);
                GodotNative.godot_uwp_set_ui_dispatcher(null);
                _state = (int)EngineState.Stopped;
                _current = null;
                CloseLogFile();

                var handler = Stopped;
                if (handler != null)
                {
                    try { handler(wantsQuit); }
                    catch (Exception ex) { Debug.WriteLine("[GodotEngineHost] Stopped handler threw: " + ex); }
                }
            }
        }

        private void DrainWork()
        {
            Action work;
            while (_work.TryDequeue(out work))
            {
                try { work(); }
                catch (Exception ex) { WriteLogLine("Host", "Queued work threw: " + ex); }
            }
        }

        // -------------------------------------------------------------------
        // Native callbacks
        // -------------------------------------------------------------------

        private static void OnNativeLog(IntPtr messageUtf8, int level)
        {
            string msg = GodotNative.Utf8ToString(messageUtf8);
            string tag = level == 2 ? "Error" : (level == 1 ? "Warn" : "Print");
            Debug.WriteLine("[Godot/" + tag + "] " + msg);
            WriteLogLine(tag, msg);

            GodotEngineHost host = _current;
            if (host != null)
            {
                var handler = host.Log;
                if (handler != null)
                {
                    try { handler(msg, (GodotLogLevel)level); } catch { }
                }
            }
        }

        // The engine calls this FROM the engine thread; the work must run
        // synchronously ON the UI thread (SetSwapChain is UI-thread affine).
        // NOTE: no exception may escape this method — a managed exception
        // crossing the reverse-P/Invoke boundary fail-fasts .NET Native.
        private static void OnNativeUiDispatch(IntPtr work, IntPtr userdata)
        {
            try
            {
                GodotEngineHost host = _current;
                var workFn = Marshal.GetDelegateForFunctionPointer<GodotNative.WorkFunc>(work);

                CoreDispatcher dispatcher = host != null ? host._uiDispatcher : null;
                if (dispatcher == null || dispatcher.HasThreadAccess)
                {
                    workFn(userdata);
                    return;
                }

                using (var done = new ManualResetEventSlim(false))
                {
                    Exception error = null;
                    var ignore = dispatcher.RunAsync(CoreDispatcherPriority.High, () =>
                    {
                        try { workFn(userdata); }
                        catch (Exception ex) { error = ex; }
                        finally { done.Set(); }
                    });
                    done.Wait();
                    if (error != null)
                    {
                        WriteLogLine("Host", "UI dispatch work threw: " + error);
                    }
                }
            }
            catch (Exception ex)
            {
                WriteLogLine("Host", "UI dispatch failed: " + ex);
            }
        }

        // -------------------------------------------------------------------
        // Host -> engine (marshalled onto the engine thread)
        // -------------------------------------------------------------------

        public void Post(Action work)
        {
            if (work == null)
            {
                return;
            }
            EngineState s = State;
            if (s == EngineState.Stopping || s == EngineState.Stopped)
            {
                return;
            }
            _work.Enqueue(work);
        }

        /// <summary>
        /// Runs <paramref name="func"/> on the engine thread and blocks until it
        /// completes, returning its result. Runs inline when already on the
        /// engine thread.
        /// </summary>
        public T Invoke<T>(Func<T> func)
        {
            if (func == null)
            {
                throw new ArgumentNullException("func");
            }
            if (Thread.CurrentThread == _engineThread)
            {
                return func();
            }
            EngineState s = State;
            if (s == EngineState.Stopping || s == EngineState.Stopped)
            {
                throw new InvalidOperationException("Cannot Invoke — engine is not running.");
            }

            using (var done = new ManualResetEventSlim(false))
            {
                T result = default(T);
                Exception error = null;
                _work.Enqueue(() =>
                {
                    try { result = func(); }
                    catch (Exception ex) { error = ex; }
                    finally { done.Set(); }
                });
                done.Wait();
                if (error != null)
                {
                    throw error;
                }
                return result;
            }
        }

        /// <summary>Runs <paramref name="func"/> on the engine thread asynchronously.</summary>
        public System.Threading.Tasks.Task<T> InvokeAsync<T>(Func<T> func)
        {
            if (func == null)
            {
                throw new ArgumentNullException("func");
            }
            if (Thread.CurrentThread == _engineThread)
            {
                try { return System.Threading.Tasks.Task.FromResult(func()); }
                catch (Exception ex) { return System.Threading.Tasks.Task.FromException<T>(ex); }
            }
            EngineState s = State;
            if (s == EngineState.Stopping || s == EngineState.Stopped)
            {
                throw new InvalidOperationException("Cannot InvokeAsync — engine is not running.");
            }

            var tcs = new System.Threading.Tasks.TaskCompletionSource<T>(
                System.Threading.Tasks.TaskCreationOptions.RunContinuationsAsynchronously);
            _work.Enqueue(() =>
            {
                try { tcs.SetResult(func()); }
                catch (Exception ex) { tcs.SetException(ex); }
            });
            return tcs.Task;
        }

        // -------------------------------------------------------------------
        // Host <-> engine JSON message bus
        // -------------------------------------------------------------------

        /// <summary>Engine->host message handler. Return a JSON reply or null.</summary>
        public delegate string HostMessageHandler(string method, string argsJson);

        private static GodotNative.HostMsgCallback _hostMsgPin;
        private static HostMessageHandler _hostMsgHandler;

        /// <summary>
        /// Installs the engine->host message handler (GDScript
        /// UWPHost.send_to_host). Call BEFORE Start so messages emitted during
        /// _ready are not dropped. The handler runs on the ENGINE thread and must
        /// not throw. Pass null to clear.
        /// </summary>
        public static void SetHostMessageHandler(HostMessageHandler handler)
        {
            _hostMsgHandler = handler;
            if (handler == null)
            {
                GodotNative.godot_uwp_set_host_message_callback(null);
                _hostMsgPin = null;
                return;
            }

            if (_hostMsgPin == null)
            {
                _hostMsgPin = OnNativeHostMessage;
            }
            GodotNative.godot_uwp_set_host_message_callback(_hostMsgPin);
        }

        private static void OnNativeHostMessage(IntPtr methodUtf8, IntPtr argsUtf8)
        {
            try
            {
                HostMessageHandler handler = _hostMsgHandler;
                if (handler == null)
                {
                    return;
                }
                string method = GodotNative.Utf8ToString(methodUtf8);
                string args = GodotNative.Utf8ToString(argsUtf8);

                string ret = null;
                try { ret = handler(method, args); }
                catch (Exception ex) { WriteLogLine("Host", "Host message handler '" + method + "' threw: " + ex); }

                if (ret != null)
                {
                    IntPtr retUtf8 = GodotNative.Utf8Alloc(ret);
                    try { GodotNative.godot_uwp_set_call_return(retUtf8); }
                    finally { Marshal.FreeHGlobal(retUtf8); }
                }
            }
            catch (Exception ex)
            {
                // Never let an exception cross the reverse-P/Invoke boundary.
                WriteLogLine("Host", "OnNativeHostMessage failed: " + ex);
            }
        }

        /// <summary>
        /// Invokes a GDScript handler registered via UWPHost.register_handler.
        /// Engine-thread affine — call through Post/Invoke/InvokeAsync.
        /// </summary>
        public static string CallEngineRaw(string method, string argsJson)
        {
            if (string.IsNullOrEmpty(method))
            {
                throw new ArgumentException("method must be non-empty.", "method");
            }

            IntPtr methodUtf8 = GodotNative.Utf8Alloc(method);
            IntPtr argsUtf8 = argsJson != null ? GodotNative.Utf8Alloc(argsJson) : IntPtr.Zero;
            IntPtr retUtf8 = IntPtr.Zero;
            try
            {
                int ok = GodotNative.godot_uwp_call_engine(methodUtf8, argsUtf8, out retUtf8);
                if (ok == 0)
                {
                    throw new InvalidOperationException("UWPHost bridge is not initialised. Start the engine first.");
                }
                return retUtf8 == IntPtr.Zero ? null : GodotNative.Utf8ToString(retUtf8);
            }
            finally
            {
                if (retUtf8 != IntPtr.Zero)
                {
                    GodotNative.godot_uwp_free_string(retUtf8);
                }
                if (argsUtf8 != IntPtr.Zero)
                {
                    Marshal.FreeHGlobal(argsUtf8);
                }
                Marshal.FreeHGlobal(methodUtf8);
            }
        }

        public void ConfigurePanel(double widthPx, double heightPx, float scaleX, float scaleY)
        {
            int w = (int)widthPx, h = (int)heightPx;
            Post(() =>
            {
                GodotNative.godot_uwp_set_composition_scale(scaleX, scaleY);
                GodotNative.godot_uwp_notify_panel_resize(w, h);
            });
        }

        public void InjectMouseButton(GodotMouseButton button, bool pressed, float x, float y)
        {
            Post(() => GodotNative.godot_uwp_inject_mouse_button((int)button, pressed ? 1 : 0, x, y, 0));
        }

        public void InjectMouseMotion(float x, float y, float relX, float relY)
        {
            Post(() => GodotNative.godot_uwp_inject_mouse_motion(x, y, relX, relY));
        }

        public void InjectMouseWheel(float x, float y, float deltaX, float deltaY)
        {
            Post(() => GodotNative.godot_uwp_inject_mouse_wheel(x, y, deltaX, deltaY));
        }

        public void InjectKey(uint winVk, bool pressed, bool echo, uint unicode)
        {
            Post(() => GodotNative.godot_uwp_inject_key(winVk, pressed ? 1 : 0, echo ? 1 : 0, unicode));
        }

        // -------------------------------------------------------------------
        // File logging (LocalFolder\Logs)
        // -------------------------------------------------------------------

        private static readonly object _logLock = new object();
        private static StreamWriter _logWriter;
        public static string LogFilePath { get; private set; }

        /// <summary>Writes a host-side line into the engine log file.</summary>
        public static void LogLine(string tag, string message)
        {
            WriteLogLine(tag, message);
        }

        private static void OpenLogFile()
        {
            try
            {
                string dir = Path.Combine(Windows.Storage.ApplicationData.Current.LocalFolder.Path, "Logs");
                Directory.CreateDirectory(dir);
                string path = Path.Combine(dir, "godot_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".log");
                lock (_logLock)
                {
                    if (_logWriter != null)
                    {
                        _logWriter.Dispose();
                    }
                    _logWriter = new StreamWriter(new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read), new UTF8Encoding(false));
                    _logWriter.AutoFlush = true;
                    LogFilePath = path;
                    _logWriter.WriteLine("=== Godot UWP engine log opened " + DateTime.Now.ToString("O") + " ===");
                }
                Debug.WriteLine("[GodotEngineHost] Log file: " + path);
            }
            catch (Exception ex)
            {
                Debug.WriteLine("[GodotEngineHost] Failed to open log file: " + ex.Message);
            }
        }

        private static void WriteLogLine(string tag, string message)
        {
            try
            {
                lock (_logLock)
                {
                    if (_logWriter != null)
                    {
                        _logWriter.WriteLine(DateTime.Now.ToString("HH:mm:ss.fff") + " [" + tag + "] " + message);
                    }
                }
            }
            catch { }
        }

        private static void CloseLogFile()
        {
            lock (_logLock)
            {
                if (_logWriter == null)
                {
                    return;
                }
                try
                {
                    _logWriter.WriteLine("=== Log closed " + DateTime.Now.ToString("O") + " ===");
                    _logWriter.Dispose();
                }
                catch { }
                _logWriter = null;
            }
        }

        // -------------------------------------------------------------------

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            Stop();
        }
    }
}
