// MainPage.xaml.cs
// Hosts an in-process Godot engine inside a XAML SwapChainPanel. The engine
// runs on a dedicated thread owned by GodotEngineHost; this page only forwards
// input and panel sizing onto that thread. No HWND is involved: the engine
// binds a composition swap chain to GodotPanel and all input is injected from
// the XAML pointer/key events below.

using System;
using System.Runtime.InteropServices;
using Windows.Storage;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Input;

namespace Godot.Uwp.Embedding
{
    public sealed partial class MainPage : Page
    {
        private readonly GodotEngineHost _host = new GodotEngineHost();

        // Generic host<->engine JSON message bus. The sample only logs incoming
        // messages; real apps answer them and/or push messages with _sender.
        private readonly EngineMessageReceiver _receiver;
        private readonly EngineMessageSender _sender;

        private double _lastX, _lastY;
        private bool _engineStarted;

        public MainPage()
        {
            this.InitializeComponent();

            _receiver = new EngineMessageReceiver();
            _sender = new EngineMessageSender(_host);

            // Key events arrive on the CoreWindow in UWP.
            Window.Current.CoreWindow.KeyDown += OnCoreKeyDown;
            Window.Current.CoreWindow.KeyUp += OnCoreKeyUp;

            // PLM terminates suspended apps that keep presenting — pause the
            // engine loop across suspend/resume.
            Application.Current.Suspending += (s, e2) => _host.Pause();
            Application.Current.Resuming += (s, e2) => _host.Resume();

            _host.Log += OnGodotLog;
            _host.Stopped += OnEngineStopped;
        }

        // -------------------------------------------------------------------
        // Engine startup
        // -------------------------------------------------------------------

        private void OnPanelLoaded(object sender, RoutedEventArgs e)
        {
            if (_engineStarted)
            {
                return;
            }
            _engineStarted = true;

            // Load a bundled "project.pck" if present, else the loose
            // GodotProject folder shipped in the package.
            string installed = Windows.ApplicationModel.Package.Current.InstalledLocation.Path;
            string pckPath = System.IO.Path.Combine(installed, "Assets", "project.pck");
            string projectPath = System.IO.File.Exists(pckPath)
                ? pckPath
                : System.IO.Path.Combine(installed, "GodotProject");

            _host.ProjectPath = projectPath;

            // The embedded display server is D3D12 / RenderingDevice-only, so
            // force a RenderingDevice method in case the project defaults to
            // the gl_compatibility renderer.
            _host.ExtraArgs = new string[] { "--rendering-method", "forward_plus" };

            // Wire the message bus before Start so messages emitted during
            // GDScript _ready are not dropped. The sample just logs them.
            _receiver.OnMessage += OnBusMessage;
            _receiver.Initialize();

            float scaleX = GodotPanel.CompositionScaleX;
            float scaleY = GodotPanel.CompositionScaleY;
            int widthPx = Math.Max(64, (int)(GodotPanel.ActualWidth * scaleX));
            int heightPx = Math.Max(64, (int)(GodotPanel.ActualHeight * scaleY));

            // IUnknown* of the panel; the engine QIs ISwapChainPanelNative
            // internally (UWP and WinUI3 panel IIDs are both supported).
            IntPtr panelUnknown = Marshal.GetIUnknownForObject(GodotPanel);

            StatusText.Text = "Starting Godot engine...\nLogs: " +
                ApplicationData.Current.LocalFolder.Path + @"\Logs";

            _host.Start(panelUnknown, widthPx, heightPx, scaleX, scaleY, Dispatcher);
        }

        // -------------------------------------------------------------------
        // Engine -> host messages (raised on the UI thread by the receiver)
        // -------------------------------------------------------------------

        private void OnBusMessage(object sender, EngineMessageEventArgs e)
        {
            string args = e.ArgsJson != null && e.ArgsJson.Length > 200
                ? e.ArgsJson.Substring(0, 200) + "..."
                : e.ArgsJson;
            GodotEngineHost.LogLine("Bus", "<- " + e.Method + " " + args);
        }

        private void OnEngineStopped(bool engineRequestedQuit)
        {
            var ignore = Dispatcher.RunAsync(CoreDispatcherPriority.Normal, () =>
            {
                StatusText.Visibility = Visibility.Visible;
                StatusText.Text = engineRequestedQuit
                    ? "Godot engine exited."
                    : "Godot engine stopped (see logs in LocalState\\Logs).";
            });
        }

        private int _overlayHidden; // 0 = visible, 1 = hide already queued

        private void OnGodotLog(string message, GodotLogLevel level)
        {
            // Called on the ENGINE thread — never touch XAML here directly.
            // Hide the status overlay once, after the engine is running.
            if (_host.IsRunning && System.Threading.Interlocked.Exchange(ref _overlayHidden, 1) == 0)
            {
                var ignore = Dispatcher.RunAsync(CoreDispatcherPriority.Low, () =>
                {
                    StatusText.Visibility = Visibility.Collapsed;
                });
            }
        }

        // -------------------------------------------------------------------
        // Panel sizing / DPI
        // -------------------------------------------------------------------

        private void ConfigurePanel()
        {
            float scaleX = GodotPanel.CompositionScaleX;
            float scaleY = GodotPanel.CompositionScaleY;
            double width = GodotPanel.ActualWidth * scaleX;
            double height = GodotPanel.ActualHeight * scaleY;
            _host.ConfigurePanel(width, height, scaleX, scaleY);
        }

        private void OnPanelSizeChanged(object sender, SizeChangedEventArgs e)
        {
            ConfigurePanel();
        }

        private void OnPanelCompositionScaleChanged(SwapChainPanel sender, object args)
        {
            ConfigurePanel();
        }

        // -------------------------------------------------------------------
        // Input forwarding (physical pixels = DIP * CompositionScale)
        // -------------------------------------------------------------------

        private float DpiScale
        {
            get { return GodotPanel.CompositionScaleX; }
        }

        private void OnPointerPressed(object sender, PointerRoutedEventArgs e)
        {
            var pt = e.GetCurrentPoint(GodotPanel);
            float scale = DpiScale;
            float x = (float)(pt.Position.X * scale);
            float y = (float)(pt.Position.Y * scale);
            if (pt.Properties.IsLeftButtonPressed) _host.InjectMouseButton(GodotMouseButton.Left, true, x, y);
            if (pt.Properties.IsRightButtonPressed) _host.InjectMouseButton(GodotMouseButton.Right, true, x, y);
            if (pt.Properties.IsMiddleButtonPressed) _host.InjectMouseButton(GodotMouseButton.Middle, true, x, y);
            e.Handled = true;
        }

        private void OnPointerReleased(object sender, PointerRoutedEventArgs e)
        {
            var pt = e.GetCurrentPoint(GodotPanel);
            float scale = DpiScale;
            float x = (float)(pt.Position.X * scale);
            float y = (float)(pt.Position.Y * scale);
            if (!pt.Properties.IsLeftButtonPressed) _host.InjectMouseButton(GodotMouseButton.Left, false, x, y);
            if (!pt.Properties.IsRightButtonPressed) _host.InjectMouseButton(GodotMouseButton.Right, false, x, y);
            if (!pt.Properties.IsMiddleButtonPressed) _host.InjectMouseButton(GodotMouseButton.Middle, false, x, y);
            e.Handled = true;
        }

        private void OnPointerMoved(object sender, PointerRoutedEventArgs e)
        {
            var pt = e.GetCurrentPoint(GodotPanel);
            float scale = DpiScale;
            float px = (float)(pt.Position.X * scale);
            float py = (float)(pt.Position.Y * scale);
            _host.InjectMouseMotion(px, py,
                (float)((pt.Position.X - _lastX) * scale),
                (float)((pt.Position.Y - _lastY) * scale));
            _lastX = pt.Position.X;
            _lastY = pt.Position.Y;
            e.Handled = true;
        }

        private void OnPointerWheelChanged(object sender, PointerRoutedEventArgs e)
        {
            var pt = e.GetCurrentPoint(GodotPanel);
            float scale = DpiScale;
            float x = (float)(pt.Position.X * scale);
            float y = (float)(pt.Position.Y * scale);
            float notches = pt.Properties.MouseWheelDelta / 120.0f;
            if (pt.Properties.IsHorizontalMouseWheel)
            {
                _host.InjectMouseWheel(x, y, notches, 0f);
            }
            else
            {
                _host.InjectMouseWheel(x, y, 0f, notches);
            }
            e.Handled = true;
        }

        private void OnCoreKeyDown(CoreWindow sender, KeyEventArgs e)
        {
            _host.InjectKey((uint)e.VirtualKey, true, e.KeyStatus.WasKeyDown, 0);
        }

        private void OnCoreKeyUp(CoreWindow sender, KeyEventArgs e)
        {
            _host.InjectKey((uint)e.VirtualKey, false, false, 0);
        }
    }
}
