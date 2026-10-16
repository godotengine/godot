using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Loader;
using Godot;
using GodotTools.Build;
using GodotTools.Internals;
using JetBrains.Annotations;

namespace GodotTools
{
    public partial class HotReloadAssemblyWatcher : Node
    {
        private enum UnloadNotifyStatus
        {
            None,
            TakingTooLong,
            Failed,
        }
        private class AlcUnloadInfo(int unloadStartTime)
        {
            public int UnloadStartTime { get; } = unloadStartTime;
            public UnloadNotifyStatus NotifyStatus { get; set; }
        }

#nullable disable
        private Timer _watchTimer;
        private Timer _unloadAlcTimer;
#nullable enable
        private static ConditionalWeakTable<AssemblyLoadContext, AlcUnloadInfo> _unloadingAlc = new();

        public static bool IsAssemblyBeingUnloaded(System.Reflection.Assembly assembly)
        {
            var alc = AssemblyLoadContext.GetLoadContext(assembly);
            return alc is not null && _unloadingAlc.TryGetValue(alc, out _);
        }

        public override void _Notification(int what)
        {
            if (what == Node.NotificationWMWindowFocusIn)
            {
                RestartTimer();

                if (Internal.IsAssembliesReloadingNeeded())
                {
                    BuildManager.UpdateLastValidBuildDateTime();
                    Internal.ReloadAssemblies();
                }
            }
        }

        private void WatchTimerTimeout()
        {
            if (Internal.IsAssembliesReloadingNeeded())
            {
                BuildManager.UpdateLastValidBuildDateTime();
                Internal.ReloadAssemblies();
            }
        }

        private void UnloadNoticeTimerTimeout()
        {
            AlcUnloadInfo[] unloadInfos;
            if (!CheckAlcNeedsUnload())
                return;

            // Try to unload ALC in the background.
            System.GC.Collect(System.GC.MaxGeneration, System.GCCollectionMode.Forced);
            System.GC.WaitForPendingFinalizers();

            if (!CheckAlcNeedsUnload())
                return;

            bool CheckAlcNeedsUnload()
            {
                unloadInfos = _unloadingAlc.Select(kv => kv.Value).ToArray();
                if (unloadInfos.Length == 0 || unloadInfos.All(info => info.NotifyStatus == UnloadNotifyStatus.Failed))
                {
                    _unloadAlcTimer.Stop();
                    return false;
                }
                return true;
            }

            bool notifyTooLong = false;
            bool notifyFail = false;
            foreach (var info in unloadInfos)
            {
                int elapsedTimeMs = System.Environment.TickCount - info.UnloadStartTime;

                if (elapsedTimeMs >= 200 && info.NotifyStatus == UnloadNotifyStatus.None)
                {
                    info.NotifyStatus = UnloadNotifyStatus.TakingTooLong;
                    notifyTooLong = true;
                }
                if (elapsedTimeMs >= 3000 && info.NotifyStatus != UnloadNotifyStatus.Failed)
                {
                    info.NotifyStatus = UnloadNotifyStatus.Failed;
                    notifyFail = true;
                }
            }

            var toaster = EditorInterface.Singleton.GetEditorToaster();
            if (notifyTooLong)
            {
                toaster.PushToast(".NET: Assembly unloading is taking longer than expected...", EditorToaster.Severity.Info);
            }
            if (notifyFail)
            {
                string message = ".NET: Failed to unload assemblies. The editor will continue working using new assemblies.\n" +
                    $"The number of leaked assemblies: {unloadInfos.Length}. The leaked memory won't be reclaimed until editor restart.\n" +
                    "Possible causes: Strong GC handles, running threads, etc.\n" +
                    "Please check https://github.com/godotengine/godot/issues/78513 for more information.";

                toaster.PushToast(message, EditorToaster.Severity.Warning);
                GD.PushWarning(message);
            }
        }

        [UsedImplicitly]
        public void RestartTimer()
        {
            _watchTimer.Stop();
            _watchTimer.Start();
        }

        public override void _Ready()
        {
            base._Ready();

            _watchTimer = new Timer
            {
                OneShot = false,
                WaitTime = 0.5f
            };
            _watchTimer.Timeout += WatchTimerTimeout;
            AddChild(_watchTimer);
            _watchTimer.Start();

            _unloadAlcTimer = new Timer
            {
                OneShot = false,
                WaitTime = 0.1f
            };
            _unloadAlcTimer.Timeout += UnloadNoticeTimerTimeout;
            AddChild(_unloadAlcTimer);

        }

        public override void _EnterTree()
        {
            AlcTracker.AlcBeginUnloading += OnAlcBeginUnloading;
        }

        public override void _ExitTree()
        {
            AlcTracker.AlcBeginUnloading -= OnAlcBeginUnloading;
        }

        private void OnAlcBeginUnloading(AssemblyLoadContext alc)
        {
            _unloadingAlc.Add(alc, new AlcUnloadInfo(System.Environment.TickCount));
            _unloadAlcTimer.Start();
        }
    }
}
