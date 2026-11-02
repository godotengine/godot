using Godot;
using GodotTools.Build;
using GodotTools.Internals;
using JetBrains.Annotations;

namespace GodotTools
{
    public partial class HotReloadAssemblyWatcher : Node
    {
#nullable disable
        private Timer _watchTimer;
#nullable enable

        public override void _Notification(int what)
        {
            if (what == Node.NotificationWMWindowFocusIn && CanWatchAssemblies())
            {
                RestartTimer();

                if (Internal.IsAssembliesReloadingNeeded())
                {
                    BuildManager.UpdateLastValidBuildDateTime();
                    Internal.ReloadAssemblies(softReload: false);
                }
            }
        }

        private void TimerTimeout()
        {
            if (CanWatchAssemblies() && Internal.IsAssembliesReloadingNeeded())
            {
                BuildManager.UpdateLastValidBuildDateTime();
                Internal.ReloadAssemblies(softReload: false);
            }
        }

        [UsedImplicitly]
        public void RestartTimer()
        {
            if (!CanWatchAssemblies())
                return;

            _watchTimer.Stop();
            _watchTimer.Start();
        }

        // Children leave the tree before their parent. Focus notifications can
        // still arrive during editor teardown, after the timer has already left.
        private bool CanWatchAssemblies() =>
            IsInsideTree() && _watchTimer != null && _watchTimer.IsInsideTree();

        public override void _Ready()
        {
            base._Ready();

            _watchTimer = new Timer
            {
                OneShot = false,
                WaitTime = 0.5f
            };
            _watchTimer.Timeout += TimerTimeout;
            AddChild(_watchTimer);
            _watchTimer.Start();
        }
    }
}
