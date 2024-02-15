using Godot;
using GodotTools.Build;
using GodotTools.Internals;
using JetBrains.Annotations;

namespace GodotTools
{
    public partial class HotReloadAssemblyWatcher : Node
    {
        private Timer _watchTimer;

        public override void _Notification(int what)
        {
            if (what == Node.NotificationWMWindowFocusIn)
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
            if (Internal.IsAssembliesReloadingNeeded())
            {
                BuildManager.UpdateLastValidBuildDateTime();
                Internal.ReloadAssemblies(softReload: false);
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
            _watchTimer.Timeout += TimerTimeout;
            AddChild(_watchTimer);
            _watchTimer.Start();
        }
    }
}
