using Godot;
using GodotTools.Internals;
using static GodotTools.Internals.Globals;

namespace GodotTools
{
    public class HotReloadAssemblyWatcher : Node
    {
        private Timer _watchTimer;

        public override void _Notification(int what)
        {
            if (what == MainLoop.NotificationWmFocusIn)
            {
                RestartTimer();

                if (Internal.IsAssembliesReloadingNeeded())
                    Internal.ReloadAssemblies(softReload: false);
            }
        }

        private void TimerTimeout()
        {
            if (Internal.IsAssembliesReloadingNeeded())
                Internal.ReloadAssemblies(softReload: false);
        }

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
                WaitTime = (float)EditorDef("mono/assembly_watch_interval_sec", 0.5)
            };
            _watchTimer.Connect("timeout", this, nameof(TimerTimeout));
            AddChild(_watchTimer);
            _watchTimer.Start();
        }
    }
}
