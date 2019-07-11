using Godot;
using GodotTools.Internals;

namespace GodotTools
{
    public class HotReloadAssemblyWatcher : Node
    {
        private Timer watchTimer;

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
            watchTimer.Stop();
            watchTimer.Start();
        }

        public override void _Ready()
        {
            base._Ready();

            watchTimer = new Timer
            {
                OneShot = false,
                WaitTime = (float) Internal.EditorDef("mono/assembly_watch_interval_sec", 0.5)
            };
            watchTimer.Connect("timeout", this, nameof(TimerTimeout));
            AddChild(watchTimer);
            watchTimer.Start();
        }
    }
}
