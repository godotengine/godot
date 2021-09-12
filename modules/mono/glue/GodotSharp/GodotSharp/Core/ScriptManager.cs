namespace Godot
{
    internal class ScriptManager
    {
        internal static void FrameCallback()
        {
            Dispatcher.DefaultGodotTaskScheduler?.Activate();
        }
    }
}
