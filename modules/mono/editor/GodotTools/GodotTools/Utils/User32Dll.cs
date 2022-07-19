using System.Runtime.InteropServices;

namespace GodotTools.Utils
{
    public static class User32Dll
    {
        [DllImport("user32.dll")]
        public static extern bool AllowSetForegroundWindow(int dwProcessId);
    }
}
