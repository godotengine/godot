namespace GodotTools
{
    public readonly struct PlaySettings
    {
        public bool HasDebugger { get; }
        public string DebuggerHost { get; }
        public int DebuggerPort { get; }

        public bool BuildBeforePlaying { get; }

        public PlaySettings(string debuggerHost, int debuggerPort, bool buildBeforePlaying)
        {
            HasDebugger = true;
            DebuggerHost = debuggerHost;
            DebuggerPort = debuggerPort;
            BuildBeforePlaying = buildBeforePlaying;
        }
    }
}
