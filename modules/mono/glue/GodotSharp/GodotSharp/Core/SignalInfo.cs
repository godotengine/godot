namespace Godot
{
    public struct SignalInfo
    {
        private readonly Object _owner;
        private readonly StringName _signalName;

        public Object Owner => _owner;
        public StringName Name => _signalName;

        public SignalInfo(Object owner, StringName name)
        {
            _owner = owner;
            _signalName = name;
        }
    }
}
