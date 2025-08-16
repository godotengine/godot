namespace Godot
{
    /// <summary>
    /// Represents a signal defined in an object.
    /// </summary>
    public readonly struct Signal : IAwaitable<Variant[]>
    {
        private readonly GodotObject _owner;
        private readonly StringName _signalName;

        /// <summary>
        /// Object that contains the signal.
        /// </summary>
        public GodotObject Owner => _owner;

        /// <summary>
        /// Name of the signal.
        /// </summary>
        public StringName Name => _signalName;

        /// <summary>
        /// Creates a new <see cref="Signal"/> with the name <paramref name="name"/>
        /// in the specified <paramref name="owner"/>.
        /// </summary>
        /// <param name="owner">Object that contains the signal.</param>
        /// <param name="name">Name of the signal.</param>
        public Signal(GodotObject owner, StringName name)
        {
            _owner = owner;
            _signalName = name;
        }

        public IAwaiter<Variant[]> GetAwaiter()
        {
            return new SignalAwaiter(_owner, _signalName, _owner);
        }
    }
}
