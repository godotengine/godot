namespace Godot.Bridge
{
    public sealed class ScriptSignalRegistry<T> : ScriptRegistry<T, ScriptSignalMethod, ScriptCache<ScriptSignalMethod>,
            ScriptSignalRegistry<T>>
        where T : GodotObject
    {
        protected override ScriptCache<ScriptSignalMethod> InitializeCache((MethodKey, ScriptSignalMethod)[] methods)
        {
            return new ScriptCache<ScriptSignalMethod>(methods, []);
        }
    }
}
