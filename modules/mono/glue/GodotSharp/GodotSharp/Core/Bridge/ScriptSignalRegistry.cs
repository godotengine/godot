namespace Godot.Bridge
{
    public sealed class ScriptSignalRegistry<T> : ScriptRegistry<T, ScriptSignalMethod<GodotObject>, ScriptCache<ScriptSignalMethod<GodotObject>>,
            ScriptSignalRegistry<T>>
        where T : GodotObject
    {
        protected override ScriptCache<ScriptSignalMethod<GodotObject>> InitializeCache((MethodKey, ScriptSignalMethod<GodotObject>)[] methods)
        {
            return new ScriptCache<ScriptSignalMethod<GodotObject>>(methods, []);
        }
    }
}
