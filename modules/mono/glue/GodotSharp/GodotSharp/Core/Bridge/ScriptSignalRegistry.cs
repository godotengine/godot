namespace Godot.Bridge
{
    public sealed class ScriptSignalRegistry<T> : ScriptRegistry<T, ScriptSignalMethod<GodotObject>, ScriptCache<T, ScriptSignalMethod<GodotObject>>,
            ScriptSignalRegistry<T>>
        where T : GodotObject
    {
        protected override void InitializeCache((MethodKey, ScriptSignalMethod<GodotObject>)[] methods)
        {
            ScriptCache<T, ScriptSignalMethod<GodotObject>>.Initialize(methods, []);
        }
    }
}
