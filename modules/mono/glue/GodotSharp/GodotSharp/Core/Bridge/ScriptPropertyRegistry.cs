namespace Godot.Bridge
{
    public sealed class ScriptPropertyRegistry<T> :
        ScriptRegistry<T, ScriptPropertyMethod<GodotObject>, ScriptCache<ScriptPropertyMethod<GodotObject>>,
            ScriptPropertyRegistry<T>>
        where T : GodotObject
    {
        protected override ScriptCache<ScriptPropertyMethod<GodotObject>> InitializeCache((MethodKey, ScriptPropertyMethod<GodotObject>)[] methods)
        {
            return new ScriptCache<ScriptPropertyMethod<GodotObject>>(methods, []);
        }
    }
}
