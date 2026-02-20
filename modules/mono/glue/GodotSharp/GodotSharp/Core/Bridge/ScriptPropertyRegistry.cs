namespace Godot.Bridge
{
    public sealed class ScriptPropertyRegistry<T> :
        ScriptRegistry<T, ScriptPropertyMethod<GodotObject>, ScriptCache<T, ScriptPropertyMethod<GodotObject>>,
            ScriptPropertyRegistry<T>>
        where T : GodotObject
    {
        protected override void InitializeCache((MethodKey, ScriptPropertyMethod<GodotObject>)[] methods)
        {
            ScriptCache<T, ScriptPropertyMethod<GodotObject>>.Initialize(methods, []);
        }
    }
}
