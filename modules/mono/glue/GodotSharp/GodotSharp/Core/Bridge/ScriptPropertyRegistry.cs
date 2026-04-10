namespace Godot.Bridge
{
    public sealed class ScriptPropertyRegistry<T> :
        ScriptRegistry<T, ScriptPropertyMethod, ScriptCache<ScriptPropertyMethod>,
            ScriptPropertyRegistry<T>>
        where T : GodotObject
    {
        protected override ScriptCache<ScriptPropertyMethod> InitializeCache((MethodKey, ScriptPropertyMethod)[] methods)
        {
            return new ScriptCache<ScriptPropertyMethod>(methods, []);
        }
    }
}
