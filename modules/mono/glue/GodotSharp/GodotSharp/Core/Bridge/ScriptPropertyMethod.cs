using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate godot_variant ScriptPropertyMethod<T>(T godotObject, scoped in godot_variant value)
        where T : GodotObject;
}
