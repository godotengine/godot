using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate godot_variant ScriptPropertyMethod(GodotObject godotObject, scoped in godot_variant value);
}
