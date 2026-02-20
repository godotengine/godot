using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate godot_variant ScriptMethod(GodotObject godotObject, scoped in NativeVariantPtrArgs args);
}
