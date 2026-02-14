using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate godot_variant ScriptMethod<T>(T godotObject, scoped in NativeVariantPtrArgs args)
        where T : GodotObject;
}
