using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate void ScriptSignalMethod<T>(T godotObject, scoped in NativeVariantPtrArgs args)
        where T : GodotObject;
}
