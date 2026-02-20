using Godot.NativeInterop;

namespace Godot.Bridge
{
    public delegate void ScriptSignalMethod(GodotObject godotObject, scoped in NativeVariantPtrArgs args);
}
