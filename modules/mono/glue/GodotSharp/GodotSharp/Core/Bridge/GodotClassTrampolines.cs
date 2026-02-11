using System;

namespace Godot.Bridge;

#nullable enable

public readonly struct MethodTrampoline
{
    public unsafe MethodTrampoline(MethodTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe MethodTrampolineDelegate TrampolineDelegate { get; }
}

public readonly struct PropertyGetterTrampoline
{
    public unsafe PropertyGetterTrampoline(PropertyGetterTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe PropertyGetterTrampolineDelegate TrampolineDelegate { get; }
}

public readonly struct PropertySetterTrampoline
{
    public unsafe PropertySetterTrampoline(PropertySetterTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe PropertySetterTrampolineDelegate TrampolineDelegate { get; }
}

public readonly struct RaiseSignalTrampoline
{
    public unsafe RaiseSignalTrampoline(RaiseSignalTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe RaiseSignalTrampolineDelegate TrampolineDelegate { get; }
}
