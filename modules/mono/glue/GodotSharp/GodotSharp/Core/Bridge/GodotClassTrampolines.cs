using System;
using Godot.NativeInterop;

namespace Godot.Bridge;

#nullable enable
using unsafe MethodTrampolineDelegate = delegate* unmanaged<
    /* godotObjectGCHandle: */
    IntPtr,
    /* args: */
    godot_variant**,
    /* argCount: */
    int,
    /* outRefCallError: */
    godot_variant_call_error*,
    /* outRet: */
    godot_variant*,
    void>;
using unsafe PropertyGetterTrampolineDelegate = delegate* unmanaged<
    /* godotObjectGCHandle: */
    IntPtr,
    /* outValue: */
    godot_variant*,
    /* return (success): */
    godot_bool>;
using unsafe PropertySetterTrampolineDelegate = delegate* unmanaged<
    /* godotObjectGCHandle: */
    IntPtr,
    /* value: */
    godot_variant*,
    /* return (success): */
    godot_bool>;
using unsafe RaiseSignalTrampolineDelegate = delegate* unmanaged<
    /* godotObjectGCHandle: */
    IntPtr,
    /* args: */
    godot_variant**,
    /* argCount: */
    int,
    /* outOwnerIsNull: */
    godot_bool*,
    void>;

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
