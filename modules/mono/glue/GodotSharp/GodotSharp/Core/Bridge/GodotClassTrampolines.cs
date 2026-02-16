using System;

namespace Godot.Bridge;

#nullable enable

/// <summary>
/// Struct representing a method trampoline, containing a delegate to the trampoline function.
/// The trampoline is a <c>delegate* managed</c> with a signature equivalent to the following C# delegate:
/// <code>
/// delegate godot_variant MethodTrampoline(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError);
/// </code>
/// </summary>
public readonly struct MethodTrampoline
{
    public unsafe MethodTrampoline(MethodTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe MethodTrampolineDelegate TrampolineDelegate { get; }
}

/// <summary>
/// Struct representing a property getter trampoline, containing a delegate to the trampoline function.
/// The trampoline is a <c>delegate* managed</c> with a signature equivalent to the following C# delegate:
/// <code>
/// delegate godot_variant PropertyGetterTrampoline(object godotObject);
/// </code>
/// </summary>
public readonly struct PropertyGetterTrampoline
{
    public unsafe PropertyGetterTrampoline(PropertyGetterTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe PropertyGetterTrampolineDelegate TrampolineDelegate { get; }
}

/// <summary>
/// Struct representing a property setter trampoline, containing a delegate to the trampoline function.
/// The trampoline is a <c>delegate* managed</c> with a signature equivalent to the following C# delegate:
/// <code>
/// delegate void PropertySetterTrampoline(object godotObject, in godot_variant value);
/// </code>
/// </summary>
public readonly struct PropertySetterTrampoline
{
    public unsafe PropertySetterTrampoline(PropertySetterTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe PropertySetterTrampolineDelegate TrampolineDelegate { get; }
}

/// <summary>
/// Struct representing a raise signal trampoline, containing a delegate to the trampoline function.
/// The trampoline is a <c>delegate* managed</c> with a signature equivalent to the following C# delegate:
/// <code>
/// delegate void RaiseSignalTrampoline(object godotObject, NativeVariantPtrArgs args);
/// </code>
/// </summary>
public readonly struct RaiseSignalTrampoline
{
    public unsafe RaiseSignalTrampoline(RaiseSignalTrampolineDelegate trampolineDelegate)
        => TrampolineDelegate = trampolineDelegate;

    public unsafe RaiseSignalTrampolineDelegate TrampolineDelegate { get; }
}
