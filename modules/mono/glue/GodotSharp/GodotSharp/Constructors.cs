namespace Godot;

using System;
using System.Collections.Generic;

internal static partial class Constructors
{
    internal readonly struct BuiltInConstructorTrampoline
    {
        public unsafe BuiltInConstructorTrampoline(BuiltInConstructorTrampolineDelegate trampolineDelegate) =>
            TrampolineDelegate = trampolineDelegate;

        public unsafe BuiltInConstructorTrampoline(IntPtr trampolineDelegate) =>
            TrampolineDelegate = (BuiltInConstructorTrampolineDelegate)trampolineDelegate;

        public unsafe BuiltInConstructorTrampolineDelegate TrampolineDelegate { get; }
    }

    internal static readonly Dictionary<string, BuiltInConstructorTrampoline> BuiltInMethodConstructors;

    public static GodotObject Invoke(string nativeTypeNameStr, IntPtr nativeObjectPtr)
    {
        if (!BuiltInMethodConstructors.TryGetValue(nativeTypeNameStr, out var constructor))
            throw new InvalidOperationException("Wrapper class not found for type: " + nativeTypeNameStr);
        unsafe
        {
            return constructor.TrampolineDelegate(nativeObjectPtr);
        }
    }
}
