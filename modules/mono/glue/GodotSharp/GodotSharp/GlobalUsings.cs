#if REAL_T_IS_DOUBLE
global using real_t = System.Double;
#else
global using real_t = System.Single;
#endif

global using unsafe ConstructorTrampolineDelegate = delegate* managed<
    /* godotObjectPtr: */ System.IntPtr,
    /* args: */ Godot.NativeInterop.NativeVariantPtrArgs,
    /* return */ Godot.GodotObject>;
global using unsafe MethodTrampolineDelegate = delegate* managed<
    /* godotObject: */ object,
    /* args: */ Godot.NativeInterop.NativeVariantPtrArgs,
    /* callError: */ ref Godot.NativeInterop.godot_variant_call_error,
    /* return (value): */ Godot.NativeInterop.godot_variant>;
global using unsafe PropertyGetterTrampolineDelegate = delegate* managed<
    /* godotObject: */ object,
    /* return (value): */ Godot.NativeInterop.godot_variant>;
global using unsafe PropertySetterTrampolineDelegate = delegate* managed<
    /* godotObject: */ object,
    /* value: */ in Godot.NativeInterop.godot_variant,
    /* return: */ void>;
global using unsafe RaiseSignalTrampolineDelegate = delegate* managed<
    /* godotObject: */ object,
    /* args: */ Godot.NativeInterop.NativeVariantPtrArgs,
    /* return */ void>;
global using unsafe BuiltInConstructorTrampolineDelegate = delegate* managed<
    /* godotObjectPtr: */ System.IntPtr,
    /* return */ Godot.GodotObject>;

global using unsafe TryAddConstructorTrampolineDelegate = delegate* unmanaged<
    /* scriptPtr: */ System.IntPtr,
    /* argCount: */ int,
    /* trampolineDelegate: */ void*,
    /* return */ void>;
global using unsafe TryAddMethodTrampolineDelegate = delegate* unmanaged<
    /* scriptPtr: */ System.IntPtr,
    /* name: */ Godot.NativeInterop.godot_string_name*,
    /* argCount: */ int,
    /* trampolineDelegate: */ void*,
    /* isStatic: */ Godot.NativeInterop.godot_bool,
    /* return */ void>;
global using unsafe TryAddPropertyTrampolineDelegate = delegate* unmanaged<
    /* scriptPtr: */ System.IntPtr,
    /* name: */ Godot.NativeInterop.godot_string_name*,
    /* getterTrampolineDelegate: */ void*,
    /* setterTrampolineDelegate: */ void*,
    /* return */ void>;
global using unsafe TryAddRaiseSignalTrampolineDelegate = delegate* unmanaged<
    /* scriptPtr: */ System.IntPtr,
    /* name: */ Godot.NativeInterop.godot_string_name*,
    /* argCount: */ int,
    /* trampolineDelegate: */ void*,
    /* return */ void>;
