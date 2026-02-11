#if REAL_T_IS_DOUBLE
global using real_t = System.Double;
#else
global using real_t = System.Single;
#endif

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
