using Godot;
using Godot.NativeInterop;

partial class ScriptBoilerplate
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.Node.MethodName {
        /// <summary>
        /// Cached name for the '_Process' method.
        /// </summary>
        public new static readonly global::Godot.StringName @_Process = "_Process";
        /// <summary>
        /// Cached name for the 'Bazz' method.
        /// </summary>
        public new static readonly global::Godot.StringName @Bazz = "Bazz";
    }
    /// <summary>
    /// Get the method information for all the methods declared in this class.
    /// This method is used by Godot to register the available methods in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotMethodList()
    {
        var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(2);
        methods.Add(new(name: MethodName.@_Process, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)3, name: "delta", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        methods.Add(new(name: MethodName.@Bazz, returnVal: new(type: (global::Godot.Variant.Type)2, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)21, name: "name", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return methods;
    }
    private static partial class GodotInternal
    {
        public new static unsafe void GetGodotMethodTrampolines(global::Godot.Bridge.ScriptManagerBridge.MethodTrampolineCollector collector)
        {
            static godot_variant trampoline_1__Process(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 1) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 1, provided: args.Count);
                    return default;
                }
                ((global::Godot.Node)godotObject).@_Process(global::Godot.NativeInterop.VariantUtils.ConvertTo<double>(args[0]));
                return default;
            }
            collector.TryAdd(new(MethodName.@_Process, 1), new(&trampoline_1__Process, isStatic: false));
            collector.TryAdd(new(global::Godot.Node.MethodName.@_Process, 1), new(&trampoline_1__Process, isStatic: false));
            static godot_variant trampoline_1_Bazz(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 1) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 1, provided: args.Count);
                    return default;
                }
                var callRet = ((global::ScriptBoilerplate)godotObject).@Bazz(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(callRet);
            }
            collector.TryAdd(new(MethodName.@Bazz, 1), new(&trampoline_1_Bazz, isStatic: false));
        }
        private static readonly global::System.Type CachedType = typeof(global::ScriptBoilerplate);
        private static partial class Accessors
        {
            [global::System.Runtime.CompilerServices.UnsafeAccessor(global::System.Runtime.CompilerServices.UnsafeAccessorKind.Method, Name = ".ctor")]
            public extern static void CtorAsMethod(global::ScriptBoilerplate godotObject);
        }
        public new static unsafe void GetGodotConstructorTrampolines(global::Godot.Bridge.ScriptManagerBridge.ConstructorTrampolineCollector collector)
        {
            static global::Godot.GodotObject trampoline_0(global::System.IntPtr godotObjectPtr, NativeVariantPtrArgs args)
            {
                if (args.Count != 0) {
                    throw new global::System.MissingMemberException($"Invalid argument count for constructor of class 'ScriptBoilerplate'. Expected 0, but got {args.Count}.");
                }
                var godotObject = (global::ScriptBoilerplate)global::System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(global::ScriptBoilerplate.GodotInternal.CachedType);
                global::Godot.Bridge.ScriptManagerBridge.Accessors.UnsafeSetGodotObjectNativePtr(godotObject, godotObjectPtr);
                global::ScriptBoilerplate.GodotInternal.Accessors.CtorAsMethod(godotObject);
                return godotObject;
            }
            collector.TryAdd(0, new(&trampoline_0));
        }
    }
#pragma warning restore CS0109
}
