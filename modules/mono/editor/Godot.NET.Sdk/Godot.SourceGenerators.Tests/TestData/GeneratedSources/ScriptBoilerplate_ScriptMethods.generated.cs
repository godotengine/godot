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
    protected internal new static partial class GodotInternal
    {
        public static void GetGodotMethodTrampolines(global::Godot.Bridge.MethodTrampolineCollector collector)
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
            var aux_delegate_1__Process = trampoline_1__Process;
            collector.TryAdd(new(MethodName.@_Process, 1), new(aux_delegate_1__Process.Method.MethodHandle.GetFunctionPointer(), isStatic: false));
            collector.TryAdd(new(global::Godot.Node.MethodName.@_Process, 1), new(aux_delegate_1__Process.Method.MethodHandle.GetFunctionPointer(), isStatic: false));
            static godot_variant trampoline_1_Bazz(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 1) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 1, provided: args.Count);
                    return default;
                }
                var callRet = ((global::ScriptBoilerplate)godotObject).@Bazz(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(callRet);
            }
            var aux_delegate_1_Bazz = trampoline_1_Bazz;
            collector.TryAdd(new(MethodName.@Bazz, 1), new(aux_delegate_1_Bazz.Method.MethodHandle.GetFunctionPointer(), isStatic: false));
        }
    }
#pragma warning restore CS0109
}
