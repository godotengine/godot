using Godot;
using Godot.NativeInterop;

partial struct OuterClass
{
partial class NestedClass
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.RefCounted.MethodName {
        /// <summary>
        /// Cached name for the '_Get' method.
        /// </summary>
        public new static readonly global::Godot.StringName @_Get = "_Get";
    }
    /// <summary>
    /// Get the method information for all the methods declared in this class.
    /// This method is used by Godot to register the available methods in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo> GetGodotMethodList()
    {
        var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(1);
        methods.Add(new(name: MethodName.@_Get, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)131078, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)21, name: "property", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        return methods;
    }
    private static partial class GodotInternal
    {
        public new static unsafe void GetGodotMethodTrampolines(global::Godot.Bridge.ScriptManagerBridge.MethodTrampolineCollector collector)
        {
            static godot_variant trampoline_1__Get(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 1) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 1, provided: args.Count);
                    return default;
                }
                var callRet = ((global::Godot.GodotObject)godotObject).@_Get(global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.StringName>(args[0]));
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Variant>(callRet);
            }
            collector.TryAdd(new(MethodName.@_Get, 1), new(&trampoline_1__Get, isStatic: false));
            collector.TryAdd(new(global::Godot.GodotObject.MethodName.@_Get, 1), new(&trampoline_1__Get, isStatic: false));
        }
        private static readonly global::System.Type CachedType = typeof(global::OuterClass.NestedClass);
        private static partial class Accessors
        {
            [global::System.Runtime.CompilerServices.UnsafeAccessor(global::System.Runtime.CompilerServices.UnsafeAccessorKind.Method, Name = ".ctor")]
            public extern static void CtorAsMethod(global::OuterClass.NestedClass godotObject);
        }
        public new static unsafe void GetGodotConstructorTrampolines(global::Godot.Bridge.ScriptManagerBridge.ConstructorTrampolineCollector collector)
        {
            static global::Godot.GodotObject trampoline_0(global::System.IntPtr godotObjectPtr, NativeVariantPtrArgs args)
            {
                if (args.Count != 0) {
                    throw new global::System.MissingMemberException($"Invalid argument count for constructor of class 'OuterClass.NestedClass'. Expected 0, but got {args.Count}.");
                }
                var godotObject = (global::OuterClass.NestedClass)global::System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(global::OuterClass.NestedClass.GodotInternal.CachedType);
                global::Godot.Bridge.ScriptManagerBridge.Accessors.UnsafeSetGodotObjectNativePtr(godotObject, godotObjectPtr);
                global::OuterClass.NestedClass.GodotInternal.Accessors.CtorAsMethod(godotObject);
                return godotObject;
            }
            collector.TryAdd(0, new(&trampoline_0));
        }
    }
#pragma warning restore CS0109
}
}
