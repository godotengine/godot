using Godot;
using Godot.NativeInterop;

partial class Methods
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.GodotObject.MethodName {
        /// <summary>
        /// Cached name for the 'MethodWithOverload' method.
        /// </summary>
        public new static readonly global::Godot.StringName @MethodWithOverload = "MethodWithOverload";
    }
    protected new static partial class GodotInternal
    {
        /// <summary>
        /// Get the method information for all the methods declared in this class.
        /// This method is used by Godot to register the available methods in the editor.
        /// Do not call this method.
        /// </summary>
        public static
#nullable enable
            global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>?
#nullable restore
            GetGodotMethodList()
        {
            var methods = new global::System.Collections.Generic.List<global::Godot.Bridge.MethodInfo>(3);
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: null, defaultArguments: null));
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)2, name: "a", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
        methods.Add(new(name: MethodName.@MethodWithOverload, returnVal: new(type: (global::Godot.Variant.Type)0, name: "", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), flags: (global::Godot.MethodFlags)1, arguments: new() { new(type: (global::Godot.Variant.Type)2, name: "a", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false), new(type: (global::Godot.Variant.Type)2, name: "b", hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)6, exported: false),  }, defaultArguments: null));
            return methods;
        }
        /// <summary>
        /// Get the method information for all the methods declared in this class.
        /// This method is used by Godot to register the available methods in the editor.
        /// Do not call this method.
        /// </summary>
        public static void GetGodotRpcMethods(global::Godot.Bridge.RpcMethodCollector collector)
        {
        }
        private static unsafe void GetGodotMethodTrampolines(global::Godot.Bridge.MethodTrampolineCollector collector)
        {
            static godot_variant trampoline_0_MethodWithOverload(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 0) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 0, provided: args.Count);
                    return default;
                }
                ((global::Methods)godotObject).@MethodWithOverload();
                return default;
            }
            collector.TryAdd(new(MethodName.@MethodWithOverload, 0), new(&trampoline_0_MethodWithOverload, isStatic: false));
            static godot_variant trampoline_1_MethodWithOverload(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 1) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 1, provided: args.Count);
                    return default;
                }
                ((global::Methods)godotObject).@MethodWithOverload(global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[0]));
                return default;
            }
            collector.TryAdd(new(MethodName.@MethodWithOverload, 1), new(&trampoline_1_MethodWithOverload, isStatic: false));
            static godot_variant trampoline_2_MethodWithOverload(object godotObject, NativeVariantPtrArgs args, ref godot_variant_call_error callError)
            {
                if (args.Count != 2) {
                    callError = godot_variant_call_error.CreateInvalidArgumentCountError(expected: 2, provided: args.Count);
                    return default;
                }
                ((global::Methods)godotObject).@MethodWithOverload(global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[0]), global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(args[1]));
                return default;
            }
            collector.TryAdd(new(MethodName.@MethodWithOverload, 2), new(&trampoline_2_MethodWithOverload, isStatic: false));
        }
        [global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMembers(global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes.PublicConstructors | global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes.NonPublicConstructors)]
        public static global::System.Type CachedType { get; } = typeof(global::Methods);
        private static partial class Accessors
        {
            [global::System.Runtime.CompilerServices.UnsafeAccessor(global::System.Runtime.CompilerServices.UnsafeAccessorKind.Method, Name = ".ctor")]
            public extern static void CtorAsMethod(global::Methods godotObject);
        }
        private static unsafe void GetGodotConstructorTrampolines(global::Godot.Bridge.ConstructorTrampolineCollector collector)
        {
            static global::Godot.GodotObject trampoline_0(global::System.IntPtr godotObjectPtr, NativeVariantPtrArgs args)
            {
                if (args.Count != 0) {
                    throw new global::System.MissingMemberException($"Invalid argument count for constructor of class 'Methods'. Expected 0, but got {args.Count}.");
                }
                var godotObject = (global::Methods)global::System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(global::Methods.GodotInternal.CachedType);
                global::Godot.Bridge.ScriptManagerBridge.Accessors.UnsafeSetGodotObjectNativePtr(godotObject, godotObjectPtr);
                global::Methods.GodotInternal.Accessors.CtorAsMethod(godotObject);
                return godotObject;
            }
            collector.TryAdd(0, new(&trampoline_0));
        }
    }
#pragma warning restore CS0109
}
