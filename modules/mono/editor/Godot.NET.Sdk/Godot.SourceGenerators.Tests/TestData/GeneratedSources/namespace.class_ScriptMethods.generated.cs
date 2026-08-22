using Godot;
using Godot.NativeInterop;

namespace @namespace {

partial class @class
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the methods contained in this class, for fast lookup.
    /// </summary>
    public new class MethodName : global::Godot.GodotObject.MethodName {
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
            return null;
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
        }
        [global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMembers(global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes.PublicConstructors | global::System.Diagnostics.CodeAnalysis.DynamicallyAccessedMemberTypes.NonPublicConstructors)]
        public static global::System.Type CachedType { get; } = typeof(global::@namespace.@class);
        private static partial class Accessors
        {
            [global::System.Runtime.CompilerServices.UnsafeAccessor(global::System.Runtime.CompilerServices.UnsafeAccessorKind.Method, Name = ".ctor")]
            public extern static void CtorAsMethod(global::@namespace.@class godotObject);
        }
        private static unsafe void GetGodotConstructorTrampolines(global::Godot.Bridge.ConstructorTrampolineCollector collector)
        {
            static global::Godot.GodotObject trampoline_0(global::System.IntPtr godotObjectPtr, NativeVariantPtrArgs args)
            {
                if (args.Count != 0) {
                    throw new global::System.MissingMemberException($"Invalid argument count for constructor of class '@namespace.@class'. Expected 0, but got {args.Count}.");
                }
                var godotObject = (global::@namespace.@class)global::System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(global::@namespace.@class.GodotInternal.CachedType);
                global::Godot.Bridge.ScriptManagerBridge.Accessors.UnsafeSetGodotObjectNativePtr(godotObject, godotObjectPtr);
                global::@namespace.@class.GodotInternal.Accessors.CtorAsMethod(godotObject);
                return godotObject;
            }
            collector.TryAdd(0, new(&trampoline_0));
        }
    }
#pragma warning restore CS0109
}

}
