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
    private static partial class GodotInternal
    {
        public new static unsafe void GetGodotMethodTrampolines(global::Godot.Bridge.ScriptManagerBridge.MethodTrampolineCollector collector)
        {
        }
        private static readonly global::System.Type CachedType = typeof(global::@namespace.@class);
        private static partial class Accessors
        {
            [global::System.Runtime.CompilerServices.UnsafeAccessor(global::System.Runtime.CompilerServices.UnsafeAccessorKind.Method, Name = ".ctor")]
            public extern static void CtorAsMethod(global::@namespace.@class godotObject);
        }
        public new static unsafe void GetGodotConstructorTrampolines(global::Godot.Bridge.ScriptManagerBridge.ConstructorTrampolineCollector collector)
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
