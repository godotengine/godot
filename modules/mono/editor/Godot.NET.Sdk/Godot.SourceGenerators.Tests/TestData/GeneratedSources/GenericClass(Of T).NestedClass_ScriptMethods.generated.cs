using Godot;
using Godot.NativeInterop;

partial class GenericClass<T>
{
partial class NestedClass
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
    }
#pragma warning restore CS0109
}
}
