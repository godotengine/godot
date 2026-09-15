using Godot;
using Godot.NativeInterop;

partial struct OuterClass
{
partial class NestedClass
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.RefCounted.PropertyName {
    }
    protected new static partial class GodotInternal
    {
        private static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.PropertyTrampolineCollector collector)
        {
        }
        /// <summary>
        /// Get the property information for all the properties declared in this class.
        /// This method is used by Godot to register the available properties in the editor.
        /// Do not call this method.
        /// </summary>
        public static
#nullable enable
            global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>?
#nullable restore
            GetGodotPropertyList()
        {
            return null;
        }
    }
#pragma warning restore CS0109
}
}
