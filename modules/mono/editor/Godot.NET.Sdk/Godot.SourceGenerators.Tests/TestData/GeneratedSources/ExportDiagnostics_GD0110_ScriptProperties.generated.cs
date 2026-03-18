using Godot;
using Godot.NativeInterop;

partial class ExportDiagnostics_GD0110
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the 'MyButton' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButton = "MyButton";
    }
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_MyButton(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0110)godotObject).@MyButton;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
            }
            collector.TryAdd(PropertyName.@MyButton, (new(&trampoline_get_MyButton), new(null)));
        }
    }
    /// <summary>
    /// Get the property information for all the properties declared in this class.
    /// This method is used by Godot to register the available properties in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo> GetGodotPropertyList()
    {
        var properties = new global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>();
        return properties;
    }
#pragma warning restore CS0109
}
