using Godot;
using Godot.NativeInterop;

partial class AllReadOnly
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.GodotObject.PropertyName {
        /// <summary>
        /// Cached name for the 'ReadOnlyAutoProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @ReadOnlyAutoProperty = "ReadOnlyAutoProperty";
        /// <summary>
        /// Cached name for the 'ReadOnlyProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @ReadOnlyProperty = "ReadOnlyProperty";
        /// <summary>
        /// Cached name for the 'InitOnlyAutoProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @InitOnlyAutoProperty = "InitOnlyAutoProperty";
        /// <summary>
        /// Cached name for the 'ReadOnlyField' field.
        /// </summary>
        public new static readonly global::Godot.StringName @ReadOnlyField = "ReadOnlyField";
    }
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_ReadOnlyAutoProperty(object godotObject)
            {
                var ret = ((global::AllReadOnly)godotObject).@ReadOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get_ReadOnlyProperty(object godotObject)
            {
                var ret = ((global::AllReadOnly)godotObject).@ReadOnlyProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get_InitOnlyAutoProperty(object godotObject)
            {
                var ret = ((global::AllReadOnly)godotObject).@InitOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get_ReadOnlyField(object godotObject)
            {
                var ret = ((global::AllReadOnly)godotObject).@ReadOnlyField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            collector.TryAdd(PropertyName.@ReadOnlyAutoProperty, (new(&trampoline_get_ReadOnlyAutoProperty), new(null)));
            collector.TryAdd(PropertyName.@ReadOnlyProperty, (new(&trampoline_get_ReadOnlyProperty), new(null)));
            collector.TryAdd(PropertyName.@InitOnlyAutoProperty, (new(&trampoline_get_InitOnlyAutoProperty), new(null)));
            collector.TryAdd(PropertyName.@ReadOnlyField, (new(&trampoline_get_ReadOnlyField), new(null)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@ReadOnlyField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@ReadOnlyAutoProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@ReadOnlyProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@InitOnlyAutoProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
