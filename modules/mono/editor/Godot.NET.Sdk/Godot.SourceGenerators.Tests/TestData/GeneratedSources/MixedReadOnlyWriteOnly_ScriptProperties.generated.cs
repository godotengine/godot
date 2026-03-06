using Godot;
using Godot.NativeInterop;

partial class MixedReadOnlyWriteOnly
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
        /// Cached name for the 'WriteOnlyProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @WriteOnlyProperty = "WriteOnlyProperty";
        /// <summary>
        /// Cached name for the 'ReadOnlyField' field.
        /// </summary>
        public new static readonly global::Godot.StringName @ReadOnlyField = "ReadOnlyField";
        /// <summary>
        /// Cached name for the '_writeOnlyBackingField' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_writeOnlyBackingField = "_writeOnlyBackingField";
    }
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_ReadOnlyAutoProperty(object godotObject)
            {
                var ret = ((global::MixedReadOnlyWriteOnly)godotObject).@ReadOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get_ReadOnlyProperty(object godotObject)
            {
                var ret = ((global::MixedReadOnlyWriteOnly)godotObject).@ReadOnlyProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get_InitOnlyAutoProperty(object godotObject)
            {
                var ret = ((global::MixedReadOnlyWriteOnly)godotObject).@InitOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static void trampoline_set_WriteOnlyProperty(object godotObject, in godot_variant value)
            {
                ((global::MixedReadOnlyWriteOnly)godotObject).@WriteOnlyProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
            }
            static godot_variant trampoline_get_ReadOnlyField(object godotObject)
            {
                var ret = ((global::MixedReadOnlyWriteOnly)godotObject).@ReadOnlyField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static godot_variant trampoline_get__writeOnlyBackingField(object godotObject)
            {
                var ret = ((global::MixedReadOnlyWriteOnly)godotObject).@_writeOnlyBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<bool>(ret);
            }
            static void trampoline_set__writeOnlyBackingField(object godotObject, in godot_variant value)
            {
                ((global::MixedReadOnlyWriteOnly)godotObject).@_writeOnlyBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
            }
            collector.TryAdd(PropertyName.@ReadOnlyAutoProperty, (new(&trampoline_get_ReadOnlyAutoProperty), new(null)));
            collector.TryAdd(PropertyName.@ReadOnlyProperty, (new(&trampoline_get_ReadOnlyProperty), new(null)));
            collector.TryAdd(PropertyName.@InitOnlyAutoProperty, (new(&trampoline_get_InitOnlyAutoProperty), new(null)));
            collector.TryAdd(PropertyName.@WriteOnlyProperty, (new(null), new(&trampoline_set_WriteOnlyProperty)));
            collector.TryAdd(PropertyName.@ReadOnlyField, (new(&trampoline_get_ReadOnlyField), new(null)));
            collector.TryAdd(PropertyName.@_writeOnlyBackingField, (new(&trampoline_get__writeOnlyBackingField), new(&trampoline_set__writeOnlyBackingField)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)1, name: PropertyName.@_writeOnlyBackingField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)1, name: PropertyName.@WriteOnlyProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
