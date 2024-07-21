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
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
    {
        if (name == PropertyName.@WriteOnlyProperty) {
            this.@WriteOnlyProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
            return true;
        }
        if (name == PropertyName.@_writeOnlyBackingField) {
            this.@_writeOnlyBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
            return true;
        }
        return base.SetGodotClassPropertyValue(name, value);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        if (name == PropertyName.@ReadOnlyAutoProperty) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.@ReadOnlyAutoProperty);
            return true;
        }
        if (name == PropertyName.@ReadOnlyProperty) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.@ReadOnlyProperty);
            return true;
        }
        if (name == PropertyName.@InitOnlyAutoProperty) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.@InitOnlyAutoProperty);
            return true;
        }
        if (name == PropertyName.@ReadOnlyField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.@ReadOnlyField);
            return true;
        }
        if (name == PropertyName.@_writeOnlyBackingField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<bool>(this.@_writeOnlyBackingField);
            return true;
        }
        return base.GetGodotClassPropertyValue(name, out value);
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
