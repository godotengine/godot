using Godot;
using Godot.NativeInterop;

partial class Generic<T>
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.GodotObject.PropertyName {
        /// <summary>
        /// Cached name for the 'RegularProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName RegularProperty = "RegularProperty";
        /// <summary>
        /// Cached name for the 'ArrayProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName ArrayProperty = "ArrayProperty";
        /// <summary>
        /// Cached name for the 'RegularField' field.
        /// </summary>
        public new static readonly global::Godot.StringName RegularField = "RegularField";
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
    {
        if (name == PropertyName.RegularProperty) {
            this.RegularProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<T>(value);
            return true;
        }
        if (name == PropertyName.ArrayProperty) {
            this.ArrayProperty = global::Godot.NativeInterop.VariantUtils.ConvertToArray<T>(value);
            return true;
        }
        if (name == PropertyName.RegularField) {
            this.RegularField = global::Godot.NativeInterop.VariantUtils.ConvertTo<T>(value);
            return true;
        }
        return base.SetGodotClassPropertyValue(name, value);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        if (name == PropertyName.RegularProperty) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<T>(this.RegularProperty);
            return true;
        }
        if (name == PropertyName.ArrayProperty) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFromArray(this.ArrayProperty);
            return true;
        }
        if (name == PropertyName.RegularField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<T>(this.RegularField);
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
        properties.Add(global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: PropertyName.RegularField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<T>(name: PropertyName.RegularProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        properties.Add(global::Godot.Bridge.GenericUtils.PropertyInfoFromGenericType<global::Godot.Collections.Array<T>>(name: PropertyName.ArrayProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        return properties;
    }
#pragma warning restore CS0109
}
