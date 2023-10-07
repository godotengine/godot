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
        /// Cached name for the 'readonly_auto_property' property.
        /// </summary>
        public new static readonly global::Godot.StringName readonly_auto_property = "readonly_auto_property";
        /// <summary>
        /// Cached name for the 'readonly_property' property.
        /// </summary>
        public new static readonly global::Godot.StringName readonly_property = "readonly_property";
        /// <summary>
        /// Cached name for the 'initonly_auto_property' property.
        /// </summary>
        public new static readonly global::Godot.StringName initonly_auto_property = "initonly_auto_property";
        /// <summary>
        /// Cached name for the 'readonly_field' field.
        /// </summary>
        public new static readonly global::Godot.StringName readonly_field = "readonly_field";
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        if (name == PropertyName.readonly_auto_property) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.readonly_auto_property);
            return true;
        }
        else if (name == PropertyName.readonly_property) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.readonly_property);
            return true;
        }
        else if (name == PropertyName.initonly_auto_property) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.initonly_auto_property);
            return true;
        }
        else if (name == PropertyName.readonly_field) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(this.readonly_field);
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
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.readonly_field, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.readonly_auto_property, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.readonly_property, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.initonly_auto_property, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
