using Godot;
using Godot.NativeInterop;

partial class ExportDiagnostics_GD0111
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the 'MyButtonGet' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonGet = "MyButtonGet";
        /// <summary>
        /// Cached name for the 'MyButtonGetSet' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonGetSet = "MyButtonGetSet";
        /// <summary>
        /// Cached name for the 'MyButtonGetWithBackingField' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonGetWithBackingField = "MyButtonGetWithBackingField";
        /// <summary>
        /// Cached name for the 'MyButtonGetSetWithBackingField' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonGetSetWithBackingField = "MyButtonGetSetWithBackingField";
        /// <summary>
        /// Cached name for the 'MyButtonOkWithCallableCreationExpression' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonOkWithCallableCreationExpression = "MyButtonOkWithCallableCreationExpression";
        /// <summary>
        /// Cached name for the 'MyButtonOkWithImplicitCallableCreationExpression' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonOkWithImplicitCallableCreationExpression = "MyButtonOkWithImplicitCallableCreationExpression";
        /// <summary>
        /// Cached name for the 'MyButtonOkWithCallableFromExpression' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButtonOkWithCallableFromExpression = "MyButtonOkWithCallableFromExpression";
        /// <summary>
        /// Cached name for the '_backingField' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_backingField = "_backingField";
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
    {
        if (name == PropertyName.@MyButtonGetSet) {
            this.@MyButtonGetSet = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            return true;
        }
        if (name == PropertyName.@MyButtonGetSetWithBackingField) {
            this.@MyButtonGetSetWithBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            return true;
        }
        if (name == PropertyName.@_backingField) {
            this.@_backingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            return true;
        }
        return base.SetGodotClassPropertyValue(name, value);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        if (name == PropertyName.@MyButtonGet) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonGet);
            return true;
        }
        if (name == PropertyName.@MyButtonGetSet) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonGetSet);
            return true;
        }
        if (name == PropertyName.@MyButtonGetWithBackingField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonGetWithBackingField);
            return true;
        }
        if (name == PropertyName.@MyButtonGetSetWithBackingField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonGetSetWithBackingField);
            return true;
        }
        if (name == PropertyName.@MyButtonOkWithCallableCreationExpression) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonOkWithCallableCreationExpression);
            return true;
        }
        if (name == PropertyName.@MyButtonOkWithImplicitCallableCreationExpression) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonOkWithImplicitCallableCreationExpression);
            return true;
        }
        if (name == PropertyName.@MyButtonOkWithCallableFromExpression) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@MyButtonOkWithCallableFromExpression);
            return true;
        }
        if (name == PropertyName.@_backingField) {
            value = global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(this.@_backingField);
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
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@_backingField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithImplicitCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableFromExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        return properties;
    }
#pragma warning restore CS0109
}
