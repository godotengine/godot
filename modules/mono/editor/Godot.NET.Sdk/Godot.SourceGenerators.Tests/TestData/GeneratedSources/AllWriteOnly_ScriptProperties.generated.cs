using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class AllWriteOnly
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.GodotObject.PropertyName {
        /// <summary>
        /// Cached name for the 'WriteOnlyProperty' property.
        /// </summary>
        public new static readonly global::Godot.StringName @WriteOnlyProperty = "WriteOnlyProperty";
        /// <summary>
        /// Cached name for the '_writeOnlyBackingField' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_writeOnlyBackingField = "_writeOnlyBackingField";
    }
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<AllWriteOnly> PropertyRegistry = new ScriptPropertyRegistry<AllWriteOnly>()
        .Register(global::Godot.GodotObject.PropertyRegistry)
        .Register(PropertyName.@WriteOnlyProperty, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, AllWriteOnly>(ref scriptInstance).@WriteOnlyProperty = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
                return value;
            })
        .Register(PropertyName.@_writeOnlyBackingField, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, AllWriteOnly>(ref scriptInstance).@_writeOnlyBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<bool>(value);
                return value;
            })
        .Register(PropertyName.@_writeOnlyBackingField, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, AllWriteOnly>(ref scriptInstance).@_writeOnlyBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<bool>(ret);
            })
        .Build();
#pragma warning restore CS0618 // Type or member is obsolete

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool SetGodotClassPropertyValue(in godot_string_name name, in godot_variant value)
    {
        ref readonly var propertySetter = ref PropertyRegistry.GetMethodOrNullRef(in name, 1);
        if (!Unsafe.IsNullRef(in propertySetter))
        {
            propertySetter(this, value);
            return true;
        }
        return false;
    }

    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override bool GetGodotClassPropertyValue(in godot_string_name name, out godot_variant value)
    {
        ref readonly var propertyGetter = ref PropertyRegistry.GetMethodOrNullRef(in name, 0);
        if (!Unsafe.IsNullRef(in propertyGetter))
        {
            value = propertyGetter(this, default);
            return true;
        }
        value = default;
        return false;
    }

#pragma warning disable CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
    /// <summary>
    /// Get the property information for all the properties declared in this class.
    /// This method is used by Godot to register the available properties in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo> GetGodotPropertyList()
    {
        var properties = new global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>();
        properties.Add(new(type: (global::Godot.Variant.Type)1, name: PropertyName.@_writeOnlyBackingField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)1, name: PropertyName.@WriteOnlyProperty, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
}
