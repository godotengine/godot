using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

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
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<AllReadOnly> PropertyRegistry = new ScriptPropertyRegistry<AllReadOnly>()
        .Register(global::Godot.GodotObject.PropertyRegistry)
        .Register(PropertyName.@ReadOnlyAutoProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, AllReadOnly>(ref scriptInstance).@ReadOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@ReadOnlyProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, AllReadOnly>(ref scriptInstance).@ReadOnlyProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@InitOnlyAutoProperty, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, AllReadOnly>(ref scriptInstance).@InitOnlyAutoProperty;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Register(PropertyName.@ReadOnlyField, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, AllReadOnly>(ref scriptInstance).@ReadOnlyField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            })
        .Build();
#pragma warning restore CS0618 // Type or member is obsolete

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
