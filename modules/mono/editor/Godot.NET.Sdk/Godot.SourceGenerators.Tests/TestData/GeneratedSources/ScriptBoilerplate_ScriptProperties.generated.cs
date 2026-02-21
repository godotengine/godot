using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class ScriptBoilerplate
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the '_nodePath' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_nodePath = "_nodePath";
        /// <summary>
        /// Cached name for the '_velocity' field.
        /// </summary>
        public new static readonly global::Godot.StringName @_velocity = "_velocity";
    }
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<ScriptBoilerplate> PropertyRegistry = new ScriptPropertyRegistry<ScriptBoilerplate>()
        .Register(global::Godot.Node.PropertyRegistry)
        .Register(PropertyName.@_nodePath, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance).@_nodePath = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.NodePath>(value);
                return value;
            })
        .Register(PropertyName.@_velocity, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance).@_velocity = global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(value);
                return value;
            })
        .Register(PropertyName.@_nodePath, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance).@_nodePath;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.NodePath>(ret);
            })
        .Register(PropertyName.@_velocity, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ScriptBoilerplate>(ref scriptInstance).@_velocity;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
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

    /// <summary>
    /// Get the property information for all the properties declared in this class.
    /// This method is used by Godot to register the available properties in the editor.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo> GetGodotPropertyList()
    {
        var properties = new global::System.Collections.Generic.List<global::Godot.Bridge.PropertyInfo>();
        properties.Add(new(type: (global::Godot.Variant.Type)22, name: PropertyName.@_nodePath, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@_velocity, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
