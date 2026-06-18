using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

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
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<ExportDiagnostics_GD0111> PropertyRegistry = new ScriptPropertyRegistry<ExportDiagnostics_GD0111>()
        .Register(global::Godot.Node.PropertyRegistry)
        .Register(PropertyName.@MyButtonGetSet, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGetSet = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
                return value;
            })
        .Register(PropertyName.@MyButtonGetSetWithBackingField, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGetSetWithBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
                return value;
            })
        .Register(PropertyName.@_backingField, 1,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant value) =>
            {
                Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@_backingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
                return value;
            })
        .Register(PropertyName.@MyButtonGet, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGet;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonGetSet, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGetSet;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonGetWithBackingField, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGetWithBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonGetSetWithBackingField, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonGetSetWithBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonOkWithCallableCreationExpression, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonOkWithCallableCreationExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonOkWithImplicitCallableCreationExpression, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonOkWithImplicitCallableCreationExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@MyButtonOkWithCallableFromExpression, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@MyButtonOkWithCallableFromExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            })
        .Register(PropertyName.@_backingField, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0111>(ref scriptInstance).@_backingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
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
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@_backingField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithImplicitCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableFromExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        return properties;
    }
#pragma warning restore CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
}
