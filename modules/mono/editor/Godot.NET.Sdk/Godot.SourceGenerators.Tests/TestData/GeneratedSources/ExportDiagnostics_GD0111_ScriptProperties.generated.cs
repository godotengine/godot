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
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_MyButtonGet(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGet;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static godot_variant trampoline_get_MyButtonGetSet(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGetSet;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static void trampoline_set_MyButtonGetSet(object godotObject, in godot_variant value)
            {
                ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGetSet = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            }
            static godot_variant trampoline_get_MyButtonGetWithBackingField(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGetWithBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static godot_variant trampoline_get_MyButtonGetSetWithBackingField(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGetSetWithBackingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static void trampoline_set_MyButtonGetSetWithBackingField(object godotObject, in godot_variant value)
            {
                ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonGetSetWithBackingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            }
            static godot_variant trampoline_get_MyButtonOkWithCallableCreationExpression(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonOkWithCallableCreationExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static godot_variant trampoline_get_MyButtonOkWithImplicitCallableCreationExpression(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonOkWithImplicitCallableCreationExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static godot_variant trampoline_get_MyButtonOkWithCallableFromExpression(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@MyButtonOkWithCallableFromExpression;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static godot_variant trampoline_get__backingField(object godotObject)
            {
                var ret = ((global::ExportDiagnostics_GD0111)godotObject).@_backingField;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
            }
            static void trampoline_set__backingField(object godotObject, in godot_variant value)
            {
                ((global::ExportDiagnostics_GD0111)godotObject).@_backingField = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.Callable>(value);
            }
            collector.TryAdd(PropertyName.@MyButtonGet, (new(&trampoline_get_MyButtonGet), new(null)));
            collector.TryAdd(PropertyName.@MyButtonGetSet, (new(&trampoline_get_MyButtonGetSet), new(&trampoline_set_MyButtonGetSet)));
            collector.TryAdd(PropertyName.@MyButtonGetWithBackingField, (new(&trampoline_get_MyButtonGetWithBackingField), new(null)));
            collector.TryAdd(PropertyName.@MyButtonGetSetWithBackingField, (new(&trampoline_get_MyButtonGetSetWithBackingField), new(&trampoline_set_MyButtonGetSetWithBackingField)));
            collector.TryAdd(PropertyName.@MyButtonOkWithCallableCreationExpression, (new(&trampoline_get_MyButtonOkWithCallableCreationExpression), new(null)));
            collector.TryAdd(PropertyName.@MyButtonOkWithImplicitCallableCreationExpression, (new(&trampoline_get_MyButtonOkWithImplicitCallableCreationExpression), new(null)));
            collector.TryAdd(PropertyName.@MyButtonOkWithCallableFromExpression, (new(&trampoline_get_MyButtonOkWithCallableFromExpression), new(null)));
            collector.TryAdd(PropertyName.@_backingField, (new(&trampoline_get__backingField), new(&trampoline_set__backingField)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@_backingField, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithImplicitCallableCreationExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        properties.Add(new(type: (global::Godot.Variant.Type)25, name: PropertyName.@MyButtonOkWithCallableFromExpression, hint: (global::Godot.PropertyHint)39, hintString: "", usage: (global::Godot.PropertyUsageFlags)4, exported: true));
        return properties;
    }
#pragma warning restore CS0109
}
