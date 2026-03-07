using Godot;
using Godot.NativeInterop;
using Godot.Bridge;
using System.Runtime.CompilerServices;

partial class ExportDiagnostics_GD0108
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the 'MyButton' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyButton = "MyButton";
    }
#pragma warning restore CS0109 // Disable warning about redundant 'new' keyword

#pragma warning disable CS0618 // Type or member is obsolete
    protected new static readonly ScriptPropertyRegistry<ExportDiagnostics_GD0108> PropertyRegistry = new ScriptPropertyRegistry<ExportDiagnostics_GD0108>()
        .Register(global::Godot.Node.PropertyRegistry)
        .Register(PropertyName.@MyButton, 0,
            [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
            static (GodotObject scriptInstance, scoped in godot_variant _) =>
            {
                var ret = Unsafe.As<GodotObject, ExportDiagnostics_GD0108>(ref scriptInstance).@MyButton;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.Callable>(ret);
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
        return properties;
    }
#pragma warning restore CS0109 // The member 'member' does not hide an inherited member. The new keyword is not required
}
