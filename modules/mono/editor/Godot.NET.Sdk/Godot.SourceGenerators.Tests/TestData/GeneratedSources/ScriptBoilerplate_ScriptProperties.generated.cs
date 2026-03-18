using Godot;
using Godot.NativeInterop;

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
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get__nodePath(object godotObject)
            {
                var ret = ((global::ScriptBoilerplate)godotObject).@_nodePath;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<global::Godot.NodePath>(ret);
            }
            static void trampoline_set__nodePath(object godotObject, in godot_variant value)
            {
                ((global::ScriptBoilerplate)godotObject).@_nodePath = global::Godot.NativeInterop.VariantUtils.ConvertTo<global::Godot.NodePath>(value);
            }
            static godot_variant trampoline_get__velocity(object godotObject)
            {
                var ret = ((global::ScriptBoilerplate)godotObject).@_velocity;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
            }
            static void trampoline_set__velocity(object godotObject, in godot_variant value)
            {
                ((global::ScriptBoilerplate)godotObject).@_velocity = global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(value);
            }
            collector.TryAdd(PropertyName.@_nodePath, (new(&trampoline_get__nodePath), new(&trampoline_set__nodePath)));
            collector.TryAdd(PropertyName.@_velocity, (new(&trampoline_get__velocity), new(&trampoline_set__velocity)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)22, name: PropertyName.@_nodePath, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@_velocity, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
