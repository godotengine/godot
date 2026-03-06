using Godot;
using Godot.NativeInterop;

partial class AbstractGenericNode<T>
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the 'MyArray' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyArray = "MyArray";
    }
    private static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.ScriptManagerBridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_MyArray(object godotObject)
            {
                var ret = ((global::AbstractGenericNode<T>)godotObject).@MyArray;
                return global::Godot.NativeInterop.VariantUtils.CreateFromArray(ret);
            }
            static void trampoline_set_MyArray(object godotObject, in godot_variant value)
            {
                ((global::AbstractGenericNode<T>)godotObject).@MyArray = global::Godot.NativeInterop.VariantUtils.ConvertToArray<T>(value);
            }
            collector.TryAdd(PropertyName.@MyArray, (new(&trampoline_get_MyArray), new(&trampoline_set_MyArray)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)28, name: PropertyName.@MyArray, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4102, exported: true));
        return properties;
    }
#pragma warning restore CS0109
}
