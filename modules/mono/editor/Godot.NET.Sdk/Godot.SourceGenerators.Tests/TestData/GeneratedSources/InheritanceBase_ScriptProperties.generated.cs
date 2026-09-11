using Godot;
using Godot.NativeInterop;

partial class InheritanceBase
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
    /// <summary>
    /// Cached StringNames for the properties and fields contained in this class, for fast lookup.
    /// </summary>
    public new class PropertyName : global::Godot.Node.PropertyName {
        /// <summary>
        /// Cached name for the 'MyString' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyString = "MyString";
        /// <summary>
        /// Cached name for the 'MyInteger' property.
        /// </summary>
        public new static readonly global::Godot.StringName @MyInteger = "MyInteger";
    }
    protected internal new static partial class GodotInternal
    {
        internal new static unsafe void GetGodotPropertyTrampolines(global::Godot.Bridge.PropertyTrampolineCollector collector)
        {
            static godot_variant trampoline_get_MyString(object godotObject)
            {
                var ret = ((global::InheritanceBase)godotObject).@MyString;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<string>(ret);
            }
            static void trampoline_set_MyString(object godotObject, in godot_variant value)
            {
                ((global::InheritanceBase)godotObject).@MyString = global::Godot.NativeInterop.VariantUtils.ConvertTo<string>(value);
            }
            static godot_variant trampoline_get_MyInteger(object godotObject)
            {
                var ret = ((global::InheritanceBase)godotObject).@MyInteger;
                return global::Godot.NativeInterop.VariantUtils.CreateFrom<int>(ret);
            }
            static void trampoline_set_MyInteger(object godotObject, in godot_variant value)
            {
                ((global::InheritanceBase)godotObject).@MyInteger = global::Godot.NativeInterop.VariantUtils.ConvertTo<int>(value);
            }
            collector.TryAdd(PropertyName.@MyString, (new(&trampoline_get_MyString), new(&trampoline_set_MyString)));
            collector.TryAdd(PropertyName.@MyInteger, (new(&trampoline_get_MyInteger), new(&trampoline_set_MyInteger)));
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
        properties.Add(new(type: (global::Godot.Variant.Type)4, name: PropertyName.@MyString, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        properties.Add(new(type: (global::Godot.Variant.Type)2, name: PropertyName.@MyInteger, hint: (global::Godot.PropertyHint)0, hintString: "", usage: (global::Godot.PropertyUsageFlags)4096, exported: false));
        return properties;
    }
#pragma warning restore CS0109
}
