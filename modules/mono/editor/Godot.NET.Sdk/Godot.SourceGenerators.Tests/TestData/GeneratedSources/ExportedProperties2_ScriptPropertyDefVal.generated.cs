partial class ExportedProperties2
{
#pragma warning disable CS0109 // Disable warning about redundant 'new' keyword
#if TOOLS
    /// <summary>
    /// Get the default values for all properties declared in this class.
    /// This method is used by Godot to determine the value that will be
    /// used by the inspector when resetting properties.
    /// Do not call this method.
    /// </summary>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    internal new static global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant> GetGodotPropertyDefaultValues()
    {
        var values = new global::System.Collections.Generic.Dictionary<global::Godot.StringName, global::Godot.Variant>(3);
        int __Health_default_value = default;
        values.Add(PropertyName.@Health, global::Godot.Variant.From<int>(__Health_default_value));
        global::Godot.Resource __SubResource_default_value = default;
        values.Add(PropertyName.@SubResource, global::Godot.Variant.From<global::Godot.Resource>(__SubResource_default_value));
        string[] __Strings_default_value = default;
        values.Add(PropertyName.@Strings, global::Godot.Variant.From<string[]>(__Strings_default_value));
        return values;
    }
#endif // TOOLS
#pragma warning restore CS0109
}
