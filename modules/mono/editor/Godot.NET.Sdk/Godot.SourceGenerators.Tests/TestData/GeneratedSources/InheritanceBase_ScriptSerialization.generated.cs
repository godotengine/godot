using Godot;
using Godot.NativeInterop;

partial class InheritanceBase
{
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void SaveGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.SaveGodotObjectData(info);
        info.AddProperty(PropertyName.@MyString, global::Godot.Variant.From<string>(this.@MyString));
        info.AddProperty(PropertyName.@MyInteger, global::Godot.Variant.From<int>(this.@MyInteger));
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void RestoreGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.RestoreGodotObjectData(info);
        if (info.TryGetProperty(PropertyName.@MyString, out var _value_MyString))
            this.@MyString = _value_MyString.As<string>();
        if (info.TryGetProperty(PropertyName.@MyInteger, out var _value_MyInteger))
            this.@MyInteger = _value_MyInteger.As<int>();
    }
}
