using Godot;
using Godot.NativeInterop;

partial class InheritanceChild
{
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override void SaveGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.SaveGodotObjectData(info);
        info.AddProperty(PropertyName.@MyString, global::Godot.Variant.From<string>(this.@MyString));
        info.AddProperty(PropertyName.@MyInteger, global::Godot.Variant.From<int>(this.@MyInteger));
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    protected override void RestoreGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.RestoreGodotObjectData(info);
        if (info.TryGetProperty(PropertyName.@MyString, out var _value_MyString))
            this.@MyString = _value_MyString.As<string>();
        if (info.TryGetProperty(PropertyName.@MyInteger, out var _value_MyInteger))
            this.@MyInteger = _value_MyInteger.As<int>();
    }
}
