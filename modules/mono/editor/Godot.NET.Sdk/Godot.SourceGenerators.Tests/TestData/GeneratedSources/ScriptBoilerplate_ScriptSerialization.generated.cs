using Godot;
using Godot.NativeInterop;

partial class ScriptBoilerplate
{
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void SaveGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.SaveGodotObjectData(info);
        info.AddProperty(PropertyName.@_nodePath, global::Godot.Variant.From<global::Godot.NodePath>(this.@_nodePath));
        info.AddProperty(PropertyName.@_velocity, global::Godot.Variant.From<int>(this.@_velocity));
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void RestoreGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.RestoreGodotObjectData(info);
        if (info.TryGetProperty(PropertyName.@_nodePath, out var _value__nodePath))
            this.@_nodePath = _value__nodePath.As<global::Godot.NodePath>();
        if (info.TryGetProperty(PropertyName.@_velocity, out var _value__velocity))
            this.@_velocity = _value__velocity.As<int>();
    }
}
