using Godot;
using Godot.NativeInterop;

partial struct OuterClass
{
partial class NestedClass
{
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void SaveGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.SaveGodotObjectData(info);
    }
    /// <inheritdoc/>
    [global::System.ComponentModel.EditorBrowsable(global::System.ComponentModel.EditorBrowsableState.Never)]
    [global::System.Diagnostics.CodeAnalysis.RequiresUnreferencedCode("This method is for use by the Godot editor only. The overriding methods might not be compatible with trimming.")]
    [global::System.Diagnostics.CodeAnalysis.RequiresDynamicCode("This method is for use by the Godot editor only. The overriding methods might require dynamic code, for which native code might not be available at runtime.")]
    protected override void RestoreGodotObjectData(global::Godot.Bridge.GodotSerializationInfo info)
    {
        base.RestoreGodotObjectData(info);
    }
}
}
