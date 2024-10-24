using Godot;
using GodotTools.Internals;

namespace GodotTools.Inspector;

public partial class ExternalTypeReminder : HBoxContainer
{
    public override void _Ready()
    {
        SetAnchorsPreset(LayoutPreset.TopWide);

        var iconTexture = GetThemeIcon("NodeInfo", "EditorIcons");

        var icon = new TextureRect()
        {
            Texture = iconTexture,
            ExpandMode = TextureRect.ExpandModeEnum.FitWidthProportional,
            CustomMinimumSize = iconTexture.GetSize(),
        };

        icon.SizeFlagsVertical = SizeFlags.ShrinkCenter;

        var label = new Label()
        {
            Text = "This inspector inherits an external type. Please build the C# project when the external type changes.".TTR(),
            AutowrapMode = TextServer.AutowrapMode.WordSmart,
            CustomMinimumSize = new Vector2(100f, 0f),
        };

        label.SizeFlagsHorizontal = SizeFlags.Fill | SizeFlags.Expand;

        AddChild(icon);
        AddChild(label);
    }
}
