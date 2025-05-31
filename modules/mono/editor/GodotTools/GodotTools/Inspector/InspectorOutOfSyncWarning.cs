using Godot;
using GodotTools.Internals;

namespace GodotTools.Inspector
{
    public partial class InspectorOutOfSyncWarning : HBoxContainer
    {
        public override void _Ready()
        {
            SetAnchorsPreset(LayoutPreset.TopWide);

            var iconTexture = GetThemeIcon("StatusWarning", "EditorIcons");

            var icon = new TextureRect()
            {
                Texture = iconTexture,
                ExpandMode = TextureRect.ExpandModeEnum.FitWidthProportional,
                CustomMinimumSize = iconTexture.GetSize(),
            };

            icon.SizeFlagsVertical = SizeFlags.ShrinkCenter;

            var label = new Label()
            {
                Text = "This inspector might be out of date. Please build the C# project.".TTR(),
                AutowrapMode = TextServer.AutowrapMode.WordSmart,
                CustomMinimumSize = new Vector2(100f, 0f),
            };

            label.AddThemeColorOverride("font_color", GetThemeColor("warning_color", "Editor"));
            label.SizeFlagsHorizontal = SizeFlags.Fill | SizeFlags.Expand;

            AddChild(icon);
            AddChild(label);
        }
    }
}
