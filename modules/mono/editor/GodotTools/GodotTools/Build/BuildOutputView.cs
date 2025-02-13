using Godot;
using static GodotTools.Internals.Globals;

namespace GodotTools.Build
{
    public partial class BuildOutputView : HBoxContainer
    {
#nullable disable
        private RichTextLabel _log;

        private Button _clearButton;
        private Button _copyButton;
#nullable enable

        public void Append(string text)
        {
            _log.AddText(text);
        }

        public void Clear()
        {
            _log.Clear();
        }

        private void CopyRequested()
        {
            string text = _log.GetSelectedText();

            if (string.IsNullOrEmpty(text))
                text = _log.GetParsedText();

            if (!string.IsNullOrEmpty(text))
                DisplayServer.ClipboardSet(text);
        }

        public override void _Ready()
        {
            Name = "Output".TTR();

            var vbLeft = new VBoxContainer
            {
                CustomMinimumSize = new Vector2(0, 180 * EditorScale),
                SizeFlagsVertical = SizeFlags.ExpandFill,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            };
            AddChild(vbLeft);

            // Log - Rich Text Label.
            _log = new RichTextLabel
            {
                BbcodeEnabled = true,
                ScrollFollowing = true,
                SelectionEnabled = true,
                ContextMenuEnabled = true,
                FocusMode = FocusModeEnum.Click,
                SizeFlagsVertical = SizeFlags.ExpandFill,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                DeselectOnFocusLossEnabled = false,

            };
            vbLeft.AddChild(_log);

            var vbRight = new VBoxContainer();
            AddChild(vbRight);

            // Tools grid
            var hbTools = new HBoxContainer
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            };
            vbRight.AddChild(hbTools);

            // Clear.
            _clearButton = new Button
            {
                ThemeTypeVariation = "FlatButton",
                FocusMode = FocusModeEnum.None,
                Shortcut = EditorDefShortcut("editor/clear_output", "Clear Output".TTR(), (Key)KeyModifierMask.MaskCmdOrCtrl | (Key)KeyModifierMask.MaskShift | Key.K),
            };
            _clearButton.Pressed += Clear;
            hbTools.AddChild(_clearButton);

            // Copy.
            _copyButton = new Button
            {
                ThemeTypeVariation = "FlatButton",
                FocusMode = FocusModeEnum.None,
                Shortcut = EditorDefShortcut("editor/copy_output", "Copy Selection".TTR(), (Key)KeyModifierMask.MaskCmdOrCtrl | Key.C),
                ShortcutContext = this,
            };
            _copyButton.Pressed += CopyRequested;
            hbTools.AddChild(_copyButton);

            UpdateTheme();
        }

        public override void _Notification(int what)
        {
            base._Notification(what);

            if (what == NotificationThemeChanged)
            {
                UpdateTheme();
            }
        }

        private void UpdateTheme()
        {
            // Nodes will be null until _Ready is called.
            if (_log == null)
                return;

            var normalFont = GetThemeFont("output_source", "EditorFonts");
            if (normalFont != null)
                _log.AddThemeFontOverride("normal_font", normalFont);

            var boldFont = GetThemeFont("output_source_bold", "EditorFonts");
            if (boldFont != null)
                _log.AddThemeFontOverride("bold_font", boldFont);

            var italicsFont = GetThemeFont("output_source_italic", "EditorFonts");
            if (italicsFont != null)
                _log.AddThemeFontOverride("italics_font", italicsFont);

            var boldItalicsFont = GetThemeFont("output_source_bold_italic", "EditorFonts");
            if (boldItalicsFont != null)
                _log.AddThemeFontOverride("bold_italics_font", boldItalicsFont);

            var monoFont = GetThemeFont("output_source_mono", "EditorFonts");
            if (monoFont != null)
                _log.AddThemeFontOverride("mono_font", monoFont);

            // Disable padding for highlighted background/foreground to prevent highlights from overlapping on close lines.
            // This also better matches terminal output, which does not use any form of padding.
            _log.AddThemeConstantOverride("text_highlight_h_padding", 0);
            _log.AddThemeConstantOverride("text_highlight_v_padding", 0);

            int font_size = GetThemeFontSize("output_source_size", "EditorFonts");
            _log.AddThemeFontSizeOverride("normal_font_size", font_size);
            _log.AddThemeFontSizeOverride("bold_font_size", font_size);
            _log.AddThemeFontSizeOverride("italics_font_size", font_size);
            _log.AddThemeFontSizeOverride("mono_font_size", font_size);

            _clearButton.Icon = GetThemeIcon("Clear", "EditorIcons");
            _copyButton.Icon = GetThemeIcon("ActionCopy", "EditorIcons");
        }
    }
}
