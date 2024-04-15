using Godot;
using System.Globalization;

namespace GodotTools.Build
{
    public class BuildProblemsFilter
    {
        public BuildDiagnostic.DiagnosticType Type { get; }

        public Button ToggleButton { get; }

        private int _problemsCount;

        public int ProblemsCount
        {
            get => _problemsCount;
            set
            {
                _problemsCount = value;
                ToggleButton.Text = _problemsCount.ToString(CultureInfo.InvariantCulture);
            }
        }

        public bool IsActive => ToggleButton.ButtonPressed;

        public BuildProblemsFilter(BuildDiagnostic.DiagnosticType type)
        {
            Type = type;
            ToggleButton = new Button
            {
                ToggleMode = true,
                ButtonPressed = true,
                Text = "0",
                FocusMode = Control.FocusModeEnum.None,
                ThemeTypeVariation = "EditorLogFilterButton",
            };
        }
    }
}
