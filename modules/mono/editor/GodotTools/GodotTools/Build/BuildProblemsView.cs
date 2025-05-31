using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Godot;
using GodotTools.Internals;
using static GodotTools.Internals.Globals;
using FileAccess = Godot.FileAccess;

namespace GodotTools.Build
{
    public partial class BuildProblemsView : HBoxContainer
    {
#nullable disable
        private Button _clearButton;
        private Button _copyButton;

        private Button _toggleLayoutButton;

        private Button _showSearchButton;
        private LineEdit _searchBox;
#nullable enable

        private readonly Dictionary<BuildDiagnostic.DiagnosticType, BuildProblemsFilter> _filtersByType = new();

#nullable disable
        private Tree _problemsTree;
        private PopupMenu _problemsContextMenu;
#nullable enable

        public enum ProblemsLayout { List, Tree }
        private ProblemsLayout _layout = ProblemsLayout.Tree;

        private readonly List<BuildDiagnostic> _diagnostics = new();

        public int TotalDiagnosticCount => _diagnostics.Count;

        private readonly Dictionary<BuildDiagnostic.DiagnosticType, int> _problemCountByType = new();

        public int WarningCount =>
            GetProblemCountForType(BuildDiagnostic.DiagnosticType.Warning);

        public int ErrorCount =>
            GetProblemCountForType(BuildDiagnostic.DiagnosticType.Error);

        private int GetProblemCountForType(BuildDiagnostic.DiagnosticType type)
        {
            if (!_problemCountByType.TryGetValue(type, out int count))
            {
                count = _diagnostics.Count(d => d.Type == type);
                _problemCountByType[type] = count;
            }

            return count;
        }

        private static IEnumerable<BuildDiagnostic> ReadDiagnosticsFromFile(string csvFile)
        {
            using var file = FileAccess.Open(csvFile, FileAccess.ModeFlags.Read);

            if (file == null)
                yield break;

            while (!file.EofReached())
            {
                string[] csvColumns = file.GetCsvLine();

                if (csvColumns.Length == 1 && string.IsNullOrEmpty(csvColumns[0]))
                    yield break;

                if (csvColumns.Length != 7)
                {
                    GD.PushError($"Expected 7 columns, got {csvColumns.Length}");
                    continue;
                }

                var diagnostic = new BuildDiagnostic
                {
                    Type = csvColumns[0] switch
                    {
                        "warning" => BuildDiagnostic.DiagnosticType.Warning,
                        "error" or _ => BuildDiagnostic.DiagnosticType.Error,
                    },
                    File = csvColumns[1],
                    Line = int.Parse(csvColumns[2], CultureInfo.InvariantCulture),
                    Column = int.Parse(csvColumns[3], CultureInfo.InvariantCulture),
                    Code = csvColumns[4],
                    Message = csvColumns[5],
                    ProjectFile = csvColumns[6],
                };

                // If there's no ProjectFile but the File is a csproj, then use that.
                if (string.IsNullOrEmpty(diagnostic.ProjectFile) &&
                    !string.IsNullOrEmpty(diagnostic.File) &&
                    diagnostic.File.EndsWith(".csproj", StringComparison.OrdinalIgnoreCase))
                {
                    diagnostic.ProjectFile = diagnostic.File;
                }

                yield return diagnostic;
            }
        }

        public void SetDiagnosticsFromFile(string csvFile)
        {
            var diagnostics = ReadDiagnosticsFromFile(csvFile);
            SetDiagnostics(diagnostics);
        }

        public void SetDiagnostics(IEnumerable<BuildDiagnostic> diagnostics)
        {
            _diagnostics.Clear();
            _problemCountByType.Clear();

            _diagnostics.AddRange(diagnostics);
            UpdateProblemsView();
        }

        public void Clear()
        {
            _problemsTree.Clear();
            _diagnostics.Clear();
            _problemCountByType.Clear();

            UpdateProblemsView();
        }

        private void CopySelectedProblems()
        {
            var selectedItem = _problemsTree.GetNextSelected(null);
            if (selectedItem == null)
                return;

            var selectedIdxs = new List<int>();
            while (selectedItem != null)
            {
                int selectedIdx = (int)selectedItem.GetMetadata(0);
                selectedIdxs.Add(selectedIdx);

                selectedItem = _problemsTree.GetNextSelected(selectedItem);
            }

            if (selectedIdxs.Count == 0)
                return;

            var selectedDiagnostics = selectedIdxs.Select(i => _diagnostics[i]);

            var sb = new StringBuilder();

            foreach (var diagnostic in selectedDiagnostics)
            {
                if (!string.IsNullOrEmpty(diagnostic.Code))
                    sb.Append(CultureInfo.InvariantCulture, $"{diagnostic.Code}: ");

                sb.AppendLine(CultureInfo.InvariantCulture, $"{diagnostic.Message} {diagnostic.File}({diagnostic.Line},{diagnostic.Column})");
            }

            string text = sb.ToString();

            if (!string.IsNullOrEmpty(text))
                DisplayServer.ClipboardSet(text);
        }

        private void ToggleLayout(bool pressed)
        {
            _layout = pressed ? ProblemsLayout.List : ProblemsLayout.Tree;

            var editorSettings = EditorInterface.Singleton.GetEditorSettings();
            editorSettings.SetSetting(GodotSharpEditor.Settings.ProblemsLayout, Variant.From(_layout));

            _toggleLayoutButton.Icon = GetToggleLayoutIcon();
            _toggleLayoutButton.TooltipText = GetToggleLayoutTooltipText();

            UpdateProblemsView();
        }

        private bool GetToggleLayoutPressedState()
        {
            // If pressed: List layout.
            // If not pressed: Tree layout.
            return _layout == ProblemsLayout.List;
        }

        private Texture2D? GetToggleLayoutIcon()
        {
            return _layout switch
            {
                ProblemsLayout.List => GetThemeIcon("FileList", "EditorIcons"),
                ProblemsLayout.Tree or _ => GetThemeIcon("FileTree", "EditorIcons"),
            };
        }

        private string GetToggleLayoutTooltipText()
        {
            return _layout switch
            {
                ProblemsLayout.List => "View as a Tree".TTR(),
                ProblemsLayout.Tree or _ => "View as a List".TTR(),
            };
        }

        private void ToggleSearchBoxVisibility(bool pressed)
        {
            _searchBox.Visible = pressed;
            if (pressed)
            {
                _searchBox.GrabFocus();
            }
        }

        private void SearchTextChanged(string text)
        {
            UpdateProblemsView();
        }

        private void ToggleFilter(bool pressed)
        {
            UpdateProblemsView();
        }

        private void GoToSelectedProblem()
        {
            var selectedItem = _problemsTree.GetSelected();
            if (selectedItem == null)
                throw new InvalidOperationException("Item tree has no selected items.");

            // Get correct diagnostic index from problems tree.
            int diagnosticIndex = (int)selectedItem.GetMetadata(0);

            if (diagnosticIndex < 0 || diagnosticIndex >= _diagnostics.Count)
                throw new InvalidOperationException("Diagnostic index out of range.");

            var diagnostic = _diagnostics[diagnosticIndex];

            if (string.IsNullOrEmpty(diagnostic.ProjectFile) && string.IsNullOrEmpty(diagnostic.File))
                return;

            string? projectDir = !string.IsNullOrEmpty(diagnostic.ProjectFile) ?
                diagnostic.ProjectFile.GetBaseDir() :
                GodotSharpEditor.Instance.MSBuildPanel.LastBuildInfo?.Solution.GetBaseDir();
            if (string.IsNullOrEmpty(projectDir))
                return;

            string? file = !string.IsNullOrEmpty(diagnostic.File) ?
                Path.Combine(projectDir.SimplifyGodotPath(), diagnostic.File.SimplifyGodotPath()) :
                null;

            if (!File.Exists(file))
                return;

            file = ProjectSettings.LocalizePath(file);

            if (file.StartsWith("res://", StringComparison.Ordinal))
            {
                var script = (Script)ResourceLoader.Load(file, typeHint: Internal.CSharpLanguageType);

                // Godot's ScriptEditor.Edit is 0-based but the diagnostic lines are 1-based.
                if (script != null && Internal.ScriptEditorEdit(script, diagnostic.Line - 1, diagnostic.Column - 1))
                    Internal.EditorNodeShowScriptScreen();
            }
        }

        private void ShowProblemContextMenu(Vector2 position, long mouseButtonIndex)
        {
            if (mouseButtonIndex != (long)MouseButton.Right)
                return;

            _problemsContextMenu.Clear();
            _problemsContextMenu.Size = new Vector2I(1, 1);

            var selectedItem = _problemsTree.GetSelected();
            if (selectedItem != null)
            {
                // Add menu entries for the selected item.
                _problemsContextMenu.AddIconItem(GetThemeIcon("ActionCopy", "EditorIcons"),
                    label: "Copy Error".TTR(), (int)ProblemContextMenuOption.Copy);
            }

            if (_problemsContextMenu.ItemCount > 0)
            {
                _problemsContextMenu.Position = (Vector2I)(GetScreenPosition() + position);
                _problemsContextMenu.Popup();
            }
        }

        private enum ProblemContextMenuOption
        {
            Copy,
        }

        private void ProblemContextOptionPressed(long id)
        {
            switch ((ProblemContextMenuOption)id)
            {
                case ProblemContextMenuOption.Copy:
                    CopySelectedProblems();
                    break;

                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Invalid problem context menu option.");
            }
        }

        private bool ShouldDisplayDiagnostic(BuildDiagnostic diagnostic)
        {
            if (!_filtersByType[diagnostic.Type].IsActive)
                return false;

            string searchText = _searchBox.Text;
            if (string.IsNullOrEmpty(searchText))
                return true;
            if (diagnostic.Message.Contains(searchText, StringComparison.OrdinalIgnoreCase))
                return true;
            if (diagnostic.File?.Contains(searchText, StringComparison.OrdinalIgnoreCase) ?? false)
                return true;

            return false;
        }

        private Color? GetProblemItemColor(BuildDiagnostic diagnostic)
        {
            return diagnostic.Type switch
            {
                BuildDiagnostic.DiagnosticType.Warning => GetThemeColor("warning_color", "Editor"),
                BuildDiagnostic.DiagnosticType.Error => GetThemeColor("error_color", "Editor"),
                _ => null,
            };
        }

        public void UpdateProblemsView()
        {
            switch (_layout)
            {
                case ProblemsLayout.List:
                    UpdateProblemsList();
                    break;

                case ProblemsLayout.Tree:
                default:
                    UpdateProblemsTree();
                    break;
            }

            foreach (var (type, filter) in _filtersByType)
            {
                int count = _diagnostics.Count(d => d.Type == type);
                filter.ProblemsCount = count;
            }

            if (_diagnostics.Count == 0)
                Name = "Problems".TTR();
            else
                Name = $"{"Problems".TTR()} ({_diagnostics.Count})";
        }

        private void UpdateProblemsList()
        {
            _problemsTree.Clear();

            var root = _problemsTree.CreateItem();

            for (int i = 0; i < _diagnostics.Count; i++)
            {
                var diagnostic = _diagnostics[i];

                if (!ShouldDisplayDiagnostic(diagnostic))
                    continue;

                var item = CreateProblemItem(diagnostic, includeFileInText: true);

                var problemItem = _problemsTree.CreateItem(root);
                problemItem.SetIcon(0, item.Icon);
                problemItem.SetText(0, item.Text);
                problemItem.SetTooltipText(0, item.TooltipText);
                problemItem.SetMetadata(0, i);

                var color = GetProblemItemColor(diagnostic);
                if (color.HasValue)
                    problemItem.SetCustomColor(0, color.Value);
            }
        }

        private void UpdateProblemsTree()
        {
            _problemsTree.Clear();

            var root = _problemsTree.CreateItem();

            var groupedDiagnostics = _diagnostics.Select((d, i) => (Diagnostic: d, Index: i))
                .Where(x => ShouldDisplayDiagnostic(x.Diagnostic))
                .GroupBy(x => x.Diagnostic.ProjectFile)
                .Select(g => (ProjectFile: g.Key, Diagnostics: g.GroupBy(x => x.Diagnostic.File)
                    .Select(x => (File: x.Key, Diagnostics: x.ToArray()))))
                .ToArray();

            if (groupedDiagnostics.Length == 0)
                return;

            foreach (var (projectFile, projectDiagnostics) in groupedDiagnostics)
            {
                TreeItem projectItem;

                if (groupedDiagnostics.Length == 1)
                {
                    // Don't create a project item if there's only one project.
                    projectItem = root;
                }
                else
                {
                    string projectFilePath = !string.IsNullOrEmpty(projectFile)
                        ? projectFile
                        : "Unknown project".TTR();
                    projectItem = _problemsTree.CreateItem(root);
                    projectItem.SetText(0, projectFilePath);
                    projectItem.SetSelectable(0, false);
                }

                foreach (var (file, fileDiagnostics) in projectDiagnostics)
                {
                    if (fileDiagnostics.Length == 0)
                        continue;

                    string? projectDir = Path.GetDirectoryName(projectFile);
                    string relativeFilePath = !string.IsNullOrEmpty(file) && !string.IsNullOrEmpty(projectDir)
                        ? Path.GetRelativePath(projectDir, file)
                        : "Unknown file".TTR();

                    string fileItemText = string.Format(CultureInfo.InvariantCulture, "{0} ({1} issues)".TTR(), relativeFilePath, fileDiagnostics.Length);

                    var fileItem = _problemsTree.CreateItem(projectItem);
                    fileItem.SetText(0, fileItemText);
                    fileItem.SetSelectable(0, false);

                    foreach (var (diagnostic, index) in fileDiagnostics)
                    {
                        var item = CreateProblemItem(diagnostic);

                        var problemItem = _problemsTree.CreateItem(fileItem);
                        problemItem.SetIcon(0, item.Icon);
                        problemItem.SetText(0, item.Text);
                        problemItem.SetTooltipText(0, item.TooltipText);
                        problemItem.SetMetadata(0, index);

                        var color = GetProblemItemColor(diagnostic);
                        if (color.HasValue)
                            problemItem.SetCustomColor(0, color.Value);
                    }
                }
            }
        }

        private class ProblemItem
        {
            public string? Text { get; set; }
            public string? TooltipText { get; set; }
            public Texture2D? Icon { get; set; }
        }

        private ProblemItem CreateProblemItem(BuildDiagnostic diagnostic, bool includeFileInText = false)
        {
            var text = new StringBuilder();
            var tooltip = new StringBuilder();

            ReadOnlySpan<char> shortMessage = diagnostic.Message.AsSpan();
            int lineBreakIdx = shortMessage.IndexOf('\n');
            if (lineBreakIdx != -1)
                shortMessage = shortMessage[..lineBreakIdx];
            text.Append(shortMessage);

            tooltip.Append(CultureInfo.InvariantCulture, $"Message: {diagnostic.Message}");

            if (!string.IsNullOrEmpty(diagnostic.Code))
                tooltip.Append(CultureInfo.InvariantCulture, $"\nCode: {diagnostic.Code}");

            string type = diagnostic.Type switch
            {
                BuildDiagnostic.DiagnosticType.Hidden => "hidden",
                BuildDiagnostic.DiagnosticType.Info => "info",
                BuildDiagnostic.DiagnosticType.Warning => "warning",
                BuildDiagnostic.DiagnosticType.Error => "error",
                _ => "unknown",
            };
            tooltip.Append(CultureInfo.InvariantCulture, $"\nType: {type}");

            if (!string.IsNullOrEmpty(diagnostic.File))
            {
                text.Append(' ');
                if (includeFileInText)
                {
                    text.Append(diagnostic.File);
                }

                text.Append(CultureInfo.InvariantCulture, $"({diagnostic.Line},{diagnostic.Column})");

                tooltip.Append(CultureInfo.InvariantCulture, $"\nFile: {diagnostic.File}");
                tooltip.Append(CultureInfo.InvariantCulture, $"\nLine: {diagnostic.Line}");
                tooltip.Append(CultureInfo.InvariantCulture, $"\nColumn: {diagnostic.Column}");
            }

            if (!string.IsNullOrEmpty(diagnostic.ProjectFile))
                tooltip.Append(CultureInfo.InvariantCulture, $"\nProject: {diagnostic.ProjectFile}");

            return new ProblemItem()
            {
                Text = text.ToString(),
                TooltipText = tooltip.ToString(),
                Icon = diagnostic.Type switch
                {
                    BuildDiagnostic.DiagnosticType.Warning => GetThemeIcon("Warning", "EditorIcons"),
                    BuildDiagnostic.DiagnosticType.Error => GetThemeIcon("Error", "EditorIcons"),
                    _ => null,
                },
            };
        }

        public override void _Ready()
        {
            var editorSettings = EditorInterface.Singleton.GetEditorSettings();
            _layout = editorSettings.GetSetting(GodotSharpEditor.Settings.ProblemsLayout).As<ProblemsLayout>();

            Name = "Problems".TTR();

            var vbLeft = new VBoxContainer
            {
                CustomMinimumSize = new Vector2(0, 180 * EditorScale),
                SizeFlagsVertical = SizeFlags.ExpandFill,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            };
            AddChild(vbLeft);

            // Problem Tree.
            _problemsTree = new Tree
            {
                SizeFlagsVertical = SizeFlags.ExpandFill,
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                AllowRmbSelect = true,
                HideRoot = true,
            };
            _problemsTree.ItemActivated += GoToSelectedProblem;
            _problemsTree.ItemMouseSelected += ShowProblemContextMenu;
            vbLeft.AddChild(_problemsTree);

            // Problem context menu.
            _problemsContextMenu = new PopupMenu();
            _problemsContextMenu.IdPressed += ProblemContextOptionPressed;
            _problemsTree.AddChild(_problemsContextMenu);

            // Search box.
            _searchBox = new LineEdit
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
                PlaceholderText = "Filter Problems".TTR(),
                ClearButtonEnabled = true,
            };
            _searchBox.TextChanged += SearchTextChanged;
            vbLeft.AddChild(_searchBox);

            var vbRight = new VBoxContainer();
            AddChild(vbRight);

            // Tools grid.
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
                ShortcutContext = this,
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
            _copyButton.Pressed += CopySelectedProblems;
            hbTools.AddChild(_copyButton);

            // A second hbox to make a 2x2 grid of buttons.
            var hbTools2 = new HBoxContainer
            {
                SizeFlagsHorizontal = SizeFlags.ShrinkCenter,
            };
            vbRight.AddChild(hbTools2);

            // Toggle List/Tree.
            _toggleLayoutButton = new Button
            {
                Flat = true,
                FocusMode = FocusModeEnum.None,
                TooltipText = GetToggleLayoutTooltipText(),
                ToggleMode = true,
                ButtonPressed = GetToggleLayoutPressedState(),
            };
            // Don't tint the icon even when in "pressed" state.
            _toggleLayoutButton.AddThemeColorOverride("icon_pressed_color", Colors.White);
            _toggleLayoutButton.Toggled += ToggleLayout;
            hbTools2.AddChild(_toggleLayoutButton);

            // Show Search.
            _showSearchButton = new Button
            {
                ThemeTypeVariation = "FlatButton",
                FocusMode = FocusModeEnum.None,
                ToggleMode = true,
                ButtonPressed = true,
                Shortcut = EditorDefShortcut("editor/open_search", "Focus Search/Filter Bar".TTR(), (Key)KeyModifierMask.MaskCmdOrCtrl | Key.F),
                ShortcutContext = this,
            };
            _showSearchButton.Toggled += ToggleSearchBoxVisibility;
            hbTools2.AddChild(_showSearchButton);

            // Diagnostic Type Filters.
            vbRight.AddChild(new HSeparator());

            var infoFilter = new BuildProblemsFilter(BuildDiagnostic.DiagnosticType.Info);
            infoFilter.ToggleButton.TooltipText = "Toggle visibility of info diagnostics.".TTR();
            infoFilter.ToggleButton.Toggled += ToggleFilter;
            vbRight.AddChild(infoFilter.ToggleButton);
            _filtersByType[BuildDiagnostic.DiagnosticType.Info] = infoFilter;

            var errorFilter = new BuildProblemsFilter(BuildDiagnostic.DiagnosticType.Error);
            errorFilter.ToggleButton.TooltipText = "Toggle visibility of errors.".TTR();
            errorFilter.ToggleButton.Toggled += ToggleFilter;
            vbRight.AddChild(errorFilter.ToggleButton);
            _filtersByType[BuildDiagnostic.DiagnosticType.Error] = errorFilter;

            var warningFilter = new BuildProblemsFilter(BuildDiagnostic.DiagnosticType.Warning);
            warningFilter.ToggleButton.TooltipText = "Toggle visibility of warnings.".TTR();
            warningFilter.ToggleButton.Toggled += ToggleFilter;
            vbRight.AddChild(warningFilter.ToggleButton);
            _filtersByType[BuildDiagnostic.DiagnosticType.Warning] = warningFilter;

            UpdateTheme();

            UpdateProblemsView();
        }

        public override void _Notification(int what)
        {
            base._Notification(what);

            switch ((long)what)
            {
                case EditorSettings.NotificationEditorSettingsChanged:
                    var editorSettings = EditorInterface.Singleton.GetEditorSettings();
                    _layout = editorSettings.GetSetting(GodotSharpEditor.Settings.ProblemsLayout).As<ProblemsLayout>();
                    _toggleLayoutButton.ButtonPressed = GetToggleLayoutPressedState();
                    UpdateProblemsView();
                    break;

                case NotificationThemeChanged:
                    UpdateTheme();
                    break;
            }
        }

        private void UpdateTheme()
        {
            // Nodes will be null until _Ready is called.
            if (_clearButton == null)
                return;

            foreach (var (type, filter) in _filtersByType)
            {
                filter.ToggleButton.Icon = type switch
                {
                    BuildDiagnostic.DiagnosticType.Info => GetThemeIcon("Popup", "EditorIcons"),
                    BuildDiagnostic.DiagnosticType.Warning => GetThemeIcon("StatusWarning", "EditorIcons"),
                    BuildDiagnostic.DiagnosticType.Error => GetThemeIcon("StatusError", "EditorIcons"),
                    _ => null,
                };
            }

            _clearButton.Icon = GetThemeIcon("Clear", "EditorIcons");
            _copyButton.Icon = GetThemeIcon("ActionCopy", "EditorIcons");
            _toggleLayoutButton.Icon = GetToggleLayoutIcon();
            _showSearchButton.Icon = GetThemeIcon("Search", "EditorIcons");
            _searchBox.RightIcon = GetThemeIcon("Search", "EditorIcons");
        }
    }
}
