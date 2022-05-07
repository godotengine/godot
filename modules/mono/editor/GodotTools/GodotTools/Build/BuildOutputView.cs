using Godot;
using System;
using Godot.Collections;
using GodotTools.Internals;
using JetBrains.Annotations;
using File = GodotTools.Utils.File;
using Path = System.IO.Path;

namespace GodotTools.Build
{
    public class BuildOutputView : VBoxContainer, ISerializationListener
    {
        [Serializable]
        private class BuildIssue : Reference // TODO Remove Reference once we have proper serialization
        {
            public bool Warning { get; set; }
            public string File { get; set; }
            public int Line { get; set; }
            public int Column { get; set; }
            public string Code { get; set; }
            public string Message { get; set; }
            public string ProjectFile { get; set; }
        }

        [Signal]
        public delegate void BuildStateChanged();

        public bool HasBuildExited { get; private set; } = false;

        public BuildResult? BuildResult { get; private set; } = null;

        public int ErrorCount { get; private set; } = 0;

        public int WarningCount { get; private set; } = 0;

        public bool ErrorsVisible { get; set; } = true;
        public bool WarningsVisible { get; set; } = true;

        public Texture BuildStateIcon
        {
            get
            {
                if (!HasBuildExited)
                    return GetIcon("Stop", "EditorIcons");

                if (BuildResult == Build.BuildResult.Error)
                    return GetIcon("Error", "EditorIcons");

                if (WarningCount > 1)
                    return GetIcon("Warning", "EditorIcons");

                return null;
            }
        }

        public bool LogVisible
        {
            set => _buildLog.Visible = value;
        }

        // TODO Use List once we have proper serialization.
        private readonly Array<BuildIssue> _issues = new Array<BuildIssue>();
        private ItemList _issuesList;
        private PopupMenu _issuesListContextMenu;
        private TextEdit _buildLog;
        private BuildInfo _buildInfo;

        private readonly object _pendingBuildLogTextLock = new object();
        [NotNull] private string _pendingBuildLogText = string.Empty;

        private void LoadIssuesFromFile(string csvFile)
        {
            using (var file = new Godot.File())
            {
                try
                {
                    Error openError = file.Open(csvFile, Godot.File.ModeFlags.Read);

                    if (openError != Error.Ok)
                        return;

                    while (!file.EofReached())
                    {
                        string[] csvColumns = file.GetCsvLine();

                        if (csvColumns.Length == 1 && string.IsNullOrEmpty(csvColumns[0]))
                            return;

                        if (csvColumns.Length != 7)
                        {
                            GD.PushError($"Expected 7 columns, got {csvColumns.Length}");
                            continue;
                        }

                        var issue = new BuildIssue
                        {
                            Warning = csvColumns[0] == "warning",
                            File = csvColumns[1],
                            Line = int.Parse(csvColumns[2]),
                            Column = int.Parse(csvColumns[3]),
                            Code = csvColumns[4],
                            Message = csvColumns[5],
                            ProjectFile = csvColumns[6]
                        };

                        if (issue.Warning)
                            WarningCount += 1;
                        else
                            ErrorCount += 1;

                        _issues.Add(issue);
                    }
                }
                finally
                {
                    file.Close(); // Disposing it is not enough. We need to call Close()
                }
            }
        }

        private void IssueActivated(int idx)
        {
            if (idx < 0 || idx >= _issuesList.GetItemCount())
                throw new IndexOutOfRangeException("Item list index out of range");

            // Get correct issue idx from issue list
            int issueIndex = (int)_issuesList.GetItemMetadata(idx);

            if (issueIndex < 0 || issueIndex >= _issues.Count)
                throw new IndexOutOfRangeException("Issue index out of range");

            BuildIssue issue = _issues[issueIndex];

            if (string.IsNullOrEmpty(issue.ProjectFile) && string.IsNullOrEmpty(issue.File))
                return;

            string projectDir = issue.ProjectFile.Length > 0 ? issue.ProjectFile.GetBaseDir() : _buildInfo.Solution.GetBaseDir();

            string file = Path.Combine(projectDir.SimplifyGodotPath(), issue.File.SimplifyGodotPath());

            if (!File.Exists(file))
                return;

            file = ProjectSettings.LocalizePath(file);

            if (file.StartsWith("res://"))
            {
                var script = (Script)ResourceLoader.Load(file, typeHint: Internal.CSharpLanguageType);

                if (script != null && Internal.ScriptEditorEdit(script, issue.Line, issue.Column))
                    Internal.EditorNodeShowScriptScreen();
            }
        }

        public void UpdateIssuesList()
        {
            _issuesList.Clear();

            using (var warningIcon = GetIcon("Warning", "EditorIcons"))
            using (var errorIcon = GetIcon("Error", "EditorIcons"))
            {
                for (int i = 0; i < _issues.Count; i++)
                {
                    BuildIssue issue = _issues[i];

                    if (!(issue.Warning ? WarningsVisible : ErrorsVisible))
                        continue;

                    string tooltip = string.Empty;
                    tooltip += $"Message: {issue.Message}";

                    if (!string.IsNullOrEmpty(issue.Code))
                        tooltip += $"\nCode: {issue.Code}";

                    tooltip += $"\nType: {(issue.Warning ? "warning" : "error")}";

                    string text = string.Empty;

                    if (!string.IsNullOrEmpty(issue.File))
                    {
                        text += $"{issue.File}({issue.Line},{issue.Column}): ";

                        tooltip += $"\nFile: {issue.File}";
                        tooltip += $"\nLine: {issue.Line}";
                        tooltip += $"\nColumn: {issue.Column}";
                    }

                    if (!string.IsNullOrEmpty(issue.ProjectFile))
                        tooltip += $"\nProject: {issue.ProjectFile}";

                    text += issue.Message;

                    int lineBreakIdx = text.IndexOf("\n", StringComparison.Ordinal);
                    string itemText = lineBreakIdx == -1 ? text : text.Substring(0, lineBreakIdx);
                    _issuesList.AddItem(itemText, issue.Warning ? warningIcon : errorIcon);

                    int index = _issuesList.GetItemCount() - 1;
                    _issuesList.SetItemTooltip(index, tooltip);
                    _issuesList.SetItemMetadata(index, i);
                }
            }
        }

        private void BuildLaunchFailed(BuildInfo buildInfo, string cause)
        {
            HasBuildExited = true;
            BuildResult = Build.BuildResult.Error;

            _issuesList.Clear();

            var issue = new BuildIssue {Message = cause, Warning = false};

            ErrorCount += 1;
            _issues.Add(issue);

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void BuildStarted(BuildInfo buildInfo)
        {
            _buildInfo = buildInfo;
            HasBuildExited = false;

            _issues.Clear();
            WarningCount = 0;
            ErrorCount = 0;
            _buildLog.Text = string.Empty;

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void BuildFinished(BuildResult result)
        {
            HasBuildExited = true;
            BuildResult = result;

            LoadIssuesFromFile(Path.Combine(_buildInfo.LogsDirPath, BuildManager.MsBuildIssuesFileName));

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void UpdateBuildLogText()
        {
            lock (_pendingBuildLogTextLock)
            {
                _buildLog.Text += _pendingBuildLogText;
                _pendingBuildLogText = string.Empty;
                ScrollToLastNonEmptyLogLine();
            }
        }

        private void StdOutputReceived(string text)
        {
            lock (_pendingBuildLogTextLock)
            {
                if (_pendingBuildLogText.Length == 0)
                    CallDeferred(nameof(UpdateBuildLogText));
                _pendingBuildLogText += text + "\n";
            }
        }

        private void StdErrorReceived(string text)
        {
            lock (_pendingBuildLogTextLock)
            {
                if (_pendingBuildLogText.Length == 0)
                    CallDeferred(nameof(UpdateBuildLogText));
                _pendingBuildLogText += text + "\n";
            }
        }

        private void ScrollToLastNonEmptyLogLine()
        {
            int line;
            for (line = _buildLog.GetLineCount(); line > 0; line--)
            {
                string lineText = _buildLog.GetLine(line);

                if (!string.IsNullOrEmpty(lineText) || !string.IsNullOrEmpty(lineText?.Trim()))
                    break;
            }

            _buildLog.CursorSetLine(line);
        }

        public void RestartBuild()
        {
            if (!HasBuildExited)
                throw new InvalidOperationException("Build already started");

            BuildManager.RestartBuild(this);
        }

        public void StopBuild()
        {
            if (!HasBuildExited)
                throw new InvalidOperationException("Build is not in progress");

            BuildManager.StopBuild(this);
        }

        private enum IssuesContextMenuOption
        {
            Copy
        }

        private void IssuesListContextOptionPressed(IssuesContextMenuOption id)
        {
            switch (id)
            {
                case IssuesContextMenuOption.Copy:
                {
                    // We don't allow multi-selection but just in case that changes later...
                    string text = null;

                    foreach (int issueIndex in _issuesList.GetSelectedItems())
                    {
                        if (text != null)
                            text += "\n";
                        text += _issuesList.GetItemText(issueIndex);
                    }

                    if (text != null)
                        OS.Clipboard = text;
                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Invalid issue context menu option");
            }
        }

        private void IssuesListRmbSelected(int index, Vector2 atPosition)
        {
            _ = index; // Unused

            _issuesListContextMenu.Clear();
            _issuesListContextMenu.SetSize(new Vector2(1, 1));

            if (_issuesList.IsAnythingSelected())
            {
                // Add menu entries for the selected item
                _issuesListContextMenu.AddIconItem(GetIcon("ActionCopy", "EditorIcons"),
                    label: "Copy Error".TTR(), (int)IssuesContextMenuOption.Copy);
            }

            if (_issuesListContextMenu.GetItemCount() > 0)
            {
                _issuesListContextMenu.SetPosition(_issuesList.RectGlobalPosition + atPosition);
                _issuesListContextMenu.Popup_();
            }
        }

        public override void _Ready()
        {
            base._Ready();

            SizeFlagsVertical = (int)SizeFlags.ExpandFill;

            var hsc = new HSplitContainer
            {
                SizeFlagsHorizontal = (int)SizeFlags.ExpandFill,
                SizeFlagsVertical = (int)SizeFlags.ExpandFill
            };
            AddChild(hsc);

            _issuesList = new ItemList
            {
                SizeFlagsVertical = (int)SizeFlags.ExpandFill,
                SizeFlagsHorizontal = (int)SizeFlags.ExpandFill // Avoid being squashed by the build log
            };
            _issuesList.Connect("item_activated", this, nameof(IssueActivated));
            _issuesList.AllowRmbSelect = true;
            _issuesList.Connect("item_rmb_selected", this, nameof(IssuesListRmbSelected));
            hsc.AddChild(_issuesList);

            _issuesListContextMenu = new PopupMenu();
            _issuesListContextMenu.Connect("id_pressed", this, nameof(IssuesListContextOptionPressed));
            _issuesList.AddChild(_issuesListContextMenu);

            _buildLog = new TextEdit
            {
                Readonly = true,
                SizeFlagsVertical = (int)SizeFlags.ExpandFill,
                SizeFlagsHorizontal = (int)SizeFlags.ExpandFill // Avoid being squashed by the issues list
            };
            hsc.AddChild(_buildLog);

            AddBuildEventListeners();
        }

        private void AddBuildEventListeners()
        {
            BuildManager.BuildLaunchFailed += BuildLaunchFailed;
            BuildManager.BuildStarted += BuildStarted;
            BuildManager.BuildFinished += BuildFinished;
            // StdOutput/Error can be received from different threads, so we need to use CallDeferred
            BuildManager.StdOutputReceived += StdOutputReceived;
            BuildManager.StdErrorReceived += StdErrorReceived;
        }

        public void OnBeforeSerialize()
        {
            // In case it didn't update yet. We don't want to have to serialize any pending output.
            UpdateBuildLogText();
        }

        public void OnAfterDeserialize()
        {
            AddBuildEventListeners(); // Re-add them
        }
    }
}
