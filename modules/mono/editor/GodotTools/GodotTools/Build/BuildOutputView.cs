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

        private readonly Array<BuildIssue> issues = new Array<BuildIssue>(); // TODO Use List once we have proper serialization
        private ItemList issuesList;
        private TextEdit buildLog;
        private PopupMenu issuesListContextMenu;

        private readonly object pendingBuildLogTextLock = new object();
        [NotNull] private string pendingBuildLogText = string.Empty;

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

        private BuildInfo BuildInfo { get; set; }

        public bool LogVisible
        {
            set => buildLog.Visible = value;
        }

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

                        issues.Add(issue);
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
            if (idx < 0 || idx >= issuesList.GetItemCount())
                throw new IndexOutOfRangeException("Item list index out of range");

            // Get correct issue idx from issue list
            int issueIndex = (int)issuesList.GetItemMetadata(idx);

            if (issueIndex < 0 || issueIndex >= issues.Count)
                throw new IndexOutOfRangeException("Issue index out of range");

            BuildIssue issue = issues[issueIndex];

            if (string.IsNullOrEmpty(issue.ProjectFile) && string.IsNullOrEmpty(issue.File))
                return;

            string projectDir = issue.ProjectFile.Length > 0 ? issue.ProjectFile.GetBaseDir() : BuildInfo.Solution.GetBaseDir();

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
            issuesList.Clear();

            using (var warningIcon = GetIcon("Warning", "EditorIcons"))
            using (var errorIcon = GetIcon("Error", "EditorIcons"))
            {
                for (int i = 0; i < issues.Count; i++)
                {
                    BuildIssue issue = issues[i];

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
                    issuesList.AddItem(itemText, issue.Warning ? warningIcon : errorIcon);

                    int index = issuesList.GetItemCount() - 1;
                    issuesList.SetItemTooltip(index, tooltip);
                    issuesList.SetItemMetadata(index, i);
                }
            }
        }

        private void BuildLaunchFailed(BuildInfo buildInfo, string cause)
        {
            HasBuildExited = true;
            BuildResult = Build.BuildResult.Error;

            issuesList.Clear();

            var issue = new BuildIssue {Message = cause, Warning = false};

            ErrorCount += 1;
            issues.Add(issue);

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void BuildStarted(BuildInfo buildInfo)
        {
            BuildInfo = buildInfo;
            HasBuildExited = false;

            issues.Clear();
            WarningCount = 0;
            ErrorCount = 0;
            buildLog.Text = string.Empty;

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void BuildFinished(BuildResult result)
        {
            HasBuildExited = true;
            BuildResult = result;

            LoadIssuesFromFile(Path.Combine(BuildInfo.LogsDirPath, BuildManager.MsBuildIssuesFileName));

            UpdateIssuesList();

            EmitSignal(nameof(BuildStateChanged));
        }

        private void UpdateBuildLogText()
        {
            lock (pendingBuildLogTextLock)
            {
                buildLog.Text += pendingBuildLogText;
                pendingBuildLogText = string.Empty;
                ScrollToLastNonEmptyLogLine();
            }
        }

        private void StdOutputReceived(string text)
        {
            lock (pendingBuildLogTextLock)
            {
                if (pendingBuildLogText.Length == 0)
                    CallDeferred(nameof(UpdateBuildLogText));
                pendingBuildLogText += text + "\n";
            }
        }

        private void StdErrorReceived(string text)
        {
            lock (pendingBuildLogTextLock)
            {
                if (pendingBuildLogText.Length == 0)
                    CallDeferred(nameof(UpdateBuildLogText));
                pendingBuildLogText += text + "\n";
            }
        }

        private void ScrollToLastNonEmptyLogLine()
        {
            int line;
            for (line = buildLog.GetLineCount(); line > 0; line--)
            {
                string lineText = buildLog.GetLine(line);

                if (!string.IsNullOrEmpty(lineText) || !string.IsNullOrEmpty(lineText?.Trim()))
                    break;
            }

            buildLog.CursorSetLine(line);
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

                    foreach (int issueIndex in issuesList.GetSelectedItems())
                    {
                        if (text != null)
                            text += "\n";
                        text += issuesList.GetItemText(issueIndex);
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

            issuesListContextMenu.Clear();
            issuesListContextMenu.SetSize(new Vector2(1, 1));

            if (issuesList.IsAnythingSelected())
            {
                // Add menu entries for the selected item
                issuesListContextMenu.AddIconItem(GetIcon("ActionCopy", "EditorIcons"),
                    label: "Copy Error".TTR(), (int)IssuesContextMenuOption.Copy);
            }

            if (issuesListContextMenu.GetItemCount() > 0)
            {
                issuesListContextMenu.SetPosition(issuesList.RectGlobalPosition + atPosition);
                issuesListContextMenu.Popup_();
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

            issuesList = new ItemList
            {
                SizeFlagsVertical = (int)SizeFlags.ExpandFill,
                SizeFlagsHorizontal = (int)SizeFlags.ExpandFill // Avoid being squashed by the build log
            };
            issuesList.Connect("item_activated", this, nameof(IssueActivated));
            issuesList.AllowRmbSelect = true;
            issuesList.Connect("item_rmb_selected", this, nameof(IssuesListRmbSelected));
            hsc.AddChild(issuesList);

            issuesListContextMenu = new PopupMenu();
            issuesListContextMenu.Connect("id_pressed", this, nameof(IssuesListContextOptionPressed));
            issuesList.AddChild(issuesListContextMenu);

            buildLog = new TextEdit
            {
                Readonly = true,
                SizeFlagsVertical = (int)SizeFlags.ExpandFill,
                SizeFlagsHorizontal = (int)SizeFlags.ExpandFill // Avoid being squashed by the issues list
            };
            hsc.AddChild(buildLog);

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
