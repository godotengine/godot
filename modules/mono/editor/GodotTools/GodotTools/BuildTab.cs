using Godot;
using System;
using Godot.Collections;
using GodotTools.Internals;
using File = GodotTools.Utils.File;
using Path = System.IO.Path;

namespace GodotTools
{
    public class BuildTab : VBoxContainer
    {
        public enum BuildResults
        {
            Error,
            Success
        }

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

        public bool BuildExited { get; private set; } = false;

        public BuildResults? BuildResult { get; private set; } = null;

        public int ErrorCount { get; private set; } = 0;

        public int WarningCount { get; private set; } = 0;

        public bool ErrorsVisible { get; set; } = true;
        public bool WarningsVisible { get; set; } = true;

        public Texture2D IconTexture
        {
            get
            {
                if (!BuildExited)
                    return GetThemeIcon("Stop", "EditorIcons");

                if (BuildResult == BuildResults.Error)
                    return GetThemeIcon("StatusError", "EditorIcons");

                return GetThemeIcon("StatusSuccess", "EditorIcons");
            }
        }

        public BuildInfo BuildInfo { get; private set; }

        private void _LoadIssuesFromFile(string csvFile)
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

                        if (csvColumns.Length == 1 && csvColumns[0].Empty())
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

        private void _IssueActivated(int idx)
        {
            if (idx < 0 || idx >= issuesList.GetItemCount())
                throw new IndexOutOfRangeException("Item list index out of range");

            // Get correct issue idx from issue list
            int issueIndex = (int)issuesList.GetItemMetadata(idx);

            if (idx < 0 || idx >= issues.Count)
                throw new IndexOutOfRangeException("Issue index out of range");

            BuildIssue issue = issues[issueIndex];

            if (issue.ProjectFile.Empty() && issue.File.Empty())
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

            using (var warningIcon = GetThemeIcon("Warning", "EditorIcons"))
            using (var errorIcon = GetThemeIcon("Error", "EditorIcons"))
            {
                for (int i = 0; i < issues.Count; i++)
                {
                    BuildIssue issue = issues[i];

                    if (!(issue.Warning ? WarningsVisible : ErrorsVisible))
                        continue;

                    string tooltip = string.Empty;
                    tooltip += $"Message: {issue.Message}";

                    if (!issue.Code.Empty())
                        tooltip += $"\nCode: {issue.Code}";

                    tooltip += $"\nType: {(issue.Warning ? "warning" : "error")}";

                    string text = string.Empty;

                    if (!issue.File.Empty())
                    {
                        text += $"{issue.File}({issue.Line},{issue.Column}): ";

                        tooltip += $"\nFile: {issue.File}";
                        tooltip += $"\nLine: {issue.Line}";
                        tooltip += $"\nColumn: {issue.Column}";
                    }

                    if (!issue.ProjectFile.Empty())
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

        public void OnBuildStart()
        {
            BuildExited = false;

            issues.Clear();
            WarningCount = 0;
            ErrorCount = 0;
            UpdateIssuesList();

            GodotSharpEditor.Instance.BottomPanel.RaiseBuildTab(this);
        }

        public void OnBuildExit(BuildResults result)
        {
            BuildExited = true;
            BuildResult = result;

            _LoadIssuesFromFile(Path.Combine(BuildInfo.LogsDirPath, BuildManager.MsBuildIssuesFileName));
            UpdateIssuesList();

            GodotSharpEditor.Instance.BottomPanel.RaiseBuildTab(this);
        }

        public void OnBuildExecFailed(string cause)
        {
            BuildExited = true;
            BuildResult = BuildResults.Error;

            issuesList.Clear();

            var issue = new BuildIssue { Message = cause, Warning = false };

            ErrorCount += 1;
            issues.Add(issue);

            UpdateIssuesList();

            GodotSharpEditor.Instance.BottomPanel.RaiseBuildTab(this);
        }

        public void RestartBuild()
        {
            if (!BuildExited)
                throw new InvalidOperationException("Build already started");

            BuildManager.RestartBuild(this);
        }

        public void StopBuild()
        {
            if (!BuildExited)
                throw new InvalidOperationException("Build is not in progress");

            BuildManager.StopBuild(this);
        }

        public override void _Ready()
        {
            base._Ready();

            issuesList = new ItemList { SizeFlagsVertical = (int)SizeFlags.ExpandFill };
            issuesList.ItemActivated += _IssueActivated;
            AddChild(issuesList);
        }

        private BuildTab()
        {
        }

        public BuildTab(BuildInfo buildInfo)
        {
            BuildInfo = buildInfo;
        }
    }
}
