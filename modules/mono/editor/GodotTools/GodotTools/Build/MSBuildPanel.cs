using System;
using System.IO;
using Godot;
using GodotTools.Internals;
using static GodotTools.Internals.Globals;
using File = GodotTools.Utils.File;

namespace GodotTools.Build
{
    public partial class MSBuildPanel : VBoxContainer
    {
        public BuildOutputView BuildOutputView { get; private set; }

        private MenuButton _buildMenuBtn;
        private Button _errorsBtn;
        private Button _warningsBtn;
        private Button _viewLogBtn;
        private Button _openLogsFolderBtn;

        private void WarningsToggled(bool pressed)
        {
            BuildOutputView.WarningsVisible = pressed;
            BuildOutputView.UpdateIssuesList();
        }

        private void ErrorsToggled(bool pressed)
        {
            BuildOutputView.ErrorsVisible = pressed;
            BuildOutputView.UpdateIssuesList();
        }

        public void BuildProject()
        {
            if (!File.Exists(GodotSharpDirs.ProjectCsProjPath))
                return; // No project to build.

            if (!BuildManager.BuildProjectBlocking("Debug"))
                return; // Build failed.

            // Notify running game for hot-reload.
            Internal.EditorDebuggerNodeReloadScripts();

            // Hot-reload in the editor.
            GodotSharpEditor.Instance.GetNode<HotReloadAssemblyWatcher>("HotReloadAssemblyWatcher").RestartTimer();

            if (Internal.IsAssembliesReloadingNeeded())
                Internal.ReloadAssemblies(softReload: false);
        }

        private void RebuildProject()
        {
            if (!File.Exists(GodotSharpDirs.ProjectCsProjPath))
                return; // No project to build.

            if (!BuildManager.BuildProjectBlocking("Debug", rebuild: true))
                return; // Build failed.

            // Notify running game for hot-reload.
            Internal.EditorDebuggerNodeReloadScripts();

            // Hot-reload in the editor.
            GodotSharpEditor.Instance.GetNode<HotReloadAssemblyWatcher>("HotReloadAssemblyWatcher").RestartTimer();

            if (Internal.IsAssembliesReloadingNeeded())
                Internal.ReloadAssemblies(softReload: false);
        }

        private void CleanProject()
        {
            if (!File.Exists(GodotSharpDirs.ProjectCsProjPath))
                return; // No project to build.

            _ = BuildManager.CleanProjectBlocking("Debug");
        }

        private void ViewLogToggled(bool pressed) => BuildOutputView.LogVisible = pressed;

        private void OpenLogsFolderPressed() => OS.ShellOpen(
            $"file://{GodotSharpDirs.LogsDirPathFor("Debug")}"
        );

        private void BuildMenuOptionPressed(long id)
        {
            switch ((BuildMenuOptions)id)
            {
                case BuildMenuOptions.BuildProject:
                    BuildProject();
                    break;
                case BuildMenuOptions.RebuildProject:
                    RebuildProject();
                    break;
                case BuildMenuOptions.CleanProject:
                    CleanProject();
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Invalid build menu option");
            }
        }

        private enum BuildMenuOptions
        {
            BuildProject,
            RebuildProject,
            CleanProject
        }

        public override void _Ready()
        {
            base._Ready();

            CustomMinimumSize = new Vector2(0, 228 * EditorScale);
            SizeFlagsVertical = SizeFlags.ExpandFill;

            var toolBarHBox = new HBoxContainer { SizeFlagsHorizontal = SizeFlags.ExpandFill };
            AddChild(toolBarHBox);

            _buildMenuBtn = new MenuButton { Text = "Build", Icon = GetThemeIcon("BuildCSharp", "EditorIcons") };
            toolBarHBox.AddChild(_buildMenuBtn);

            var buildMenu = _buildMenuBtn.GetPopup();
            buildMenu.AddItem("Build Project".TTR(), (int)BuildMenuOptions.BuildProject);
            buildMenu.AddItem("Rebuild Project".TTR(), (int)BuildMenuOptions.RebuildProject);
            buildMenu.AddItem("Clean Project".TTR(), (int)BuildMenuOptions.CleanProject);
            buildMenu.IdPressed += BuildMenuOptionPressed;

            _errorsBtn = new Button
            {
                TooltipText = "Show Errors".TTR(),
                Icon = GetThemeIcon("StatusError", "EditorIcons"),
                ExpandIcon = false,
                ToggleMode = true,
                ButtonPressed = true,
                FocusMode = FocusModeEnum.None
            };
            _errorsBtn.Toggled += ErrorsToggled;
            toolBarHBox.AddChild(_errorsBtn);

            _warningsBtn = new Button
            {
                TooltipText = "Show Warnings".TTR(),
                Icon = GetThemeIcon("NodeWarning", "EditorIcons"),
                ExpandIcon = false,
                ToggleMode = true,
                ButtonPressed = true,
                FocusMode = FocusModeEnum.None
            };
            _warningsBtn.Toggled += WarningsToggled;
            toolBarHBox.AddChild(_warningsBtn);

            _viewLogBtn = new Button
            {
                Text = "Show Output".TTR(),
                ToggleMode = true,
                ButtonPressed = true,
                FocusMode = FocusModeEnum.None
            };
            _viewLogBtn.Toggled += ViewLogToggled;
            toolBarHBox.AddChild(_viewLogBtn);

            // Horizontal spacer, push everything to the right.
            toolBarHBox.AddChild(new Control
            {
                SizeFlagsHorizontal = SizeFlags.ExpandFill,
            });

            _openLogsFolderBtn = new Button
            {
                Text = "Show Logs in File Manager".TTR(),
                Icon = GetThemeIcon("Filesystem", "EditorIcons"),
                ExpandIcon = false,
                FocusMode = FocusModeEnum.None,
            };
            _openLogsFolderBtn.Pressed += OpenLogsFolderPressed;
            toolBarHBox.AddChild(_openLogsFolderBtn);

            BuildOutputView = new BuildOutputView();
            AddChild(BuildOutputView);
        }

        public override void _Notification(int what)
        {
            base._Notification(what);

            if (what == NotificationThemeChanged)
            {
                if (_buildMenuBtn != null)
                    _buildMenuBtn.Icon = GetThemeIcon("BuildCSharp", "EditorIcons");
                if (_errorsBtn != null)
                    _errorsBtn.Icon = GetThemeIcon("StatusError", "EditorIcons");
                if (_warningsBtn != null)
                    _warningsBtn.Icon = GetThemeIcon("NodeWarning", "EditorIcons");
            }
        }
    }
}
