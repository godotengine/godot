using Godot;
using System;
using System.IO;
using Godot.Collections;
using GodotTools.Internals;
using File = GodotTools.Utils.File;
using Path = System.IO.Path;

namespace GodotTools
{
    public class MonoBottomPanel : VBoxContainer
    {
        private EditorInterface editorInterface;

        private TabContainer panelTabs;

        private VBoxContainer panelBuildsTab;

        private ItemList buildTabsList;
        private TabContainer buildTabs;

        private ToolButton warningsBtn;
        private ToolButton errorsBtn;
        private Button viewLogBtn;

        private void _UpdateBuildTabsList()
        {
            buildTabsList.Clear();

            int currentTab = buildTabs.CurrentTab;

            bool noCurrentTab = currentTab < 0 || currentTab >= buildTabs.GetTabCount();

            for (int i = 0; i < buildTabs.GetChildCount(); i++)
            {
                var tab = (MonoBuildTab) buildTabs.GetChild(i);

                if (tab == null)
                    continue;

                string itemName = Path.GetFileNameWithoutExtension(tab.BuildInfo.Solution);
                itemName += " [" + tab.BuildInfo.Configuration + "]";

                buildTabsList.AddItem(itemName, tab.IconTexture);

                string itemTooltip = "Solution: " + tab.BuildInfo.Solution;
                itemTooltip += "\nConfiguration: " + tab.BuildInfo.Configuration;
                itemTooltip += "\nStatus: ";

                if (tab.BuildExited)
                    itemTooltip += tab.BuildResult == MonoBuildTab.BuildResults.Success ? "Succeeded" : "Errored";
                else
                    itemTooltip += "Running";

                if (!tab.BuildExited || tab.BuildResult == MonoBuildTab.BuildResults.Error)
                    itemTooltip += $"\nErrors: {tab.ErrorCount}";

                itemTooltip += $"\nWarnings: {tab.WarningCount}";

                buildTabsList.SetItemTooltip(i, itemTooltip);

                if (noCurrentTab || currentTab == i)
                {
                    buildTabsList.Select(i);
                    _BuildTabsItemSelected(i);
                }
            }
        }

        public MonoBuildTab GetBuildTabFor(MonoBuildInfo buildInfo)
        {
            foreach (var buildTab in new Array<MonoBuildTab>(buildTabs.GetChildren()))
            {
                if (buildTab.BuildInfo.Equals(buildInfo))
                    return buildTab;
            }

            var newBuildTab = new MonoBuildTab(buildInfo);
            AddBuildTab(newBuildTab);

            return newBuildTab;
        }

        private void _BuildTabsItemSelected(int idx)
        {
            if (idx < 0 || idx >= buildTabs.GetTabCount())
                throw new IndexOutOfRangeException();

            buildTabs.CurrentTab = idx;
            if (!buildTabs.Visible)
                buildTabs.Visible = true;

            warningsBtn.Visible = true;
            errorsBtn.Visible = true;
            viewLogBtn.Visible = true;
        }

        private void _BuildTabsNothingSelected()
        {
            if (buildTabs.GetTabCount() != 0)
            {
                // just in case
                buildTabs.Visible = false;

                // This callback is called when clicking on the empty space of the list.
                // ItemList won't deselect the items automatically, so we must do it ourselves.
                buildTabsList.UnselectAll();
            }

            warningsBtn.Visible = false;
            errorsBtn.Visible = false;
            viewLogBtn.Visible = false;
        }

        private void _WarningsToggled(bool pressed)
        {
            int currentTab = buildTabs.CurrentTab;

            if (currentTab < 0 || currentTab >= buildTabs.GetTabCount())
                throw new InvalidOperationException("No tab selected");

            var buildTab = (MonoBuildTab) buildTabs.GetChild(currentTab);
            buildTab.WarningsVisible = pressed;
            buildTab.UpdateIssuesList();
        }

        private void _ErrorsToggled(bool pressed)
        {
            int currentTab = buildTabs.CurrentTab;

            if (currentTab < 0 || currentTab >= buildTabs.GetTabCount())
                throw new InvalidOperationException("No tab selected");

            var buildTab = (MonoBuildTab) buildTabs.GetChild(currentTab);
            buildTab.ErrorsVisible = pressed;
            buildTab.UpdateIssuesList();
        }

        public void BuildProjectPressed()
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
                return; // No solution to build

            string editorScriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, "scripts_metadata.editor");
            string playerScriptsMetadataPath = Path.Combine(GodotSharpDirs.ResMetadataDir, "scripts_metadata.editor_player");

            CSharpProject.GenerateScriptsMetadata(GodotSharpDirs.ProjectCsProjPath, editorScriptsMetadataPath);

            if (File.Exists(editorScriptsMetadataPath))
            {
                try
                {
                    File.Copy(editorScriptsMetadataPath, playerScriptsMetadataPath);
                }
                catch (IOException e)
                {
                    GD.PushError($"Failed to copy scripts metadata file. Exception message: {e.Message}");
                    return;
                }
            }

            var godotDefines = new[]
            {
                OS.GetName(),
                Internal.GodotIs32Bits() ? "32" : "64"
            };

            bool buildSuccess = GodotSharpBuilds.BuildProjectBlocking("Tools", godotDefines);

            if (!buildSuccess)
                return;

            // Notify running game for hot-reload
            Internal.ScriptEditorDebuggerReloadScripts();

            // Hot-reload in the editor
            GodotSharpEditor.Instance.GetNode<HotReloadAssemblyWatcher>("HotReloadAssemblyWatcher").RestartTimer();

            if (Internal.IsAssembliesReloadingNeeded())
                Internal.ReloadAssemblies(softReload: false);
        }

        private void _ViewLogPressed()
        {
            if (!buildTabsList.IsAnythingSelected())
                return;

            var selectedItems = buildTabsList.GetSelectedItems();

            if (selectedItems.Length != 1)
                throw new InvalidOperationException($"Expected 1 selected item, got {selectedItems.Length}");

            int selectedItem = selectedItems[0];

            var buildTab = (MonoBuildTab) buildTabs.GetTabControl(selectedItem);

            OS.ShellOpen(Path.Combine(buildTab.BuildInfo.LogsDirPath, GodotSharpBuilds.MsBuildLogFileName));
        }

        public override void _Notification(int what)
        {
            base._Notification(what);

            if (what == EditorSettings.NotificationEditorSettingsChanged)
            {
                var editorBaseControl = editorInterface.GetBaseControl();
                panelTabs.AddStyleboxOverride("panel", editorBaseControl.GetStylebox("DebuggerPanel", "EditorStyles"));
                panelTabs.AddStyleboxOverride("tab_fg", editorBaseControl.GetStylebox("DebuggerTabFG", "EditorStyles"));
                panelTabs.AddStyleboxOverride("tab_bg", editorBaseControl.GetStylebox("DebuggerTabBG", "EditorStyles"));
            }
        }

        public void AddBuildTab(MonoBuildTab buildTab)
        {
            buildTabs.AddChild(buildTab);
            RaiseBuildTab(buildTab);
        }

        public void RaiseBuildTab(MonoBuildTab buildTab)
        {
            if (buildTab.GetParent() != buildTabs)
                throw new InvalidOperationException("Build tab is not in the tabs list");

            buildTabs.MoveChild(buildTab, 0);
            _UpdateBuildTabsList();
        }

        public void ShowBuildTab()
        {
            for (int i = 0; i < panelTabs.GetTabCount(); i++)
            {
                if (panelTabs.GetTabControl(i) == panelBuildsTab)
                {
                    panelTabs.CurrentTab = i;
                    GodotSharpEditor.Instance.MakeBottomPanelItemVisible(this);
                    return;
                }
            }

            GD.PushError("Builds tab not found");
        }

        public override void _Ready()
        {
            base._Ready();

            editorInterface = GodotSharpEditor.Instance.GetEditorInterface();

            var editorBaseControl = editorInterface.GetBaseControl();

            SizeFlagsVertical = (int) SizeFlags.ExpandFill;
            SetAnchorsAndMarginsPreset(LayoutPreset.Wide);

            panelTabs = new TabContainer
            {
                TabAlign = TabContainer.TabAlignEnum.Left,
                RectMinSize = new Vector2(0, 228) * Internal.EditorScale,
                SizeFlagsVertical = (int) SizeFlags.ExpandFill
            };
            panelTabs.AddStyleboxOverride("panel", editorBaseControl.GetStylebox("DebuggerPanel", "EditorStyles"));
            panelTabs.AddStyleboxOverride("tab_fg", editorBaseControl.GetStylebox("DebuggerTabFG", "EditorStyles"));
            panelTabs.AddStyleboxOverride("tab_bg", editorBaseControl.GetStylebox("DebuggerTabBG", "EditorStyles"));
            AddChild(panelTabs);

            {
                // Builds tab
                panelBuildsTab = new VBoxContainer
                {
                    Name = "Builds", // TTR
                    SizeFlagsHorizontal = (int) SizeFlags.ExpandFill
                };
                panelTabs.AddChild(panelBuildsTab);

                var toolBarHBox = new HBoxContainer {SizeFlagsHorizontal = (int) SizeFlags.ExpandFill};
                panelBuildsTab.AddChild(toolBarHBox);

                var buildProjectBtn = new Button
                {
                    Text = "Build Project", // TTR
                    FocusMode = FocusModeEnum.None
                };
                buildProjectBtn.Connect("pressed", this, nameof(BuildProjectPressed));
                toolBarHBox.AddChild(buildProjectBtn);

                toolBarHBox.AddSpacer(begin: false);

                warningsBtn = new ToolButton
                {
                    Text = "Warnings", // TTR
                    ToggleMode = true,
                    Pressed = true,
                    Visible = false,
                    FocusMode = FocusModeEnum.None
                };
                warningsBtn.Connect("toggled", this, nameof(_WarningsToggled));
                toolBarHBox.AddChild(warningsBtn);

                errorsBtn = new ToolButton
                {
                    Text = "Errors", // TTR
                    ToggleMode = true,
                    Pressed = true,
                    Visible = false,
                    FocusMode = FocusModeEnum.None
                };
                errorsBtn.Connect("toggled", this, nameof(_ErrorsToggled));
                toolBarHBox.AddChild(errorsBtn);

                toolBarHBox.AddSpacer(begin: false);

                viewLogBtn = new Button
                {
                    Text = "View log", // TTR
                    FocusMode = FocusModeEnum.None,
                    Visible = false
                };
                viewLogBtn.Connect("pressed", this, nameof(_ViewLogPressed));
                toolBarHBox.AddChild(viewLogBtn);

                var hsc = new HSplitContainer
                {
                    SizeFlagsHorizontal = (int) SizeFlags.ExpandFill,
                    SizeFlagsVertical = (int) SizeFlags.ExpandFill
                };
                panelBuildsTab.AddChild(hsc);

                buildTabsList = new ItemList {SizeFlagsHorizontal = (int) SizeFlags.ExpandFill};
                buildTabsList.Connect("item_selected", this, nameof(_BuildTabsItemSelected));
                buildTabsList.Connect("nothing_selected", this, nameof(_BuildTabsNothingSelected));
                hsc.AddChild(buildTabsList);

                buildTabs = new TabContainer
                {
                    TabAlign = TabContainer.TabAlignEnum.Left,
                    SizeFlagsHorizontal = (int) SizeFlags.ExpandFill,
                    TabsVisible = false
                };
                hsc.AddChild(buildTabs);
            }
        }
    }
}
