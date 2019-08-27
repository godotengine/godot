using Godot;
using GodotTools.Utils;
using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using GodotTools.Ides;
using GodotTools.Internals;
using GodotTools.ProjectEditor;
using static GodotTools.Internals.Globals;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;

namespace GodotTools
{
    [SuppressMessage("ReSharper", "ClassNeverInstantiated.Global")]
    public class GodotSharpEditor : EditorPlugin, ISerializationListener
    {
        private EditorSettings editorSettings;

        private PopupMenu menuPopup;

        private AcceptDialog errorDialog;
        private AcceptDialog aboutDialog;
        private CheckBox aboutDialogCheckBox;

        private ToolButton bottomPanelBtn;

        public GodotIdeManager GodotIdeManager { get; private set; }

        private WeakRef exportPluginWeak; // TODO Use WeakReference once we have proper serialization

        public BottomPanel BottomPanel { get; private set; }

        private bool CreateProjectSolution()
        {
            using (var pr = new EditorProgress("create_csharp_solution", "Generating solution...".TTR(), 2))
            {
                pr.Step("Generating C# project...".TTR());

                string resourceDir = ProjectSettings.GlobalizePath("res://");

                string path = resourceDir;
                string name = (string) ProjectSettings.GetSetting("application/config/name");
                if (name.Empty())
                    name = "UnnamedProject";

                string guid = CsProjOperations.GenerateGameProject(path, name);

                if (guid.Length > 0)
                {
                    var solution = new DotNetSolution(name)
                    {
                        DirectoryPath = path
                    };

                    var projectInfo = new DotNetSolution.ProjectInfo
                    {
                        Guid = guid,
                        PathRelativeToSolution = name + ".csproj",
                        Configs = new List<string> {"Debug", "Release", "Tools"}
                    };

                    solution.AddNewProject(name, projectInfo);

                    try
                    {
                        solution.Save();
                    }
                    catch (IOException e)
                    {
                        ShowErrorDialog("Failed to save solution. Exception message: ".TTR() + e.Message);
                        return false;
                    }

                    // Make sure to update the API assemblies if they happen to be missing. Just in
                    // case the user decided to delete them at some point after they were loaded.
                    Internal.UpdateApiAssembliesFromPrebuilt();

                    pr.Step("Done".TTR());

                    // Here, after all calls to progress_task_step
                    CallDeferred(nameof(_RemoveCreateSlnMenuOption));
                }
                else
                {
                    ShowErrorDialog("Failed to create C# project.".TTR());
                }

                return true;
            }
        }

        private void _RemoveCreateSlnMenuOption()
        {
            menuPopup.RemoveItem(menuPopup.GetItemIndex((int) MenuOptions.CreateSln));
            bottomPanelBtn.Show();
        }

        private void _ShowAboutDialog()
        {
            bool showOnStart = (bool) editorSettings.GetSetting("mono/editor/show_info_on_start");
            aboutDialogCheckBox.Pressed = showOnStart;
            aboutDialog.PopupCenteredMinsize();
        }

        private void _ToggleAboutDialogOnStart(bool enabled)
        {
            bool showOnStart = (bool) editorSettings.GetSetting("mono/editor/show_info_on_start");
            if (showOnStart != enabled)
                editorSettings.SetSetting("mono/editor/show_info_on_start", enabled);
        }

        private void _MenuOptionPressed(MenuOptions id)
        {
            switch (id)
            {
                case MenuOptions.CreateSln:
                    CreateProjectSolution();
                    break;
                case MenuOptions.AboutCSharp:
                    _ShowAboutDialog();
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Invalid menu option");
            }
        }

        private void _BuildSolutionPressed()
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
            {
                if (!CreateProjectSolution())
                    return; // Failed to create solution
            }

            Instance.BottomPanel.BuildProjectPressed();
        }

        public override void _Notification(int what)
        {
            base._Notification(what);

            if (what == NotificationReady)
            {
                bool showInfoDialog = (bool) editorSettings.GetSetting("mono/editor/show_info_on_start");
                if (showInfoDialog)
                {
                    aboutDialog.PopupExclusive = true;
                    _ShowAboutDialog();
                    // Once shown a first time, it can be seen again via the Mono menu - it doesn't have to be exclusive from that time on.
                    aboutDialog.PopupExclusive = false;
                }
            }
        }

        private enum MenuOptions
        {
            CreateSln,
            AboutCSharp,
        }

        public void ShowErrorDialog(string message, string title = "Error")
        {
            errorDialog.WindowTitle = title;
            errorDialog.DialogText = message;
            errorDialog.PopupCenteredMinsize();
        }

        private static string _vsCodePath = string.Empty;

        private static readonly string[] VsCodeNames =
        {
            "code", "code-oss", "vscode", "vscode-oss", "visual-studio-code", "visual-studio-code-oss"
        };

        public Error OpenInExternalEditor(Script script, int line, int col)
        {
            var editor = (ExternalEditorId) editorSettings.GetSetting("mono/editor/external_editor");

            switch (editor)
            {
                case ExternalEditorId.None:
                    // Tells the caller to fallback to the global external editor settings or the built-in editor
                    return Error.Unavailable;
                case ExternalEditorId.VisualStudio:
                    throw new NotSupportedException();
                case ExternalEditorId.VisualStudioForMac:
                    goto case ExternalEditorId.MonoDevelop;
                case ExternalEditorId.MonoDevelop:
                {
                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);

                    if (line >= 0)
                        GodotIdeManager.SendOpenFile(scriptPath, line + 1, col);
                    else
                        GodotIdeManager.SendOpenFile(scriptPath);

                    break;
                }

                case ExternalEditorId.VsCode:
                {
                    if (_vsCodePath.Empty() || !File.Exists(_vsCodePath))
                    {
                        // Try to search it again if it wasn't found last time or if it was removed from its location
                        _vsCodePath = VsCodeNames.SelectFirstNotNull(OS.PathWhich, orElse: string.Empty);
                    }

                    var args = new List<string>();

                    bool osxAppBundleInstalled = false;

                    if (OS.IsOSX())
                    {
                        // The package path is '/Applications/Visual Studio Code.app'
                        const string vscodeBundleId = "com.microsoft.VSCode";

                        osxAppBundleInstalled = Internal.IsOsxAppBundleInstalled(vscodeBundleId);

                        if (osxAppBundleInstalled)
                        {
                            args.Add("-b");
                            args.Add(vscodeBundleId);

                            // The reusing of existing windows made by the 'open' command might not choose a wubdiw that is
                            // editing our folder. It's better to ask for a new window and let VSCode do the window management.
                            args.Add("-n");

                            // The open process must wait until the application finishes (which is instant in VSCode's case)
                            args.Add("--wait-apps");

                            args.Add("--args");
                        }
                    }

                    var resourcePath = ProjectSettings.GlobalizePath("res://");
                    args.Add(resourcePath);

                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);

                    if (line >= 0)
                    {
                        args.Add("-g");
                        args.Add($"{scriptPath}:{line + 1}:{col}");
                    }
                    else
                    {
                        args.Add(scriptPath);
                    }

                    string command;

                    if (OS.IsOSX())
                    {
                        if (!osxAppBundleInstalled && _vsCodePath.Empty())
                        {
                            GD.PushError("Cannot find code editor: VSCode");
                            return Error.FileNotFound;
                        }

                        command = osxAppBundleInstalled ? "/usr/bin/open" : _vsCodePath;
                    }
                    else
                    {
                        if (_vsCodePath.Empty())
                        {
                            GD.PushError("Cannot find code editor: VSCode");
                            return Error.FileNotFound;
                        }

                        command = _vsCodePath;
                    }

                    try
                    {
                        OS.RunProcess(command, args);
                    }
                    catch (Exception e)
                    {
                        GD.PushError($"Error when trying to run code editor: VSCode. Exception message: '{e.Message}'");
                    }

                    break;
                }

                default:
                    throw new ArgumentOutOfRangeException();
            }

            return Error.Ok;
        }

        public bool OverridesExternalEditor()
        {
            return (ExternalEditorId) editorSettings.GetSetting("mono/editor/external_editor") != ExternalEditorId.None;
        }

        public override bool Build()
        {
            return BuildManager.EditorBuildCallback();
        }

        public override void EnablePlugin()
        {
            base.EnablePlugin();

            if (Instance != null)
                throw new InvalidOperationException();
            Instance = this;

            var editorInterface = GetEditorInterface();
            var editorBaseControl = editorInterface.GetBaseControl();

            editorSettings = editorInterface.GetEditorSettings();

            errorDialog = new AcceptDialog();
            editorBaseControl.AddChild(errorDialog);

            BottomPanel = new BottomPanel();

            bottomPanelBtn = AddControlToBottomPanel(BottomPanel, "Mono".TTR());

            AddChild(new HotReloadAssemblyWatcher {Name = "HotReloadAssemblyWatcher"});

            menuPopup = new PopupMenu();
            menuPopup.Hide();
            menuPopup.SetAsToplevel(true);

            AddToolSubmenuItem("Mono", menuPopup);

            // TODO: Remove or edit this info dialog once Mono support is no longer in alpha
            {
                menuPopup.AddItem("About C# support".TTR(), (int) MenuOptions.AboutCSharp);
                aboutDialog = new AcceptDialog();
                editorBaseControl.AddChild(aboutDialog);
                aboutDialog.WindowTitle = "Important: C# support is not feature-complete";

                // We don't use DialogText as the default AcceptDialog Label doesn't play well with the TextureRect and CheckBox
                // we'll add. Instead we add containers and a new autowrapped Label inside.

                // Main VBoxContainer (icon + label on top, checkbox at bottom)
                var aboutVBox = new VBoxContainer();
                aboutDialog.AddChild(aboutVBox);

                // HBoxContainer for icon + label
                var aboutHBox = new HBoxContainer();
                aboutVBox.AddChild(aboutHBox);

                var aboutIcon = new TextureRect();
                aboutIcon.Texture = aboutIcon.GetIcon("NodeWarning", "EditorIcons");
                aboutHBox.AddChild(aboutIcon);

                var aboutLabel = new Label();
                aboutHBox.AddChild(aboutLabel);
                aboutLabel.RectMinSize = new Vector2(600, 150) * EditorScale;
                aboutLabel.SizeFlagsVertical = (int) Control.SizeFlags.ExpandFill;
                aboutLabel.Autowrap = true;
                aboutLabel.Text =
                    "C# support in Godot Engine is in late alpha stage and, while already usable, " +
                    "it is not meant for use in production.\n\n" +
                    "Projects can be exported to Linux, macOS, Windows and Android, but not yet to iOS, HTML5 or UWP. " +
                    "Bugs and usability issues will be addressed gradually over future releases, " +
                    "potentially including compatibility breaking changes as new features are implemented for a better overall C# experience.\n\n" +
                    "If you experience issues with this Mono build, please report them on Godot's issue tracker with details about your system, MSBuild version, IDE, etc.:\n\n" +
                    "        https://github.com/godotengine/godot/issues\n\n" +
                    "Your critical feedback at this stage will play a great role in shaping the C# support in future releases, so thank you!";

                EditorDef("mono/editor/show_info_on_start", true);

                // CheckBox in main container
                aboutDialogCheckBox = new CheckBox {Text = "Show this warning when starting the editor"};
                aboutDialogCheckBox.Connect("toggled", this, nameof(_ToggleAboutDialogOnStart));
                aboutVBox.AddChild(aboutDialogCheckBox);
            }

            if (File.Exists(GodotSharpDirs.ProjectSlnPath) && File.Exists(GodotSharpDirs.ProjectCsProjPath))
            {
                // Make sure the existing project has Api assembly references configured correctly
                CsProjOperations.FixApiHintPath(GodotSharpDirs.ProjectCsProjPath);
            }
            else
            {
                bottomPanelBtn.Hide();
                menuPopup.AddItem("Create C# solution".TTR(), (int) MenuOptions.CreateSln);
            }

            menuPopup.Connect("id_pressed", this, nameof(_MenuOptionPressed));

            var buildButton = new ToolButton
            {
                Text = "Build",
                HintTooltip = "Build solution",
                FocusMode = Control.FocusModeEnum.None
            };
            buildButton.Connect("pressed", this, nameof(_BuildSolutionPressed));
            AddControlToContainer(CustomControlContainer.Toolbar, buildButton);

            // External editor settings
            EditorDef("mono/editor/external_editor", ExternalEditorId.None);

            string settingsHintStr = "Disabled";

            if (OS.IsWindows())
            {
                settingsHintStr += $",MonoDevelop:{(int) ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int) ExternalEditorId.VsCode}";
            }
            else if (OS.IsOSX())
            {
                settingsHintStr += $",Visual Studio:{(int) ExternalEditorId.VisualStudioForMac}" +
                                   $",MonoDevelop:{(int) ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int) ExternalEditorId.VsCode}";
            }
            else if (OS.IsUnix())
            {
                settingsHintStr += $",MonoDevelop:{(int) ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int) ExternalEditorId.VsCode}";
            }

            editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
            {
                ["type"] = Variant.Type.Int,
                ["name"] = "mono/editor/external_editor",
                ["hint"] = PropertyHint.Enum,
                ["hint_string"] = settingsHintStr
            });

            // Export plugin
            var exportPlugin = new GodotSharpExport();
            AddExportPlugin(exportPlugin);
            exportPluginWeak = WeakRef(exportPlugin);

            BuildManager.Initialize();

            GodotIdeManager = new GodotIdeManager();
            AddChild(GodotIdeManager);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            if (exportPluginWeak != null)
            {
                // We need to dispose our export plugin before the editor destroys EditorSettings.
                // Otherwise, if the GC disposes it at a later time, EditorExportPlatformAndroid
                // will be freed after EditorSettings already was, and its device polling thread
                // will try to access the EditorSettings singleton, resulting in null dereferencing.
                (exportPluginWeak.GetRef() as GodotSharpExport)?.Dispose();

                exportPluginWeak.Dispose();
            }

            GodotIdeManager?.Dispose();
        }

        public void OnBeforeSerialize()
        {
        }

        public void OnAfterDeserialize()
        {
            Instance = this;
        }

        // Singleton

        public static GodotSharpEditor Instance { get; private set; }

        private GodotSharpEditor()
        {
        }
    }
}
