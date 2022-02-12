using Godot;
using GodotTools.Core;
using GodotTools.Export;
using GodotTools.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using GodotTools.Build;
using GodotTools.Ides;
using GodotTools.Ides.Rider;
using GodotTools.Internals;
using GodotTools.ProjectEditor;
using JetBrains.Annotations;
using static GodotTools.Internals.Globals;
using File = GodotTools.Utils.File;
using OS = GodotTools.Utils.OS;
using Path = System.IO.Path;

namespace GodotTools
{
    public class GodotSharpEditor : EditorPlugin, ISerializationListener
    {
        private EditorSettings _editorSettings;

        private PopupMenu _menuPopup;

        private AcceptDialog _errorDialog;

        private Button _bottomPanelBtn;
        private Button _toolBarBuildButton;

        // TODO Use WeakReference once we have proper serialization.
        private WeakRef _exportPluginWeak;

        public GodotIdeManager GodotIdeManager { get; private set; }

        public MSBuildPanel MSBuildPanel { get; private set; }

        public bool SkipBuildBeforePlaying { get; set; } = false;

        public static string ProjectAssemblyName
        {
            get
            {
                string projectAssemblyName = (string)ProjectSettings.GetSetting("application/config/name");
                projectAssemblyName = projectAssemblyName.ToSafeDirName();
                if (string.IsNullOrEmpty(projectAssemblyName))
                    projectAssemblyName = "UnnamedProject";
                return projectAssemblyName;
            }
        }

        private bool CreateProjectSolution()
        {
            using (var pr = new EditorProgress("create_csharp_solution", "Generating solution...".TTR(), 3))
            {
                pr.Step("Generating C# project...".TTR());

                string resourceDir = ProjectSettings.GlobalizePath("res://");

                string path = resourceDir;
                string name = ProjectAssemblyName;

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
                        Configs = new List<string> {"Debug", "ExportDebug", "ExportRelease"}
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

                    pr.Step("Updating Godot API assemblies...".TTR());

                    string debugApiAssembliesError = Internal.UpdateApiAssembliesFromPrebuilt("Debug");

                    if (!string.IsNullOrEmpty(debugApiAssembliesError))
                    {
                        ShowErrorDialog("Failed to update the Godot API assemblies: " + debugApiAssembliesError);
                        return false;
                    }

                    string releaseApiAssembliesError = Internal.UpdateApiAssembliesFromPrebuilt("Release");

                    if (!string.IsNullOrEmpty(releaseApiAssembliesError))
                    {
                        ShowErrorDialog("Failed to update the Godot API assemblies: " + releaseApiAssembliesError);
                        return false;
                    }

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
            _menuPopup.RemoveItem(_menuPopup.GetItemIndex((int)MenuOptions.CreateSln));
            _bottomPanelBtn.Show();
            _toolBarBuildButton.Show();
        }

        private void _MenuOptionPressed(int id)
        {
            switch ((MenuOptions)id)
            {
                case MenuOptions.CreateSln:
                    CreateProjectSolution();
                    break;
                case MenuOptions.SetupGodotNugetFallbackFolder:
                {
                    try
                    {
                        string fallbackFolder = NuGetUtils.GodotFallbackFolderPath;
                        NuGetUtils.AddFallbackFolderToUserNuGetConfigs(NuGetUtils.GodotFallbackFolderName, fallbackFolder);
                        NuGetUtils.AddBundledPackagesToFallbackFolder(fallbackFolder);
                    }
                    catch (Exception e)
                    {
                        ShowErrorDialog("Failed to setup Godot NuGet Offline Packages: " + e.Message);
                    }

                    break;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(id), id, "Invalid menu option");
            }
        }

        private void BuildSolutionPressed()
        {
            if (!File.Exists(GodotSharpDirs.ProjectSlnPath))
            {
                if (!CreateProjectSolution())
                    return; // Failed to create solution
            }

            Instance.MSBuildPanel.BuildSolution();
        }

        public override void _Ready()
        {
            base._Ready();

            MSBuildPanel.BuildOutputView.BuildStateChanged += BuildStateChanged;
        }

        private enum MenuOptions
        {
            CreateSln,
            SetupGodotNugetFallbackFolder,
        }

        public void ShowErrorDialog(string message, string title = "Error")
        {
            _errorDialog.Title = title;
            _errorDialog.DialogText = message;
            _errorDialog.PopupCentered();
        }

        private static string _vsCodePath = string.Empty;

        private static readonly string[] VsCodeNames =
        {
            "code", "code-oss", "vscode", "vscode-oss", "visual-studio-code", "visual-studio-code-oss"
        };

        [UsedImplicitly]
        public Error OpenInExternalEditor(Script script, int line, int col)
        {
            var editorId = (ExternalEditorId)_editorSettings.GetSetting("mono/editor/external_editor");

            switch (editorId)
            {
                case ExternalEditorId.None:
                    // Not an error. Tells the caller to fallback to the global external editor settings or the built-in editor.
                    return Error.Unavailable;
                case ExternalEditorId.VisualStudio:
                {
                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);

                    var args = new List<string>
                    {
                        GodotSharpDirs.ProjectSlnPath,
                        line >= 0 ? $"{scriptPath};{line + 1};{col + 1}" : scriptPath
                    };

                    string command = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "GodotTools.OpenVisualStudio.exe");

                    try
                    {
                        if (Godot.OS.IsStdoutVerbose())
                            Console.WriteLine($"Running: \"{command}\" {string.Join(" ", args.Select(a => $"\"{a}\""))}");

                        OS.RunProcess(command, args);
                    }
                    catch (Exception e)
                    {
                        GD.PushError($"Error when trying to run code editor: VisualStudio. Exception message: '{e.Message}'");
                    }

                    break;
                }
                case ExternalEditorId.VisualStudioForMac:
                    goto case ExternalEditorId.MonoDevelop;
                case ExternalEditorId.Rider:
                {
                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);
                    RiderPathManager.OpenFile(GodotSharpDirs.ProjectSlnPath, scriptPath, line);
                    return Error.Ok;
                }
                case ExternalEditorId.MonoDevelop:
                {
                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);

                    GodotIdeManager.LaunchIdeAsync().ContinueWith(launchTask =>
                    {
                        var editorPick = launchTask.Result;
                        if (line >= 0)
                            editorPick?.SendOpenFile(scriptPath, line + 1, col);
                        else
                            editorPick?.SendOpenFile(scriptPath);
                    });

                    break;
                }
                case ExternalEditorId.VsCode:
                {
                    if (string.IsNullOrEmpty(_vsCodePath) || !File.Exists(_vsCodePath))
                    {
                        // Try to search it again if it wasn't found last time or if it was removed from its location
                        _vsCodePath = VsCodeNames.SelectFirstNotNull(OS.PathWhich, orElse: string.Empty);
                    }

                    var args = new List<string>();

                    bool osxAppBundleInstalled = false;

                    if (OS.IsMacOS)
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

                    string resourcePath = ProjectSettings.GlobalizePath("res://");
                    args.Add(resourcePath);

                    string scriptPath = ProjectSettings.GlobalizePath(script.ResourcePath);

                    if (line >= 0)
                    {
                        args.Add("-g");
                        args.Add($"{scriptPath}:{line}:{col}");
                    }
                    else
                    {
                        args.Add(scriptPath);
                    }

                    string command;

                    if (OS.IsMacOS)
                    {
                        if (!osxAppBundleInstalled && string.IsNullOrEmpty(_vsCodePath))
                        {
                            GD.PushError("Cannot find code editor: VSCode");
                            return Error.FileNotFound;
                        }

                        command = osxAppBundleInstalled ? "/usr/bin/open" : _vsCodePath;
                    }
                    else
                    {
                        if (string.IsNullOrEmpty(_vsCodePath))
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

        [UsedImplicitly]
        public bool OverridesExternalEditor()
        {
            return (ExternalEditorId)_editorSettings.GetSetting("mono/editor/external_editor") != ExternalEditorId.None;
        }

        public override bool _Build()
        {
            return BuildManager.EditorBuildCallback();
        }

        private void ApplyNecessaryChangesToSolution()
        {
            try
            {
                // Migrate solution from old configuration names to: Debug, ExportDebug and ExportRelease
                DotNetSolution.MigrateFromOldConfigNames(GodotSharpDirs.ProjectSlnPath);

                var msbuildProject = ProjectUtils.Open(GodotSharpDirs.ProjectCsProjPath)
                                     ?? throw new Exception("Cannot open C# project");

                // NOTE: The order in which changes are made to the project is important

                // Migrate to MSBuild project Sdks style if using the old style
                ProjectUtils.MigrateToProjectSdksStyle(msbuildProject, ProjectAssemblyName);

                ProjectUtils.EnsureGodotSdkIsUpToDate(msbuildProject);

                if (msbuildProject.HasUnsavedChanges)
                {
                    // Save a copy of the project before replacing it
                    FileUtils.SaveBackupCopy(GodotSharpDirs.ProjectCsProjPath);

                    msbuildProject.Save();
                }
            }
            catch (Exception e)
            {
                GD.PushError(e.ToString());
            }
        }

        private void BuildStateChanged()
        {
            if (_bottomPanelBtn != null)
                _bottomPanelBtn.Icon = MSBuildPanel.BuildOutputView.BuildStateIcon;
        }

        public override void _EnablePlugin()
        {
            base._EnablePlugin();

            if (Instance != null)
                throw new InvalidOperationException();
            Instance = this;

            var editorInterface = GetEditorInterface();
            var editorBaseControl = editorInterface.GetBaseControl();

            _editorSettings = editorInterface.GetEditorSettings();

            _errorDialog = new AcceptDialog();
            editorBaseControl.AddChild(_errorDialog);

            MSBuildPanel = new MSBuildPanel();
            _bottomPanelBtn = AddControlToBottomPanel(MSBuildPanel, "MSBuild".TTR());

            AddChild(new HotReloadAssemblyWatcher {Name = "HotReloadAssemblyWatcher"});

            _menuPopup = new PopupMenu();
            _menuPopup.Hide();

            AddToolSubmenuItem("C#", _menuPopup);

            var buildSolutionShortcut = (Shortcut)EditorShortcut("mono/build_solution");

            _toolBarBuildButton = new Button
            {
                Text = "Build",
                HintTooltip = "Build Solution".TTR(),
                FocusMode = Control.FocusModeEnum.None,
                Shortcut = buildSolutionShortcut,
                ShortcutInTooltip = true
            };
            _toolBarBuildButton.Pressed += BuildSolutionPressed;
            AddControlToContainer(CustomControlContainer.Toolbar, _toolBarBuildButton);

            if (File.Exists(GodotSharpDirs.ProjectSlnPath) && File.Exists(GodotSharpDirs.ProjectCsProjPath))
            {
                ApplyNecessaryChangesToSolution();
            }
            else
            {
                _bottomPanelBtn.Hide();
                _toolBarBuildButton.Hide();
                _menuPopup.AddItem("Create C# solution".TTR(), (int)MenuOptions.CreateSln);
            }

            _menuPopup.IdPressed += _MenuOptionPressed;

            // External editor settings
            EditorDef("mono/editor/external_editor", ExternalEditorId.None);

            string settingsHintStr = "Disabled";

            if (OS.IsWindows)
            {
                settingsHintStr += $",Visual Studio:{(int)ExternalEditorId.VisualStudio}" +
                                   $",MonoDevelop:{(int)ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int)ExternalEditorId.VsCode}" +
                                   $",JetBrains Rider:{(int)ExternalEditorId.Rider}";
            }
            else if (OS.IsMacOS)
            {
                settingsHintStr += $",Visual Studio:{(int)ExternalEditorId.VisualStudioForMac}" +
                                   $",MonoDevelop:{(int)ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int)ExternalEditorId.VsCode}" +
                                   $",JetBrains Rider:{(int)ExternalEditorId.Rider}";
            }
            else if (OS.IsUnixLike)
            {
                settingsHintStr += $",MonoDevelop:{(int)ExternalEditorId.MonoDevelop}" +
                                   $",Visual Studio Code:{(int)ExternalEditorId.VsCode}" +
                                   $",JetBrains Rider:{(int)ExternalEditorId.Rider}";
            }

            _editorSettings.AddPropertyInfo(new Godot.Collections.Dictionary
            {
                ["type"] = Variant.Type.Int,
                ["name"] = "mono/editor/external_editor",
                ["hint"] = PropertyHint.Enum,
                ["hint_string"] = settingsHintStr
            });

            // Export plugin
            var exportPlugin = new ExportPlugin();
            AddExportPlugin(exportPlugin);
            exportPlugin.RegisterExportSettings();
            _exportPluginWeak = WeakRef(exportPlugin);

            try
            {
                // At startup we make sure NuGet.Config files have our Godot NuGet fallback folder included
                NuGetUtils.AddFallbackFolderToUserNuGetConfigs(NuGetUtils.GodotFallbackFolderName, NuGetUtils.GodotFallbackFolderPath);
            }
            catch (Exception e)
            {
                GD.PushError("Failed to add Godot NuGet Offline Packages to NuGet.Config: " + e.Message);
            }

            BuildManager.Initialize();
            RiderPathManager.Initialize();

            GodotIdeManager = new GodotIdeManager();
            AddChild(GodotIdeManager);
        }

        protected override void Dispose(bool disposing)
        {
            base.Dispose(disposing);

            if (_exportPluginWeak != null)
            {
                // We need to dispose our export plugin before the editor destroys EditorSettings.
                // Otherwise, if the GC disposes it at a later time, EditorExportPlatformAndroid
                // will be freed after EditorSettings already was, and its device polling thread
                // will try to access the EditorSettings singleton, resulting in null dereferencing.
                (_exportPluginWeak.GetRef() as ExportPlugin)?.Dispose();

                _exportPluginWeak.Dispose();
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

        [UsedImplicitly]
        private GodotSharpEditor()
        {
        }
    }
}
