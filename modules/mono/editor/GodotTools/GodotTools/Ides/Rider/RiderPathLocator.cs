using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Godot;
using JetBrains.Annotations;
using Microsoft.Win32;
using Newtonsoft.Json;
using Directory = System.IO.Directory;
using Environment = System.Environment;
using File = System.IO.File;
using Path = System.IO.Path;
using OS = GodotTools.Utils.OS;

// ReSharper disable UnassignedField.Local
// ReSharper disable InconsistentNaming
// ReSharper disable UnassignedField.Global
// ReSharper disable MemberHidesStaticFromOuterClass

namespace GodotTools.Ides.Rider
{
    /// <summary>
    /// This code is a modified version of the JetBrains resharper-unity plugin listed under Apache License 2.0 license:
    /// https://github.com/JetBrains/resharper-unity/blob/master/unity/JetBrains.Rider.Unity.Editor/EditorPlugin/RiderPathLocator.cs
    /// </summary>
    public static class RiderPathLocator
    {
        public static RiderInfo[] GetAllRiderPaths()
        {
            try
            {
                if (OS.IsWindows)
                {
                    return CollectRiderInfosWindows();
                }
                if (OS.IsOSX)
                {
                    return CollectRiderInfosMac();
                }
                if (OS.IsUnixLike)
                {
                    return CollectAllRiderPathsLinux();
                }
                throw new Exception("Unexpected OS.");
            }
            catch (Exception e)
            {
                GD.PushWarning(e.Message);
            }

            return Array.Empty<RiderInfo>();
        }

        private static RiderInfo[] CollectAllRiderPathsLinux()
        {
            var installInfos = new List<RiderInfo>();
            string home = Environment.GetEnvironmentVariable("HOME");
            if (!string.IsNullOrEmpty(home))
            {
                string toolboxRiderRootPath = GetToolboxBaseDir();
                installInfos.AddRange(CollectPathsFromToolbox(toolboxRiderRootPath, "bin", "rider.sh", false)
                  .Select(a => new RiderInfo(a, true)).ToList());

                //$Home/.local/share/applications/jetbrains-rider.desktop
                var shortcut = new FileInfo(Path.Combine(home, @".local/share/applications/jetbrains-rider.desktop"));

                if (shortcut.Exists)
                {
                    string[] lines = File.ReadAllLines(shortcut.FullName);
                    foreach (string line in lines)
                    {
                        if (!line.StartsWith("Exec=\""))
                            continue;
                        string path = line.Split('"').Where((item, index) => index == 1).SingleOrDefault();
                        if (string.IsNullOrEmpty(path))
                            continue;

                        if (installInfos.Any(a => a.Path == path)) // avoid adding similar build as from toolbox
                            continue;
                        installInfos.Add(new RiderInfo(path, false));
                    }
                }
            }

            // snap install
            string snapInstallPath = "/snap/rider/current/bin/rider.sh";
            if (new FileInfo(snapInstallPath).Exists)
                installInfos.Add(new RiderInfo(snapInstallPath, false));

            return installInfos.ToArray();
        }

        private static RiderInfo[] CollectRiderInfosMac()
        {
            var installInfos = new List<RiderInfo>();
            // "/Applications/*Rider*.app"
            // should be combined with "Contents/MacOS/rider"
            var folder = new DirectoryInfo("/Applications");
            if (folder.Exists)
            {
                installInfos.AddRange(folder.GetDirectories("*Rider*.app")
                    .Select(a => new RiderInfo(Path.Combine(a.FullName, "Contents/MacOS/rider"), false))
                    .ToList());
            }

            // /Users/user/Library/Application Support/JetBrains/Toolbox/apps/Rider/ch-1/181.3870.267/Rider EAP.app
            // should be combined with "Contents/MacOS/rider"
            string toolboxRiderRootPath = GetToolboxBaseDir();
            var paths = CollectPathsFromToolbox(toolboxRiderRootPath, "", "Rider*.app", true)
                .Select(a => new RiderInfo(Path.Combine(a, "Contents/MacOS/rider"), true));
            installInfos.AddRange(paths);

            return installInfos.ToArray();
        }

        private static RiderInfo[] CollectRiderInfosWindows()
        {
            var installInfos = new List<RiderInfo>();
            var toolboxRiderRootPath = GetToolboxBaseDir();
            var installPathsToolbox = CollectPathsFromToolbox(toolboxRiderRootPath, "bin", "rider64.exe", false).ToList();
            installInfos.AddRange(installPathsToolbox.Select(a => new RiderInfo(a, true)).ToList());

            var installPaths = new List<string>();
            const string registryKey = @"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall";
            CollectPathsFromRegistry(registryKey, installPaths);
            const string wowRegistryKey = @"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall";
            CollectPathsFromRegistry(wowRegistryKey, installPaths);

            installInfos.AddRange(installPaths.Select(a => new RiderInfo(a, false)).ToList());

            return installInfos.ToArray();
        }

        private static string GetToolboxBaseDir()
        {
            if (OS.IsWindows)
            {
                string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
                return GetToolboxRiderRootPath(localAppData);
            }

            if (OS.IsOSX)
            {
                var home = Environment.GetEnvironmentVariable("HOME");
                if (string.IsNullOrEmpty(home))
                    return string.Empty;
                var localAppData = Path.Combine(home, @"Library/Application Support");
                return GetToolboxRiderRootPath(localAppData);
            }

            if (OS.IsUnixLike)
            {
                var home = Environment.GetEnvironmentVariable("HOME");
                if (string.IsNullOrEmpty(home))
                    return string.Empty;
                var localAppData = Path.Combine(home, @".local/share");
                return GetToolboxRiderRootPath(localAppData);
            }

            return string.Empty;
        }


        private static string GetToolboxRiderRootPath(string localAppData)
        {
            var toolboxPath = Path.Combine(localAppData, @"JetBrains/Toolbox");
            var settingsJson = Path.Combine(toolboxPath, ".settings.json");

            if (File.Exists(settingsJson))
            {
                var path = SettingsJson.GetInstallLocationFromJson(File.ReadAllText(settingsJson));
                if (!string.IsNullOrEmpty(path))
                    toolboxPath = path;
            }

            var toolboxRiderRootPath = Path.Combine(toolboxPath, @"apps/Rider");
            return toolboxRiderRootPath;
        }

        internal static ProductInfo GetBuildVersion(string path)
        {
            var buildTxtFileInfo = new FileInfo(Path.Combine(path, GetRelativePathToBuildTxt()));
            var dir = buildTxtFileInfo.DirectoryName;
            if (!Directory.Exists(dir))
                return null;
            var buildVersionFile = new FileInfo(Path.Combine(dir, "product-info.json"));
            if (!buildVersionFile.Exists)
                return null;
            var json = File.ReadAllText(buildVersionFile.FullName);
            return ProductInfo.GetProductInfo(json);
        }

        internal static Version GetBuildNumber(string path)
        {
            var file = new FileInfo(Path.Combine(path, GetRelativePathToBuildTxt()));
            if (!file.Exists)
                return null;
            var text = File.ReadAllText(file.FullName);
            if (text.Length <= 3)
                return null;

            var versionText = text.Substring(3);
            return Version.TryParse(versionText, out var v) ? v : null;
        }

        internal static bool IsToolbox(string path)
        {
            return path.StartsWith(GetToolboxBaseDir());
        }

        private static string GetRelativePathToBuildTxt()
        {
            if (OS.IsWindows || OS.IsUnixLike)
                return "../../build.txt";
            if (OS.IsOSX)
                return "Contents/Resources/build.txt";
            throw new Exception("Unknown OS.");
        }

        private static void CollectPathsFromRegistry(string registryKey, List<string> installPaths)
        {
            using (var key = Registry.CurrentUser.OpenSubKey(registryKey))
            {
                CollectPathsFromRegistry(installPaths, key);
            }
            using (var key = Registry.LocalMachine.OpenSubKey(registryKey))
            {
                CollectPathsFromRegistry(installPaths, key);
            }
        }

        private static void CollectPathsFromRegistry(List<string> installPaths, RegistryKey key)
        {
            if (key == null) return;
            foreach (var subkeyName in key.GetSubKeyNames().Where(a => a.Contains("Rider")))
            {
                using (var subkey = key.OpenSubKey(subkeyName))
                {
                    var folderObject = subkey?.GetValue("InstallLocation");
                    if (folderObject == null) continue;
                    var folder = folderObject.ToString();
                    var possiblePath = Path.Combine(folder, @"bin\rider64.exe");
                    if (File.Exists(possiblePath))
                        installPaths.Add(possiblePath);
                }
            }
        }

        private static string[] CollectPathsFromToolbox(string toolboxRiderRootPath, string dirName, string searchPattern,
          bool isMac)
        {
            if (!Directory.Exists(toolboxRiderRootPath))
                return Array.Empty<string>();

            var channelDirs = Directory.GetDirectories(toolboxRiderRootPath);
            var paths = channelDirs.SelectMany(channelDir =>
              {
                  try
                  {
                      // use history.json - last entry stands for the active build https://jetbrains.slack.com/archives/C07KNP99D/p1547807024066500?thread_ts=1547731708.057700&cid=C07KNP99D
                      var historyFile = Path.Combine(channelDir, ".history.json");
                      if (File.Exists(historyFile))
                      {
                          var json = File.ReadAllText(historyFile);
                          var build = ToolboxHistory.GetLatestBuildFromJson(json);
                          if (build != null)
                          {
                              var buildDir = Path.Combine(channelDir, build);
                              var executablePaths = GetExecutablePaths(dirName, searchPattern, isMac, buildDir);
                              if (executablePaths.Any())
                                  return executablePaths;
                          }
                      }

                      var channelFile = Path.Combine(channelDir, ".channel.settings.json");
                      if (File.Exists(channelFile))
                      {
                          var json = File.ReadAllText(channelFile).Replace("active-application", "active_application");
                          var build = ToolboxInstallData.GetLatestBuildFromJson(json);
                          if (build != null)
                          {
                              var buildDir = Path.Combine(channelDir, build);
                              var executablePaths = GetExecutablePaths(dirName, searchPattern, isMac, buildDir);
                              if (executablePaths.Any())
                                  return executablePaths;
                          }
                      }

                      // changes in toolbox json files format may brake the logic above, so return all found Rider installations
                      return Directory.GetDirectories(channelDir)
                          .SelectMany(buildDir => GetExecutablePaths(dirName, searchPattern, isMac, buildDir));
                  }
                  catch (Exception e)
                  {
                      // do not write to Debug.Log, just log it.
                      Logger.Warn($"Failed to get RiderPath from {channelDir}", e);
                  }

                  return Array.Empty<string>();
              })
              .Where(c => !string.IsNullOrEmpty(c))
              .ToArray();
            return paths;
        }

        private static string[] GetExecutablePaths(string dirName, string searchPattern, bool isMac, string buildDir)
        {
            var folder = new DirectoryInfo(Path.Combine(buildDir, dirName));
            if (!folder.Exists)
                return Array.Empty<string>();

            if (!isMac)
                return new[] { Path.Combine(folder.FullName, searchPattern) }.Where(File.Exists).ToArray();
            return folder.GetDirectories(searchPattern).Select(f => f.FullName)
              .Where(Directory.Exists).ToArray();
        }

        // Disable the "field is never assigned" compiler warning. We never assign it, but Unity does.
        // Note that Unity disable this warning in the generated C# projects
#pragma warning disable 0649

        [Serializable]
        class SettingsJson
        {
            public string install_location;

            [CanBeNull]
            public static string GetInstallLocationFromJson(string json)
            {
                try
                {
                    return JsonConvert.DeserializeObject<SettingsJson>(json).install_location;
                }
                catch (Exception)
                {
                    Logger.Warn($"Failed to get install_location from json {json}");
                }

                return null;
            }
        }

        [Serializable]
        class ToolboxHistory
        {
            public List<ItemNode> history;

            public static string GetLatestBuildFromJson(string json)
            {
                try
                {
                    return JsonConvert.DeserializeObject<ToolboxHistory>(json).history.LastOrDefault()?.item.build;
                }
                catch (Exception)
                {
                    Logger.Warn($"Failed to get latest build from json {json}");
                }

                return null;
            }
        }

        [Serializable]
        class ItemNode
        {
            public BuildNode item;
        }

        [Serializable]
        class BuildNode
        {
            public string build;
        }

        [Serializable]
        public class ProductInfo
        {
            public string version;
            public string versionSuffix;

            [CanBeNull]
            internal static ProductInfo GetProductInfo(string json)
            {
                try
                {
                    var productInfo = JsonConvert.DeserializeObject<ProductInfo>(json);
                    return productInfo;
                }
                catch (Exception)
                {
                    Logger.Warn($"Failed to get version from json {json}");
                }

                return null;
            }
        }

        // ReSharper disable once ClassNeverInstantiated.Global
        [Serializable]
        class ToolboxInstallData
        {
            // ReSharper disable once InconsistentNaming
            public ActiveApplication active_application;

            [CanBeNull]
            public static string GetLatestBuildFromJson(string json)
            {
                try
                {
                    var toolbox = JsonConvert.DeserializeObject<ToolboxInstallData>(json);
                    var builds = toolbox.active_application.builds;
                    if (builds != null && builds.Any())
                        return builds.First();
                }
                catch (Exception)
                {
                    Logger.Warn($"Failed to get latest build from json {json}");
                }

                return null;
            }
        }

        [Serializable]
        class ActiveApplication
        {
            public List<string> builds;
        }

#pragma warning restore 0649

        public struct RiderInfo
        {
            // ReSharper disable once NotAccessedField.Global
            public bool IsToolbox;
            public string Presentation;
            public Version BuildNumber;
            public ProductInfo ProductInfo;
            public string Path;

            public RiderInfo(string path, bool isToolbox)
            {
                BuildNumber = GetBuildNumber(path);
                ProductInfo = GetBuildVersion(path);
                Path = new FileInfo(path).FullName; // normalize separators
                var presentation = $"Rider {BuildNumber}";

                if (ProductInfo != null && !string.IsNullOrEmpty(ProductInfo.version))
                {
                    var suffix = string.IsNullOrEmpty(ProductInfo.versionSuffix) ? "" : $" {ProductInfo.versionSuffix}";
                    presentation = $"Rider {ProductInfo.version}{suffix}";
                }

                if (isToolbox)
                    presentation += " (JetBrains Toolbox)";

                Presentation = presentation;
                IsToolbox = isToolbox;
            }
        }

        private static class Logger
        {
            internal static void Warn(string message, Exception e = null)
            {
                throw new Exception(message, e);
            }
        }
    }
}
