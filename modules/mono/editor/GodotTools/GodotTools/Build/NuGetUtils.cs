using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Xml;
using Godot;
using GodotTools.Internals;
using GodotTools.Shared;
using Directory = GodotTools.Utils.Directory;
using Environment = System.Environment;
using File = GodotTools.Utils.File;

namespace GodotTools.Build
{
    public static class NuGetUtils
    {
        public const string GodotFallbackFolderName = "Godot Offline Packages";

        public static string GodotFallbackFolderPath
            => Path.Combine(GodotSharpDirs.MonoUserDir, "GodotNuGetFallbackFolder");

        private static void AddFallbackFolderToNuGetConfig(string nuGetConfigPath, string name, string path)
        {
            var xmlDoc = new XmlDocument();
            xmlDoc.Load(nuGetConfigPath);

            const string nuGetConfigRootName = "configuration";

            var rootNode = xmlDoc.DocumentElement;

            if (rootNode == null)
            {
                // No root node, create it
                rootNode = xmlDoc.CreateElement(nuGetConfigRootName);
                xmlDoc.AppendChild(rootNode);

                // Since this can be considered pretty much a new NuGet.Config, add the default nuget.org source as well
                XmlElement nugetOrgSourceEntry = xmlDoc.CreateElement("add");
                nugetOrgSourceEntry.Attributes.Append(xmlDoc.CreateAttribute("key")).Value = "nuget.org";
                nugetOrgSourceEntry.Attributes.Append(xmlDoc.CreateAttribute("value")).Value = "https://api.nuget.org/v3/index.json";
                nugetOrgSourceEntry.Attributes.Append(xmlDoc.CreateAttribute("protocolVersion")).Value = "3";
                rootNode.AppendChild(xmlDoc.CreateElement("packageSources")).AppendChild(nugetOrgSourceEntry);
            }
            else
            {
                // Check that the root node is the expected one
                if (rootNode.Name != nuGetConfigRootName)
                    throw new Exception("Invalid root Xml node for NuGet.Config. " +
                                        $"Expected '{nuGetConfigRootName}' got '{rootNode.Name}'.");
            }

            var fallbackFoldersNode = rootNode["fallbackPackageFolders"] ??
                                      rootNode.AppendChild(xmlDoc.CreateElement("fallbackPackageFolders"));

            // Check if it already has our fallback package folder
            for (var xmlNode = fallbackFoldersNode.FirstChild; xmlNode != null; xmlNode = xmlNode.NextSibling)
            {
                if (xmlNode.NodeType != XmlNodeType.Element)
                    continue;

                var xmlElement = (XmlElement)xmlNode;
                if (xmlElement.Name == "add" &&
                    xmlElement.Attributes["key"]?.Value == name &&
                    xmlElement.Attributes["value"]?.Value == path)
                {
                    return;
                }
            }

            XmlElement newEntry = xmlDoc.CreateElement("add");
            newEntry.Attributes.Append(xmlDoc.CreateAttribute("key")).Value = name;
            newEntry.Attributes.Append(xmlDoc.CreateAttribute("value")).Value = path;

            fallbackFoldersNode.AppendChild(newEntry);

            xmlDoc.Save(nuGetConfigPath);
        }

        /// <summary>
        /// Returns all the paths where the user NuGet.Config files can be found.
        /// Does not determine whether the returned files exist or not.
        /// </summary>
        private static string[] GetAllUserNuGetConfigFilePaths()
        {
            // Where to find 'NuGet/NuGet.Config':
            //
            // - Mono/.NETFramework (standalone NuGet):
            //     Uses Environment.SpecialFolder.ApplicationData
            //     - Windows: '%APPDATA%'
            //     - Linux/macOS: '$HOME/.config'
            // - CoreCLR (dotnet CLI NuGet):
            //     - Windows: '%APPDATA%'
            //     - Linux/macOS: '$DOTNET_CLI_HOME/.nuget' otherwise '$HOME/.nuget'

            string applicationData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

            if (Utils.OS.IsWindows)
            {
                // %APPDATA% for both
                return new[] {Path.Combine(applicationData, "NuGet", "NuGet.Config")};
            }

            var paths = new string[2];

            // CoreCLR (dotnet CLI NuGet)

            string dotnetCliHome = Environment.GetEnvironmentVariable("DOTNET_CLI_HOME");
            if (!string.IsNullOrEmpty(dotnetCliHome))
            {
                paths[0] = Path.Combine(dotnetCliHome, ".nuget", "NuGet", "NuGet.Config");
            }
            else
            {
                string home = Environment.GetEnvironmentVariable("HOME");
                if (string.IsNullOrEmpty(home))
                    throw new InvalidOperationException("Required environment variable 'HOME' is not set.");
                paths[0] = Path.Combine(home, ".nuget", "NuGet", "NuGet.Config");
            }

            // Mono/.NETFramework (standalone NuGet)

            // ApplicationData is $HOME/.config on Linux/macOS
            paths[1] = Path.Combine(applicationData, "NuGet", "NuGet.Config");

            return paths;
        }

        // nupkg extraction
        //
        // Exclude: (NuGet.Client -> NuGet.Packaging.PackageHelper.ExcludePaths)
        // package/
        // _rels/
        // [Content_Types].xml
        //
        // Don't ignore files that begin with a dot (.)
        //
        // The nuspec is not lower case inside the nupkg but must be made lower case when extracted.

        /// <summary>
        /// Adds the specified fallback folder to the user NuGet.Config files,
        /// for both standalone NuGet (Mono/.NETFramework) and dotnet CLI NuGet.
        /// </summary>
        public static void AddFallbackFolderToUserNuGetConfigs(string name, string path)
        {
            foreach (string nuGetConfigPath in GetAllUserNuGetConfigFilePaths())
            {
                if (!System.IO.File.Exists(nuGetConfigPath))
                {
                    // It doesn't exist, so we create a default one
                    const string defaultConfig = @"<?xml version=""1.0"" encoding=""utf-8""?>
<configuration>
  <packageSources>
    <add key=""nuget.org"" value=""https://api.nuget.org/v3/index.json"" protocolVersion=""3"" />
  </packageSources>
</configuration>
";
                    System.IO.File.WriteAllText(nuGetConfigPath, defaultConfig, Encoding.UTF8); // UTF-8 with BOM
                }

                AddFallbackFolderToNuGetConfig(nuGetConfigPath, name, path);
            }
        }

        private static void AddPackageToFallbackFolder(string fallbackFolder,
            string nupkgPath, string packageId, string packageVersion)
        {
            // dotnet CLI provides no command for this, but we can do it manually.
            //
            // - The expected structure is as follows:
            //     fallback_folder/
            //         <package.name>/<version>/
            //             <package.name>.<version>.nupkg
            //             <package.name>.<version>.nupkg.sha512
            //             <package.name>.nuspec
            //             ... extracted nupkg files (check code for excluded files) ...
            //
            // - <package.name> and <version> must be in lower case.
            // - The sha512 of the nupkg is base64 encoded.
            // - We can get the nuspec from the nupkg which is a Zip file.

            string packageIdLower = packageId.ToLower();
            string packageVersionLower = packageVersion.ToLower();

            string destDir = Path.Combine(fallbackFolder, packageIdLower, packageVersionLower);
            string nupkgDestPath = Path.Combine(destDir, $"{packageIdLower}.{packageVersionLower}.nupkg");
            string nupkgSha512DestPath = Path.Combine(destDir, $"{packageIdLower}.{packageVersionLower}.nupkg.sha512");

            if (File.Exists(nupkgDestPath) && File.Exists(nupkgSha512DestPath))
                return; // Already added (for speed we don't check if every file is properly extracted)

            Directory.CreateDirectory(destDir);

            // Generate .nupkg.sha512 file

            using (var alg = SHA512.Create())
            {
                alg.ComputeHash(File.ReadAllBytes(nupkgPath));
                string base64Hash = Convert.ToBase64String(alg.Hash);
                File.WriteAllText(nupkgSha512DestPath, base64Hash);
            }

            // Extract nupkg
            ExtractNupkg(destDir, nupkgPath, packageId, packageVersion);

            // Copy .nupkg
            File.Copy(nupkgPath, nupkgDestPath);
        }

        private static readonly string[] NupkgExcludePaths =
        {
            "_rels/",
            "package/",
            "[Content_Types].xml"
        };

        private static void ExtractNupkg(string destDir, string nupkgPath, string packageId, string packageVersion)
        {
            // NOTE: Must use SimplifyGodotPath to make sure we don't extract files outside the destination directory.

            using (var archive = ZipFile.OpenRead(nupkgPath))
            {
                // Extract .nuspec manually as it needs to be in lower case

                var nuspecEntry = archive.GetEntry(packageId + ".nuspec");

                if (nuspecEntry == null)
                    throw new InvalidOperationException($"Failed to extract package {packageId}.{packageVersion}. Could not find the nuspec file.");

                nuspecEntry.ExtractToFile(Path.Combine(destDir, nuspecEntry.Name.ToLower().SimplifyGodotPath()));

                // Extract the other package files

                foreach (var entry in archive.Entries)
                {
                    // NOTE: SimplifyGodotPath() removes trailing slash and backslash,
                    // so we can't use the result to check if the entry is a directory.

                    string entryFullName = entry.FullName.Replace('\\', '/');

                    // Check if the file must be ignored
                    if ( // Excluded files.
                        NupkgExcludePaths.Any(e => entryFullName.StartsWith(e, StringComparison.OrdinalIgnoreCase)) ||
                        // Nupkg hash files and nupkg metadata files on all directory.
                        entryFullName.EndsWith(".nupkg.sha512", StringComparison.OrdinalIgnoreCase) ||
                        entryFullName.EndsWith(".nupkg.metadata", StringComparison.OrdinalIgnoreCase) ||
                        // Nuspec at root level. We already extracted it previously but in lower case.
                        entryFullName.IndexOf('/') == -1 && entryFullName.EndsWith(".nuspec"))
                    {
                        continue;
                    }

                    string entryFullNameSimplified = entryFullName.SimplifyGodotPath();
                    string destFilePath = Path.Combine(destDir, entryFullNameSimplified);
                    bool isDir = entryFullName.EndsWith("/");

                    if (isDir)
                    {
                        Directory.CreateDirectory(destFilePath);
                    }
                    else
                    {
                        Directory.CreateDirectory(Path.GetDirectoryName(destFilePath));
                        entry.ExtractToFile(destFilePath, overwrite: true);
                    }
                }
            }
        }

        /// <summary>
        /// Copies and extracts all the Godot bundled packages to the Godot NuGet fallback folder.
        /// Does nothing if the packages were already copied.
        /// </summary>
        public static void AddBundledPackagesToFallbackFolder(string fallbackFolder)
        {
            GD.Print("Copying Godot Offline Packages...");

            string nupkgsLocation = Path.Combine(GodotSharpDirs.DataEditorToolsDir, "nupkgs");

            void AddPackage(string packageId, string packageVersion)
            {
                string nupkgPath = Path.Combine(nupkgsLocation, $"{packageId}.{packageVersion}.nupkg");
                AddPackageToFallbackFolder(fallbackFolder, nupkgPath, packageId, packageVersion);
            }

            foreach (var (packageId, packageVersion) in PackagesToAdd)
                AddPackage(packageId, packageVersion);
        }

        private static readonly (string packageId, string packageVersion)[] PackagesToAdd =
        {
            ("Godot.NET.Sdk", GeneratedGodotNupkgsVersions.GodotNETSdk),
            ("Godot.SourceGenerators", GeneratedGodotNupkgsVersions.GodotSourceGenerators),
        };
    }
}
