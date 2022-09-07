using System;
using System.Globalization;
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

        /// <summary>
        /// Returns all the paths where the Godot.Offline.Config files can be found.
        /// Does not determine whether the returned files exist or not.
        /// </summary>
        private static string[] GetAllGodotNuGetConfigFilePaths()
        {
            // Where to find 'NuGet/config/Godot.Offline.Config':
            //
            // - Mono/.NETFramework (standalone NuGet):
            //     Uses Environment.SpecialFolder.ApplicationData
            //     - Windows: '%APPDATA%'
            //     - Linux/macOS: '$HOME/.config'
            // - CoreCLR (dotnet CLI NuGet):
            //     - Windows: '%APPDATA%'
            //     - Linux/macOS: '$DOTNET_CLI_HOME/.nuget' otherwise '$HOME/.nuget'

            string applicationData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);

            const string configFileName = "Godot.Offline.Config";

            if (Utils.OS.IsWindows)
            {
                // %APPDATA% for both
                return new[] { Path.Combine(applicationData, "NuGet", "config", configFileName) };
            }

            var paths = new string[2];

            // CoreCLR (dotnet CLI NuGet)

            string dotnetCliHome = Environment.GetEnvironmentVariable("DOTNET_CLI_HOME");
            if (!string.IsNullOrEmpty(dotnetCliHome))
            {
                paths[0] = Path.Combine(dotnetCliHome, ".nuget", "NuGet", "config", configFileName);
            }
            else
            {
                string home = Environment.GetEnvironmentVariable("HOME");
                if (string.IsNullOrEmpty(home))
                    throw new InvalidOperationException("Required environment variable 'HOME' is not set.");
                paths[0] = Path.Combine(home, ".nuget", "NuGet", "config", configFileName);
            }

            // Mono/.NETFramework (standalone NuGet)

            // ApplicationData is $HOME/.config on Linux/macOS
            paths[1] = Path.Combine(applicationData, "NuGet", "config", configFileName);

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
        /// Adds the specified fallback folder to the Godot.Offline.Config files,
        /// for both standalone NuGet (Mono/.NETFramework) and dotnet CLI NuGet.
        /// </summary>
        public static void AddFallbackFolderToGodotNuGetConfigs(string name, string path)
        {
            // Make sure the fallback folder exists to avoid error:
            // MSB4018: The "ResolvePackageAssets" task failed unexpectedly.
            System.IO.Directory.CreateDirectory(path);

            foreach (string nuGetConfigPath in GetAllGodotNuGetConfigFilePaths())
            {
                string defaultConfig = @$"<?xml version=""1.0"" encoding=""utf-8""?>
<configuration>
  <fallbackPackageFolders>
    <add key=""{name}"" value=""{path}"" />
  </fallbackPackageFolders>
</configuration>
";
                System.IO.Directory.CreateDirectory(Path.GetDirectoryName(nuGetConfigPath));
                System.IO.File.WriteAllText(nuGetConfigPath, defaultConfig, Encoding.UTF8); // UTF-8 with BOM
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

            string packageIdLower = packageId.ToLowerInvariant();
            string packageVersionLower = packageVersion.ToLowerInvariant();

            string destDir = Path.Combine(fallbackFolder, packageIdLower, packageVersionLower);
            string nupkgDestPath = Path.Combine(destDir, $"{packageIdLower}.{packageVersionLower}.nupkg");
            string nupkgSha512DestPath = Path.Combine(destDir, $"{packageIdLower}.{packageVersionLower}.nupkg.sha512");
            string nupkgMetadataDestPath = Path.Combine(destDir, ".nupkg.metadata");

            if (File.Exists(nupkgDestPath) && File.Exists(nupkgSha512DestPath))
                return; // Already added (for speed we don't check if every file is properly extracted)

            Directory.CreateDirectory(destDir);

            // Generate .nupkg.sha512 file

            byte[] hash = SHA512.HashData(File.ReadAllBytes(nupkgPath));
            string base64Hash = Convert.ToBase64String(hash);
            File.WriteAllText(nupkgSha512DestPath, base64Hash);

            // Generate .nupkg.metadata file
            // Spec: https://github.com/NuGet/Home/wiki/Nupkg-Metadata-File

            File.WriteAllText(nupkgMetadataDestPath, @$"{{
    ""version"": 2,
    ""contentHash"": ""{base64Hash}"",
    ""source"": null
}}");

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
                    throw new InvalidOperationException(
                        $"Failed to extract package {packageId}.{packageVersion}. Could not find the nuspec file.");

                nuspecEntry.ExtractToFile(Path.Combine(destDir, nuspecEntry.Name
                    .ToLowerInvariant().SimplifyGodotPath()));

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
                        !entryFullName.Contains('/') && entryFullName.EndsWith(".nuspec"))
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
            ("GodotSharp", GeneratedGodotNupkgsVersions.GodotSharp),
            ("GodotSharpEditor", GeneratedGodotNupkgsVersions.GodotSharp),
        };
    }
}
