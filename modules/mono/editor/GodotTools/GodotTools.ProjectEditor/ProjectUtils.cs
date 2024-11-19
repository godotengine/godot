using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Build.Construction;
using Microsoft.Build.Locator;

namespace GodotTools.ProjectEditor
{
    public sealed class MSBuildProject
    {
        internal ProjectRootElement Root { get; set; }

        public bool HasUnsavedChanges { get; set; }

        public void Save() => Root.Save();

        public MSBuildProject(ProjectRootElement root)
        {
            Root = root;
        }
    }

    public static class ProjectUtils
    {
        private struct PlatformFramework
        {
            public string Platform { get; init; }
            public string TargetFramework { get; init; }
        }

        private const string MinimumTargetFramework = "net6.0";

        private static readonly List<PlatformFramework> _platformsFrameworks = new()
        {
            new PlatformFramework { Platform = "ios", TargetFramework = "net8.0" },
            new PlatformFramework { Platform = "android", TargetFramework = "net7.0" },
        };

        public static void MSBuildLocatorRegisterLatest(out Version version, out string path)
        {
            var instance = MSBuildLocator.QueryVisualStudioInstances()
                .OrderByDescending(x => x.Version)
                .First();
            MSBuildLocator.RegisterInstance(instance);
            version = instance.Version;
            path = instance.MSBuildPath;
        }

        public static void MSBuildLocatorRegisterMSBuildPath(string msbuildPath)
            => MSBuildLocator.RegisterMSBuildPath(msbuildPath);

        public static MSBuildProject? Open(string path)
        {
            var root = ProjectRootElement.Open(path);
            return root != null ? new MSBuildProject(root) : null;
        }

        public static void MigrateToProjectSdksStyle(MSBuildProject project, string projectName)
        {
            var origRoot = project.Root;

            if (!string.IsNullOrEmpty(origRoot.Sdk))
                return;

            project.Root = ProjectGenerator.GenGameProject(projectName);
            project.Root.FullPath = origRoot.FullPath;
            project.HasUnsavedChanges = true;
        }

        public static void EnsureGodotSdkIsUpToDate(MSBuildProject project)
        {
            var root = project.Root;
            string godotSdkAttrValue = ProjectGenerator.GodotSdkAttrValue;

            if (!string.IsNullOrEmpty(root.Sdk) &&
                root.Sdk.Trim().Equals(godotSdkAttrValue, StringComparison.OrdinalIgnoreCase))
                return;

            root.Sdk = godotSdkAttrValue;
            project.HasUnsavedChanges = true;
        }

        public static void EnsureTargetFrameworkIsSet(MSBuildProject project)
        {
            bool HasNoCondition(ProjectElement element) => string.IsNullOrEmpty(element.Condition);

            bool ConditionMatches(ProjectElement element, string platform) =>
                element.Condition.Trim().Replace(" ", "").Equals($"'$(GodotTargetPlatform)'=='{platform}'", StringComparison.OrdinalIgnoreCase);

            // if the existing framework is equal or higher than what we need, we're good.
            bool IsVersionUsable(string theirs, string ours) =>
                Version.TryParse(theirs[3..], out var versionTheirs) &&
                Version.TryParse(ours[3..], out var versionOurs) &&
                versionTheirs >= versionOurs;

            // if the property already has our condition, we're good.
            // if the property has no conditions and the group has no conditions, we can use it.
            // if the property has no conditions but the group matches our condition, we're cool.
            bool IsFrameworkUsable(ProjectPropertyGroupElement group, ProjectPropertyElement prop,
                string platform, string framework)
            {
                return IsVersionUsable(prop.Value, framework) &&
                       (ConditionMatches(prop, platform) ||
                        (HasNoCondition(prop) &&
                         (HasNoCondition(group) || ConditionMatches(group, platform))));
            }

            // If the condition matches (in the property or group) but the target framework isn't high enough, replace it.
            bool ShouldReplaceProperty(ProjectPropertyGroupElement group, ProjectPropertyElement prop,
                string platform, string framework)
            {
                return !IsVersionUsable(prop.Value, framework) &&
                       (ConditionMatches(prop, platform) ||
                        (HasNoCondition(prop) && ConditionMatches(group, platform)));
            }

            ProjectPropertyGroupElement? mainGroup = null;
            Dictionary<ProjectPropertyGroupElement, List<ProjectPropertyElement>> propertiesToRemove = new();

            var root = project.Root;

            // We'll go through all the platforms that need specific target framework versions, based
            // on the configuration set at the top, and check if there is a TargetFramework property that
            // covers the specific platform and is high enough to match the platform requirement.
            //
            // The property could have no conditions (matching all platforms)
            // <PropertyGroup>
            //   <TargetFramework>net8.0</TargetFramework>
            // </PropertyGroup>
            //
            // or could have a specific condition matching one or more platforms
            // <PropertyGroup>
            //   <TargetFramework Condition="'$(Platform)' == 'ios' or '$(Platform)' == 'iossimulator'">net8.0</TargetFramework>
            // </PropertyGroup>
            //
            // or could be part of a group that has the right conditions.
            // <PropertyGroup Condition="'$(Platform)' == 'ios'>
            //   <TargetFramework>net8.0</TargetFramework>
            // </PropertyGroup>
            //
            // Any such property that we find that matches the platform, we check whether the framework version is equal or higher
            // than what the platform requires, and if yes, it is added to this list. At the end, any platforms not on this list
            // will get a new TargetFramework property added to the first PropertyGroup with existing TargetFramework properties,
            // so they're all together as much as possible.
            var platformsAlreadySupported = new List<PlatformFramework>();

            // If the user sets the GodotSkipAutomaticTargetFrameworkUpdate property, we don't do anything.
            if (root.PropertyGroups.SelectMany(x => x.Properties)
                .Any(x =>
                    x.Name.Equals("GodotSkipAutomaticTargetFrameworkUpdate", StringComparison.OrdinalIgnoreCase) &&
                    (x.Value?.Equals("true", StringComparison.OrdinalIgnoreCase) ?? false)))
            {
                return;
            }

            var defaultTargetFramework = root.PropertyGroups
                    .Where(HasNoCondition)
                    .SelectMany(group => group.Properties)
                    .FirstOrDefault(prop => prop.Name.Equals("TargetFramework", StringComparison.OrdinalIgnoreCase) && HasNoCondition(prop));

            if (defaultTargetFramework == null)
            {
                mainGroup = root.PropertyGroups.FirstOrDefault(HasNoCondition);
                mainGroup ??= root.AddPropertyGroup();
                mainGroup.AddProperty("TargetFramework", MinimumTargetFramework);
                project.HasUnsavedChanges = true;
            }
            else
            {
                // if we need to add the property, we'll add it on the first conditionless property group that
                // has a TargetFramework, just so everything is close together
                mainGroup = (ProjectPropertyGroupElement)defaultTargetFramework.Parent;
                if (!IsVersionUsable(defaultTargetFramework.Value, MinimumTargetFramework))
                {
                    // we should have already found the first property group that has a TargetFramework, but if not...
                    defaultTargetFramework.Value = MinimumTargetFramework;
                    project.HasUnsavedChanges = true;
                }
            }

            foreach (var prop in root.PropertyGroups
                         .SelectMany(group => group.Properties)
                         .Where(prop => prop.Name.Equals("TargetFramework", StringComparison.OrdinalIgnoreCase)))
            {
                var group = (ProjectPropertyGroupElement)prop.Parent;
                foreach (var pf in _platformsFrameworks)
                {
                    if (ShouldReplaceProperty(group, prop, pf.Platform, pf.TargetFramework))
                    {
                        if (!propertiesToRemove.ContainsKey(group))
                            propertiesToRemove.Add(group, new List<ProjectPropertyElement>());

                        propertiesToRemove[group].Add(prop);
                    }

                    if (!platformsAlreadySupported.Contains(pf) &&
                        IsFrameworkUsable(group, prop, pf.Platform, pf.TargetFramework))
                    {
                        platformsAlreadySupported.Add(pf);
                    }
                }
            }

            foreach (var toRemove in propertiesToRemove.SelectMany(group => group.Value.Select(prop => (group.Key, prop))))
            {
                toRemove.Key.RemoveChild(toRemove.prop);
            }

            foreach (var pf in _platformsFrameworks.Where(x => !platformsAlreadySupported.Contains(x)))
            {
                // we should have already found the first property group that has a TargetFramework, but if not...
                mainGroup ??= root.AddPropertyGroup();
                var prop = mainGroup.AddProperty("TargetFramework", pf.TargetFramework);
                prop.Condition = $" '$(GodotTargetPlatform)' == '{pf.Platform}' ";
                project.HasUnsavedChanges = true;
            }
        }
    }
}
