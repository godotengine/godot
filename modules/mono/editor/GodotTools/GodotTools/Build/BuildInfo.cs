using System;
using Godot;
using Godot.Collections;
using GodotTools.Internals;
using Path = System.IO.Path;

#nullable enable

namespace GodotTools.Build
{
    [Serializable]
    public sealed partial class BuildInfo : RefCounted // TODO Remove RefCounted once we have proper serialization
    {
        public string Solution { get; private set; }
        public string Project { get; private set; }
        public string Configuration { get; private set; }
        public string? RuntimeIdentifier { get; private set; }
        public string? PublishOutputDir { get; private set; }
        public bool Restore { get; private set; }
        public bool Rebuild { get; private set; }
        public bool OnlyClean { get; private set; }

        // TODO Use List once we have proper serialization
        public Godot.Collections.Array CustomProperties { get; private set; } = new();

        public string LogsDirPath => GodotSharpDirs.LogsDirPathFor(Solution, Configuration);

        public override bool Equals(object? obj)
        {
            return obj is BuildInfo other &&
                other.Solution == Solution &&
                other.Project == Project &&
                other.Configuration == Configuration && other.RuntimeIdentifier == RuntimeIdentifier &&
                other.PublishOutputDir == PublishOutputDir && other.Restore == Restore &&
                other.Rebuild == Rebuild && other.OnlyClean == OnlyClean &&
                other.CustomProperties == CustomProperties &&
                other.LogsDirPath == LogsDirPath;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 29) + Solution.GetHashCode();
                hash = (hash * 29) + Project.GetHashCode();
                hash = (hash * 29) + Configuration.GetHashCode();
                hash = (hash * 29) + (RuntimeIdentifier?.GetHashCode() ?? 0);
                hash = (hash * 29) + (PublishOutputDir?.GetHashCode() ?? 0);
                hash = (hash * 29) + Restore.GetHashCode();
                hash = (hash * 29) + Rebuild.GetHashCode();
                hash = (hash * 29) + OnlyClean.GetHashCode();
                hash = (hash * 29) + CustomProperties.GetHashCode();
                hash = (hash * 29) + LogsDirPath.GetHashCode();
                return hash;
            }
        }

        // Needed for instantiation from Godot, after reloading assemblies
        private BuildInfo()
        {
            Solution = string.Empty;
            Project = string.Empty;
            Configuration = string.Empty;
        }

        public BuildInfo(string solution, string project, string configuration, bool restore, bool rebuild, bool onlyClean)
        {
            Solution = solution;
            Project = project;
            Configuration = configuration;
            Restore = restore;
            Rebuild = rebuild;
            OnlyClean = onlyClean;
        }

        public BuildInfo(string solution, string project, string configuration, string runtimeIdentifier,
            string publishOutputDir, bool restore, bool rebuild, bool onlyClean)
        {
            Solution = solution;
            Project = project;
            Configuration = configuration;
            RuntimeIdentifier = runtimeIdentifier;
            PublishOutputDir = publishOutputDir;
            Restore = restore;
            Rebuild = rebuild;
            OnlyClean = onlyClean;
        }
    }
}
