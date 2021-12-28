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
        public string Solution { get; }
        public string Configuration { get; }
        public string? RuntimeIdentifier { get; }
        public string? PublishOutputDir { get; }
        public bool Restore { get; }
        public bool Rebuild { get; }
        public bool OnlyClean { get; }

        // TODO Use List once we have proper serialization
        public Array<string> CustomProperties { get; } = new Array<string>();

        public string LogsDirPath =>
            Path.Combine(GodotSharpDirs.BuildLogsDirs, $"{Solution.MD5Text()}_{Configuration}");

        public override bool Equals(object? obj)
        {
            if (obj is BuildInfo other)
                return other.Solution == Solution &&
                       other.Configuration == Configuration && other.RuntimeIdentifier == RuntimeIdentifier &&
                       other.PublishOutputDir == PublishOutputDir && other.Restore == Restore &&
                       other.Rebuild == Rebuild && other.OnlyClean == OnlyClean &&
                       other.CustomProperties == CustomProperties &&
                       other.LogsDirPath == LogsDirPath;

            return false;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 29) + Solution.GetHashCode();
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

        public BuildInfo(string solution, string configuration, bool restore, bool rebuild, bool onlyClean)
        {
            Solution = solution;
            Configuration = configuration;
            Restore = restore;
            Rebuild = rebuild;
            OnlyClean = onlyClean;
        }

        public BuildInfo(string solution, string configuration, string runtimeIdentifier,
            string publishOutputDir, bool restore, bool rebuild, bool onlyClean)
        {
            Solution = solution;
            Configuration = configuration;
            RuntimeIdentifier = runtimeIdentifier;
            PublishOutputDir = publishOutputDir;
            Restore = restore;
            Rebuild = rebuild;
            OnlyClean = onlyClean;
        }
    }
}
