using System;
using Godot;
using Godot.Collections;
using GodotTools.Internals;
using Path = System.IO.Path;

namespace GodotTools.Build
{
    [Serializable]
    public sealed class BuildInfo : Reference // TODO Remove Reference once we have proper serialization
    {
        public string Solution { get; }
        public string[] Targets { get; }
        public string Configuration { get; }
        public bool Restore { get; }
        // TODO Use List once we have proper serialization
        public Array<string> CustomProperties { get; } = new Array<string>();

        public string LogsDirPath => Path.Combine(GodotSharpDirs.BuildLogsDirs, $"{Solution.MD5Text()}_{Configuration}");

        public override bool Equals(object obj)
        {
            if (obj is BuildInfo other)
                return other.Solution == Solution && other.Targets == Targets &&
                       other.Configuration == Configuration && other.Restore == Restore &&
                       other.CustomProperties == CustomProperties && other.LogsDirPath == LogsDirPath;

            return false;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = (hash * 29) + Solution.GetHashCode();
                hash = (hash * 29) + Targets.GetHashCode();
                hash = (hash * 29) + Configuration.GetHashCode();
                hash = (hash * 29) + Restore.GetHashCode();
                hash = (hash * 29) + CustomProperties.GetHashCode();
                hash = (hash * 29) + LogsDirPath.GetHashCode();
                return hash;
            }
        }

        private BuildInfo()
        {
        }

        public BuildInfo(string solution, string[] targets, string configuration, bool restore)
        {
            Solution = solution;
            Targets = targets;
            Configuration = configuration;
            Restore = restore;
        }
    }
}
