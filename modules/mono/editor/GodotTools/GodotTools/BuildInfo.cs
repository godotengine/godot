using System;
using Godot;
using Godot.Collections;
using GodotTools.Internals;
using Path = System.IO.Path;

namespace GodotTools
{
    [Serializable]
    public sealed class BuildInfo : Reference // TODO Remove Reference once we have proper serialization
    {
        public string Solution { get; }
        public string Configuration { get; }
        public Array<string> CustomProperties { get; } = new Array<string>(); // TODO Use List once we have proper serialization

        public string LogsDirPath => Path.Combine(GodotSharpDirs.BuildLogsDirs, $"{Solution.MD5Text()}_{Configuration}");

        public override bool Equals(object obj)
        {
            if (obj is BuildInfo other)
                return other.Solution == Solution && other.Configuration == Configuration;

            return false;
        }

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 29 + Solution.GetHashCode();
                hash = hash * 29 + Configuration.GetHashCode();
                return hash;
            }
        }

        private BuildInfo()
        {
        }

        public BuildInfo(string solution, string configuration)
        {
            Solution = solution;
            Configuration = configuration;
        }
    }
}
