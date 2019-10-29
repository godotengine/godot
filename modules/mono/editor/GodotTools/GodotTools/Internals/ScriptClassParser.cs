using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Godot;
using Godot.Collections;

namespace GodotTools.Internals
{
    public static class ScriptClassParser
    {
        public class ClassDecl
        {
            public string Name { get; }
            public string Namespace { get; }
            public bool Nested { get; }
            public int BaseCount { get; }

            public ClassDecl(string name, string @namespace, bool nested, int baseCount)
            {
                Name = name;
                Namespace = @namespace;
                Nested = nested;
                BaseCount = baseCount;
            }
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Error internal_ParseFile(string filePath, Array<Dictionary> classes);

        public static void ParseFileOrThrow(string filePath, out IEnumerable<ClassDecl> classes)
        {
            var classesArray = new Array<Dictionary>();
            var error = internal_ParseFile(filePath, classesArray);
            if (error != Error.Ok)
                throw new Exception($"Failed to determine namespace and class for script: {filePath}. Parse error: {error}");

            var classesList = new List<ClassDecl>();

            foreach (var classDeclDict in classesArray)
            {
                classesList.Add(new ClassDecl(
                    (string) classDeclDict["name"],
                    (string) classDeclDict["namespace"],
                    (bool) classDeclDict["nested"],
                    (int) classDeclDict["base_count"]
                ));
            }

            classes = classesList;
        }
    }
}
