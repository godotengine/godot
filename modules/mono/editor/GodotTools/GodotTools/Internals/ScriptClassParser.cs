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

            public string SearchName => Nested ?
                Name.Substring(Name.LastIndexOf(".", StringComparison.Ordinal) + 1) :
                Name;

            public ClassDecl(string name, string @namespace, bool nested, int baseCount)
            {
                Name = name;
                Namespace = @namespace;
                Nested = nested;
                BaseCount = baseCount;
            }
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        private static extern Error internal_ParseFile(string filePath, Array<Dictionary> classes, out string errorStr);

        public static Error ParseFile(string filePath, out IEnumerable<ClassDecl> classes, out string errorStr)
        {
            var classesArray = new Array<Dictionary>();
            var error = internal_ParseFile(filePath, classesArray, out errorStr);
            if (error != Error.Ok)
            {
                classes = null;
                return error;
            }

            var classesList = new List<ClassDecl>();

            foreach (var classDeclDict in classesArray)
            {
                classesList.Add(new ClassDecl(
                    (string)classDeclDict["name"],
                    (string)classDeclDict["namespace"],
                    (bool)classDeclDict["nested"],
                    (int)classDeclDict["base_count"]
                ));
            }

            classes = classesList;

            return Error.Ok;
        }
    }
}
