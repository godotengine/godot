using System.Collections.Generic;

namespace Godot.SourceGenerators
{
    internal readonly struct MethodInfo
    {
        public MethodInfo(string name, PropertyInfo returnVal, MethodFlags flags,
            List<PropertyInfo>? arguments,
            List<string?>? defaultArguments)
        {
            Name = name;
            ReturnVal = returnVal;
            Flags = flags;
            Arguments = arguments;
            DefaultArguments = defaultArguments;
        }

        public string Name { get; }
        public PropertyInfo ReturnVal { get; }
        public MethodFlags Flags { get; }
        public List<PropertyInfo>? Arguments { get; }
        public List<string?>? DefaultArguments { get; }
    }
}
