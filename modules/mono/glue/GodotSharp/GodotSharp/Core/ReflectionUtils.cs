using System;
using System.Linq;

#nullable enable

namespace Godot;

internal class ReflectionUtils
{
    public static Type? FindTypeInLoadedAssemblies(string assemblyName, string typeFullName)
    {
        return Type.GetType($"{assemblyName}.{typeFullName}");
    }
}
