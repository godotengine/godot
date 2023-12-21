using System;
using System.Linq;

#nullable enable

namespace Godot;

internal class ReflectionUtils
{
    public static Type? FindTypeInLoadedAssemblies(string assemblyName, string typeFullName)
    {
        // TODO: Validate the side effects of this workaround
        return null;
    }
}
