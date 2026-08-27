#pragma warning disable CS1591
#nullable enable

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;

namespace Godot;

[EditorBrowsable(EditorBrowsableState.Never)]
public static class AssemblyUtils
{
    const string CORE_ASSEMBLY = "GodotSharp";
    const string EDITOR_ASSEMBLY = "GodotSharpEditor";

    public static void RemoveNonGodotSharpTypes<TValue>(Dictionary<Type, TValue> dictionary)
    {
        foreach (var ty in dictionary.Keys.ToArray())
        {
            string tyAssembly = ty.Assembly.GetName().Name!;
            if (tyAssembly is not (CORE_ASSEMBLY or EDITOR_ASSEMBLY))
            {
                dictionary.Remove(ty);
            }
        }
    }

    public static void RemoveNonGodotSharpTypes<TKey>(Dictionary<TKey, Type> dictionary) where TKey: notnull
    {
        foreach (var (key, ty) in dictionary.ToArray())
        {
            string tyAssembly = ty.Assembly.GetName().Name!;
            if (tyAssembly is not (CORE_ASSEMBLY or EDITOR_ASSEMBLY))
            {
                dictionary.Remove(key);
            }
        }
    }
}
