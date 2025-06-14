#nullable enable

using System.Collections.Generic;

namespace Godot;

/// <summary>
/// Extension methods to manipulate arrays and lists.
/// </summary>
public static class ListExtensions
{
    /// <summary>
    /// Assigns the given value to all elements in the array.
    /// </summary>
    public static void Fill<T>(this IList<T> list, T value)
    {
        for (int i = 0; i < list.Count; i++)
            list[i] = value;
    }

    /// <summary>
    /// Returns a random value from the target array.
    /// Note that this uses <see cref="GD.RandRange(int, int)"/> to advance the Godot global seed.
    /// </summary>
    /// <returns>A random element from the array, or <see langword="null"/> if the array is empty.</returns>
    public static T? PickRandom<T>(this IList<T> list)
    {
        if (list.Count == 0)
            return default;
        if (list.Count == 1)
            return list[0];

        return list[GD.RandRange(0, list.Count - 1)];
    }
}
