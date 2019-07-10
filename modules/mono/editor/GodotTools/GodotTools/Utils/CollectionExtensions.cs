using System;
using System.Collections.Generic;

namespace GodotTools.Utils
{
    public static class CollectionExtensions
    {
        public static T SelectFirstNotNull<T>(this IEnumerable<T> enumerable, Func<T, T> predicate, T orElse = null)
            where T : class
        {
            foreach (T elem in enumerable)
            {
                if (predicate(elem) != null)
                    return elem;
            }

            return orElse;
        }
    }
}
