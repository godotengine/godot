using System;
using System.Collections.Generic;
using System.IO;

namespace GodotTools.Utils
{
    public static class CollectionExtensions
    {
        public static T SelectFirstNotNull<T>(this IEnumerable<T> enumerable, Func<T, T> predicate, T orElse = null)
            where T : class
        {
            foreach (T elem in enumerable)
            {
                T result = predicate(elem);
                if (result != null)
                    return result;
            }

            return orElse;
        }

        public static IEnumerable<string> EnumerateLines(this TextReader textReader)
        {
            string line;
            while ((line = textReader.ReadLine()) != null)
                yield return line;
        }
    }
}
