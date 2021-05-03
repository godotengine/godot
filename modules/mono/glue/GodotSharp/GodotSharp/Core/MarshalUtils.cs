using System;
using System.Collections.Generic;

namespace Godot
{
    internal static class MarshalUtils
    {
        /// <summary>
        /// Returns <see langword="true"/> if the <see cref="FlagsAttribute"/> is applied to the given type.
        /// </summary>
        private static bool TypeHasFlagsAttribute(Type type) => type.IsDefined(typeof(FlagsAttribute), false);
    }
}
