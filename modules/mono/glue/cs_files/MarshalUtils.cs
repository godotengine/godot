using System;

namespace Godot
{
    static class MarshalUtils
    {
        static bool IsArrayGenericType(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Array<>);
        }

        static bool IsDictionaryGenericType(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Dictionary<, >);
        }
    }
}
