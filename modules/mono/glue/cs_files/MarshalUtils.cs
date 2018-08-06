using System;

namespace Godot
{
    static class MarshalUtils
    {
        static bool IsArrayGenericType(Type type)
        {
            var ret = new Dictionary<object, object>();

            for (int i = 0; i < keys.Length; i++)
            {
                ret.Add(keys[i], values[i]);
            }

            return ret;
        }

        private static void DictionaryToArrays(Dictionary<object, object> from, out object[] keysTo, out object[] valuesTo)
        {
            var keys = from.Keys;
            keysTo = new object[keys.Count];
            keys.CopyTo(keysTo, 0);

            var values = from.Values;
            valuesTo = new object[values.Count];
            values.CopyTo(valuesTo, 0);
        }

        static bool IsDictionaryGenericType(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Dictionary<, >);
        }
    }
}
