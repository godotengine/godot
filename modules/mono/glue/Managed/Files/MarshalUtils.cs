using System;
using System.Collections;

namespace Godot
{
    using Array = Godot.Collections.Array;
    using Dictionary = Godot.Collections.Dictionary;

    static class MarshalUtils
    {
        static bool TypeIsGenericArray(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Godot.Collections.Array<>);
        }

        static bool TypeIsGenericDictionary(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Godot.Collections.Dictionary<,>);
        }

        static void ArrayGetElementType(Type type, out Type elementType)
        {
            elementType = type.GetGenericArguments()[0];
        }

        static void DictionaryGetKeyValueTypes(Type type, out Type keyType, out Type valueType)
        {
            var genericArgs = type.GetGenericArguments();

            keyType = genericArgs[0];
            valueType = genericArgs[1];
        }

        // TODO Add support for IEnumerable<T> and IDictionary<TKey, TValue>
        // TODO: EnumerableToArray and IDictionaryToDictionary can be optimized

        internal static void EnumerableToArray(IEnumerable enumerable, IntPtr godotArrayPtr)
        {
            if (enumerable is ICollection collection)
            {
                int count = collection.Count;

                object[] tempArray = new object[count];
                collection.CopyTo(tempArray, 0);

                for (int i = 0; i < count; i++)
                {
                    Array.godot_icall_Array_Add(godotArrayPtr, tempArray[i]);
                }
            }
            else
            {
                foreach (object element in enumerable)
                {
                    Array.godot_icall_Array_Add(godotArrayPtr, element);
                }
            }
        }

        internal static void IDictionaryToDictionary(IDictionary dictionary, IntPtr godotDictionaryPtr)
        {
            foreach (DictionaryEntry entry in dictionary)
            {
                Dictionary.godot_icall_Dictionary_Add(godotDictionaryPtr, entry.Key, entry.Value);
            }
        }
    }
}
