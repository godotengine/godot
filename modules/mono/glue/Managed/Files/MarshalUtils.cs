using System;
using System.Collections;
using System.Collections.Generic;

namespace Godot
{
    using Array = Godot.Collections.Array;
    using Dictionary = Godot.Collections.Dictionary;

    static class MarshalUtils
    {
        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Godot.Collections.Array{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// <paramref name="type"/> is not a generic type. That is, IsGenericType returns false.
        /// </exception>
        static bool TypeIsGenericArray(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Godot.Collections.Array<>);
        }

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Godot.Collections.Dictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// <paramref name="type"/> is not a generic type. That is, IsGenericType returns false.
        /// </exception>
        static bool TypeIsGenericDictionary(Type type)
        {
            return type.GetGenericTypeDefinition() == typeof(Godot.Collections.Dictionary<,>);
        }

        static void ArrayGetElementType(Type arrayType, out Type elementType)
        {
            elementType = arrayType.GetGenericArguments()[0];
        }

        static void DictionaryGetKeyValueTypes(Type dictionaryType, out Type keyType, out Type valueType)
        {
            var genericArgs = dictionaryType.GetGenericArguments();
            keyType = genericArgs[0];
            valueType = genericArgs[1];
        }

        static bool GenericIEnumerableIsAssignableFromType(Type type)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                return true;

            foreach (var interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType && interfaceType.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                    return true;
            }

            Type baseType = type.BaseType;

            if (baseType == null)
                return false;

            return GenericIEnumerableIsAssignableFromType(baseType);
        }

        static bool GenericIDictionaryIsAssignableFromType(Type type)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IDictionary<,>))
                return true;

            foreach (var interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType && interfaceType.GetGenericTypeDefinition() == typeof(IDictionary<,>))
                    return true;
            }

            Type baseType = type.BaseType;

            if (baseType == null)
                return false;

            return GenericIDictionaryIsAssignableFromType(baseType);
        }

        static bool GenericIEnumerableIsAssignableFromType(Type type, out Type elementType)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IEnumerable<>))
            {
                elementType = type.GetGenericArguments()[0];
                return true;
            }

            foreach (var interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType && interfaceType.GetGenericTypeDefinition() == typeof(IEnumerable<>))
                {
                    elementType = interfaceType.GetGenericArguments()[0];
                    return true;
                }
            }

            Type baseType = type.BaseType;

            if (baseType == null)
            {
                elementType = null;
                return false;
            }

            return GenericIEnumerableIsAssignableFromType(baseType, out elementType);
        }

        static bool GenericIDictionaryIsAssignableFromType(Type type, out Type keyType, out Type valueType)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(IDictionary<,>))
            {
                var genericArgs = type.GetGenericArguments();
                keyType = genericArgs[0];
                valueType = genericArgs[1];
                return true;
            }

            foreach (var interfaceType in type.GetInterfaces())
            {
                if (interfaceType.IsGenericType && interfaceType.GetGenericTypeDefinition() == typeof(IDictionary<,>))
                {
                    var genericArgs = interfaceType.GetGenericArguments();
                    keyType = genericArgs[0];
                    valueType = genericArgs[1];
                    return true;
                }
            }

            Type baseType = type.BaseType;

            if (baseType == null)
            {
                keyType = null;
                valueType = null;
                return false;
            }

            return GenericIDictionaryIsAssignableFromType(baseType, out keyType, out valueType);
        }

        static Type MakeGenericArrayType(Type elemType)
        {
            return typeof(Godot.Collections.Array<>).MakeGenericType(elemType);
        }

        static Type MakeGenericDictionaryType(Type keyType, Type valueType)
        {
            return typeof(Godot.Collections.Dictionary<,>).MakeGenericType(keyType, valueType);
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

        internal static void GenericIDictionaryToDictionary(object dictionary, IntPtr godotDictionaryPtr)
        {
#if DEBUG
            if (!GenericIDictionaryIsAssignableFromType(dictionary.GetType()))
                throw new InvalidOperationException("The type does not implement IDictionary<,>");
#endif

            // TODO: Can we optimize this?

            var keys = ((IEnumerable)dictionary.GetType().GetProperty("Keys").GetValue(dictionary)).GetEnumerator();
            var values = ((IEnumerable)dictionary.GetType().GetProperty("Values").GetValue(dictionary)).GetEnumerator();

            while (keys.MoveNext() && values.MoveNext())
            {
                object key = keys.Current;
                object value = values.Current;

                Dictionary.godot_icall_Dictionary_Add(godotDictionaryPtr, key, value);
            }
        }
    }
}
