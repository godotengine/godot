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
        static bool TypeIsGenericArray(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Godot.Collections.Array<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Godot.Collections.Dictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// <paramref name="type"/> is not a generic type. That is, IsGenericType returns false.
        /// </exception>
        static bool TypeIsGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Godot.Collections.Dictionary<,>);

        static bool TypeIsSystemGenericList(Type type) =>
            type.GetGenericTypeDefinition() == typeof(System.Collections.Generic.List<>);

        static bool TypeIsSystemGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(System.Collections.Generic.Dictionary<,>);

        static bool TypeIsGenericIEnumerable(Type type) => type.GetGenericTypeDefinition() == typeof(IEnumerable<>);

        static bool TypeIsGenericICollection(Type type) => type.GetGenericTypeDefinition() == typeof(ICollection<>);

        static bool TypeIsGenericIDictionary(Type type) => type.GetGenericTypeDefinition() == typeof(IDictionary<,>);

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

        static Type MakeGenericArrayType(Type elemType)
        {
            return typeof(Godot.Collections.Array<>).MakeGenericType(elemType);
        }

        static Type MakeGenericDictionaryType(Type keyType, Type valueType)
        {
            return typeof(Godot.Collections.Dictionary<,>).MakeGenericType(keyType, valueType);
        }
    }
}
