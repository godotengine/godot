using System;
using System.Collections.Generic;

namespace Godot
{
    internal static class MarshalUtils
    {
        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Godot.Collections.Array{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// <paramref name="type"/> is not a generic type. That is, IsGenericType returns false.
        /// </exception>
        private static bool TypeIsGenericArray(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Godot.Collections.Array<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Godot.Collections.Dictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="System.InvalidOperationException">
        /// <paramref name="type"/> is not a generic type. That is, IsGenericType returns false.
        /// </exception>
        private static bool TypeIsGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Godot.Collections.Dictionary<,>);

        private static bool TypeIsSystemGenericList(Type type) =>
            type.GetGenericTypeDefinition() == typeof(System.Collections.Generic.List<>);

        private static bool TypeIsSystemGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(System.Collections.Generic.Dictionary<,>);

        private static bool TypeIsGenericIEnumerable(Type type) => type.GetGenericTypeDefinition() == typeof(IEnumerable<>);

        private static bool TypeIsGenericICollection(Type type) => type.GetGenericTypeDefinition() == typeof(ICollection<>);

        private static bool TypeIsGenericIDictionary(Type type) => type.GetGenericTypeDefinition() == typeof(IDictionary<,>);

        private static void ArrayGetElementType(Type arrayType, out Type elementType)
        {
            elementType = arrayType.GetGenericArguments()[0];
        }

        private static void DictionaryGetKeyValueTypes(Type dictionaryType, out Type keyType, out Type valueType)
        {
            var genericArgs = dictionaryType.GetGenericArguments();
            keyType = genericArgs[0];
            valueType = genericArgs[1];
        }

        private static Type MakeGenericArrayType(Type elemType)
        {
            return typeof(Godot.Collections.Array<>).MakeGenericType(elemType);
        }

        private static Type MakeGenericDictionaryType(Type keyType, Type valueType)
        {
            return typeof(Godot.Collections.Dictionary<,>).MakeGenericType(keyType, valueType);
        }
    }
}
