using System;
using System.Collections.Generic;

namespace Godot
{
    internal static class MarshalUtils
    {
        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Collections.Array{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsGenericArray(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Collections.Array<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Collections.Dictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Collections.Dictionary<,>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="List{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsSystemGenericList(Type type) =>
            type.GetGenericTypeDefinition() == typeof(List<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="Dictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsSystemGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Dictionary<,>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="IEnumerable{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsGenericIEnumerable(Type type) => type.GetGenericTypeDefinition() == typeof(IEnumerable<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="ICollection{T}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsGenericICollection(Type type) => type.GetGenericTypeDefinition() == typeof(ICollection<>);

        /// <summary>
        /// Returns <see langword="true"/> if the generic type definition of <paramref name="type"/>
        /// is <see cref="IDictionary{TKey, TValue}"/>; otherwise returns <see langword="false"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static bool TypeIsGenericIDictionary(Type type) => type.GetGenericTypeDefinition() == typeof(IDictionary<,>);

        /// <summary>
        /// Returns the generic type definition of <paramref name="type"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="type"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static void GetGenericTypeDefinition(Type type, out Type genericTypeDefinition)
        {
            genericTypeDefinition = type.GetGenericTypeDefinition();
        }

        /// <summary>
        /// Gets the element type for the given <paramref name="arrayType"/>.
        /// </summary>
        /// <param name="arrayType">Type for the generic array.</param>
        /// <param name="elementType">Element type for the generic array.</param>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="arrayType"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static void ArrayGetElementType(Type arrayType, out Type elementType)
        {
            elementType = arrayType.GetGenericArguments()[0];
        }

        /// <summary>
        /// Gets the key type and the value type for the given <paramref name="dictionaryType"/>.
        /// </summary>
        /// <param name="dictionaryType">The type for the generic dictionary.</param>
        /// <param name="keyType">Key type for the generic dictionary.</param>
        /// <param name="valueType">Value type for the generic dictionary.</param>
        /// <exception cref="InvalidOperationException">
        /// Thrown when the given <paramref name="dictionaryType"/> is not a generic type.
        /// That is, <see cref="Type.IsGenericType"/> returns <see langword="false"/>.
        /// </exception>
        private static void DictionaryGetKeyValueTypes(Type dictionaryType, out Type keyType, out Type valueType)
        {
            var genericArgs = dictionaryType.GetGenericArguments();
            keyType = genericArgs[0];
            valueType = genericArgs[1];
        }

        /// <summary>
        /// Constructs a new <see cref="Type"/> from <see cref="Collections.Array{T}"/>
        /// where the generic type for the elements is <paramref name="elemType"/>.
        /// </summary>
        /// <param name="elemType">Element type for the array.</param>
        /// <returns>The generic array type with the specified element type.</returns>
        private static Type MakeGenericArrayType(Type elemType)
        {
            return typeof(Collections.Array<>).MakeGenericType(elemType);
        }

        /// <summary>
        /// Constructs a new <see cref="Type"/> from <see cref="Collections.Dictionary{TKey, TValue}"/>
        /// where the generic type for the keys is <paramref name="keyType"/> and
        /// for the values is <paramref name="valueType"/>.
        /// </summary>
        /// <param name="keyType">Key type for the dictionary.</param>
        /// <param name="valueType">Key type for the dictionary.</param>
        /// <returns>The generic dictionary type with the specified key and value types.</returns>
        private static Type MakeGenericDictionaryType(Type keyType, Type valueType)
        {
            return typeof(Collections.Dictionary<,>).MakeGenericType(keyType, valueType);
        }
    }
}
