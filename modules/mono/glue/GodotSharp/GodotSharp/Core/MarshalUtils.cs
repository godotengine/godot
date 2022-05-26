using System;
using System.Collections.Generic;

namespace Godot
{
    internal static class MarshalUtils
    {
        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="Collections.Array{T}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsGenericArray(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Collections.Array<>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="Collections.Dictionary{TKey, TValue}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Collections.Dictionary<,>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="List{T}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsSystemGenericList(Type type) =>
            type.GetGenericTypeDefinition() == typeof(List<>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="Dictionary{TKey, TValue}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsSystemGenericDictionary(Type type) =>
            type.GetGenericTypeDefinition() == typeof(Dictionary<,>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="IEnumerable{T}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsGenericIEnumerable(Type type) => type.GetGenericTypeDefinition() == typeof(IEnumerable<>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="ICollection{T}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static bool TypeIsGenericICollection(Type type) => type.GetGenericTypeDefinition() == typeof(ICollection<>);

        /// <summary>
        /// 如果 <paramref name="type"/> 的泛型类型定义返回 <see langword="true"/>
        /// 是 <see cref="IDictionary{TKey, TValue}"/>; 否则返回 <see langword="false"/>。
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="type"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
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
        /// 获取给定 <paramref name="arrayType"/> 的元素类型。
        /// </summary>
        /// <param name="arrayType">泛型数组的类型。</param>
        /// <param name="elementType">泛型数组的元素类型。</param>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="arrayType"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static void ArrayGetElementType(Type arrayType, out Type elementType)
        {
            elementType = arrayType.GetGenericArguments()[0];
        }

        /// <summary>
        /// 获取给定 <paramref name="dictionaryType"/> 的键类型和值类型。
        /// </summary>
        /// <param name="dictionaryType">通用字典的类型。</param>
        /// <param name="keyType">通用字典的键类型。</param>
        /// <param name="valueType">通用字典的值类型。</param>
        /// <exception cref="InvalidOperationException">
        /// 当给定的 <paramref name="dictionaryType"/> 不是泛型类型时抛出。
        /// 即 <see cref="Type.IsGenericType"/> 返回 <see langword="false"/>。
        /// </exception>
        private static void DictionaryGetKeyValueTypes(Type dictionaryType, out Type keyType, out Type valueType)
        {
            var genericArgs = dictionaryType.GetGenericArguments();
            keyType = genericArgs[0];
            valueType = genericArgs[1];
        }

        /// <summary>
        /// 从 <see cref="Collections.Array{T}"/> 构造一个新的 <see cref="Type"/>
        /// 其中元素的通用类型是 <paramref name="elemType"/>。
        /// </summary>
        /// <param name="elemType">数组的元素类型。</param>
        /// <returns>具有指定元素类型的泛型数组类型。</returns>
        private static Type MakeGenericArrayType(Type elemType)
        {
            return typeof(Collections.Array<>).MakeGenericType(elemType);
        }

        /// <summary>
        /// 从 <see cref="Collections.Dictionary{TKey, TValue}"/> 构造一个新的 <see cref="Type"/>
        /// 其中键的通用类型是 <paramref name="keyType"/> 和
        /// 值是 <paramref name="valueType"/>。
        /// </summary>
        /// <param name="keyType">字典的键类型。</param>
        /// <param name="valueType">字典的键类型。</param>
        /// <returns>具有指定键值类型的通用字典类型。</returns>
        private static Type MakeGenericDictionaryType(Type keyType, Type valueType)
        {
            return typeof(Collections.Dictionary<,>).MakeGenericType(keyType, valueType);
        }
    }
}
