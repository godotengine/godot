

using System.Collections.Concurrent;
using Godot.Collections;
using Godot.NativeInterop;

#nullable enable

namespace Godot;

/// <summary>
/// Object pools instance for Array, Dictionary, StringName and NodePath. Internally, they are used for all Godot methods that return the corresponding object.
/// <para>You can return objects that are no longer used to the pools to reuse them, which reduces the pressure on the garbage collector (GC) and improves performance.</para>
/// </summary>
public static class Pools
{
    /// <summary>
    /// The object pool instance for <see cref="Array"/>
    /// </summary>
    public static readonly ArrayPool ArrayPoolInstance = ArrayPool.Instance;

    /// <summary>
    /// The object pool instance for <see cref="Dictionary"/>
    /// </summary>
    public static readonly DictionaryPool DictionaryPoolInstance = DictionaryPool.Instance;

    /// <summary>
    /// The object pool instance for <see cref="StringName"/>
    /// </summary>
    public static readonly StringNamePool StringNamePoolInstance = StringNamePool.Instance;

    /// <summary>
    /// The object pool instance for <see cref="NodePath"/>
    /// </summary>
    public static readonly NodePathPool NodePathPoolInstance = NodePathPool.Instance;

    /// <summary>
    /// The object pool for <see cref="Array"/>
    /// </summary>
    public sealed class ArrayPool
    {
        internal static ArrayPool Instance = new();

        private ArrayPool() { }

        private readonly ConcurrentBag<Array> _objects = [];

        internal Array Rent(godot_array nativeValueToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item.TakeOwnershipOfDisposableValue(nativeValueToOwn);
                    return item;
                }
            }
            return Array.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="Array"/> object.</param>
        public void Return(Array obj)
        {
            obj.Dispose();
            _objects.Add(obj);
        }
    }


    /// <summary>
    /// The object pool for <see cref="Dictionary"/>
    /// </summary>
    public sealed class DictionaryPool
    {
        internal static DictionaryPool Instance = new();

        private DictionaryPool() { }

        private readonly ConcurrentBag<Dictionary> _objects = [];

        internal Dictionary Rent(godot_dictionary nativeValueToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item.TakeOwnershipOfDisposableValue(nativeValueToOwn);
                    return item;
                }
            }
            return Dictionary.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="Dictionary"/> object.</param>
        public void Return(Dictionary obj)
        {
            obj.Dispose();
            _objects.Add(obj);
        }
    }

    /// <summary>
    /// The object pool for <see cref="NodePath"/>
    /// </summary>
    public sealed class NodePathPool
    {
        internal static NodePathPool Instance = new();

        private NodePathPool() { }

        private readonly ConcurrentBag<NodePath> _objects = [];

        internal NodePath Rent(godot_node_path nativeValueToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item.TakeOwnershipOfDisposableValue(nativeValueToOwn);
                    return item;
                }
            }
            return NodePath.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="NodePath"/> object.</param>
        public void Return(NodePath obj)
        {
            obj.Dispose();
            _objects.Add(obj);
        }
    }

    /// <summary>
    /// The object pool for <see cref="StringName"/>
    /// </summary>
    public sealed class StringNamePool
    {
        internal static StringNamePool Instance = new();

        private StringNamePool() { }

        private readonly ConcurrentBag<StringName> _objects = [];

        internal StringName Rent(godot_string_name nativeValueToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item.TakeOwnershipOfDisposableValue(nativeValueToOwn);
                    return item;
                }
            }
            return StringName.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="StringName"/> object.</param>
        public void Return(StringName obj)
        {
            obj.Dispose();
            _objects.Add(obj);
        }
    }

    /// <summary>
    /// The object pool for <see cref="Array{T}"/>
    /// </summary>
    /// <typeparam name="T">The type of the array.</typeparam>
    public class ArrayPool<T>
    {
        /// <summary>
        /// The object pool instance for <see cref="Array{T}"/>
        /// </summary>
        public static readonly ArrayPool<T> Instance = new();

        private ArrayPool() { }

        private readonly ConcurrentBag<Array<T>> _objects = [];

        internal Array<T> Rent(Array underlyingToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item._underlyingArray = underlyingToOwn;
                    return item;
                }
            }
            return new Array<T>(underlyingToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="Array{T}"/> object.</param>
        public void Return(Array<T> obj)
        {
            ArrayPoolInstance.Return(obj._underlyingArray);
            _objects.Add(obj);
        }
    }

    /// <summary>
    /// The object pool for <see cref="Dictionary{TKey, TValue}"/>
    /// </summary>
    /// <typeparam name="TKey">The type of the dictionary's keys.</typeparam>
    /// <typeparam name="TValue">The type of the dictionary's values.</typeparam>
    public class DictionaryPool<TKey, TValue>
    {
        /// <summary>
        /// The object pool instance for <see cref="Dictionary{TKey, TValue}"/>
        /// </summary>
        public static readonly DictionaryPool<TKey, TValue> Instance = new();

        private DictionaryPool() { }

        private readonly ConcurrentBag<Dictionary<TKey, TValue>> _objects = [];

        internal Dictionary<TKey, TValue> Rent(Dictionary underlyingToOwn)
        {
            if (_objects.TryTake(out var item))
            {
                if (item != null)
                {
                    item._underlyingDict = underlyingToOwn;
                    return item;
                }
            }
            return new Dictionary<TKey, TValue>(underlyingToOwn);
        }

        /// <summary>
        /// Returns the <paramref name="obj"/> to the pool to reuse it.
        /// <para><b>Note</b>: After this, the <paramref name="obj"/> will be disposed and should not be accessed again. And it should not return the same object twice.</para>
        /// </summary>
        /// <param name="obj">The returned <see cref="Dictionary{TKey,TValue}"/> object.</param>
        public void Return(Dictionary<TKey, TValue> obj)
        {
            DictionaryPoolInstance.Return(obj._underlyingDict);
            _objects.Add(obj);
        }
    }
}
