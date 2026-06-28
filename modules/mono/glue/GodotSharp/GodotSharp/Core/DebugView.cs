using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace Godot.Collections
{
    internal sealed class ArrayDebugView<T>
    {
        private readonly IList<T> _array;

        public ArrayDebugView(IList<T> array)
        {
            ArgumentNullException.ThrowIfNull(array);

            _array = array;
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public T[] Items
        {
            get
            {
                var items = new T[_array.Count];
                _array.CopyTo(items, 0);
                return items;
            }
        }
    }

    internal sealed class DictionaryDebugView<TKey, TValue>
    {
        private readonly IDictionary<TKey, TValue> _dictionary;

        public DictionaryDebugView(IDictionary<TKey, TValue> dictionary)
        {
            ArgumentNullException.ThrowIfNull(dictionary);

            _dictionary = dictionary;
        }

        [DebuggerBrowsable(DebuggerBrowsableState.RootHidden)]
        public DictionaryKeyItemDebugView<TKey, TValue>[] Items
        {
            get
            {
                var items = new KeyValuePair<TKey, TValue>[_dictionary.Count];
                var views = new DictionaryKeyItemDebugView<TKey, TValue>[_dictionary.Count];
                _dictionary.CopyTo(items, 0);
                for (int i = 0; i < items.Length; i++)
                {
                    views[i] = new DictionaryKeyItemDebugView<TKey, TValue>(items[i]);
                }
                return views;
            }
        }
    }

    [DebuggerDisplay("{Value}", Name = "[{Key}]")]
    internal readonly struct DictionaryKeyItemDebugView<TKey, TValue>
    {
        public DictionaryKeyItemDebugView(KeyValuePair<TKey, TValue> keyValue)
        {
            Key = keyValue.Key;
            Value = keyValue.Value;
        }

        [DebuggerBrowsable(DebuggerBrowsableState.Collapsed)]
        public TKey Key { get; }

        [DebuggerBrowsable(DebuggerBrowsableState.Collapsed)]
        public TValue Value { get; }
    }
}
