using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Diagnostics.CodeAnalysis;

namespace Godot.Collections
{
    internal class DictionarySafeHandle : SafeHandle
    {
        public DictionarySafeHandle(IntPtr handle) : base(IntPtr.Zero, true)
        {
            this.handle = handle;
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            Dictionary.godot_icall_Dictionary_Dtor(handle);
            return true;
        }
    }

    /// <summary>
    /// Wrapper around Godot's Dictionary class, a dictionary of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine.
    /// </summary>
    public class Dictionary : IDictionary, IDisposable
    {
        private DictionarySafeHandle _safeHandle;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new empty <see cref="Dictionary"/>.
        /// </summary>
        public Dictionary()
        {
            _safeHandle = new DictionarySafeHandle(godot_icall_Dictionary_Ctor());
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(IDictionary dictionary) : this()
        {
            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            foreach (DictionaryEntry entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        internal Dictionary(DictionarySafeHandle handle)
        {
            _safeHandle = handle;
        }

        internal Dictionary(IntPtr handle)
        {
            _safeHandle = new DictionarySafeHandle(handle);
        }

        internal IntPtr GetPtr()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().FullName);

            return _safeHandle.DangerousGetHandle();
        }

        /// <summary>
        /// Disposes of this <see cref="Dictionary"/>.
        /// </summary>
        public void Dispose()
        {
            if (_disposed)
                return;

            if (_safeHandle != null)
            {
                _safeHandle.Dispose();
                _safeHandle = null;
            }

            _disposed = true;
        }

        /// <summary>
        /// Duplicates this <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary Duplicate(bool deep = false)
        {
            return new Dictionary(godot_icall_Dictionary_Duplicate(GetPtr(), deep));
        }

        // IDictionary

        /// <summary>
        /// Gets the collection of keys in this <see cref="Dictionary"/>.
        /// </summary>
        public ICollection Keys
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Keys(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        /// <summary>
        /// Gets the collection of elements in this <see cref="Dictionary"/>.
        /// </summary>
        public ICollection Values
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Values(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        private (Array keys, Array values, int count) GetKeyValuePairs()
        {
            int count = godot_icall_Dictionary_KeyValuePairs(GetPtr(), out IntPtr keysHandle, out IntPtr valuesHandle);
            Array keys = new Array(new ArraySafeHandle(keysHandle));
            Array values = new Array(new ArraySafeHandle(valuesHandle));
            return (keys, values, count);
        }

        bool IDictionary.IsFixedSize => false;

        bool IDictionary.IsReadOnly => false;

        /// <summary>
        /// Returns the object at the given <paramref name="key"/>.
        /// </summary>
        /// <value>The object at the given <paramref name="key"/>.</value>
        public object this[object key]
        {
            get => godot_icall_Dictionary_GetValue(GetPtr(), key);
            set => godot_icall_Dictionary_SetValue(GetPtr(), key, value);
        }

        /// <summary>
        /// Adds an object <paramref name="value"/> at key <paramref name="key"/>
        /// to this <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="key">The key at which to add the object.</param>
        /// <param name="value">The object to add.</param>
        public void Add(object key, object value) => godot_icall_Dictionary_Add(GetPtr(), key, value);

        /// <summary>
        /// Erases all items from this <see cref="Dictionary"/>.
        /// </summary>
        public void Clear() => godot_icall_Dictionary_Clear(GetPtr());

        /// <summary>
        /// Checks if this <see cref="Dictionary"/> contains the given key.
        /// </summary>
        /// <param name="key">The key to look for.</param>
        /// <returns>Whether or not this dictionary contains the given key.</returns>
        public bool Contains(object key) => godot_icall_Dictionary_ContainsKey(GetPtr(), key);

        /// <summary>
        /// Gets an enumerator for this <see cref="Dictionary"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IDictionaryEnumerator GetEnumerator() => new DictionaryEnumerator(this);

        /// <summary>
        /// Removes an element from this <see cref="Dictionary"/> by key.
        /// </summary>
        /// <param name="key">The key of the element to remove.</param>
        public void Remove(object key) => godot_icall_Dictionary_RemoveKey(GetPtr(), key);

        // ICollection

        object ICollection.SyncRoot => this;

        bool ICollection.IsSynchronized => false;

        /// <summary>
        /// Returns the number of elements in this <see cref="Dictionary"/>.
        /// This is also known as the size or length of the dictionary.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => godot_icall_Dictionary_Count(GetPtr());

        /// <summary>
        /// Copies the elements of this <see cref="Dictionary"/> to the given
        /// untyped C# array, starting at the given index.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        /// <param name="index">The index to start at.</param>
        public void CopyTo(System.Array array, int index)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Number was less than the array's lower bound in the first dimension.");

            var (keys, values, count) = GetKeyValuePairs();

            if (array.Length < (index + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array.SetValue(new DictionaryEntry(keys[i], values[i]), index);
                index++;
            }
        }

        // IEnumerable

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private class DictionaryEnumerator : IDictionaryEnumerator
        {
            private readonly Dictionary _dictionary;
            private readonly int _count;
            private int _index = -1;
            private bool _dirty = true;

            private DictionaryEntry _entry;

            public DictionaryEnumerator(Dictionary dictionary)
            {
                _dictionary = dictionary;
                _count = dictionary.Count;
            }

            public object Current => Entry;

            public DictionaryEntry Entry
            {
                get
                {
                    if (_dirty)
                    {
                        UpdateEntry();
                    }
                    return _entry;
                }
            }

            private void UpdateEntry()
            {
                _dirty = false;
                godot_icall_Dictionary_KeyValuePairAt(_dictionary.GetPtr(), _index, out object key, out object value);
                _entry = new DictionaryEntry(key, value);
            }

            public object Key => Entry.Key;

            public object Value => Entry.Value;

            public bool MoveNext()
            {
                _index++;
                _dirty = true;
                return _index < _count;
            }

            public void Reset()
            {
                _index = -1;
                _dirty = true;
            }
        }

        /// <summary>
        /// Converts this <see cref="Dictionary"/> to a string.
        /// </summary>
        /// <returns>A string representation of this dictionary.</returns>
        public override string ToString()
        {
            return godot_icall_Dictionary_ToString(GetPtr());
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Dictionary_GetValue(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Dictionary_GetValue_Generic(IntPtr ptr, object key, int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_SetValue(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Keys(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Values(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Dictionary_Count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Dictionary_KeyValuePairs(IntPtr ptr, out IntPtr keys, out IntPtr values);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_KeyValuePairAt(IntPtr ptr, int index, out object key, out object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_KeyValuePairAt_Generic(IntPtr ptr, int index, out object key, out object value, int valueTypeEncoding, IntPtr valueTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Add(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Clear(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_Contains(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_ContainsKey(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Duplicate(IntPtr ptr, bool deep);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_RemoveKey(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_Remove(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_TryGetValue(IntPtr ptr, object key, out object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_TryGetValue_Generic(IntPtr ptr, object key, out object value, int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Generic_GetValueTypeInfo(Type valueType, out int valTypeEncoding, out IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Dictionary_ToString(IntPtr ptr);
    }

    /// <summary>
    /// Typed wrapper around Godot's Dictionary class, a dictionary of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/>.
    /// </summary>
    /// <typeparam name="TKey">The type of the dictionary's keys.</typeparam>
    /// <typeparam name="TValue">The type of the dictionary's values.</typeparam>
    public class Dictionary<TKey, TValue> : IDictionary<TKey, TValue>
    {
        private readonly Dictionary _objectDict;

        internal static int valTypeEncoding;
        internal static IntPtr valTypeClass;

        static Dictionary()
        {
            Dictionary.godot_icall_Dictionary_Generic_GetValueTypeInfo(typeof(TValue), out valTypeEncoding, out valTypeClass);
        }

        /// <summary>
        /// Constructs a new empty <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public Dictionary()
        {
            _objectDict = new Dictionary();
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary{TKey, TValue}"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(IDictionary<TKey, TValue> dictionary)
        {
            _objectDict = new Dictionary();

            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            // TODO: Can be optimized

            IntPtr godotDictionaryPtr = GetPtr();

            foreach (KeyValuePair<TKey, TValue> entry in dictionary)
            {
                Dictionary.godot_icall_Dictionary_Add(godotDictionaryPtr, entry.Key, entry.Value);
            }
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary{TKey, TValue}"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(Dictionary dictionary)
        {
            _objectDict = dictionary;
        }

        internal Dictionary(IntPtr handle)
        {
            _objectDict = new Dictionary(handle);
        }

        internal Dictionary(DictionarySafeHandle handle)
        {
            _objectDict = new Dictionary(handle);
        }

        /// <summary>
        /// Converts this typed <see cref="Dictionary{TKey, TValue}"/> to an untyped <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="from">The typed dictionary to convert.</param>
        public static explicit operator Dictionary(Dictionary<TKey, TValue> from)
        {
            return from._objectDict;
        }

        internal IntPtr GetPtr()
        {
            return _objectDict.GetPtr();
        }

        /// <summary>
        /// Duplicates this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary<TKey, TValue> Duplicate(bool deep = false)
        {
            return new Dictionary<TKey, TValue>(_objectDict.Duplicate(deep));
        }

        // IDictionary<TKey, TValue>

        /// <summary>
        /// Returns the value at the given <paramref name="key"/>.
        /// </summary>
        /// <value>The value at the given <paramref name="key"/>.</value>
        public TValue this[TKey key]
        {
            get { return (TValue)Dictionary.godot_icall_Dictionary_GetValue_Generic(_objectDict.GetPtr(), key, valTypeEncoding, valTypeClass); }
            set { _objectDict[key] = value; }
        }

        /// <summary>
        /// Gets the collection of keys in this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public ICollection<TKey> Keys
        {
            get
            {
                IntPtr handle = Dictionary.godot_icall_Dictionary_Keys(_objectDict.GetPtr());
                return new Array<TKey>(new ArraySafeHandle(handle));
            }
        }

        /// <summary>
        /// Gets the collection of elements in this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public ICollection<TValue> Values
        {
            get
            {
                IntPtr handle = Dictionary.godot_icall_Dictionary_Values(_objectDict.GetPtr());
                return new Array<TValue>(new ArraySafeHandle(handle));
            }
        }

        private KeyValuePair<TKey, TValue> GetKeyValuePair(int index)
        {
            Dictionary.godot_icall_Dictionary_KeyValuePairAt_Generic(GetPtr(), index, out object key, out object value, valTypeEncoding, valTypeClass);
            return new KeyValuePair<TKey, TValue>((TKey)key, (TValue)value);
        }

        /// <summary>
        /// Adds an object <paramref name="value"/> at key <paramref name="key"/>
        /// to this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <param name="key">The key at which to add the object.</param>
        /// <param name="value">The object to add.</param>
        public void Add(TKey key, TValue value)
        {
            _objectDict.Add(key, value);
        }

        /// <summary>
        /// Checks if this <see cref="Dictionary{TKey, TValue}"/> contains the given key.
        /// </summary>
        /// <param name="key">The key to look for.</param>
        /// <returns>Whether or not this dictionary contains the given key.</returns>
        public bool ContainsKey(TKey key)
        {
            return _objectDict.Contains(key);
        }

        /// <summary>
        /// Removes an element from this <see cref="Dictionary{TKey, TValue}"/> by key.
        /// </summary>
        /// <param name="key">The key of the element to remove.</param>
        public bool Remove(TKey key)
        {
            return Dictionary.godot_icall_Dictionary_RemoveKey(GetPtr(), key);
        }

        /// <summary>
        /// Gets the object at the given <paramref name="key"/>.
        /// </summary>
        /// <param name="key">The key of the element to get.</param>
        /// <param name="value">The value at the given <paramref name="key"/>.</param>
        /// <returns>If an object was found for the given <paramref name="key"/>.</returns>
        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
        {
            bool found = Dictionary.godot_icall_Dictionary_TryGetValue_Generic(GetPtr(), key, out object retValue, valTypeEncoding, valTypeClass);
            value = found ? (TValue)retValue : default;
            return found;
        }

        // ICollection<KeyValuePair<TKey, TValue>>

        /// <summary>
        /// Returns the number of elements in this <see cref="Dictionary{TKey, TValue}"/>.
        /// This is also known as the size or length of the dictionary.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count
        {
            get { return _objectDict.Count; }
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.IsReadOnly => false;

        void ICollection<KeyValuePair<TKey, TValue>>.Add(KeyValuePair<TKey, TValue> item)
        {
            _objectDict.Add(item.Key, item.Value);
        }

        /// <summary>
        /// Erases all the items from this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public void Clear()
        {
            _objectDict.Clear();
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Contains(KeyValuePair<TKey, TValue> item)
        {
            return _objectDict.Contains(new KeyValuePair<object, object>(item.Key, item.Value));
        }

        /// <summary>
        /// Copies the elements of this <see cref="Dictionary{TKey, TValue}"/> to the given
        /// untyped C# array, starting at the given index.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        /// <param name="arrayIndex">The index to start at.</param>
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            int count = Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = GetKeyValuePair(i);
                arrayIndex++;
            }
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Remove(KeyValuePair<TKey, TValue> item)
        {
            return Dictionary.godot_icall_Dictionary_Remove(GetPtr(), item.Key, item.Value);
            ;
        }

        // IEnumerable<KeyValuePair<TKey, TValue>>

        /// <summary>
        /// Gets an enumerator for this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
            {
                yield return GetKeyValuePair(i);
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Converts this <see cref="Dictionary{TKey, TValue}"/> to a string.
        /// </summary>
        /// <returns>A string representation of this dictionary.</returns>
        public override string ToString() => _objectDict.ToString();
    }
}
