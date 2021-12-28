using System;
using System.Collections.Generic;
using System.Collections;
using Godot.NativeInterop;
using System.Diagnostics.CodeAnalysis;

namespace Godot.Collections
{
    /// <summary>
    /// Wrapper around Godot's Dictionary class, a dictionary of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine.
    /// </summary>
    public sealed class Dictionary :
        IDictionary,
        IDisposable
    {
        internal godot_dictionary.movable NativeValue;

        private WeakReference<IDisposable> _weakReferenceToSelf;

        /// <summary>
        /// Constructs a new empty <see cref="Dictionary"/>.
        /// </summary>
        public Dictionary()
        {
            NativeValue = (godot_dictionary.movable)NativeFuncs.godotsharp_dictionary_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(IDictionary dictionary) : this()
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            foreach (DictionaryEntry entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        private Dictionary(godot_dictionary nativeValueToOwn)
        {
            NativeValue = (godot_dictionary.movable)(nativeValueToOwn.IsAllocated ?
                nativeValueToOwn :
                NativeFuncs.godotsharp_dictionary_new());
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        // Explicit name to make it very clear
        internal static Dictionary CreateTakingOwnershipOfDisposableValue(godot_dictionary nativeValueToOwn)
            => new Dictionary(nativeValueToOwn);

        ~Dictionary()
        {
            Dispose(false);
        }

        /// <summary>
        /// Disposes of this <see cref="Dictionary"/>.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            // Always dispose `NativeValue` even if disposing is true
            NativeValue.DangerousSelfRef.Dispose();
            DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
        }

        /// <summary>
        /// Duplicates this <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary Duplicate(bool deep = false)
        {
            godot_dictionary newDictionary;
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_duplicate(ref self, deep.ToGodotBool(), out newDictionary);
            return CreateTakingOwnershipOfDisposableValue(newDictionary);
        }

        // IDictionary

        /// <summary>
        /// Gets the collection of keys in this <see cref="Dictionary"/>.
        /// </summary>
        public ICollection Keys
        {
            get
            {
                godot_array keysArray;
                var self = (godot_dictionary)NativeValue;
                NativeFuncs.godotsharp_dictionary_keys(ref self, out keysArray);
                return Array.CreateTakingOwnershipOfDisposableValue(keysArray);
            }
        }

        /// <summary>
        /// Gets the collection of elements in this <see cref="Dictionary"/>.
        /// </summary>
        public ICollection Values
        {
            get
            {
                godot_array valuesArray;
                var self = (godot_dictionary)NativeValue;
                NativeFuncs.godotsharp_dictionary_values(ref self, out valuesArray);
                return Array.CreateTakingOwnershipOfDisposableValue(valuesArray);
            }
        }

        private (Array keys, Array values, int count) GetKeyValuePairs()
        {
            var self = (godot_dictionary)NativeValue;

            godot_array keysArray;
            NativeFuncs.godotsharp_dictionary_keys(ref self, out keysArray);
            var keys = Array.CreateTakingOwnershipOfDisposableValue(keysArray);

            godot_array valuesArray;
            NativeFuncs.godotsharp_dictionary_keys(ref self, out valuesArray);
            var values = Array.CreateTakingOwnershipOfDisposableValue(valuesArray);

            int count = NativeFuncs.godotsharp_dictionary_count(ref self);

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
            get
            {
                using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
                var self = (godot_dictionary)NativeValue;
                if (NativeFuncs.godotsharp_dictionary_try_get_value(ref self, variantKey,
                        out godot_variant value).ToBool())
                {
                    using (value)
                        return Marshaling.ConvertVariantToManagedObject(value);
                }
                else
                {
                    throw new KeyNotFoundException();
                }
            }
            set
            {
                using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
                using godot_variant variantValue = Marshaling.ConvertManagedObjectToVariant(value);
                var self = (godot_dictionary)NativeValue;
                NativeFuncs.godotsharp_dictionary_set_value(ref self, variantKey, variantValue);
            }
        }

        /// <summary>
        /// Adds an object <paramref name="value"/> at key <paramref name="key"/>
        /// to this <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="key">The key at which to add the object.</param>
        /// <param name="value">The object to add.</param>
        public void Add(object key, object value)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);

            var self = (godot_dictionary)NativeValue;

            if (NativeFuncs.godotsharp_dictionary_contains_key(ref self, variantKey).ToBool())
                throw new ArgumentException("An element with the same key already exists", nameof(key));

            using godot_variant variantValue = Marshaling.ConvertManagedObjectToVariant(value);
            NativeFuncs.godotsharp_dictionary_add(ref self, variantKey, variantValue);
        }

        /// <summary>
        /// Erases all items from this <see cref="Dictionary"/>.
        /// </summary>
        public void Clear()
        {
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_clear(ref self);
        }

        /// <summary>
        /// Checks if this <see cref="Dictionary"/> contains the given key.
        /// </summary>
        /// <param name="key">The key to look for.</param>
        /// <returns>Whether or not this dictionary contains the given key.</returns>
        public bool Contains(object key)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
            var self = (godot_dictionary)NativeValue;
            return NativeFuncs.godotsharp_dictionary_contains_key(ref self, variantKey).ToBool();
        }

        /// <summary>
        /// Gets an enumerator for this <see cref="Dictionary"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IDictionaryEnumerator GetEnumerator() => new DictionaryEnumerator(this);

        /// <summary>
        /// Removes an element from this <see cref="Dictionary"/> by key.
        /// </summary>
        /// <param name="key">The key of the element to remove.</param>
        public void Remove(object key)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_remove_key(ref self, variantKey);
        }

        // ICollection

        object ICollection.SyncRoot => this;

        bool ICollection.IsSynchronized => false;

        /// <summary>
        /// Returns the number of elements in this <see cref="Dictionary"/>.
        /// This is also known as the size or length of the dictionary.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count
        {
            get
            {
                var self = (godot_dictionary)NativeValue;
                return NativeFuncs.godotsharp_dictionary_count(ref self);
            }
        }

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
                throw new ArgumentOutOfRangeException(nameof(index),
                    "Number was less than the array's lower bound in the first dimension.");

            var (keys, values, count) = GetKeyValuePairs();

            if (array.Length < (index + count))
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

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
                var self = (godot_dictionary)_dictionary.NativeValue;
                NativeFuncs.godotsharp_dictionary_key_value_pair_at(ref self, _index,
                    out godot_variant key,
                    out godot_variant value);
                using (key)
                using (value)
                {
                    // FIXME: DictionaryEntry keys cannot be null, but Godot dictionaries allow null keys
                    _entry = new DictionaryEntry(Marshaling.ConvertVariantToManagedObject(key)!,
                        Marshaling.ConvertVariantToManagedObject(value));
                }
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
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_to_string(ref self, out godot_string str);
            using (str)
                return Marshaling.ConvertStringToManaged(str);
        }
    }

    internal interface IGenericGodotDictionary
    {
        Dictionary UnderlyingDictionary { get; }
        Type TypeOfKeys { get; }
        Type TypeOfValues { get; }
    }

    // TODO: Now we should be able to avoid boxing

    /// <summary>
    /// Typed wrapper around Godot's Dictionary class, a dictionary of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/>.
    /// </summary>
    /// <typeparam name="TKey">The type of the dictionary's keys.</typeparam>
    /// <typeparam name="TValue">The type of the dictionary's values.</typeparam>
    public sealed class Dictionary<TKey, TValue> :
        IDictionary<TKey, TValue>, IGenericGodotDictionary
    {
        private readonly Dictionary _underlyingDict;

        // ReSharper disable StaticMemberInGenericType
        // Warning is about unique static fields being created for each generic type combination:
        // https://www.jetbrains.com/help/resharper/StaticMemberInGenericType.html
        // In our case this is exactly what we want.
        private static readonly Type TypeOfKeys = typeof(TKey);

        private static readonly Type TypeOfValues = typeof(TValue);
        // ReSharper restore StaticMemberInGenericType

        Dictionary IGenericGodotDictionary.UnderlyingDictionary => _underlyingDict;
        Type IGenericGodotDictionary.TypeOfKeys => TypeOfKeys;
        Type IGenericGodotDictionary.TypeOfValues => TypeOfValues;

        /// <summary>
        /// Constructs a new empty <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public Dictionary()
        {
            _underlyingDict = new Dictionary();
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary{TKey, TValue}"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(IDictionary<TKey, TValue> dictionary)
        {
            if (dictionary == null)
                throw new ArgumentNullException(nameof(dictionary));

            _underlyingDict = new Dictionary();

            foreach (KeyValuePair<TKey, TValue> entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        /// <summary>
        /// Constructs a new <see cref="Dictionary{TKey, TValue}"/> from the given dictionary's elements.
        /// </summary>
        /// <param name="dictionary">The dictionary to construct from.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary(Dictionary dictionary)
        {
            _underlyingDict = dictionary;
        }

        // Explicit name to make it very clear
        internal static Dictionary<TKey, TValue> CreateTakingOwnershipOfDisposableValue(
            godot_dictionary nativeValueToOwn)
            => new Dictionary<TKey, TValue>(Dictionary.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn));

        /// <summary>
        /// Converts this typed <see cref="Dictionary{TKey, TValue}"/> to an untyped <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="from">The typed dictionary to convert.</param>
        public static explicit operator Dictionary(Dictionary<TKey, TValue> from)
        {
            return from?._underlyingDict;
        }

        /// <summary>
        /// Duplicates this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Dictionary.</returns>
        public Dictionary<TKey, TValue> Duplicate(bool deep = false)
        {
            return new Dictionary<TKey, TValue>(_underlyingDict.Duplicate(deep));
        }

        // IDictionary<TKey, TValue>

        /// <summary>
        /// Returns the value at the given <paramref name="key"/>.
        /// </summary>
        /// <value>The value at the given <paramref name="key"/>.</value>
        public TValue this[TKey key]
        {
            get
            {
                using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
                var self = (godot_dictionary)_underlyingDict.NativeValue;
                if (NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                        variantKey, out godot_variant value).ToBool())
                {
                    using (value)
                        return (TValue)Marshaling.ConvertVariantToManagedObjectOfType(value, TypeOfValues);
                }
                else
                {
                    throw new KeyNotFoundException();
                }
            }
            set => _underlyingDict[key] = value;
        }

        /// <summary>
        /// Gets the collection of keys in this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public ICollection<TKey> Keys
        {
            get
            {
                godot_array keyArray;
                var self = (godot_dictionary)_underlyingDict.NativeValue;
                NativeFuncs.godotsharp_dictionary_keys(ref self, out keyArray);
                return Array<TKey>.CreateTakingOwnershipOfDisposableValue(keyArray);
            }
        }

        /// <summary>
        /// Gets the collection of elements in this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public ICollection<TValue> Values
        {
            get
            {
                godot_array valuesArray;
                var self = (godot_dictionary)_underlyingDict.NativeValue;
                NativeFuncs.godotsharp_dictionary_values(ref self, out valuesArray);
                return Array<TValue>.CreateTakingOwnershipOfDisposableValue(valuesArray);
            }
        }

        private KeyValuePair<TKey, TValue> GetKeyValuePair(int index)
        {
            var self = (godot_dictionary)_underlyingDict.NativeValue;
            NativeFuncs.godotsharp_dictionary_key_value_pair_at(ref self, index,
                out godot_variant key,
                out godot_variant value);
            using (key)
            using (value)
            {
                return new KeyValuePair<TKey, TValue>((TKey)Marshaling.ConvertVariantToManagedObject(key),
                    (TValue)Marshaling.ConvertVariantToManagedObject(value));
            }
        }

        /// <summary>
        /// Adds an object <paramref name="value"/> at key <paramref name="key"/>
        /// to this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        /// <param name="key">The key at which to add the object.</param>
        /// <param name="value">The object to add.</param>
        public void Add(TKey key, TValue value)
        {
            _underlyingDict.Add(key, value);
        }

        /// <summary>
        /// Checks if this <see cref="Dictionary{TKey, TValue}"/> contains the given key.
        /// </summary>
        /// <param name="key">The key to look for.</param>
        /// <returns>Whether or not this dictionary contains the given key.</returns>
        public bool ContainsKey(TKey key)
        {
            return _underlyingDict.Contains(key);
        }

        /// <summary>
        /// Removes an element from this <see cref="Dictionary{TKey, TValue}"/> by key.
        /// </summary>
        /// <param name="key">The key of the element to remove.</param>
        public bool Remove(TKey key)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
            var self = (godot_dictionary)_underlyingDict.NativeValue;
            return NativeFuncs.godotsharp_dictionary_remove_key(ref self, variantKey).ToBool();
        }

        /// <summary>
        /// Gets the object at the given <paramref name="key"/>.
        /// </summary>
        /// <param name="key">The key of the element to get.</param>
        /// <param name="value">The value at the given <paramref name="key"/>.</param>
        /// <returns>If an object was found for the given <paramref name="key"/>.</returns>
        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(key);
            var self = (godot_dictionary)_underlyingDict.NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                variantKey, out godot_variant retValue).ToBool();

            using (retValue)
            {
                value = found ?
                    (TValue)Marshaling.ConvertVariantToManagedObjectOfType(retValue, TypeOfValues) :
                    default;
            }

            return found;
        }

        // ICollection<KeyValuePair<TKey, TValue>>

        /// <summary>
        /// Returns the number of elements in this <see cref="Dictionary{TKey, TValue}"/>.
        /// This is also known as the size or length of the dictionary.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => _underlyingDict.Count;

        bool ICollection<KeyValuePair<TKey, TValue>>.IsReadOnly => false;

        void ICollection<KeyValuePair<TKey, TValue>>.Add(KeyValuePair<TKey, TValue> item)
        {
            _underlyingDict.Add(item.Key, item.Value);
        }

        /// <summary>
        /// Erases all the items from this <see cref="Dictionary{TKey, TValue}"/>.
        /// </summary>
        public void Clear()
        {
            _underlyingDict.Clear();
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Contains(KeyValuePair<TKey, TValue> item)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(item.Key);
            var self = (godot_dictionary)_underlyingDict.NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                variantKey, out godot_variant retValue).ToBool();

            using (retValue)
            {
                if (!found)
                    return false;

                return NativeFuncs.godotsharp_variant_equals(variantKey, retValue).ToBool();
            }
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
                throw new ArgumentOutOfRangeException(nameof(arrayIndex),
                    "Number was less than the array's lower bound in the first dimension.");

            int count = Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = GetKeyValuePair(i);
                arrayIndex++;
            }
        }

        bool ICollection<KeyValuePair<TKey, TValue>>.Remove(KeyValuePair<TKey, TValue> item)
        {
            using godot_variant variantKey = Marshaling.ConvertManagedObjectToVariant(item.Key);
            var self = (godot_dictionary)_underlyingDict.NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                variantKey, out godot_variant retValue).ToBool();

            using (retValue)
            {
                if (!found)
                    return false;

                if (NativeFuncs.godotsharp_variant_equals(variantKey, retValue).ToBool())
                {
                    return NativeFuncs.godotsharp_dictionary_remove_key(
                        ref self, variantKey).ToBool();
                }

                return false;
            }
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

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Converts this <see cref="Dictionary{TKey, TValue}"/> to a string.
        /// </summary>
        /// <returns>A string representation of this dictionary.</returns>
        public override string ToString() => _underlyingDict.ToString();
    }
}
