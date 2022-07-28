using System;
using System.Collections.Generic;
using System.Collections;
using Godot.NativeInterop;

namespace Godot.Collections
{
    /// <summary>
    /// Wrapper around Godot's Dictionary class, a dictionary of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine.
    /// </summary>
    public sealed class Dictionary :
        IDictionary<Variant, Variant>,
        IReadOnlyDictionary<Variant, Variant>,
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

            if (_weakReferenceToSelf != null)
            {
                DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
            }
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
        public ICollection<Variant> Keys
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
        public ICollection<Variant> Values
        {
            get
            {
                godot_array valuesArray;
                var self = (godot_dictionary)NativeValue;
                NativeFuncs.godotsharp_dictionary_values(ref self, out valuesArray);
                return Array.CreateTakingOwnershipOfDisposableValue(valuesArray);
            }
        }

        IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Keys => Keys;

        IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Values => Values;

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

        /// <summary>
        /// Returns the value at the given <paramref name="key"/>.
        /// </summary>
        /// <value>The value at the given <paramref name="key"/>.</value>
        public Variant this[Variant key]
        {
            get
            {
                var self = (godot_dictionary)NativeValue;

                if (NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                        (godot_variant)key.NativeVar, out godot_variant value).ToBool())
                {
                    return Variant.CreateTakingOwnershipOfDisposableValue(value);
                }
                else
                {
                    throw new KeyNotFoundException();
                }
            }
            set
            {
                var self = (godot_dictionary)NativeValue;
                NativeFuncs.godotsharp_dictionary_set_value(ref self,
                    (godot_variant)key.NativeVar, (godot_variant)value.NativeVar);
            }
        }

        /// <summary>
        /// Adds an value <paramref name="value"/> at key <paramref name="key"/>
        /// to this <see cref="Dictionary"/>.
        /// </summary>
        /// <param name="key">The key at which to add the value.</param>
        /// <param name="value">The value to add.</param>
        public void Add(Variant key, Variant value)
        {
            var variantKey = (godot_variant)key.NativeVar;
            var self = (godot_dictionary)NativeValue;

            if (NativeFuncs.godotsharp_dictionary_contains_key(ref self, variantKey).ToBool())
                throw new ArgumentException("An element with the same key already exists", nameof(key));

            godot_variant variantValue = (godot_variant)value.NativeVar;
            NativeFuncs.godotsharp_dictionary_add(ref self, variantKey, variantValue);
        }

        void ICollection<KeyValuePair<Variant, Variant>>.Add(KeyValuePair<Variant, Variant> item)
            => Add(item.Key, item.Value);

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
        public bool ContainsKey(Variant key)
        {
            var self = (godot_dictionary)NativeValue;
            return NativeFuncs.godotsharp_dictionary_contains_key(ref self, (godot_variant)key.NativeVar).ToBool();
        }

        public bool Contains(KeyValuePair<Variant, Variant> item)
        {
            godot_variant variantKey = (godot_variant)item.Key.NativeVar;
            var self = (godot_dictionary)NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                variantKey, out godot_variant retValue).ToBool();

            using (retValue)
            {
                if (!found)
                    return false;

                godot_variant variantValue = (godot_variant)item.Value.NativeVar;
                return NativeFuncs.godotsharp_variant_equals(variantValue, retValue).ToBool();
            }
        }

        /// <summary>
        /// Removes an element from this <see cref="Dictionary"/> by key.
        /// </summary>
        /// <param name="key">The key of the element to remove.</param>
        public bool Remove(Variant key)
        {
            var self = (godot_dictionary)NativeValue;
            return NativeFuncs.godotsharp_dictionary_remove_key(ref self, (godot_variant)key.NativeVar).ToBool();
        }

        public bool Remove(KeyValuePair<Variant, Variant> item)
        {
            godot_variant variantKey = (godot_variant)item.Key.NativeVar;
            var self = (godot_dictionary)NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                variantKey, out godot_variant retValue).ToBool();

            using (retValue)
            {
                if (!found)
                    return false;

                godot_variant variantValue = (godot_variant)item.Value.NativeVar;
                if (NativeFuncs.godotsharp_variant_equals(variantValue, retValue).ToBool())
                {
                    return NativeFuncs.godotsharp_dictionary_remove_key(
                        ref self, variantKey).ToBool();
                }

                return false;
            }
        }

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

        bool ICollection<KeyValuePair<Variant, Variant>>.IsReadOnly => false;

        public bool TryGetValue(Variant key, out Variant value)
        {
            var self = (godot_dictionary)NativeValue;
            bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                (godot_variant)key.NativeVar, out godot_variant retValue).ToBool();

            value = found ? Variant.CreateTakingOwnershipOfDisposableValue(retValue) : default;

            return found;
        }

        /// <summary>
        /// Copies the elements of this <see cref="Dictionary"/> to the given untyped
        /// <see cref="KeyValuePair{TKey, TValue}"/> array, starting at the given index.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        /// <param name="arrayIndex">The index to start at.</param>
        public void CopyTo(KeyValuePair<Variant, Variant>[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex),
                    "Number was less than the array's lower bound in the first dimension.");

            var (keys, values, count) = GetKeyValuePairs();

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = new(keys[i], values[i]);
                arrayIndex++;
            }
        }

        // IEnumerable

        /// <summary>
        /// Gets an enumerator for this <see cref="Dictionary"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<KeyValuePair<Variant, Variant>> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
            {
                yield return GetKeyValuePair(i);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        private KeyValuePair<Variant, Variant> GetKeyValuePair(int index)
        {
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_key_value_pair_at(ref self, index,
                out godot_variant key,
                out godot_variant value);
            return new KeyValuePair<Variant, Variant>(Variant.CreateTakingOwnershipOfDisposableValue(key),
                Variant.CreateTakingOwnershipOfDisposableValue(value));
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
}
