using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot.Collections
{
    public sealed class Dictionary :
        IDictionary,
        IDisposable
    {
        internal godot_dictionary NativeValue;

        public Dictionary()
        {
            godot_icall_Dictionary_Ctor(out NativeValue);
        }

        public Dictionary(IDictionary dictionary) : this()
        {
            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            foreach (DictionaryEntry entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        private Dictionary(godot_dictionary nativeValueToOwn)
        {
            NativeValue = nativeValueToOwn;
        }

        // Explicit name to make it very clear
        internal static Dictionary CreateTakingOwnershipOfDisposableValue(godot_dictionary nativeValueToOwn)
            => new Dictionary(nativeValueToOwn);

        ~Dictionary()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            // Always dispose `NativeValue` even if disposing is true
            NativeValue.Dispose();
        }

        public Dictionary Duplicate(bool deep = false)
        {
            godot_dictionary newDictionary;
            godot_icall_Dictionary_Duplicate(ref NativeValue, deep, out newDictionary);
            return CreateTakingOwnershipOfDisposableValue(newDictionary);
        }

        // IDictionary

        public ICollection Keys
        {
            get
            {
                godot_array keysArray;
                godot_icall_Dictionary_Keys(ref NativeValue, out keysArray);
                return Array.CreateTakingOwnershipOfDisposableValue(keysArray);
            }
        }

        public ICollection Values
        {
            get
            {
                godot_array valuesArray;
                godot_icall_Dictionary_Values(ref NativeValue, out valuesArray);
                return Array.CreateTakingOwnershipOfDisposableValue(valuesArray);
            }
        }

        public bool IsFixedSize => false;

        public bool IsReadOnly => false;

        public object this[object key]
        {
            get
            {
                godot_icall_Dictionary_GetValue(ref NativeValue, key, out godot_variant value);
                unsafe
                {
                    using (value)
                        return Marshaling.variant_to_mono_object(&value);
                }
            }
            set => godot_icall_Dictionary_SetValue(ref NativeValue, key, value);
        }

        public void Add(object key, object value) => godot_icall_Dictionary_Add(ref NativeValue, key, value);

        public void Clear() => godot_icall_Dictionary_Clear(ref NativeValue);

        public bool Contains(object key) => godot_icall_Dictionary_ContainsKey(ref NativeValue, key);

        public IDictionaryEnumerator GetEnumerator() => new DictionaryEnumerator(this);

        public void Remove(object key) => godot_icall_Dictionary_RemoveKey(ref NativeValue, key);

        // ICollection

        public object SyncRoot => this;

        public bool IsSynchronized => false;

        public int Count => godot_icall_Dictionary_Count(ref NativeValue);

        public void CopyTo(System.Array array, int index)
        {
            // TODO Can be done with single internal call

            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Number was less than the array's lower bound in the first dimension.");

            Array keys = (Array)Keys;
            Array values = (Array)Values;
            int count = Count;

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
            Array keys;
            Array values;
            int count;
            int _index = -1;

            public DictionaryEnumerator(Dictionary dictionary)
            {
                // TODO 3 internal calls, can reduce to 1
                keys = (Array)dictionary.Keys;
                values = (Array)dictionary.Values;
                count = dictionary.Count;
            }

            public object Current => Entry;

            public DictionaryEntry Entry =>
                // TODO 2 internal calls, can reduce to 1
                new DictionaryEntry(keys[_index], values[_index]);

            public object Key => Entry.Key;

            public object Value => Entry.Value;

            public bool MoveNext()
            {
                _index++;
                return _index < count;
            }

            public void Reset()
            {
                _index = -1;
            }
        }

        public override string ToString()
        {
            return godot_icall_Dictionary_ToString(ref NativeValue);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Ctor(out godot_dictionary dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_GetValue(ref godot_dictionary ptr, object key, out godot_variant value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_SetValue(ref godot_dictionary ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Keys(ref godot_dictionary ptr, out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Values(ref godot_dictionary ptr, out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Dictionary_Count(ref godot_dictionary ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Add(ref godot_dictionary ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Clear(ref godot_dictionary ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_Contains(ref godot_dictionary ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_ContainsKey(ref godot_dictionary ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Duplicate(ref godot_dictionary ptr, bool deep, out godot_dictionary dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_RemoveKey(ref godot_dictionary ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_Remove(ref godot_dictionary ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Dictionary_TryGetValue(ref godot_dictionary ptr, object key, out godot_variant value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Dictionary_ToString(ref godot_dictionary ptr);
    }

    internal interface IGenericGodotDictionary
    {
        Dictionary UnderlyingDictionary { get; }
        Type TypeOfKeys { get; }
        Type TypeOfValues { get; }
    }

    // TODO: Now we should be able to avoid boxing

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

        public Dictionary()
        {
            _underlyingDict = new Dictionary();
        }

        public Dictionary(IDictionary<TKey, TValue> dictionary)
        {
            _underlyingDict = new Dictionary();

            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            foreach (KeyValuePair<TKey, TValue> entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        public Dictionary(Dictionary dictionary)
        {
            _underlyingDict = dictionary;
        }

        // Explicit name to make it very clear
        internal static Dictionary<TKey, TValue> CreateTakingOwnershipOfDisposableValue(godot_dictionary nativeValueToOwn)
            => new Dictionary<TKey, TValue>(Dictionary.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn));

        public static explicit operator Dictionary(Dictionary<TKey, TValue> from)
        {
            return from._underlyingDict;
        }

        public Dictionary<TKey, TValue> Duplicate(bool deep = false)
        {
            return new Dictionary<TKey, TValue>(_underlyingDict.Duplicate(deep));
        }

        // IDictionary<TKey, TValue>

        public TValue this[TKey key]
        {
            get
            {
                Dictionary.godot_icall_Dictionary_GetValue(ref _underlyingDict.NativeValue, key, out godot_variant value);
                unsafe
                {
                    using (value)
                        return (TValue)Marshaling.variant_to_mono_object_of_type(&value, TypeOfValues);
                }
            }
            set => _underlyingDict[key] = value;
        }

        public ICollection<TKey> Keys
        {
            get
            {
                godot_array keyArray;
                Dictionary.godot_icall_Dictionary_Keys(ref _underlyingDict.NativeValue, out keyArray);
                return Array<TKey>.CreateTakingOwnershipOfDisposableValue(keyArray);
            }
        }

        public ICollection<TValue> Values
        {
            get
            {
                godot_array valuesArray;
                Dictionary.godot_icall_Dictionary_Values(ref _underlyingDict.NativeValue, out valuesArray);
                return Array<TValue>.CreateTakingOwnershipOfDisposableValue(valuesArray);
            }
        }

        public void Add(TKey key, TValue value)
        {
            _underlyingDict.Add(key, value);
        }

        public bool ContainsKey(TKey key)
        {
            return _underlyingDict.Contains(key);
        }

        public bool Remove(TKey key)
        {
            return Dictionary.godot_icall_Dictionary_RemoveKey(ref _underlyingDict.NativeValue, key);
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            bool found = Dictionary.godot_icall_Dictionary_TryGetValue(ref _underlyingDict.NativeValue, key, out godot_variant retValue);

            unsafe
            {
                using (retValue)
                {
                    value = found ?
                        (TValue)Marshaling.variant_to_mono_object_of_type(&retValue, TypeOfValues) :
                        default;
                }
            }

            return found;
        }

        // ICollection<KeyValuePair<TKey, TValue>>

        public int Count => _underlyingDict.Count;

        public bool IsReadOnly => _underlyingDict.IsReadOnly;

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            _underlyingDict.Add(item.Key, item.Value);
        }

        public void Clear()
        {
            _underlyingDict.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return _underlyingDict.Contains(new KeyValuePair<object, object>(item.Key, item.Value));
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            // TODO 3 internal calls, can reduce to 1
            Array<TKey> keys = (Array<TKey>)Keys;
            Array<TValue> values = (Array<TValue>)Values;
            int count = Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                array[arrayIndex] = new KeyValuePair<TKey, TValue>(keys[i], values[i]);
                arrayIndex++;
            }
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return Dictionary.godot_icall_Dictionary_Remove(ref _underlyingDict.NativeValue, item.Key, item.Value);
        }

        // IEnumerable<KeyValuePair<TKey, TValue>>

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            // TODO 3 internal calls, can reduce to 1
            Array<TKey> keys = (Array<TKey>)Keys;
            Array<TValue> values = (Array<TValue>)Values;
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                yield return new KeyValuePair<TKey, TValue>(keys[i], values[i]);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => _underlyingDict.ToString();
    }
}
