using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Collections
{
    internal class DictionarySafeHandle : SafeHandle
    {
        public DictionarySafeHandle(IntPtr handle) : base(IntPtr.Zero, true) => this.handle = handle;

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            Dictionary.godot_icall_Dictionary_Dtor(handle);
            return true;
        }
    }

    public class Dictionary :
        IDictionary,
        IDisposable
    {
        private DictionarySafeHandle _safeHandle;
        private bool _disposed = false;

        public Dictionary() => _safeHandle = new DictionarySafeHandle(godot_icall_Dictionary_Ctor());

        public Dictionary(IDictionary dictionary) : this()
        {
            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            foreach (DictionaryEntry entry in dictionary)
                Add(entry.Key, entry.Value);
        }

        internal Dictionary(DictionarySafeHandle handle) => _safeHandle = handle;

        internal Dictionary(IntPtr handle) => _safeHandle = new DictionarySafeHandle(handle);

        internal IntPtr GetPtr()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().FullName);

            return _safeHandle.DangerousGetHandle();
        }

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

        public Dictionary Duplicate(bool deep = false) =>
            new Dictionary(godot_icall_Dictionary_Duplicate(GetPtr(), deep));

        // IDictionary

        public ICollection Keys
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Keys(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        public ICollection Values
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Values(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        public bool IsFixedSize => false;

        public bool IsReadOnly => false;

        public object this[object key]
        {
            get => godot_icall_Dictionary_GetValue(GetPtr(), key);
            set => godot_icall_Dictionary_SetValue(GetPtr(), key, value);
        }

        public void Add(object key, object value) => godot_icall_Dictionary_Add(GetPtr(), key, value);

        public void Clear() => godot_icall_Dictionary_Clear(GetPtr());

        public bool Contains(object key) => godot_icall_Dictionary_ContainsKey(GetPtr(), key);

        public IDictionaryEnumerator GetEnumerator() => new DictionaryEnumerator(this);

        public void Remove(object key) => godot_icall_Dictionary_RemoveKey(GetPtr(), key);

        // ICollection

        public object SyncRoot => this;

        public bool IsSynchronized => false;

        public int Count => godot_icall_Dictionary_Count(GetPtr());

        public void CopyTo(System.Array array, int index)
        {
            // TODO Can be done with single internal call

            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Number was less than the array's lower bound in the first dimension.");

            var keys = (Array)Keys;
            var values = (Array)Values;
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
            private readonly Array _keys;
            private readonly Array _values;
            private readonly int _count;
            private int _index = -1;

            public DictionaryEnumerator(Dictionary dictionary)
            {
                // TODO 3 internal calls, can reduce to 1
                _keys = (Array)dictionary.Keys;
                _values = (Array)dictionary.Values;
                _count = dictionary.Count;
            }

            public object Current => Entry;

            public DictionaryEntry Entry =>
                // TODO 2 internal calls, can reduce to 1
                new DictionaryEntry(_keys[_index], _values[_index]);

            public object Key => Entry.Key;

            public object Value => Entry.Value;

            public bool MoveNext()
            {
                _index++;
                return _index < _count;
            }

            public void Reset() => _index = -1;
        }

        public override string ToString() => godot_icall_Dictionary_ToString(GetPtr());

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Dictionary_GetValue(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Dictionary_GetValue_Generic(IntPtr ptr, object key,
            int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_SetValue(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Keys(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Dictionary_Values(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Dictionary_Count(IntPtr ptr);

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
        internal static extern bool godot_icall_Dictionary_TryGetValue_Generic(IntPtr ptr, object key, out object value,
            int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Dictionary_Generic_GetValueTypeInfo(Type valueType,
            out int valTypeEncoding, out IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Dictionary_ToString(IntPtr ptr);
    }

    public class Dictionary<TKey, TValue> :
        IDictionary<TKey, TValue>
    {
        private Dictionary objectDict;

        internal static int valTypeEncoding;
        internal static IntPtr valTypeClass;

        static Dictionary() =>
            Dictionary.godot_icall_Dictionary_Generic_GetValueTypeInfo(typeof(TValue), out valTypeEncoding,
                out valTypeClass);

        public Dictionary() => objectDict = new Dictionary();

        public Dictionary(IDictionary<TKey, TValue> dictionary)
        {
            objectDict = new Dictionary();

            if (dictionary == null)
                throw new NullReferenceException($"Parameter '{nameof(dictionary)} cannot be null.'");

            // TODO: Can be optimized

            IntPtr godotDictionaryPtr = GetPtr();

            foreach (KeyValuePair<TKey, TValue> entry in dictionary)
            {
                Dictionary.godot_icall_Dictionary_Add(godotDictionaryPtr, entry.Key, entry.Value);
            }
        }

        public Dictionary(Dictionary dictionary) => objectDict = dictionary;

        internal Dictionary(IntPtr handle) => objectDict = new Dictionary(handle);

        internal Dictionary(DictionarySafeHandle handle) => objectDict = new Dictionary(handle);

        public static explicit operator Dictionary(Dictionary<TKey, TValue> from) => from.objectDict;

        internal IntPtr GetPtr() => objectDict.GetPtr();

        public Dictionary<TKey, TValue> Duplicate(bool deep = false) =>
            new Dictionary<TKey, TValue>(objectDict.Duplicate(deep));

        // IDictionary<TKey, TValue>

        public TValue this[TKey key]
        {
            get => (TValue)Dictionary.godot_icall_Dictionary_GetValue_Generic(objectDict.GetPtr(), key, valTypeEncoding,
                valTypeClass);
            set => objectDict[key] = value;
        }

        public ICollection<TKey> Keys
        {
            get
            {
                IntPtr handle = Dictionary.godot_icall_Dictionary_Keys(objectDict.GetPtr());
                return new Array<TKey>(new ArraySafeHandle(handle));
            }
        }

        public ICollection<TValue> Values
        {
            get
            {
                IntPtr handle = Dictionary.godot_icall_Dictionary_Values(objectDict.GetPtr());
                return new Array<TValue>(new ArraySafeHandle(handle));
            }
        }

        public void Add(TKey key, TValue value) => objectDict.Add(key, value);

        public bool ContainsKey(TKey key) => objectDict.Contains(key);

        public bool Remove(TKey key) => Dictionary.godot_icall_Dictionary_RemoveKey(GetPtr(), key);

        public bool TryGetValue(TKey key, out TValue value)
        {
            object retValue;
            bool found =
                Dictionary.godot_icall_Dictionary_TryGetValue_Generic(GetPtr(), key, out retValue, valTypeEncoding,
                    valTypeClass);
            value = found ? (TValue)retValue : default;
            return found;
        }

        // ICollection<KeyValuePair<TKey, TValue>>

        public int Count => objectDict.Count;

        public bool IsReadOnly => objectDict.IsReadOnly;

        public void Add(KeyValuePair<TKey, TValue> item) => objectDict.Add(item.Key, item.Value);

        public void Clear() => objectDict.Clear();

        public bool Contains(KeyValuePair<TKey, TValue> item) =>
            objectDict.Contains(new KeyValuePair<object, object>(item.Key, item.Value));

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            // TODO 3 internal calls, can reduce to 1
            var keys = (Array<TKey>)Keys;
            var values = (Array<TValue>)Values;
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

        public bool Remove(KeyValuePair<TKey, TValue> item) =>
            Dictionary.godot_icall_Dictionary_Remove(GetPtr(), item.Key, item.Value);

        // IEnumerable<KeyValuePair<TKey, TValue>>

        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
        {
            // TODO 3 internal calls, can reduce to 1
            var keys = (Array<TKey>)Keys;
            var values = (Array<TValue>)Values;
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                yield return new KeyValuePair<TKey, TValue>(keys[i], values[i]);
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => objectDict.ToString();
    }
}
