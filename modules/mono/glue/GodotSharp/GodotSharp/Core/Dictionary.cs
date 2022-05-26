using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
    /// 对 Godot 的 Dictionary 类的封装，一个 Variant 的字典
    /// 在 C++ 引擎中分配的类型化元素。 有用的时候
    /// 与引擎交互。
    /// </summary>
    public class Dictionary : IDictionary, IDisposable
    {
        private DictionarySafeHandle _safeHandle;
        private bool _disposed = false;

        /// <summary>
        /// 构造一个新的空 <see cref="Dictionary"/>。
        /// </summary>
        public Dictionary()
        {
            _safeHandle = new DictionarySafeHandle(godot_icall_Dictionary_Ctor());
        }

        /// <summary>
        /// 从给定字典的元素构造一个新的 <see cref="Dictionary"/>。
        /// </summary>
        /// <param name="dictionary">要构造的字典.</param>
        /// <returns>新的 godot 词典.</returns>
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
        /// 处理这个 <see cref="Dictionary"/>。
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
        /// 复制此 <see cref="Dictionary"/>。
        /// </summary>
        /// <param name="deep">如果<see langword="true"/>，执行深拷贝.</param>
        /// <returns>新的 Godot 词典.</returns>
        public Dictionary Duplicate(bool deep = false)
        {
            return new Dictionary(godot_icall_Dictionary_Duplicate(GetPtr(), deep));
        }

        // IDictionary

        /// <summary>
        /// 获取此 <see cref="Dictionary"/> 中的键集合。
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
        /// 获取此 <see cref="Dictionary"/> 中的元素集合。
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
        /// 返回给定 <paramref name="key"/> 处的对象。
        /// </summary>
        /// <value>给定 <paramref name="key"/> 处的对象.</value>
        public object this[object key]
        {
            get => godot_icall_Dictionary_GetValue(GetPtr(), key);
            set => godot_icall_Dictionary_SetValue(GetPtr(), key, value);
        }

        /// <summary>
        /// 在键 <paramref name="key"/> 处添加对象 <paramref name="value"/>
        /// 到这个<see cref="Dictionary"/>。
        /// </summary>
        /// <param name="key">添加对象的键.</param>
        /// <param name="value">要添加的对象.</param>
        public void Add(object key, object value) => godot_icall_Dictionary_Add(GetPtr(), key, value);

        /// <summary>
        /// 删除此 <see cref="Dictionary"/> 中的所有项目。
        /// </summary>
        public void Clear() => godot_icall_Dictionary_Clear(GetPtr());

        /// <summary>
        /// 检查此 <see cref="Dictionary"/> 是否包含给定的键。
        /// </summary>
        /// <param name="key">寻找的键.</param>
        /// <returns>这个字典是否包含给定的键.</returns>
        public bool Contains(object key) => godot_icall_Dictionary_ContainsKey(GetPtr(), key);

        /// <summary>
        /// 获取此 <see cref="Dictionary"/> 的枚举器。
        /// </summary>
        /// <returns>枚举器.</returns>
        public IDictionaryEnumerator GetEnumerator() => new DictionaryEnumerator(this);

        /// <summary>
        /// 按键从此 <see cref="Dictionary"/> 中删除一个元素。
        /// </summary>
        /// <param name="key">要删除的元素的键.</param>
        public void Remove(object key) => godot_icall_Dictionary_RemoveKey(GetPtr(), key);

        // ICollection

        object ICollection.SyncRoot => this;

        bool ICollection.IsSynchronized => false;

        /// <summary>
        /// 返回此 <see cref="Dictionary"/> 中的元素数。
        /// 这也称为字典的大小或长度。
        /// </summary>
        /// <returns>元素数量.</returns>
        public int Count => godot_icall_Dictionary_Count(GetPtr());

        /// <summary>
        /// 将此 <see cref="Dictionary"/> 的元素复制到给定的
        /// 无类型的 C# 数组，从给定的索引开始。
        /// </summary>
        /// <param name="array">要复制到的数组.</param>
        /// <param name="index">开始的索引.</param>
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
        /// 将此 <see cref="Dictionary"/> 转换为字符串。
        /// </summary>
        /// <returns>此字典的字符串表示形式.</returns>
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
    /// 围绕 Godot 的 Dictionary 类的类型化包装，一个 Variant 的字典
    /// 在 C++ 引擎中分配的类型化元素。 有用的时候
    /// 与引擎交互。 否则更喜欢 .NET 集合
    /// 例如 <see cref="System.Collections.Generic.Dictionary{TKey, TValue}"/>。
    /// </summary>
    /// <typeparam name="TKey">字典键的类型.</typeparam>
    /// <typeparam name="TValue">字典值的类型.</typeparam>
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
        /// 构造一个新的空 <see cref="Dictionary{TKey, TValue}"/>。
        /// </summary>
        public Dictionary()
        {
            _objectDict = new Dictionary();
        }

        /// <summary>
        ///从给定字典的元素构造一个新的 <see cref="Dictionary{TKey, TValue}"/>。
        /// </summary>
        /// <param name="dictionary">要构造的字典.</param>
        /// <returns>新的 Godot 词典.</returns>
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
        /// 从给定字典的元素构造一个新的 <see cref="Dictionary{TKey, TValue}"/>。
        /// </summary>
        /// <param name="dictionary">要构造的字典.</param>
        /// <returns>新的 Godot 词典.</returns>
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
        /// 将此类型化的 <see cref="Dictionary{TKey, TValue}"/> 转换为非类型化的 <see cref="Dictionary"/>。
        /// </summary>
        /// <param name="from">要转换的类型化字典.</param>
        public static explicit operator Dictionary(Dictionary<TKey, TValue> from)
        {
            return from._objectDict;
        }

        internal IntPtr GetPtr()
        {
            return _objectDict.GetPtr();
        }

        /// <summary>
        /// 复制此 <see cref="Dictionary{TKey, TValue}"/>。
        /// </summary>
        /// <param name="deep">如果<see langword="true"/>，执行深拷贝.</param>
        /// <returns>新的 Godot 词典.</returns>
        public Dictionary<TKey, TValue> Duplicate(bool deep = false)
        {
            return new Dictionary<TKey, TValue>(_objectDict.Duplicate(deep));
        }

        // IDictionary<TKey, TValue>

        /// <summary>
        /// 返回给定 <paramref name="key"/> 的值。
        /// </summary>
        /// <value>给定 <paramref name="key"/> 的值.</value>
        public TValue this[TKey key]
        {
            get { return (TValue)Dictionary.godot_icall_Dictionary_GetValue_Generic(_objectDict.GetPtr(), key, valTypeEncoding, valTypeClass); }
            set { _objectDict[key] = value; }
        }

        /// <summary>
        /// 获取此 <see cref="Dictionary{TKey, TValue}"/> 中的键集合。
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
        /// 获取此 <see cref="Dictionary{TKey, TValue}"/> 中的元素集合。
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
        /// 在键 <paramref name="key"/> 处添加一个对象 <paramref name="value"/>
        /// 到这个 <see cref="Dictionary{TKey, TValue}"/>。
        /// </summary>
        /// <param name="key">添加对象的键.</param>
        /// <param name="value">要添加的对象.</param>
        public void Add(TKey key, TValue value)
        {
            _objectDict.Add(key, value);
        }

        /// <summary>
        /// 检查这个 <see cref="Dictionary{TKey, TValue}"/> 是否包含给定的键。
        /// </summary>
        /// <param name="key">寻找的键.</param>
        /// <returns>这个字典是否包含给定的键。</returns>
        public bool ContainsKey(TKey key)
        {
            return _objectDict.Contains(key);
        }

        /// <summary>
        /// 按键从此 <see cref="Dictionary{TKey, TValue}"/> 中删除一个元素。
        /// </summary>
        /// <param name="key">要移除的元素的键。</param>
        public bool Remove(TKey key)
        {
            return Dictionary.godot_icall_Dictionary_RemoveKey(GetPtr(), key);
        }

        /// <summary>
        /// 获取给定 <paramref name="key"/> 处的对象。
        /// </summary>
        /// <param name="key">要获取的元素的key。</param>
        /// <param name="value">给定<paramref name="key"/>的值。</param>
        /// <returns>如果为给定的 <paramref name="key"/> 找到对象。</returns>
        public bool TryGetValue(TKey key, out TValue value)
        {
            bool found = Dictionary.godot_icall_Dictionary_TryGetValue_Generic(GetPtr(), key, out object retValue, valTypeEncoding, valTypeClass);
            value = found ? (TValue)retValue : default;
            return found;
        }

        // ICollection<KeyValuePair<TKey, TValue>>

        /// <summary>
        /// 返回此 <see cref="Dictionary{TKey, TValue}"/> 中的元素数。
        /// 这也称为字典的大小或长度。
        /// </summary>
        /// <returns>元素个数。</returns>
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
        /// 删除此 <see cref="Dictionary{TKey, TValue}"/> 中的所有项目。
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
        /// 将此 <see cref="Dictionary{TKey, TValue}"/> 的元素复制到给定的
        /// 无类型的 C# 数组，从给定的索引开始。
        /// </summary>
        /// <param name="array">要复制到的数组。</param>
        /// <param name="arrayIndex">开始的索引。</param>
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
        /// 获取此 <see cref="Dictionary{TKey, TValue}"/> 的枚举器。
        /// </summary>
        /// <returns>枚举器.</returns>
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
        /// 将此 <see cref="Dictionary{TKey, TValue}"/> 转换为字符串。
        /// </summary>
        /// <returns>此字典的字符串表示形式.</returns>
        public override string ToString() => _objectDict.ToString();
    }
}
