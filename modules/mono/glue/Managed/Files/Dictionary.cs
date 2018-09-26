using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Collections
{
    class DictionarySafeHandle : SafeHandle
    {
        public DictionarySafeHandle(IntPtr handle) : base(IntPtr.Zero, true)
        {
            this.handle = handle;
        }

        public override bool IsInvalid
        {
            get
            {
                return handle == IntPtr.Zero;
            }
        }

        protected override bool ReleaseHandle()
        {
            Dictionary.godot_icall_Dictionary_Dtor(handle);
            return true;
        }
    }

    public class Dictionary :
        IDictionary<object, object>,
        ICollection<KeyValuePair<object, object>>,
        IEnumerable<KeyValuePair<object, object>>,
        IDisposable
    {
        DictionarySafeHandle safeHandle;
        bool disposed = false;

        public Dictionary()
        {
            safeHandle = new DictionarySafeHandle(godot_icall_Dictionary_Ctor());
        }

        internal Dictionary(DictionarySafeHandle handle)
        {
            safeHandle = handle;
        }

        internal Dictionary(IntPtr handle)
        {
            safeHandle = new DictionarySafeHandle(handle);
        }

        internal IntPtr GetPtr()
        {
            return safeHandle.DangerousGetHandle();
        }

        public void Dispose()
        {
            if (disposed)
                return;

            if (safeHandle != null)
            {
                safeHandle.Dispose();
                safeHandle = null;
            }

            disposed = true;
        }

        public object this[object key]
        {
            get
            {
                return godot_icall_Dictionary_GetValue(GetPtr(), key);
            }
            set
            {
                godot_icall_Dictionary_SetValue(GetPtr(), key, value);
            }
        }

        public ICollection<object> Keys
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Keys(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        public ICollection<object> Values
        {
            get
            {
                IntPtr handle = godot_icall_Dictionary_Values(GetPtr());
                return new Array(new ArraySafeHandle(handle));
            }
        }

        public int Count
        {
            get
            {
                return godot_icall_Dictionary_Count(GetPtr());
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return false;
            }
        }

        public void Add(object key, object value)
        {
            godot_icall_Dictionary_Add(GetPtr(), key, value);
        }

        public void Add(KeyValuePair<object, object> item)
        {
            Add(item.Key, item.Value);
        }

        public void Clear()
        {
            godot_icall_Dictionary_Clear(GetPtr());
        }

        public bool Contains(KeyValuePair<object, object> item)
        {
            return godot_icall_Dictionary_Contains(GetPtr(), item.Key, item.Value);
        }

        public bool ContainsKey(object key)
        {
            return godot_icall_Dictionary_ContainsKey(GetPtr(), key);
        }

        public void CopyTo(KeyValuePair<object, object>[] array, int arrayIndex)
        {
            // TODO 3 internal calls, can reduce to 1
            Array keys = (Array)Keys;
            Array values = (Array)Values;
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                array[arrayIndex] = new KeyValuePair<object, object>(keys[i], values[i]);
                arrayIndex++;
            }
        }

        public IEnumerator<KeyValuePair<object, object>> GetEnumerator()
        {
            // TODO 3 internal calls, can reduce to 1
            Array keys = (Array)Keys;
            Array values = (Array)Values;
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                yield return new KeyValuePair<object, object>(keys[i], values[i]);
            }
        }

        public bool Remove(object key)
        {
            return godot_icall_Dictionary_RemoveKey(GetPtr(), key);
        }

        public bool Remove(KeyValuePair<object, object> item)
        {
            return godot_icall_Dictionary_Remove(GetPtr(), item.Key, item.Value);
        }

        public bool TryGetValue(object key, out object value)
        {
            object retValue;
            bool found = godot_icall_Dictionary_TryGetValue(GetPtr(), key, out retValue);
            value = found ? retValue : default(object);
            return found;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Dictionary_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Dictionary_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_Dictionary_GetValue(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_Dictionary_GetValue_Generic(IntPtr ptr, object key, int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Dictionary_SetValue(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Dictionary_Keys(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Dictionary_Values(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_Dictionary_Count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Dictionary_Add(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Dictionary_Clear(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_Contains(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_ContainsKey(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_RemoveKey(IntPtr ptr, object key);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_Remove(IntPtr ptr, object key, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_TryGetValue(IntPtr ptr, object key, out object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Dictionary_TryGetValue_Generic(IntPtr ptr, object key, out object value, int valTypeEncoding, IntPtr valTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Dictionary_Generic_GetValueTypeInfo(Type valueType, out int valTypeEncoding, out IntPtr valTypeClass);
    }

    public class Dictionary<TKey, TValue> :
        IDictionary<TKey, TValue>,
        ICollection<KeyValuePair<TKey, TValue>>,
        IEnumerable<KeyValuePair<TKey, TValue>>
    {
        Dictionary objectDict;

        internal static int valTypeEncoding;
        internal static IntPtr valTypeClass;

        static Dictionary()
        {
            Dictionary.godot_icall_Dictionary_Generic_GetValueTypeInfo(typeof(TValue), out valTypeEncoding, out valTypeClass);
        }

        public Dictionary()
        {
            objectDict = new Dictionary();
        }

        public Dictionary(Dictionary dictionary)
        {
            objectDict = dictionary;
        }

        internal Dictionary(IntPtr handle)
        {
            objectDict = new Dictionary(handle);
        }

        internal Dictionary(DictionarySafeHandle handle)
        {
            objectDict = new Dictionary(handle);
        }

        public static explicit operator Dictionary(Dictionary<TKey, TValue> from)
        {
            return from.objectDict;
        }

        public TValue this[TKey key]
        {
            get
            {
                return (TValue)Dictionary.godot_icall_Dictionary_GetValue_Generic(objectDict.GetPtr(), key, valTypeEncoding, valTypeClass);
            }
            set
            {
                objectDict[key] = value;
            }
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

        public int Count
        {
            get
            {
                return objectDict.Count;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return objectDict.IsReadOnly;
            }
        }

        public void Add(TKey key, TValue value)
        {
            objectDict.Add(key, value);
        }

        public void Add(KeyValuePair<TKey, TValue> item)
        {
            objectDict.Add(item.Key, item.Value);
        }

        public void Clear()
        {
            objectDict.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
        {
            return objectDict.Contains(new KeyValuePair<object, object>(item.Key, item.Value));
        }

        public bool ContainsKey(TKey key)
        {
            return objectDict.ContainsKey(key);
        }

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
        {
            // TODO 3 internal calls, can reduce to 1
            Array<TKey> keys = (Array<TKey>)Keys;
            Array<TValue> values = (Array<TValue>)Values;
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                // TODO 2 internal calls, can reduce to 1
                array[arrayIndex] = new KeyValuePair<TKey, TValue>(keys[i], values[i]);
                arrayIndex++;
            }
        }

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

        public bool Remove(TKey key)
        {
            return objectDict.Remove(key);
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            return objectDict.Remove(new KeyValuePair<object, object>(item.Key, item.Value));
        }

        public bool TryGetValue(TKey key, out TValue value)
        {
            object retValue;
            bool found = Dictionary.godot_icall_Dictionary_TryGetValue_Generic(GetPtr(), key, out retValue, valTypeEncoding, valTypeClass);
            value = found ? (TValue)retValue : default(TValue);
            return found;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        internal IntPtr GetPtr()
        {
            return objectDict.GetPtr();
        }
    }
}
