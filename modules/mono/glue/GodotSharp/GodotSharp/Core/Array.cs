using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Collections
{
    internal class ArraySafeHandle : SafeHandle
    {
        public ArraySafeHandle(IntPtr handle) : base(IntPtr.Zero, true) => this.handle = handle;

        public override bool IsInvalid => handle == IntPtr.Zero;

        protected override bool ReleaseHandle()
        {
            Array.godot_icall_Array_Dtor(handle);
            return true;
        }
    }

    public class Array : IList, IDisposable
    {
        private ArraySafeHandle _safeHandle;
        private bool _disposed = false;

        public Array() => _safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor());

        public Array(IEnumerable collection) : this()
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            foreach (object element in collection)
                Add(element);
        }

        public Array(params object[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }

            _safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor_MonoArray(array));
        }

        internal Array(ArraySafeHandle handle) => _safeHandle = handle;

        internal Array(IntPtr handle) => _safeHandle = new ArraySafeHandle(handle);

        internal IntPtr GetPtr()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().FullName);

            return _safeHandle.DangerousGetHandle();
        }

        public Array Duplicate(bool deep = false) => new Array(godot_icall_Array_Duplicate(GetPtr(), deep));

        public Error Resize(int newSize) => godot_icall_Array_Resize(GetPtr(), newSize);

        public void Shuffle() => godot_icall_Array_Shuffle(GetPtr());

        public static Array operator +(Array left, Array right) =>
            new Array(godot_icall_Array_Concatenate(left.GetPtr(), right.GetPtr()));

        // IDisposable

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

        // IList

        public bool IsReadOnly => false;

        public bool IsFixedSize => false;

        public object this[int index]
        {
            get => godot_icall_Array_At(GetPtr(), index);
            set => godot_icall_Array_SetAt(GetPtr(), index, value);
        }

        public int Add(object value) => godot_icall_Array_Add(GetPtr(), value);

        public bool Contains(object value) => godot_icall_Array_Contains(GetPtr(), value);

        public void Clear() => godot_icall_Array_Clear(GetPtr());

        public int IndexOf(object value) => godot_icall_Array_IndexOf(GetPtr(), value);

        public void Insert(int index, object value) => godot_icall_Array_Insert(GetPtr(), index, value);

        public void Remove(object value) => godot_icall_Array_Remove(GetPtr(), value);

        public void RemoveAt(int index) => godot_icall_Array_RemoveAt(GetPtr(), index);

        // ICollection

        public int Count => godot_icall_Array_Count(GetPtr());

        public object SyncRoot => this;

        public bool IsSynchronized => false;

        public void CopyTo(System.Array array, int index)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (index < 0)
                throw new ArgumentOutOfRangeException(nameof(index), "Number was less than the array's lower bound in the first dimension.");

            // Internal call may throw ArgumentException
            godot_icall_Array_CopyTo(GetPtr(), array, index);
        }

        // IEnumerable

        public IEnumerator GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        public override string ToString() => godot_icall_Array_ToString(GetPtr());

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Ctor_MonoArray(System.Array array);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Array_At(IntPtr ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Array_At_Generic(IntPtr ptr, int index, int elemTypeEncoding,
            IntPtr elemTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_SetAt(IntPtr ptr, int index, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_Count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_Add(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Clear(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Concatenate(IntPtr left, IntPtr right);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Array_Contains(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_CopyTo(IntPtr ptr, System.Array array, int arrayIndex);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Duplicate(IntPtr ptr, bool deep);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_IndexOf(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Insert(IntPtr ptr, int index, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Array_Remove(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_RemoveAt(IntPtr ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_Array_Resize(IntPtr ptr, int newSize);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_Array_Shuffle(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Generic_GetElementTypeInfo(Type elemType,
            out int elemTypeEncoding, out IntPtr elemTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Array_ToString(IntPtr ptr);
    }

    public class Array<T> : IList<T>, ICollection<T>, IEnumerable<T>
    {
        private Array objectArray;

        internal static int elemTypeEncoding;
        internal static IntPtr elemTypeClass;

        static Array() =>
            Array.godot_icall_Array_Generic_GetElementTypeInfo(typeof(T), out elemTypeEncoding, out elemTypeClass);

        public Array() => objectArray = new Array();

        public Array(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            objectArray = new Array(collection);
        }

        public Array(params T[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }

            objectArray = new Array(array);
        }

        public Array(Array array) => objectArray = array;

        internal Array(IntPtr handle) => objectArray = new Array(handle);

        internal Array(ArraySafeHandle handle) => objectArray = new Array(handle);

        internal IntPtr GetPtr() => objectArray.GetPtr();

        public static explicit operator Array(Array<T> from) => from.objectArray;

        public Array<T> Duplicate(bool deep = false) => new Array<T>(objectArray.Duplicate(deep));

        public Error Resize(int newSize) => objectArray.Resize(newSize);

        public void Shuffle() => objectArray.Shuffle();

        public static Array<T> operator +(Array<T> left, Array<T> right) =>
            new Array<T>(left.objectArray + right.objectArray);

        // IList<T>

        public T this[int index]
        {
            get => (T)Array.godot_icall_Array_At_Generic(GetPtr(), index, elemTypeEncoding, elemTypeClass);
            set => objectArray[index] = value;
        }

        public int IndexOf(T item) => objectArray.IndexOf(item);

        public void Insert(int index, T item) => objectArray.Insert(index, item);

        public void RemoveAt(int index) => objectArray.RemoveAt(index);

        // ICollection<T>

        public int Count => objectArray.Count;

        public bool IsReadOnly => objectArray.IsReadOnly;

        public void Add(T item) => objectArray.Add(item);

        public void Clear() => objectArray.Clear();

        public bool Contains(T item) => objectArray.Contains(item);

        public void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            // TODO This may be quite slow because every element access is an internal call.
            // It could be moved entirely to an internal call if we find out how to do the cast there.

            int count = objectArray.Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = (T)this[i];
                arrayIndex++;
            }
        }

        public bool Remove(T item) => Array.godot_icall_Array_Remove(GetPtr(), item);

        // IEnumerable<T>

        public IEnumerator<T> GetEnumerator()
        {
            int count = objectArray.Count;

            for (int i = 0; i < count; i++)
            {
                yield return (T)this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => objectArray.ToString();
    }
}
