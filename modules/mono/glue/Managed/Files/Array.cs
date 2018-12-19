using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Collections
{
    class ArraySafeHandle : SafeHandle
    {
        public ArraySafeHandle(IntPtr handle) : base(IntPtr.Zero, true)
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
            Array.godot_icall_Array_Dtor(handle);
            return true;
        }
    }

    public class Array : IList<object>, ICollection<object>, IEnumerable<object>, IDisposable
    {
        ArraySafeHandle safeHandle;
        bool disposed = false;

        public Array()
        {
            safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor());
        }

        internal Array(ArraySafeHandle handle)
        {
            safeHandle = handle;
        }

        internal Array(IntPtr handle)
        {
            safeHandle = new ArraySafeHandle(handle);
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

        public object this[int index]
        {
            get
            {
                return godot_icall_Array_At(GetPtr(), index);
            }
            set
            {
                godot_icall_Array_SetAt(GetPtr(), index, value);
            }
        }

        public int Count
        {
            get
            {
                return godot_icall_Array_Count(GetPtr());
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return false;
            }
        }

        public void Add(object item)
        {
            godot_icall_Array_Add(GetPtr(), item);
        }

        public void Clear()
        {
            godot_icall_Array_Clear(GetPtr());
        }

        public bool Contains(object item)
        {
            return godot_icall_Array_Contains(GetPtr(), item);
        }

        public void CopyTo(object[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            // Internal call may throw ArgumentException
            godot_icall_Array_CopyTo(GetPtr(), array, arrayIndex);
        }

        public IEnumerator<object> GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        public int IndexOf(object item)
        {
            return godot_icall_Array_IndexOf(GetPtr(), item);
        }

        public void Insert(int index, object item)
        {
            godot_icall_Array_Insert(GetPtr(), index, item);
        }

        public bool Remove(object item)
        {
            return godot_icall_Array_Remove(GetPtr(), item);
        }

        public void RemoveAt(int index)
        {
            godot_icall_Array_RemoveAt(GetPtr(), index);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static IntPtr godot_icall_Array_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_Array_At(IntPtr ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static object godot_icall_Array_At_Generic(IntPtr ptr, int index, int elemTypeEncoding, IntPtr elemTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_SetAt(IntPtr ptr, int index, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_Array_Count(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_Add(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_Clear(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Array_Contains(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_CopyTo(IntPtr ptr, object[] array, int arrayIndex);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static int godot_icall_Array_IndexOf(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_Insert(IntPtr ptr, int index, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static bool godot_icall_Array_Remove(IntPtr ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_RemoveAt(IntPtr ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal extern static void godot_icall_Array_Generic_GetElementTypeInfo(Type elemType, out int elemTypeEncoding, out IntPtr elemTypeClass);
    }

    public class Array<T> : IList<T>, ICollection<T>, IEnumerable<T>
    {
        Array objectArray;

        internal static int elemTypeEncoding;
        internal static IntPtr elemTypeClass;

        static Array()
        {
            Array.godot_icall_Array_Generic_GetElementTypeInfo(typeof(T), out elemTypeEncoding, out elemTypeClass);
        }

        public Array()
        {
            objectArray = new Array();
        }

        public Array(Array array)
        {
            objectArray = array;
        }

        internal Array(IntPtr handle)
        {
            objectArray = new Array(handle);
        }

        internal Array(ArraySafeHandle handle)
        {
            objectArray = new Array(handle);
        }

        public static explicit operator Array(Array<T> from)
        {
            return from.objectArray;
        }

        public T this[int index]
        {
            get
            {
                return (T)Array.godot_icall_Array_At_Generic(GetPtr(), index, elemTypeEncoding, elemTypeClass);
            }
            set
            {
                objectArray[index] = value;
            }
        }

        public int Count
        {
            get
            {
                return objectArray.Count;
            }
        }

        public bool IsReadOnly
        {
            get
            {
                return objectArray.IsReadOnly;
            }
        }

        public void Add(T item)
        {
            objectArray.Add(item);
        }

        public void Clear()
        {
            objectArray.Clear();
        }

        public bool Contains(T item)
        {
            return objectArray.Contains(item);
        }

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

        public IEnumerator<T> GetEnumerator()
        {
            int count = objectArray.Count;

            for (int i = 0; i < count; i++)
            {
                yield return (T)this[i];
            }
        }

        public int IndexOf(T item)
        {
            return objectArray.IndexOf(item);
        }

        public void Insert(int index, T item)
        {
            objectArray.Insert(index, item);
        }

        public bool Remove(T item)
        {
            return objectArray.Remove(item);
        }

        public void RemoveAt(int index)
        {
            objectArray.RemoveAt(index);
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        internal IntPtr GetPtr()
        {
            return objectArray.GetPtr();
        }
    }
}
