using System;
using System.Collections.Generic;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Godot.Collections
{
    internal class ArraySafeHandle : SafeHandle
    {
        public ArraySafeHandle(IntPtr handle) : base(IntPtr.Zero, true)
        {
            this.handle = handle;
        }

        public override bool IsInvalid
        {
            get { return handle == IntPtr.Zero; }
        }

        protected override bool ReleaseHandle()
        {
            Array.godot_icall_Array_Dtor(handle);
            return true;
        }
    }

    /// <summary>
    /// Wrapper around Godot's Array class, an array of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as <see cref="System.Array"/> or <see cref="List{T}"/>.
    /// </summary>
    public class Array : IList, IDisposable
    {
        private ArraySafeHandle _safeHandle;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a new empty <see cref="Array"/>.
        /// </summary>
        public Array()
        {
            _safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor());
        }

        /// <summary>
        /// Constructs a new <see cref="Array"/> from the given collection's elements.
        /// </summary>
        /// <param name="collection">The collection of elements to construct from.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(IEnumerable collection) : this()
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            foreach (object element in collection)
                Add(element);
        }

        /// <summary>
        /// Constructs a new <see cref="Array"/> from the given objects.
        /// </summary>
        /// <param name="array">The objects to put in the new array.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(params object[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }
            _safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor_MonoArray(array));
        }

        internal Array(ArraySafeHandle handle)
        {
            _safeHandle = handle;
        }

        internal Array(IntPtr handle)
        {
            _safeHandle = new ArraySafeHandle(handle);
        }

        internal IntPtr GetPtr()
        {
            if (_disposed)
                throw new ObjectDisposedException(GetType().FullName);

            return _safeHandle.DangerousGetHandle();
        }

        /// <summary>
        /// Duplicates this <see cref="Array"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array Duplicate(bool deep = false)
        {
            return new Array(godot_icall_Array_Duplicate(GetPtr(), deep));
        }

        /// <summary>
        /// Resizes this <see cref="Array"/> to the given size.
        /// </summary>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            return godot_icall_Array_Resize(GetPtr(), newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array"/> into a random order.
        /// </summary>
        public void Shuffle()
        {
            godot_icall_Array_Shuffle(GetPtr());
        }

        /// <summary>
        /// Concatenates these two <see cref="Array"/>s.
        /// </summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns>A new Godot Array with the contents of both arrays.</returns>
        public static Array operator +(Array left, Array right)
        {
            return new Array(godot_icall_Array_Concatenate(left.GetPtr(), right.GetPtr()));
        }

        // IDisposable

        /// <summary>
        /// Disposes of this <see cref="Array"/>.
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

        // IList

        bool IList.IsReadOnly => false;

        bool IList.IsFixedSize => false;

        /// <summary>
        /// Returns the object at the given <paramref name="index"/>.
        /// </summary>
        /// <value>The object at the given <paramref name="index"/>.</value>
        public object this[int index]
        {
            get => godot_icall_Array_At(GetPtr(), index);
            set => godot_icall_Array_SetAt(GetPtr(), index, value);
        }

        /// <summary>
        /// Adds an object to the end of this <see cref="Array"/>.
        /// This is the same as <c>append</c> or <c>push_back</c> in GDScript.
        /// </summary>
        /// <param name="value">The object to add.</param>
        /// <returns>The new size after adding the object.</returns>
        public int Add(object value) => godot_icall_Array_Add(GetPtr(), value);

        /// <summary>
        /// Checks if this <see cref="Array"/> contains the given object.
        /// </summary>
        /// <param name="value">The item to look for.</param>
        /// <returns>Whether or not this array contains the given object.</returns>
        public bool Contains(object value) => godot_icall_Array_Contains(GetPtr(), value);

        /// <summary>
        /// Erases all items from this <see cref="Array"/>.
        /// </summary>
        public void Clear() => godot_icall_Array_Clear(GetPtr());

        /// <summary>
        /// Searches this <see cref="Array"/> for an object
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="value">The object to search for.</param>
        /// <returns>The index of the object, or -1 if not found.</returns>
        public int IndexOf(object value) => godot_icall_Array_IndexOf(GetPtr(), value);

        /// <summary>
        /// Inserts a new object at a given position in the array.
        /// The position must be a valid position of an existing item,
        /// or the position at the end of the array.
        /// Existing items will be moved to the right.
        /// </summary>
        /// <param name="index">The index to insert at.</param>
        /// <param name="value">The object to insert.</param>
        public void Insert(int index, object value) => godot_icall_Array_Insert(GetPtr(), index, value);

        /// <summary>
        /// Removes the first occurrence of the specified <paramref name="value"/>
        /// from this <see cref="Array"/>.
        /// </summary>
        /// <param name="value">The value to remove.</param>
        public void Remove(object value) => godot_icall_Array_Remove(GetPtr(), value);

        /// <summary>
        /// Removes an element from this <see cref="Array"/> by index.
        /// </summary>
        /// <param name="index">The index of the element to remove.</param>
        public void RemoveAt(int index) => godot_icall_Array_RemoveAt(GetPtr(), index);

        // ICollection

        /// <summary>
        /// Returns the number of elements in this <see cref="Array"/>.
        /// This is also known as the size or length of the array.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => godot_icall_Array_Count(GetPtr());

        object ICollection.SyncRoot => this;

        bool ICollection.IsSynchronized => false;

        /// <summary>
        /// Copies the elements of this <see cref="Array"/> to the given
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

            // Internal call may throw ArgumentException
            godot_icall_Array_CopyTo(GetPtr(), array, index);
        }

        // IEnumerable

        /// <summary>
        /// Gets an enumerator for this <see cref="Array"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        /// <summary>
        /// Converts this <see cref="Array"/> to a string.
        /// </summary>
        /// <returns>A string representation of this array.</returns>
        public override string ToString()
        {
            return godot_icall_Array_ToString(GetPtr());
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Ctor();

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern IntPtr godot_icall_Array_Ctor_MonoArray(System.Array array);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Dtor(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Array_At(IntPtr ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern object godot_icall_Array_At_Generic(IntPtr ptr, int index, int elemTypeEncoding, IntPtr elemTypeClass);

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
        internal static extern void godot_icall_Array_Shuffle(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Generic_GetElementTypeInfo(Type elemType, out int elemTypeEncoding, out IntPtr elemTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Array_ToString(IntPtr ptr);
    }

    /// <summary>
    /// Typed wrapper around Godot's Array class, an array of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as arrays or <see cref="List{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the array.</typeparam>
    public class Array<T> : IList<T>, ICollection<T>, IEnumerable<T>
    {
        private Array _objectArray;

        internal static int elemTypeEncoding;
        internal static IntPtr elemTypeClass;

        static Array()
        {
            Array.godot_icall_Array_Generic_GetElementTypeInfo(typeof(T), out elemTypeEncoding, out elemTypeClass);
        }

        /// <summary>
        /// Constructs a new empty <see cref="Array{T}"/>.
        /// </summary>
        public Array()
        {
            _objectArray = new Array();
        }

        /// <summary>
        /// Constructs a new <see cref="Array{T}"/> from the given collection's elements.
        /// </summary>
        /// <param name="collection">The collection of elements to construct from.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            _objectArray = new Array(collection);
        }

        /// <summary>
        /// Constructs a new <see cref="Array{T}"/> from the given items.
        /// </summary>
        /// <param name="array">The items to put in the new array.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(params T[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }
            _objectArray = new Array(array);
        }

        /// <summary>
        /// Constructs a typed <see cref="Array{T}"/> from an untyped <see cref="Array"/>.
        /// </summary>
        /// <param name="array">The untyped array to construct from.</param>
        public Array(Array array)
        {
            _objectArray = array;
        }

        internal Array(IntPtr handle)
        {
            _objectArray = new Array(handle);
        }

        internal Array(ArraySafeHandle handle)
        {
            _objectArray = new Array(handle);
        }

        internal IntPtr GetPtr()
        {
            return _objectArray.GetPtr();
        }

        /// <summary>
        /// Converts this typed <see cref="Array{T}"/> to an untyped <see cref="Array"/>.
        /// </summary>
        /// <param name="from">The typed array to convert.</param>
        public static explicit operator Array(Array<T> from)
        {
            return from._objectArray;
        }

        /// <summary>
        /// Duplicates this <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array<T> Duplicate(bool deep = false)
        {
            return new Array<T>(_objectArray.Duplicate(deep));
        }

        /// <summary>
        /// Resizes this <see cref="Array{T}"/> to the given size.
        /// </summary>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            return _objectArray.Resize(newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array{T}"/> into a random order.
        /// </summary>
        public void Shuffle()
        {
            _objectArray.Shuffle();
        }

        /// <summary>
        /// Concatenates these two <see cref="Array{T}"/>s.
        /// </summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns>A new Godot Array with the contents of both arrays.</returns>
        public static Array<T> operator +(Array<T> left, Array<T> right)
        {
            return new Array<T>(left._objectArray + right._objectArray);
        }

        // IList<T>

        /// <summary>
        /// Returns the value at the given <paramref name="index"/>.
        /// </summary>
        /// <value>The value at the given <paramref name="index"/>.</value>
        public T this[int index]
        {
            get { return (T)Array.godot_icall_Array_At_Generic(GetPtr(), index, elemTypeEncoding, elemTypeClass); }
            set { _objectArray[index] = value; }
        }

        /// <summary>
        /// Searches this <see cref="Array{T}"/> for an item
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="item">The item to search for.</param>
        /// <returns>The index of the item, or -1 if not found.</returns>
        public int IndexOf(T item)
        {
            return _objectArray.IndexOf(item);
        }

        /// <summary>
        /// Inserts a new item at a given position in the <see cref="Array{T}"/>.
        /// The position must be a valid position of an existing item,
        /// or the position at the end of the array.
        /// Existing items will be moved to the right.
        /// </summary>
        /// <param name="index">The index to insert at.</param>
        /// <param name="item">The item to insert.</param>
        public void Insert(int index, T item)
        {
            _objectArray.Insert(index, item);
        }

        /// <summary>
        /// Removes an element from this <see cref="Array{T}"/> by index.
        /// </summary>
        /// <param name="index">The index of the element to remove.</param>
        public void RemoveAt(int index)
        {
            _objectArray.RemoveAt(index);
        }

        // ICollection<T>

        /// <summary>
        /// Returns the number of elements in this <see cref="Array{T}"/>.
        /// This is also known as the size or length of the array.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count
        {
            get { return _objectArray.Count; }
        }

        bool ICollection<T>.IsReadOnly => false;

        /// <summary>
        /// Adds an item to the end of this <see cref="Array{T}"/>.
        /// This is the same as <c>append</c> or <c>push_back</c> in GDScript.
        /// </summary>
        /// <param name="item">The item to add.</param>
        /// <returns>The new size after adding the item.</returns>
        public void Add(T item)
        {
            _objectArray.Add(item);
        }

        /// <summary>
        /// Erases all items from this <see cref="Array{T}"/>.
        /// </summary>
        public void Clear()
        {
            _objectArray.Clear();
        }

        /// <summary>
        /// Checks if this <see cref="Array{T}"/> contains the given item.
        /// </summary>
        /// <param name="item">The item to look for.</param>
        /// <returns>Whether or not this array contains the given item.</returns>
        public bool Contains(T item)
        {
            return _objectArray.Contains(item);
        }

        /// <summary>
        /// Copies the elements of this <see cref="Array{T}"/> to the given
        /// C# array, starting at the given index.
        /// </summary>
        /// <param name="array">The C# array to copy to.</param>
        /// <param name="arrayIndex">The index to start at.</param>
        public void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
                throw new ArgumentOutOfRangeException(nameof(arrayIndex), "Number was less than the array's lower bound in the first dimension.");

            // TODO This may be quite slow because every element access is an internal call.
            // It could be moved entirely to an internal call if we find out how to do the cast there.

            int count = _objectArray.Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = (T)this[i];
                arrayIndex++;
            }
        }

        /// <summary>
        /// Removes the first occurrence of the specified value
        /// from this <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="item">The value to remove.</param>
        /// <returns>A <see langword="bool"/> indicating success or failure.</returns>
        public bool Remove(T item)
        {
            return Array.godot_icall_Array_Remove(GetPtr(), item);
        }

        // IEnumerable<T>

        /// <summary>
        /// Gets an enumerator for this <see cref="Array{T}"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            int count = _objectArray.Count;

            for (int i = 0; i < count; i++)
            {
                yield return (T)this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        /// <summary>
        /// Converts this <see cref="Array{T}"/> to a string.
        /// </summary>
        /// <returns>A string representation of this array.</returns>
        public override string ToString() => _objectArray.ToString();
    }
}
