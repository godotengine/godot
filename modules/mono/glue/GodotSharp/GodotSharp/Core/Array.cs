using System;
using System.Collections.Generic;
using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot.Collections
{
    /// <summary>
    /// Wrapper around Godot's Array class, an array of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as <see cref="System.Array"/> or <see cref="List{T}"/>.
    /// </summary>
    public sealed class Array : IList, IDisposable
    {
        internal godot_array NativeValue;

        /// <summary>
        /// Constructs a new empty <see cref="Array"/>.
        /// </summary>
        public Array()
        {
            godot_icall_Array_Ctor(out NativeValue);
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

        // TODO: This must be removed. Lots of silent mistakes as it takes pretty much anything.
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

            godot_icall_Array_Ctor_MonoArray(array, out NativeValue);
        }

        private Array(godot_array nativeValueToOwn)
        {
            NativeValue = nativeValueToOwn;
        }

        // Explicit name to make it very clear
        internal static Array CreateTakingOwnershipOfDisposableValue(godot_array nativeValueToOwn)
            => new Array(nativeValueToOwn);

        ~Array()
        {
            Dispose(false);
        }

        /// <summary>
        /// Disposes of this <see cref="Array"/>.
        /// </summary>
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

        /// <summary>
        /// Duplicates this <see cref="Array"/>.
        /// </summary>
        /// <param name="deep">If true, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array Duplicate(bool deep = false)
        {
            godot_array newArray;
            godot_icall_Array_Duplicate(ref NativeValue, deep, out newArray);
            return CreateTakingOwnershipOfDisposableValue(newArray);
        }

        /// <summary>
        /// Resizes this <see cref="Array"/> to the given size.
        /// </summary>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            return godot_icall_Array_Resize(ref NativeValue, newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array"/> into a random order.
        /// </summary>
        public void Shuffle()
        {
            godot_icall_Array_Shuffle(ref NativeValue);
        }

        /// <summary>
        /// Concatenates these two <see cref="Array"/>s.
        /// </summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns>A new Godot Array with the contents of both arrays.</returns>
        public static Array operator +(Array left, Array right)
        {
            godot_array newArray;
            godot_icall_Array_Concatenate(ref left.NativeValue, ref right.NativeValue, out newArray);
            return CreateTakingOwnershipOfDisposableValue(newArray);
        }

        // IList

        bool IList.IsReadOnly => false;

        bool IList.IsFixedSize => false;

        /// <summary>
        /// Returns the object at the given index.
        /// </summary>
        /// <value>The object at the given index.</value>
        public object this[int index]
        {
            get
            {
                godot_icall_Array_At(ref NativeValue, index, out godot_variant elem);
                unsafe
                {
                    using (elem)
                        return Marshaling.variant_to_mono_object(&elem);
                }
            }
            set => godot_icall_Array_SetAt(ref NativeValue, index, value);
        }

        /// <summary>
        /// Adds an object to the end of this <see cref="Array"/>.
        /// This is the same as `append` or `push_back` in GDScript.
        /// </summary>
        /// <param name="value">The object to add.</param>
        /// <returns>The new size after adding the object.</returns>
        public int Add(object value) => godot_icall_Array_Add(ref NativeValue, value);

        /// <summary>
        /// Checks if this <see cref="Array"/> contains the given object.
        /// </summary>
        /// <param name="value">The item to look for.</param>
        /// <returns>Whether or not this array contains the given object.</returns>
        public bool Contains(object value) => godot_icall_Array_Contains(ref NativeValue, value);

        /// <summary>
        /// Erases all items from this <see cref="Array"/>.
        /// </summary>
        public void Clear() => godot_icall_Array_Clear(ref NativeValue);

        /// <summary>
        /// Searches this <see cref="Array"/> for an object
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="value">The object to search for.</param>
        /// <returns>The index of the object, or -1 if not found.</returns>
        public int IndexOf(object value) => godot_icall_Array_IndexOf(ref NativeValue, value);

        /// <summary>
        /// Inserts a new object at a given position in the array.
        /// The position must be a valid position of an existing item,
        /// or the position at the end of the array.
        /// Existing items will be moved to the right.
        /// </summary>
        /// <param name="index">The index to insert at.</param>
        /// <param name="value">The object to insert.</param>
        public void Insert(int index, object value) => godot_icall_Array_Insert(ref NativeValue, index, value);

        /// <summary>
        /// Removes the first occurrence of the specified value
        /// from this <see cref="Array"/>.
        /// </summary>
        /// <param name="value">The value to remove.</param>
        public void Remove(object value) => godot_icall_Array_Remove(ref NativeValue, value);

        /// <summary>
        /// Removes an element from this <see cref="Array"/> by index.
        /// </summary>
        /// <param name="index">The index of the element to remove.</param>
        public void RemoveAt(int index) => godot_icall_Array_RemoveAt(ref NativeValue, index);

        // ICollection

        /// <summary>
        /// Returns the number of elements in this <see cref="Array"/>.
        /// This is also known as the size or length of the array.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => godot_icall_Array_Count(ref NativeValue);

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
            godot_icall_Array_CopyTo(ref NativeValue, array, index);
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
            return godot_icall_Array_ToString(ref NativeValue);
        }

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Ctor(out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Ctor_MonoArray(System.Array array, out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_At(ref godot_array ptr, int index, out godot_variant elem);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_SetAt(ref godot_array ptr, int index, object value);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_Count(ref godot_array ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_Add(ref godot_array ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Clear(ref godot_array ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Concatenate(ref godot_array left, ref godot_array right, out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Array_Contains(ref godot_array ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_CopyTo(ref godot_array ptr, System.Array array, int arrayIndex);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Duplicate(ref godot_array ptr, bool deep, out godot_array dest);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern int godot_icall_Array_IndexOf(ref godot_array ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Insert(ref godot_array ptr, int index, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern bool godot_icall_Array_Remove(ref godot_array ptr, object item);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_RemoveAt(ref godot_array ptr, int index);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_Array_Resize(ref godot_array ptr, int newSize);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern Error godot_icall_Array_Shuffle(ref godot_array ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Array_ToString(ref godot_array ptr);
    }

    internal interface IGenericGodotArray
    {
        Array UnderlyingArray { get; }
        Type TypeOfElements { get; }
    }

    // TODO: Now we should be able to avoid boxing
    /// <summary>
    /// Typed wrapper around Godot's Array class, an array of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as arrays or <see cref="List{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the array.</typeparam>
    [SuppressMessage("ReSharper", "RedundantExtendsListEntry")]
    public sealed class Array<T> : IList<T>, ICollection<T>, IEnumerable<T>, IGenericGodotArray
    {
        private readonly Array _underlyingArray;

        // ReSharper disable StaticMemberInGenericType
        // Warning is about unique static fields being created for each generic type combination:
        // https://www.jetbrains.com/help/resharper/StaticMemberInGenericType.html
        // In our case this is exactly what we want.
        private static readonly Type TypeOfElements = typeof(T);
        // ReSharper restore StaticMemberInGenericType

        Array IGenericGodotArray.UnderlyingArray => _underlyingArray;
        Type IGenericGodotArray.TypeOfElements => TypeOfElements;

        /// <summary>
        /// Constructs a new empty <see cref="Array{T}"/>.
        /// </summary>
        public Array()
        {
            _underlyingArray = new Array();
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

            _underlyingArray = new Array(collection);
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

            _underlyingArray = new Array(array);
        }

        /// <summary>
        /// Constructs a typed <see cref="Array{T}"/> from an untyped <see cref="Array"/>.
        /// </summary>
        /// <param name="array">The untyped array to construct from.</param>
        public Array(Array array)
        {
            _underlyingArray = array;
        }

        // Explicit name to make it very clear
        internal static Array<T> CreateTakingOwnershipOfDisposableValue(godot_array nativeValueToOwn)
            => new Array<T>(Array.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn));

        /// <summary>
        /// Converts this typed <see cref="Array{T}"/> to an untyped <see cref="Array"/>.
        /// </summary>
        /// <param name="from">The typed array to convert.</param>
        public static explicit operator Array(Array<T> from)
        {
            return from._underlyingArray;
        }

        /// <summary>
        /// Duplicates this <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="deep">If true, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array<T> Duplicate(bool deep = false)
        {
            return new Array<T>(_underlyingArray.Duplicate(deep));
        }

        /// <summary>
        /// Resizes this <see cref="Array{T}"/> to the given size.
        /// </summary>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            return _underlyingArray.Resize(newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array{T}"/> into a random order.
        /// </summary>
        public void Shuffle()
        {
            _underlyingArray.Shuffle();
        }

        /// <summary>
        /// Concatenates these two <see cref="Array{T}"/>s.
        /// </summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns>A new Godot Array with the contents of both arrays.</returns>
        public static Array<T> operator +(Array<T> left, Array<T> right)
        {
            return new Array<T>(left._underlyingArray + right._underlyingArray);
        }

        // IList<T>

        /// <summary>
        /// Returns the value at the given index.
        /// </summary>
        /// <value>The value at the given index.</value>
        public T this[int index]
        {
            get
            {
                Array.godot_icall_Array_At(ref _underlyingArray.NativeValue, index, out godot_variant elem);
                unsafe
                {
                    using (elem)
                        return (T)Marshaling.variant_to_mono_object_of_type(&elem, TypeOfElements);
                }
            }
            set => _underlyingArray[index] = value;
        }

        /// <summary>
        /// Searches this <see cref="Array{T}"/> for an item
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="item">The item to search for.</param>
        /// <returns>The index of the item, or -1 if not found.</returns>
        public int IndexOf(T item)
        {
            return _underlyingArray.IndexOf(item);
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
            _underlyingArray.Insert(index, item);
        }

        /// <summary>
        /// Removes an element from this <see cref="Array{T}"/> by index.
        /// </summary>
        /// <param name="index">The index of the element to remove.</param>
        public void RemoveAt(int index)
        {
            _underlyingArray.RemoveAt(index);
        }

        // ICollection<T>

        /// <summary>
        /// Returns the number of elements in this <see cref="Array{T}"/>.
        /// This is also known as the size or length of the array.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => _underlyingArray.Count;

        bool ICollection<T>.IsReadOnly => false;

        /// <summary>
        /// Adds an item to the end of this <see cref="Array{T}"/>.
        /// This is the same as `append` or `push_back` in GDScript.
        /// </summary>
        /// <param name="item">The item to add.</param>
        /// <returns>The new size after adding the item.</returns>
        public void Add(T item)
        {
            _underlyingArray.Add(item);
        }

        /// <summary>
        /// Erases all items from this <see cref="Array{T}"/>.
        /// </summary>
        public void Clear()
        {
            _underlyingArray.Clear();
        }

        /// <summary>
        /// Checks if this <see cref="Array{T}"/> contains the given item.
        /// </summary>
        /// <param name="item">The item to look for.</param>
        /// <returns>Whether or not this array contains the given item.</returns>
        public bool Contains(T item)
        {
            return _underlyingArray.Contains(item);
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

            int count = _underlyingArray.Count;

            if (array.Length < (arrayIndex + count))
                throw new ArgumentException("Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = this[i];
                arrayIndex++;
            }
        }

        /// <summary>
        /// Removes the first occurrence of the specified value
        /// from this <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="item">The value to remove.</param>
        /// <returns>A bool indicating success or failure.</returns>
        public bool Remove(T item)
        {
            return Array.godot_icall_Array_Remove(ref _underlyingArray.NativeValue, item);
        }

        // IEnumerable<T>

        /// <summary>
        /// Gets an enumerator for this <see cref="Array{T}"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<T> GetEnumerator()
        {
            int count = _underlyingArray.Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Converts this <see cref="Array{T}"/> to a string.
        /// </summary>
        /// <returns>A string representation of this array.</returns>
        public override string ToString() => _underlyingArray.ToString();
    }
}
