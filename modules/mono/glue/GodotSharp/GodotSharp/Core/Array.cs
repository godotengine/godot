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
    public sealed class Array :
        IList<Variant>,
        IReadOnlyList<Variant>,
        ICollection,
        IDisposable
    {
        internal godot_array.movable NativeValue;

        private WeakReference<IDisposable> _weakReferenceToSelf;

        /// <summary>
        /// Constructs a new empty <see cref="Array"/>.
        /// </summary>
        public Array()
        {
            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
        }

        /// <summary>
        /// Constructs a new <see cref="Array"/> from the given collection's elements.
        /// </summary>
        /// <param name="collection">The collection of elements to construct from.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(IEnumerable<Variant> collection) : this()
        {
            if (collection == null)
                throw new ArgumentNullException(nameof(collection));

            foreach (Variant element in collection)
                Add(element);
        }

        /// <summary>
        /// Constructs a new <see cref="Array"/> from the given objects.
        /// </summary>
        /// <param name="array">The objects to put in the new array.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(Variant[] array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);

            int length = array.Length;

            Resize(length);

            for (int i = 0; i < length; i++)
                this[i] = array[i];
        }

        public Array(Span<StringName> array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);

            int length = array.Length;

            Resize(length);

            for (int i = 0; i < length; i++)
                this[i] = array[i];
        }

        public Array(Span<NodePath> array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);

            int length = array.Length;

            Resize(length);

            for (int i = 0; i < length; i++)
                this[i] = array[i];
        }

        public Array(Span<Rid> array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);

            int length = array.Length;

            Resize(length);

            for (int i = 0; i < length; i++)
                this[i] = array[i];
        }

        // We must use ReadOnlySpan instead of Span here as this can accept implicit conversions
        // from derived types (e.g.: Node[]). Implicit conversion from Derived[] to Base[] are
        // fine as long as the array is not mutated. However, Span does this type checking at
        // instantiation, so it's not possible to use it even when not mutating anything.
        // ReSharper disable once RedundantNameQualifier
        public Array(ReadOnlySpan<GodotObject> array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new();
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);

            int length = array.Length;

            Resize(length);

            for (int i = 0; i < length; i++)
                this[i] = array[i];
        }

        private Array(godot_array nativeValueToOwn)
        {
            NativeValue = (godot_array.movable)(nativeValueToOwn.IsAllocated ?
                nativeValueToOwn :
                NativeFuncs.godotsharp_array_new());
            _weakReferenceToSelf = DisposablesTracker.RegisterDisposable(this);
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
            NativeValue.DangerousSelfRef.Dispose();

            if (_weakReferenceToSelf != null)
            {
                DisposablesTracker.UnregisterDisposable(_weakReferenceToSelf);
            }
        }

        /// <summary>
        /// Duplicates this <see cref="Array"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array Duplicate(bool deep = false)
        {
            godot_array newArray;
            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_duplicate(ref self, deep.ToGodotBool(), out newArray);
            return CreateTakingOwnershipOfDisposableValue(newArray);
        }

        /// <summary>
        /// Resizes this <see cref="Array"/> to the given size.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            ThrowIfReadOnly();

            var self = (godot_array)NativeValue;
            return NativeFuncs.godotsharp_array_resize(ref self, newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array"/> into a random order.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        public void Shuffle()
        {
            ThrowIfReadOnly();

            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_shuffle(ref self);
        }

        /// <summary>
        /// Concatenates these two <see cref="Array"/>s.
        /// </summary>
        /// <param name="left">The first array.</param>
        /// <param name="right">The second array.</param>
        /// <returns>A new Godot Array with the contents of both arrays.</returns>
        public static Array operator +(Array left, Array right)
        {
            if (left == null)
            {
                if (right == null)
                    return new Array();

                return right.Duplicate(deep: false);
            }

            if (right == null)
                return left.Duplicate(deep: false);

            int leftCount = left.Count;
            int rightCount = right.Count;

            Array newArray = left.Duplicate(deep: false);
            newArray.Resize(leftCount + rightCount);

            for (int i = 0; i < rightCount; i++)
                newArray[i + leftCount] = right[i];

            return newArray;
        }

        /// <summary>
        /// Returns the item at the given <paramref name="index"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The property is assigned and the array is read-only.
        /// </exception>
        /// <value>The <see cref="Variant"/> item at the given <paramref name="index"/>.</value>
        public unsafe Variant this[int index]
        {
            get
            {
                GetVariantBorrowElementAt(index, out godot_variant borrowElem);
                return Variant.CreateCopyingBorrowed(borrowElem);
            }
            set
            {
                ThrowIfReadOnly();

                if (index < 0 || index >= Count)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var self = (godot_array)NativeValue;
                godot_variant* ptrw = NativeFuncs.godotsharp_array_ptrw(ref self);
                godot_variant* itemPtr = &ptrw[index];
                (*itemPtr).Dispose();
                *itemPtr = value.CopyNativeVariant();
            }
        }

        /// <summary>
        /// Adds an item to the end of this <see cref="Array"/>.
        /// This is the same as <c>append</c> or <c>push_back</c> in GDScript.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="item">The <see cref="Variant"/> item to add.</param>
        public void Add(Variant item)
        {
            ThrowIfReadOnly();

            godot_variant variantValue = (godot_variant)item.NativeVar;
            var self = (godot_array)NativeValue;
            _ = NativeFuncs.godotsharp_array_add(ref self, variantValue);
        }

        /// <summary>
        /// Checks if this <see cref="Array"/> contains the given item.
        /// </summary>
        /// <param name="item">The <see cref="Variant"/> item to look for.</param>
        /// <returns>Whether or not this array contains the given item.</returns>
        public bool Contains(Variant item) => IndexOf(item) != -1;

        /// <summary>
        /// Erases all items from this <see cref="Array"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        public void Clear() => Resize(0);

        /// <summary>
        /// Searches this <see cref="Array"/> for an item
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="item">The <see cref="Variant"/> item to search for.</param>
        /// <returns>The index of the item, or -1 if not found.</returns>
        public int IndexOf(Variant item)
        {
            godot_variant variantValue = (godot_variant)item.NativeVar;
            var self = (godot_array)NativeValue;
            return NativeFuncs.godotsharp_array_index_of(ref self, variantValue);
        }

        /// <summary>
        /// Inserts a new item at a given position in the array.
        /// The position must be a valid position of an existing item,
        /// or the position at the end of the array.
        /// Existing items will be moved to the right.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="index">The index to insert at.</param>
        /// <param name="item">The <see cref="Variant"/> item to insert.</param>
        public void Insert(int index, Variant item)
        {
            ThrowIfReadOnly();

            if (index < 0 || index > Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            godot_variant variantValue = (godot_variant)item.NativeVar;
            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_insert(ref self, index, variantValue);
        }

        /// <summary>
        /// Removes the first occurrence of the specified <paramref name="item"/>
        /// from this <see cref="Array"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="item">The value to remove.</param>
        public bool Remove(Variant item)
        {
            ThrowIfReadOnly();

            int index = IndexOf(item);
            if (index >= 0)
            {
                RemoveAt(index);
                return true;
            }

            return false;
        }

        /// <summary>
        /// Removes an element from this <see cref="Array"/> by index.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="index">The index of the element to remove.</param>
        public void RemoveAt(int index)
        {
            ThrowIfReadOnly();

            if (index < 0 || index > Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_remove_at(ref self, index);
        }

        // ICollection

        /// <summary>
        /// Returns the number of elements in this <see cref="Array"/>.
        /// This is also known as the size or length of the array.
        /// </summary>
        /// <returns>The number of elements.</returns>
        public int Count => NativeValue.DangerousSelfRef.Size;

        bool ICollection.IsSynchronized => false;

        object ICollection.SyncRoot => false;

        /// <summary>
        /// Returns <see langword="true"/> if the array is read-only.
        /// See <see cref="MakeReadOnly"/>.
        /// </summary>
        public bool IsReadOnly => NativeValue.DangerousSelfRef.IsReadOnly;

        /// <summary>
        /// Makes the <see cref="Array"/> read-only, i.e. disabled modying of the
        /// array's elements. Does not apply to nested content, e.g. content of
        /// nested arrays.
        /// </summary>
        public void MakeReadOnly()
        {
            if (IsReadOnly)
            {
                // Avoid interop call when the array is already read-only.
                return;
            }

            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_make_read_only(ref self);
        }

        /// <summary>
        /// Copies the elements of this <see cref="Array"/> to the given
        /// <see cref="Variant"/> C# array, starting at the given index.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        /// <param name="arrayIndex">The index to start at.</param>
        public void CopyTo(Variant[] array, int arrayIndex)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (arrayIndex < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(arrayIndex),
                    "Number was less than the array's lower bound in the first dimension.");
            }

            int count = Count;

            if (array.Length < (arrayIndex + count))
            {
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");
            }

            unsafe
            {
                for (int i = 0; i < count; i++)
                {
                    array[arrayIndex] = Variant.CreateCopyingBorrowed(NativeValue.DangerousSelfRef.Elements[i]);
                    arrayIndex++;
                }
            }
        }

        void ICollection.CopyTo(System.Array array, int index)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array), "Value cannot be null.");

            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(index),
                    "Number was less than the array's lower bound in the first dimension.");
            }

            int count = Count;

            if (array.Length < (index + count))
            {
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");
            }

            unsafe
            {
                for (int i = 0; i < count; i++)
                {
                    object boxedVariant = Variant.CreateCopyingBorrowed(NativeValue.DangerousSelfRef.Elements[i]);
                    array.SetValue(boxedVariant, index);
                    index++;
                }
            }
        }

        // IEnumerable

        /// <summary>
        /// Gets an enumerator for this <see cref="Array"/>.
        /// </summary>
        /// <returns>An enumerator.</returns>
        public IEnumerator<Variant> GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>
        /// Converts this <see cref="Array"/> to a string.
        /// </summary>
        /// <returns>A string representation of this array.</returns>
        public override string ToString()
        {
            var self = (godot_array)NativeValue;
            NativeFuncs.godotsharp_array_to_string(ref self, out godot_string str);
            using (str)
                return Marshaling.ConvertStringToManaged(str);
        }

        /// <summary>
        /// The variant returned via the <paramref name="elem"/> parameter is owned by the Array and must not be disposed.
        /// </summary>
        internal void GetVariantBorrowElementAt(int index, out godot_variant elem)
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));
            GetVariantBorrowElementAtUnchecked(index, out elem);
        }

        /// <summary>
        /// The variant returned via the <paramref name="elem"/> parameter is owned by the Array and must not be disposed.
        /// </summary>
        internal unsafe void GetVariantBorrowElementAtUnchecked(int index, out godot_variant elem)
        {
            elem = NativeValue.DangerousSelfRef.Elements[index];
        }

        private void ThrowIfReadOnly()
        {
            if (IsReadOnly)
            {
                throw new InvalidOperationException("Array instance is read-only.");
            }
        }
    }

    internal interface IGenericGodotArray
    {
        public Array UnderlyingArray { get; }
    }

    /// <summary>
    /// Typed wrapper around Godot's Array class, an array of Variant
    /// typed elements allocated in the engine in C++. Useful when
    /// interfacing with the engine. Otherwise prefer .NET collections
    /// such as arrays or <see cref="List{T}"/>.
    /// </summary>
    /// <typeparam name="T">The type of the array.</typeparam>
    [SuppressMessage("ReSharper", "RedundantExtendsListEntry")]
    [SuppressMessage("Naming", "CA1710", MessageId = "Identifiers should have correct suffix")]
    public sealed class Array<[MustBeVariant] T> :
        IList<T>,
        IReadOnlyList<T>,
        ICollection<T>,
        IEnumerable<T>,
        IGenericGodotArray
    {
        private static godot_variant ToVariantFunc(in Array<T> godotArray) =>
            VariantUtils.CreateFromArray(godotArray);

        private static Array<T> FromVariantFunc(in godot_variant variant) =>
            VariantUtils.ConvertToArray<T>(variant);

        static unsafe Array()
        {
            VariantUtils.GenericConversion<Array<T>>.ToVariantCb = &ToVariantFunc;
            VariantUtils.GenericConversion<Array<T>>.FromVariantCb = &FromVariantFunc;
        }

        private readonly Array _underlyingArray;

        Array IGenericGodotArray.UnderlyingArray => _underlyingArray;

        internal ref godot_array.movable NativeValue
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => ref _underlyingArray.NativeValue;
        }

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
                throw new ArgumentNullException(nameof(collection));

            _underlyingArray = new Array();

            foreach (T element in collection)
                Add(element);
        }

        /// <summary>
        /// Constructs a new <see cref="Array{T}"/> from the given items.
        /// </summary>
        /// <param name="array">The items to put in the new array.</param>
        /// <returns>A new Godot Array.</returns>
        public Array(T[] array) : this()
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            _underlyingArray = new Array();

            foreach (T element in array)
                Add(element);
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
            return from?._underlyingArray;
        }

        /// <summary>
        /// Duplicates this <see cref="Array{T}"/>.
        /// </summary>
        /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
        /// <returns>A new Godot Array.</returns>
        public Array<T> Duplicate(bool deep = false)
        {
            return new Array<T>(_underlyingArray.Duplicate(deep));
        }

        /// <summary>
        /// Resizes this <see cref="Array{T}"/> to the given size.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="newSize">The new size of the array.</param>
        /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
        public Error Resize(int newSize)
        {
            return _underlyingArray.Resize(newSize);
        }

        /// <summary>
        /// Shuffles the contents of this <see cref="Array{T}"/> into a random order.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
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
            if (left == null)
            {
                if (right == null)
                    return new Array<T>();

                return right.Duplicate(deep: false);
            }

            if (right == null)
                return left.Duplicate(deep: false);

            return new Array<T>(left._underlyingArray + right._underlyingArray);
        }

        // IList<T>

        /// <summary>
        /// Returns the value at the given <paramref name="index"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The property is assigned and the array is read-only.
        /// </exception>
        /// <value>The value at the given <paramref name="index"/>.</value>
        public unsafe T this[int index]
        {
            get
            {
                _underlyingArray.GetVariantBorrowElementAt(index, out godot_variant borrowElem);
                return VariantUtils.ConvertTo<T>(borrowElem);
            }
            set
            {
                ThrowIfReadOnly();

                if (index < 0 || index >= Count)
                    throw new ArgumentOutOfRangeException(nameof(index));

                var self = (godot_array)_underlyingArray.NativeValue;
                godot_variant* ptrw = NativeFuncs.godotsharp_array_ptrw(ref self);
                godot_variant* itemPtr = &ptrw[index];
                (*itemPtr).Dispose();
                *itemPtr = VariantUtils.CreateFrom(value);
            }
        }

        /// <summary>
        /// Searches this <see cref="Array{T}"/> for an item
        /// and returns its index or -1 if not found.
        /// </summary>
        /// <param name="item">The item to search for.</param>
        /// <returns>The index of the item, or -1 if not found.</returns>
        public int IndexOf(T item)
        {
            using var variantValue = VariantUtils.CreateFrom(item);
            var self = (godot_array)_underlyingArray.NativeValue;
            return NativeFuncs.godotsharp_array_index_of(ref self, variantValue);
        }

        /// <summary>
        /// Inserts a new item at a given position in the <see cref="Array{T}"/>.
        /// The position must be a valid position of an existing item,
        /// or the position at the end of the array.
        /// Existing items will be moved to the right.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="index">The index to insert at.</param>
        /// <param name="item">The item to insert.</param>
        public void Insert(int index, T item)
        {
            ThrowIfReadOnly();

            if (index < 0 || index > Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            using var variantValue = VariantUtils.CreateFrom(item);
            var self = (godot_array)_underlyingArray.NativeValue;
            NativeFuncs.godotsharp_array_insert(ref self, index, variantValue);
        }

        /// <summary>
        /// Removes an element from this <see cref="Array{T}"/> by index.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
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

        /// <summary>
        /// Returns <see langword="true"/> if the array is read-only.
        /// See <see cref="MakeReadOnly"/>.
        /// </summary>
        public bool IsReadOnly => _underlyingArray.IsReadOnly;

        /// <summary>
        /// Makes the <see cref="Array{T}"/> read-only, i.e. disabled modying of the
        /// array's elements. Does not apply to nested content, e.g. content of
        /// nested arrays.
        /// </summary>
        public void MakeReadOnly()
        {
            _underlyingArray.MakeReadOnly();
        }

        /// <summary>
        /// Adds an item to the end of this <see cref="Array{T}"/>.
        /// This is the same as <c>append</c> or <c>push_back</c> in GDScript.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="item">The item to add.</param>
        /// <returns>The new size after adding the item.</returns>
        public void Add(T item)
        {
            ThrowIfReadOnly();

            using var variantValue = VariantUtils.CreateFrom(item);
            var self = (godot_array)_underlyingArray.NativeValue;
            _ = NativeFuncs.godotsharp_array_add(ref self, variantValue);
        }

        /// <summary>
        /// Erases all items from this <see cref="Array{T}"/>.
        /// </summary>
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        public void Clear()
        {
            _underlyingArray.Clear();
        }

        /// <summary>
        /// Checks if this <see cref="Array{T}"/> contains the given item.
        /// </summary>
        /// <param name="item">The item to look for.</param>
        /// <returns>Whether or not this array contains the given item.</returns>
        public bool Contains(T item) => IndexOf(item) != -1;

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
            {
                throw new ArgumentOutOfRangeException(nameof(arrayIndex),
                    "Number was less than the array's lower bound in the first dimension.");
            }

            int count = Count;

            if (array.Length < (arrayIndex + count))
            {
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");
            }

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
        /// <exception cref="InvalidOperationException">
        /// The array is read-only.
        /// </exception>
        /// <param name="item">The value to remove.</param>
        /// <returns>A <see langword="bool"/> indicating success or failure.</returns>
        public bool Remove(T item)
        {
            ThrowIfReadOnly();

            int index = IndexOf(item);
            if (index >= 0)
            {
                RemoveAt(index);
                return true;
            }

            return false;
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static implicit operator Variant(Array<T> from) => Variant.CreateFrom(from);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static explicit operator Array<T>(Variant from) => from.AsGodotArray<T>();

        private void ThrowIfReadOnly()
        {
            if (IsReadOnly)
            {
                throw new InvalidOperationException("Array instance is read-only.");
            }
        }
    }
}
