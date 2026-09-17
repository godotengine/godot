using System;
using System.Collections;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace Godot.NativeInterop.UnsafeCollections;

internal struct UnsafeGodotArray :
    IList<Variant>,
    IReadOnlyList<Variant>,
    ICollection,
    IDisposable
{
    internal godot_array.movable NativeValue;

    public struct Borrowing(UnsafeGodotArray target) : IEnumerable<Variant>
    {
        public UnsafeGodotArray Target = target;

        [MustDisposeResource]
        public Enumerator GetEnumerator() => Target.GetEnumerator();

        [MustDisposeResource]
        IEnumerator<Variant> IEnumerable<Variant>.GetEnumerator() => Target.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() =>
            throw new InvalidOperationException(
                $"IEnumerable.GetEnumerator() cannot be implemented for this '{typeof(Borrowing)}' "
                + "because IEnumerable doesn't implement IDisposable and would leak the native resources.");

        public static implicit operator Variant(Borrowing unsafeArray) =>
            Variant.CreateConsuming(VariantUtils.CreateFromArray((godot_array)unsafeArray.Target.NativeValue));
    }

    public struct Consuming(UnsafeGodotArray target) : IDisposable, IEnumerable<Variant>
    {
        public UnsafeGodotArray Target = target;

        public void Dispose() => Target.Dispose();

        [MustDisposeResource]
        public Enumerator GetEnumerator() => Target.GetEnumerator();

        [MustDisposeResource]
        IEnumerator<Variant> IEnumerable<Variant>.GetEnumerator() => Target.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() =>
            throw new InvalidOperationException(
                $"IEnumerable.GetEnumerator() cannot be implemented for this '{typeof(Consuming)}' "
                + "because IEnumerable doesn't implement IDisposable and would leak the native resources.");
    }

    public static UnsafeGodotArray Create() =>
        new() { NativeValue = (godot_array.movable)NativeFuncs.godotsharp_array_new() };

    public static Consuming CreateConsuming(godot_array consumingNativeValue) =>
        new(new UnsafeGodotArray(consumingNativeValue, consuming: true));

    public static Borrowing CreateBorrowing(godot_array borrowingNativeValue) =>
        new(new UnsafeGodotArray(borrowingNativeValue, consuming: false));

    private UnsafeGodotArray(godot_array nativeValue, bool consuming)
    {
        if (consuming)
        {
            NativeValue = (godot_array.movable)(nativeValue.IsAllocated
                ? nativeValue
                : NativeFuncs.godotsharp_array_new());
        }
        else
        {
            if (!nativeValue.IsAllocated)
                throw new ArgumentException($"The {nameof(nativeValue)} native array is not allocated.");
            NativeValue = (godot_array.movable)nativeValue;
        }
    }

    public readonly Consuming CloneDisposable() =>
        CreateConsuming(NativeFuncs.godotsharp_array_new_copy((godot_array)NativeValue));

    public readonly Borrowing BorrowDisposable() => new(this);

    public void Dispose() => NativeValue.DangerousSelfRef.Dispose();

    public readonly int Count => NativeValue.DangerousSelfRef.Size;
    readonly int ICollection.Count => Count;
    readonly int ICollection<Variant>.Count => Count;
    readonly int IReadOnlyCollection<Variant>.Count => Count;

    public struct Enumerator : IEnumerator<Variant>
    {
        private UnsafeGodotArray _consumingUnsafeArray;
        private int _index;
        private readonly int _count;

        internal Enumerator(Consuming consumingUnsafeArray, int count)
        {
            _consumingUnsafeArray = consumingUnsafeArray.Target;
            _index = -1;
            _count = count;
        }

        public bool MoveNext()
        {
            _index++;
            return _index < _count;
        }

        public void Reset() => _index = -1;

        Variant IEnumerator<Variant>.Current => _consumingUnsafeArray[Current];

        object IEnumerator.Current => Current;

        public int Current => _index;

        public void Dispose() => _consumingUnsafeArray.Dispose();
    }

    public readonly Enumerator GetEnumerator() => new(CloneDisposable(), Count);

    readonly IEnumerator<Variant> IEnumerable<Variant>.GetEnumerator() => GetEnumerator();

    readonly IEnumerator IEnumerable.GetEnumerator() =>
        throw new InvalidOperationException(
            $"IEnumerable.GetEnumerator() cannot be implemented for this '{typeof(UnsafeGodotArray)}' "
            + "because IEnumerable doesn't implement IDisposable and would leak the native resources.");

    public readonly void Add(Variant borrowedItem)
    {
        ThrowIfReadOnly();

        godot_variant variantValue = (godot_variant)borrowedItem.NativeVar;
        var self = (godot_array)NativeValue;
        _ = NativeFuncs.godotsharp_array_add(ref self, variantValue);
    }

    /// <summary>
    /// Returns a copy of the <see cref="UnsafeGodotArray"/>.
    /// If <paramref name="deep"/> is <see langword="true"/>, a deep copy if performed:
    /// all nested arrays and dictionaries are duplicated and will not be shared with
    /// the original array. If <see langword="false"/>, a shallow copy is made and
    /// references to the original nested arrays and dictionaries are kept, so that
    /// modifying a sub-array or dictionary in the copy will also impact those
    /// referenced in the source array. Note that any <see cref="GodotObject"/> derived
    /// elements will be shallow copied regardless of the <paramref name="deep"/>
    /// setting.
    /// </summary>
    /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
    /// <returns>A new Godot Array.</returns>
    public readonly Consuming Duplicate(bool deep = false)
    {
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_duplicate(ref self, deep.ToGodotBool(), out godot_array newArray);
        return CreateConsuming(newArray);
    }

    /// <summary>
    /// Assigns the given value to all elements in the array. This can typically be
    /// used together with <see cref="Resize(int)"/> to create an array with a given
    /// size and initialized elements.
    /// Note: If <paramref name="value"/> is of a reference type (<see cref="GodotObject"/>
    /// derived, <see cref="Collections.Array"/> or <see cref="Collections.Dictionary"/>, etc.) then the array
    /// is filled with the references to the same object, i.e. no duplicates are
    /// created.
    /// </summary>
    /// <example>
    /// <code>
    /// var array = new Godot.Collections.Array();
    /// array.Resize(10);
    /// array.Fill(0); // Initialize the 10 elements to 0.
    /// </code>
    /// </example>
    /// <exception cref="InvalidOperationException">
    /// The array is read-only.
    /// </exception>
    /// <param name="value">The value to fill the array with.</param>
    public readonly void Fill(Variant value)
    {
        ThrowIfReadOnly();

        godot_variant variantValue = (godot_variant)value.NativeVar;
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_fill(ref self, variantValue);
    }

    /// <summary>
    /// Returns the maximum value contained in the array if all elements are of
    /// comparable types. If the elements can't be compared, <see langword="null"/>
    /// is returned.
    /// </summary>
    /// <returns>The maximum value contained in the array.</returns>
    public readonly Variant Max()
    {
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_max(ref self, out godot_variant resVariant);
        return Variant.CreateConsuming(resVariant);
    }

    /// <summary>
    /// Returns the minimum value contained in the array if all elements are of
    /// comparable types. If the elements can't be compared, <see langword="null"/>
    /// is returned.
    /// </summary>
    /// <returns>The minimum value contained in the array.</returns>
    public readonly Variant Min()
    {
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_min(ref self, out godot_variant resVariant);
        return Variant.CreateConsuming(resVariant);
    }

    /// <summary>
    /// Returns a random value from the target array.
    /// </summary>
    /// <example>
    /// <code>
    /// var array = new Godot.Collections.Array { 1, 2, 3, 4 };
    /// GD.Print(array.PickRandom()); // Prints either of the four numbers.
    /// </code>
    /// </example>
    /// <returns>A random element from the array.</returns>
    public readonly Variant PickRandom()
    {
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_pick_random(ref self, out godot_variant resVariant);
        return Variant.CreateConsuming(resVariant);
    }

    /// <summary>
    /// Compares this <see cref="UnsafeGodotArray"/> against the <paramref name="other"/>
    /// <see cref="UnsafeGodotArray"/> recursively. Returns <see langword="true"/> if the
    /// sizes and contents of the arrays are equal, <see langword="false"/>
    /// otherwise.
    /// </summary>
    /// <param name="other">The other array to compare against.</param>
    /// <returns>
    /// <see langword="true"/> if the sizes and contents of the arrays are equal,
    /// <see langword="false"/> otherwise.
    /// </returns>
    public readonly bool RecursiveEqual(Borrowing other)
    {
        var self = (godot_array)NativeValue;
        var otherVariant = (godot_array)other.Target.NativeValue;
        return NativeFuncs.godotsharp_array_recursive_equal(ref self, otherVariant).ToBool();
    }

    /// <summary>
    /// Resizes the array to contain a different number of elements. If the array
    /// size is smaller, elements are cleared, if bigger, new elements are
    /// <see langword="null"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The array is read-only.
    /// </exception>
    /// <param name="newSize">The new size of the array.</param>
    /// <returns><see cref="Error.Ok"/> if successful, or an error code.</returns>
    public readonly Error Resize(int newSize)
    {
        ThrowIfReadOnly();

        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_resize(ref self, newSize);
    }

    /// <summary>
    /// Reverses the order of the elements in the array.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The array is read-only.
    /// </exception>
    public readonly void Reverse()
    {
        ThrowIfReadOnly();

        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_reverse(ref self);
    }

    /// <summary>
    /// Shuffles the array such that the items will have a random order.
    /// This method uses the global random number generator common to methods
    /// such as <see cref="GD.Randi"/>. Call <see cref="GD.Randomize"/> to
    /// ensure that a new seed will be used each time if you want
    /// non-reproducible shuffling.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The array is read-only.
    /// </exception>
    public readonly void Shuffle()
    {
        ThrowIfReadOnly();

        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_shuffle(ref self);
    }

    /// <summary>
    /// Creates a shallow copy of a range of elements in the source <see cref="UnsafeGodotArray"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="start"/> is less than 0 or greater than the array's size.
    /// </exception>
    /// <param name="start">The zero-based index at which the range starts.</param>
    /// <returns>A new array that contains the elements inside the slice range.</returns>
    public readonly Consuming Slice(int start)
    {
        if (start < 0 || start > Count)
            throw new ArgumentOutOfRangeException(nameof(start));

        return GetSliceRange(start, Count, step: 1, deep: false);
    }

    /// <summary>
    /// Creates a shallow copy of a range of elements in the source <see cref="UnsafeGodotArray"/>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="start"/> is less than 0 or greater than the array's size.
    /// -or-
    /// <paramref name="length"/> is less than 0 or greater than the array's size.
    /// </exception>
    /// <param name="start">The zero-based index at which the range starts.</param>
    /// <param name="length">The length of the range.</param>
    /// <returns>A new array that contains the elements inside the slice range.</returns>
    // The Slice method must have this signature to get implicit Range support.
    public readonly Consuming Slice(int start, int length)
    {
        if (start < 0 || start > Count)
            throw new ArgumentOutOfRangeException(nameof(start));

        if (length < 0 || length > Count)
            throw new ArgumentOutOfRangeException(nameof(start));

        return GetSliceRange(start, start + length, step: 1, deep: false);
    }

    /// <summary>
    /// Returns the slice of the <see cref="UnsafeGodotArray"/>, from <paramref name="start"/>
    /// (inclusive) to <paramref name="end"/> (exclusive), as a new <see cref="UnsafeGodotArray"/>.
    /// The absolute value of <paramref name="start"/> and <paramref name="end"/>
    /// will be clamped to the array size.
    /// If either <paramref name="start"/> or <paramref name="end"/> are negative, they
    /// will be relative to the end of the array (i.e. <c>arr.GetSliceRange(0, -2)</c>
    /// is a shorthand for <c>arr.GetSliceRange(0, arr.Count - 2)</c>).
    /// If specified, <paramref name="step"/> is the relative index between source
    /// elements. It can be negative, then <paramref name="start"/> must be higher than
    /// <paramref name="end"/>. For example, <c>[0, 1, 2, 3, 4, 5].GetSliceRange(5, 1, -2)</c>
    /// returns <c>[5, 3]</c>.
    /// If <paramref name="deep"/> is true, each element will be copied by value
    /// rather than by reference.
    /// </summary>
    /// <param name="start">The zero-based index at which the range starts.</param>
    /// <param name="end">The zero-based index at which the range ends.</param>
    /// <param name="step">The relative index between source elements to take.</param>
    /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
    /// <returns>A new array that contains the elements inside the slice range.</returns>
    public readonly Consuming GetSliceRange(int start, int end, int step = 1, bool deep = false)
    {
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_slice(ref self, start, end, step, deep.ToGodotBool(), out godot_array newArray);
        return CreateConsuming(newArray);
    }

    /// <summary>
    /// Sorts the array.
    /// Note: The sorting algorithm used is not stable. This means that values
    /// considered equal may have their order changed when using <see cref="Sort"/>.
    /// Note: Strings are sorted in alphabetical order (as opposed to natural order).
    /// This may lead to unexpected behavior when sorting an array of strings ending
    /// with a sequence of numbers.
    /// To sort with a custom predicate use
    /// <see cref="System.Linq.Enumerable.OrderBy{TSource, TKey}(IEnumerable{TSource}, Func{TSource, TKey})"/>.
    /// </summary>
    /// <example>
    /// <code>
    /// var strings = new Godot.Collections.Array { "string1", "string2", "string10", "string11" };
    /// strings.Sort();
    /// GD.Print(strings); // Prints [string1, string10, string11, string2]
    /// </code>
    /// </example>
    /// <exception cref="InvalidOperationException">
    /// The array is read-only.
    /// </exception>
    public readonly void Sort()
    {
        ThrowIfReadOnly();

        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_sort(ref self);
    }

    /// <summary>
    /// Finds the index of an existing value using binary search.
    /// If the value is not present in the array, it returns the bitwise
    /// complement of the insertion index that maintains sorting order.
    /// Note: Calling <see cref="BinarySearch(int, int, Variant)"/> on an
    /// unsorted array results in unexpected behavior.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="index"/> is less than 0.
    /// -or-
    /// <paramref name="count"/> is less than 0.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// <paramref name="index"/> and <paramref name="count"/> do not denote
    /// a valid range in the <see cref="UnsafeGodotArray"/>.
    /// </exception>
    /// <param name="index">The starting index of the range to search.</param>
    /// <param name="count">The length of the range to search.</param>
    /// <param name="item">The object to locate.</param>
    /// <returns>
    /// The index of the item in the array, if <paramref name="item"/> is found;
    /// otherwise, a negative number that is the bitwise complement of the index
    /// of the next element that is larger than <paramref name="item"/> or, if
    /// there is no larger element, the bitwise complement of <see cref="Count"/>.
    /// </returns>
    public readonly int BinarySearch(int index, int count, Variant item)
    {
        if (index < 0)
            throw new ArgumentOutOfRangeException(nameof(index), "index cannot be negative.");
        if (count < 0)
            throw new ArgumentOutOfRangeException(nameof(count), "count cannot be negative.");
        if (Count - index < count)
            throw new ArgumentException("length is out of bounds or count is greater than the number of elements.");

        if (Count == 0)
        {
            // Special case for empty array to avoid an interop call.
            return -1;
        }

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_binary_search(ref self, index, count, variantValue);
    }

    /// <summary>
    /// Finds the index of an existing value using binary search.
    /// If the value is not present in the array, it returns the bitwise
    /// complement of the insertion index that maintains sorting order.
    /// Note: Calling <see cref="BinarySearch(Variant)"/> on an unsorted
    /// array results in unexpected behavior.
    /// </summary>
    /// <param name="item">The object to locate.</param>
    /// <returns>
    /// The index of the item in the array, if <paramref name="item"/> is found;
    /// otherwise, a negative number that is the bitwise complement of the index
    /// of the next element that is larger than <paramref name="item"/> or, if
    /// there is no larger element, the bitwise complement of <see cref="Count"/>.
    /// </returns>
    public readonly int BinarySearch(Variant item) => BinarySearch(0, Count, item);

    /// <summary>
    /// Makes the <see cref="UnsafeGodotArray"/> read-only, i.e. disabled modifying of the
    /// array's elements. Does not apply to nested content, e.g. content of
    /// nested arrays.
    /// </summary>
    public readonly void MakeReadOnly()
    {
        if (IsReadOnly)
        {
            // Avoid interop call when the array is already read-only.
            return;
        }

        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_make_read_only(ref self);
    }

    public readonly void Clear() => Resize(0);

    public readonly bool Contains(Variant item) => IndexOf(item) != -1;

    public readonly void CopyTo(Variant[] array, int arrayIndex)
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

    public readonly bool Remove(Variant item)
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

    public readonly void CopyTo(Array array, int index)
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

    public readonly bool IsSynchronized => false;
    public readonly object SyncRoot => false;

    public readonly bool IsReadOnly => NativeValue.DangerousSelfRef.IsReadOnly;

    public readonly int IndexOf(Variant item)
    {
        if (Count == 0)
        {
            // Special case for empty array to avoid an interop call.
            return -1;
        }

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_index_of(ref self, variantValue);
    }

    /// <summary>
    /// Searches the array for a value and returns its index or <c>-1</c> if not found.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="index"/> is less than 0 or greater than the array's size.
    /// </exception>
    /// <param name="item">The <see cref="Variant"/> item to search for.</param>
    /// <param name="index">The initial search index to start from.</param>
    /// <returns>The index of the item, or -1 if not found.</returns>
    public readonly int IndexOf(Variant item, int index)
    {
        if (index < 0 || index > Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (Count == 0)
        {
            // Special case for empty array to avoid an interop call.
            return -1;
        }

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_index_of(ref self, variantValue, index);
    }

    /// <summary>
    /// Searches the array for a value in reverse order and returns its index
    /// or <c>-1</c> if not found.
    /// </summary>
    /// <param name="item">The <see cref="Variant"/> item to search for.</param>
    /// <returns>The index of the item, or -1 if not found.</returns>
    public readonly int LastIndexOf(Variant item)
    {
        if (Count == 0)
        {
            // Special case for empty array to avoid an interop call.
            return -1;
        }

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_last_index_of(ref self, variantValue, Count - 1);
    }

    /// <summary>
    /// Searches the array for a value in reverse order and returns its index
    /// or <c>-1</c> if not found.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="index"/> is less than 0 or greater than the array's size.
    /// </exception>
    /// <param name="item">The <see cref="Variant"/> item to search for.</param>
    /// <param name="index">The initial search index to start from.</param>
    /// <returns>The index of the item, or -1 if not found.</returns>
    public readonly int LastIndexOf(Variant item, int index)
    {
        if (index < 0 || index >= Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (Count == 0)
        {
            // Special case for empty array to avoid an interop call.
            return -1;
        }

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        return NativeFuncs.godotsharp_array_last_index_of(ref self, variantValue, index);
    }

    public readonly void Insert(int index, Variant item)
    {
        ThrowIfReadOnly();

        if (index < 0 || index > Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        godot_variant variantValue = (godot_variant)item.NativeVar;
        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_insert(ref self, index, variantValue);
    }

    public readonly void RemoveAt(int index)
    {
        ThrowIfReadOnly();

        if (index < 0 || index > Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        var self = (godot_array)NativeValue;
        NativeFuncs.godotsharp_array_remove_at(ref self, index);
    }

    /// <summary>
    /// The variant returned via the <paramref name="elem"/> parameter is owned by the Array and must not be disposed.
    /// </summary>
    internal readonly unsafe void GetVariantBorrowElementAtUnchecked(int index, out godot_variant elem)
        => elem = NativeValue.DangerousSelfRef.Elements[index];

    /// <summary>
    /// The variant returned via the <paramref name="elem"/> parameter is owned by the Array and must not be disposed.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="index"/> is less than 0 or greater than the array's size.
    /// </exception>
    internal readonly void GetVariantBorrowElementAt(int index, out godot_variant elem)
    {
        if (index < 0 || index >= Count)
            throw new ArgumentOutOfRangeException(nameof(index));
        GetVariantBorrowElementAtUnchecked(index, out elem);
    }

    public readonly unsafe Variant this[int index]
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

    private readonly void ThrowIfReadOnly()
    {
        if (IsReadOnly)
            throw new InvalidOperationException("Array instance is read-only.");
    }
}
