using System;
using System.Collections;
using System.Collections.Generic;

namespace Godot.NativeInterop.UnsafeCollections;

internal struct UnsafeGodotDictionary :
    IDictionary<Variant, Variant>,
    IReadOnlyDictionary<Variant, Variant>,
    IDisposable
{
    internal godot_dictionary.movable NativeValue;

    public struct Borrowing(UnsafeGodotDictionary target)
    {
        public UnsafeGodotDictionary Target = target;

        public static implicit operator Variant(Borrowing unsafeDict) =>
            Variant.CreateConsuming(VariantUtils.CreateFromDictionary((godot_dictionary)unsafeDict.Target.NativeValue));
    }

    public struct Consuming(UnsafeGodotDictionary target) : IDisposable
    {
        public UnsafeGodotDictionary Target = target;

        public void Dispose() => Target.Dispose();
    }

    public static UnsafeGodotDictionary Create() =>
        new() { NativeValue = (godot_dictionary.movable)NativeFuncs.godotsharp_dictionary_new() };

    public static Consuming CreateConsuming(godot_dictionary consumingNativeValue) =>
        new(new UnsafeGodotDictionary(consumingNativeValue, consuming: true));

    public static Borrowing CreateBorrowing(godot_dictionary borrowingNativeValue) =>
        new(new UnsafeGodotDictionary(borrowingNativeValue, consuming: false));

    private UnsafeGodotDictionary(godot_dictionary nativeValue, bool consuming)
    {
        if (consuming)
        {
            NativeValue = (godot_dictionary.movable)(nativeValue.IsAllocated
                ? nativeValue
                : NativeFuncs.godotsharp_dictionary_new());
        }
        else
        {
            if (!nativeValue.IsAllocated)
                throw new ArgumentException($"The {nameof(nativeValue)} native dictionary is not allocated.");
            NativeValue = (godot_dictionary.movable)nativeValue;
        }
    }

    public readonly Consuming CloneDisposable() =>
        CreateConsuming(NativeFuncs.godotsharp_dictionary_new_copy((godot_dictionary)NativeValue));

    public readonly Borrowing BorrowDisposable() => new(this);

    public void Dispose() => NativeValue.DangerousSelfRef.Dispose();

    public readonly int Count
    {
        get
        {
            var self = (godot_dictionary)NativeValue;
            return NativeFuncs.godotsharp_dictionary_count(ref self);
        }
    }

    readonly int ICollection<KeyValuePair<Variant, Variant>>.Count => Count;

    readonly int IReadOnlyCollection<KeyValuePair<Variant, Variant>>.Count => Count;

    public struct Enumerator : IEnumerator<KeyValuePair<Variant, Variant>>
    {
        private UnsafeGodotDictionary _consumingUnsafeDict;
        private int _index;
        private readonly int _count;

        internal Enumerator(Consuming consumingUnsafeDict, int count)
        {
            _consumingUnsafeDict = consumingUnsafeDict.Target;
            _index = -1;
            _count = count;
        }

        public bool MoveNext()
        {
            _index++;
            return _index < _count;
        }

        public void Reset() => _index = -1;

        KeyValuePair<Variant, Variant> IEnumerator<KeyValuePair<Variant, Variant>>.Current
            => _consumingUnsafeDict.GetKeyValuePair(Current);

        object IEnumerator.Current => Current;

        public int Current => _index;

        public void Dispose() => _consumingUnsafeDict.Dispose();
    }

    public readonly Enumerator GetEnumerator() => new(CloneDisposable(), Count);

    readonly IEnumerator<KeyValuePair<Variant, Variant>> IEnumerable<KeyValuePair<Variant, Variant>>.GetEnumerator()
        => GetEnumerator();

    readonly IEnumerator IEnumerable.GetEnumerator()
    {
        throw new InvalidOperationException(
            $"IEnumerable.GetEnumerator() cannot be implemented for this '{typeof(UnsafeGodotDictionary)}' "
            + "because IEnumerable doesn't implement IDisposable and would leak the native resources.");
    }

    internal readonly KeyValuePair<Variant, Variant> GetKeyValuePair(int index)
    {
        var self = (godot_dictionary)NativeValue;
        NativeFuncs.godotsharp_dictionary_key_value_pair_at(ref self, index,
            out godot_variant key,
            out godot_variant value);
        return new KeyValuePair<Variant, Variant>(Variant.CreateConsuming(key),
            Variant.CreateConsuming(value));
    }

    /// <summary>
    /// Adds a value <paramref name="value"/> at key <paramref name="key"/>
    /// to this <see cref="UnsafeGodotDictionary"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The dictionary is read-only.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// An entry for <paramref name="key"/> already exists in the dictionary.
    /// </exception>
    /// <param name="key">The key at which to add the value.</param>
    /// <param name="value">The value to add.</param>
    public readonly void Add(Variant key, Variant value)
    {
        ThrowIfReadOnly();

        var variantKey = (godot_variant)key.NativeVar;
        var self = (godot_dictionary)NativeValue;

        if (NativeFuncs.godotsharp_dictionary_contains_key(ref self, variantKey).ToBool())
            throw new ArgumentException("An element with the same key already exists.", nameof(key));

        godot_variant variantValue = (godot_variant)value.NativeVar;
        NativeFuncs.godotsharp_dictionary_add(ref self, variantKey, variantValue);
    }

    readonly void ICollection<KeyValuePair<Variant, Variant>>.Add(KeyValuePair<Variant, Variant> item)
        => Add(item.Key, item.Value);

    /// <summary>
    /// Clears the dictionary, removing all entries from it.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The dictionary is read-only.
    /// </exception>
    public readonly void Clear()
    {
        ThrowIfReadOnly();

        var self = (godot_dictionary)NativeValue;
        NativeFuncs.godotsharp_dictionary_clear(ref self);
    }

    public readonly bool ContainsKey(Variant key)
    {
        var self = (godot_dictionary)NativeValue;
        return NativeFuncs.godotsharp_dictionary_contains_key(ref self, (godot_variant)key.NativeVar).ToBool();
    }

    public readonly bool Contains(KeyValuePair<Variant, Variant> item)
    {
        godot_variant variantKey = (godot_variant)item.Key.NativeVar;
        var self = (godot_dictionary)NativeValue;
        bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
            variantKey, out godot_variant retValue).ToBool();

        using (retValue)
        {
            if (!found)
                return false;

            godot_variant variantValue = (godot_variant)item.Value.NativeVar;
            return NativeFuncs.godotsharp_variant_equals(variantValue, retValue).ToBool();
        }
    }

    public readonly void CopyTo(KeyValuePair<Variant, Variant>[] array, int arrayIndex)
    {
        if (array == null)
            throw new ArgumentNullException(nameof(array), "Value cannot be null.");

        if (arrayIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(arrayIndex),
                "Number was less than the array's lower bound in the first dimension.");

        var (keys, values, count) = GetKeyValuePairs();

        using (keys)
        using (values)
        {
            if (array.Length < (arrayIndex + count))
                throw new ArgumentException(
                    "Destination array was not long enough. Check destIndex and length, and the array's lower bounds.");

            for (int i = 0; i < count; i++)
            {
                array[arrayIndex] = new(keys.Target[i], values.Target[i]);
                arrayIndex++;
            }
        }
    }

    private readonly (UnsafeGodotArray.Consuming keys, UnsafeGodotArray.Consuming values, int count) GetKeyValuePairs()
    {
        var self = (godot_dictionary)NativeValue;

        NativeFuncs.godotsharp_dictionary_keys(ref self, out godot_array keysArray);
        var keys = UnsafeGodotArray.CreateConsuming(keysArray);

        NativeFuncs.godotsharp_dictionary_values(ref self, out godot_array valuesArray);
        var values = UnsafeGodotArray.CreateConsuming(valuesArray);

        int count = NativeFuncs.godotsharp_dictionary_count(ref self);

        return (keys, values, count);
    }

    public readonly bool TryGetValue(Variant key, out Variant value)
    {
        var self = (godot_dictionary)NativeValue;
        bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
            (godot_variant)key.NativeVar, out godot_variant retValue).ToBool();

        value = found ? Variant.CreateConsuming(retValue) : default;

        return found;
    }

    /// <summary>
    /// Removes an element from this <see cref="UnsafeGodotDictionary"/> by key.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The dictionary is read-only.
    /// </exception>
    /// <param name="key">The key of the element to remove.</param>
    public readonly bool Remove(Variant key)
    {
        ThrowIfReadOnly();

        var self = (godot_dictionary)NativeValue;
        return NativeFuncs.godotsharp_dictionary_remove_key(ref self, (godot_variant)key.NativeVar).ToBool();
    }

    public readonly bool Remove(KeyValuePair<Variant, Variant> item)
    {
        ThrowIfReadOnly();

        godot_variant variantKey = (godot_variant)item.Key.NativeVar;
        var self = (godot_dictionary)NativeValue;
        bool found = NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
            variantKey, out godot_variant retValue).ToBool();

        using (retValue)
        {
            if (!found)
                return false;

            godot_variant variantValue = (godot_variant)item.Value.NativeVar;
            if (NativeFuncs.godotsharp_variant_equals(variantValue, retValue).ToBool())
            {
                return NativeFuncs.godotsharp_dictionary_remove_key(
                    ref self, variantKey).ToBool();
            }

            return false;
        }
    }

    /// <summary>
    /// Returns a copy of the <see cref="UnsafeGodotDictionary"/>.
    /// If <paramref name="deep"/> is <see langword="true"/>, a deep copy is performed:
    /// all nested arrays and dictionaries are duplicated and will not be shared with
    /// the original dictionary. If <see langword="false"/>, a shallow copy is made and
    /// references to the original nested arrays and dictionaries are kept, so that
    /// modifying a sub-array or dictionary in the copy will also impact those
    /// referenced in the source dictionary. Note that any <see cref="GodotObject"/> derived
    /// elements will be shallow copied regardless of the <paramref name="deep"/>
    /// setting.
    /// </summary>
    /// <param name="deep">If <see langword="true"/>, performs a deep copy.</param>
    /// <returns>A new Godot Dictionary.</returns>
    public readonly Consuming Duplicate(bool deep = false)
    {
        var self = (godot_dictionary)NativeValue;
        NativeFuncs.godotsharp_dictionary_duplicate(ref self, deep.ToGodotBool(),
            out godot_dictionary newDictionary);
        return CreateConsuming(newDictionary);
    }

    /// <summary>
    /// Adds entries from <paramref name="dictionary"/> to this dictionary.
    /// By default, duplicate keys are not copied over, unless <paramref name="overwrite"/>
    /// is <see langword="true"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The dictionary is read-only.
    /// </exception>
    /// <param name="dictionary">Dictionary to copy entries from.</param>
    /// <param name="overwrite">If duplicate keys should be copied over as well.</param>
    public readonly void Merge(Borrowing dictionary, bool overwrite = false)
    {
        ThrowIfReadOnly();

        var self = (godot_dictionary)NativeValue;
        var other = (godot_dictionary)dictionary.Target.NativeValue;
        NativeFuncs.godotsharp_dictionary_merge(ref self, in other, overwrite.ToGodotBool());
    }

    /// <summary>
    /// Compares this <see cref="UnsafeGodotDictionary"/> against the <paramref name="other"/>
    /// <see cref="UnsafeGodotDictionary"/> recursively. Returns <see langword="true"/> if the
    /// two dictionaries contain the same keys and values. The order of the entries
    /// does not matter.
    /// otherwise.
    /// </summary>
    /// <param name="other">The other dictionary to compare against.</param>
    /// <returns>
    /// <see langword="true"/> if the dictionaries contain the same keys and values,
    /// <see langword="false"/> otherwise.
    /// </returns>
    public readonly bool RecursiveEqual(Borrowing other)
    {
        var self = (godot_dictionary)NativeValue;
        var otherVariant = (godot_dictionary)other.Target.NativeValue;
        return NativeFuncs.godotsharp_dictionary_recursive_equal(ref self, otherVariant).ToBool();
    }

    /// <summary>
    /// Returns the value at the given <paramref name="key"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The property is assigned and the dictionary is read-only.
    /// </exception>
    /// <exception cref="KeyNotFoundException">
    /// The property is retrieved and an entry for <paramref name="key"/>
    /// does not exist in the dictionary.
    /// </exception>
    /// <value>The value at the given <paramref name="key"/>.</value>
    public readonly Variant this[Variant key]
    {
        get
        {
            var self = (godot_dictionary)NativeValue;

            if (NativeFuncs.godotsharp_dictionary_try_get_value(ref self,
                    (godot_variant)key.NativeVar, out godot_variant value).ToBool())
            {
                return Variant.CreateConsuming(value);
            }

            throw new KeyNotFoundException();
        }
        set
        {
            ThrowIfReadOnly();

            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_set_value(ref self,
                (godot_variant)key.NativeVar, (godot_variant)value.NativeVar);
        }
    }

    /// <summary>
    /// Gets the collection of keys in this <see cref="UnsafeGodotDictionary"/>.
    /// </summary>
    private readonly UnsafeGodotArray.Consuming Keys
    {
        get
        {
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_keys(ref self, out godot_array keysArray);
            return UnsafeGodotArray.CreateConsuming(keysArray);
        }
    }

    /// <summary>
    /// Gets the collection of elements in this <see cref="UnsafeGodotDictionary"/>.
    /// </summary>
    private readonly UnsafeGodotArray.Consuming Values
    {
        get
        {
            var self = (godot_dictionary)NativeValue;
            NativeFuncs.godotsharp_dictionary_keys(ref self, out godot_array keysArray);
            return UnsafeGodotArray.CreateConsuming(keysArray);
        }
    }

    readonly IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Keys => Keys;

    readonly IEnumerable<Variant> IReadOnlyDictionary<Variant, Variant>.Values => Values;

    readonly ICollection<Variant> IDictionary<Variant, Variant>.Keys => Keys.Target;

    readonly ICollection<Variant> IDictionary<Variant, Variant>.Values => Values.Target;

    public readonly bool IsReadOnly => NativeValue.DangerousSelfRef.IsReadOnly;

    /// <summary>
    /// Makes the <see cref="UnsafeGodotDictionary"/> read-only, i.e. disabled modifying of the
    /// dictionary's elements. Does not apply to nested content, e.g. content of
    /// nested dictionaries.
    /// </summary>
    public readonly void MakeReadOnly()
    {
        if (IsReadOnly)
        {
            // Avoid interop call when the dictionary is already read-only.
            return;
        }

        var self = (godot_dictionary)NativeValue;
        NativeFuncs.godotsharp_dictionary_make_read_only(ref self);
    }

    private readonly void ThrowIfReadOnly()
    {
        if (IsReadOnly)
            throw new InvalidOperationException("Dictionary instance is read-only.");
    }
}
