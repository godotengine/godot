using System;
using System.Collections.Generic;
using System.Collections;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using Godot.NativeInterop;

namespace Godot.Collections
{
    public sealed class Array : IList, IDisposable
    {
        internal godot_array NativeValue;

        public Array()
        {
            godot_icall_Array_Ctor(out NativeValue);
        }

        public Array(IEnumerable collection) : this()
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            foreach (object element in collection)
                Add(element);
        }

        // TODO: This must be removed. Lots of silent mistakes as it takes pretty much anything.
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

        public Array Duplicate(bool deep = false)
        {
            godot_array newArray;
            godot_icall_Array_Duplicate(ref NativeValue, deep, out newArray);
            return CreateTakingOwnershipOfDisposableValue(newArray);
        }

        public Error Resize(int newSize)
        {
            return godot_icall_Array_Resize(ref NativeValue, newSize);
        }

        public void Shuffle()
        {
            godot_icall_Array_Shuffle(ref NativeValue);
        }

        public static Array operator +(Array left, Array right)
        {
            godot_array newArray;
            godot_icall_Array_Concatenate(ref left.NativeValue, ref right.NativeValue, out newArray);
            return CreateTakingOwnershipOfDisposableValue(newArray);
        }

        // IList

        public bool IsReadOnly => false;

        public bool IsFixedSize => false;

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

        public int Add(object value) => godot_icall_Array_Add(ref NativeValue, value);

        public bool Contains(object value) => godot_icall_Array_Contains(ref NativeValue, value);

        public void Clear() => godot_icall_Array_Clear(ref NativeValue);

        public int IndexOf(object value) => godot_icall_Array_IndexOf(ref NativeValue, value);

        public void Insert(int index, object value) => godot_icall_Array_Insert(ref NativeValue, index, value);

        public void Remove(object value) => godot_icall_Array_Remove(ref NativeValue, value);

        public void RemoveAt(int index) => godot_icall_Array_RemoveAt(ref NativeValue, index);

        // ICollection

        public int Count => godot_icall_Array_Count(ref NativeValue);

        public object SyncRoot => this;

        public bool IsSynchronized => false;

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

        public IEnumerator GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

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

        public Array()
        {
            _underlyingArray = new Array();
        }

        public Array(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            _underlyingArray = new Array(collection);
        }

        public Array(params T[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }

            _underlyingArray = new Array(array);
        }

        public Array(Array array)
        {
            _underlyingArray = array;
        }

        // Explicit name to make it very clear
        internal static Array<T> CreateTakingOwnershipOfDisposableValue(godot_array nativeValueToOwn)
            => new Array<T>(Array.CreateTakingOwnershipOfDisposableValue(nativeValueToOwn));

        public static explicit operator Array(Array<T> from)
        {
            return from._underlyingArray;
        }

        public Array<T> Duplicate(bool deep = false)
        {
            return new Array<T>(_underlyingArray.Duplicate(deep));
        }

        public Error Resize(int newSize)
        {
            return _underlyingArray.Resize(newSize);
        }

        public void Shuffle()
        {
            _underlyingArray.Shuffle();
        }

        public static Array<T> operator +(Array<T> left, Array<T> right)
        {
            return new Array<T>(left._underlyingArray + right._underlyingArray);
        }

        // IList<T>

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

        public int IndexOf(T item)
        {
            return _underlyingArray.IndexOf(item);
        }

        public void Insert(int index, T item)
        {
            _underlyingArray.Insert(index, item);
        }

        public void RemoveAt(int index)
        {
            _underlyingArray.RemoveAt(index);
        }

        // ICollection<T>

        public int Count => _underlyingArray.Count;

        public bool IsReadOnly => _underlyingArray.IsReadOnly;

        public void Add(T item)
        {
            _underlyingArray.Add(item);
        }

        public void Clear()
        {
            _underlyingArray.Clear();
        }

        public bool Contains(T item)
        {
            return _underlyingArray.Contains(item);
        }

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

        public bool Remove(T item)
        {
            return Array.godot_icall_Array_Remove(ref _underlyingArray.NativeValue, item);
        }

        // IEnumerable<T>

        public IEnumerator<T> GetEnumerator()
        {
            int count = _underlyingArray.Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        public override string ToString() => _underlyingArray.ToString();
    }
}
