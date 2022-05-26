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
    /// 围绕 Godot 的 Array 类的包装，一个 Variant 数组
    /// 在 C++ 引擎中分配的类型化元素。 有用的时候
    /// 与引擎的接口。 否则更喜欢 .NET 集合
    /// 例如 <see cref="System.Array"/> 或 <see cref="List{T}"/>.
    /// </summary>
    public class Array : IList, IDisposable
    {
        private ArraySafeHandle _safeHandle;
        private bool _disposed = false;

        /// <summary>
        /// 构造一个新的空数组 <see cref="Array"/>.
        /// </summary>
        public Array()
        {
            _safeHandle = new ArraySafeHandle(godot_icall_Array_Ctor());
        }

        /// <summary>
        /// 从给定集合的元素构造一个新的 <see cref="Array"/>。
        /// </summary>
        /// <param name="collection">要构造的元素的集合.</param>
        /// <returns>新的 godot 数组.</returns>
        public Array(IEnumerable collection) : this()
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            foreach (object element in collection)
                Add(element);
        }

        /// <summary>
        /// 从给定的对象构造一个新的 <see cref="Array"/>.
        /// </summary>
        /// <param name="array">放入新数组的对象.</param>
        /// <returns>新的 godot 数组.</returns>
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
        ///复制此 <see cref="Array"/>。
        /// </summary>
        /// <param name="deep">如果<see langword="true"/>，执行深拷贝</param>
        /// <returns>A new Godot Array.</returns>
        public Array Duplicate(bool deep = false)
        {
            return new Array(godot_icall_Array_Duplicate(GetPtr(), deep));
        }

        /// <summary>
        /// 将此 <see cref="Array"/> 调整为给定大小
        /// </summary>
        /// <param name="newSize">数组的新大小.</param>
        /// <returns><see cref="Error.Ok"/> 返回成功或错误代码.</returns>
        public Error Resize(int newSize)
        {
            return godot_icall_Array_Resize(GetPtr(), newSize);
        }

        /// <summary>
        /// 将此 <see cref="Array"/> 的内容随机排列.
        /// </summary>
        public void Shuffle()
        {
            godot_icall_Array_Shuffle(GetPtr());
        }

        /// <summary>
        /// 连接这两个 <see cref="Array"/>.
        /// </summary>
        /// <param name="left">第一个数组.</param>
        /// <param name="right">第二个数组.</param>
        /// <returns>包含两个数组内容的新 Godot 数组.</returns>
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
        /// 返回给定 <paramref name="index"/> 处的对象.
        /// </summary>
        /// <value>给定 <paramref name="index"/> 处的对象.</value>
        public object this[int index]
        {
            get => godot_icall_Array_At(GetPtr(), index);
            set => godot_icall_Array_SetAt(GetPtr(), index, value);
        }

        /// <summary>
        /// 将一个对象添加到此 <see cref="Array"/> 的末尾。
        ///这与 GDScript 中的<c> append</c> 或<c> push_back</c> 相同。
        /// </summary>
        /// <param name="value">要添加的对象.</param>
        /// <returns>添加对象后的新尺寸.</returns>
        public int Add(object value) => godot_icall_Array_Add(GetPtr(), value);

        /// <summary>
        /// 检查此 <see cref="Array"/> 是否包含给定对象.
        /// </summary>
        /// <param name="value">要查找的项目.</param>
        /// <returns>此数组是否包含给定对象.</returns>
        public bool Contains(object value) => godot_icall_Array_Contains(GetPtr(), value);

        /// <summary>
        /// 从中删除所有项目 <see cref="Array"/>.
        /// </summary>
        public void Clear() => godot_icall_Array_Clear(GetPtr());

        /// <summary>
        /// 在这个<see cref="Array"/> 中搜索一个对象
        ///并返回它的索引，如果没有找到则返回 -1。
        /// </summary>
        /// <param name="value">要搜索的对象.</param>
        /// <returns>对象的索引，如果未找到则为 -1.</returns>
        public int IndexOf(object value) => godot_icall_Array_IndexOf(GetPtr(), value);

        /// <summary>
        /// 在数组中的给定位置插入一个新对象。
        ///该位置必须是现有项目的有效位置，
        ///或数组末尾的位置。
        ///现有项目将向右移动。
        /// </summary>
        /// <param name="index">要插入的索引.</param>
        /// <param name="value">要插入的对象.</param>
        public void Insert(int index, object value) => godot_icall_Array_Insert(GetPtr(), index, value);

        /// <summary>
        ///删除指定<paramref name="value"/> 的第一次出现
        ///从这个<see cref="Array"/>.
        /// </summary>
        /// <param name="value">要删除的值.</param>
        public void Remove(object value) => godot_icall_Array_Remove(GetPtr(), value);

        /// <summary>
        /// 按索引从此 <see cref="Array"/> 中删除一个元素。
        /// </summary>
        /// <param name="index">要删除的元素的索引.</param>
        public void RemoveAt(int index) => godot_icall_Array_RemoveAt(GetPtr(), index);

        // ICollection

        /// <summary>
        ///返回此<see cref="Array"/> 中的元素数。
        ///这也称为数组的大小或长度。
        /// </summary>
        /// <returns>元素数量.</returns>
        public int Count => godot_icall_Array_Count(GetPtr());

        object ICollection.SyncRoot => this;

        bool ICollection.IsSynchronized => false;

        /// <summary>
        ///将此<see cref="Array"/> 的元素复制到给定的
        ///无类型的 C# 数组，从给定的索引开始。
        /// </summary>
        /// <param name="array">要复制到的数组.</param>
        /// <param name="index">开始的索引.</param>
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
        /// 获取此 <see cref="Array"/> 的枚举器。
        /// </summary>
        /// <returns>枚举器.</returns>
        public IEnumerator GetEnumerator()
        {
            int count = Count;

            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        /// <summary>
        /// 将此 <see cref="Array"/> 转换为字符串。
        /// </summary>
        /// <returns>此数组的字符串表示形式。</returns>
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
        internal static extern Error godot_icall_Array_Shuffle(IntPtr ptr);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern void godot_icall_Array_Generic_GetElementTypeInfo(Type elemType, out int elemTypeEncoding, out IntPtr elemTypeClass);

        [MethodImpl(MethodImplOptions.InternalCall)]
        internal static extern string godot_icall_Array_ToString(IntPtr ptr);
    }

    /// <summary>
    /// 围绕 Godot 的 Array 类的类型化包装器，一个 Variant 数组
    /// 在 C++ 引擎中分配的类型化元素。 有用的时候
    /// 与引擎交互。 否则更喜欢 .NET 集合
    /// 例如数组或<see cref="List{T}"/>。
    /// </summary>
    /// <typeparam name="T">数组的类型.</typeparam>
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
        /// 构造一个新的空 <see cref="Array{T}"/>。
        /// </summary>
        public Array()
        {
            _objectArray = new Array();
        }

        /// <summary>
        /// 从给定集合的元素构造一个新的 <see cref="Array{T}"/>。
        /// </summary>
        /// <param name="collection">要构造的元素的集合.</param>
        /// <returns>一个新的 godot 数组.</returns>
        public Array(IEnumerable<T> collection)
        {
            if (collection == null)
                throw new NullReferenceException($"Parameter '{nameof(collection)} cannot be null.'");

            _objectArray = new Array(collection);
        }

        /// <summary>
        /// 从给定的项目构造一个新的 <see cref="Array{T}"/>。
        /// </summary>
        /// <param name="array">要放入新数组的项目.</param>
        /// <returns>一个新的 godot 数组.</returns>
        public Array(params T[] array) : this()
        {
            if (array == null)
            {
                throw new NullReferenceException($"Parameter '{nameof(array)} cannot be null.'");
            }
            _objectArray = new Array(array);
        }

        /// <summary>
        ///从无类型的 <see cref="Array"/> 构造一个有类型的 <see cref="Array{T}"/>。
        /// </summary>
        /// <param name="array">要构造的无类型数组.</param>
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
        /// 将此类型化的 <see cref="Array{T}"/> 转换为非类型化的 <see cref="Array"/>。
        /// </summary>
        /// <param name="from">要转换的类型化数组.</param>
        public static explicit operator Array(Array<T> from)
        {
            return from._objectArray;
        }

        /// <summary>
        /// 复制此 <see cref="Array{T}"/>。
        /// </summary>
        /// <param name="deep">如果<see langword="true"/>，执行深拷贝.</param>
        /// <returns>一个新的 godot 数组.</returns>
        public Array<T> Duplicate(bool deep = false)
        {
            return new Array<T>(_objectArray.Duplicate(deep));
        }

        /// <summary>
        /// 将此 <see cref="Array{T}"/> 调整为给定大小。
        /// </summary>
        /// <param name="newSize">数组的新大小.</param>
        /// <returns><see cref="Error.Ok"/> 成功或错误代码.</returns>
        public Error Resize(int newSize)
        {
            return _objectArray.Resize(newSize);
        }

        /// <summary>
        /// 将此 <see cref="Array{T}"/> 的内容随机排列。
        /// </summary>
        public void Shuffle()
        {
            _objectArray.Shuffle();
        }

        /// <summary>
        /// 连接这两个 <see cref="Array{T}"/>s。
        /// </summary>
        /// <param name="left">第一个数组.</param>
        /// <param name="right">第二个数组.</param>
        /// <returns>包含两个数组内容的新 Godot 数组.</returns>
        public static Array<T> operator +(Array<T> left, Array<T> right)
        {
            return new Array<T>(left._objectArray + right._objectArray);
        }

        // IList<T>

        /// <summary>
        /// 返回给定 <paramref name="index"/> 的值。
        /// </summary>
        /// <value>给定 <paramref name="index"/> 的值.</value>
        public T this[int index]
        {
            get { return (T)Array.godot_icall_Array_At_Generic(GetPtr(), index, elemTypeEncoding, elemTypeClass); }
            set { _objectArray[index] = value; }
        }

        /// <summary>
        ///在这个<see cref="Array{T}" /> 中搜索一个项目
        ///并返回它的索引，如果没有找到则返回 -1。
        /// </summary>
        /// <param name="item">要搜索的项目.</param>
        /// <returns>项目的索引，如果未找到，则为 -1.</returns>
        public int IndexOf(T item)
        {
            return _objectArray.IndexOf(item);
        }

        /// <summary>
        ///在<see cref="Array{T}"/> 的给定位置插入一个新项目。
        ///该位置必须是现有项目的有效位置，
        ///或数组末尾的位置。
        ///现有项目将向右移动。
        /// </summary>
        /// <param name="index">要插入的索引.</param>
        /// <param name="item">要插入的项目.</param>
        public void Insert(int index, T item)
        {
            _objectArray.Insert(index, item);
        }

        /// <summary>
        /// 按索引从此 <see cref="Array{T}"/> 中删除一个元素。
        /// </summary>
        /// <param name="index">要删除的元素的索引.</param>
        public void RemoveAt(int index)
        {
            _objectArray.RemoveAt(index);
        }

        // ICollection<T>

        /// <summary>
        /// 返回此 <see cref="Array{T}"/> 中的元素数。
        /// 这也称为数组的大小或长度。
        /// </summary>
        /// <returns>元素数量.</returns>
        public int Count
        {
            get { return _objectArray.Count; }
        }

        bool ICollection<T>.IsReadOnly => false;

        /// <summary>
        ///在<see cref="Array{T}"/> 的末尾添加一个项目。
        ///这与 GDScript 中的<c> append</c> 或<c> push_back</c> 相同。
        /// </summary>
        /// <param name="item">要添加的项目.</param>
        /// <returns>添加项目后的新尺寸.</returns>
        public void Add(T item)
        {
            _objectArray.Add(item);
        }

        /// <summary>
        /// 删除此 <see cref="Array{T}"/> 中的所有项目。
        /// </summary>
        public void Clear()
        {
            _objectArray.Clear();
        }

        /// <summary>
        /// 检查此 <see cref="Array{T}"/> 是否包含给定项目。
        /// </summary>
        /// <param name="item">要查找的项目.</param>
        /// <returns>此数组是否包含给定项目.</returns>
        public bool Contains(T item)
        {
            return _objectArray.Contains(item);
        }

        /// <summary>
        /// 将此<see cref="Array{T}"/> 的元素复制到给定的
        /// C# 数组，从给定的索引开始。
        /// </summary>
        /// <param name="array">要复制到的 C# 数组.</param>
        /// <param name="arrayIndex">开始的索引.</param>
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
        ///删除指定值的第一次出现
        ///从这个<see cref= "Array{T}" />.
        /// </summary>
        /// <param name="item">要删除的值.</param>
        /// <returns>A <see langword="bool"/> 表示成功或失败.</returns>
        public bool Remove(T item)
        {
            return Array.godot_icall_Array_Remove(GetPtr(), item);
        }

        // IEnumerable<T>

        /// <summary>
        /// 获取此 <see cref="Array{T}"/> 的枚举器。
        /// </summary>
        /// <returns>枚举器.</returns>
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
        ///将此 <see cref="Array{T}"/> 转换为字符串。
        /// </summary>
        /// <returns>此数组的字符串表示形式.</returns>
        public override string ToString() => _objectArray.ToString();
    }
}
