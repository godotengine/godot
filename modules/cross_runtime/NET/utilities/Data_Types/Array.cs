using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

#nullable enable

namespace Godot
{
	public sealed class Array : IList<Variant>, IReadOnlyList<Variant>, ICollection, IDisposable
	{
		private readonly List<Variant> _items = new();
		internal IntPtr NativePtr;

		public Array()
		{
		}

		public Array(IEnumerable<Variant> collection)
		{
			if (collection == null) throw new ArgumentNullException(nameof(collection));
			_items.AddRange(collection);
		}

		public Array(Variant[] array)
		{
			if (array == null) throw new ArgumentNullException(nameof(array));
			_items.AddRange(array);
		}

		// ---- new constructor (used by the generated return statements) ----
		public Array(ulong nativePtr)
		{
			NativePtr = (IntPtr)nativePtr;
			// The rest of the class will need to read from this handle when
			// elements are accessed.  For now, just storing the handle is
			// enough to make the generated code compile.
		}

		public Array(scoped ReadOnlySpan<StringName> span)
		{
			for (int i = 0; i < span.Length; i++)
				_items.Add(Variant.CreateFrom(span[i]));
		}

		public Array(scoped Span<StringName> span) : this((ReadOnlySpan<StringName>)span) { }

		public Array(scoped ReadOnlySpan<NodePath> span)
		{
			for (int i = 0; i < span.Length; i++)
				_items.Add(Variant.CreateFrom(span[i]));
		}

		public Array(scoped Span<NodePath> span) : this((ReadOnlySpan<NodePath>)span) { }

		public Array(scoped ReadOnlySpan<RID> span)
		{
			for (int i = 0; i < span.Length; i++)
				_items.Add(Variant.CreateFrom(span[i]));
		}

		public Array(scoped Span<RID> span) : this((ReadOnlySpan<RID>)span) { }

		public Array(scoped ReadOnlySpan<GodotObject> span)
		{
			for (int i = 0; i < span.Length; i++)
				_items.Add(Variant.CreateFrom(span[i]));
		}

		public Array(scoped Span<GodotObject> span) : this((ReadOnlySpan<GodotObject>)span) { }

		public void Dispose()
		{
			_items.Clear();
		}

		public Array Duplicate(bool deep = false)
		{
			if (!deep)
				return new Array(_items.ToArray());

			var copy = new Array();
			foreach (var item in _items)
				copy.Add(DeepCopyVariant(item));
			return copy;
		}

		public void Fill(Variant value)
		{
			for (int i = 0; i < _items.Count; i++)
				_items[i] = value;
		}

		public Variant Max()
		{
			if (_items.Count == 0)
				return Variant.CreateFrom((object?)null);

			Variant best = _items[0];
			for (int i = 1; i < _items.Count; i++)
			{
				if (CompareVariant(_items[i], best) > 0)
					best = _items[i];
			}
			return best;
		}

		public Variant Min()
		{
			if (_items.Count == 0)
				return Variant.CreateFrom((object?)null);

			Variant best = _items[0];
			for (int i = 1; i < _items.Count; i++)
			{
				if (CompareVariant(_items[i], best) < 0)
					best = _items[i];
			}
			return best;
		}

		public Variant PickRandom()
		{
			if (_items.Count == 0)
				return Variant.CreateFrom((object?)null);

			return _items[Random.Shared.Next(_items.Count)];
		}

		public bool RecursiveEqual(Array other)
		{
			if (other == null) return false;
			if (_items.Count != other._items.Count) return false;

			for (int i = 0; i < _items.Count; i++)
			{
				if (!VariantDeepEqual(_items[i], other._items[i]))
					return false;
			}

			return true;
		}

		public Error Resize(int newSize)
		{
			if (newSize < 0)
				return Error.ERR_INVALID_PARAMETER;

			if (newSize < _items.Count)
			{
				_items.RemoveRange(newSize, _items.Count - newSize);
			}
			else if (newSize > _items.Count)
			{
				while (_items.Count < newSize)
					_items.Add(Variant.CreateFrom((object?)null));
			}

			return Error.OK;
		}

		public void Reverse() => _items.Reverse();

		public void Shuffle()
		{
			for (int i = _items.Count - 1; i > 0; i--)
			{
				int j = Random.Shared.Next(i + 1);
				(_items[i], _items[j]) = (_items[j], _items[i]);
			}
		}

		public Array Slice(int start)
		{
			if (start < 0 || start > Count)
				throw new ArgumentOutOfRangeException(nameof(start));

			return GetSliceRange(start, Count, 1, false);
		}

		public Array Slice(int start, int length)
		{
			if (start < 0 || start > Count)
				throw new ArgumentOutOfRangeException(nameof(start));
			if (length < 0 || length > Count)
				throw new ArgumentOutOfRangeException(nameof(length));

			return GetSliceRange(start, start + length, 1, false);
		}

		public Array GetSliceRange(int start, int end, int step = 1, bool deep = false)
		{
			if (step == 0)
				throw new ArgumentOutOfRangeException(nameof(step));

			int count = Count;

			start = NormalizeIndex(start, count);
			end = NormalizeIndex(end, count);

			var result = new Array();

			if (step > 0)
			{
				for (int i = start; i < end && i < count; i += step)
					result.Add(deep ? DeepCopyVariant(_items[i]) : _items[i]);
			}
			else
			{
				for (int i = start; i > end && i >= 0; i += step)
					result.Add(deep ? DeepCopyVariant(_items[i]) : _items[i]);
			}

			return result;
		}

		public void Sort()
		{
			_items.Sort(CompareVariant);
		}

		public static Array operator +(Array left, Array right)
		{
			if (left == null)
			{
				if (right == null) return new Array();
				return right.Duplicate(false);
			}

			if (right == null)
				return left.Duplicate(false);

			var result = left.Duplicate(false);
			result._items.AddRange(right._items);
			return result;
		}

		public Variant this[int index]
		{
			get => _items[index];
			set => _items[index] = value;
		}

		public void Add(Variant item) => _items.Add(item);

		public void AddRange(IEnumerable<Variant> collection)
		{
			if (collection == null) throw new ArgumentNullException(nameof(collection));
			_items.AddRange(collection);
		}

		public void AddRange(Variant[] array)
		{
			if (array == null) throw new ArgumentNullException(nameof(array));
			_items.AddRange(array);
		}

		public void AddRange(ReadOnlySpan<Variant> span)
		{
			for (int i = 0; i < span.Length; i++)
				_items.Add(span[i]);
		}

		public int BinarySearch(int index, int count, Variant item)
		{
			if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
			if (count < 0) throw new ArgumentOutOfRangeException(nameof(count));
			if (index + count > _items.Count) throw new ArgumentException("Invalid range.");

			return _items.GetRange(index, count).BinarySearch(item, Comparer<Variant>.Create(CompareVariant));
		}

		public int BinarySearch(Variant item) => BinarySearch(0, Count, item);

		public bool Contains(Variant item) => IndexOf(item) != -1;

		public void Clear() => _items.Clear();

		public int IndexOf(Variant item) => _items.IndexOf(item);

		public int IndexOf(Variant item, int index)
		{
			if (index < 0 || index > Count)
				throw new ArgumentOutOfRangeException(nameof(index));

			for (int i = index; i < _items.Count; i++)
			{
				if (_items[i].Equals(item))
					return i;
			}

			return -1;
		}

		public int LastIndexOf(Variant item)
		{
			return LastIndexOf(item, Count - 1);
		}

		public int LastIndexOf(Variant item, int index)
		{
			if (index < 0 || index >= Count)
				throw new ArgumentOutOfRangeException(nameof(index));

			for (int i = index; i >= 0; i--)
			{
				if (_items[i].Equals(item))
					return i;
			}

			return -1;
		}

		public void Insert(int index, Variant item) => _items.Insert(index, item);

		public bool Remove(Variant item) => _items.Remove(item);

		public void RemoveAt(int index) => _items.RemoveAt(index);

		public int Count => _items.Count;

		public bool IsSynchronized => false;

		public object SyncRoot => this;

		public bool IsReadOnly => false;

		public void MakeReadOnly()
		{
			// Managed version has no engine-side read-only storage.
		}

		public void CopyTo(Variant[] array, int arrayIndex)
		{
			_items.CopyTo(array, arrayIndex);
		}

		void ICollection.CopyTo(System.Array array, int index)
		{
			for (int i = 0; i < _items.Count; i++)
				array.SetValue(_items[i], index + i);
		}

		public IEnumerator<Variant> GetEnumerator() => _items.GetEnumerator();

		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

		public override string ToString()
		{
			return "[" + string.Join(", ", _items.Select(x => x.Obj?.ToString() ?? "null")) + "]";
		}

		private static int NormalizeIndex(int index, int count)
		{
			if (index < 0)
				return Math.Max(count + index, 0);
			return Math.Min(index, count);
		}

		private static Variant DeepCopyVariant(Variant value)
		{
			if (value.Obj is Array a)
				return Variant.CreateFrom(a.Duplicate(true));

			if (value.Obj is Dictionary<object, object> d)
				return Variant.CreateFrom(new Dictionary<object, object>(d));

			return value;
		}

		private static bool VariantDeepEqual(Variant a, Variant b)
		{
			if (a.Obj is Array aa && b.Obj is Array bb)
				return aa.RecursiveEqual(bb);

			if (a.Obj is Dictionary<object, object> da && b.Obj is Dictionary<object, object> db)
				return da.Count == db.Count && !da.Except(db).Any();

			return a.Equals(b);
		}

		private static int CompareVariant(Variant a, Variant b)
		{
			object? ao = a.Obj;
			object? bo = b.Obj;

			if (ao is null && bo is null) return 0;
			if (ao is null) return -1;
			if (bo is null) return 1;

			if (ao is IComparable ca && bo != null && ao.GetType() == bo.GetType())
				return ca.CompareTo(bo);

			return string.Compare(ao.ToString(), bo.ToString(), StringComparison.Ordinal);
		}
	}
}
