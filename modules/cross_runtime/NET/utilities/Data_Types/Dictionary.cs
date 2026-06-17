using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

#nullable enable

namespace Godot
{

	public sealed class Dictionary :
		IDictionary<object, object>,
		IReadOnlyDictionary<object, object>,
		IDisposable
	{
		private readonly List<KeyValuePair<object, object>> _items = new();
		internal IntPtr NativePtr;

		public Dictionary() { }

		public Dictionary(ulong nativePtr)
		{
			NativePtr = (IntPtr)nativePtr;
		}

		public void Dispose() => _items.Clear();

		private int IndexOfKey(object key)
		{
			for (int i = 0; i < _items.Count; i++)
				if (Equals(_items[i].Key, key)) return i;
			return -1;
		}

		public object this[object key]
		{
			get
			{
				int i = IndexOfKey(key);
				if (i < 0) throw new KeyNotFoundException();
				return _items[i].Value;
			}
			set
			{
				int i = IndexOfKey(key);
				if (i >= 0) _items[i] = new KeyValuePair<object, object>(key, value);
				else _items.Add(new KeyValuePair<object, object>(key, value));
			}
		}

		public ICollection<object> Keys => _items.Select(kvp => kvp.Key).ToList();
		public ICollection<object> Values => _items.Select(kvp => kvp.Value).ToList();

		IEnumerable<object> IReadOnlyDictionary<object, object>.Keys => Keys;
		IEnumerable<object> IReadOnlyDictionary<object, object>.Values => Values;

		public int Count => _items.Count;
		public bool IsReadOnly => false;

		public void Add(object key, object value)
		{
			if (IndexOfKey(key) >= 0)
				throw new ArgumentException("An element with the same key already exists.", nameof(key));
			_items.Add(new KeyValuePair<object, object>(key, value));
		}

		void ICollection<KeyValuePair<object, object>>.Add(KeyValuePair<object, object> item)
			=> Add(item.Key, item.Value);

		public bool ContainsKey(object key) => IndexOfKey(key) >= 0;

		bool ICollection<KeyValuePair<object, object>>.Contains(KeyValuePair<object, object> item)
			=> TryGetValue(item.Key, out var v) && Equals(v, item.Value);

		public bool Remove(object key)
		{
			int i = IndexOfKey(key);
			if (i < 0) return false;
			_items.RemoveAt(i);
			return true;
		}

		bool ICollection<KeyValuePair<object, object>>.Remove(KeyValuePair<object, object> item)
		{
			int i = IndexOfKey(item.Key);
			if (i < 0 || !Equals(_items[i].Value, item.Value)) return false;
			_items.RemoveAt(i);
			return true;
		}

		public bool TryGetValue(object key, out object value)
		{
			int i = IndexOfKey(key);
			if (i < 0) { value = default!; return false; }
			value = _items[i].Value;
			return true;
		}

		public void Clear() => _items.Clear();

		void ICollection<KeyValuePair<object, object>>.CopyTo(KeyValuePair<object, object>[] array, int arrayIndex)
		{
			if (array == null) throw new ArgumentNullException(nameof(array));
			if (arrayIndex < 0) throw new ArgumentOutOfRangeException(nameof(arrayIndex));
			if (array.Length - arrayIndex < _items.Count) throw new ArgumentException("Destination array is too small.");
			for (int i = 0; i < _items.Count; i++)
				array[arrayIndex + i] = _items[i];
		}

		public Dictionary Duplicate()
		{
			var copy = new Dictionary();
			foreach (var kvp in _items) copy._items.Add(kvp);
			return copy;
		}

		public void Merge(Dictionary other, bool overwrite = false)
		{
			foreach (var kvp in other._items)
			{
				int i = IndexOfKey(kvp.Key);
				if (i >= 0) { if (overwrite) _items[i] = kvp; }
				else _items.Add(kvp);
			}
		}

		public bool RecursiveEqual(Dictionary other)
		{
			if (other == null || _items.Count != other._items.Count) return false;
			foreach (var kvp in _items)
			{
				int i = other.IndexOfKey(kvp.Key);
				if (i < 0 || !Equals(other._items[i].Value, kvp.Value)) return false;
			}
			return true;
		}

		public IEnumerator<KeyValuePair<object, object>> GetEnumerator()
		{
			foreach (var kvp in _items) yield return kvp;
		}

		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

		public override string ToString()
			=> "{" + string.Join(", ", _items.Select(kvp => $"{kvp.Key}: {kvp.Value}")) + "}";
	}


	// ═══════════════════════════════════════════════════════════════════════════
	// Typed wrapper
	// ═══════════════════════════════════════════════════════════════════════════

	public sealed class Dictionary<TKey, TValue> :
		IDictionary<TKey, TValue>,
		IReadOnlyDictionary<TKey, TValue>,
		IDisposable
	{
		private readonly List<KeyValuePair<object, object>> _items = new();
		internal IntPtr NativePtr;

		public Dictionary() { }

		public Dictionary(ulong nativePtr)
		{
			NativePtr = (IntPtr)nativePtr;
		}

		public Dictionary(IDictionary<TKey, TValue> dictionary)
		{
			if (dictionary == null) throw new ArgumentNullException(nameof(dictionary));
			foreach (var kvp in dictionary)
				Add(kvp.Key, kvp.Value);
		}

		public void Dispose() => _items.Clear();

		private int IndexOfKey(object key)
		{
			for (int i = 0; i < _items.Count; i++)
				if (Equals(_items[i].Key, key)) return i;
			return -1;
		}

		public TValue this[TKey key]
		{
			get
			{
				int i = IndexOfKey(key!);
				if (i < 0) throw new KeyNotFoundException();
				return (TValue)_items[i].Value;
			}
			set
			{
				int i = IndexOfKey(key!);
				if (i >= 0) _items[i] = new KeyValuePair<object, object>(key!, value!);
				else _items.Add(new KeyValuePair<object, object>(key!, value!));
			}
		}

		public ICollection<TKey> Keys => _items.Select(kvp => (TKey)kvp.Key).ToList();
		public ICollection<TValue> Values => _items.Select(kvp => (TValue)kvp.Value).ToList();

		IEnumerable<TKey> IReadOnlyDictionary<TKey, TValue>.Keys => Keys;
		IEnumerable<TValue> IReadOnlyDictionary<TKey, TValue>.Values => Values;

		public int Count => _items.Count;
		public bool IsReadOnly => false;

		public void Add(TKey key, TValue value)
		{
			if (IndexOfKey(key!) >= 0)
				throw new ArgumentException("An element with the same key already exists.", nameof(key));
			_items.Add(new KeyValuePair<object, object>(key!, value!));
		}

		void ICollection<KeyValuePair<TKey, TValue>>.Add(KeyValuePair<TKey, TValue> item)
			=> Add(item.Key, item.Value);

		public bool ContainsKey(TKey key) => IndexOfKey(key!) >= 0;

		bool ICollection<KeyValuePair<TKey, TValue>>.Contains(KeyValuePair<TKey, TValue> item)
			=> TryGetValue(item.Key, out var v) && Equals(v, item.Value);

		public bool Remove(TKey key)
		{
			int i = IndexOfKey(key!);
			if (i < 0) return false;
			_items.RemoveAt(i);
			return true;
		}

		bool ICollection<KeyValuePair<TKey, TValue>>.Remove(KeyValuePair<TKey, TValue> item)
		{
			int i = IndexOfKey(item.Key!);
			if (i < 0 || !Equals(_items[i].Value, item.Value)) return false;
			_items.RemoveAt(i);
			return true;
		}

		public bool TryGetValue(TKey key, out TValue value)
		{
			int i = IndexOfKey(key!);
			if (i < 0) { value = default!; return false; }
			value = (TValue)_items[i].Value;
			return true;
		}

		public void Clear() => _items.Clear();

		void ICollection<KeyValuePair<TKey, TValue>>.CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
		{
			if (array == null) throw new ArgumentNullException(nameof(array));
			if (arrayIndex < 0) throw new ArgumentOutOfRangeException(nameof(arrayIndex));
			if (array.Length - arrayIndex < _items.Count) throw new ArgumentException("Destination array is too small.");
			for (int i = 0; i < _items.Count; i++)
				array[arrayIndex + i] = new KeyValuePair<TKey, TValue>((TKey)_items[i].Key, (TValue)_items[i].Value);
		}

		public Dictionary<TKey, TValue> Duplicate()
		{
			var copy = new Dictionary<TKey, TValue>();
			foreach (var kvp in _items) copy._items.Add(kvp);
			return copy;
		}

		public void Merge(Dictionary<TKey, TValue> other, bool overwrite = false)
		{
			foreach (var kvp in other._items)
			{
				int i = IndexOfKey(kvp.Key);
				if (i >= 0) { if (overwrite) _items[i] = kvp; }
				else _items.Add(kvp);
			}
		}

		public bool RecursiveEqual(Dictionary<TKey, TValue> other)
		{
			if (other == null || _items.Count != other._items.Count) return false;
			foreach (var kvp in _items)
			{
				if (!other.TryGetValue((TKey)kvp.Key, out var val)) return false;
				if (!Equals(kvp.Value, val)) return false;
			}
			return true;
		}

		public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
		{
			foreach (var kvp in _items)
				yield return new KeyValuePair<TKey, TValue>((TKey)kvp.Key, (TValue)kvp.Value);
		}

		IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

		public override string ToString()
			=> "{" + string.Join(", ", _items.Select(kvp => $"{kvp.Key}: {kvp.Value}")) + "}";
	}
}
