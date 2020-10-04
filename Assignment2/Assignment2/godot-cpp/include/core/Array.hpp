#ifndef ARRAY_H
#define ARRAY_H

#include <gdnative/array.h>

#include "String.hpp"

namespace godot {

namespace helpers {
template <typename T, typename ValueT>
T append_all(T appendable, ValueT value) {
	appendable.append(value);
	return appendable;
}

template <typename T, typename ValueT, typename... Args>
T append_all(T appendable, ValueT value, Args... args) {
	appendable.append(value);
	return append_all(appendable, args...);
}

template <typename T>
T append_all(T appendable) {
	return appendable;
}

template <typename KV, typename KeyT, typename ValueT>
KV add_all(KV kv, KeyT key, ValueT value) {
	kv[key] = value;
	return kv;
}

template <typename KV, typename KeyT, typename ValueT, typename... Args>
KV add_all(KV kv, KeyT key, ValueT value, Args... args) {
	kv[key] = value;
	return add_all(kv, args...);
}

template <typename KV>
KV add_all(KV kv) {
	return kv;
}
} // namespace helpers

class Variant;
class PoolByteArray;
class PoolIntArray;
class PoolRealArray;
class PoolStringArray;
class PoolVector2Array;
class PoolVector3Array;
class PoolColorArray;

class Object;

class Array {
	godot_array _godot_array;

	friend class Variant;
	inline explicit Array(const godot_array &other) {
		_godot_array = other;
	}

public:
	Array();
	Array(const Array &other);
	Array &operator=(const Array &other);

	Array(const PoolByteArray &a);

	Array(const PoolIntArray &a);

	Array(const PoolRealArray &a);

	Array(const PoolStringArray &a);

	Array(const PoolVector2Array &a);

	Array(const PoolVector3Array &a);

	Array(const PoolColorArray &a);

	template <class... Args>
	static Array make(Args... args) {
		return helpers::append_all(Array(), args...);
	}

	Variant &operator[](const int idx);

	Variant operator[](const int idx) const;

	void append(const Variant &v);

	void clear();

	int count(const Variant &v);

	bool empty() const;

	void erase(const Variant &v);

	Variant front() const;

	Variant back() const;

	int find(const Variant &what, const int from = 0);

	int find_last(const Variant &what);

	bool has(const Variant &what) const;

	uint32_t hash() const;

	void insert(const int pos, const Variant &value);

	void invert();

	bool is_shared() const;

	Variant pop_back();

	Variant pop_front();

	void push_back(const Variant &v);

	void push_front(const Variant &v);

	void remove(const int idx);

	int size() const;

	void resize(const int size);

	int rfind(const Variant &what, const int from = -1);

	void sort();

	void sort_custom(Object *obj, const String &func);

	int bsearch(const Variant &value, const bool before = true);

	int bsearch_custom(const Variant &value, const Object *obj,
			const String &func, const bool before = true);

	Array duplicate(const bool deep = false) const;

	Variant max() const;

	Variant min() const;

	void shuffle();

	~Array();
};

} // namespace godot

#endif // ARRAY_H
