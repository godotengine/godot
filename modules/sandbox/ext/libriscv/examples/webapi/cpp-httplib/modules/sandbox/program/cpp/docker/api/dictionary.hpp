#pragma once

#include "variant.hpp"
struct DictAccessor;

struct Dictionary {
	constexpr Dictionary() {} // DON'T TOUCH
	static Dictionary Create();

	Dictionary &operator =(const Dictionary &other);

	DictAccessor operator[](const Variant &key);
	Variant get(const Variant &key) const;
	void set(const Variant &key, const Variant &value);
	Variant get_or_add(const Variant &key, const Variant &default_value = Variant());

	int size() const;
	bool is_empty() const { return size() == 0; }

	void clear();
	void erase(const Variant &key);
	bool has(const Variant &key) const;
	void merge(const Dictionary &other);
	Dictionary duplicate(bool deep = false) const;
	Variant find_key(const Variant &key) const;
	bool has_all(const Array &keys) const;
	int hash() const;
	bool is_read_only() const;
	Variant keys() const;
	void make_read_only();
	void merge(const Dictionary &dictionary, bool overwrite = false);
	Dictionary merged(const Dictionary &dictionary, bool overwrite = false) const;
	bool recursive_equal(const Dictionary &dictionary, int recursion_count) const;
	Variant values() const;

	// Call methods on the Dictionary
	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args);
	template <typename... Args>
	Variant operator () (std::string_view method, Args&&... args) const;

	static Dictionary from_variant_index(unsigned idx) { Dictionary d; d.m_idx = idx; return d; }
	unsigned get_variant_index() const noexcept { return m_idx; }
	bool is_permanent() const { return Variant::is_permanent_index(m_idx); }

private:
	unsigned m_idx = INT32_MIN;
};

inline Dictionary Variant::as_dictionary() const {
	if (m_type != DICTIONARY) {
		api_throw("std::bad_cast", "Failed to cast Variant to Dictionary", this);
	}
	return Dictionary::from_variant_index(v.i);
}

inline Variant::Variant(const Dictionary &d) {
	m_type = DICTIONARY;
	v.i = d.get_variant_index();
}

inline Variant::operator Dictionary() const {
	return as_dictionary();
}

struct DictAccessor {
	DictAccessor(const Dictionary &dict, const Variant &key) : m_dict_idx(dict.get_variant_index()), m_key(key) {}

	operator Variant() const { return dict().get(m_key); }
	Variant operator *() const { return dict().get(m_key); }
	Variant value() const { return dict().get(m_key); }
	Variant value_or(const Variant &def) const { return dict().get_or_add(m_key, def); }

	void operator=(const Variant &value) { dict().set(m_key, value); }

	template <typename... Args>
	Variant operator ()(Args &&...args) {
		return value()(std::forward<Args>(args)...);
	}

	Dictionary dict() const { return Dictionary::from_variant_index(m_dict_idx); }

private:
	unsigned m_dict_idx;
	Variant m_key;
};

inline DictAccessor Dictionary::operator[](const Variant &key) {
	return DictAccessor(*this, key);
}

template <typename... Args>
inline Variant Dictionary::operator () (std::string_view method, Args&&... args) {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}

template <typename... Args>
inline Variant Dictionary::operator () (std::string_view method, Args&&... args) const {
	return Variant(*this).method_call(method, std::forward<Args>(args)...);
}

inline Dictionary Dictionary::duplicate(bool deep) const {
	return this->operator()("duplicate", deep);
}
inline Variant Dictionary::find_key(const Variant &key) const {
	return this->operator()("find_key", key);
}
inline bool Dictionary::has_all(const Array &keys) const {
	return this->operator()("has_all", Variant(keys));
}
inline int Dictionary::hash() const {
	return this->operator()("hash");
}
inline bool Dictionary::is_read_only() const {
	return this->operator()("is_read_only");
}
inline Variant Dictionary::keys() const {
	return this->operator()("keys");
}
inline void Dictionary::make_read_only() {
	this->operator()("make_read_only");
}
inline void Dictionary::merge(const Dictionary &dictionary, bool overwrite) {
	this->operator()("merge", Variant(dictionary), overwrite);
}
inline Dictionary Dictionary::merged(const Dictionary &dictionary, bool overwrite) const {
	return this->operator()("merged", Variant(dictionary), overwrite);
}
inline bool Dictionary::recursive_equal(const Dictionary &dictionary, int recursion_count) const {
	return this->operator()("recursive_equal", Variant(dictionary), recursion_count);
}
inline Variant Dictionary::values() const {
	return this->operator()("values");
}
