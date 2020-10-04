#ifndef DICTIONARY_H
#define DICTIONARY_H

#include "Variant.hpp"

#include "Array.hpp"

#include <gdnative/dictionary.h>

namespace godot {

class Dictionary {
	godot_dictionary _godot_dictionary;

	friend Variant::operator Dictionary() const;
	inline explicit Dictionary(const godot_dictionary &other) {
		_godot_dictionary = other;
	}

public:
	Dictionary();
	Dictionary(const Dictionary &other);
	Dictionary &operator=(const Dictionary &other);

	template <class... Args>
	static Dictionary make(Args... args) {
		return helpers::add_all(Dictionary(), args...);
	}

	void clear();

	bool empty() const;

	void erase(const Variant &key);

	bool has(const Variant &key) const;

	bool has_all(const Array &keys) const;

	uint32_t hash() const;

	Array keys() const;

	Variant &operator[](const Variant &key);

	const Variant &operator[](const Variant &key) const;

	int size() const;

	String to_json() const;

	Array values() const;

	~Dictionary();
};

} // namespace godot

#endif // DICTIONARY_H
