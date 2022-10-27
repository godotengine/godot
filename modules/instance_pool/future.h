#ifndef FUTURE_H
#define FUTURE_H

#include <chrono>
#include <cstdint>

#include "core/reference.h"

class Future : public Reference {
	GDCLASS(Future, Reference)
private:
	Variant _future_to;
	bool available = false;
	bool legit = false;

	inline void swap_with(const Variant& new_vessel) { _future_to = new_vessel; available = true; }
protected:
	static void _bind_methods();
public:
	Future() = default;
	~Future() = default;

	friend class WorkPool;

	inline Variant get_value() const { return _future_to; }
	Variant await() const;
	inline bool is_available() const { return available; }
	inline bool is_legit() const { return legit; }
};

#endif
