#ifndef LUATUPLE_H
#define LUATUPLE_H

#ifndef LAPI_GDEXTENSION
#include "core/core_bind.h"
#include "core/object/ref_counted.h"
#else
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/callable.hpp>
#endif

#ifdef LAPI_GDEXTENSION
using namespace godot;
#endif

class LuaTuple : public RefCounted {
	GDCLASS(LuaTuple, RefCounted);

protected:
	static void _bind_methods();

public:
	static Ref<LuaTuple> fromArray(Array elms);

	void pushBack(Variant var);
	void pushFront(Variant var);
	void set(int i, Variant var);
	void clear();

	bool isEmpty();

	int size();

	Variant popBack();
	Variant popFront();
	Variant get(int i);

	Array toArray();

private:
	Array elements;
};
#endif
