#ifndef LUACALLABLEEXTRA_H
#define LUACALLABLEEXTRA_H

#ifndef LAPI_GDEXTENSION
#include "core/core_bind.h"
#include "core/object/ref_counted.h"
#else
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/templates/vector.hpp>
#endif

#include <lua/lua.hpp>

#ifdef LAPI_GDEXTENSION
using namespace godot;
#endif

class LuaCallableExtra : public RefCounted {
	GDCLASS(LuaCallableExtra, RefCounted);

protected:
	static void _bind_methods();

public:
	static Ref<LuaCallableExtra> withTuple(Callable function, int argc);
	static Ref<LuaCallableExtra> withRef(Callable function);
	static Ref<LuaCallableExtra> withRefAndTuple(Callable function, int argc);

	void setInfo(Callable function, int argc, bool isTuple, bool wantsRef);

	void setTuple(bool value);
	bool getTuple();

	void setWantsRef(bool value);
	bool getWantsRef();

	void setArgc(int value);
	int getArgc();

	static int call(lua_State *state);

private:
	bool isTuple = false;
	bool wantsRef = false;
	int argc = 0;

	Callable function;
};
#endif
