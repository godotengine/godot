#ifndef LUAFUNCTIONREF_H
#define LUAFUNCTIONREF_H

#ifndef LAPI_GDEXTENSION
#include "core/core_bind.h"
#include "core/object/ref_counted.h"
#else
#include <godot_cpp/classes/ref.hpp>
#endif

#include <lua/lua.hpp>

#ifdef LAPI_GDEXTENSION
using namespace godot;
#endif

class LuaFunctionRef : public RefCounted {
	GDCLASS(LuaFunctionRef, RefCounted);

protected:
	static void _bind_methods();

public:
	LuaFunctionRef();
	~LuaFunctionRef();

	void setLuaState(lua_State *state);
	void setRef(int ref);

	Variant invoke(Array args);

	inline int getRef() const { return m_ref; }
	inline lua_State *getLuaState() const { return L; }

private:
	lua_State *L;
	int m_ref;
};

#endif