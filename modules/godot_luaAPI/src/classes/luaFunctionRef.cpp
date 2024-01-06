#include "luaFunctionRef.h"

#include <luaState.h>

void LuaFunctionRef::_bind_methods() {
	ClassDB::bind_method(D_METHOD("invoke", "args"), &LuaFunctionRef::invoke);
}

LuaFunctionRef::LuaFunctionRef() {
	L = nullptr;
	m_ref = LUA_NOREF;
}

LuaFunctionRef::~LuaFunctionRef() {
	luaL_unref(L, LUA_REGISTRYINDEX, m_ref);
}

void LuaFunctionRef::setLuaState(lua_State *state) {
	L = state;
}

void LuaFunctionRef::setRef(int ref) {
	this->m_ref = ref;
}

Variant LuaFunctionRef::invoke(Array args) {
	lua_pushcfunction(L, LuaState::luaErrorHandler);
	lua_rawgeti(L, LUA_REGISTRYINDEX, m_ref);

	for (int i = 0; i < args.size(); i++) {
		Variant arg = args[i];
		LuaState::pushVariant(L, arg);
	}

	int err = lua_pcall(L, args.size(), 1, -2 - args.size());
	Variant ret;
	if (err) {
		ret = LuaState::handleError(L, ret);
	} else {
		ret = LuaState::getVariant(L, -1);
	}

	lua_pop(L, 1);

	return ret;
}