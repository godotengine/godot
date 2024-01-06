#include "luaCallable.h"
#include "luaAPI.h"

#ifndef LAPI_GDEXTENSION
#include "core/templates/hashfuncs.h"
#else
#include <gdextension_interface.h>
#include <godot_cpp/templates/hashfuncs.hpp>
#endif

// I used "GDScriptLambdaCallable" as a template for this
LuaCallable::LuaCallable(Ref<LuaAPI> obj, int ref, lua_State *p_state) {
	objectID = obj->get_instance_id();
	funcRef = ref;
	state = p_state;
	h = (uint32_t)hash_djb2_one_64((uint64_t)this);
}

LuaCallable::~LuaCallable() {
	luaL_unref(state, LUA_REGISTRYINDEX, funcRef);
}

bool LuaCallable::compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) {
	// Lua callables are only compared by reference.
	return p_a == p_b;
}

bool LuaCallable::compare_less(const CallableCustom *p_a, const CallableCustom *p_b) {
	// Lua callables are only compared by reference.
	return p_a < p_b;
}

CallableCustom::CompareEqualFunc LuaCallable::get_compare_equal_func() const {
	return compare_equal;
}

CallableCustom::CompareLessFunc LuaCallable::get_compare_less_func() const {
	return compare_less;
}

ObjectID LuaCallable::get_object() const {
	return objectID;
}

String LuaCallable::get_as_text() const {
	// I dont know of a way to get a useful name from the function
	// For now we are just using the callables hash.
	return vformat("LuaCallable 0x%X", h);
}

lua_State *LuaCallable::getLuaState() const {
	return state;
}

uint32_t LuaCallable::hash() const {
	return h;
}

bool LuaCallable::is_valid() const {
	return ObjectDB::get_instance(objectID);
}

void LuaCallable::call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, LAPI_CALL_ERROR &r_call_error) const {
	lua_pushcfunction(state, LuaState::luaErrorHandler);

	// Getting the lua function via the reference stored in funcRef
	lua_rawgeti(state, LUA_REGISTRYINDEX, funcRef);

	// Push all the argument on to the stack
	for (int i = 0; i < p_argcount; i++) {
		LuaState::pushVariant(state, *p_arguments[i]);
	}

	// execute the function using a protected call.
	int ret = lua_pcall(state, p_argcount, 1, -2 - p_argcount);
	if (ret != LUA_OK) {
		r_return_value = LuaState::handleError(state, ret);
	} else {
		r_return_value = LuaState::getVariant(state, -1);
	}

	lua_pop(state, 1);
// TODO: Tie the error handling systems together?
#ifndef LAPI_GDEXTENSION
	r_call_error.error = LAPI_CALL_ERROR::CALL_OK;
#else
	r_call_error.error = GDExtensionCallErrorType::GDEXTENSION_CALL_OK;
#endif
}

int LuaCallable::getFuncRef() {
	return funcRef;
}