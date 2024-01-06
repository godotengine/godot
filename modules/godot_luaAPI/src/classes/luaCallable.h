#ifndef LUACALLABLE_H
#define LUACALLABLE_H

#ifndef LAPI_GDEXTENSION
#include "core/object/ref_counted.h"
#include "core/variant/callable.h"
#else
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/variant/callable_custom.hpp>
#endif

#include <classes/luaAPI.h>

#include <lua/lua.hpp>

#ifdef LAPI_GDEXTENSION
using namespace godot;
#define LAPI_CALL_ERROR GDExtensionCallError
#else
#define LAPI_CALL_ERROR Callable::CallError
#endif

class LuaCallable : public CallableCustom {
public:
	LuaCallable(Ref<LuaAPI> obj, int ref, lua_State *p_state);
	virtual ~LuaCallable() override;

	virtual uint32_t hash() const override;
	virtual String get_as_text() const override;
	virtual CompareEqualFunc get_compare_equal_func() const override;
	virtual CompareLessFunc get_compare_less_func() const override;

	virtual ObjectID get_object() const override;

	virtual void call(const Variant **p_argument, int p_argcount, Variant &r_return_value, LAPI_CALL_ERROR &r_call_error) const override;
	virtual bool is_valid() const override;

	int getFuncRef();
	lua_State *getLuaState() const;

private:
	int funcRef;
	ObjectID objectID;
	lua_State *state = nullptr;
	uint32_t h;

	static bool compare_equal(const CallableCustom *p_a, const CallableCustom *p_b);
	static bool compare_less(const CallableCustom *p_a, const CallableCustom *p_b);
};

#endif