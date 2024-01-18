#include "luaState.h"

#include <classes/luaAPI.h>
#include <classes/luaCallableExtra.h>
#include <classes/luaObjectMetatable.h>
#include <classes/luaTuple.h>

// These 2 macros helps us in constructing general metamethods.
// We can use "lua" as a "Lua" pointer and arg1, arg2, ..., arg5 as Variants objects
// Check examples in createVector2Metatable
#define LUA_LAMBDA_TEMPLATE(_f_)                             \
	[](lua_State *inner_state) -> int {                      \
	    Variant* arg1_ptr = nullptr; \
	    Variant* arg2_ptr = nullptr; \
	    Variant* arg3_ptr = nullptr; \
	    Variant* arg4_ptr = nullptr; \
	    Variant* arg5_ptr = nullptr; \
		Variant* arg6_ptr = nullptr; \
		Variant* arg7_ptr = nullptr; \
		Variant* arg8_ptr = nullptr; \
		Variant arg1 = LuaState::getVariant(inner_state, 1, &arg1_ptr); \
		Variant arg2 = LuaState::getVariant(inner_state, 2, &arg2_ptr); \
		Variant arg3 = LuaState::getVariant(inner_state, 3, &arg3_ptr); \
		Variant arg4 = LuaState::getVariant(inner_state, 4	, &arg4_ptr); \
		Variant arg5 = LuaState::getVariant(inner_state, 5, &arg5_ptr); \
		Variant arg6 = LuaState::getVariant(inner_state, 6, &arg6_ptr); \
		Variant arg7 = LuaState::getVariant(inner_state, 7, &arg7_ptr); \
		Variant arg8 = LuaState::getVariant(inner_state, 8, &arg8_ptr); \
		_f_                                                  \
	}

#define LUA_METAMETHOD_TEMPLATE(lua_state, metatable_index, metamethod_name, _f_) \
	lua_pushstring(lua_state, metamethod_name);                                   \
	lua_pushcfunction(lua_state, LUA_LAMBDA_TEMPLATE(_f_));                       \
	lua_settable(lua_state, metatable_index - 2);

// Expose the default constructors
void LuaState::exposeConstructors() {
	lua_pushcfunction(L, LUA_LAMBDA_TEMPLATE({
		int argc = lua_gettop(inner_state);
		if (argc == 0) {
			LuaState::pushVariant(inner_state, Vector2());
		} else {
			LuaState::pushVariant(inner_state, Vector2(arg1.operator double(), arg2.operator double()));
		}
		return 1;
	}));
	lua_setglobal(L, "Vector2");

	lua_pushcfunction(L, LUA_LAMBDA_TEMPLATE({
		int argc = lua_gettop(inner_state);
		if (argc == 0) {
			LuaState::pushVariant(inner_state, Vector3());
		} else {
			LuaState::pushVariant(inner_state, Vector3(arg1.operator double(), arg2.operator double(), arg3.operator double()));
		}
		return 1;
	}));
	lua_setglobal(L, "Vector3");

	lua_pushcfunction(L, LUA_LAMBDA_TEMPLATE({
		int argc = lua_gettop(inner_state);
		if (argc == 3) {
			LuaState::pushVariant(inner_state, Color(arg1.operator double(), arg2.operator double(), arg3.operator double()));
		} else if (argc == 4) {
			LuaState::pushVariant(inner_state, Color(arg1.operator double(), arg2.operator double(), arg3.operator double(), arg4.operator double()));
		} else {
			LuaState::pushVariant(inner_state, Color());
		}
		return 1;
	}));
	lua_setglobal(L, "Color");

	lua_pushcfunction(L, LUA_LAMBDA_TEMPLATE({
		int argc = lua_gettop(inner_state);
		if (argc == 2) {
			LuaState::pushVariant(inner_state, Rect2(arg1.operator Vector2(), arg2.operator Vector2()));
		} else if (argc == 4) {
			LuaState::pushVariant(inner_state, Rect2(arg1.operator double(), arg2.operator double(), arg3.operator double(), arg4.operator double()));
		} else {
			LuaState::pushVariant(inner_state, Rect2());
		}
		return 1;
	}));
	lua_setglobal(L, "Rect2");

	lua_pushcfunction(L, LUA_LAMBDA_TEMPLATE({
		int argc = lua_gettop(inner_state);
		if (argc == 4) {
			LuaState::pushVariant(inner_state, Plane(arg1.operator double(), arg2.operator double(), arg3.operator double(), arg4.operator double()));
		} else if (argc == 3) {
			LuaState::pushVariant(inner_state, Plane(arg1.operator Vector3(), arg2.operator Vector3(), arg3.operator Vector3()));
		} else {
			LuaState::pushVariant(inner_state, Plane(arg1.operator Vector3(), arg1.operator double()));
		}
		return 1;
	}));
	lua_setglobal(L, "Plane");
}

// Create metatable for Vector2 and saves it at LUA_REGISTRYINDEX with name "mt_Vector2"
void LuaState::createVector2Metatable() {
	luaL_newmetatable(L, "mt_Vector2");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		// We can't use arg1 here because we need to reference the userdata
		((Variant *)lua_touserdata(inner_state, 1))->set(arg2, arg3);
		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__add", {
		LuaState::pushVariant(inner_state, arg1.operator Vector2() + arg2.operator Vector2());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__sub", {
		LuaState::pushVariant(inner_state, arg1.operator Vector2() - arg2.operator Vector2());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mul", {
		switch (arg2.get_type()) {
			case Variant::Type::VECTOR2:
				LuaState::pushVariant(inner_state, arg1.operator Vector2() * arg2.operator Vector2());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Vector2() * arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__div", {
		switch (arg2.get_type()) {
			case Variant::Type::VECTOR2:
				LuaState::pushVariant(inner_state, arg1.operator Vector2() / arg2.operator Vector2());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Vector2() / arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaState::pushVariant(inner_state, arg1.operator Vector2() == arg2.operator Vector2());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__lt", {
		LuaState::pushVariant(inner_state, arg1.operator Vector2() < arg2.operator Vector2());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__le", {
		LuaState::pushVariant(inner_state, arg1.operator Vector2() <= arg2.operator Vector2());
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for Vector3 and saves it at LUA_REGISTRYINDEX with name "mt_Vector3"
void LuaState::createVector3Metatable() {
	luaL_newmetatable(L, "mt_Vector3");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		// We can't use arg1 here because we need to reference the userdata
		((Variant *)lua_touserdata(inner_state, 1))->set(arg2, arg3);
		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__add", {
		LuaState::pushVariant(inner_state, arg1.operator Vector3() + arg2.operator Vector3());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__sub", {
		LuaState::pushVariant(inner_state, arg1.operator Vector3() - arg2.operator Vector3());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mul", {
		switch (arg2.get_type()) {
			case Variant::Type::VECTOR3:
				LuaState::pushVariant(inner_state, arg1.operator Vector3() * arg2.operator Vector3());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Vector3() * arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__div", {
		switch (arg2.get_type()) {
			case Variant::Type::VECTOR3:
				LuaState::pushVariant(inner_state, arg1.operator Vector3() / arg2.operator Vector3());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Vector3() / arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaState::pushVariant(inner_state, arg1.operator Vector3() == arg2.operator Vector3());
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for Rect2 and saves it at LUA_REGISTRYINDEX with name "mt_Rect2"
void LuaState::createRect2Metatable() {
	luaL_newmetatable(L, "mt_Rect2");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		// We can't use arg1 here because we need to reference the userdata
		((Variant *)lua_touserdata(inner_state, 1))->set(arg2, arg3);
		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaState::pushVariant(inner_state, arg1.operator Rect2() == arg2.operator Rect2());
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for Plane and saves it at LUA_REGISTRYINDEX with name "mt_Plane"
void LuaState::createPlaneMetatable() {
	luaL_newmetatable(L, "mt_Plane");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		// We can't use arg1 here because we need to reference the userdata
		((Variant *)lua_touserdata(inner_state, 1))->set(arg2, arg3);
		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaState::pushVariant(inner_state, arg1.operator Plane() == arg2.operator Plane());
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for Color and saves it at LUA_REGISTRYINDEX with name "mt_Color"
void LuaState::createColorMetatable() {
	luaL_newmetatable(L, "mt_Color");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		// We can't use arg1 here because we need to reference the userdata
		((Variant *)lua_touserdata(inner_state, 1))->set(arg2, arg3);
		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__add", {
		LuaState::pushVariant(inner_state, arg1.operator Color() + arg2.operator Color());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__sub", {
		LuaState::pushVariant(inner_state, arg1.operator Color() - arg2.operator Color());
		return 1;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mul", {
		switch (arg2.get_type()) {
			case Variant::Type::COLOR:
				LuaState::pushVariant(inner_state, arg1.operator Color() * arg2.operator Color());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Color() * arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__div", {
		switch (arg2.get_type()) {
			case Variant::Type::COLOR:
				LuaState::pushVariant(inner_state, arg1.operator Color() / arg2.operator Color());
				return 1;
			case Variant::Type::INT:
			case Variant::Type::FLOAT:
				LuaState::pushVariant(inner_state, arg1.operator Color() / arg2.operator double());
				return 1;
			default:
				return 0;
		}
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaState::pushVariant(inner_state, arg1.operator Color() == arg2.operator Color());
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for Signal and saves it at LUA_REGISTRYINDEX with name "mt_Signal"
void LuaState::createSignalMetatable() {
	luaL_newmetatable(L, "mt_Signal");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		if (arg1.has_method(arg2.operator String())) {
			lua_pushlightuserdata(inner_state, lua_touserdata(inner_state, 1));
			LuaState::pushVariant(inner_state, arg2);
			lua_pushcclosure(inner_state, luaUserdataFuncCall, 2);
			return 1;
		}

		LuaState::pushVariant(inner_state, arg1.get(arg2));
		return 1;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1); // Stack is now unmodified
}

// Create metatable for any Object and saves it at LUA_REGISTRYINDEX with name "mt_Object"
void LuaState::createObjectMetatable() {
	luaL_newmetatable(L, "mt_Object");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			Variant ret = mt->__index(arg1, api, arg2);
			LuaState::pushVariant(inner_state, ret);
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			Ref<LuaError> err = mt->__newindex(arg1, api, arg2, arg3);
			if (!err.is_null()) {
				LuaState::pushVariant(inner_state, err);
				return 1;
			}
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__call", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			int argc = lua_gettop(inner_state);

			Array args;
			for (int i = 1; i < argc; i++) {
				args.push_back(LuaState::getVariant(inner_state, i + 1));
			}

			Variant ret = mt->__call(arg1, api, LuaTuple::fromArray(args));
			LuaState::pushVariant(inner_state, ret);
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__gc", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		// Sometimes the api ref is cleaned up first, so we need to check for that
		if (!mt.is_valid() && api != nullptr) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			Ref<LuaError> err = mt->__gc(arg1, api);
			if (!err.is_null()) {
				LuaState::pushVariant(inner_state, err);
			}
		}

		// We need to manually uncount the ref
		if (Ref<RefCounted> ref = Object::cast_to<RefCounted>(arg1); ref.is_valid()) {
			ref->unreference();
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__tostring", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__tostring(arg1, api));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__len", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__len(arg1, api));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__unm", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__unm(arg1, api));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__add", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__add(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__sub", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__sub(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mul", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__mul(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__div", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__div(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__idiv", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__idiv(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mod", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__mod(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__pow", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__pow(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__concat", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__concat(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__band", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__band(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__bor", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__bor(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__bxor", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__bxor(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__bnot", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__bnot(arg1, api));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__shl", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");
		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__shl(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__shr", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");

		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__shr(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");

		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__eq(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__lt", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");

		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__lt(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	LUA_METAMETHOD_TEMPLATE(L, -1, "__le", {
		LuaAPI* api =LuaAPI::get_singleton();
		Ref<LuaObjectMetatable> mt = arg1.get("lua_metatable");

		if (!mt.is_valid()) {
			mt = api->getObjectMetatable();
		}

		if (mt.is_valid()) {
			LuaState::pushVariant(inner_state, mt->__le(arg1, api, arg2));
			return 1;
		}

		return 0;
	});

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1);
}

// Create metatable for any Callable and saves it at LUA_REGISTRYINDEX with name "mt_Callable"
void LuaState::createCallableMetatable() {
	luaL_newmetatable(L, "mt_Callable");

	lua_pushstring(L, "__call");
	lua_pushcfunction(L, luaCallableCall);
	lua_settable(L, -3);

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1);
}

// Create metatable for any Callable and saves it at LUA_REGISTRYINDEX with name "mt_Callable"
void LuaState::createCallableExtraMetatable() {
	luaL_newmetatable(L, "mt_CallableExtra");

	LUA_METAMETHOD_TEMPLATE(L, -1, "__gc", {
		// We need to manually uncount the ref
		if (Ref<RefCounted> ref = Object::cast_to<RefCounted>(arg1); ref.is_valid()) {
			ref->unreference();
		}

		return 0;
	});

	lua_pushstring(L, "__call");
	lua_pushcfunction(L, LuaCallableExtra::call);
	lua_settable(L, -3);

	lua_pushliteral(L, "__metatable");
	lua_pushliteral(L, METATABLE_DISCLAIMER);
	lua_settable(L, -3);

	lua_pop(L, 1);
}

void LuaState::createInt64Metatable()
{
	 luaL_newmetatable(L, "mt_Int64");

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		 lua_pushboolean(inner_state, arg1.operator int64_t() == arg2.operator int64_t());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__lt", {
		 lua_pushboolean(inner_state, arg1.operator int64_t() < arg2.operator int64_t()); 
		 return 1;
	 })


	 LUA_METAMETHOD_TEMPLATE(L, -1, "__le", {
		 lua_pushboolean(inner_state, arg1.operator int64_t() <= arg2.operator int64_t());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__tostring", {
		 lua_pushfstring(inner_state, "%lld", arg1.operator int64_t());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__add", {
		 LuaState::pushVariant(inner_state, arg1.operator int64_t() + arg2.operator int64_t());
		 return 1;
		 
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__sub", {
		 LuaState::pushVariant(inner_state, arg1.operator int64_t() - arg2.operator int64_t());
		 return 1; 
	 })


	 LUA_METAMETHOD_TEMPLATE(L, -1, "__mul", {
		 LuaState::pushVariant(inner_state, arg1.operator int64_t() * arg2.operator int64_t()); 
		 return 1;
	 })

	LUA_METAMETHOD_TEMPLATE(L, -1, "__div", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() / arg2.operator int64_t());
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__mod", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() % arg2.operator int64_t());
		return 1;	
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__unm", {
		LuaState::pushVariant(inner_state, -arg1.operator int64_t());
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__bnot", {
		LuaState::pushVariant(inner_state, ~arg1.operator int64_t());
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__band", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() & arg2.operator int64_t());
		return 1;
	})


	LUA_METAMETHOD_TEMPLATE(L, -1, "__bor", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() | arg2.operator int64_t());
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__bxor", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() ^ arg2.operator int64_t());
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__shl", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() << arg2.operator int64_t());
		return 1;
	})
     

	LUA_METAMETHOD_TEMPLATE(L, -1, "__shr", {
		LuaState::pushVariant(inner_state, arg1.operator int64_t() >> arg2.operator int64_t());	
		return 1;
	})

	LUA_METAMETHOD_TEMPLATE(L, -1, "__pow", {
		LuaState::pushVariant(inner_state, pow(arg1.operator int64_t(), arg2.operator int64_t()));
		return 1;
	})




}


#define LUA_VECTOR_METATABLE(mate_name,type,type_name,push_set_func,none_con)\
{\
     luaL_newmetatable(L, mate_name);\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {\
		const Vector<type> pba = arg1.operator Vector<type>();\
		int index = arg2;\
		if (pba.size() > index) {\
			push_set_func(inner_state, pba[index]);\
			return 1;\
		}\
		push_set_func(inner_state,none_con);\
		return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 int index = arg2;\
		 if (pba->size() > index) {\
			 pba->set(index, arg3);\
			 return 0;\
		 }\
		 return 0;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__len", {\
		 lua_pushinteger(inner_state, arg1.operator Vector<type>().size());\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {\
		 const Vector<type> pba = arg1.operator Vector<type>();\
		 const Vector<type> pba2 = arg2.operator Vector<type>();\
		 lua_pushboolean(inner_state, pba == pba2);\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__lt", {\
		 const Vector<type> pba = arg1.operator Vector<type>();\
		 const Vector<type> pba2 = arg2.operator Vector<type>();\
		 lua_pushboolean(inner_state, pba < pba2);\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "__le", {\
		 const Vector<type> pba = arg1.operator Vector<type>();\
		 const Vector<type> pba2 = arg2.operator Vector<type>();\
		 lua_pushboolean(inner_state, pba <= pba2);\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "clear", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->clear();\
		 return 0;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "sort", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->sort();\
		 return 0;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "push_back", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->push_back(arg2);\
		 return 0;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "fill", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->fill(arg2);\
		 return 0;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "slice", {\
		 const Vector<type> pba = arg1.operator Vector<type>();\
		 const Vector<type> pba2 = pba.slice(arg2, arg3);\
		 LuaState::pushVariant(inner_state, pba2);\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "find", {\
		 const Vector<type> pba = arg1.operator Vector<type>();\
		 lua_pushinteger(inner_state, pba.find(arg2, arg3));\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "resize", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->resize(arg2);\
		 return 1;\
	 })\
	 LUA_METAMETHOD_TEMPLATE(L, -1, "remove_at", {\
		 Vector<type>* pba = arg1_ptr->get_ ## type_name ## _array();\
		 pba->remove_at(arg2);\
		 return 0;\
	 })\
}

void LuaState::createVectorBytesMetatable()
LUA_VECTOR_METATABLE("mt_VectorBytes",uint8_t,byte,lua_pushinteger,0)


void LuaState::createVectorInt64Metatable()
LUA_VECTOR_METATABLE("mt_VectorInt64",int64_t,int64,LuaState::push_int64,0LL)

void LuaState::createVectorFloat32Metatable()
LUA_VECTOR_METATABLE("mt_VectorFloat32",float,float32,lua_pushnumber,0.0f)


void LuaState::createVectorFloat64Metatable()
LUA_VECTOR_METATABLE("mt_VectorFloat64",double,float64,lua_pushnumber,0.0)

void LuaState::createVectorStringMetatable()
LUA_VECTOR_METATABLE("mt_VectorString",String,string,LuaState::pushVariant,String())



void LuaState::createVectorVector2Metatable()
LUA_VECTOR_METATABLE("mt_VectorVector2",Vector2,vector2,LuaState::pushVariant,Vector2())

void LuaState::createVectorVector3Metatable()
LUA_VECTOR_METATABLE("mt_VectorVector3",Vector3,vector3,LuaState::pushVariant,Vector3())
	

void LuaState::createVectorColorMetatable()
LUA_VECTOR_METATABLE("mt_VectorColor",Color,color,LuaState::pushVariant,Color())

void LuaState::createArrayMetatable()
{
     luaL_newmetatable(L, "mt_Array");

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		
		Array pba = arg1.operator Array();
		int index = arg2;
		if (pba.size() > index) {
			LuaState::pushVariant(inner_state, pba[index]);
			return 1;
		}
		lua_pushnil(inner_state);
		return 1;

	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		 Array pba = arg1.operator Array();
		 int index = arg2;
		 if (pba.size() > index) {
			 pba[index] = arg3;
			 return 0;
		 }
		 return 0;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__len", {
		 lua_pushinteger(inner_state, arg1.operator Array().size());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		 Array pba = arg1.operator Array();
		 Array pba2 = arg2.operator Array();
		 lua_pushboolean(inner_state, pba == pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__lt", {
		 Array pba = arg1.operator Array();
		 Array pba2 = arg2.operator Array();
		 lua_pushboolean(inner_state, pba < pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__le", {
		 Array pba = arg1.operator Array();
		 Array pba2 = arg2.operator Array();
		 lua_pushboolean(inner_state, pba <= pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "clear", {
		 Array pba = arg1.operator Array();
		 pba.clear();
		 return 0;
	 })

	 
	 LUA_METAMETHOD_TEMPLATE(L, -1, "sort", {
		 Array pba = arg1.operator Array();
		 pba.sort();
		 return 0;
	 })

	 
	 LUA_METAMETHOD_TEMPLATE(L, -1, "push_back", {
		 Array pba = arg1.operator Array();
		 pba.push_back(arg2);
		 return 0;
	 })

	 
	 LUA_METAMETHOD_TEMPLATE(L, -1, "fill", {
		 Array pba = arg1.operator Array();
		 pba.fill(arg2);
		 return 0;
	 })


	 LUA_METAMETHOD_TEMPLATE(L, -1, "slice", {
		 Array pba = arg1.operator Array();
		 Array pba2 = pba.slice(arg2, arg3);
		 LuaState::pushVariant(inner_state, pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "find", {
		 Array pba = arg1.operator Array();
		 lua_pushinteger(inner_state, pba.find(arg2, arg3));
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "resize", {
		 Array pba = arg1.operator Array();
		 pba.resize(arg2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "remove_at", {
		 Array pba = arg1.operator Array();
		 pba.remove_at(arg2);
		 return 0;
	 })
}


void LuaState::createDictionaryMetatable()
{
     luaL_newmetatable(L, "mt_Dictionary");

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__index", {
		
		Dictionary pba = arg1.operator Dictionary();
		if (pba.has(arg2)) {
			LuaState::pushVariant(inner_state, pba[arg2]);
			return 1;
		}
		lua_pushnil(inner_state);
		return 1;

	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__newindex", {
		 Dictionary pba = arg1.operator Dictionary();
		 int index = arg2;
		 pba[index] = arg3;
		 return 0;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__len", {
		 lua_pushinteger(inner_state, arg1.operator Dictionary().size());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "__eq", {
		 Dictionary pba = arg1.operator Dictionary();
		 Dictionary pba2 = arg2.operator Dictionary();
		 lua_pushboolean(inner_state, pba == pba2);
		 return 1;
	 })


	 LUA_METAMETHOD_TEMPLATE(L, -1, "is_empty", {
		 Dictionary pba = arg1.operator Dictionary();
		 lua_pushboolean(inner_state, pba.is_empty());
		 return 1;
	 })



	 LUA_METAMETHOD_TEMPLATE(L, -1, "merge", {
		 Dictionary pba = arg1.operator Dictionary();
		 Dictionary pba2 = arg2.operator Dictionary();
		 pba.merge(pba2, arg3);
		 return 0;
	 })
	 LUA_METAMETHOD_TEMPLATE(L, -1, "has", {
		 Dictionary pba = arg1.operator Dictionary();
		 lua_pushboolean(inner_state, pba.has(arg2));
		 return 1;
	 })
	 LUA_METAMETHOD_TEMPLATE(L, -1, "keys", {
		 Dictionary p = arg1.operator Dictionary();
		 Array pba = p.keys();
		 LuaState::pushVariant(inner_state, pba);
		 return 1;
	 })
	 LUA_METAMETHOD_TEMPLATE(L, -1, "erase", {
		 Dictionary pba = arg1.operator Dictionary();
		 pba.erase(arg2);
		 return 0;
	 })
	 LUA_METAMETHOD_TEMPLATE(L, -1, "values", {
		 Dictionary pba = arg1.operator Dictionary();
		 Array pba2 = pba.values();
		 LuaState::pushVariant(inner_state, pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "duplicate", {
		 Dictionary pba = arg1.operator Dictionary();
		 Dictionary pba2 = pba.duplicate();
		 LuaState::pushVariant(inner_state, pba2);
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "make_read_only", {
		 Dictionary pba = arg1.operator Dictionary();
		 pba.make_read_only();
		 return 0;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "is_read_only", {
		 Dictionary pba = arg1.operator Dictionary();
		 lua_pushboolean(inner_state, pba.is_read_only());
		 return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "find_key", {
		 Dictionary pba = arg1.operator Dictionary();
		lua_pushinteger(inner_state, pba.find_key(arg2));
		return 1;
	 })

	 LUA_METAMETHOD_TEMPLATE(L, -1, "init_form", {
		*arg1_ptr = (arg2);
		return 0;
	 })



}




















