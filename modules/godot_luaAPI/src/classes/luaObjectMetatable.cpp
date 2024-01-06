#include "luaObjectMetatable.h"

#ifdef LAPI_GDEXTENSION
#define GDVIRTUAL_BIND(m, ...) BIND_VIRTUAL_METHOD(LuaObjectMetatable, m);
#define VIRTUAL_CALL(m, r, ...) r = call(#m, __VA_ARGS__);
#else
#define VIRTUAL_CALL(m, r, ...) GDVIRTUAL_CALL(m, __VA_ARGS__, r);
#endif

void LuaObjectMetatable::_bind_methods() {
	GDVIRTUAL_BIND(__index, "obj", "lua", "index");
	GDVIRTUAL_BIND(__newindex, "obj", "lua", "index", "value");
	GDVIRTUAL_BIND(__call, "obj", "lua", "args");
	GDVIRTUAL_BIND(__gc, "obj", "lua");
	GDVIRTUAL_BIND(__tostring, "obj", "lua");
	GDVIRTUAL_BIND(__len, "obj", "lua");
	GDVIRTUAL_BIND(__unm, "obj", "lua");
	GDVIRTUAL_BIND(__add, "obj", "lua", "other");
	GDVIRTUAL_BIND(__sub, "obj", "lua", "other");
	GDVIRTUAL_BIND(__mul, "obj", "lua", "other");
	GDVIRTUAL_BIND(__div, "obj", "lua", "other");
	GDVIRTUAL_BIND(__idiv, "obj", "lua", "other");
	GDVIRTUAL_BIND(__mod, "obj", "lua", "other");
	GDVIRTUAL_BIND(__pow, "obj", "lua", "other");
	GDVIRTUAL_BIND(__band, "obj", "lua", "other");
	GDVIRTUAL_BIND(__bor, "obj", "lua", "other");
	GDVIRTUAL_BIND(__bxor, "obj", "lua", "other");
	GDVIRTUAL_BIND(__bnot, "obj", "lua");
	GDVIRTUAL_BIND(__shl, "obj", "lua", "other");
	GDVIRTUAL_BIND(__shr, "obj", "lua", "other");
	GDVIRTUAL_BIND(__concat, "obj", "lua", "other");
	GDVIRTUAL_BIND(__eq, "obj", "lua", "other");
	GDVIRTUAL_BIND(__lt, "obj", "lua", "other");
	GDVIRTUAL_BIND(__le, "obj", "lua", "other");
}

Variant LuaObjectMetatable::__index(Object *obj, Ref<LuaAPI> api, Variant index) {
	Variant ret;
	VIRTUAL_CALL(__index, ret, obj, api, index);
	return ret;
}

Ref<LuaError> LuaObjectMetatable::__newindex(Object *obj, Ref<LuaAPI> api, Variant index, Variant value) {
	Ref<LuaError> ret;
	VIRTUAL_CALL(__newindex, ret, obj, api, index, value);
	return ret;
}

Variant LuaObjectMetatable::__call(Object *obj, Ref<LuaAPI> api, Ref<LuaTuple> args) {
	Variant ret;
	VIRTUAL_CALL(__call, ret, obj, api, args);
	return ret;
}

Ref<LuaError> LuaObjectMetatable::__gc(Object *obj, Ref<LuaAPI> api) {
	Ref<LuaError> ret;
	VIRTUAL_CALL(__gc, ret, obj, api);
	return ret;
}

String LuaObjectMetatable::__tostring(Object *obj, Ref<LuaAPI> api) {
	String ret;
	VIRTUAL_CALL(__tostring, ret, obj, api);
	return ret;
}

int LuaObjectMetatable::__len(Object *obj, Ref<LuaAPI> api) {
	int ret = 0;
	VIRTUAL_CALL(__len, ret, obj, api);
	return ret;
}

Variant LuaObjectMetatable::__unm(Object *obj, Ref<LuaAPI> api) {
	Variant ret;
	VIRTUAL_CALL(__unm, ret, obj, api);
	return ret;
}

Variant LuaObjectMetatable::__add(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__add, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__sub(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__sub, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__mul(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__mul, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__div(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__div, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__idiv(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__idiv, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__mod(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__mod, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__pow(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__pow, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__band(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__band, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__bor(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__bor, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__bxor(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__bxor, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__bnot(Object *obj, Ref<LuaAPI> api) {
	Variant ret;
	VIRTUAL_CALL(__bnot, ret, obj, api);
	return ret;
}

Variant LuaObjectMetatable::__shl(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__shl, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__shr(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__shr, ret, obj, api, other);
	return ret;
}

Variant LuaObjectMetatable::__concat(Object *obj, Ref<LuaAPI> api, Variant other) {
	Variant ret;
	VIRTUAL_CALL(__concat, ret, obj, api, other);
	return ret;
}

bool LuaObjectMetatable::__eq(Object *obj, Ref<LuaAPI> api, Variant other) {
	bool ret = false;
	VIRTUAL_CALL(__eq, ret, obj, api, other);
	return ret;
}

bool LuaObjectMetatable::__lt(Object *obj, Ref<LuaAPI> api, Variant other) {
	bool ret = false;
	VIRTUAL_CALL(__lt, ret, obj, api, other);
	return ret;
}

bool LuaObjectMetatable::__le(Object *obj, Ref<LuaAPI> api, Variant other) {
	bool ret = false;
	VIRTUAL_CALL(__le, ret, obj, api, other);
	return ret;
}

// Default object metatable

void LuaDefaultObjectMetatable::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_permissive"), &LuaDefaultObjectMetatable::getPermissive);
	ClassDB::bind_method(D_METHOD("set_permissive", "value"), &LuaDefaultObjectMetatable::setPermissive);
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "permissive"), "set_permissive", "get_permissive");
}

void LuaDefaultObjectMetatable::setPermissive(bool value) {
	permissive = value;
}

bool LuaDefaultObjectMetatable::getPermissive() const {
	return permissive;
}

Variant LuaDefaultObjectMetatable::__index(Object *obj, Ref<LuaAPI> api, Variant index) {
	if (obj->has_method("__index")) {
		return obj->call("__index", api, index);
	}

	Array fields = Array();
	if (obj->has_method("lua_fields")) {
		fields = obj->call("lua_fields");
	}

	if ((!permissive && fields.has((String)index)) || (permissive && !fields.has((String)index))) {
		return obj->get((String)index);
	}

	return Variant();
}

Ref<LuaError> LuaDefaultObjectMetatable::__newindex(Object *obj, Ref<LuaAPI> api, Variant index, Variant value) {
	if (obj->has_method("__newindex")) {
		Variant ret = obj->call("__newindex", api, index, value);
		if (ret.get_type() == Variant::OBJECT) {
#ifndef LAPI_GDEXTENSION
			return Object::cast_to<LuaError>(ret.operator Object *());
#else
			return dynamic_cast<LuaError *>(ret.operator Object *());
#endif
		}
	}

	Array fields = Array();
	if (obj->has_method("lua_fields")) {
		fields = obj->call("lua_fields");
	}

	if ((!permissive && fields.has(index)) || (permissive && !fields.has(index))) {
		obj->set(index, value);
		return nullptr;
	}

	return LuaError::newError(vformat("Attempt to set field '%s' on object of type '%s' which is not a valid field.", index, obj->get_class()), LuaError::ERR_RUNTIME);
}

Variant LuaDefaultObjectMetatable::__call(Object *obj, Ref<LuaAPI> api, Ref<LuaTuple> args) {
	if (obj->has_method("__call")) {
		return obj->call("__call", api, args);
	}

	return Variant();
}

Ref<LuaError> LuaDefaultObjectMetatable::__gc(Object *obj, Ref<LuaAPI> api) {
	if (obj->has_method("__gc")) {
		Variant ret = obj->call("__gc", api);
		if (ret.get_type() == Variant::OBJECT) {
#ifndef LAPI_GDEXTENSION
			return Object::cast_to<LuaError>(ret.operator Object *());
#else
			return dynamic_cast<LuaError *>(ret.operator Object *());
#endif
		}
	}

	return nullptr;
}

String LuaDefaultObjectMetatable::__tostring(Object *obj, Ref<LuaAPI> api) {
	if (obj->has_method("__tostring")) {
		return obj->call("__tostring", api);
	}

	return String();
}

Variant LuaDefaultObjectMetatable::__unm(Object *obj, Ref<LuaAPI> api) {
	if (obj->has_method("__unm")) {
		return obj->call("__unm", api);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__add(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__add")) {
		return obj->call("__add", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__sub(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__sub")) {
		return obj->call("__sub", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__mul(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__mul")) {
		return obj->call("__mul", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__div(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__div")) {
		return obj->call("__div", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__idiv(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__idiv")) {
		return obj->call("__idiv", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__mod(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__mod")) {
		return obj->call("__mod", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__pow(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__pow")) {
		return obj->call("__pow", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__band(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__band")) {
		return obj->call("__band", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__bor(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__bor")) {
		return obj->call("__bor", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__bxor(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__bxor")) {
		return obj->call("__bxor", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__bnot(Object *obj, Ref<LuaAPI> api) {
	if (obj->has_method("__bnot")) {
		return obj->call("__bnot", api);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__shl(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__shl")) {
		return obj->call("__shl", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__shr(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__shr")) {
		return obj->call("__shr", api, other);
	}

	return Variant();
}

Variant LuaDefaultObjectMetatable::__concat(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__concat")) {
		return obj->call("__concat", api, other);
	}

	return Variant();
}

int LuaDefaultObjectMetatable::__len(Object *obj, Ref<LuaAPI> api) {
	if (obj->has_method("__len")) {
		return obj->call("__len", api);
	}

	return Variant();
}

bool LuaDefaultObjectMetatable::__eq(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__eq")) {
		return obj->call("__eq", api, other);
	}

	return Variant();
}

bool LuaDefaultObjectMetatable::__lt(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__lt")) {
		return obj->call("__lt", api, other);
	}

	return Variant();
}

bool LuaDefaultObjectMetatable::__le(Object *obj, Ref<LuaAPI> api, Variant other) {
	if (obj->has_method("__le")) {
		return obj->call("__le", api, other);
	}

	return Variant();
}
