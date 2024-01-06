#ifndef LUAOBJECTMETATABLE_H
#define LUAOBJECTMETATABLE_H

#ifndef LAPI_GDEXTENSION
#include "core/core_bind.h"
#include "core/object/ref_counted.h"
#else
#include <godot_cpp/classes/ref.hpp>
#endif

#include "luaAPI.h"
#include "luaError.h"
#include "luaTuple.h"

#ifdef LAPI_GDEXTENSION
using namespace godot;
#endif

class LuaObjectMetatable : public RefCounted {
	GDCLASS(LuaObjectMetatable, RefCounted);

protected:
	static void _bind_methods();

#ifndef LAPI_GDEXTENSION
	GDVIRTUAL3R(Variant, __index, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL4R(Ref<LuaError>, __newindex, Object *, Ref<LuaAPI>, Variant, Variant);
	GDVIRTUAL3R(Variant, __call, Object *, Ref<LuaAPI>, Ref<LuaTuple>);
	GDVIRTUAL2R(Ref<LuaError>, __gc, Object *, Ref<LuaAPI>);
	GDVIRTUAL2R(String, __tostring, Object *, Ref<LuaAPI>);
	GDVIRTUAL2R(int, __len, Object *, Ref<LuaAPI>);
	GDVIRTUAL2R(Variant, __unm, Object *, Ref<LuaAPI>);
	GDVIRTUAL3R(Variant, __add, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __sub, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __mul, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __div, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __idiv, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __mod, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __pow, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __band, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __bor, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __bxor, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL2R(Variant, __bnot, Object *, Ref<LuaAPI>);
	GDVIRTUAL3R(Variant, __shl, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __shr, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(Variant, __concat, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(bool, __eq, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(bool, __lt, Object *, Ref<LuaAPI>, Variant);
	GDVIRTUAL3R(bool, __le, Object *, Ref<LuaAPI>, Variant);
#endif

public:
	virtual Variant __index(Object *obj, Ref<LuaAPI> api, Variant index);
	virtual Ref<LuaError> __newindex(Object *obj, Ref<LuaAPI> api, Variant index, Variant value);
	virtual Variant __call(Object *obj, Ref<LuaAPI> api, Ref<LuaTuple> args);
	virtual Ref<LuaError> __gc(Object *obj, Ref<LuaAPI> api);
	virtual String __tostring(Object *obj, Ref<LuaAPI> api);
	virtual int __len(Object *obj, Ref<LuaAPI> api);
	virtual Variant __unm(Object *obj, Ref<LuaAPI> api);
	virtual Variant __add(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __sub(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __mul(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __div(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __idiv(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __mod(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __pow(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __band(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __bor(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __bxor(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __bnot(Object *obj, Ref<LuaAPI> api);
	virtual Variant __shl(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __shr(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual Variant __concat(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual bool __eq(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual bool __lt(Object *obj, Ref<LuaAPI> api, Variant other);
	virtual bool __le(Object *obj, Ref<LuaAPI> api, Variant other);

private:
};

// Default object metatable

class LuaDefaultObjectMetatable : public LuaObjectMetatable {
	GDCLASS(LuaDefaultObjectMetatable, LuaObjectMetatable);

protected:
	static void _bind_methods();

public:
	Variant __index(Object *obj, Ref<LuaAPI> api, Variant index) override;
	Ref<LuaError> __newindex(Object *obj, Ref<LuaAPI> api, Variant index, Variant value) override;
	Variant __call(Object *obj, Ref<LuaAPI> api, Ref<LuaTuple> args) override;
	Ref<LuaError> __gc(Object *obj, Ref<LuaAPI> api) override;
	String __tostring(Object *obj, Ref<LuaAPI> api) override;
	int __len(Object *obj, Ref<LuaAPI> api) override;
	Variant __unm(Object *obj, Ref<LuaAPI> api) override;
	Variant __add(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __sub(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __mul(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __div(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __idiv(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __mod(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __pow(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __band(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __bor(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __bxor(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __bnot(Object *obj, Ref<LuaAPI> api) override;
	Variant __shl(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __shr(Object *obj, Ref<LuaAPI> api, Variant other) override;
	Variant __concat(Object *obj, Ref<LuaAPI> api, Variant other) override;
	bool __eq(Object *obj, Ref<LuaAPI> api, Variant other) override;
	bool __lt(Object *obj, Ref<LuaAPI> api, Variant other) override;
	bool __le(Object *obj, Ref<LuaAPI> api, Variant other) override;

	void setPermissive(bool permissive);
	bool getPermissive() const;

private:
	bool permissive = true;
};

#endif
