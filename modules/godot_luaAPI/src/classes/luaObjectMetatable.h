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
	GDVIRTUAL3R(Variant, __index, Object *, LuaAPI*, Variant);
	GDVIRTUAL4R(Ref<LuaError>, __newindex, Object *, LuaAPI*, Variant, Variant);
	GDVIRTUAL3R(Variant, __call, Object *, LuaAPI*, Ref<LuaTuple>);
	GDVIRTUAL2R(Ref<LuaError>, __gc, Object *, LuaAPI*);
	GDVIRTUAL2R(String, __tostring, Object *, LuaAPI*);
	GDVIRTUAL2R(int, __len, Object *, LuaAPI*);
	GDVIRTUAL2R(Variant, __unm, Object *, LuaAPI*);
	GDVIRTUAL3R(Variant, __add, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __sub, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __mul, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __div, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __idiv, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __mod, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __pow, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __band, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __bor, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __bxor, Object *, LuaAPI*, Variant);
	GDVIRTUAL2R(Variant, __bnot, Object *, LuaAPI*);
	GDVIRTUAL3R(Variant, __shl, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __shr, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(Variant, __concat, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(bool, __eq, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(bool, __lt, Object *, LuaAPI*, Variant);
	GDVIRTUAL3R(bool, __le, Object *, LuaAPI*, Variant);
#endif

public:
	virtual Variant __index(Object *obj, LuaAPI*api, Variant index);
	virtual Ref<LuaError> __newindex(Object *obj, LuaAPI*api, Variant index, Variant value);
	virtual Variant __call(Object *obj, LuaAPI*api, Ref<LuaTuple> args);
	virtual Ref<LuaError> __gc(Object *obj, LuaAPI*api);
	virtual String __tostring(Object *obj, LuaAPI*api);
	virtual int __len(Object *obj, LuaAPI*api);
	virtual Variant __unm(Object *obj, LuaAPI*api);
	virtual Variant __add(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __sub(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __mul(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __div(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __idiv(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __mod(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __pow(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __band(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __bor(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __bxor(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __bnot(Object *obj, LuaAPI*api);
	virtual Variant __shl(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __shr(Object *obj, LuaAPI*api, Variant other);
	virtual Variant __concat(Object *obj, LuaAPI*api, Variant other);
	virtual bool __eq(Object *obj, LuaAPI*api, Variant other);
	virtual bool __lt(Object *obj, LuaAPI*api, Variant other);
	virtual bool __le(Object *obj, LuaAPI*api, Variant other);

private:
};

// Default object metatable

class LuaDefaultObjectMetatable : public LuaObjectMetatable {
	GDCLASS(LuaDefaultObjectMetatable, LuaObjectMetatable);

protected:
	static void _bind_methods();

public:
	Variant __index(Object *obj, LuaAPI*api, Variant index) override;
	Ref<LuaError> __newindex(Object *obj, LuaAPI*api, Variant index, Variant value) override;
	Variant __call(Object *obj, LuaAPI*api, Ref<LuaTuple> args) override;
	Ref<LuaError> __gc(Object *obj, LuaAPI*api) override;
	String __tostring(Object *obj, LuaAPI*api) override;
	int __len(Object *obj, LuaAPI*api) override;
	Variant __unm(Object *obj, LuaAPI*api) override;
	Variant __add(Object *obj, LuaAPI*api, Variant other) override;
	Variant __sub(Object *obj, LuaAPI*api, Variant other) override;
	Variant __mul(Object *obj, LuaAPI*api, Variant other) override;
	Variant __div(Object *obj, LuaAPI*api, Variant other) override;
	Variant __idiv(Object *obj, LuaAPI*api, Variant other) override;
	Variant __mod(Object *obj, LuaAPI*api, Variant other) override;
	Variant __pow(Object *obj, LuaAPI*api, Variant other) override;
	Variant __band(Object *obj, LuaAPI*api, Variant other) override;
	Variant __bor(Object *obj, LuaAPI*api, Variant other) override;
	Variant __bxor(Object *obj, LuaAPI*api, Variant other) override;
	Variant __bnot(Object *obj, LuaAPI*api) override;
	Variant __shl(Object *obj, LuaAPI*api, Variant other) override;
	Variant __shr(Object *obj, LuaAPI*api, Variant other) override;
	Variant __concat(Object *obj, LuaAPI*api, Variant other) override;
	bool __eq(Object *obj, LuaAPI*api, Variant other) override;
	bool __lt(Object *obj, LuaAPI*api, Variant other) override;
	bool __le(Object *obj, LuaAPI*api, Variant other) override;

	void setPermissive(bool permissive);
	bool getPermissive() const;

private:
	bool permissive = true;
};

#endif
