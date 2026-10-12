/**************************************************************************/
/*  as_typed_containers.cpp                                               */
/**************************************************************************/
#include "as_typed_containers.h"
#include "as_binding_object.h"

#include "core/object/object.h"
#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/safe_refcount.h"
#include "core/templates/vector.h"
#include "core/variant/variant.h"

#include <angelscript.h>

namespace {

static HashSet<asIScriptEngine *> registered_engines;
static SafeNumeric<int64_t> live_instances;

static void script_exception(const char *p_message) {
	asIScriptContext *ctx = asGetActiveContext();
	if (ctx != nullptr) ctx->SetException(p_message);
}

template <typename Storage>
class RefContainer {
	SafeRefCount refs;

public:
	Storage values;
	RefContainer() { refs.init(); live_instances.increment(); }
	virtual ~RefContainer() { live_instances.decrement(); }
	void add_ref() { refs.ref(); }
	void release() { if (refs.unref()) memdelete(this); }
};

template <typename T> class ArrayContainer : public RefContainer<Vector<T>> {};
using Int64Array = ArrayContainer<int64_t>;
using Float64Array = ArrayContainer<double>;
using Vector2Array = ArrayContainer<Vector2>;
using Vector3Array = ArrayContainer<Vector3>;
using ColorArray = ArrayContainer<Color>;
using StringNameArray = ArrayContainer<StringName>;
using ObjectArray = ArrayContainer<ObjectID>;

template <typename K, typename V> class MapContainer : public RefContainer<HashMap<K, V>> {};
using Int64Int64Map = MapContainer<int64_t, int64_t>;
using StringNameInt64Map = MapContainer<StringName, int64_t>;
using StringNameFloat64Map = MapContainer<StringName, double>;
using StringNameVector2Map = MapContainer<StringName, Vector2>;
using StringNameColorMap = MapContainer<StringName, Color>;
using StringNameObjectMap = MapContainer<StringName, ObjectID>;

template <typename C> static C *factory() { return memnew(C); }
template <typename C> static void add_ref(C *self) { self->add_ref(); }
template <typename C> static void release(C *self) { self->release(); }
template <typename C> static int64_t size(const C *self) { return self->values.size(); }
template <typename C> static bool is_empty(const C *self) { return self->values.is_empty(); }
template <typename C> static void clear(C *self) { self->values.clear(); }

template <typename C> static void array_reserve(C *self, int64_t amount) {
	if (amount < 0 || amount > INT32_MAX) { script_exception("invalid array capacity"); return; }
	self->values.reserve((int)amount);
}
template <typename C> static void array_resize(C *self, int64_t amount) {
	if (amount < 0 || amount > INT32_MAX) { script_exception("invalid array size"); return; }
	self->values.resize((int)amount);
}
template <typename C, typename T> static void array_push(C *self, T value) { self->values.push_back(value); }
template <typename C, typename T> static void array_set(C *self, int64_t index, T value) {
	if (index < 0 || index >= self->values.size()) { script_exception("index out of bounds"); return; }
	self->values.write[(int)index] = value;
}
template <typename C, typename T> static T array_get(const C *self, int64_t index) {
	if (index < 0 || index >= self->values.size()) { script_exception("index out of bounds"); return T(); }
	return self->values[(int)index];
}
template <typename C, typename T> static void array_fill(C *self, T value) { self->values.fill(value); }
template <typename C> static void array_append(C *self, const C *other) { self->values.append_array(other->values); }

template <typename C, typename T> static T array_sum(const C *self) {
	T result{};
	for (const T &v : self->values) result += v;
	return result;
}
template <typename C, typename T> static T array_min(const C *self) {
	if (self->values.is_empty()) { script_exception("min_value on empty array"); return T(); }
	T result = self->values[0]; for (const T &v : self->values) if (v < result) result = v; return result;
}
template <typename C, typename T> static T array_max(const C *self) {
	if (self->values.is_empty()) { script_exception("max_value on empty array"); return T(); }
	T result = self->values[0]; for (const T &v : self->values) if (result < v) result = v; return result;
}
template <typename C, typename T> static void array_add_scalar(C *self, T value) { for (T &v : self->values) v += value; }
template <typename C, typename T> static void array_mul_scalar(C *self, T value) { for (T &v : self->values) v *= value; }
template <typename C, typename T> static void array_add_in_place(C *self, const C *other) {
	if (self->values.size() != other->values.size()) { script_exception("array sizes differ"); return; }
	for (int i = 0; i < self->values.size(); i++) self->values.write[i] += other->values[i];
}
template <typename C, typename T> static void array_add_scaled(C *self, const C *other, double scale) {
	if (self->values.size() != other->values.size()) { script_exception("array sizes differ"); return; }
	for (int i = 0; i < self->values.size(); i++) self->values.write[i] += other->values[i] * (real_t)scale;
}

// asCALL_CDECL_OBJFIRST 对大值返回使用隐藏返回槽，沿用 native value ops 的 ABI 形状。
template <typename C, typename T> static void array_get_obj(T *ret, const C *self, int64_t index) { new (ret) T(array_get<C, T>(self, index)); }
template <typename C, typename T> static void array_sum_obj(T *ret, const C *self) { new (ret) T(array_sum<C, T>(self)); }
template <typename C, typename T> static void array_push_obj(C *self, const T *value) { array_push<C, T>(self, *value); }
template <typename C, typename T> static void array_set_obj(C *self, int64_t index, const T *value) { array_set<C, T>(self, index, *value); }
template <typename C, typename T> static void array_fill_obj(C *self, const T *value) { array_fill<C, T>(self, *value); }

static bool string_name_has(const StringNameArray *self, const Variant *value) {
	const StringName name = *value; for (const StringName &v : self->values) if (v == name) return true; return false;
}
static int64_t string_name_find(const StringNameArray *self, const Variant *value) {
	const StringName name = *value; for (int i = 0; i < self->values.size(); i++) if (self->values[i] == name) return i; return -1;
}
static void string_name_push(StringNameArray *self, const Variant *value) { self->values.push_back((StringName)*value); }
static void string_name_set(StringNameArray *self, int64_t index, const Variant *value) { array_set<StringNameArray, StringName>(self, index, (StringName)*value); }
static void string_name_get(Variant *ret, const StringNameArray *self, int64_t index) { new (ret) Variant(array_get<StringNameArray, StringName>(self, index)); }

static void object_push(ObjectArray *self, void *value_slot) { Object *value=as_handle_decode(value_slot,AS_KIND_OBJECT_NONOWNING); self->values.push_back(value ? value->get_instance_id() : ObjectID()); }
static void *object_get(const ObjectArray *self, int64_t index) {
	const ObjectID id = array_get<ObjectArray, ObjectID>(self, index); return as_handle_encode(id.is_null()?nullptr:ObjectDB::get_instance(id),AS_KIND_OBJECT_NONOWNING);
}
static int64_t object_remove_invalid(ObjectArray *self) {
	int64_t removed = 0;
	for (int i = self->values.size() - 1; i >= 0; i--) if (self->values[i].is_null() || ObjectDB::get_instance(self->values[i]) == nullptr) { self->values.remove_at(i); removed++; }
	return removed;
}

#define DEFINE_PACKED_CONVERSION(PREFIX, C, PACKED, TYPE, VTYPE) \
static void PREFIX##_assign(C *self, const Variant *boxed) { \
	if (boxed->get_type() != VTYPE) { script_exception("packed array type mismatch"); return; } \
	PACKED source = *boxed; self->values.resize(source.size()); for (int i=0;i<source.size();i++) self->values.write[i]=source[i]; \
} \
static void PREFIX##_to(Variant *ret, const C *self) { \
	PACKED out; out.resize(self->values.size()); for (int i=0;i<self->values.size();i++) out.set(i,self->values[i]); new(ret) Variant(out); \
}
DEFINE_PACKED_CONVERSION(i64_packed, Int64Array, PackedInt64Array, int64_t, Variant::PACKED_INT64_ARRAY)
DEFINE_PACKED_CONVERSION(f64_packed, Float64Array, PackedFloat64Array, double, Variant::PACKED_FLOAT64_ARRAY)
DEFINE_PACKED_CONVERSION(v2_packed, Vector2Array, PackedVector2Array, Vector2, Variant::PACKED_VECTOR2_ARRAY)
DEFINE_PACKED_CONVERSION(v3_packed, Vector3Array, PackedVector3Array, Vector3, Variant::PACKED_VECTOR3_ARRAY)
DEFINE_PACKED_CONVERSION(color_packed, ColorArray, PackedColorArray, Color, Variant::PACKED_COLOR_ARRAY)
#undef DEFINE_PACKED_CONVERSION

static void string_name_assign_array(StringNameArray *self, const Variant *boxed) {
	if (boxed->get_type() != Variant::ARRAY) { script_exception("expected Array"); return; }
	Array source = *boxed; Vector<StringName> out; out.resize(source.size());
	for (int i=0;i<source.size();i++) { if (source[i].get_type()!=Variant::STRING_NAME && source[i].get_type()!=Variant::STRING) { script_exception("Array element type mismatch"); return; } out.write[i]=(StringName)source[i]; }
	self->values = out;
}
static void string_name_to_array(Variant *ret, const StringNameArray *self) { Array out; out.resize(self->values.size()); for(int i=0;i<self->values.size();i++) out[i]=self->values[i]; new(ret) Variant(out); }
static void object_assign_array(ObjectArray *self, const Variant *boxed) {
	if (boxed->get_type()!=Variant::ARRAY) { script_exception("expected Array"); return; } Array source=*boxed; Vector<ObjectID> out; out.resize(source.size());
	for(int i=0;i<source.size();i++){ if(source[i].get_type()!=Variant::OBJECT){script_exception("Array element type mismatch");return;} Object *obj=source[i]; out.write[i]=obj?obj->get_instance_id():ObjectID(); } self->values=out;
}
static void object_to_array(Variant *ret,const ObjectArray *self){Array out;out.resize(self->values.size());for(int i=0;i<self->values.size();i++)out[i]=self->values[i].is_null()?Variant():Variant(ObjectDB::get_instance(self->values[i]));new(ret)Variant(out);}

template <typename C, typename K> static bool map_has(const C *self, K key) { return self->values.has(key); }
template <typename C, typename K> static bool map_erase(C *self, K key) { return self->values.erase(key); }
template <typename C, typename K, typename V> static void map_set(C *self, K key, V value) { self->values.insert(key, value); }
template <typename C, typename K, typename V> static V map_get_or(const C *self, K key, V fallback) { const V *v = self->values.getptr(key); return v ? *v : fallback; }

static StringName key_from_variant(const Variant *key) { return (StringName)*key; }
template <typename C, typename V> static bool sn_map_has(const C *self, const Variant *key) { return self->values.has(key_from_variant(key)); }
template <typename C, typename V> static bool sn_map_erase(C *self, const Variant *key) { return self->values.erase(key_from_variant(key)); }
template <typename C, typename V> static void sn_map_set(C *self, const Variant *key, V value) { self->values.insert(key_from_variant(key), value); }
template <typename C, typename V> static V sn_map_get_or(const C *self, const Variant *key, V fallback) { const V *v = self->values.getptr(key_from_variant(key)); return v ? *v : fallback; }
template <typename C, typename V> static void sn_map_set_obj(C *self, const Variant *key, const V *value) { self->values.insert(key_from_variant(key), *value); }
template <typename C, typename V> static void sn_map_get_obj(V *ret, const C *self, const Variant *key, const V *fallback) { const V *v = self->values.getptr(key_from_variant(key)); new (ret) V(v ? *v : *fallback); }

static void sn_object_set(StringNameObjectMap *self, const Variant *key, void *value_slot) { Object *value=as_handle_decode(value_slot,AS_KIND_OBJECT_NONOWNING); self->values.insert(key_from_variant(key), value ? value->get_instance_id() : ObjectID()); }
static void *sn_object_get(const StringNameObjectMap *self, const Variant *key, void *fallback_slot) {
	Object *fallback=as_handle_decode(fallback_slot,AS_KIND_OBJECT_NONOWNING); const ObjectID *id = self->values.getptr(key_from_variant(key)); if (!id || id->is_null()) return as_handle_encode(fallback,AS_KIND_OBJECT_NONOWNING); Object *obj = ObjectDB::get_instance(*id); return as_handle_encode(obj?obj:fallback,AS_KIND_OBJECT_NONOWNING);
}
static int64_t sn_object_remove_invalid(StringNameObjectMap *self) {
	Vector<StringName> dead; for (const KeyValue<StringName, ObjectID> &e : self->values) if (e.value.is_null() || ObjectDB::get_instance(e.value) == nullptr) dead.push_back(e.key);
	for (const StringName &key : dead) self->values.erase(key); return dead.size();
}

static bool reg(asIScriptEngine *e, int result) { return result >= 0; }
template <typename C> static bool register_ref(asIScriptEngine *e, const char *name) {
	String factory_decl = String(name) + "@ f()";
	return reg(e, e->RegisterObjectType(name, 0, asOBJ_REF)) &&
		reg(e, e->RegisterObjectBehaviour(name, asBEHAVE_FACTORY, factory_decl.utf8().get_data(), asFUNCTION((factory<C>)), asCALL_CDECL)) &&
		reg(e, e->RegisterObjectBehaviour(name, asBEHAVE_ADDREF, "void f()", asFUNCTION((add_ref<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectBehaviour(name, asBEHAVE_RELEASE, "void f()", asFUNCTION((release<C>)), asCALL_CDECL_OBJFIRST));
}
template <typename C> static bool register_common_array(asIScriptEngine *e, const char *name) {
	String n(name);
	return register_ref<C>(e, name) &&
		reg(e, e->RegisterObjectMethod(name, "int64 size() const", asFUNCTION((size<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, "bool is_empty() const", asFUNCTION((is_empty<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, "void reserve(int64)", asFUNCTION((array_reserve<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, "void resize(int64)", asFUNCTION((array_resize<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, "void clear()", asFUNCTION((clear<C>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void append_array(const " + n + "@)").utf8().get_data(), asFUNCTION((array_append<C>)), asCALL_CDECL_OBJFIRST));
}

template <typename C, typename T> static bool register_scalar_array(asIScriptEngine *e, const char *name, const char *type) {
	String t(type);
	return register_common_array<C>(e, name) &&
		reg(e, e->RegisterObjectMethod(name, ("void push_back(" + t + ")").utf8().get_data(), asFUNCTION((array_push<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " get(int64) const").utf8().get_data(), asFUNCTION((array_get<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void set(int64," + t + ")").utf8().get_data(), asFUNCTION((array_set<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void fill(" + t + ")").utf8().get_data(), asFUNCTION((array_fill<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " sum() const").utf8().get_data(), asFUNCTION((array_sum<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " min_value() const").utf8().get_data(), asFUNCTION((array_min<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " max_value() const").utf8().get_data(), asFUNCTION((array_max<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void add_scalar(" + t + ")").utf8().get_data(), asFUNCTION((array_add_scalar<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void multiply_scalar(" + t + ")").utf8().get_data(), asFUNCTION((array_mul_scalar<C,T>)), asCALL_CDECL_OBJFIRST));
}
template <typename C, typename T> static bool register_obj_array(asIScriptEngine *e, const char *name, const char *type, bool scaled) {
	String t(type), n(name);
	bool ok = register_common_array<C>(e, name) &&
		reg(e, e->RegisterObjectMethod(name, ("void push_back(const " + t + " &in)").utf8().get_data(), asFUNCTION((array_push_obj<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " get(int64) const").utf8().get_data(), asFUNCTION((array_get_obj<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void set(int64,const " + t + " &in)").utf8().get_data(), asFUNCTION((array_set_obj<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void fill(const " + t + " &in)").utf8().get_data(), asFUNCTION((array_fill_obj<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, (t + " sum() const").utf8().get_data(), asFUNCTION((array_sum_obj<C,T>)), asCALL_CDECL_OBJFIRST)) &&
		reg(e, e->RegisterObjectMethod(name, ("void add_in_place(const " + n + "@)").utf8().get_data(), asFUNCTION((array_add_in_place<C,T>)), asCALL_CDECL_OBJFIRST));
	if (scaled) ok = ok && reg(e, e->RegisterObjectMethod(name, ("void add_scaled_in_place(const " + n + "@, double)").utf8().get_data(), asFUNCTION((array_add_scaled<C,T>)), asCALL_CDECL_OBJFIRST));
	return ok;
}

template <typename C> static bool register_common_map(asIScriptEngine *e, const char *name) {
	return register_ref<C>(e,name) && reg(e,e->RegisterObjectMethod(name,"int64 size() const",asFUNCTION((size<C>)),asCALL_CDECL_OBJFIRST)) && reg(e,e->RegisterObjectMethod(name,"bool is_empty() const",asFUNCTION((is_empty<C>)),asCALL_CDECL_OBJFIRST)) && reg(e,e->RegisterObjectMethod(name,"void clear()",asFUNCTION((clear<C>)),asCALL_CDECL_OBJFIRST));
}

} // namespace

Error ASTypedContainers::register_all(asIScriptEngine *e) {
	ERR_FAIL_NULL_V(e, ERR_INVALID_PARAMETER);
	if (registered_engines.has(e)) return OK;
	registered_engines.insert(e);
	bool ok = true;
	ok = ok && register_scalar_array<Int64Array,int64_t>(e,"Int64Array","int64");
	ok = ok && register_scalar_array<Float64Array,double>(e,"Float64Array","double");
	ok = ok && register_obj_array<Vector2Array,Vector2>(e,"Vector2Array","Vector2",true);
	ok = ok && register_obj_array<Vector3Array,Vector3>(e,"Vector3Array","Vector3",true);
	ok = ok && register_obj_array<ColorArray,Color>(e,"ColorArray","Color",false);
#define REG_PACKED(C,N,PREFIX,PACKEDNAME) do { ok=ok&&reg(e,e->RegisterObjectMethod(N,"void assign_packed(const " PACKEDNAME " &in)",asFUNCTION(PREFIX##_assign),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,PACKEDNAME " to_packed() const",asFUNCTION(PREFIX##_to),asCALL_CDECL_OBJFIRST)); } while(false)
	REG_PACKED(Int64Array,"Int64Array",i64_packed,"PackedInt64Array");
	REG_PACKED(Float64Array,"Float64Array",f64_packed,"PackedFloat64Array");
	REG_PACKED(Vector2Array,"Vector2Array",v2_packed,"PackedVector2Array");
	REG_PACKED(Vector3Array,"Vector3Array",v3_packed,"PackedVector3Array");
	REG_PACKED(ColorArray,"ColorArray",color_packed,"PackedColorArray");
#undef REG_PACKED
	ok = ok && register_common_array<StringNameArray>(e,"StringNameArray");
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","void push_back(const StringName &in)",asFUNCTION(string_name_push),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","void set(int64,const StringName &in)",asFUNCTION(string_name_set),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","StringName get(int64) const",asFUNCTION(string_name_get),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","bool has(const StringName &in) const",asFUNCTION(string_name_has),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","int64 find(const StringName &in) const",asFUNCTION(string_name_find),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","void assign_array(const Array &in)",asFUNCTION(string_name_assign_array),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("StringNameArray","Array to_array() const",asFUNCTION(string_name_to_array),asCALL_CDECL_OBJFIRST));
	ok = ok && register_common_array<ObjectArray>(e,"ObjectArray");
	ok = ok && reg(e,e->RegisterObjectMethod("ObjectArray","void push_back(Object@)",asFUNCTION(object_push),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("ObjectArray","Object@ get(int64) const",asFUNCTION(object_get),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("ObjectArray","int64 remove_invalid()",asFUNCTION(object_remove_invalid),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("ObjectArray","void assign_array(const Array &in)",asFUNCTION(object_assign_array),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("ObjectArray","Array to_array() const",asFUNCTION(object_to_array),asCALL_CDECL_OBJFIRST));

	ok = ok && register_common_map<Int64Int64Map>(e,"Int64Int64Map");
	ok = ok && reg(e,e->RegisterObjectMethod("Int64Int64Map","bool has(int64) const",asFUNCTION((map_has<Int64Int64Map,int64_t>)),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("Int64Int64Map","bool erase(int64)",asFUNCTION((map_erase<Int64Int64Map,int64_t>)),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("Int64Int64Map","void set(int64,int64)",asFUNCTION((map_set<Int64Int64Map,int64_t,int64_t>)),asCALL_CDECL_OBJFIRST));
	ok = ok && reg(e,e->RegisterObjectMethod("Int64Int64Map","int64 get_or(int64,int64) const",asFUNCTION((map_get_or<Int64Int64Map,int64_t,int64_t>)),asCALL_CDECL_OBJFIRST));

#define REG_SN_SCALAR(C,N,T,ASNAME) do { ok=ok&&register_common_map<C>(e,N); ok=ok&&reg(e,e->RegisterObjectMethod(N,"bool has(const StringName &in) const",asFUNCTION((sn_map_has<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,"bool erase(const StringName &in)",asFUNCTION((sn_map_erase<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,"void set(const StringName &in," ASNAME ")",asFUNCTION((sn_map_set<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,ASNAME " get_or(const StringName &in," ASNAME ") const",asFUNCTION((sn_map_get_or<C,T>)),asCALL_CDECL_OBJFIRST)); } while(false)
	REG_SN_SCALAR(StringNameInt64Map,"StringNameInt64Map",int64_t,"int64");
	REG_SN_SCALAR(StringNameFloat64Map,"StringNameFloat64Map",double,"double");
#undef REG_SN_SCALAR
#define REG_SN_OBJ(C,N,T,ASNAME) do { ok=ok&&register_common_map<C>(e,N); ok=ok&&reg(e,e->RegisterObjectMethod(N,"bool has(const StringName &in) const",asFUNCTION((sn_map_has<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,"bool erase(const StringName &in)",asFUNCTION((sn_map_erase<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,"void set(const StringName &in,const " ASNAME " &in)",asFUNCTION((sn_map_set_obj<C,T>)),asCALL_CDECL_OBJFIRST)); ok=ok&&reg(e,e->RegisterObjectMethod(N,ASNAME " get_or(const StringName &in,const " ASNAME " &in) const",asFUNCTION((sn_map_get_obj<C,T>)),asCALL_CDECL_OBJFIRST)); } while(false)
	REG_SN_OBJ(StringNameVector2Map,"StringNameVector2Map",Vector2,"Vector2");
	REG_SN_OBJ(StringNameColorMap,"StringNameColorMap",Color,"Color");
#undef REG_SN_OBJ
	ok=ok&&register_common_map<StringNameObjectMap>(e,"StringNameObjectMap");
	ok=ok&&reg(e,e->RegisterObjectMethod("StringNameObjectMap","void set(const StringName &in,Object@)",asFUNCTION(sn_object_set),asCALL_CDECL_OBJFIRST));
	ok=ok&&reg(e,e->RegisterObjectMethod("StringNameObjectMap","Object@ get_or(const StringName &in,Object@) const",asFUNCTION(sn_object_get),asCALL_CDECL_OBJFIRST));
	ok=ok&&reg(e,e->RegisterObjectMethod("StringNameObjectMap","int64 remove_invalid()",asFUNCTION(sn_object_remove_invalid),asCALL_CDECL_OBJFIRST));
	return ok ? OK : ERR_CANT_CREATE;
}

int64_t ASTypedContainers::live_instance_count() { return live_instances.get(); }
