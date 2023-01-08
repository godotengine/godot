#ifndef RTSCOM_META_H
#define RTSCOM_META_H

// #include <string>

#include "servers/physics_server.h"
#include "core/reference.h"
#include "core/string_name.h"
#include "core/script_language.h"
#include "core/print_string.h"
#include "core/list.h"
#include "rcs_types.h"
// #include "combat_server.h"

#define MAX_ALLOCATABLE_WORKER_PER_OBJECT 4

//#define USE_SAFE_RID_COUNT
#ifdef  USE_SAFE_RID_COUNT
#define RID_TYPE RIR
#else
#define RID_TYPE RID
#endif

// #endif


// #define USE_STL_WRAPPER
// #define USE_STL_VECTOR
// #define STL_WRAPPER_SAME_API

#if defined(STL_WRAPPER_SAME_API) && defined(USE_STL_WRAPPER)
#define VECTOR WrappedVector
#define VEC_FIND(vec, elem, i)                                             \
	auto iter = std::find(vec.begin(), vec.end(), elem);                   \
	i = (iter == vec.end() ? -1 : iter - vec.begin());
#define VEC_HAS(vec, elem){                                                \
	int i = 0;                                                             \
	VEC_FIND(vec, elem, i);                                                \
	return i != -1;                                                        \
}
#define VEC_ERASE(vec, elem)                                               \
{                                                                          \
	auto __size = vec.size();                                              \
	for (unsigned int __i = 0; __i < __size; __i++){                       \
		if (vec[__i] == elem){                                             \
			auto iter = vec.begin() + __i;                                 \
			vec.erase(iter);                                               \
			break;                                                         \
		}                                                                  \
	}                                                                      \
}
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(const VECTOR<T>& vec, const T& elem) VEC_HAS(vec, elem)
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(VECTOR<T>* vec, const T& elem) {
	auto iterator = std::find(vec->begin(), vec->end(), elem);
	return (iter == vec->end() ? -1, iter - vec->begin());
}
#elif defined(USE_STL_WRAPPER) && !defined(STL_WRAPPER_SAME_API) && !defined(USE_STL_VECTOR)
#define VECTOR WrappedVector
#define VEC_FIND(vec, elem, i) i = vec.find(elem)
#define VEC_HAS(vec, elem) { return (vec.find(elem) != -1); }
#define VEC_ERASE(vec, elem) { vec.erase(elem); }
#define VEC_REMOVE(vec, idx) vec.remove(idx);
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(const VECTOR<T>& vec, const T& elem) VEC_HAS(vec, elem)
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(VECTOR<T>* vec, const T& elem) {
	return vec->find(elem) != -1;
}
#elif defined(USE_STL_VECTOR) && !defined(USE_STL_WRAPPER)
#include <vector>
#define VECTOR std::vector
#define VEC_FIND(vec, elem, i)                                             \
	auto iter = std::find(vec.begin(), vec.end(), elem);                   \
	i = (iter == vec.end() ? -1 : iter - vec.begin());
#define VEC_HAS(vec, elem){                                                \
	int i = 0;                                                             \
	VEC_FIND(vec, elem, i);                                                \
	return i != -1;                                                        \
}
#define VEC_ERASE(vec, elem)                                               \
{                                                                          \
	auto __size = vec.size();                                              \
	for (unsigned int __i = 0; __i < __size; __i++){                       \
		if (vec[__i] == elem){                                             \
			auto iter = vec.begin() + __i;                                 \
			vec.erase(iter);                                               \
			break;                                                         \
		}                                                                  \
	}                                                                      \
}
#define VEC_REMOVE(vec, idx) vec.erase(vec.begin() + idx)
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(const VECTOR<T>& vec, const T& elem) VEC_HAS(vec, elem)
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(VECTOR<T>* vec, const T& elem) {
	auto iterator = std::find(vec->begin(), vec->end(), elem);
	return (iter == vec->end() ? -1, iter - vec->begin());
}
#else
#include "core/vector.h"
#define VECTOR Vector
#define VEC_FIND(vec, elem, i)                                             \
	i = vec.find(elem);
#define VEC_HAS(vec, elem)                                                 \
	{ return vec.find(elem) != -1; }
#define VEC_ERASE(vec, elem)                                               \
	{ vec.erase(elem); }
#define VEC_REMOVE(vec, idx) vec.remove(idx)
#endif

#define VEC2GDARRAY(vec, arr) {                                            \
	for (uint32_t __i = 0, __size = vec.size(); __i < __size; __i++){      \
		arr.push_back(vec[__i]);                                           \
	}                                                                      \
}

#define VEC_TRANSFER(from, to) {                                           \
	for (uint32_t __i = 0, __size = from.size(); __i < __size; __i++){     \
		to.push_back(from[i]);                                             \
	}                                                                      \
}
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(const VECTOR<T>& vec, const T& elem) VEC_HAS(vec, elem)
template <typename T> static _ALWAYS_INLINE_ bool VectorHas(const VECTOR<T>* vec, const T& elem) {
	return vec->find(elem) != -1;
}
#ifdef USE_SAFE_RID_COUNT
#define rcsnew(classptr) std::shared_ptr<classptr>()
#define rcsdel(ptr) { ptr.reset(); }
#else
#define rcsnew(classptr) new classptr
#define rcsdel(ptr) { delete ptr; ptr = nullptr; }
#endif

enum CombatantStatus {
	UNINITIALIZED,
	STANDBY,
	ACTIVE,
	DESTROYED,
};

class RID_RCS;
class RCS_OwnerBase;
// class RCSCompatPassthrough;

#define RIR uint32_t
// typedef uint32_t RIR;

#define PUSH_RECORD_PRIMITIVE(m_record, m_val) \
	m_record->table.push_back(Pair<StringName, Variant>(StringName(#m_val), m_val))


#include "modules/record/record.h"
#ifdef USE_SAFE_RID_COUNT
class RID_RCS {
protected:
	RCS_OwnerBase *_owner;
	RIR _id;
	Ref<RCSChip> chip;
	RID_TYPE self;
	Sentrience *combat_server;
	std::weak_ptr<RID_RCS> self_ref;
	friend class RCS_OwnerBase;

public:
	virtual Ref<RawRecord> serialize() const { return Ref<RawRecord>(); }
	virtual bool serialize(const Ref<RawRecord> &from) { return false; }
	virtual void deserialize(const RawRecord &rec) {}

	friend class Sentrience;

	_FORCE_INLINE_ void set_self(const RID_TYPE &p_self) { self = p_self; }
	_FORCE_INLINE_ RID_TYPE get_self() const { return self; }

	_FORCE_INLINE_ void set_self_ref(const std::weak_ptr<RID_RCS> &p_self) { self_ref = p_self; }
	_FORCE_INLINE_ std::weak_ptr<RID_RCS> get_self_ref() const { return self_ref; }

	_FORCE_INLINE_ void _set_combat_server(Sentrience *p_combatServer) { combat_server = p_combatServer; }
	_FORCE_INLINE_ Sentrience *_get_combat_server() const { return combat_server; }

	virtual void capture_event(RID_RCS *from, void *event = nullptr) {}

	virtual void poll(const float &delta) {
		if (chip.is_valid())
			chip->callback(delta);
	}

	virtual void set_chip(const Ref<RCSChip> &new_chip) {
		chip = new_chip;
		if (chip.is_valid())
			chip->set_host(self);
	}
	virtual Ref<RCSChip> get_chip() const { return chip; }
};
#else

class Sentrience;

class RCSChip : public Reference{
	GDCLASS(RCSChip, Reference);
private:
	RID_TYPE host;

	void callback(const float& delta);
protected:
	virtual void internal_callback(const float& delta) {};
	virtual void internal_init() {};
	static void _bind_methods();
public:
	RCSChip();
	~RCSChip();

	friend class RID_RCS;

	void set_host(const RID_TYPE& r_host);
	_FORCE_INLINE_ RID_TYPE get_host() const { return host; }
};


class RID_RCS : public RID_Data {
protected:
	Ref<RCSChip> chip;
	RID_TYPE self;
	Sentrience *combat_server;

public:
	virtual Ref<RawRecord> serialize() const { return Ref<RawRecord>(); }
	virtual bool serialize(const Ref<RawRecord> &from) { return false; }
	virtual void deserialize(const RawRecord &rec) {}

	friend class Sentrience;

	_FORCE_INLINE_ void set_self(const RID_TYPE &p_self) { self = p_self; }
	_FORCE_INLINE_ RID_TYPE get_self() const noexcept { return self; }

	_FORCE_INLINE_ void _set_combat_server(Sentrience *p_combatServer) { combat_server = p_combatServer; }
	_FORCE_INLINE_ Sentrience *_get_combat_server() const { return combat_server; }

	virtual void capture_event(RID_RCS *from, void *event = nullptr) {}

	virtual void poll(const float &delta) {
		if (chip.is_valid())
			chip->callback(delta);
	}

	virtual void set_chip(const Ref<RCSChip> &new_chip) {
		chip = new_chip;
		if (chip.is_valid())
			chip->set_host(self);
	}
	virtual Ref<RCSChip> get_chip() const { return chip; }
};
#endif


#ifdef USE_SAFE_RID_COUNT
class RCS_OwnerBase {
	SafeRefCount ref_counter;

public:
	RCS_OwnerBase() = default;
	_FORCE_INLINE_ void _set_data(RID_TYPE& rid, RID_RCS* ref) {
		rid = ref_counter.refval();
		ref->_id = rid;
		ref->_owner = this;
	}
	_FORCE_INLINE_ bool _is_owner(const RID_RCS* ref) const {
		return ref->_owner == this;
	}
};
template <class T>
class RCS_Owner : public RCS_OwnerBase {
public:
	using rcs_reference = std::shared_ptr<T>;
private:
	HashMap<RIR, std::shared_ptr<T>> ownership_record;
public:
	_FORCE_INLINE_ RCS_Owner() {

	}
	_FORCE_INLINE_ RIR make_rid(std::shared_ptr<T>& ref){
		RIR rir;
		_set_data(rir, ref.get());
		ownership_record.operator[](rir) = ref;
	}
	_FORCE_INLINE_ bool owns(const RIR& rir){
		auto first_check = ownership_record.has(rir);
		if (!first_check)
			return false;
		auto obj = ownership_record.operator[](rir);
		return _is_owner(obj.get());
	}
	_FORCE_INLINE_ std::weak_ptr<T> get(const RIR& rir) const {
		if (!owns(rir))
			return std::weak_ptr<T>(std::shared_ptr<T>(nullptr));
		return std::weak_ptr<T>(ownership_record.operator[](rir));
	}
	_FORCE_INLINE_ std::shared_ptr<T> get_locked(const RIR& rir) const {
		if (!owns(rir)) return std::shared_ptr<T>(nullptr);
		return std::shared_ptr<T>(ownership_record.operator[](rir));
	}
	_FORCE_INLINE_ void free(const RIR& rir){
		if (!has(rir))
			return;
		auto stuff = ownership_record.operator[](rir);
		stuff->_owner = nullptr;
		ref->_id = 0;
		ownership_record.erase(rir);
	}
	_FORCE_INLINE_ void get_owned_list(List<RID> *p_owned){
		ownership_record.get_key_list(p_owned);
	}
};
#endif

// static _ALWAYS_INLINE_ float F_rsqrt(const float& number){
// 	const float threehalfs = 1.5F;

// 	float x2 = number * 0.5F;
// 	float y = number;

// 	long i = * ( long * ) &y;

// 	i = 0x5f3759df - ( i >> 1 );
// 	y = * ( float * ) &i;

// 	y = y * ( threehalfs - ( x2 * y * y ) );
// 	// y = y * ( threehalfs - ( x2 * y * y ) );

// 	return y;
// }

// static _ALWAYS_INLINE_ float F_sqrt(const float& number){
// 	return 1 / F_rsqrt(number);
// }

#endif
