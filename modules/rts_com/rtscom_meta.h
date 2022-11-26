#ifndef RTSCOM_META_H
#define RTSCOM_META_H

#define USE_STL_VECTOR

#if defined USE_STL_VECTOR
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
#else
#include "core/vector.h"
#define VECTOR Vector
#define VEC_FIND(vec, elem, i)                                             \
	i = vec.find(elem);
#define VEC_ERASE(vec, elem)                                               \
	vec.erase(elem);
#endif

#define VEC2GDARRAY(vec, arr) {                                            \
	for (uint32_t __i = 0, __size = vec.size(); __i < __size; __i++){      \
		arr.push_back(vec[__i]);                                           \
	}                                                                      \
}

#include "servers/physics_server.h"
#include "core/reference.h"
#include "core/string_name.h"
#include "core/script_language.h"

#define rcsnew(classptr) new classptr
#define rcsdel(ptr) { delete ptr; ptr = nullptr; }
// #define rcsnew(ptr) memnew(ptr)
// #define rcsdel(ptr) memdelete(ptr);

enum CombatantStatus {
	UNINITIALIZED,
	STANDBY,
	ACTIVE,
	DESTROYED,
};

class RID_RCS;

class RCSChip : public Reference{
	GDCLASS(RCSChip, Reference);
private:
	RID host;

	void callback(const float& delta);
protected:
	virtual void internal_callback(const float& delta) {};
	virtual void internal_init() {};
	static void _bind_methods();
public:
	RCSChip();
	~RCSChip();

	friend class RID_RCS;

	void set_host(const RID& r_host);
	_FORCE_INLINE_ RID get_host() const { return host; }
};

#endif
