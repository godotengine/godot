#pragma once
#include <cstdint>
#include "variant.hpp"
struct Variant;

struct RID {
	int64_t index;

	constexpr RID() : index(0) {}
	constexpr RID(int64_t p_index) : index(p_index) {}
	RID(const Variant &v);
};

inline Variant::Variant(const ::RID& rid) {
	m_type = RID;
	v.i = rid.index;
}

inline Variant::operator ::RID() const {
	return ::RID{v.i};
}
