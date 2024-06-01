#pragma once

#include "containers/inline_allocator.hpp"
#include "core/templates/local_vector.h"

template<typename TElement, int32_t TCapacity>
class InlineVector final : public LocalVector<TElement, int32_t> {
	using Base = LocalVector<TElement, int32_t>;

public:
	InlineVector() { Base::reserve(TCapacity); }

	explicit InlineVector(int32_t p_capacity) {
		const int32_t capacity = MAX(p_capacity, TCapacity);
		Base::reserve(capacity);
	}

	InlineVector(const InlineVector& p_other) {
		const int32_t capacity = MAX(p_other.size(), TCapacity);
		Base::reserve(capacity);
		Base::operator=(p_other);
	}

	InlineVector(InlineVector&& p_other) noexcept {
		const int32_t capacity = MAX(p_other.size(), TCapacity);
		Base::reserve(capacity);
		Base::operator=(std::move(p_other));
	}

	InlineVector(std::initializer_list<TElement> p_list) {
		const auto list_size = (int32_t)std::distance(p_list.begin(), p_list.end());
		const int32_t capacity = MAX(list_size, TCapacity);
		Base::reserve(capacity);
		Base::operator=(p_list);
	}

	InlineVector& operator=(const InlineVector& p_other) {
		Base::operator=(p_other);
		return *this;
	}

	InlineVector& operator=(InlineVector&& p_other) noexcept {
		Base::operator=(std::move(p_other));
		return *this;
	}
};
