#pragma once

template<typename TType, size_t TCapacity, typename TOriginalType = TType>
class InlineAllocator : private std::allocator<TType> {
	using Fallback = std::allocator<TType>;

	static constexpr size_t BUFFER_COUNT = TCapacity;
	static constexpr size_t BUFFER_SIZE = sizeof(TType) * BUFFER_COUNT;

public:
	// NOLINTBEGIN(readability-identifier-naming)

	using value_type = TType;
	using size_type = size_t;
	using difference_type = ptrdiff_t;
	using is_always_equal = std::false_type;
	using propagate_on_container_copy_assignment = std::false_type;
	using propagate_on_container_move_assignment = std::false_type;
	using propagate_on_container_swap = std::false_type;

	template<class TNewType>
	struct rebind {
		using other = InlineAllocator<TNewType, TCapacity, TOriginalType>;
	};

	// NOLINTEND(readability-identifier-naming)

	constexpr InlineAllocator() noexcept = default;

	template<class TOtherType>
	explicit constexpr InlineAllocator(
		[[maybe_unused]] const InlineAllocator<TOtherType, TCapacity, TOriginalType>& p_other
	) noexcept { }

	constexpr InlineAllocator(const InlineAllocator& p_other) noexcept
		: using_buffer(p_other.using_buffer) { }

	constexpr InlineAllocator([[maybe_unused]] InlineAllocator&& p_other) noexcept { }

	constexpr TType* allocate(size_t p_count) {
		if constexpr (std::is_same_v<TType, TOriginalType>) {
			if (p_count <= BUFFER_COUNT) {
				using_buffer = true;
				return reinterpret_cast<TType*>(&buffer);
			}
		}

		using_buffer = false;

		return Fallback::allocate(p_count);
	}

	constexpr void deallocate(TType* p_ptr, size_t p_count) {
		if (p_ptr != reinterpret_cast<TType*>(&buffer)) {
			Fallback::deallocate(p_ptr, p_count);
		}

		using_buffer = false;
	}

	constexpr InlineAllocator& operator=(const InlineAllocator& p_other) noexcept {
		if (this != &p_other) {
			using_buffer = p_other.using_buffer;
		}

		return *this;
	}

	constexpr InlineAllocator& operator=([[maybe_unused]] InlineAllocator&& p_other) noexcept {
		return *this;
	}

	friend constexpr bool operator==(const InlineAllocator& p_lhs, const InlineAllocator& p_rhs) {
		// This must return false when either operand is using the inline buffer, in order to force
		// an element-wise move during container move-assignment. This is technically non-conforming
		// with the standard's requirements for allocators.
		return !p_lhs.using_buffer && !p_rhs.using_buffer;
	}

	friend constexpr bool operator!=(const InlineAllocator& p_lhs, const InlineAllocator& p_rhs) {
		return !(p_lhs == p_rhs);
	}

private:
	alignas(alignof(TType)) uint8_t buffer[BUFFER_SIZE];

	bool using_buffer = false;
};
