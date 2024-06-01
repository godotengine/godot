#pragma once

template<typename TElement>
class FreeList {
	using Implementation = JPH::FixedSizeFreeList<TElement>;

public:
	explicit FreeList(int32_t p_max_elements) {
		impl.Init((JPH::uint)p_max_elements, (JPH::uint)p_max_elements);
	}

	template<typename... TParams>
	_FORCE_INLINE_ TElement* construct(TParams&&... p_params) {
		const JPH::uint32 index = impl.ConstructObject(std::forward<TParams>(p_params)...);

		if (index == Implementation::cInvalidObjectIndex) {
			return nullptr;
		}

		return &impl.Get(index);
	}

	_FORCE_INLINE_ void destruct(TElement* p_value) { impl.DestructObject(p_value); }

private:
	Implementation impl;
};
