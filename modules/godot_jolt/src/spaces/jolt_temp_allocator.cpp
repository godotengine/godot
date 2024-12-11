#include "jolt_temp_allocator.hpp"

#include "servers/jolt_project_settings.hpp"

JoltTempAllocator::JoltTempAllocator()
	: capacity((uint64_t)JoltProjectSettings::get_max_temp_memory_b())
	, base(static_cast<uint8_t*>(JPH::Allocate((size_t)capacity))) { }

JoltTempAllocator::~JoltTempAllocator() {
	JPH::Free(base);
}

void* JoltTempAllocator::Allocate(uint32_t p_size) {
	if (p_size == 0) {
		return nullptr;
	}

	p_size = align_up(p_size, 16U);

	const uint64_t new_top = top + p_size;

	void* ptr = nullptr;

	if (new_top <= capacity) {
		ptr = base + top;
	} else {
		WARN_PRINT_ONCE(vformat(
			"Godot Jolt's temporary memory allocator exceeded capacity of %d MiB. "
			"Falling back to slower general-purpose allocator. "
			"Consider increasing maximum temporary memory in project settings.",
			JoltProjectSettings::get_max_temp_memory_mib()
		));

		ptr = JPH::Allocate(p_size);
	}

	top = new_top;

	return ptr;
}

void JoltTempAllocator::Free(void* p_ptr, uint32_t p_size) {
	if (p_ptr == nullptr) {
		return;
	}

	p_size = align_up(p_size, 16U);

	const uint64_t new_top = top - p_size;

	if (top <= capacity) {
		if (base + new_top != p_ptr) {
			CRASH_NOW_REPORT("Temporary memory was freed in the wrong order.");
		}
	} else {
		JPH::Free(p_ptr);
	}

	top = new_top;
}
