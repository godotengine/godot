#pragma once
#include "../common.h"
#include "misc/utility_functions.hpp"




class JoltTempAllocator final : public JPH::TempAllocator {
public:
	explicit JoltTempAllocator();

	~JoltTempAllocator() override;

	void* Allocate(uint32_t p_size) override;

	void Free(void* p_ptr, uint32_t p_size) override;

private:
	uint64_t capacity = 0;

	uint64_t top = 0;

	uint8_t* base = nullptr;
};
