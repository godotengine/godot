// This file is part of meshoptimizer library; see meshoptimizer.h for version/license details
#include "meshoptimizer.h"

#ifdef MESHOPTIMIZER_ALLOC_EXPORT
meshopt_Allocator::Storage& meshopt_Allocator::storage()
{
	static Storage s = {::operator new, ::operator delete };
	return s;
}
#endif

void meshopt_setAllocator(void* (MESHOPTIMIZER_ALLOC_CALLCONV* allocate)(size_t), void (MESHOPTIMIZER_ALLOC_CALLCONV* deallocate)(void*))
{
	meshopt_Allocator::Storage& s = meshopt_Allocator::storage();
	s.allocate = allocate;
	s.deallocate = deallocate;
}
