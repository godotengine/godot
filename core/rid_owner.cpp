#include "rid_owner.h"

#include <atomic>

std::atomic<uint64_t> RID_AllocBase::base_id(1);
