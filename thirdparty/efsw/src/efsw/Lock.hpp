#ifndef EFSW_LOCK_HPP
#define EFSW_LOCK_HPP

#include <mutex>
#include <efsw/Mutex.hpp>

namespace efsw {
	using Lock = std::unique_lock<Mutex>;
} // namespace efsw

#endif
