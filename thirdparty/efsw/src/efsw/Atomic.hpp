#ifndef EFSW_ATOMIC_BOOL_HPP
#define EFSW_ATOMIC_BOOL_HPP

#include <efsw/base.hpp>

#include <atomic>

namespace efsw {

template <typename T> class Atomic {
  public:
	explicit Atomic( T set = false ) : set_( set ) {}

	Atomic& operator=( T set ) {
		set_.store( set, std::memory_order_release );
		return *this;
	}

	explicit operator T() const {
		return set_.load( std::memory_order_acquire );
	}

	T load() const {
		return set_.load( std::memory_order_acquire );
	}

  private:
	std::atomic<T> set_;
};

} // namespace efsw

#endif
