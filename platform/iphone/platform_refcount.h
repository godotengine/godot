#include "safe_refcount.h"

#ifdef IPHONE_ENABLED

#define REFCOUNT_T int
#define REFCOUNT_GET_T int const volatile&

#include <libkern/OSAtomic.h>

inline int atomic_conditional_increment(volatile int* v) {
	return (*v==0)? 0 : OSAtomicIncrement32(v);
}

inline int atomic_decrement(volatile int* v) {
	return OSAtomicDecrement32(v);
}

#endif
