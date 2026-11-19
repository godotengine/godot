// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2026 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#pragma once

JPH_NAMESPACE_BEGIN

/// Prefetch the given address to L1 cache. Can be used to avoid cache misses, but should be used with care as it can also cause cache pollution if used incorrectly.
template <typename T>
inline void PrefetchL1(const T *inAddress)
{
#ifdef JPH_USE_SSE
	_mm_prefetch(reinterpret_cast<const char *>(inAddress), _MM_HINT_T0);
#elif defined(JPH_COMPILER_GCC) || defined(JPH_COMPILER_CLANG)
	__builtin_prefetch(inAddress, 0, 3);
#endif
}

JPH_NAMESPACE_END
