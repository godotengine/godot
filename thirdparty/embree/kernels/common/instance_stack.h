// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore.h"

namespace embree {
namespace instance_id_stack {

static_assert(RTC_MAX_INSTANCE_LEVEL_COUNT > 0, 
              "RTC_MAX_INSTANCE_LEVEL_COUNT must be greater than 0.");

/*******************************************************************************
 * Instance ID stack manipulation.
 * This is used from the instance intersector.
 ******************************************************************************/

/* 
 * Push an instance to the stack. 
 */
RTC_FORCEINLINE bool push(RTCIntersectContext* context, 
                          unsigned instanceId)
{
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  const bool spaceAvailable = context->instStackSize < RTC_MAX_INSTANCE_LEVEL_COUNT;
  /* We assert here because instances are silently dropped when the stack is full. 
     This might be quite hard to find in production. */
  assert(spaceAvailable); 
  if (likely(spaceAvailable))
    context->instID[context->instStackSize++] = instanceId;
  return spaceAvailable;
#else
  const bool spaceAvailable = (context->instID[0] == RTC_INVALID_GEOMETRY_ID);
  assert(spaceAvailable); 
  if (likely(spaceAvailable))
    context->instID[0] = instanceId;
  return spaceAvailable;
#endif
}


/* 
 * Pop the last instance pushed to the stack. 
 * Do not call on an empty stack. 
 */
RTC_FORCEINLINE void pop(RTCIntersectContext* context)
{
  assert(context);
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  assert(context->instStackSize > 0);
  context->instID[--context->instStackSize] = RTC_INVALID_GEOMETRY_ID;
#else
  assert(context->instID[0] != RTC_INVALID_GEOMETRY_ID);
  context->instID[0] = RTC_INVALID_GEOMETRY_ID;
#endif
}

/*******************************************************************************
 * Optimized instance id stack copy.
 * The copy() function at the bottom of this block will either copy full
 * stacks or copy only until the last valid element has been copied, depending
 * on RTC_MAX_INSTANCE_LEVEL_COUNT.
 ******************************************************************************/

/*
 * Plain array assignment. This works for scalar->scalar,
 * scalar->vector, and vector->vector.
 */
template <class Src, class Tgt>
RTC_FORCEINLINE void level_copy(unsigned level, Src* src, Tgt* tgt)
{
  tgt[level] = src[level];
}

/*
 * Masked SIMD vector->vector store.
 */
template <int K>
RTC_FORCEINLINE void level_copy(unsigned level, const vuint<K>* src, vuint<K>* tgt, const vbool<K>& mask)
{
  vuint<K>::storeu(mask, tgt + level, src[level]);
}

/*
 * Masked scalar->SIMD vector store.
 */
template <int K>
RTC_FORCEINLINE void level_copy(unsigned level, const unsigned* src, vuint<K>* tgt, const vbool<K>& mask)
{
  vuint<K>::store(mask, tgt + level, src[level]);
}

/*
 * Indexed assign from vector to scalar.
 */
template <int K>
RTC_FORCEINLINE void level_copy(unsigned level, const vuint<K>* src, unsigned* tgt, const size_t& idx)
{
  tgt[level] = src[level][idx];
}

/*
 * Indexed assign from scalar to vector.
 */
template <int K>
RTC_FORCEINLINE void level_copy(unsigned level, const unsigned* src, vuint<K>* tgt, const size_t& idx)
{
  tgt[level][idx] = src[level];
}

/*
 * Indexed assign from vector to vector.
 */
template <int K>
RTC_FORCEINLINE void level_copy(unsigned level, const vuint<K>* src, vuint<K>* tgt, const size_t& i, const size_t& j)
{
  tgt[level][j] = src[level][i];
}

/*
 * Check if the given stack level is valid.
 * These are only used for large max stack sizes.
 */
RTC_FORCEINLINE bool level_valid(unsigned level, const unsigned* stack)
{
  return stack[level] != RTC_INVALID_GEOMETRY_ID;
}
RTC_FORCEINLINE bool level_valid(unsigned level, const unsigned* stack, const size_t& /*i*/)
{
  return stack[level] != RTC_INVALID_GEOMETRY_ID;
}
template <int K>
RTC_FORCEINLINE bool level_valid(unsigned level, const unsigned* stack, const vbool<K>& /*mask*/)
{
  return stack[level] != RTC_INVALID_GEOMETRY_ID;
}

template <int K>
RTC_FORCEINLINE bool level_valid(unsigned level, const vuint<K>* stack)
{
  return any(stack[level] != RTC_INVALID_GEOMETRY_ID);
}
template <int K>
RTC_FORCEINLINE bool level_valid(unsigned level, const vuint<K>* stack, const vbool<K>& mask)
{
  return any(mask & (stack[level] != RTC_INVALID_GEOMETRY_ID));
}

template <int K>
RTC_FORCEINLINE bool level_valid(unsigned level, const vuint<K>* stack, const size_t& i)
{
  return stack[level][i] != RTC_INVALID_GEOMETRY_ID;
}
template <int K>
RTC_FORCEINLINE bool level_valid(unsigned level, const vuint<K>* stack, const size_t& i, const size_t& /*j*/)
{
  return stack[level][i] != RTC_INVALID_GEOMETRY_ID;
}

/*
 * Copy an instance ID stack.
 *
 * This function automatically selects a LevelFunctor from the above Assign 
 * structs.
 */
template <class Src, class Tgt, class... Args>
RTC_FORCEINLINE void copy(Src src, Tgt tgt, Args&&... args)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  /* 
   * Avoid all loops for only one level. 
   */
  level_copy(0, src, tgt, std::forward<Args>(args)...);

#elif (RTC_MAX_INSTANCE_LEVEL_COUNT <= 4)
  /* 
   * It is faster to avoid the valid test for low level counts.
   * Just copy the whole stack.
   */
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
    level_copy(l, src, tgt, std::forward<Args>(args)...);

#else
  /* 
   * For general stack sizes, it pays off to test for validity.
   */
  bool valid = true;
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT && valid; ++l)
  {
    level_copy(l, src, tgt, std::forward<Args>(args)...);
    valid = level_valid(l, src, std::forward<Args>(args)...);
  }
#endif
}

} // namespace instance_id_stack
} // namespace embree

