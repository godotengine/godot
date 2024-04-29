// Copyright 2009-2021 Intel Corporation
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

/*
 * Optimized instance id stack copy.
 * The copy() functions will either copy full
 * stacks or copy only until the last valid element has been copied, depending
 * on RTC_MAX_INSTANCE_LEVEL_COUNT.
 */
RTC_FORCEINLINE void copy_UU(const unsigned* src, unsigned* tgt)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0] = src[0];
  
#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    tgt[l] = src[l];
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_UV(const unsigned* src, vuint<K>* tgt)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0] = src[0];

#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    tgt[l] = src[l];
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_UV(const unsigned* src, vuint<K>* tgt, size_t j)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0][j] = src[0];

#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    tgt[l][j] = src[l];
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_UV(const unsigned* src, vuint<K>* tgt, const vbool<K>& mask)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  vuint<K>::store(mask, tgt, src[0]);

#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    vuint<K>::store(mask, tgt + l, src[l]);
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_VU(const vuint<K>* src, unsigned* tgt, size_t i)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0] = src[0][i];

#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    tgt[l] = src[l][i];
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l][i] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_VV(const vuint<K>* src, vuint<K>* tgt, size_t i, size_t j)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0][j] = src[0][i];

#else
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    tgt[l][j] = src[l][i];
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4)
      if (src[l][i] == RTC_INVALID_GEOMETRY_ID)
        break;
  }
#endif
}

template <int K>
RTC_FORCEINLINE void copy_VV(const vuint<K>* src, vuint<K>* tgt, const vbool<K>& mask)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  vuint<K>::store(mask, tgt, src[0]);

#else
  vbool<K> done = !mask;
  for (unsigned l = 0; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l) {
    vuint<K>::store(mask, tgt + l, src[l]);
    if (RTC_MAX_INSTANCE_LEVEL_COUNT > 4) {
      done |= src[l] == RTC_INVALID_GEOMETRY_ID;
      if (all(done)) break;
    }
  }
#endif
}

} // namespace instance_id_stack
} // namespace embree
