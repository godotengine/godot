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
template<typename Context>
RTC_FORCEINLINE bool push(Context context,
                          unsigned instanceId,
                          unsigned instancePrimId)
{
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  const bool spaceAvailable = context->instStackSize < RTC_MAX_INSTANCE_LEVEL_COUNT;
  /* We assert here because instances are silently dropped when the stack is full.
     This might be quite hard to find in production. */
  assert(spaceAvailable);
  if (likely(spaceAvailable)) {
    context->instID[context->instStackSize] = instanceId;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
    context->instPrimID[context->instStackSize] = instancePrimId;
#endif
    context->instStackSize++;
  }
  return spaceAvailable;
#else
  const bool spaceAvailable = (context->instID[0] == RTC_INVALID_GEOMETRY_ID);
  assert(spaceAvailable);
  if (likely(spaceAvailable)) {
    context->instID[0] = instanceId;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
    context->instPrimID[0] = instancePrimId;
#endif
  }
  return spaceAvailable;
#endif
}

/* 
 * Pop the last instance pushed to the stack. 
 * Do not call on an empty stack. 
 */
template<typename Context>
RTC_FORCEINLINE void pop(Context context)
{
  assert(context);
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  assert(context->instStackSize > 0);
  --context->instStackSize;
  context->instID[context->instStackSize] = RTC_INVALID_GEOMETRY_ID;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
  context->instPrimID[context->instStackSize] = RTC_INVALID_GEOMETRY_ID;
#endif
#else
  assert(context->instID[0] != RTC_INVALID_GEOMETRY_ID);
  context->instID[0] = RTC_INVALID_GEOMETRY_ID;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
  context->instPrimID[0] = RTC_INVALID_GEOMETRY_ID;
#endif
#endif
}


/* Push an instance to the stack. Used for point queries*/
RTC_FORCEINLINE bool push(RTCPointQueryContext* context,
                          unsigned int instanceId,
                          unsigned int instancePrimId,
                          AffineSpace3fa const& w2i,
                          AffineSpace3fa const& i2w)
{
  assert(context);
  const size_t stackSize = context->instStackSize;
  assert(stackSize < RTC_MAX_INSTANCE_LEVEL_COUNT);
  context->instID[stackSize] = instanceId;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
  context->instPrimID[stackSize] = instancePrimId;
#endif

  AffineSpace3fa_store_unaligned(w2i,(AffineSpace3fa*)context->world2inst[stackSize]);
  AffineSpace3fa_store_unaligned(i2w,(AffineSpace3fa*)context->inst2world[stackSize]);

#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  if (unlikely(stackSize > 0))
  {
    const AffineSpace3fa world2inst = AffineSpace3fa_load_unaligned((AffineSpace3fa*)context->world2inst[stackSize  ])
                                    * AffineSpace3fa_load_unaligned((AffineSpace3fa*)context->world2inst[stackSize-1]);
    const AffineSpace3fa inst2world = AffineSpace3fa_load_unaligned((AffineSpace3fa*)context->inst2world[stackSize-1])
                                    * AffineSpace3fa_load_unaligned((AffineSpace3fa*)context->inst2world[stackSize  ]);
    AffineSpace3fa_store_unaligned(world2inst,(AffineSpace3fa*)context->world2inst[stackSize]);
    AffineSpace3fa_store_unaligned(inst2world,(AffineSpace3fa*)context->inst2world[stackSize]);
  }
#endif
  context->instStackSize++;
  return true;
}

template<>
RTC_FORCEINLINE void pop(RTCPointQueryContext* context)
{
  assert(context);
#if RTC_MAX_INSTANCE_LEVEL_COUNT > 1
  assert(context->instStackSize > 0);
#else
  assert(context->instID[0] != RTC_INVALID_GEOMETRY_ID);
#endif
  --context->instStackSize;
  context->instID[context->instStackSize] = RTC_INVALID_GEOMETRY_ID;
#if defined(RTC_GEOMETRY_INSTANCE_ARRAY)
  context->instPrimID[context->instStackSize] = RTC_INVALID_GEOMETRY_ID;
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

RTC_FORCEINLINE void copy_UU(const RTCRayQueryContext* context, const unsigned* src, unsigned* tgt)
{
#if (RTC_MAX_INSTANCE_LEVEL_COUNT == 1)
  tgt[0] = src[0];
  
#else
  
  unsigned int depth = context->instStackSize;
  
  for (unsigned l = 0; l < depth; ++l)
    tgt[l] = src[l];
  
  for (unsigned l = depth; l < RTC_MAX_INSTANCE_LEVEL_COUNT; ++l)
    tgt[l] = RTC_INVALID_GEOMETRY_ID;

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
