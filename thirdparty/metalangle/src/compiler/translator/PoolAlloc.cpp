//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#include "compiler/translator/PoolAlloc.h"

#include <assert.h>
#include "common/tls.h"

TLSIndex PoolIndex = TLS_INVALID_INDEX;

bool InitializePoolIndex()
{
    assert(PoolIndex == TLS_INVALID_INDEX);

    PoolIndex = CreateTLSIndex();
    return PoolIndex != TLS_INVALID_INDEX;
}

void FreePoolIndex()
{
    assert(PoolIndex != TLS_INVALID_INDEX);

    DestroyTLSIndex(PoolIndex);
    PoolIndex = TLS_INVALID_INDEX;
}

angle::PoolAllocator *GetGlobalPoolAllocator()
{
    assert(PoolIndex != TLS_INVALID_INDEX);
    return static_cast<angle::PoolAllocator *>(GetTLSValue(PoolIndex));
}

void SetGlobalPoolAllocator(angle::PoolAllocator *poolAllocator)
{
    assert(PoolIndex != TLS_INVALID_INDEX);
    SetTLSValue(PoolIndex, poolAllocator);
}
