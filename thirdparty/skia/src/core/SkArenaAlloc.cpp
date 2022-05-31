/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/core/SkArenaAlloc.h"
#include <algorithm>
#include <new>

static char* end_chain(char*) { return nullptr; }

SkArenaAlloc::SkArenaAlloc(char* block, size_t size, size_t firstHeapAllocation)
    : fDtorCursor {block}
    , fCursor     {block}
    , fEnd        {block + SkToU32(size)}
    , fFibonacciProgression{SkToU32(size), SkToU32(firstHeapAllocation)}
{
    if (size < sizeof(Footer)) {
        fEnd = fCursor = fDtorCursor = nullptr;
    }

    if (fCursor != nullptr) {
        this->installFooter(end_chain, 0);
    }
}

SkArenaAlloc::~SkArenaAlloc() {
    RunDtorsOnBlock(fDtorCursor);
}

void SkArenaAlloc::installFooter(FooterAction* action, uint32_t padding) {
    assert(SkTFitsIn<uint8_t>(padding));
    this->installRaw(action);
    this->installRaw((uint8_t)padding);
    fDtorCursor = fCursor;
}

char* SkArenaAlloc::SkipPod(char* footerEnd) {
    char* objEnd = footerEnd - (sizeof(Footer) + sizeof(uint32_t));
    uint32_t skip;
    memmove(&skip, objEnd, sizeof(uint32_t));
    return objEnd - (ptrdiff_t) skip;
}

void SkArenaAlloc::RunDtorsOnBlock(char* footerEnd) {
    while (footerEnd != nullptr) {
        FooterAction* action;
        uint8_t       padding;

        memcpy(&action,  footerEnd - sizeof( Footer), sizeof( action));
        memcpy(&padding, footerEnd - sizeof(padding), sizeof(padding));

        footerEnd = action(footerEnd) - (ptrdiff_t)padding;
    }
}

char* SkArenaAlloc::NextBlock(char* footerEnd) {
    char* objEnd = footerEnd - (sizeof(char*) + sizeof(Footer));
    char* next;
    memmove(&next, objEnd, sizeof(char*));
    RunDtorsOnBlock(next);
    delete [] objEnd;
    return nullptr;
}


void SkArenaAlloc::ensureSpace(uint32_t size, uint32_t alignment) {
    constexpr uint32_t headerSize = sizeof(Footer) + sizeof(ptrdiff_t);
    constexpr uint32_t maxSize = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t overhead = headerSize + sizeof(Footer);
    AssertRelease(size <= maxSize - overhead);
    uint32_t objSizeAndOverhead = size + overhead;

    const uint32_t alignmentOverhead = alignment - 1;
    AssertRelease(objSizeAndOverhead <= maxSize - alignmentOverhead);
    objSizeAndOverhead += alignmentOverhead;

    uint32_t minAllocationSize = fFibonacciProgression.nextBlockSize();
    uint32_t allocationSize = std::max(objSizeAndOverhead, minAllocationSize);

    // Round up to a nice size. If > 32K align to 4K boundary else up to max_align_t. The > 32K
    // heuristic is from the JEMalloc behavior.
    {
        uint32_t mask = allocationSize > (1 << 15) ? (1 << 12) - 1 : 16 - 1;
        AssertRelease(allocationSize <= maxSize - mask);
        allocationSize = (allocationSize + mask) & ~mask;
    }

    char* newBlock = new char[allocationSize];

    auto previousDtor = fDtorCursor;
    fCursor = newBlock;
    fDtorCursor = newBlock;
    fEnd = fCursor + allocationSize;
    this->installRaw(previousDtor);
    this->installFooter(NextBlock, 0);
}

char* SkArenaAlloc::allocObjectWithFooter(uint32_t sizeIncludingFooter, uint32_t alignment) {
    uintptr_t mask = alignment - 1;

restart:
    uint32_t skipOverhead = 0;
    const bool needsSkipFooter = fCursor != fDtorCursor;
    if (needsSkipFooter) {
        skipOverhead = sizeof(Footer) + sizeof(uint32_t);
    }
    const uint32_t totalSize = sizeIncludingFooter + skipOverhead;

    // Math on null fCursor/fEnd is undefined behavior, so explicitly check for first alloc.
    if (!fCursor) {
        this->ensureSpace(totalSize, alignment);
        goto restart;
    }

    assert(fEnd);
    // This test alone would be enough nullptr were defined to be 0, but it's not.
    char* objStart = (char*)((uintptr_t)(fCursor + skipOverhead + mask) & ~mask);
    if ((ptrdiff_t)totalSize > fEnd - objStart) {
        this->ensureSpace(totalSize, alignment);
        goto restart;
    }

    AssertRelease((ptrdiff_t)totalSize <= fEnd - objStart);

    // Install a skip footer if needed, thus terminating a run of POD data. The calling code is
    // responsible for installing the footer after the object.
    if (needsSkipFooter) {
        this->installRaw(SkToU32(fCursor - fDtorCursor));
        this->installFooter(SkipPod, 0);
    }

    return objStart;
}

SkArenaAllocWithReset::SkArenaAllocWithReset(char* block,
                                             size_t size,
                                             size_t firstHeapAllocation)
        : SkArenaAlloc(block, size, firstHeapAllocation)
        , fFirstBlock{block}
        , fFirstSize{SkToU32(size)}
        , fFirstHeapAllocationSize{SkToU32(firstHeapAllocation)} {}

void SkArenaAllocWithReset::reset() {
    char* const    firstBlock              = fFirstBlock;
    const uint32_t firstSize               = fFirstSize;
    const uint32_t firstHeapAllocationSize = fFirstHeapAllocationSize;
    this->~SkArenaAllocWithReset();
    new (this) SkArenaAllocWithReset{firstBlock, firstSize, firstHeapAllocationSize};
}

// SkFibonacci47 is the first 47 Fibonacci numbers. Fib(47) is the largest value less than 2 ^ 32.
// Used by SkFibBlockSizes.
std::array<const uint32_t, 47> SkFibonacci47 {
                1,         1,          2,          3,          5,          8,
               13,        21,         34,         55,         89,        144,
              233,       377,        610,        987,       1597,       2584,
             4181,      6765,      10946,      17711,      28657,      46368,
            75025,    121393,     196418,     317811,     514229,     832040,
          1346269,   2178309,    3524578,    5702887,    9227465,   14930352,
         24157817,  39088169,   63245986,  102334155,  165580141,  267914296,
        433494437, 701408733, 1134903170, 1836311903, 2971215073,
};
