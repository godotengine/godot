/*
 * Copyright 2016 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkArenaAlloc_DEFINED
#define SkArenaAlloc_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkTFitsIn.h"
#include "include/private/SkTo.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

// We found allocating strictly doubling amounts of memory from the heap left too
// much unused slop, particularly on Android.  Instead we'll follow a Fibonacci-like
// progression.

// SkFibonacci47 is the first 47 Fibonacci numbers. Fib(47) is the largest value less than 2 ^ 32.
extern std::array<const uint32_t, 47> SkFibonacci47;
template<uint32_t kMaxSize>
class SkFibBlockSizes {
public:
    // staticBlockSize, and firstAllocationSize are parameters describing the initial memory
    // layout. staticBlockSize describes the size of the inlined memory, and firstAllocationSize
    // describes the size of the first block to be allocated if the static block is exhausted. By
    // convention, firstAllocationSize is the first choice for the block unit size followed by
    // staticBlockSize followed by the default of 1024 bytes.
    SkFibBlockSizes(uint32_t staticBlockSize, uint32_t firstAllocationSize) : fIndex{0} {
        fBlockUnitSize = firstAllocationSize > 0 ? firstAllocationSize :
                         staticBlockSize     > 0 ? staticBlockSize     : 1024;

        SkASSERT_RELEASE(0 < fBlockUnitSize);
        SkASSERT_RELEASE(fBlockUnitSize < std::min(kMaxSize, (1u << 26) - 1));
    }

    uint32_t nextBlockSize() {
        uint32_t result = SkFibonacci47[fIndex] * fBlockUnitSize;

        if (SkTo<size_t>(fIndex + 1) < SkFibonacci47.size() &&
            SkFibonacci47[fIndex + 1] < kMaxSize / fBlockUnitSize)
        {
            fIndex += 1;
        }

        return result;
    }

private:
    uint32_t fIndex : 6;
    uint32_t fBlockUnitSize : 26;
};

// SkArenaAlloc allocates object and destroys the allocated objects when destroyed. It's designed
// to minimize the number of underlying block allocations. SkArenaAlloc allocates first out of an
// (optional) user-provided block of memory, and when that's exhausted it allocates on the heap,
// starting with an allocation of firstHeapAllocation bytes.  If your data (plus a small overhead)
// fits in the user-provided block, SkArenaAlloc never uses the heap, and if it fits in
// firstHeapAllocation bytes, it'll use the heap only once. If 0 is specified for
// firstHeapAllocation, then blockSize is used unless that too is 0, then 1024 is used.
//
// Examples:
//
//   char block[mostCasesSize];
//   SkArenaAlloc arena(block, mostCasesSize);
//
// If mostCasesSize is too large for the stack, you can use the following pattern.
//
//   std::unique_ptr<char[]> block{new char[mostCasesSize]};
//   SkArenaAlloc arena(block.get(), mostCasesSize, almostAllCasesSize);
//
// If the program only sometimes allocates memory, use the following pattern.
//
//   SkArenaAlloc arena(nullptr, 0, almostAllCasesSize);
//
// The storage does not necessarily need to be on the stack. Embedding the storage in a class also
// works.
//
//   class Foo {
//       char storage[mostCasesSize];
//       SkArenaAlloc arena (storage, mostCasesSize);
//   };
//
// In addition, the system is optimized to handle POD data including arrays of PODs (where
// POD is really data with no destructors). For POD data it has zero overhead per item, and a
// typical per block overhead of 8 bytes. For non-POD objects there is a per item overhead of 4
// bytes. For arrays of non-POD objects there is a per array overhead of typically 8 bytes. There
// is an addition overhead when switching from POD data to non-POD data of typically 8 bytes.
//
// If additional blocks are needed they are increased exponentially. This strategy bounds the
// recursion of the RunDtorsOnBlock to be limited to O(log size-of-memory). Block size grow using
// the Fibonacci sequence which means that for 2^32 memory there are 48 allocations, and for 2^48
// there are 71 allocations.
class SkArenaAlloc {
public:
    SkArenaAlloc(char* block, size_t blockSize, size_t firstHeapAllocation);

    explicit SkArenaAlloc(size_t firstHeapAllocation)
        : SkArenaAlloc(nullptr, 0, firstHeapAllocation) {}

    SkArenaAlloc(const SkArenaAlloc&) = delete;
    SkArenaAlloc& operator=(const SkArenaAlloc&) = delete;
    SkArenaAlloc(SkArenaAlloc&&) = delete;
    SkArenaAlloc& operator=(SkArenaAlloc&&) = delete;

    ~SkArenaAlloc();

    template <typename Ctor>
    auto make(Ctor&& ctor) -> decltype(ctor(nullptr)) {
        using T = std::remove_pointer_t<decltype(ctor(nullptr))>;

        uint32_t size      = SkToU32(sizeof(T));
        uint32_t alignment = SkToU32(alignof(T));
        char* objStart;
        if (std::is_trivially_destructible<T>::value) {
            objStart = this->allocObject(size, alignment);
            fCursor = objStart + size;
        } else {
            objStart = this->allocObjectWithFooter(size + sizeof(Footer), alignment);
            // Can never be UB because max value is alignof(T).
            uint32_t padding = SkToU32(objStart - fCursor);

            // Advance to end of object to install footer.
            fCursor = objStart + size;
            FooterAction* releaser = [](char* objEnd) {
                char* objStart = objEnd - (sizeof(T) + sizeof(Footer));
                ((T*)objStart)->~T();
                return objStart;
            };
            this->installFooter(releaser, padding);
        }

        // This must be last to make objects with nested use of this allocator work.
        return ctor(objStart);
    }

    template <typename T, typename... Args>
    T* make(Args&&... args) {
        return this->make([&](void* objStart) {
            return new(objStart) T(std::forward<Args>(args)...);
        });
    }

    template <typename T>
    T* makeArrayDefault(size_t count) {
        T* array = this->allocUninitializedArray<T>(count);
        for (size_t i = 0; i < count; i++) {
            // Default initialization: if T is primitive then the value is left uninitialized.
            new (&array[i]) T;
        }
        return array;
    }

    template <typename T>
    T* makeArray(size_t count) {
        T* array = this->allocUninitializedArray<T>(count);
        for (size_t i = 0; i < count; i++) {
            // Value initialization: if T is primitive then the value is zero-initialized.
            new (&array[i]) T();
        }
        return array;
    }

    template <typename T, typename Initializer>
    T* makeInitializedArray(size_t count, Initializer initializer) {
        T* array = this->allocUninitializedArray<T>(count);
        for (size_t i = 0; i < count; i++) {
            new (&array[i]) T(initializer(i));
        }
        return array;
    }

    // Only use makeBytesAlignedTo if none of the typed variants are impractical to use.
    void* makeBytesAlignedTo(size_t size, size_t align) {
        AssertRelease(SkTFitsIn<uint32_t>(size));
        auto objStart = this->allocObject(SkToU32(size), SkToU32(align));
        fCursor = objStart + size;
        return objStart;
    }

private:
    static void AssertRelease(bool cond) { if (!cond) { ::abort(); } }

    using FooterAction = char* (char*);
    struct Footer {
        uint8_t unaligned_action[sizeof(FooterAction*)];
        uint8_t padding;
    };

    static char* SkipPod(char* footerEnd);
    static void RunDtorsOnBlock(char* footerEnd);
    static char* NextBlock(char* footerEnd);

    template <typename T>
    void installRaw(const T& val) {
        memcpy(fCursor, &val, sizeof(val));
        fCursor += sizeof(val);
    }
    void installFooter(FooterAction* releaser, uint32_t padding);

    void ensureSpace(uint32_t size, uint32_t alignment);

    char* allocObject(uint32_t size, uint32_t alignment) {
        uintptr_t mask = alignment - 1;
        uintptr_t alignedOffset = (~reinterpret_cast<uintptr_t>(fCursor) + 1) & mask;
        uintptr_t totalSize = size + alignedOffset;
        AssertRelease(totalSize >= size);
        if (totalSize > static_cast<uintptr_t>(fEnd - fCursor)) {
            this->ensureSpace(size, alignment);
            alignedOffset = (~reinterpret_cast<uintptr_t>(fCursor) + 1) & mask;
        }

        char* object = fCursor + alignedOffset;

        SkASSERT((reinterpret_cast<uintptr_t>(object) & (alignment - 1)) == 0);
        SkASSERT(object + size <= fEnd);

        return object;
    }

    char* allocObjectWithFooter(uint32_t sizeIncludingFooter, uint32_t alignment);

    template <typename T>
    T* allocUninitializedArray(size_t countZ) {
        AssertRelease(SkTFitsIn<uint32_t>(countZ));
        uint32_t count = SkToU32(countZ);

        char* objStart;
        AssertRelease(count <= std::numeric_limits<uint32_t>::max() / sizeof(T));
        uint32_t arraySize = SkToU32(count * sizeof(T));
        uint32_t alignment = SkToU32(alignof(T));

        if (std::is_trivially_destructible<T>::value) {
            objStart = this->allocObject(arraySize, alignment);
            fCursor = objStart + arraySize;
        } else {
            constexpr uint32_t overhead = sizeof(Footer) + sizeof(uint32_t);
            AssertRelease(arraySize <= std::numeric_limits<uint32_t>::max() - overhead);
            uint32_t totalSize = arraySize + overhead;
            objStart = this->allocObjectWithFooter(totalSize, alignment);

            // Can never be UB because max value is alignof(T).
            uint32_t padding = SkToU32(objStart - fCursor);

            // Advance to end of array to install footer.
            fCursor = objStart + arraySize;
            this->installRaw(SkToU32(count));
            this->installFooter(
                [](char* footerEnd) {
                    char* objEnd = footerEnd - (sizeof(Footer) + sizeof(uint32_t));
                    uint32_t count;
                    memmove(&count, objEnd, sizeof(uint32_t));
                    char* objStart = objEnd - count * sizeof(T);
                    T* array = (T*) objStart;
                    for (uint32_t i = 0; i < count; i++) {
                        array[i].~T();
                    }
                    return objStart;
                },
                padding);
        }

        return (T*)objStart;
    }

    char*          fDtorCursor;
    char*          fCursor;
    char*          fEnd;

    SkFibBlockSizes<std::numeric_limits<uint32_t>::max()> fFibonacciProgression;
};

class SkArenaAllocWithReset : public SkArenaAlloc {
public:
    SkArenaAllocWithReset(char* block, size_t blockSize, size_t firstHeapAllocation);

    explicit SkArenaAllocWithReset(size_t firstHeapAllocation)
            : SkArenaAllocWithReset(nullptr, 0, firstHeapAllocation) {}

    // Destroy all allocated objects, free any heap allocations.
    void reset();

private:
    char* const    fFirstBlock;
    const uint32_t fFirstSize;
    const uint32_t fFirstHeapAllocationSize;
};

// Helper for defining allocators with inline/reserved storage.
// For argument declarations, stick to the base type (SkArenaAlloc).
// Note: Inheriting from the storage first means the storage will outlive the
// SkArenaAlloc, letting ~SkArenaAlloc read it as it calls destructors.
// (This is mostly only relevant for strict tools like MSAN.)
template <size_t InlineStorageSize>
class SkSTArenaAlloc : private std::array<char, InlineStorageSize>, public SkArenaAlloc {
public:
    explicit SkSTArenaAlloc(size_t firstHeapAllocation = InlineStorageSize)
        : SkArenaAlloc{this->data(), this->size(), firstHeapAllocation} {}
};

template <size_t InlineStorageSize>
class SkSTArenaAllocWithReset
        : private std::array<char, InlineStorageSize>, public SkArenaAllocWithReset {
public:
    explicit SkSTArenaAllocWithReset(size_t firstHeapAllocation = InlineStorageSize)
            : SkArenaAllocWithReset{this->data(), this->size(), firstHeapAllocation} {}
};

#endif  // SkArenaAlloc_DEFINED
