/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTArray_DEFINED
#define SkTArray_DEFINED

#include "include/core/SkMath.h"
#include "include/core/SkTypes.h"
#include "include/private/SkMalloc.h"
#include "include/private/SkSafe32.h"
#include "include/private/SkTLogic.h"
#include "include/private/SkTemplates.h"
#include "include/private/SkTo.h"

#include <algorithm>
#include <string.h>
#include <initializer_list>
#include <memory>
#include <new>
#include <utility>

/** SkTArray<T> implements a typical, mostly std::vector-like array.
    Each T will be default-initialized on allocation, and ~T will be called on destruction.

    MEM_MOVE controls the behavior when a T needs to be moved (e.g. when the array is resized)
      - true: T will be bit-copied via memcpy.
      - false: T will be moved via move-constructors.

    Modern implementations of std::vector<T> will generally provide similar performance
    characteristics when used with appropriate care. Consider using std::vector<T> in new code.
*/
template <typename T, bool MEM_MOVE = false> class SkTArray {
private:
    enum ReallocType { kExactFit, kGrowing, kShrinking };

public:
    using value_type = T;

    /**
     * Creates an empty array with no initial storage
     */
    SkTArray() { this->init(0); }

    /**
     * Creates an empty array that will preallocate space for reserveCount
     * elements.
     */
    explicit SkTArray(int reserveCount) : SkTArray() { this->reserve_back(reserveCount); }

    /**
     * Copies one array to another. The new array will be heap allocated.
     */
    SkTArray(const SkTArray& that)
        : SkTArray(that.fItemArray, that.fCount) {}

    SkTArray(SkTArray&& that) {
        if (that.fOwnMemory) {
            fItemArray = that.fItemArray;
            fCount = that.fCount;
            fAllocCount = that.fAllocCount;
            fOwnMemory = true;
            fReserved = that.fReserved;

            that.fItemArray = nullptr;
            that.fCount = 0;
            that.fAllocCount = 0;
            that.fOwnMemory = true;
            that.fReserved = false;
        } else {
            this->init(that.fCount);
            that.move(fItemArray);
            that.fCount = 0;
        }
    }

    /**
     * Creates a SkTArray by copying contents of a standard C array. The new
     * array will be heap allocated. Be careful not to use this constructor
     * when you really want the (void*, int) version.
     */
    SkTArray(const T* array, int count) {
        this->init(count);
        this->copy(array);
    }
    /**
     * Creates a SkTArray by copying contents of an initializer list.
     */
    SkTArray(std::initializer_list<T> data)
        : SkTArray(data.begin(), data.size()) {}

    SkTArray& operator=(const SkTArray& that) {
        if (this == &that) {
            return *this;
        }
        for (int i = 0; i < this->count(); ++i) {
            fItemArray[i].~T();
        }
        fCount = 0;
        this->checkRealloc(that.count(), kExactFit);
        fCount = that.fCount;
        this->copy(that.fItemArray);
        return *this;
    }

    SkTArray& operator=(SkTArray&& that) {
        if (this == &that) {
            return *this;
        }
        for (int i = 0; i < this->count(); ++i) {
            fItemArray[i].~T();
        }
        fCount = 0;
        this->checkRealloc(that.count(), kExactFit);
        fCount = that.fCount;
        that.move(fItemArray);
        that.fCount = 0;
        return *this;
    }

    ~SkTArray() {
        for (int i = 0; i < this->count(); ++i) {
            fItemArray[i].~T();
        }
        if (fOwnMemory) {
            sk_free(fItemArray);
        }
    }

    /**
     * Resets to count() == 0 and resets any reserve count.
     */
    void reset() {
        this->pop_back_n(fCount);
        fReserved = false;
    }

    /**
     * Resets to count() = n newly constructed T objects and resets any reserve count.
     */
    void reset(int n) {
        SkASSERT(n >= 0);
        for (int i = 0; i < this->count(); ++i) {
            fItemArray[i].~T();
        }
        // Set fCount to 0 before calling checkRealloc so that no elements are moved.
        fCount = 0;
        this->checkRealloc(n, kExactFit);
        fCount = n;
        for (int i = 0; i < this->count(); ++i) {
            new (fItemArray + i) T;
        }
        fReserved = false;
    }

    /**
     * Resets to a copy of a C array and resets any reserve count.
     */
    void reset(const T* array, int count) {
        for (int i = 0; i < this->count(); ++i) {
            fItemArray[i].~T();
        }
        fCount = 0;
        this->checkRealloc(count, kExactFit);
        fCount = count;
        this->copy(array);
        fReserved = false;
    }

    /**
     * Ensures there is enough reserved space for n additional elements. The is guaranteed at least
     * until the array size grows above n and subsequently shrinks below n, any version of reset()
     * is called, or reserve_back() is called again.
     */
    void reserve_back(int n) {
        SkASSERT(n >= 0);
        if (n > 0) {
            this->checkRealloc(n, kExactFit);
            fReserved = fOwnMemory;
        } else {
            fReserved = false;
        }
    }

    void removeShuffle(int n) {
        SkASSERT(n < this->count());
        int newCount = fCount - 1;
        fCount = newCount;
        fItemArray[n].~T();
        if (n != newCount) {
            this->move(n, newCount);
        }
    }

    /**
     * Number of elements in the array.
     */
    int count() const { return fCount; }

    /**
     * Is the array empty.
     */
    bool empty() const { return !fCount; }

    /**
     * Adds 1 new default-initialized T value and returns it by reference. Note
     * the reference only remains valid until the next call that adds or removes
     * elements.
     */
    T& push_back() {
        void* newT = this->push_back_raw(1);
        return *new (newT) T;
    }

    /**
     * Version of above that uses a copy constructor to initialize the new item
     */
    T& push_back(const T& t) {
        void* newT = this->push_back_raw(1);
        return *new (newT) T(t);
    }

    /**
     * Version of above that uses a move constructor to initialize the new item
     */
    T& push_back(T&& t) {
        void* newT = this->push_back_raw(1);
        return *new (newT) T(std::move(t));
    }

    /**
     *  Construct a new T at the back of this array.
     */
    template<class... Args> T& emplace_back(Args&&... args) {
        void* newT = this->push_back_raw(1);
        return *new (newT) T(std::forward<Args>(args)...);
    }

    /**
     * Allocates n more default-initialized T values, and returns the address of
     * the start of that new range. Note: this address is only valid until the
     * next API call made on the array that might add or remove elements.
     */
    T* push_back_n(int n) {
        SkASSERT(n >= 0);
        void* newTs = this->push_back_raw(n);
        for (int i = 0; i < n; ++i) {
            new (static_cast<char*>(newTs) + i * sizeof(T)) T;
        }
        return static_cast<T*>(newTs);
    }

    /**
     * Version of above that uses a copy constructor to initialize all n items
     * to the same T.
     */
    T* push_back_n(int n, const T& t) {
        SkASSERT(n >= 0);
        void* newTs = this->push_back_raw(n);
        for (int i = 0; i < n; ++i) {
            new (static_cast<char*>(newTs) + i * sizeof(T)) T(t);
        }
        return static_cast<T*>(newTs);
    }

    /**
     * Version of above that uses a copy constructor to initialize the n items
     * to separate T values.
     */
    T* push_back_n(int n, const T t[]) {
        SkASSERT(n >= 0);
        this->checkRealloc(n, kGrowing);
        for (int i = 0; i < n; ++i) {
            new (fItemArray + fCount + i) T(t[i]);
        }
        fCount += n;
        return fItemArray + fCount - n;
    }

    /**
     * Version of above that uses the move constructor to set n items.
     */
    T* move_back_n(int n, T* t) {
        SkASSERT(n >= 0);
        this->checkRealloc(n, kGrowing);
        for (int i = 0; i < n; ++i) {
            new (fItemArray + fCount + i) T(std::move(t[i]));
        }
        fCount += n;
        return fItemArray + fCount - n;
    }

    /**
     * Removes the last element. Not safe to call when count() == 0.
     */
    void pop_back() {
        SkASSERT(fCount > 0);
        --fCount;
        fItemArray[fCount].~T();
        this->checkRealloc(0, kShrinking);
    }

    /**
     * Removes the last n elements. Not safe to call when count() < n.
     */
    void pop_back_n(int n) {
        SkASSERT(n >= 0);
        SkASSERT(this->count() >= n);
        fCount -= n;
        for (int i = 0; i < n; ++i) {
            fItemArray[fCount + i].~T();
        }
        this->checkRealloc(0, kShrinking);
    }

    /**
     * Pushes or pops from the back to resize. Pushes will be default
     * initialized.
     */
    void resize_back(int newCount) {
        SkASSERT(newCount >= 0);

        if (newCount > this->count()) {
            this->push_back_n(newCount - fCount);
        } else if (newCount < this->count()) {
            this->pop_back_n(fCount - newCount);
        }
    }

    /** Swaps the contents of this array with that array. Does a pointer swap if possible,
        otherwise copies the T values. */
    void swap(SkTArray& that) {
        using std::swap;
        if (this == &that) {
            return;
        }
        if (fOwnMemory && that.fOwnMemory) {
            swap(fItemArray, that.fItemArray);

            auto count = fCount;
            fCount = that.fCount;
            that.fCount = count;

            auto allocCount = fAllocCount;
            fAllocCount = that.fAllocCount;
            that.fAllocCount = allocCount;
        } else {
            // This could be more optimal...
            SkTArray copy(std::move(that));
            that = std::move(*this);
            *this = std::move(copy);
        }
    }

    T* begin() {
        return fItemArray;
    }
    const T* begin() const {
        return fItemArray;
    }
    T* end() {
        return fItemArray ? fItemArray + fCount : nullptr;
    }
    const T* end() const {
        return fItemArray ? fItemArray + fCount : nullptr;
    }
    T* data() { return fItemArray; }
    const T* data() const { return fItemArray; }
    size_t size() const { return (size_t)fCount; }
    void resize(size_t count) { this->resize_back((int)count); }

    /**
     * Get the i^th element.
     */
    T& operator[] (int i) {
        SkASSERT(i < this->count());
        SkASSERT(i >= 0);
        return fItemArray[i];
    }

    const T& operator[] (int i) const {
        SkASSERT(i < this->count());
        SkASSERT(i >= 0);
        return fItemArray[i];
    }

    T& at(int i) { return (*this)[i]; }
    const T& at(int i) const { return (*this)[i]; }

    /**
     * equivalent to operator[](0)
     */
    T& front() { SkASSERT(fCount > 0); return fItemArray[0];}

    const T& front() const { SkASSERT(fCount > 0); return fItemArray[0];}

    /**
     * equivalent to operator[](count() - 1)
     */
    T& back() { SkASSERT(fCount); return fItemArray[fCount - 1];}

    const T& back() const { SkASSERT(fCount > 0); return fItemArray[fCount - 1];}

    /**
     * equivalent to operator[](count()-1-i)
     */
    T& fromBack(int i) {
        SkASSERT(i >= 0);
        SkASSERT(i < this->count());
        return fItemArray[fCount - i - 1];
    }

    const T& fromBack(int i) const {
        SkASSERT(i >= 0);
        SkASSERT(i < this->count());
        return fItemArray[fCount - i - 1];
    }

    bool operator==(const SkTArray<T, MEM_MOVE>& right) const {
        int leftCount = this->count();
        if (leftCount != right.count()) {
            return false;
        }
        for (int index = 0; index < leftCount; ++index) {
            if (fItemArray[index] != right.fItemArray[index]) {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const SkTArray<T, MEM_MOVE>& right) const {
        return !(*this == right);
    }

    int capacity() const {
        return fAllocCount;
    }

protected:
    /**
     * Creates an empty array that will use the passed storage block until it
     * is insufficiently large to hold the entire array.
     */
    template <int N>
    SkTArray(SkAlignedSTStorage<N,T>* storage) {
        this->initWithPreallocatedStorage(0, storage->get(), N);
    }

    /**
     * Copy a C array, using preallocated storage if preAllocCount >=
     * count. Otherwise storage will only be used when array shrinks
     * to fit.
     */
    template <int N>
    SkTArray(const T* array, int count, SkAlignedSTStorage<N,T>* storage) {
        this->initWithPreallocatedStorage(count, storage->get(), N);
        this->copy(array);
    }

private:
    // We disable Control-Flow Integrity sanitization (go/cfi) when updating the item-array buffer.
    // CFI flags this code as dangerous because we are casting `buffer` to a T* while the buffer's
    // contents might still be uninitialized memory. When T has a vtable, this is especially risky
    // because we could hypothetically access a virtual method on fItemArray and jump to an
    // unpredictable location in memory. Of course, SkTArray won't actually use fItemArray in this
    // way, and we don't want to construct a T before the user requests one. There's no real risk
    // here, so disable CFI when doing these casts.
// -- GODOT start --
    //SK_ATTRIBUTE(no_sanitize("cfi"))
// -- GODOT end --
    void setItemArray(void* buffer) {
        fItemArray = (T*)buffer;
    }

// -- GODOT start --
    //SK_ATTRIBUTE(no_sanitize("cfi"))
// -- GODOT end --
    void recreateItemArray() {
        T* newItemArray = (T*)sk_malloc_throw((size_t)fAllocCount, sizeof(T));
        this->move(newItemArray);
        if (fOwnMemory) {
            sk_free(fItemArray);
        }
        fItemArray = newItemArray;
    }

    void init(int count) {
        fCount = SkToU32(count);
        if (!count) {
            fAllocCount = 0;
            fItemArray = nullptr;
        } else {
            fAllocCount = SkToU32(std::max(count, kMinHeapAllocCount));
            this->setItemArray(sk_malloc_throw((size_t)fAllocCount, sizeof(T)));
        }
        fOwnMemory = true;
        fReserved = false;
    }

    void initWithPreallocatedStorage(int count, void* preallocStorage, int preallocCount) {
        SkASSERT(count >= 0);
        SkASSERT(preallocCount > 0);
        SkASSERT(preallocStorage);
        fCount = SkToU32(count);
        fItemArray = nullptr;
        fReserved = false;
        if (count > preallocCount) {
            fAllocCount = SkToU32(std::max(count, kMinHeapAllocCount));
            this->setItemArray(sk_malloc_throw(fAllocCount, sizeof(T)));
            fOwnMemory = true;
        } else {
            fAllocCount = SkToU32(preallocCount);
            this->setItemArray(preallocStorage);
            fOwnMemory = false;
        }
    }

    /** In the following move and copy methods, 'dst' is assumed to be uninitialized raw storage.
     *  In the following move methods, 'src' is destroyed leaving behind uninitialized raw storage.
     */
    void copy(const T* src) {
        // Some types may be trivially copyable, in which case we *could* use memcopy; but
        // MEM_MOVE == true implies that the type is trivially movable, and not necessarily
        // trivially copyable (think sk_sp<>).  So short of adding another template arg, we
        // must be conservative and use copy construction.
        for (int i = 0; i < this->count(); ++i) {
            new (fItemArray + i) T(src[i]);
        }
    }

    template <bool E = MEM_MOVE> std::enable_if_t<E, void> move(int dst, int src) {
        memcpy(&fItemArray[dst], &fItemArray[src], sizeof(T));
    }
    template <bool E = MEM_MOVE> std::enable_if_t<E, void> move(void* dst) {
        sk_careful_memcpy(dst, fItemArray, fCount * sizeof(T));
    }

    template <bool E = MEM_MOVE> std::enable_if_t<!E, void> move(int dst, int src) {
        new (&fItemArray[dst]) T(std::move(fItemArray[src]));
        fItemArray[src].~T();
    }
    template <bool E = MEM_MOVE> std::enable_if_t<!E, void> move(void* dst) {
        for (int i = 0; i < this->count(); ++i) {
            new (static_cast<char*>(dst) + sizeof(T) * (size_t)i) T(std::move(fItemArray[i]));
            fItemArray[i].~T();
        }
    }

    static constexpr int kMinHeapAllocCount = 8;

    // Helper function that makes space for n objects, adjusts the count, but does not initialize
    // the new objects.
    void* push_back_raw(int n) {
        this->checkRealloc(n, kGrowing);
        void* ptr = fItemArray + fCount;
        fCount += n;
        return ptr;
    }

    void checkRealloc(int delta, ReallocType reallocType) {
        SkASSERT(fCount >= 0);
        SkASSERT(fAllocCount >= 0);
        SkASSERT(-delta <= this->count());

        // Move into 64bit math temporarily, to avoid local overflows
        int64_t newCount = fCount + delta;

        // We allow fAllocCount to be in the range [newCount, 3*newCount]. We also never shrink
        // when we're currently using preallocated memory, would allocate less than
        // kMinHeapAllocCount, or a reserve count was specified that has yet to be exceeded.
        bool mustGrow = newCount > fAllocCount;
        bool shouldShrink = fAllocCount > 3 * newCount && fOwnMemory && !fReserved;
        if (!mustGrow && !shouldShrink) {
            return;
        }

        int64_t newAllocCount = newCount;
        if (reallocType != kExactFit) {
            // Whether we're growing or shrinking, leave at least 50% extra space for future growth.
            newAllocCount += ((newCount + 1) >> 1);
            // Align the new allocation count to kMinHeapAllocCount.
            static_assert(SkIsPow2(kMinHeapAllocCount), "min alloc count not power of two.");
            newAllocCount = (newAllocCount + (kMinHeapAllocCount - 1)) & ~(kMinHeapAllocCount - 1);
        }

        // At small sizes the old and new alloc count can both be kMinHeapAllocCount.
        if (newAllocCount == fAllocCount) {
            return;
        }

        fAllocCount = SkToU32(Sk64_pin_to_s32(newAllocCount));
        SkASSERT(fAllocCount >= newCount);
        this->recreateItemArray();
        fOwnMemory = true;
        fReserved = false;
    }

    T* fItemArray;
    uint32_t fOwnMemory  :  1;
    uint32_t fCount      : 31;
    uint32_t fReserved   :  1;
    uint32_t fAllocCount : 31;
};

template <typename T, bool M> static inline void swap(SkTArray<T, M>& a, SkTArray<T, M>& b) {
    a.swap(b);
}

template<typename T, bool MEM_MOVE> constexpr int SkTArray<T, MEM_MOVE>::kMinHeapAllocCount;

/**
 * Subclass of SkTArray that contains a preallocated memory block for the array.
 */
template <int N, typename T, bool MEM_MOVE = false>
class SkSTArray : private SkAlignedSTStorage<N,T>, public SkTArray<T, MEM_MOVE> {
private:
    using STORAGE   = SkAlignedSTStorage<N,T>;
    using INHERITED = SkTArray<T, MEM_MOVE>;

public:
    SkSTArray()
        : STORAGE{}, INHERITED(static_cast<STORAGE*>(this)) {}

    SkSTArray(const T* array, int count)
        : STORAGE{}, INHERITED(array, count, static_cast<STORAGE*>(this)) {}

    SkSTArray(std::initializer_list<T> data)
        : SkSTArray(data.begin(), data.size()) {}

    explicit SkSTArray(int reserveCount)
        : SkSTArray() {
        this->reserve_back(reserveCount);
    }

    SkSTArray         (const SkSTArray&  that) : SkSTArray() { *this = that; }
    explicit SkSTArray(const INHERITED&  that) : SkSTArray() { *this = that; }
    SkSTArray         (      SkSTArray&& that) : SkSTArray() { *this = std::move(that); }
    explicit SkSTArray(      INHERITED&& that) : SkSTArray() { *this = std::move(that); }

    SkSTArray& operator=(const SkSTArray& that) {
        INHERITED::operator=(that);
        return *this;
    }
    SkSTArray& operator=(const INHERITED& that) {
        INHERITED::operator=(that);
        return *this;
    }

    SkSTArray& operator=(SkSTArray&& that) {
        INHERITED::operator=(std::move(that));
        return *this;
    }
    SkSTArray& operator=(INHERITED&& that) {
        INHERITED::operator=(std::move(that));
        return *this;
    }
};

#endif
