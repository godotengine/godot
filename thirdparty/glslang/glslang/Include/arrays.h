//
// Copyright (C) 2002-2005  3Dlabs Inc. Ltd.
// Copyright (C) 2012-2013 LunarG, Inc.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//    Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
//    Redistributions in binary form must reproduce the above
//    copyright notice, this list of conditions and the following
//    disclaimer in the documentation and/or other materials provided
//    with the distribution.
//
//    Neither the name of 3Dlabs Inc. Ltd. nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//

//
// Implement types for tracking GLSL arrays, arrays of arrays, etc.
//

#ifndef _ARRAYS_INCLUDED
#define _ARRAYS_INCLUDED

#include <algorithm>

namespace glslang {

// This is used to mean there is no size yet (unsized), it is waiting to get a size from somewhere else.
const int UnsizedArraySize = 0;

class TIntermTyped;
extern bool SameSpecializationConstants(TIntermTyped*, TIntermTyped*);

// Specialization constants need both a nominal size and a node that defines
// the specialization constant being used.  Array types are the same when their
// size and specialization constant nodes are the same.
struct TArraySize {
    unsigned int size;
    TIntermTyped* node;  // nullptr means no specialization constant node
    bool operator==(const TArraySize& rhs) const
    {
        if (size != rhs.size)
            return false;
        if (node == nullptr || rhs.node == nullptr)
            return node == rhs.node;

        return SameSpecializationConstants(node, rhs.node);
    }
    bool operator!=(const TArraySize& rhs) const { return !(*this == rhs); }
};

//
// TSmallArrayVector is used as the container for the set of sizes in TArraySizes.
// It has generic-container semantics, while TArraySizes has array-of-array semantics.
// That is, TSmallArrayVector should be more focused on mechanism and TArraySizes on policy.
//
struct TSmallArrayVector {
    //
    // TODO: memory: TSmallArrayVector is intended to be smaller.
    // Almost all arrays could be handled by two sizes each fitting
    // in 16 bits, needing a real vector only in the cases where there
    // are more than 3 sizes or a size needing more than 16 bits.
    //
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TSmallArrayVector() : sizes(nullptr) { }
    virtual ~TSmallArrayVector() { dealloc(); }

    // For breaking into two non-shared copies, independently modifiable.
    TSmallArrayVector& operator=(const TSmallArrayVector& from)
    {
        if (from.sizes == nullptr)
            sizes = nullptr;
        else {
            alloc();
            *sizes = *from.sizes;
        }

        return *this;
    }

    int size() const
    {
        if (sizes == nullptr)
            return 0;
        return (int)sizes->size();
    }

    unsigned int frontSize() const
    {
        assert(sizes != nullptr && sizes->size() > 0);
        return sizes->front().size;
    }

    TIntermTyped* frontNode() const
    {
        assert(sizes != nullptr && sizes->size() > 0);
        return sizes->front().node;
    }

    void changeFront(unsigned int s)
    {
        assert(sizes != nullptr);
        // this should only happen for implicitly sized arrays, not specialization constants
        assert(sizes->front().node == nullptr);
        sizes->front().size = s;
    }

    void push_back(unsigned int e, TIntermTyped* n)
    {
        alloc();
        TArraySize pair = { e, n };
        sizes->push_back(pair);
    }

    void push_back(const TSmallArrayVector& newDims)
    {
        alloc();
        sizes->insert(sizes->end(), newDims.sizes->begin(), newDims.sizes->end());
    }

    void pop_front()
    {
        assert(sizes != nullptr && sizes->size() > 0);
        if (sizes->size() == 1)
            dealloc();
        else
            sizes->erase(sizes->begin());
    }

    void pop_back()
    {
        assert(sizes != nullptr && sizes->size() > 0);
        if (sizes->size() == 1)
            dealloc();
        else
            sizes->resize(sizes->size() - 1);
    }

    // 'this' should currently not be holding anything, and copyNonFront
    // will make it hold a copy of all but the first element of rhs.
    // (This would be useful for making a type that is dereferenced by
    // one dimension.)
    void copyNonFront(const TSmallArrayVector& rhs)
    {
        assert(sizes == nullptr);
        if (rhs.size() > 1) {
            alloc();
            sizes->insert(sizes->begin(), rhs.sizes->begin() + 1, rhs.sizes->end());
        }
    }

    unsigned int getDimSize(int i) const
    {
        assert(sizes != nullptr && (int)sizes->size() > i);
        return (*sizes)[i].size;
    }

    void setDimSize(int i, unsigned int size) const
    {
        assert(sizes != nullptr && (int)sizes->size() > i);
        assert((*sizes)[i].node == nullptr);
        (*sizes)[i].size = size;
    }

    TIntermTyped* getDimNode(int i) const
    {
        assert(sizes != nullptr && (int)sizes->size() > i);
        return (*sizes)[i].node;
    }

    bool operator==(const TSmallArrayVector& rhs) const
    {
        if (sizes == nullptr && rhs.sizes == nullptr)
            return true;
        if (sizes == nullptr || rhs.sizes == nullptr)
            return false;
        return *sizes == *rhs.sizes;
    }
    bool operator!=(const TSmallArrayVector& rhs) const { return ! operator==(rhs); }

    const TArraySize& operator[](int index) const
    {
        assert(sizes && index < (int)sizes->size());
        return (*sizes)[index];
    }

protected:
    TSmallArrayVector(const TSmallArrayVector&);

    void alloc()
    {
        if (sizes == nullptr)
            sizes = new TVector<TArraySize>;
    }
    void dealloc()
    {
        delete sizes;
        sizes = nullptr;
    }

    TVector<TArraySize>* sizes; // will either hold such a pointer, or in the future, hold the two array sizes
};

//
// Represent an array, or array of arrays, to arbitrary depth.  This is not
// done through a hierarchy of types in a type tree, rather all contiguous arrayness
// in the type hierarchy is localized into this single cumulative object.
//
// The arrayness in TTtype is a pointer, so that it can be non-allocated and zero
// for the vast majority of types that are non-array types.
//
// Order Policy: these are all identical:
//  - left to right order within a contiguous set of ...[..][..][..]... in the source language
//  - index order 0, 1, 2, ... within the 'sizes' member below
//  - outer-most to inner-most
//
struct TArraySizes {
    POOL_ALLOCATOR_NEW_DELETE(GetThreadPoolAllocator())

    TArraySizes() : implicitArraySize(0), implicitlySized(true), variablyIndexed(false){ }

    // For breaking into two non-shared copies, independently modifiable.
    TArraySizes& operator=(const TArraySizes& from)
    {
        implicitArraySize = from.implicitArraySize;
        variablyIndexed = from.variablyIndexed;
        sizes = from.sizes;
        implicitlySized = from.implicitlySized;

        return *this;
    }

    // translate from array-of-array semantics to container semantics
    int getNumDims() const { return sizes.size(); }
    int getDimSize(int dim) const { return sizes.getDimSize(dim); }
    TIntermTyped* getDimNode(int dim) const { return sizes.getDimNode(dim); }
    void setDimSize(int dim, int size) { sizes.setDimSize(dim, size); }
    int getOuterSize() const { return sizes.frontSize(); }
    TIntermTyped* getOuterNode() const { return sizes.frontNode(); }
    int getCumulativeSize() const
    {
        int size = 1;
        for (int d = 0; d < sizes.size(); ++d) {
            // this only makes sense in paths that have a known array size
            assert(sizes.getDimSize(d) != UnsizedArraySize);
            size *= sizes.getDimSize(d);
        }
        return size;
    }
    void addInnerSize() { addInnerSize((unsigned)UnsizedArraySize); }
    void addInnerSize(int s) { addInnerSize((unsigned)s, nullptr); }
    void addInnerSize(int s, TIntermTyped* n) { sizes.push_back((unsigned)s, n); }
    void addInnerSize(TArraySize pair) {
        sizes.push_back(pair.size, pair.node);
        implicitlySized = false;
    }
    void addInnerSizes(const TArraySizes& s) { sizes.push_back(s.sizes); }
    void changeOuterSize(int s) {
        sizes.changeFront((unsigned)s);
        implicitlySized = false;
    }
    int getImplicitSize() const { return implicitArraySize > 0 ? implicitArraySize : 1; }
    void updateImplicitSize(int s) {
        implicitArraySize = (std::max)(implicitArraySize, s);
    }
    bool isInnerUnsized() const
    {
        for (int d = 1; d < sizes.size(); ++d) {
            if (sizes.getDimSize(d) == (unsigned)UnsizedArraySize)
                return true;
        }

        return false;
    }
    bool clearInnerUnsized()
    {
        for (int d = 1; d < sizes.size(); ++d) {
            if (sizes.getDimSize(d) == (unsigned)UnsizedArraySize)
                setDimSize(d, 1);
        }

        return false;
    }
    bool isInnerSpecialization() const
    {
        for (int d = 1; d < sizes.size(); ++d) {
            if (sizes.getDimNode(d) != nullptr)
                return true;
        }

        return false;
    }
    bool isOuterSpecialization()
    {
        return sizes.getDimNode(0) != nullptr;
    }

    bool hasUnsized() const { return getOuterSize() == UnsizedArraySize || isInnerUnsized(); }
    bool isSized() const { return getOuterSize() != UnsizedArraySize; }
    bool isImplicitlySized() const { return implicitlySized; }
    bool isDefaultImplicitlySized() const { return implicitlySized && implicitArraySize == 0; }
    void setImplicitlySized(bool isImplicitSizing) { implicitlySized = isImplicitSizing; }
    void dereference() { sizes.pop_front(); }
    void removeLastSize() { sizes.pop_back(); }
    void copyDereferenced(const TArraySizes& rhs)
    {
        assert(sizes.size() == 0);
        if (rhs.sizes.size() > 1)
            sizes.copyNonFront(rhs.sizes);
    }

    bool sameInnerArrayness(const TArraySizes& rhs) const
    {
        if (sizes.size() != rhs.sizes.size())
            return false;

        for (int d = 1; d < sizes.size(); ++d) {
            if (sizes.getDimSize(d) != rhs.sizes.getDimSize(d) ||
                sizes.getDimNode(d) != rhs.sizes.getDimNode(d))
                return false;
        }

        return true;
    }
    const TArraySize& getArraySize(int index) const
    {
        return sizes[index];
    }

    void setVariablyIndexed() { variablyIndexed = true; }
    bool isVariablyIndexed() const { return variablyIndexed; }

    bool operator==(const TArraySizes& rhs) const { return sizes == rhs.sizes; }
    bool operator!=(const TArraySizes& rhs) const { return sizes != rhs.sizes; }

protected:
    TSmallArrayVector sizes;

    TArraySizes(const TArraySizes&);

    // For tracking maximum referenced compile-time constant index.
    // Applies only to the outer-most dimension. Potentially becomes
    // the implicit size of the array, if not variably indexed and
    // otherwise legal.
    int implicitArraySize;
    bool implicitlySized;
    bool variablyIndexed;  // true if array is indexed with a non compile-time constant
};

} // end namespace glslang

#endif // _ARRAYS_INCLUDED_
