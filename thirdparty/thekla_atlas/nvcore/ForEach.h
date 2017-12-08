// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#pragma once
#ifndef NV_CORE_FOREACH_H
#define NV_CORE_FOREACH_H

/*
These foreach macros are very non-standard and somewhat confusing, but I like them.
*/

#include "nvcore.h"


#if NV_CC_CPP11

#define NV_FOREACH(i, container) \
    for (auto i = (container).start(); !(container).isDone(i); (container).advance(i))

#elif NV_CC_GNUC // If typeof is available:

/*
Ideally we would like to write this:

#define NV_FOREACH(i, container) \
    for(decltype(container)::PseudoIndex i((container).start()); !(container).isDone(i); (container).advance(i))

But gcc versions prior to 4.7 required an intermediate type. See:
https://gcc.gnu.org/bugzilla/show_bug.cgi?id=6709
*/

#define NV_FOREACH(i, container) \
    typedef typeof(container) NV_STRING_JOIN2(cont,__LINE__); \
    for(NV_STRING_JOIN2(cont,__LINE__)::PseudoIndex i((container).start()); !(container).isDone(i); (container).advance(i))

#else // If typeof not available:

#define NV_NEED_PSEUDOINDEX_WRAPPER 1

#include <new> // placement new

struct PseudoIndexWrapper {
    template <typename T>
    PseudoIndexWrapper(const T & container) {
        nvStaticCheck(sizeof(typename T::PseudoIndex) <= sizeof(memory));
        new (memory) typename T::PseudoIndex(container.start());
    }
    // PseudoIndex cannot have a dtor!

    template <typename T> typename T::PseudoIndex & operator()(const T * /*container*/) {
        return *reinterpret_cast<typename T::PseudoIndex *>(memory);
    }
    template <typename T> const typename T::PseudoIndex & operator()(const T * /*container*/) const {
        return *reinterpret_cast<const typename T::PseudoIndex *>(memory);
    }

    uint8 memory[4];	// Increase the size if we have bigger enumerators.
};

#define NV_FOREACH(i, container) \
    for(PseudoIndexWrapper i(container); !(container).isDone(i(&(container))); (container).advance(i(&(container))))

#endif

// Declare foreach keyword.
#if !defined NV_NO_USE_KEYWORDS
#   define foreach NV_FOREACH
#   define foreach_index NV_FOREACH
#endif


#endif // NV_CORE_FOREACH_H
