/*
 * Copyright 2020 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkIDChangeListener_DEFINED
#define SkIDChangeListener_DEFINED

#include "include/core/SkRefCnt.h"
#include "include/private/SkMutex.h"
#include "include/private/SkTDArray.h"

#include <atomic>

/**
 * Used to be notified when a gen/unique ID is invalidated, typically to preemptively purge
 * associated items from a cache that are no longer reachable. The listener can
 * be marked for deregistration if the cached item is remove before the listener is
 * triggered. This prevents unbounded listener growth when cache items are routinely
 * removed before the gen ID/unique ID is invalidated.
 */
class SkIDChangeListener : public SkRefCnt {
public:
    SkIDChangeListener();

    ~SkIDChangeListener() override;

    virtual void changed() = 0;

    /**
     * Mark the listener is no longer needed. It should be removed and changed() should not be
     * called.
     */
    void markShouldDeregister() { fShouldDeregister.store(true, std::memory_order_relaxed); }

    /** Indicates whether markShouldDeregister was called. */
    bool shouldDeregister() { return fShouldDeregister.load(std::memory_order_acquire); }

    /** Manages a list of SkIDChangeListeners. */
    class List {
    public:
        List();

        ~List();

        /**
         * Add a new listener to the list. It must not already be deregistered. Also clears out
         * previously deregistered listeners.
         */
        void add(sk_sp<SkIDChangeListener> listener) SK_EXCLUDES(fMutex);

        /**
         * The number of registered listeners (including deregisterd listeners that are yet-to-be
         * removed.
         */
        int count() const SK_EXCLUDES(fMutex);

        /** Calls changed() on all listeners that haven't been deregistered and resets the list. */
        void changed() SK_EXCLUDES(fMutex);

        /** Resets without calling changed() on the listeners. */
        void reset() SK_EXCLUDES(fMutex);

    private:
        mutable SkMutex fMutex;
        SkTDArray<SkIDChangeListener*> fListeners SK_GUARDED_BY(fMutex);  // pointers are reffed
    };

private:
    std::atomic<bool> fShouldDeregister;
};

#endif
