/*
 * Copyright 2020 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/private/SkIDChangeListener.h"

/**
 * Used to be notified when a gen/unique ID is invalidated, typically to preemptively purge
 * associated items from a cache that are no longer reachable. The listener can
 * be marked for deregistration if the cached item is remove before the listener is
 * triggered. This prevents unbounded listener growth when cache items are routinely
 * removed before the gen ID/unique ID is invalidated.
 */

SkIDChangeListener::SkIDChangeListener() : fShouldDeregister(false) {}

SkIDChangeListener::~SkIDChangeListener() = default;

using List = SkIDChangeListener::List;

List::List() = default;

List::~List() {
    // We don't need the mutex. No other thread should have this list while it's being
    // destroyed.
    for (int i = 0; i < fListeners.count(); ++i) {
        if (!fListeners[i]->shouldDeregister()) {
            fListeners[i]->changed();
        }
        fListeners[i]->unref();
    }
}

void List::add(sk_sp<SkIDChangeListener> listener) {
    if (!listener) {
        return;
    }
    SkASSERT(!listener->shouldDeregister());

    SkAutoMutexExclusive lock(fMutex);
    // Clean out any stale listeners before we append the new one.
    for (int i = 0; i < fListeners.count(); ++i) {
        if (fListeners[i]->shouldDeregister()) {
            fListeners[i]->unref();
            fListeners.removeShuffle(i--);  // No need to preserve the order after i.
        }
    }
    *fListeners.append() = listener.release();
}

int List::count() const {
    SkAutoMutexExclusive lock(fMutex);
    return fListeners.count();
}

void List::changed() {
    SkAutoMutexExclusive lock(fMutex);
    for (SkIDChangeListener* listener : fListeners) {
        if (!listener->shouldDeregister()) {
            listener->changed();
        }
        // Listeners get at most one shot, so whether these triggered or not, blow them away.
        listener->unref();
    }
    fListeners.reset();
}

void List::reset() {
    SkAutoMutexExclusive lock(fMutex);
    fListeners.unrefAll();
}
