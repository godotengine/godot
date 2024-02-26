/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTLazy_DEFINED
#define SkTLazy_DEFINED

#include "include/core/SkTypes.h"
#include <optional>

/**
 *  Efficient way to defer allocating/initializing a class until it is needed
 *  (if ever).
 */
template <typename T> class SkTLazy {
public:
    SkTLazy() = default;
    explicit SkTLazy(const T* src) : fValue(src ? std::optional<T>(*src) : std::nullopt) {}
    SkTLazy(const SkTLazy& that) : fValue(that.fValue) {}
    SkTLazy(SkTLazy&& that) : fValue(std::move(that.fValue)) {}

    ~SkTLazy() = default;

    SkTLazy& operator=(const SkTLazy& that) {
        fValue = that.fValue;
        return *this;
    }

    SkTLazy& operator=(SkTLazy&& that) {
        fValue = std::move(that.fValue);
        return *this;
    }

    /**
     *  Return a pointer to an instance of the class initialized with 'args'.
     *  If a previous instance had been initialized (either from init() or
     *  set()) it will first be destroyed, so that a freshly initialized
     *  instance is always returned.
     */
    template <typename... Args> T* init(Args&&... args) {
        fValue.emplace(std::forward<Args>(args)...);
        return this->get();
    }

    /**
     *  Copy src into this, and return a pointer to a copy of it. Note this
     *  will always return the same pointer, so if it is called on a lazy that
     *  has already been initialized, then this will copy over the previous
     *  contents.
     */
    T* set(const T& src) {
        fValue = src;
        return this->get();
    }

    T* set(T&& src) {
        fValue = std::move(src);
        return this->get();
    }

    /**
     * Destroy the lazy object (if it was created via init() or set())
     */
    void reset() {
        fValue.reset();
    }

    /**
     *  Returns true if a valid object has been initialized in the SkTLazy,
     *  false otherwise.
     */
    bool isValid() const { return fValue.has_value(); }

    /**
     * Returns the object. This version should only be called when the caller
     * knows that the object has been initialized.
     */
    T* get() {
        SkASSERT(fValue.has_value());
// -- GODOT start --
        return &(*fValue);
// -- GODOT end --
    }
    const T* get() const {
        SkASSERT(fValue.has_value());
// -- GODOT start --
        return &(*fValue);
// -- GODOT end --
    }

    T* operator->() { return this->get(); }
    const T* operator->() const { return this->get(); }

    T& operator*() {
        SkASSERT(fValue.has_value());
        return *fValue;
    }
    const T& operator*() const {
        SkASSERT(fValue.has_value());
        return *fValue;
    }

    /**
     * Like above but doesn't assert if object isn't initialized (in which case
     * nullptr is returned).
     */
    const T* getMaybeNull() const { return fValue.has_value() ? this->get() : nullptr; }

private:
    std::optional<T> fValue;
};

/**
 * A helper built on top of std::optional to do copy-on-first-write. The object is initialized
 * with a const pointer but provides a non-const pointer accessor. The first time the
 * accessor is called (if ever) the object is cloned.
 *
 * In the following example at most one copy of constThing is made:
 *
 * SkTCopyOnFirstWrite<Thing> thing(&constThing);
 * ...
 * function_that_takes_a_const_thing_ptr(thing); // constThing is passed
 * ...
 * if (need_to_modify_thing()) {
 *    thing.writable()->modifyMe(); // makes a copy of constThing
 * }
 * ...
 * x = thing->readSomething();
 * ...
 * if (need_to_modify_thing_now()) {
 *    thing.writable()->changeMe(); // makes a copy of constThing if we didn't call modifyMe()
 * }
 *
 * consume_a_thing(thing); // could be constThing or a modified copy.
 */
template <typename T>
class SkTCopyOnFirstWrite {
public:
    explicit SkTCopyOnFirstWrite(const T& initial) : fObj(&initial) {}

    explicit SkTCopyOnFirstWrite(const T* initial) : fObj(initial) {}

    // Constructor for delayed initialization.
    SkTCopyOnFirstWrite() : fObj(nullptr) {}

    SkTCopyOnFirstWrite(const SkTCopyOnFirstWrite&  that) { *this = that;            }
    SkTCopyOnFirstWrite(      SkTCopyOnFirstWrite&& that) { *this = std::move(that); }

    SkTCopyOnFirstWrite& operator=(const SkTCopyOnFirstWrite& that) {
        fLazy = that.fLazy;
        fObj  = fLazy.has_value() ? &fLazy.value() : that.fObj;
        return *this;
    }

    SkTCopyOnFirstWrite& operator=(SkTCopyOnFirstWrite&& that) {
        fLazy = std::move(that.fLazy);
        fObj  = fLazy.has_value() ? &fLazy.value() : that.fObj;
        return *this;
    }

    // Should only be called once, and only if the default constructor was used.
    void init(const T& initial) {
        SkASSERT(!fObj);
        SkASSERT(!fLazy.has_value());
        fObj = &initial;
    }

    // If not already initialized, in-place instantiates the writable object
    template <typename... Args>
    void initIfNeeded(Args&&... args) {
        if (!fObj) {
            SkASSERT(!fLazy.has_value());
            fObj = fLazy.emplace(std::forward<Args>(args)...);
        }
    }

    /**
     * Returns a writable T*. The first time this is called the initial object is cloned.
     */
    T* writable() {
        SkASSERT(fObj);
        if (!fLazy.has_value()) {
            fLazy = *fObj;
            fObj = &fLazy.value();
        }
        return &fLazy.value();
    }

    const T* get() const { return fObj; }

    /**
     * Operators for treating this as though it were a const pointer.
     */

    const T *operator->() const { return fObj; }

    operator const T*() const { return fObj; }

    const T& operator *() const { return *fObj; }

private:
    const T*         fObj;
    std::optional<T> fLazy;
};

#endif
