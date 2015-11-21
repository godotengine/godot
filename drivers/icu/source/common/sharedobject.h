/*
******************************************************************************
* Copyright (C) 2014, International Business Machines
* Corporation and others.  All Rights Reserved.
******************************************************************************
* sharedobject.h
*/

#ifndef __SHAREDOBJECT_H__
#define __SHAREDOBJECT_H__


#include "unicode/uobject.h"
#include "umutex.h"

U_NAMESPACE_BEGIN

/**
 * Base class for shared, reference-counted, auto-deleted objects.
 * Subclasses can be immutable.
 * If they are mutable, then they must implement their copy constructor
 * so that copyOnWrite() works.
 *
 * Either stack-allocate, use LocalPointer, or use addRef()/removeRef().
 * Sharing requires reference-counting.
 */
class U_COMMON_API SharedObject : public UObject {
public:
    /** Initializes totalRefCount, softRefCount to 0. */
    SharedObject() : totalRefCount(0), softRefCount(0) {}

    /** Initializes totalRefCount, softRefCount to 0. */
    SharedObject(const SharedObject &other)
        : UObject(other),
          totalRefCount(0),
          softRefCount(0) {}

    virtual ~SharedObject();

    /**
     * Increments the number of references to this object. Thread-safe.
     */
    void addRef() const;

    /**
     * Increments the number of soft references to this object. Thread-safe.
     */
    void addSoftRef() const;

    /**
     * Decrements the number of references to this object. Thread-safe.
     */
    void removeRef() const;

    /**
     * Decrements the number of soft references to this object. Thread-safe.
     */
    void removeSoftRef() const;

    /**
     * Returns the reference counter including soft references.
     * Uses a memory barrier.
     */
    int32_t getRefCount() const;

    /**
     * Returns the count of soft references only. Uses a memory barrier.
     * Used for testing the cache. Regular clients won't need this.
     */
    int32_t getSoftRefCount() const;

    /**
     * If allSoftReferences() == TRUE then this object has only soft
     * references. The converse is not necessarily true.
     */
    UBool allSoftReferences() const;

    /**
     * Deletes this object if it has no references or soft references.
     */
    void deleteIfZeroRefCount() const;

    /**
     * Returns a writable version of ptr.
     * If there is exactly one owner, then ptr itself is returned as a
     *  non-const pointer.
     * If there are multiple owners, then ptr is replaced with a 
     * copy-constructed clone,
     * and that is returned.
     * Returns NULL if cloning failed.
     *
     * T must be a subclass of SharedObject.
     */
    template<typename T>
    static T *copyOnWrite(const T *&ptr) {
        const T *p = ptr;
        if(p->getRefCount() <= 1) { return const_cast<T *>(p); }
        T *p2 = new T(*p);
        if(p2 == NULL) { return NULL; }
        p->removeRef();
        ptr = p2;
        p2->addRef();
        return p2;
    }

    /**
     * Makes dest an owner of the object pointed to by src while adjusting
     * reference counts and deleting the previous object dest pointed to
     * if necessary. Before this call is made, dest must either be NULL or
     * be included in the reference count of the object it points to. 
     *
     * T must be a subclass of SharedObject.
     */
    template<typename T>
    static void copyPtr(const T *src, const T *&dest) {
        if(src != dest) {
            if(dest != NULL) { dest->removeRef(); }
            dest = src;
            if(src != NULL) { src->addRef(); }
        }
    }

    /**
     * Equivalent to copyPtr(NULL, dest).
     */
    template<typename T>
    static void clearPtr(const T *&ptr) {
        if (ptr != NULL) {
            ptr->removeRef();
            ptr = NULL;
        }
    }

private:
    mutable u_atomic_int32_t totalRefCount;
    mutable u_atomic_int32_t softRefCount;
};

U_NAMESPACE_END

#endif
