/*
******************************************************************************
* Copyright (C) 2014, International Business Machines
* Corporation and others.  All Rights Reserved.
******************************************************************************
* sharedobject.cpp
*/
#include "sharedobject.h"

U_NAMESPACE_BEGIN
SharedObject::~SharedObject() {}

void
SharedObject::addRef() const {
    umtx_atomic_inc(&totalRefCount);
}

void
SharedObject::removeRef() const {
    if(umtx_atomic_dec(&totalRefCount) == 0) {
        delete this;
    }
}

void
SharedObject::addSoftRef() const {
    addRef();
    umtx_atomic_inc(&softRefCount);
}

void
SharedObject::removeSoftRef() const {
    umtx_atomic_dec(&softRefCount);
    removeRef();
}

UBool
SharedObject::allSoftReferences() const {
    return umtx_loadAcquire(totalRefCount) == umtx_loadAcquire(softRefCount);
}

int32_t
SharedObject::getRefCount() const {
    return umtx_loadAcquire(totalRefCount);
}

int32_t
SharedObject::getSoftRefCount() const {
    return umtx_loadAcquire(softRefCount);
}

void
SharedObject::deleteIfZeroRefCount() const {
    if(getRefCount() == 0) {
        delete this;
    }
}

U_NAMESPACE_END
