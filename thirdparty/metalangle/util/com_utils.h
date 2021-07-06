//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// com_utils.h: Utility functions for working with COM objects

#ifndef UTIL_COM_UTILS_H
#define UTIL_COM_UTILS_H

template <typename outType>
inline outType *DynamicCastComObject(IUnknown *object)
{
    outType *outObject = nullptr;
    HRESULT result =
        object->QueryInterface(__uuidof(outType), reinterpret_cast<void **>(&outObject));
    if (SUCCEEDED(result))
    {
        return outObject;
    }
    else
    {
        SafeRelease(outObject);
        return nullptr;
    }
}

#endif  // UTIL_COM_UTILS_H
