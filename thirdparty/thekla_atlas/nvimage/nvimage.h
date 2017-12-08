// This code is in the public domain -- castanyo@yahoo.es

#pragma once
#ifndef NV_IMAGE_H
#define NV_IMAGE_H

#include "nvcore/nvcore.h"
#include "nvcore/Debug.h" // nvDebugCheck
#include "nvcore/Utils.h" // isPowerOfTwo

// Function linkage
#if NVIMAGE_SHARED
#ifdef NVIMAGE_EXPORTS
#define NVIMAGE_API DLL_EXPORT
#define NVIMAGE_CLASS DLL_EXPORT_CLASS
#else
#define NVIMAGE_API DLL_IMPORT
#define NVIMAGE_CLASS DLL_IMPORT
#endif
#else
#define NVIMAGE_API
#define NVIMAGE_CLASS
#endif


namespace nv {

    // Some utility functions:

    inline uint computeBitPitch(uint w, uint bitsize, uint alignmentInBits)
    {
        nvDebugCheck(isPowerOfTwo(alignmentInBits));

        return ((w * bitsize +  alignmentInBits - 1) / alignmentInBits) * alignmentInBits;
    }

    inline uint computeBytePitch(uint w, uint bitsize, uint alignmentInBytes)
    {
        uint pitch = computeBitPitch(w, bitsize, 8*alignmentInBytes);
        nvDebugCheck((pitch & 7) == 0);

        return (pitch + 7) / 8;
    }


} // nv namespace

#endif // NV_IMAGE_H
