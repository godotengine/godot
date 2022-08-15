//===- WinIncludes.h --------------------------------------------*- C++ -*-===//
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// WinIncludes.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#ifdef _MSC_VER

// mingw-w64 tends to define it as 0x0502 in its headers.
#undef _WIN32_WINNT
#undef _WIN32_IE

// Require at least Windows 7 (Updated from XP)
#define _WIN32_WINNT 0x0601
#define _WIN32_IE    0x0800 // MinGW at it again.

#define NOATOM 1
#define NOGDICAPMASKS 1
#define NOMETAFILE 1
#define NOMINMAX 1
#define NOOPENFILE 1
#define NORASTEROPS 1
#define NOSCROLL 1
#define NOSOUND 1
#define NOSYSMETRICS 1
#define NOWH 1
#define NOCOMM 1
#define NOKANJI 1
#define NOCRYPT 1
#define NOMCX 1
#define WIN32_LEAN_AND_MEAN 1
#define VC_EXTRALEAN 1
#define NONAMELESSSTRUCT 1

#include <windows.h>
#include <unknwn.h>
#include <atlbase.h> // atlbase.h needs to come before strsafe.h
#include <strsafe.h>
#include <intsafe.h>
#include <ObjIdl.h>

// Support older atlbase.h if needed
#ifndef _ATL_DECLSPEC_ALLOCATOR
#define _ATL_DECLSPEC_ALLOCATOR
#endif

/// Swap two ComPtr classes.
template <class T> void swap(CComHeapPtr<T> &a, CComHeapPtr<T> &b) {
  T *c(a.m_pData);
  a.m_pData = b.m_pData;
  b.m_pData = c;
}

#else // _MSC_VER

#include "dxc/Support/WinAdapter.h"

#ifdef __cplusplus
// Define operator overloads to enable bit operations on enum values that are
// used to define flags. Use DEFINE_ENUM_FLAG_OPERATORS(YOUR_TYPE) to enable these
// operators on YOUR_TYPE.
extern "C++" {
    template <size_t S>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE;

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<1>
    {
        typedef int8_t type;
    };

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<2>
    {
        typedef int16_t type;
    };

    template <>
    struct _ENUM_FLAG_INTEGER_FOR_SIZE<4>
    {
        typedef int32_t type;
    };

    // used as an approximation of std::underlying_type<T>
    template <class T>
    struct _ENUM_FLAG_SIZED_INTEGER
    {
        typedef typename _ENUM_FLAG_INTEGER_FOR_SIZE<sizeof(T)>::type type;
    };

}
#define DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE) \
extern "C++" { \
inline ENUMTYPE operator | (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) | ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator |= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) |= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE operator & (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) & ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator &= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) &= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE operator ~ (ENUMTYPE a) { return ENUMTYPE(~((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a)); } \
inline ENUMTYPE operator ^ (ENUMTYPE a, ENUMTYPE b) { return ENUMTYPE(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)a) ^ ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
inline ENUMTYPE &operator ^= (ENUMTYPE &a, ENUMTYPE b) { return (ENUMTYPE &)(((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type &)a) ^= ((_ENUM_FLAG_SIZED_INTEGER<ENUMTYPE>::type)b)); } \
}
#else
#define DEFINE_ENUM_FLAG_OPERATORS(ENUMTYPE) // NOP, C allows these operators.
#endif

#endif // _MSC_VER
