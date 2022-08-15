///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxcerror.h                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides definition of error codes.                                        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_ERRORS__
#define __DXC_ERRORS__

#ifndef FACILITY_GRAPHICS
#define FACILITY_GRAPHICS 36
#endif

#define DXC_EXCEPTION_CODE(name, status)                                 \
    static constexpr DWORD EXCEPTION_##name =                 \
    (0xc0000000u | (FACILITY_GRAPHICS << 16) | (0xff00u | (status & 0xffu)));

DXC_EXCEPTION_CODE(LOAD_LIBRARY_FAILED, 0x00u)
DXC_EXCEPTION_CODE(NO_HMODULE,          0x01u)
DXC_EXCEPTION_CODE(GET_PROC_FAILED,     0x02u)

#undef DXC_EXCEPTION_CODE

#endif
