///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// ErrorCodes.h                                                              //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides error code values for the DirectX compiler and tools.            //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once


// Redeclare some macros to not depend on winerror.h
#define DXC_SEVERITY_ERROR      1
#define DXC_MAKE_HRESULT(sev,fac,code) \
    ((HRESULT) (((unsigned long)(sev)<<31) | ((unsigned long)(fac)<<16) | ((unsigned long)(code))) )

#define HRESULT_IS_WIN32ERR(hr) ((HRESULT)(hr & 0xFFFF0000) == MAKE_HRESULT(SEVERITY_ERROR, FACILITY_WIN32, 0))
#define HRESULT_AS_WIN32ERR(hr) (HRESULT_CODE(hr))

// Error codes from C libraries (0n150) - 0x8096xxxx
#define FACILITY_ERRNO          (0x96)
#define HRESULT_FROM_ERRNO(x)   MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_ERRNO,(x))

// Error codes from DXC libraries (0n170) - 0x8013xxxx
#define FACILITY_DXC             (0xAA)

// 0x00000000 - The operation succeeded.
#define DXC_S_OK                        0 // _HRESULT_TYPEDEF_(0x00000000L)

// 0x80AA0001 - The operation failed because overlapping semantics were found.
#define DXC_E_OVERLAPPING_SEMANTICS                   DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0001))

// 0x80AA0002 - The operation failed because multiple depth semantics were found.
#define DXC_E_MULTIPLE_DEPTH_SEMANTICS                DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0002))

// 0x80AA0003 - Input file is too large.
#define DXC_E_INPUT_FILE_TOO_LARGE                    DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0003))

// 0x80AA0004 - Error parsing DXBC container.
#define DXC_E_INCORRECT_DXBC                          DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0004))

// 0x80AA0005 - Error parsing DXBC bytecode.
#define DXC_E_ERROR_PARSING_DXBC_BYTECODE             DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0005))

// 0x80AA0006 - Data is too large.
#define DXC_E_DATA_TOO_LARGE                          DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0006))

// 0x80AA0007 - Incompatible converter options.
#define DXC_E_INCOMPATIBLE_CONVERTER_OPTIONS          DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0007))

// 0x80AA0008 - Irreducible control flow graph.
#define DXC_E_IRREDUCIBLE_CFG                         DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0008))

// 0x80AA0009 - IR verification error.
#define DXC_E_IR_VERIFICATION_FAILED                  DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0009))

// 0x80AA000A - Scope-nested control flow recovery failed.
#define DXC_E_SCOPE_NESTED_FAILED                     DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000A))

// 0x80AA000B - Operation is not supported.
#define DXC_E_NOT_SUPPORTED                           DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000B))

// 0x80AA000C - Unable to encode string.
#define DXC_E_STRING_ENCODING_FAILED                  DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000C))

// 0x80AA000D - DXIL container is invalid.
#define DXC_E_CONTAINER_INVALID                       DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000D))

// 0x80AA000E - DXIL container is missing the DXIL part.
#define DXC_E_CONTAINER_MISSING_DXIL                  DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000E))

// 0x80AA000F - Unable to parse DxilModule metadata.
#define DXC_E_INCORRECT_DXIL_METADATA                 DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x000F))

// 0x80AA0010 - Error parsing DDI signature.
#define DXC_E_INCORRECT_DDI_SIGNATURE                 DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0010))

// 0x80AA0011 - Duplicate part exists in dxil container.
#define DXC_E_DUPLICATE_PART                          DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0011))

// 0x80AA0012 - Error finding part in dxil container.
#define DXC_E_MISSING_PART                            DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0012))

// 0x80AA0013 - Malformed DXIL Container.
#define DXC_E_MALFORMED_CONTAINER                     DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0013))

// 0x80AA0014 - Incorrect Root Signature for shader.
#define DXC_E_INCORRECT_ROOT_SIGNATURE                DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0014))

// 0X80AA0015 - DXIL container is missing DebugInfo part.
#define DXC_E_CONTAINER_MISSING_DEBUG                 DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0015))

// 0X80AA0016 - Unexpected failure in macro expansion.
#define DXC_E_MACRO_EXPANSION_FAILURE                 DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0016))

// 0X80AA0017 - DXIL optimization pass failed.
#define DXC_E_OPTIMIZATION_FAILED                     DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0017))

// 0X80AA0018 - General internal error.
#define DXC_E_GENERAL_INTERNAL_ERROR                  DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0018))

// 0X80AA0019 - Abort compilation error.
#define DXC_E_ABORT_COMPILATION_ERROR                 DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x0019))

// 0X80AA001A - Error in extension mechanism.
#define DXC_E_EXTENSION_ERROR                         DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x001A))

// 0X80AA001B - LLVM Fatal Error
#define DXC_E_LLVM_FATAL_ERROR                         DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x001B))

// 0X80AA001C - LLVM Unreachable code
#define DXC_E_LLVM_UNREACHABLE                         DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x001C))

// 0X80AA001D - LLVM Cast Failure
#define DXC_E_LLVM_CAST_ERROR                         DXC_MAKE_HRESULT(DXC_SEVERITY_ERROR,FACILITY_DXC,(0x001D))
