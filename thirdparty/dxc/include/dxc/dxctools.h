///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// dxctools.h                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Provides declarations for the DirectX Compiler tooling components.        //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef __DXC_TOOLS__
#define __DXC_TOOLS__

#include <dxc/dxcapi.h>

enum RewriterOptionMask {
  Default = 0,
  SkipFunctionBody = 1,
  SkipStatic = 2,
  GlobalExternByDefault = 4,
  KeepUserMacro = 8,
};

CROSS_PLATFORM_UUIDOF(IDxcRewriter, "c012115b-8893-4eb9-9c5a-111456ea1c45")
struct IDxcRewriter : public IUnknown {

  virtual HRESULT STDMETHODCALLTYPE RemoveUnusedGlobals(_In_ IDxcBlobEncoding *pSource,
                                                        _In_z_ LPCWSTR entryPoint,
                                                        _In_count_(defineCount) DxcDefine *pDefines,
                                                        _In_ UINT32 defineCount,
                                                        _COM_Outptr_ IDxcOperationResult **ppResult) = 0;


  virtual HRESULT STDMETHODCALLTYPE RewriteUnchanged(_In_ IDxcBlobEncoding *pSource,
                                                     _In_count_(defineCount) DxcDefine *pDefines,
                                                     _In_ UINT32 defineCount,
                                                     _COM_Outptr_ IDxcOperationResult **ppResult) = 0;

  virtual HRESULT STDMETHODCALLTYPE RewriteUnchangedWithInclude(_In_ IDxcBlobEncoding *pSource,
                                                     // Optional file name for pSource. Used in errors and include handlers.
                                                     _In_opt_ LPCWSTR pSourceName,
                                                     _In_count_(defineCount) DxcDefine *pDefines,
                                                     _In_ UINT32 defineCount,
                                                     // user-provided interface to handle #include directives (optional)
                                                     _In_opt_ IDxcIncludeHandler *pIncludeHandler,
                                                     _In_ UINT32  rewriteOption,
                                                     _COM_Outptr_ IDxcOperationResult **ppResult) = 0;
};

#ifdef _MSC_VER
#define CLSID_SCOPE __declspec(selectany) extern
#else
#define CLSID_SCOPE
#endif

CLSID_SCOPE const CLSID
    CLSID_DxcRewriter = {/* b489b951-e07f-40b3-968d-93e124734da4 */
                         0xb489b951,
                         0xe07f,
                         0x40b3,
                         {0x96, 0x8d, 0x93, 0xe1, 0x24, 0x73, 0x4d, 0xa4}};

CROSS_PLATFORM_UUIDOF(IDxcRewriter2, "261afca1-0609-4ec6-a77f-d98c7035194e")
struct IDxcRewriter2 : public IDxcRewriter {

  virtual HRESULT STDMETHODCALLTYPE RewriteWithOptions(_In_ IDxcBlobEncoding *pSource,
                                                     // Optional file name for pSource. Used in errors and include handlers.
                                                     _In_opt_ LPCWSTR pSourceName,
                                                     // Compiler arguments
                                                     _In_count_(argCount) LPCWSTR *pArguments, _In_ UINT32 argCount,
                                                     // Defines
                                                     _In_count_(defineCount) DxcDefine *pDefines, _In_ UINT32 defineCount,
                                                     // user-provided interface to handle #include directives (optional)
                                                     _In_opt_ IDxcIncludeHandler *pIncludeHandler,
                                                     _COM_Outptr_ IDxcOperationResult **ppResult) = 0;
};

#endif
