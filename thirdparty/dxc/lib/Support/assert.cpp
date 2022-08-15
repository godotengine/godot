///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// assert.cpp                                                                //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "assert.h"

#ifdef _WIN32

#include "windows.h"
#include "dxc/Support/Global.h"

void llvm_assert(_In_z_ const char *_Message,
                 _In_z_ const char *_File,
                 _In_ unsigned _Line,
                 const char *_Function) {
  OutputDebugFormatA("Error: assert(%s)\nFile:\n%s(%d)\nFunc:\t%s\n", _Message, _File, _Line, _Function);
  RaiseException(STATUS_LLVM_ASSERT, 0, 0, 0);
}

#else

#include <assert.h>

void llvm_assert(const char* message, const char*, unsigned) {
  assert(false && message);
}

#endif
