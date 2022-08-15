///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLSLOptions.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// HLSL version enumeration and parsing support                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#ifndef LLVM_HLSL_HLSLVERSION_H
#define LLVM_HLSL_HLSLVERSION_H

namespace hlsl {

 // This Updates to this enum must be reflected in HLSLOptions.td and Options.td
 // for the hlsl_version option.
enum class LangStd : unsigned long  {
  vUnset = 0,
  vError = 1,
  v2015 = 2015,
  v2016 = 2016,
  v2017 = 2017,
  v2018 = 2018,
  v2021 = 2021,
  v202x = 2029,
  vLatest = v2018
};

constexpr const char *ValidVersionsStr = "2015, 2016, 2017, 2018, and 2021";

LangStd parseHLSLVersion(llvm::StringRef Ver);

} // namespace hlsl

#endif // LLVM_HLSL_HLSLVERSION_H
