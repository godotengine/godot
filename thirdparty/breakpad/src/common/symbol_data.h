// -*- mode: c++ -*-

// Copyright (c) 2013 Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef COMMON_SYMBOL_DATA_H_
#define COMMON_SYMBOL_DATA_H_

#include<type_traits>

// Control what data is used from the symbol file.
enum SymbolData {
  NO_DATA = 0,
  SYMBOLS_AND_FILES = 1,
  CFI = 1 << 1,
  INLINES = 1 << 2,
  ALL_SYMBOL_DATA = INLINES | CFI | SYMBOLS_AND_FILES
};

inline SymbolData operator&(SymbolData data1, SymbolData data2) {
  return static_cast<SymbolData>(
      static_cast<std::underlying_type<SymbolData>::type>(data1) &
      static_cast<std::underlying_type<SymbolData>::type>(data2));
}

inline SymbolData operator|(SymbolData data1, SymbolData data2) {
  return static_cast<SymbolData>(
      static_cast<std::underlying_type<SymbolData>::type>(data1) |
      static_cast<std::underlying_type<SymbolData>::type>(data2));
}

#endif  // COMMON_SYMBOL_DATA_H_
