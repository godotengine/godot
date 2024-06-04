// -*- mode: c++ -*-

// Copyright (c) 2018 Google Inc.
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

// Original author: Gabriele Svelto <gsvelto@mozilla.com>
//                                  <gabriele.svelto@gmail.com>

// The DwarfRangeListHandler class accepts rangelist data from a DWARF parser
// and adds it to a google_breakpad::Function or other objects supporting
// ranges.

#ifndef COMMON_LINUX_DWARF_RANGE_LIST_HANDLER_H
#define COMMON_LINUX_DWARF_RANGE_LIST_HANDLER_H

#include <vector>

#include "common/module.h"
#include "common/dwarf/dwarf2reader.h"

namespace google_breakpad {

// A class for producing a vector of google_breakpad::Module::Range
// instances from a parsed DWARF range list.

class DwarfRangeListHandler: public RangeListHandler {
 public:
  DwarfRangeListHandler(vector<Module::Range>* ranges)
      : ranges_(ranges) { }

  ~DwarfRangeListHandler() { }

  // Add a range to the list
  void AddRange(uint64_t begin, uint64_t end);

  // Sort the ranges so that they are in ascending order of starting address
  void Finish();

 private:
  // The list of ranges to be populated
  vector<Module::Range>* ranges_;
};

} // namespace google_breakpad

#endif // COMMON_LINUX_DWARF_RANGE_LIST_HANDLER_H
