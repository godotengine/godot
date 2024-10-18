// -*- mode: c++ -*-

// Copyright 2010 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// The DwarfLineToModule class accepts line number information from a
// DWARF parser and adds it to a google_breakpad::Module. The Module
// can write that data out as a Breakpad symbol file.

#ifndef COMMON_LINUX_DWARF_LINE_TO_MODULE_H
#define COMMON_LINUX_DWARF_LINE_TO_MODULE_H

#include <string>

#include "common/module.h"
#include "common/dwarf/dwarf2reader.h"
#include "common/using_std_string.h"

namespace google_breakpad {

// A class for producing a vector of google_breakpad::Module::Line
// instances from parsed DWARF line number data.  
//
// An instance of this class can be provided as a handler to a
// LineInfo DWARF line number information parser. The
// handler accepts source location information from the parser and
// uses it to produce a vector of google_breakpad::Module::Line
// objects, referring to google_breakpad::Module::File objects added
// to a particular google_breakpad::Module.
//
// GNU toolchain omitted sections support:
// ======================================
//
// Given the right options, the GNU toolchain will omit unreferenced
// functions from the final executable. Unfortunately, when it does so, it
// does not remove the associated portions of the DWARF line number
// program; instead, it gives the DW_LNE_set_address instructions referring
// to the now-deleted code addresses of zero. Given this input, the DWARF
// line parser will call AddLine with a series of lines starting at address
// zero. For example, here is the output from 'readelf -wl' for a program
// with four functions, the first three of which have been omitted:
//
//   Line Number Statements:
//    Extended opcode 2: set Address to 0x0
//    Advance Line by 14 to 15
//    Copy
//    Special opcode 48: advance Address by 3 to 0x3 and Line by 1 to 16
//    Special opcode 119: advance Address by 8 to 0xb and Line by 2 to 18
//    Advance PC by 2 to 0xd
//    Extended opcode 1: End of Sequence
// 
//    Extended opcode 2: set Address to 0x0
//    Advance Line by 14 to 15
//    Copy
//    Special opcode 48: advance Address by 3 to 0x3 and Line by 1 to 16
//    Special opcode 119: advance Address by 8 to 0xb and Line by 2 to 18
//    Advance PC by 2 to 0xd
//    Extended opcode 1: End of Sequence
// 
//    Extended opcode 2: set Address to 0x0
//    Advance Line by 19 to 20
//    Copy
//    Special opcode 48: advance Address by 3 to 0x3 and Line by 1 to 21
//    Special opcode 76: advance Address by 5 to 0x8 and Line by 1 to 22
//    Advance PC by 2 to 0xa
//    Extended opcode 1: End of Sequence
// 
//    Extended opcode 2: set Address to 0x80483a4
//    Advance Line by 23 to 24
//    Copy
//    Special opcode 202: advance Address by 14 to 0x80483b2 and Line by 1 to 25
//    Special opcode 76: advance Address by 5 to 0x80483b7 and Line by 1 to 26
//    Advance PC by 6 to 0x80483bd
//    Extended opcode 1: End of Sequence
//
// Instead of collecting runs of lines describing code that is not there,
// we try to recognize and drop them. Since the linker doesn't explicitly
// distinguish references to dropped sections from genuine references to
// code at address zero, we must use a heuristic. We have chosen:
//
// - If a line starts at address zero, omit it. (On the platforms
//   breakpad targets, it is extremely unlikely that there will be code
//   at address zero.)
//
// - If a line starts immediately after an omitted line, omit it too.
class DwarfLineToModule: public LineInfoHandler {
 public:
  // As the DWARF line info parser passes us line records, add source
  // files to MODULE, and add all lines to the end of LINES. LINES
  // need not be empty. If the parser hands us a zero-length line, we
  // omit it. If the parser hands us a line that extends beyond the
  // end of the address space, we clip it. It's up to our client to
  // sort out which lines belong to which functions; we don't add them
  // to any particular function in MODULE ourselves.
  DwarfLineToModule(Module* module,
                    const string& compilation_dir,
                    vector<Module::Line>* lines,
                    std::map<uint32_t, Module::File*>* files)
      : module_(module),
        compilation_dir_(compilation_dir),
        lines_(lines),
        files_(files),
        highest_file_number_(-1),
        omitted_line_end_(0),
        warned_bad_file_number_(false),
        warned_bad_directory_number_(false) { }

  ~DwarfLineToModule() { }

  void DefineDir(const string& name, uint32_t dir_num);
  void DefineFile(const string& name, int32_t file_num,
                  uint32_t dir_num, uint64_t mod_time,
                  uint64_t length);
  void AddLine(uint64_t address, uint64_t length,
               uint32_t file_num, uint32_t line_num, uint32_t column_num);

 private:

  typedef std::map<uint32_t, string> DirectoryTable;
  typedef std::map<uint32_t, Module::File*> FileTable;

  // The module we're contributing debugging info to. Owned by our
  // client.
  Module *module_;

  // The compilation directory for the current compilation unit whose
  // lines are being accumulated.
  string compilation_dir_;

  // The vector of lines we're accumulating. Owned by our client.
  //
  // In a Module, as in a breakpad symbol file, lines belong to
  // specific functions, but DWARF simply assigns lines to addresses;
  // one must infer the line/function relationship using the
  // functions' beginning and ending addresses. So we can't add these
  // to the appropriate function from module_ until we've read the
  // function info as well. Instead, we accumulate lines here, and let
  // whoever constructed this sort it all out.
  vector<Module::Line>* lines_;

  // A table mapping directory numbers to paths.
  DirectoryTable directories_;

  // A table mapping file numbers to Module::File pointers.
  FileTable* files_;

  // The highest file number we've seen so far, or -1 if we've seen
  // none.  Used for dynamically defined file numbers.
  int32_t highest_file_number_;

  // This is the ending address of the last line we omitted, or zero if we
  // didn't omit the previous line. It is zero before we have received any
  // AddLine calls.
  uint64_t omitted_line_end_;

  // True if we've warned about:
  bool warned_bad_file_number_; // bad file numbers
  bool warned_bad_directory_number_; // bad directory numbers
};

} // namespace google_breakpad

#endif // COMMON_LINUX_DWARF_LINE_TO_MODULE_H
