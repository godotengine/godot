// -*- mode: c++ -*-

// Copyright 2011 Google LLC
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

// dump_symbols.h: Read debugging information from an ELF file, and write
// it out as a Breakpad symbol file.

#ifndef COMMON_LINUX_DUMP_SYMBOLS_H__
#define COMMON_LINUX_DUMP_SYMBOLS_H__

#include <iostream>
#include <string>
#include <vector>

#include "common/symbol_data.h"
#include "common/using_std_string.h"

namespace google_breakpad {

class Module;

struct DumpOptions {
  DumpOptions(SymbolData symbol_data,
              bool handle_inter_cu_refs,
              bool enable_multiple_field,
              bool preserve_load_address)
      : symbol_data(symbol_data),
        handle_inter_cu_refs(handle_inter_cu_refs),
        enable_multiple_field(enable_multiple_field),
        preserve_load_address(preserve_load_address) {}

  SymbolData symbol_data;
  bool handle_inter_cu_refs;
  bool enable_multiple_field;
  bool preserve_load_address;
};

// Find all the debugging information in OBJ_FILE, an ELF executable
// or shared library, and write it to SYM_STREAM in the Breakpad symbol
// file format.
// If OBJ_FILE has been stripped but contains a .gnu_debuglink section,
// then look for the debug file in DEBUG_DIRS.
// SYMBOL_DATA allows limiting the type of symbol data written.
bool WriteSymbolFile(const string& load_path,
                     const string& obj_file,
                     const string& obj_os,
                     const string& module_id,
                     const std::vector<string>& debug_dirs,
                     const DumpOptions& options,
                     std::ostream& sym_stream);

// Read the selected object file's debugging information, and write out the
// header only to |stream|. Return true on success; if an error occurs, report
// it and return false. |obj_file| becomes the MODULE file name and |obj_os|
// becomes the MODULE operating system.
bool WriteSymbolFileHeader(const string& load_path,
                           const string& obj_file,
                           const string& obj_os,
                           const string& module_id,
                           std::ostream& sym_stream);

// As above, but simply return the debugging information in MODULE
// instead of writing it to a stream. The caller owns the resulting
// Module object and must delete it when finished.
bool ReadSymbolData(const string& load_path,
                    const string& obj_file,
                    const string& obj_os,
                    const string& module_id,
                    const std::vector<string>& debug_dirs,
                    const DumpOptions& options,
                    Module** module);

}  // namespace google_breakpad

#endif  // COMMON_LINUX_DUMP_SYMBOLS_H__
