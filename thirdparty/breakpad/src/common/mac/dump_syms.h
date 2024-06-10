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

// Author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// dump_syms.h: Declaration of google_breakpad::DumpSymbols, a class for
// reading debugging information from Mach-O files and writing it out as a
// Breakpad symbol file.

#include <mach-o/loader.h>
#include <stdio.h>
#include <stdlib.h>

#include <ostream>
#include <string>
#include <vector>

#include "common/byte_cursor.h"
#include "common/dwarf/dwarf2reader.h"
#include "common/mac/arch_utilities.h"
#include "common/mac/macho_reader.h"
#include "common/mac/super_fat_arch.h"
#include "common/module.h"
#include "common/scoped_ptr.h"
#include "common/symbol_data.h"

namespace google_breakpad {

class DumpSymbols {
 public:
  DumpSymbols(SymbolData symbol_data,
              bool handle_inter_cu_refs,
              bool enable_multiple = false,
              const std::string& module_name = "",
              bool prefer_extern_name = false)
      : symbol_data_(symbol_data),
        handle_inter_cu_refs_(handle_inter_cu_refs),
        object_filename_(),
        contents_(),
        size_(0),
        from_disk_(false),
        object_files_(),
        selected_object_file_(),
        selected_object_name_(),
        enable_multiple_(enable_multiple),
        module_name_(module_name),
        prefer_extern_name_(prefer_extern_name),
        report_warnings_(true) {}
  ~DumpSymbols() = default;

  // Prepare to read debugging information from |filename|. |filename| may be
  // the name of a fat file, a Mach-O file, or a dSYM bundle containing either
  // of the above.
  //
  // If |module_name_| is empty, uses the basename of |filename| as the module
  // name. Otherwise, uses |module_name_| as the module name.
  //
  // On success, return true; if there is a problem reading
  // |filename|, report it and return false.
  bool Read(const std::string& filename);

  // Prepare to read debugging information from |contents|. |contents| is
  // expected to be the data obtained from reading a fat file, or a Mach-O file.
  // |filename| is used to determine the object filename in the generated
  // output; there will not be an attempt to open this file as the data
  // is already expected to be in memory. On success, return true; if there is a
  // problem reading |contents|, report it and return false.
  bool ReadData(uint8_t* contents, size_t size, const std::string& filename);

  // If this dumper's file includes an object file for `info`, then select that
  // object file for dumping, and return true. Otherwise, return false, and
  // leave this dumper's selected architecture unchanged.
  //
  // By default, if this dumper's file contains only one object file, then
  // the dumper will dump those symbols; and if it contains more than one
  // object file, then the dumper will dump the object file whose
  // architecture matches that of this dumper program.
  bool SetArchitecture(const ArchInfo& info);

  // Set whether or not to report DWARF warnings
  void SetReportWarnings(bool report_warnings);

  // Return a pointer to an array of SuperFatArch structures describing the
  // object files contained in this dumper's file. Set *|count| to the number
  // of elements in the array. The returned array is owned by this DumpSymbols
  // instance.
  //
  // If there are no available architectures, this function
  // may return NULL.
  const SuperFatArch* AvailableArchitectures(size_t* count) {
    *count = object_files_.size();
    if (object_files_.size() > 0)
      return &object_files_[0];
    return NULL;
  }

  // Read the selected object file's debugging information, and write out the
  // header only to |stream|. Return true on success; if an error occurs, report
  // it and return false.
  bool WriteSymbolFileHeader(std::ostream& stream);

  // Read the selected object file's debugging information and store it in
  // `module`. The caller owns the resulting module object and must delete
  // it when finished.
  bool ReadSymbolData(Module** module);

  // Return an identifier string for the file this DumpSymbols is dumping.
  std::string Identifier();

 private:
  // Used internally.
  class DumperLineToModule;
  class DumperRangesHandler;
  class LoadCommandDumper;

  // This method behaves similarly to NXFindBestFatArch, but it supports
  // SuperFatArch.
  SuperFatArch* FindBestMatchForArchitecture(
      cpu_type_t cpu_type, cpu_subtype_t cpu_subtype);

  // Creates an empty module object.
  bool CreateEmptyModule(scoped_ptr<Module>& module);

  // Process the split dwarf file referenced by reader.
  void StartProcessSplitDwarf(google_breakpad::CompilationUnit* reader,
                              Module* module,
                              google_breakpad::Endianness endianness,
                              bool handle_inter_cu_refs,
                              bool handle_inline) const;

  // Read debugging information from |dwarf_sections|, which was taken from
  // |macho_reader|, and add it to |module|.
  void ReadDwarf(google_breakpad::Module* module,
                 const mach_o::Reader& macho_reader,
                 const mach_o::SectionMap& dwarf_sections,
                 bool handle_inter_cu_refs) const;

  // Read DWARF CFI or .eh_frame data from |section|, belonging to
  // |macho_reader|, and record it in |module|.  If |eh_frame| is true,
  // then the data is .eh_frame-format data; otherwise, it is standard DWARF
  // .debug_frame data. On success, return true; on failure, report
  // the problem and return false.
  bool ReadCFI(google_breakpad::Module* module,
               const mach_o::Reader& macho_reader,
               const mach_o::Section& section,
               bool eh_frame) const;

  // The selection of what type of symbol data to read/write.
  const SymbolData symbol_data_;

  // Whether to handle references between compilation units.
  const bool handle_inter_cu_refs_;

  // The name of the file this DumpSymbols will actually read debugging
  // information from. If the filename passed to Read refers to a dSYM bundle,
  // then this is the resource file within that bundle.
  std::string object_filename_;

  // The complete contents of object_filename_, mapped into memory.
  scoped_array<uint8_t> contents_;

  // The size of contents_.
  size_t size_;

  // Indicates which entry point to DumpSymbols was used, i.e. Read vs ReadData.
  // This is used to indicate that downstream code paths can/should also read
  // from disk or not.
  bool from_disk_;

  // A vector of SuperFatArch structures describing the object files
  // object_filename_ contains. If object_filename_ refers to a fat binary,
  // this may have more than one element; if it refers to a Mach-O file, this
  // has exactly one element.
  vector<SuperFatArch> object_files_;

  // The object file in object_files_ selected to dump, or NULL if
  // SetArchitecture hasn't been called yet.
  const SuperFatArch* selected_object_file_;

  // A string that identifies the selected object file, for use in error
  // messages.  This is usually object_filename_, but if that refers to a
  // fat binary, it includes an indication of the particular architecture
  // within that binary.
  string selected_object_name_;

  // Whether symbols sharing an address should be collapsed into a single entry
  // and marked with an `m` in the output. 
  // See: https://crbug.com/google-breakpad/751 and docs at 
  // docs/symbol_files.md#records-3
  bool enable_multiple_;

  // If non-empty, used as the module name. Otherwise, the basename of
  // |object_filename_| is used as the module name.
  const std::string module_name_;

  // If a Function and an Extern share the same address but have a different
  // name, prefer the name of the Extern.
  //
  // Use this when dumping Mach-O .dSYMs built with -gmlt (Minimum Line Tables),
  // as the Function's fully-qualified name will only be present in the STABS
  // (which are placed in the Extern), not in the DWARF symbols (which are
  // placed in the Function).
  bool prefer_extern_name_;

  // Whether or not to report warnings
  bool report_warnings_;
};

}  // namespace google_breakpad
