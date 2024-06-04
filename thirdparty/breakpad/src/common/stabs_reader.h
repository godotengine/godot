// -*- mode: c++ -*-

// Copyright (c) 2010 Google Inc. All Rights Reserved.
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

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// stabs_reader.h: Define StabsReader, a parser for STABS debugging
// information. A description of the STABS debugging format can be
// found at:
//
//    http://sourceware.org/gdb/current/onlinedocs/stabs_toc.html
//
// The comments here assume you understand the format.
//
// This parser can handle big-endian and little-endian data, and the symbol
// values may be either 32 or 64 bits long. It handles both STABS in
// sections (as used on Linux) and STABS appearing directly in an
// a.out-like symbol table (as used in Darwin OS X Mach-O files).

#ifndef COMMON_STABS_READER_H__
#define COMMON_STABS_READER_H__

#include <stddef.h>
#include <stdint.h>

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_MACH_O_NLIST_H
#include <mach-o/nlist.h>
#elif defined(HAVE_A_OUT_H)
#include <a.out.h>
#endif

#include <string>
#include <vector>

#include "common/byte_cursor.h"
#include "common/using_std_string.h"

namespace google_breakpad {

class StabsHandler;

class StabsReader {
 public:
  // Create a reader for the STABS debug information whose .stab section is
  // being traversed by ITERATOR, and whose .stabstr section is referred to
  // by STRINGS. The reader will call the member functions of HANDLER to
  // report the information it finds, when the reader's 'Process' member
  // function is called.
  //
  // BIG_ENDIAN should be true if the entries in the .stab section are in
  // big-endian form, or false if they are in little-endian form.
  //
  // VALUE_SIZE should be either 4 or 8, indicating the size of the 'value'
  // field in each entry in bytes.
  //
  // UNITIZED should be true if the STABS data is stored in units with
  // N_UNDF headers. This is usually the case for STABS stored in sections,
  // like .stab/.stabstr, and usually not the case for STABS stored in the
  // actual symbol table; UNITIZED should be true when parsing Linux stabs,
  // false when parsing Mac OS X STABS. For details, see:
  // http://sourceware.org/gdb/current/onlinedocs/stabs/Stab-Section-Basics.html
  // 
  // Note that, in ELF, the .stabstr section should be found using the
  // 'sh_link' field of the .stab section header, not by name.
  StabsReader(const uint8_t* stab,    size_t stab_size,
              const uint8_t* stabstr, size_t stabstr_size,
              bool big_endian, size_t value_size, bool unitized,
              StabsHandler* handler);

  // Process the STABS data, calling the handler's member functions to
  // report what we find.  While the handler functions return true,
  // continue to process until we reach the end of the section.  If we
  // processed the entire section and all handlers returned true,
  // return true.  If any handler returned false, return false.
  // 
  // This is only meant to be called once per StabsReader instance;
  // resuming a prior processing pass that stopped abruptly isn't supported.
  bool Process();

 private:

  // An class for walking arrays of STABS entries. This isolates the main
  // STABS reader from the exact format (size; endianness) of the entries
  // themselves.
  class EntryIterator {
   public:
    // The contents of a STABS entry, adjusted for the host's endianness,
    // word size, 'struct nlist' layout, and so on.
    struct Entry {
      // True if this iterator has reached the end of the entry array. When
      // this is set, the other members of this structure are not valid.
      bool at_end;

      // The number of this entry within the list.
      size_t index;

      // The current entry's name offset. This is the offset within the
      // current compilation unit's strings, as establish by the N_UNDF entries.
      size_t name_offset;

      // The current entry's type, 'other' field, descriptor, and value.
      unsigned char type;
      unsigned char other;
      short descriptor;
      uint64_t value;
    };

    // Create a EntryIterator walking the entries in BUFFER. Treat the
    // entries as big-endian if BIG_ENDIAN is true, as little-endian
    // otherwise. Assume each entry has a 'value' field whose size is
    // VALUE_SIZE.
    //
    // This would not be terribly clean to extend to other format variations,
    // but it's enough to handle Linux and Mac, and we'd like STABS to die
    // anyway.
    //
    // For the record: on Linux, STABS entry values are always 32 bits,
    // regardless of the architecture address size (don't ask me why); on
    // Mac, they are 32 or 64 bits long. Oddly, the section header's entry
    // size for a Linux ELF .stab section varies according to the ELF class
    // from 12 to 20 even as the actual entries remain unchanged.
    EntryIterator(const ByteBuffer* buffer, bool big_endian, size_t value_size);

    // Move to the next entry. This function's behavior is undefined if
    // at_end() is true when it is called.
    EntryIterator& operator++() { Fetch(); entry_.index++; return *this; }

    // Dereferencing this iterator produces a reference to an Entry structure
    // that holds the current entry's values. The entry is owned by this
    // EntryIterator, and will be invalidated at the next call to operator++.
    const Entry& operator*() const { return entry_; }
    const Entry* operator->() const { return &entry_; }

   private:
    // Read the STABS entry at cursor_, and set entry_ appropriately.
    void Fetch();

    // The size of entries' value field, in bytes.
    size_t value_size_;

    // A byte cursor traversing buffer_.
    ByteCursor cursor_;

    // Values for the entry this iterator refers to.
    Entry entry_;
  };

  // A source line, saved to be reported later.
  struct Line {
    uint64_t address;
    const char* filename;
    int number;
  };

  // Return the name of the current symbol.
  const char* SymbolString();

  // Process a compilation unit starting at symbol_.  Return true
  // to continue processing, or false to abort.
  bool ProcessCompilationUnit();

  // Process a function in current_source_file_ starting at symbol_.
  // Return true to continue processing, or false to abort.
  bool ProcessFunction();

  // Process an exported function symbol.
  // Return true to continue processing, or false to abort.
  bool ProcessExtern();

  // The STABS entries being parsed.
  ByteBuffer entries_;

  // The string section to which the entries refer.
  ByteBuffer strings_;

  // The iterator walking the STABS entries.
  EntryIterator iterator_;

  // True if the data is "unitized"; see the explanation in the comment for
  // StabsReader::StabsReader.
  bool unitized_;

  StabsHandler* handler_;

  // The offset of the current compilation unit's strings within stabstr_.
  size_t string_offset_;

  // The value string_offset_ should have for the next compilation unit,
  // as established by N_UNDF entries.
  size_t next_cu_string_offset_;

  // The current source file name.
  const char* current_source_file_;

  // Mac OS X STABS place SLINE records before functions; we accumulate a
  // vector of these until we see the FUN record, and then report them
  // after the StartFunction call.
  std::vector<Line> queued_lines_;
};

// Consumer-provided callback structure for the STABS reader.  Clients
// of the STABS reader provide an instance of this structure.  The
// reader then invokes the member functions of that instance to report
// the information it finds.
//
// The default definitions of the member functions do nothing, and return
// true so processing will continue.
class StabsHandler {
 public:
  StabsHandler() { }
  virtual ~StabsHandler() { }

  // Some general notes about the handler callback functions:

  // Processing proceeds until the end of the .stabs section, or until
  // one of these functions returns false.

  // The addresses given are as reported in the STABS info, without
  // regard for whether the module may be loaded at different
  // addresses at different times (a shared library, say).  When
  // processing STABS from an ELF shared library, the addresses given
  // all assume the library is loaded at its nominal load address.
  // They are *not* offsets from the nominal load address.  If you
  // want offsets, you must subtract off the library's nominal load
  // address.

  // The arguments to these functions named FILENAME are all
  // references to strings stored in the .stabstr section.  Because
  // both the Linux and Solaris linkers factor out duplicate strings
  // from the .stabstr section, the consumer can assume that if two
  // FILENAME values are different addresses, they represent different
  // file names.
  //
  // Thus, it's safe to use (say) std::map<char*, ...>, which does
  // string address comparisons, not string content comparisons.
  // Since all the strings are in same array of characters --- the
  // .stabstr section --- comparing their addresses produces
  // predictable, if not lexicographically meaningful, results.

  // Begin processing a compilation unit whose main source file is
  // named FILENAME, and whose base address is ADDRESS.  If
  // BUILD_DIRECTORY is non-NULL, it is the name of the build
  // directory in which the compilation occurred.
  virtual bool StartCompilationUnit(const char* filename, uint64_t address,
                                    const char* build_directory) {
    return true;
  }

  // Finish processing the compilation unit.  If ADDRESS is non-zero,
  // it is the ending address of the compilation unit.  If ADDRESS is
  // zero, then the compilation unit's ending address is not
  // available, and the consumer must infer it by other means.
  virtual bool EndCompilationUnit(uint64_t address) { return true; }

  // Begin processing a function named NAME, whose starting address is
  // ADDRESS.  This function belongs to the compilation unit that was
  // most recently started but not ended.
  //
  // Note that, unlike filenames, NAME is not a pointer into the
  // .stabstr section; this is because the name as it appears in the
  // STABS data is followed by type information.  The value passed to
  // StartFunction is the function name alone.
  //
  // In languages that use name mangling, like C++, NAME is mangled.
  virtual bool StartFunction(const string& name, uint64_t address) {
    return true;
  }

  // Finish processing the function.  If ADDRESS is non-zero, it is
  // the ending address for the function.  If ADDRESS is zero, then
  // the function's ending address is not available, and the consumer
  // must infer it by other means.
  virtual bool EndFunction(uint64_t address) { return true; }
  
  // Report that the code at ADDRESS is attributable to line NUMBER of
  // the source file named FILENAME.  The caller must infer the ending
  // address of the line.
  virtual bool Line(uint64_t address, const char* filename, int number) {
    return true;
  }

  // Report that an exported function NAME is present at ADDRESS.
  // The size of the function is unknown.
  virtual bool Extern(const string& name, uint64_t address) {
    return true;
  }

  // Report a warning.  FORMAT is a printf-like format string,
  // specifying how to format the subsequent arguments.
  virtual void Warning(const char* format, ...) = 0;
};

} // namespace google_breakpad

#endif  // COMMON_STABS_READER_H__
