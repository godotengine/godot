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

// This file implements the google_breakpad::StabsReader class.
// See stabs_reader.h.

#include "common/stabs_reader.h"

#include <assert.h>
#include <stab.h>
#include <string.h>

#include <string>

#include "common/using_std_string.h"

using std::vector;

namespace google_breakpad {

StabsReader::EntryIterator::EntryIterator(const ByteBuffer* buffer,
                                          bool big_endian, size_t value_size)
    : value_size_(value_size), cursor_(buffer, big_endian) {
  // Actually, we could handle weird sizes just fine, but they're
  // probably mistakes --- expressed in bits, say.
  assert(value_size == 4 || value_size == 8);
  entry_.index = 0;
  Fetch();
}

void StabsReader::EntryIterator::Fetch() {
  cursor_
      .Read(4, false, &entry_.name_offset)
      .Read(1, false, &entry_.type)
      .Read(1, false, &entry_.other)
      .Read(2, false, &entry_.descriptor)
      .Read(value_size_, false, &entry_.value);
  entry_.at_end = !cursor_;
}

StabsReader::StabsReader(const uint8_t* stab,    size_t stab_size,
                         const uint8_t* stabstr, size_t stabstr_size,
                         bool big_endian, size_t value_size, bool unitized,
                         StabsHandler* handler)
    : entries_(stab, stab_size),
      strings_(stabstr, stabstr_size),
      iterator_(&entries_, big_endian, value_size),
      unitized_(unitized),
      handler_(handler),
      string_offset_(0),
      next_cu_string_offset_(0),
      current_source_file_(NULL) { }

const char* StabsReader::SymbolString() {
  ptrdiff_t offset = string_offset_ + iterator_->name_offset;
  if (offset < 0 || (size_t) offset >= strings_.Size()) {
    handler_->Warning("symbol %d: name offset outside the string section\n",
                      iterator_->index);
    // Return our null string, to keep our promise about all names being
    // taken from the string section.
    offset = 0;
  }
  return reinterpret_cast<const char*>(strings_.start + offset);
}

bool StabsReader::Process() {
  while (!iterator_->at_end) {
    if (iterator_->type == N_SO) {
      if (! ProcessCompilationUnit())
        return false;
    } else if (iterator_->type == N_UNDF && unitized_) {
      // In unitized STABS (including Linux STABS, and pretty much anything
      // else that puts STABS data in sections), at the head of each
      // compilation unit's entries there is an N_UNDF stab giving the
      // number of symbols in the compilation unit, and the number of bytes
      // that compilation unit's strings take up in the .stabstr section.
      // Each CU's strings are separate; the n_strx values are offsets
      // within the current CU's portion of the .stabstr section.
      //
      // As an optimization, the GNU linker combines all the
      // compilation units into one, with a single N_UNDF at the
      // beginning. However, other linkers, like Gold, do not perform
      // this optimization.
      string_offset_ = next_cu_string_offset_;
      next_cu_string_offset_ = iterator_->value;
      ++iterator_;
    }
#if defined(HAVE_MACH_O_NLIST_H)
    // Export symbols in Mach-O binaries look like this.
    // This is necessary in order to be able to dump symbols
    // from OS X system libraries.
    else if ((iterator_->type & N_STAB) == 0 &&
               (iterator_->type & N_TYPE) == N_SECT) {
      ProcessExtern();
    }
#endif
    else {
      ++iterator_;
    }
  }
  return true;
}

bool StabsReader::ProcessCompilationUnit() {
  assert(!iterator_->at_end && iterator_->type == N_SO);

  // There may be an N_SO entry whose name ends with a slash,
  // indicating the directory in which the compilation occurred.
  // The build directory defaults to NULL.
  const char* build_directory = NULL;
  {
    const char* name = SymbolString();
    if (name[0] && name[strlen(name) - 1] == '/') {
      build_directory = name;
      ++iterator_;
    }
  }

  // We expect to see an N_SO entry with a filename next, indicating
  // the start of the compilation unit.
  {
    if (iterator_->at_end || iterator_->type != N_SO)
      return true;
    const char* name = SymbolString();
    if (name[0] == '\0') {
      // This seems to be a stray end-of-compilation-unit marker;
      // consume it, but don't report the end, since we didn't see a
      // beginning.
      ++iterator_;
      return true;
    }
    current_source_file_ = name;
  }

  if (! handler_->StartCompilationUnit(current_source_file_,
                                       iterator_->value,
                                       build_directory))
    return false;

  ++iterator_;

  // The STABS documentation says that some compilers may emit
  // additional N_SO entries with names immediately following the
  // first, and that they should be ignored.  However, the original
  // Breakpad STABS reader doesn't ignore them, so we won't either.

  // Process the body of the compilation unit, up to the next N_SO.
  while (!iterator_->at_end && iterator_->type != N_SO) {
    if (iterator_->type == N_FUN) {
      if (! ProcessFunction())
        return false;
    } else if (iterator_->type == N_SLINE) {
      // Mac OS X STABS place SLINE records before functions.
      Line line;
      // The value of an N_SLINE entry that appears outside a function is
      // the absolute address of the line.
      line.address = iterator_->value;
      line.filename = current_source_file_;
      // The n_desc of a N_SLINE entry is the line number.  It's a
      // signed 16-bit field; line numbers from 32768 to 65535 are
      // stored as n-65536.
      line.number = (uint16_t) iterator_->descriptor;
      queued_lines_.push_back(line);
      ++iterator_;
    } else if (iterator_->type == N_SOL) {
      current_source_file_ = SymbolString();
      ++iterator_;
    } else {
      // Ignore anything else.
      ++iterator_;
    }
  }

  // An N_SO with an empty name indicates the end of the compilation
  // unit.  Default to zero.
  uint64_t ending_address = 0;
  if (!iterator_->at_end) {
    assert(iterator_->type == N_SO);
    const char* name = SymbolString();
    if (name[0] == '\0') {
      ending_address = iterator_->value;
      ++iterator_;
    }
  }

  if (! handler_->EndCompilationUnit(ending_address))
    return false;

  queued_lines_.clear();

  return true;
}

bool StabsReader::ProcessFunction() {
  assert(!iterator_->at_end && iterator_->type == N_FUN);

  uint64_t function_address = iterator_->value;
  // The STABS string for an N_FUN entry is the name of the function,
  // followed by a colon, followed by type information for the
  // function.  We want to pass the name alone to StartFunction.
  const char* stab_string = SymbolString();
  const char* name_end = strchr(stab_string, ':');
  if (! name_end)
    name_end = stab_string + strlen(stab_string);
  string name(stab_string, name_end - stab_string);
  if (! handler_->StartFunction(name, function_address))
    return false;
  ++iterator_;

  // If there were any SLINE records given before the function, report them now.
  for (vector<Line>::const_iterator it = queued_lines_.begin();
       it != queued_lines_.end(); it++) {
    if (!handler_->Line(it->address, it->filename, it->number))
      return false;
  }
  queued_lines_.clear();

  while (!iterator_->at_end) {
    if (iterator_->type == N_SO || iterator_->type == N_FUN)
      break;
    else if (iterator_->type == N_SLINE) {
      // The value of an N_SLINE entry is the offset of the line from
      // the function's start address.
      uint64_t line_address = function_address + iterator_->value;
      // The n_desc of a N_SLINE entry is the line number.  It's a
      // signed 16-bit field; line numbers from 32768 to 65535 are
      // stored as n-65536.
      uint16_t line_number = iterator_->descriptor;
      if (! handler_->Line(line_address, current_source_file_, line_number))
        return false;
      ++iterator_;
    } else if (iterator_->type == N_SOL) {
      current_source_file_ = SymbolString();
      ++iterator_;
    } else
      // Ignore anything else.
      ++iterator_;
  }

  // We've reached the end of the function. See if we can figure out its
  // ending address.
  uint64_t ending_address = 0;
  if (!iterator_->at_end) {
    assert(iterator_->type == N_SO || iterator_->type == N_FUN);
    if (iterator_->type == N_FUN) {
      const char* symbol_name = SymbolString();
      if (symbol_name[0] == '\0') {
        // An N_FUN entry with no name is a terminator for this function;
        // its value is the function's size.
        ending_address = function_address + iterator_->value;
        ++iterator_;
      } else {
        // An N_FUN entry with a name is the next function, and we can take
        // its value as our ending address. Don't advance the iterator, as
        // we'll use this symbol to start the next function as well.
        ending_address = iterator_->value;
      }
    } else {
      // An N_SO entry could be an end-of-compilation-unit marker, or the
      // start of the next compilation unit, but in either case, its value
      // is our ending address. We don't advance the iterator;
      // ProcessCompilationUnit will decide what to do with this symbol.
      ending_address = iterator_->value;
    }
  }

  if (! handler_->EndFunction(ending_address))
    return false;

  return true;
}

bool StabsReader::ProcessExtern() {
#if defined(HAVE_MACH_O_NLIST_H)
  assert(!iterator_->at_end &&
         (iterator_->type & N_STAB) == 0 &&
         (iterator_->type & N_TYPE) == N_SECT);
#endif

  // TODO(mark): only do symbols in the text section?
  if (!handler_->Extern(SymbolString(), iterator_->value))
    return false;

  ++iterator_;
  return true;
}

} // namespace google_breakpad
