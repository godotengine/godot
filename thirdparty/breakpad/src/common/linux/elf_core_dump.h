// Copyright (c) 2011, Google Inc.
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

// elf_core_dump.h: Define the google_breakpad::ElfCoreDump class, which
// encapsulates an ELF core dump file mapped into memory.

#ifndef COMMON_LINUX_ELF_CORE_DUMP_H_
#define COMMON_LINUX_ELF_CORE_DUMP_H_

#include <elf.h>
#include <limits.h>
#include <link.h>
#include <stddef.h>

#include "common/memory_range.h"

namespace google_breakpad {

// A class encapsulating an ELF core dump file mapped into memory, which
// provides methods for accessing program headers and the note section.
class ElfCoreDump {
 public:
  // ELF types based on the native word size.
  typedef ElfW(Ehdr) Ehdr;
  typedef ElfW(Nhdr) Nhdr;
  typedef ElfW(Phdr) Phdr;
  typedef ElfW(Word) Word;
  typedef ElfW(Addr) Addr;
#if ULONG_MAX == 0xffffffff
  static const int kClass = ELFCLASS32;
#elif ULONG_MAX == 0xffffffffffffffff
  static const int kClass = ELFCLASS64;
#else
#error "Unsupported word size for ElfCoreDump."
#endif

  // A class encapsulating the note content in a core dump, which provides
  // methods for accessing the name and description of a note.
  class Note {
   public:
    Note();

    // Constructor that takes the note content from |content|.
    explicit Note(const MemoryRange& content);

    // Returns true if this note is valid, i,e. a note header is found in
    // |content_|, or false otherwise.
    bool IsValid() const;

    // Returns the note header, or NULL if no note header is found in
    // |content_|.
    const Nhdr* GetHeader() const;

    // Returns the note type, or 0 if no note header is found in |content_|.
    Word GetType() const;

    // Returns a memory range covering the note name, or an empty range
    // if no valid note name is found in |content_|.
    MemoryRange GetName() const;

    // Returns a memory range covering the note description, or an empty
    // range if no valid note description is found in |content_|.
    MemoryRange GetDescription() const;

    // Returns the note following this note, or an empty note if no valid
    // note is found after this note.
    Note GetNextNote() const;

   private:
    // Returns the size in bytes round up to the word alignment, specified
    // for the note section, of a given size in bytes.
    static size_t AlignedSize(size_t size);

    // Note content.
    MemoryRange content_;
  };

  ElfCoreDump();

  // Constructor that takes the core dump content from |content|.
  explicit ElfCoreDump(const MemoryRange& content);

  ~ElfCoreDump();

  // Sets the core dump content to |content|.
  void SetContent(const MemoryRange& content);

  // Returns true if a valid ELF header in the core dump, or false otherwise.
  bool IsValid() const;

  // Returns the ELF header in the core dump, or NULL if no ELF header
  // is found in |content_|.
  const Ehdr* GetHeader() const;

  // Returns the |index|-th program header in the core dump, or NULL if no
  // ELF header is found in |content_| or |index| is out of bounds.
  const Phdr* GetProgramHeader(unsigned index) const;

  // Returns the first program header of |type| in the core dump, or NULL if
  // no ELF header is found in |content_| or no program header of |type| is
  // found.
  const Phdr* GetFirstProgramHeaderOfType(Word type) const;

  // Returns the number of program headers in the core dump, or 0 if no
  // ELF header is found in |content_|.
  unsigned GetProgramHeaderCount() const;

  // Copies |length| bytes of data starting at |virtual_address| in the core
  // dump to |buffer|. |buffer| should be a valid pointer to a buffer of at
  // least |length| bytes. Returns true if the data to be copied is found in
  // the core dump, or false otherwise.
  bool CopyData(void* buffer, Addr virtual_address, size_t length);

  // Returns the first note found in the note section of the core dump, or
  // an empty note if no note is found.
  Note GetFirstNote() const;

  // Sets the mem fd.
  void SetProcMem(const int fd);

 private:
  // Core dump content.
  MemoryRange content_;

  // Descriptor for /proc/<pid>/mem.
  int proc_mem_fd_;
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_ELF_CORE_DUMP_H_
