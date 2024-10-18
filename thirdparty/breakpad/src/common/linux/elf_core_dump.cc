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

// elf_core_dump.cc: Implement google_breakpad::ElfCoreDump.
// See elf_core_dump.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/linux/elf_core_dump.h"

#include <stddef.h>
#include <string.h>
#include <unistd.h>

namespace google_breakpad {

// Implementation of ElfCoreDump::Note.

ElfCoreDump::Note::Note() {}

ElfCoreDump::Note::Note(const MemoryRange& content) : content_(content) {}

bool ElfCoreDump::Note::IsValid() const {
  return GetHeader() != NULL;
}

const ElfCoreDump::Nhdr* ElfCoreDump::Note::GetHeader() const {
  return content_.GetData<Nhdr>(0);
}

ElfCoreDump::Word ElfCoreDump::Note::GetType() const {
  const Nhdr* header = GetHeader();
  // 0 is not being used as a NOTE type.
  return header ? header->n_type : 0;
}

MemoryRange ElfCoreDump::Note::GetName() const {
  const Nhdr* header = GetHeader();
  if (header) {
      return content_.Subrange(sizeof(Nhdr), header->n_namesz);
  }
  return MemoryRange();
}

MemoryRange ElfCoreDump::Note::GetDescription() const {
  const Nhdr* header = GetHeader();
  if (header) {
    return content_.Subrange(AlignedSize(sizeof(Nhdr) + header->n_namesz),
                             header->n_descsz);
  }
  return MemoryRange();
}

ElfCoreDump::Note ElfCoreDump::Note::GetNextNote() const {
  MemoryRange next_content;
  const Nhdr* header = GetHeader();
  if (header) {
    size_t next_offset = AlignedSize(sizeof(Nhdr) + header->n_namesz);
    next_offset = AlignedSize(next_offset + header->n_descsz);
    next_content =
        content_.Subrange(next_offset, content_.length() - next_offset);
  }
  return Note(next_content);
}

// static
size_t ElfCoreDump::Note::AlignedSize(size_t size) {
  size_t mask = sizeof(Word) - 1;
  return (size + mask) & ~mask;
}


// Implementation of ElfCoreDump.

ElfCoreDump::ElfCoreDump() : proc_mem_fd_(-1) {}

ElfCoreDump::ElfCoreDump(const MemoryRange& content)
    : content_(content), proc_mem_fd_(-1) {}

ElfCoreDump::~ElfCoreDump() {
  if (proc_mem_fd_ != -1) {
    close(proc_mem_fd_);
    proc_mem_fd_ = -1;
  }
}

void ElfCoreDump::SetContent(const MemoryRange& content) {
  content_ = content;
}

void ElfCoreDump::SetProcMem(int fd) {
  if (proc_mem_fd_ != -1) {
    close(proc_mem_fd_);
  }
  proc_mem_fd_ = fd;
}

bool ElfCoreDump::IsValid() const {
  const Ehdr* header = GetHeader();
  return (header &&
          header->e_ident[0] == ELFMAG0 &&
          header->e_ident[1] == ELFMAG1 &&
          header->e_ident[2] == ELFMAG2 &&
          header->e_ident[3] == ELFMAG3 &&
          header->e_ident[4] == kClass &&
          header->e_version == EV_CURRENT &&
          header->e_type == ET_CORE);
}

const ElfCoreDump::Ehdr* ElfCoreDump::GetHeader() const {
  return content_.GetData<Ehdr>(0);
}

const ElfCoreDump::Phdr* ElfCoreDump::GetProgramHeader(unsigned index) const {
  const Ehdr* header = GetHeader();
  if (header) {
    return reinterpret_cast<const Phdr*>(content_.GetArrayElement(
        header->e_phoff, header->e_phentsize, index));
  }
  return NULL;
}

const ElfCoreDump::Phdr* ElfCoreDump::GetFirstProgramHeaderOfType(
    Word type) const {
  for (unsigned i = 0, n = GetProgramHeaderCount(); i < n; ++i) {
    const Phdr* program = GetProgramHeader(i);
    if (program->p_type == type) {
      return program;
    }
  }
  return NULL;
}

unsigned ElfCoreDump::GetProgramHeaderCount() const {
  const Ehdr* header = GetHeader();
  return header ? header->e_phnum : 0;
}

bool ElfCoreDump::CopyData(void* buffer, Addr virtual_address, size_t length) {
  for (unsigned i = 0, n = GetProgramHeaderCount(); i < n; ++i) {
    const Phdr* program = GetProgramHeader(i);
    if (program->p_type != PT_LOAD)
      continue;

    size_t offset_in_segment = virtual_address - program->p_vaddr;
    if (virtual_address >= program->p_vaddr &&
        offset_in_segment < program->p_filesz) {
      const void* data =
          content_.GetData(program->p_offset + offset_in_segment, length);
      if (data) {
        memcpy(buffer, data, length);
        return true;
      }
    }
  }

  /* fallback: if available, read from /proc/<pid>/mem */
  if (proc_mem_fd_ != -1) {
    off_t offset = virtual_address;
    ssize_t r = pread(proc_mem_fd_, buffer, length, offset);
    if (r < ssize_t(length)) {
      return false;
    }
    return true;
  }
  return false;
}

ElfCoreDump::Note ElfCoreDump::GetFirstNote() const {
  MemoryRange note_content;
  const Phdr* program_header = GetFirstProgramHeaderOfType(PT_NOTE);
  if (program_header) {
    note_content = content_.Subrange(program_header->p_offset,
                                     program_header->p_filesz);
  }
  return Note(note_content);
}

}  // namespace google_breakpad
