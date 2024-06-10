// Copyright 2007 Google LLC
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

// file_id.cc: Return a unique identifier for a file
//
// See file_id.h for documentation
//
// Author: Alfred Peng

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <elf.h>
#include <fcntl.h>
#include <gelf.h>
#include <sys/mman.h>
#include <sys/ksyms.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>

#include "common/md5.h"
#include "common/solaris/file_id.h"
#include "common/solaris/message_output.h"
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

class AutoElfEnder {
 public:
  AutoElfEnder(Elf* elf) : elf_(elf) {}
  ~AutoElfEnder() { if (elf_) elf_end(elf_); }
 private:
  Elf* elf_;
};

// Find the text section in elf object file.
// Return the section start address and the size.
static bool FindElfTextSection(int fd, const void* elf_base,
                               const void** text_start,
                               int* text_size) {
  assert(text_start);
  assert(text_size);

  *text_start = NULL;
  *text_size = 0;

  if (elf_version(EV_CURRENT) == EV_NONE) {
    print_message2(2, "elf_version() failed: %s\n", elf_errmsg(0));
    return false;
  }

  GElf_Ehdr elf_header;
  lseek(fd, 0L, 0);
  Elf* elf = elf_begin(fd, ELF_C_READ, NULL);
  AutoElfEnder elfEnder(elf);

  if (gelf_getehdr(elf, &elf_header) == (GElf_Ehdr*)NULL) {
    print_message2(2, "failed to read elf header: %s\n", elf_errmsg(-1));
    return false;
  }

  if (elf_header.e_ident[EI_MAG0] != ELFMAG0 ||
      elf_header.e_ident[EI_MAG1] != ELFMAG1 ||
      elf_header.e_ident[EI_MAG2] != ELFMAG2 ||
      elf_header.e_ident[EI_MAG3] != ELFMAG3) {
    print_message1(2, "header magic doesn't match\n");
    return false;
  }

  static const char kTextSectionName[] = ".text";
  const GElf_Shdr* text_section = NULL;
  Elf_Scn* scn = NULL;
  GElf_Shdr shdr;

  while ((scn = elf_nextscn(elf, scn)) != NULL) {
    if (gelf_getshdr(scn, &shdr) == (GElf_Shdr*)0) {
      print_message2(2, "failed to read section header: %s\n", elf_errmsg(0));
      return false;
    }

    if (shdr.sh_type == SHT_PROGBITS) {
      const char* section_name = elf_strptr(elf, elf_header.e_shstrndx,
                                            shdr.sh_name);
      if (!section_name) {
        print_message2(2, "Section name error: %s\n", elf_errmsg(-1));
        continue;
      }

      if (strcmp(section_name, kTextSectionName) == 0) {
        text_section = &shdr;
        break;
      }
    }
  }
  if (text_section != NULL && text_section->sh_size > 0) {
    *text_start = (char*)elf_base + text_section->sh_offset;
    *text_size = text_section->sh_size;
    return true;
  }

  return false;
}

class AutoCloser {
 public:
  AutoCloser(int fd) : fd_(fd) {}
  ~AutoCloser() { if (fd_) close(fd_); }
 private:
  int fd_;
};

namespace elf {

FileID::FileID(const char* path) {
  strcpy(path_, path);
}

bool FileID::ElfFileIdentifier(unsigned char identifier[16]) {
  int fd = 0;
  if ((fd = open(path_, O_RDONLY)) < 0)
    return false;

  AutoCloser autocloser(fd);
  struct stat st;
  if (fstat(fd, &st) != 0 || st.st_size <= 0)
    return false;

  void* base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (base == MAP_FAILED)
    return false;

  bool success = false;
  const void* text_section = NULL;
  int text_size = 0;

  if (FindElfTextSection(fd, base, &text_section, &text_size)) {
    MD5Context md5;
    MD5Init(&md5);
    MD5Update(&md5, (const unsigned char*)text_section, text_size);
    MD5Final(identifier, &md5);
    success = true;
  }

  munmap((char*)base, st.st_size);
  return success;
}

// static
bool FileID::ConvertIdentifierToString(const unsigned char identifier[16],
                                       char* buffer, int buffer_length) {
  if (buffer_length < 34)
    return false;

  int buffer_idx = 0;
  for (int idx = 0; idx < 16; ++idx) {
    int hi = (identifier[idx] >> 4) & 0x0F;
    int lo = (identifier[idx]) & 0x0F;

    buffer[buffer_idx++] = (hi >= 10) ? 'A' + hi - 10 : '0' + hi;
    buffer[buffer_idx++] = (lo >= 10) ? 'A' + lo - 10 : '0' + lo;
  }

  // Add an extra "0" by the end.
  buffer[buffer_idx++] = '0';

  // NULL terminate
  buffer[buffer_idx] = 0;

  return true;
}

}  // elf
}  // namespace google_breakpad
