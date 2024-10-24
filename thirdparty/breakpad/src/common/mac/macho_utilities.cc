// Copyright 2006 Google LLC
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

// macho_utilties.cc: Utilities for dealing with mach-o files
//
// Author: Dave Camp

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/mac/byteswap.h"
#include "common/mac/macho_utilities.h"

#include <mach-o/fat.h>
#include <mach-o/loader.h>

void breakpad_swap_uuid_command(struct breakpad_uuid_command *uc) {
  uc->cmd = ByteSwap(uc->cmd);
  uc->cmdsize = ByteSwap(uc->cmdsize);
}

void breakpad_swap_load_command(struct load_command *lc) {
  lc->cmd = ByteSwap(lc->cmd);
  lc->cmdsize = ByteSwap(lc->cmdsize);
}

void breakpad_swap_dylib_command(struct dylib_command *dc) {
  dc->cmd = ByteSwap(dc->cmd);
  dc->cmdsize = ByteSwap(dc->cmdsize);

  dc->dylib.name.offset = ByteSwap(dc->dylib.name.offset);
  dc->dylib.timestamp = ByteSwap(dc->dylib.timestamp);
  dc->dylib.current_version = ByteSwap(dc->dylib.current_version);
  dc->dylib.compatibility_version = ByteSwap(dc->dylib.compatibility_version);
}

void breakpad_swap_segment_command(struct segment_command *sc) {
  sc->cmd = ByteSwap(sc->cmd);
  sc->cmdsize = ByteSwap(sc->cmdsize);

  sc->vmaddr = ByteSwap(sc->vmaddr);
  sc->vmsize = ByteSwap(sc->vmsize);
  sc->fileoff = ByteSwap(sc->fileoff);
  sc->filesize = ByteSwap(sc->filesize);
  sc->maxprot = ByteSwap(sc->maxprot);
  sc->initprot = ByteSwap(sc->initprot);
  sc->nsects = ByteSwap(sc->nsects);
  sc->flags = ByteSwap(sc->flags);
}

void breakpad_swap_segment_command_64(struct segment_command_64 *sg) {
  sg->cmd = ByteSwap(sg->cmd);
  sg->cmdsize = ByteSwap(sg->cmdsize);

  sg->vmaddr = ByteSwap(sg->vmaddr);
  sg->vmsize = ByteSwap(sg->vmsize);
  sg->fileoff = ByteSwap(sg->fileoff);
  sg->filesize = ByteSwap(sg->filesize);

  sg->maxprot = ByteSwap(sg->maxprot);
  sg->initprot = ByteSwap(sg->initprot);
  sg->nsects = ByteSwap(sg->nsects);
  sg->flags = ByteSwap(sg->flags);
}

void breakpad_swap_fat_header(struct fat_header *fh) {
  fh->magic = ByteSwap(fh->magic);
  fh->nfat_arch = ByteSwap(fh->nfat_arch);
}

void breakpad_swap_fat_arch(struct fat_arch *fa, uint32_t narchs) {
  for (uint32_t i = 0; i < narchs; ++i) {
    fa[i].cputype = ByteSwap(fa[i].cputype);
    fa[i].cpusubtype = ByteSwap(fa[i].cpusubtype);
    fa[i].offset = ByteSwap(fa[i].offset);
    fa[i].size = ByteSwap(fa[i].size);
    fa[i].align = ByteSwap(fa[i].align);
  }
}

void breakpad_swap_mach_header(struct mach_header *mh) {
  mh->magic = ByteSwap(mh->magic);
  mh->cputype = ByteSwap(mh->cputype);
  mh->cpusubtype = ByteSwap(mh->cpusubtype);
  mh->filetype = ByteSwap(mh->filetype);
  mh->ncmds = ByteSwap(mh->ncmds);
  mh->sizeofcmds = ByteSwap(mh->sizeofcmds);
  mh->flags = ByteSwap(mh->flags);
}

void breakpad_swap_mach_header_64(struct mach_header_64 *mh) {
  mh->magic = ByteSwap(mh->magic);
  mh->cputype = ByteSwap(mh->cputype);
  mh->cpusubtype = ByteSwap(mh->cpusubtype);
  mh->filetype = ByteSwap(mh->filetype);
  mh->ncmds = ByteSwap(mh->ncmds);
  mh->sizeofcmds = ByteSwap(mh->sizeofcmds);
  mh->flags = ByteSwap(mh->flags);
  mh->reserved = ByteSwap(mh->reserved);
}

void breakpad_swap_section(struct section *s,
                           uint32_t nsects) {
  for (uint32_t i = 0; i < nsects; i++) {
    s[i].addr = ByteSwap(s[i].addr);
    s[i].size = ByteSwap(s[i].size);

    s[i].offset = ByteSwap(s[i].offset);
    s[i].align = ByteSwap(s[i].align);
    s[i].reloff = ByteSwap(s[i].reloff);
    s[i].nreloc = ByteSwap(s[i].nreloc);
    s[i].flags = ByteSwap(s[i].flags);
    s[i].reserved1 = ByteSwap(s[i].reserved1);
    s[i].reserved2 = ByteSwap(s[i].reserved2);
  }
}

void breakpad_swap_section_64(struct section_64 *s,
                              uint32_t nsects) {
  for (uint32_t i = 0; i < nsects; i++) {
    s[i].addr = ByteSwap(s[i].addr);
    s[i].size = ByteSwap(s[i].size);

    s[i].offset = ByteSwap(s[i].offset);
    s[i].align = ByteSwap(s[i].align);
    s[i].reloff = ByteSwap(s[i].reloff);
    s[i].nreloc = ByteSwap(s[i].nreloc);
    s[i].flags = ByteSwap(s[i].flags);
    s[i].reserved1 = ByteSwap(s[i].reserved1);
    s[i].reserved2 = ByteSwap(s[i].reserved2);
  }
}
