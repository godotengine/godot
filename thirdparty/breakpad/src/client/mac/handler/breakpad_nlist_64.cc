/*
 * Copyright (c) 1999 Apple Computer, Inc. All rights reserved.
 *
 * @APPLE_LICENSE_HEADER_START@
 * 
 * This file contains Original Code and/or Modifications of Original Code
 * as defined in and that are subject to the Apple Public Source License
 * Version 2.0 (the 'License'). You may not use this file except in
 * compliance with the License. Please obtain a copy of the License at
 * http://www.opensource.apple.com/apsl/ and read it before using this
 * file.
 * 
 * The Original Code and all software distributed under the License are
 * distributed on an 'AS IS' basis, WITHOUT WARRANTY OF ANY KIND, EITHER
 * EXPRESS OR IMPLIED, AND APPLE HEREBY DISCLAIMS ALL SUCH WARRANTIES,
 * INCLUDING WITHOUT LIMITATION, ANY WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, QUIET ENJOYMENT OR NON-INFRINGEMENT.
 * Please see the License for the specific language governing rights and
 * limitations under the License.
 * 
 * @APPLE_LICENSE_HEADER_END@
 */
/*
 * Copyright (c) 1989, 1993
 * The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the University of
 *      California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */


/*
 * This file was copied from libc/gen/nlist.c from Darwin's source code       
 * The version of nlist used as a base is from 10.5.2, libc-498               
 * http://www.opensource.apple.com/darwinsource/10.5.2/Libc-498/gen/nlist.c   
 *                                                                            
 * The full tarball is at:                                                    
 * http://www.opensource.apple.com/darwinsource/tarballs/apsl/Libc-498.tar.gz 
 *                                                                            
 * I've modified it to be compatible with 64-bit images.
*/

#include "breakpad_nlist_64.h"

#include <CoreFoundation/CoreFoundation.h>
#include <fcntl.h>
#include <mach-o/nlist.h>
#include <mach-o/loader.h>
#include <mach-o/fat.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <TargetConditionals.h>
#include <unistd.h>

/* Stuff lifted from <a.out.h> and <sys/exec.h> since they are gone */
/*
 * Header prepended to each a.out file.
 */
struct exec {
  unsigned short  a_machtype;     /* machine type */
  unsigned short  a_magic;        /* magic number */
  unsigned long a_text;         /* size of text segment */
  unsigned long a_data;         /* size of initialized data */
  unsigned long a_bss;          /* size of uninitialized data */
  unsigned long a_syms;         /* size of symbol table */
  unsigned long a_entry;        /* entry point */
  unsigned long a_trsize;       /* size of text relocation */
  unsigned long a_drsize;       /* size of data relocation */
};

#define OMAGIC  0407            /* old impure format */
#define NMAGIC  0410            /* read-only text */
#define ZMAGIC  0413            /* demand load format */

#define N_BADMAG(x)                                                     \
  (((x).a_magic)!=OMAGIC && ((x).a_magic)!=NMAGIC && ((x).a_magic)!=ZMAGIC)
#define N_TXTOFF(x)                                     \
  ((x).a_magic==ZMAGIC ? 0 : sizeof (struct exec))
#define N_SYMOFF(x)                                                     \
  (N_TXTOFF(x) + (x).a_text+(x).a_data + (x).a_trsize+(x).a_drsize)

// Traits structs for specializing function templates to handle
// 32-bit/64-bit Mach-O files.
template<typename T>
struct MachBits {};

typedef struct nlist nlist32;
typedef struct nlist_64 nlist64;

template<>
struct MachBits<nlist32> {
  typedef mach_header mach_header_type;
  typedef uint32_t word_type;
  static const uint32_t magic = MH_MAGIC;
};

template<>
struct MachBits<nlist64> {
  typedef mach_header_64 mach_header_type;
  typedef uint64_t word_type;
  static const uint32_t magic = MH_MAGIC_64;
};

template<typename nlist_type>
int
__breakpad_fdnlist(int fd, nlist_type* list, const char** symbolNames,
                   cpu_type_t cpu_type);

/*
 * nlist - retreive attributes from name list (string table version)
 */

template <typename nlist_type>
int breakpad_nlist_common(const char* name,
                          nlist_type* list,
                          const char** symbolNames,
                          cpu_type_t cpu_type) {
  int fd = open(name, O_RDONLY, 0);
  if (fd < 0)
    return -1;
  int n = __breakpad_fdnlist(fd, list, symbolNames, cpu_type);
  close(fd);
  return n;
}

int breakpad_nlist(const char* name,
                   struct nlist* list,
                   const char** symbolNames,
                   cpu_type_t cpu_type) {
  return breakpad_nlist_common(name, list, symbolNames, cpu_type);
}

int breakpad_nlist(const char* name,
                   struct nlist_64* list,
                   const char** symbolNames,
                   cpu_type_t cpu_type) {
  return breakpad_nlist_common(name, list, symbolNames, cpu_type);
}

/* Note: __fdnlist() is called from kvm_nlist in libkvm's kvm.c */

template<typename nlist_type>
int __breakpad_fdnlist(int fd, nlist_type* list, const char** symbolNames,
                       cpu_type_t cpu_type) {
  typedef typename MachBits<nlist_type>::mach_header_type mach_header_type;
  typedef typename MachBits<nlist_type>::word_type word_type;

  const uint32_t magic = MachBits<nlist_type>::magic;

  int maxlen = 500;
  int nreq = 0;
  for (nlist_type* q = list;
       symbolNames[q-list] && symbolNames[q-list][0];
       q++, nreq++) {

    q->n_type = 0;
    q->n_value = 0;
    q->n_desc = 0;
    q->n_sect = 0;
    q->n_un.n_strx = 0;
  }

  struct exec buf;
  if (read(fd, (char*)&buf, sizeof(buf)) != sizeof(buf) ||
      (N_BADMAG(buf) && *((uint32_t*)&buf) != magic &&
        CFSwapInt32BigToHost(*((uint32_t*)&buf)) != FAT_MAGIC &&
       /* The following is the big-endian ppc64 check */
       (*((uint32_t*)&buf)) != FAT_MAGIC)) {
    return -1;
  }

  /* Deal with fat file if necessary */
  unsigned arch_offset = 0;
  if (CFSwapInt32BigToHost(*((uint32_t*)&buf)) == FAT_MAGIC ||
      /* The following is the big-endian ppc64 check */
      *((unsigned int*)&buf) == FAT_MAGIC) {
    /* Read in the fat header */
    struct fat_header fh;
    if (lseek(fd, 0, SEEK_SET) == -1) {
      return -1;
    }
    if (read(fd, (char*)&fh, sizeof(fh)) != sizeof(fh)) {
      return -1;
    }

    /* Convert fat_narchs to host byte order */
    fh.nfat_arch = CFSwapInt32BigToHost(fh.nfat_arch);

    /* Read in the fat archs */
    struct fat_arch* fat_archs =
        (struct fat_arch*)malloc(fh.nfat_arch * sizeof(struct fat_arch));
    if (fat_archs == NULL) {
      return -1;
    }
    if (read(fd, (char*)fat_archs,
             sizeof(struct fat_arch) * fh.nfat_arch) !=
        (ssize_t)(sizeof(struct fat_arch) * fh.nfat_arch)) {
      free(fat_archs);
      return -1;
    }

    /*
     * Convert archs to host byte ordering (a constraint of
     * cpusubtype_getbestarch()
     */
    for (unsigned i = 0; i < fh.nfat_arch; i++) {
      fat_archs[i].cputype =
        CFSwapInt32BigToHost(fat_archs[i].cputype);
      fat_archs[i].cpusubtype =
        CFSwapInt32BigToHost(fat_archs[i].cpusubtype);
      fat_archs[i].offset =
        CFSwapInt32BigToHost(fat_archs[i].offset);
      fat_archs[i].size =
        CFSwapInt32BigToHost(fat_archs[i].size);
      fat_archs[i].align =
        CFSwapInt32BigToHost(fat_archs[i].align);
    }

    struct fat_arch* fap = NULL;
    for (unsigned i = 0; i < fh.nfat_arch; i++) {
      if (fat_archs[i].cputype == cpu_type) {
        fap = &fat_archs[i];
        break;
      }
    }

    if (!fap) {
      free(fat_archs);
      return -1;
    }
    arch_offset = fap->offset;
    free(fat_archs);

    /* Read in the beginning of the architecture-specific file */
    if (lseek(fd, arch_offset, SEEK_SET) == -1) {
      return -1;
    }
    if (read(fd, (char*)&buf, sizeof(buf)) != sizeof(buf)) {
      return -1;
    }
  }

  off_t sa;  /* symbol address */
  off_t ss;  /* start of strings */
  register_t n;
  if (*((unsigned int*)&buf) == magic) {
    if (lseek(fd, arch_offset, SEEK_SET) == -1) {
      return -1;
    }
    mach_header_type mh;
    if (read(fd, (char*)&mh, sizeof(mh)) != sizeof(mh)) {
      return -1;
    }

    struct load_command* load_commands =
        (struct load_command*)malloc(mh.sizeofcmds);
    if (load_commands == NULL) {
      return -1;
    }
    if (read(fd, (char*)load_commands, mh.sizeofcmds) !=
        (ssize_t)mh.sizeofcmds) {
      free(load_commands);
      return -1;
    }
    struct symtab_command* stp = NULL;
    struct load_command* lcp = load_commands;
    // iterate through all load commands, looking for
    // LC_SYMTAB load command
    for (uint32_t i = 0; i < mh.ncmds; i++) {
      if (lcp->cmdsize % sizeof(word_type) != 0 ||
          lcp->cmdsize <= 0 ||
          (char*)lcp + lcp->cmdsize > (char*)load_commands + mh.sizeofcmds) {
        free(load_commands);
        return -1;
      }
      if (lcp->cmd == LC_SYMTAB) {
        if (lcp->cmdsize != sizeof(struct symtab_command)) {
          free(load_commands);
          return -1;
        }
        stp = (struct symtab_command*)lcp;
        break;
      }
      lcp = (struct load_command*)((char*)lcp + lcp->cmdsize);
    }
    if (stp == NULL) {
      free(load_commands);
      return -1;
    }
    // sa points to the beginning of the symbol table
    sa = stp->symoff + arch_offset;
    // ss points to the beginning of the string table
    ss = stp->stroff + arch_offset;
    // n is the number of bytes in the symbol table
    // each symbol table entry is an nlist structure
    n = stp->nsyms * sizeof(nlist_type);
    free(load_commands);
  } else {
    sa = N_SYMOFF(buf) + arch_offset;
    ss = sa + buf.a_syms + arch_offset;
    n = buf.a_syms;
  }

  if (lseek(fd, sa, SEEK_SET) == -1) {
    return -1;
  }

  // the algorithm here is to read the nlist entries in m-sized
  // chunks into q.  q is then iterated over. for each entry in q,
  // use the string table index(q->n_un.n_strx) to read the symbol 
  // name, then scan the nlist entries passed in by the user(via p),
  // and look for a match
  while (n) {
    nlist_type space[BUFSIZ/sizeof (nlist_type)];
    register_t m = sizeof (space);

    if (n < m)
      m = n;
    if (read(fd, (char*)space, m) != m)
      break;
    n -= m;
    off_t savpos = lseek(fd, 0, SEEK_CUR);
    if (savpos == -1) {
      return -1;
    }
    for (nlist_type* q = space; (m -= sizeof(nlist_type)) >= 0; q++) {
      char nambuf[BUFSIZ];

      if (q->n_un.n_strx == 0 || q->n_type & N_STAB)
        continue;

      // seek to the location in the binary where the symbol
      // name is stored & read it into memory
      if (lseek(fd, ss+q->n_un.n_strx, SEEK_SET) == -1) {
        return -1;
      }
      if (read(fd, nambuf, maxlen+1) == -1) {
        return -1;
      }
      const char* s2 = nambuf;
      for (nlist_type* p = list; 
           symbolNames[p-list] && symbolNames[p-list][0];
           p++) {
        // get the symbol name the user has passed in that 
        // corresponds to the nlist entry that we're looking at
        const char* s1 = symbolNames[p - list];
        while (*s1) {
          if (*s1++ != *s2++)
            goto cont;
        }
        if (*s2)
          goto cont;

        p->n_value = q->n_value;
        p->n_type = q->n_type;
        p->n_desc = q->n_desc;
        p->n_sect = q->n_sect;
        p->n_un.n_strx = q->n_un.n_strx;
        if (--nreq == 0)
          return nreq;

        break;
      cont:           ;
      }
    }
    if (lseek(fd, savpos, SEEK_SET) == -1) {
      return -1;
    }
  }
  return nreq;
}
