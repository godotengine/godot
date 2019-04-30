#ifndef __MMAP_HPP__
#define __MMAP_HPP__

#ifndef _WIN32
#  include <sys/mman.h>
#else
#  include <string.h>
#  include <sys/types.h>

#  define PROT_READ 1
#  define PROT_WRITE 2
#  define MAP_SHARED 0

void* mmap( void* addr, size_t length, int prot, int flags, int fd, off_t offset );
int munmap( void* addr, size_t length );

#endif

#endif
