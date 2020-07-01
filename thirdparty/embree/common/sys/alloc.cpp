// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "alloc.h"
#include "intrinsics.h"
#include "sysinfo.h"
#include "mutex.h"

////////////////////////////////////////////////////////////////////////////////
/// All Platforms
////////////////////////////////////////////////////////////////////////////////
  
namespace embree
{
  void* alignedMalloc(size_t size, size_t align) 
  {
    if (size == 0)
      return nullptr;
    
    assert((align & (align-1)) == 0);
    void* ptr = _mm_malloc(size,align);

    if (size != 0 && ptr == nullptr)
      throw std::bad_alloc();
    
    return ptr;
  }
  
  void alignedFree(void* ptr)
  {
    if (ptr)
      _mm_free(ptr);
  }

  static bool huge_pages_enabled = false;
  static MutexSys os_init_mutex;

  __forceinline bool isHugePageCandidate(const size_t bytes) 
  {
    if (!huge_pages_enabled)
      return false;

    /* use huge pages only when memory overhead is low */
    const size_t hbytes = (bytes+PAGE_SIZE_2M-1) & ~size_t(PAGE_SIZE_2M-1);
    return 66*(hbytes-bytes) < bytes; // at most 1.5% overhead
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Windows Platform
////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>

namespace embree
{
  bool win_enable_selockmemoryprivilege (bool verbose)
  {
    HANDLE hToken;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES, &hToken)) {
      if (verbose) std::cout << "WARNING: OpenProcessToken failed while trying to enable SeLockMemoryPrivilege: " << GetLastError() << std::endl;
      return false;
    }

    TOKEN_PRIVILEGES tp;
    tp.PrivilegeCount = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

    if (!LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
      if (verbose) std::cout << "WARNING: LookupPrivilegeValue failed while trying to enable SeLockMemoryPrivilege: " << GetLastError() << std::endl;
      return false;
    }
    
    SetLastError(ERROR_SUCCESS);
    if (!AdjustTokenPrivileges(hToken, FALSE, &tp, sizeof(tp), nullptr, 0)) {
      if (verbose) std::cout << "WARNING: AdjustTokenPrivileges failed while trying to enable SeLockMemoryPrivilege" << std::endl;
      return false;
    }
    
    if (GetLastError() == ERROR_NOT_ALL_ASSIGNED) {
      if (verbose) std::cout << "WARNING: AdjustTokenPrivileges failed to enable SeLockMemoryPrivilege: Add SeLockMemoryPrivilege for current user and run process in elevated mode (Run as administrator)." << std::endl;
      return false;
    } 

    return true;
  }

  bool os_init(bool hugepages, bool verbose) 
  {
    Lock<MutexSys> lock(os_init_mutex);

    if (!hugepages) {
      huge_pages_enabled = false;
      return true;
    }

    if (GetLargePageMinimum() != PAGE_SIZE_2M) {
      huge_pages_enabled = false;
      return false;
    }

    huge_pages_enabled = true;
    return true;
  }

  void* os_malloc(size_t bytes, bool& hugepages)
  {
    if (bytes == 0) {
      hugepages = false;
      return nullptr;
    }

    /* try direct huge page allocation first */
    if (isHugePageCandidate(bytes)) 
    {
      int flags = MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES;
      char* ptr = (char*) VirtualAlloc(nullptr,bytes,flags,PAGE_READWRITE);
      if (ptr != nullptr) {
        hugepages = true;
        return ptr;
      }
    } 

    /* fall back to 4k pages */
    int flags = MEM_COMMIT | MEM_RESERVE;
    char* ptr = (char*) VirtualAlloc(nullptr,bytes,flags,PAGE_READWRITE);
    if (ptr == nullptr) throw std::bad_alloc();
    hugepages = false;
    return ptr;
  }

  size_t os_shrink(void* ptr, size_t bytesNew, size_t bytesOld, bool hugepages) 
  {
    if (hugepages) // decommitting huge pages seems not to work under Windows
      return bytesOld;

    const size_t pageSize = hugepages ? PAGE_SIZE_2M : PAGE_SIZE_4K;
    bytesNew = (bytesNew+pageSize-1) & ~(pageSize-1);
    bytesOld = (bytesOld+pageSize-1) & ~(pageSize-1);
    if (bytesNew >= bytesOld)
      return bytesOld;

    if (!VirtualFree((char*)ptr+bytesNew,bytesOld-bytesNew,MEM_DECOMMIT))
      throw std::bad_alloc();

    return bytesNew;
  }

  void os_free(void* ptr, size_t bytes, bool hugepages) 
  {
    if (bytes == 0) 
      return;

    if (!VirtualFree(ptr,0,MEM_RELEASE))
      throw std::bad_alloc();
  }

  void os_advise(void *ptr, size_t bytes)
  {
  }
}

#endif

////////////////////////////////////////////////////////////////////////////////
/// Unix Platform
////////////////////////////////////////////////////////////////////////////////

#if defined(__UNIX__)

#include <sys/mman.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>

#if defined(__MACOSX__)
#include <mach/vm_statistics.h>
#endif

namespace embree
{
  bool os_init(bool hugepages, bool verbose) 
  {
    Lock<MutexSys> lock(os_init_mutex);

    if (!hugepages) {
      huge_pages_enabled = false;
      return true;
    }

#if defined(__LINUX__)

    int hugepagesize = 0;

    std::ifstream file; 
    file.open("/proc/meminfo",std::ios::in);
    if (!file.is_open()) {
      if (verbose) std::cout << "WARNING: Could not open /proc/meminfo. Huge page support cannot get enabled!" << std::endl;
      huge_pages_enabled = false;
      return false;
    }
    
    std::string line;
    while (getline(file,line))
    {
      std::stringstream sline(line);
      while (!sline.eof() && sline.peek() == ' ') sline.ignore();
      std::string tag; getline(sline,tag,' ');
      while (!sline.eof() && sline.peek() == ' ') sline.ignore();
      std::string val; getline(sline,val,' ');
      while (!sline.eof() && sline.peek() == ' ') sline.ignore();
      std::string unit; getline(sline,unit,' ');
      if (tag == "Hugepagesize:" && unit == "kB") {
	hugepagesize = std::stoi(val)*1024;
	break;
      }
    }
    
    if (hugepagesize != PAGE_SIZE_2M) 
    {
      if (verbose) std::cout << "WARNING: Only 2MB huge pages supported. Huge page support cannot get enabled!" << std::endl;
      huge_pages_enabled = false;
      return false;
    }
#endif

    huge_pages_enabled = true;
    return true;
  }

  void* os_malloc(size_t bytes, bool& hugepages)
  { 
    if (bytes == 0) {
      hugepages = false;
      return nullptr;
    }

    /* try direct huge page allocation first */
    if (isHugePageCandidate(bytes)) 
    {
#if defined(__MACOSX__)
      void* ptr = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, VM_FLAGS_SUPERPAGE_SIZE_2MB, 0);
      if (ptr != MAP_FAILED) {
        hugepages = true;
        return ptr;
      }
#else
      void* ptr = mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON | MAP_HUGETLB, -1, 0);
      if (ptr != MAP_FAILED) {
        hugepages = true;
        return ptr;
      }
#endif
    } 

    /* fallback to 4k pages */
    void* ptr = (char*) mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (ptr == MAP_FAILED) throw std::bad_alloc();
    hugepages = false;

    /* advise huge page hint for THP */
    os_advise(ptr,bytes);
    return ptr;
  }

  size_t os_shrink(void* ptr, size_t bytesNew, size_t bytesOld, bool hugepages) 
  {
    const size_t pageSize = hugepages ? PAGE_SIZE_2M : PAGE_SIZE_4K;
    bytesNew = (bytesNew+pageSize-1) & ~(pageSize-1);
    bytesOld = (bytesOld+pageSize-1) & ~(pageSize-1);
    if (bytesNew >= bytesOld)
      return bytesOld;

    if (munmap((char*)ptr+bytesNew,bytesOld-bytesNew) == -1)
      throw std::bad_alloc();

    return bytesNew;
  }

  void os_free(void* ptr, size_t bytes, bool hugepages) 
  {
    if (bytes == 0)
      return;

    /* for hugepages we need to also align the size */
    const size_t pageSize = hugepages ? PAGE_SIZE_2M : PAGE_SIZE_4K;
    bytes = (bytes+pageSize-1) & ~(pageSize-1);
    if (munmap(ptr,bytes) == -1)
      throw std::bad_alloc();
  }

  /* hint for transparent huge pages (THP) */
  void os_advise(void* pptr, size_t bytes)
  {
#if defined(MADV_HUGEPAGE)
    madvise(pptr,bytes,MADV_HUGEPAGE); 
#endif
  }
}

#endif
