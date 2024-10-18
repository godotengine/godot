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

//  dynamic_images.h
//
//    Implements most of the function of the dyld API, but allowing an
//    arbitrary task to be introspected, unlike the dyld API which
//    only allows operation on the current task.  The current implementation
//    is limited to use by 32-bit tasks.

#ifndef CLIENT_MAC_HANDLER_DYNAMIC_IMAGES_H__
#define CLIENT_MAC_HANDLER_DYNAMIC_IMAGES_H__

#include <mach/mach.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <sys/types.h>

#include <string>
#include <vector>

#include "mach_vm_compat.h"

namespace google_breakpad {

using std::string;
using std::vector;

//==============================================================================
// The memory layout of this struct matches the dyld_image_info struct
// defined in "dyld_gdb.h" in the darwin source.
typedef struct dyld_image_info32 {
  uint32_t                   load_address_;  // struct mach_header*
  uint32_t                   file_path_;     // char*
  uint32_t                   file_mod_date_;
} dyld_image_info32;

typedef struct dyld_image_info64 {
  uint64_t                   load_address_;  // struct mach_header*
  uint64_t                   file_path_;     // char*
  uint64_t                   file_mod_date_;
} dyld_image_info64;

//==============================================================================
// This is as defined in "dyld_gdb.h" in the darwin source.
// _dyld_all_image_infos (in dyld) is a structure of this type
// which will be used to determine which dynamic code has been loaded.
typedef struct dyld_all_image_infos32 {
  uint32_t                      version;  // == 1 in Mac OS X 10.4
  uint32_t                      infoArrayCount;
  uint32_t                      infoArray;  // const struct dyld_image_info*
  uint32_t                      notification;
  bool                          processDetachedFromSharedRegion;
} dyld_all_image_infos32;

typedef struct dyld_all_image_infos64 {
  uint32_t                      version;  // == 1 in Mac OS X 10.4
  uint32_t                      infoArrayCount;
  uint64_t                      infoArray;  // const struct dyld_image_info*
  uint64_t                      notification;
  bool                          processDetachedFromSharedRegion;
} dyld_all_image_infos64;

// some typedefs to isolate 64/32 bit differences
#ifdef __LP64__
typedef mach_header_64 breakpad_mach_header;
typedef segment_command_64 breakpad_mach_segment_command;
#else
typedef mach_header breakpad_mach_header;
typedef segment_command breakpad_mach_segment_command;
#endif

// Helper functions to deal with 32-bit/64-bit Mach-O differences.
class DynamicImage;
template<typename MachBits>
bool FindTextSection(DynamicImage& image);

template<typename MachBits>
uint32_t GetFileTypeFromHeader(DynamicImage& image);

//==============================================================================
// Represents a single dynamically loaded mach-o image
class DynamicImage {
 public:
  DynamicImage(uint8_t* header,     // data is copied
               size_t header_size,  // includes load commands
               uint64_t load_address,
               string file_path,
               uintptr_t image_mod_date,
               mach_port_t task,
               cpu_type_t cpu_type)
    : header_(header, header + header_size),
      header_size_(header_size),
      load_address_(load_address),
      vmaddr_(0),
      vmsize_(0),
      slide_(0),
      version_(0),
      file_path_(file_path),
      file_mod_date_(image_mod_date),
      task_(task),
      cpu_type_(cpu_type) {
    CalculateMemoryAndVersionInfo();
  }

  // Size of mach_header plus load commands
  size_t GetHeaderSize() const {return header_.size();}

  // Full path to mach-o binary
  string GetFilePath() {return file_path_;}

  uint64_t GetModDate() const {return file_mod_date_;}

  // Actual address where the image was loaded
  uint64_t GetLoadAddress() const {return load_address_;}

  // Address where the image should be loaded
  mach_vm_address_t GetVMAddr() const {return vmaddr_;}

  // Difference between GetLoadAddress() and GetVMAddr()
  ptrdiff_t GetVMAddrSlide() const {return slide_;}

  // Size of the image
  mach_vm_size_t GetVMSize() const {return vmsize_;}

  // Task owning this loaded image
  mach_port_t GetTask() {return task_;}

  // CPU type of the task
  cpu_type_t GetCPUType() {return cpu_type_;}

  // filetype from the Mach-O header.
  uint32_t GetFileType();

  // Return true if the task is a 64-bit architecture.
  bool Is64Bit() { return (GetCPUType() & CPU_ARCH_ABI64) == CPU_ARCH_ABI64; }

  uint32_t GetVersion() {return version_;}
  // For sorting
  bool operator<(const DynamicImage& inInfo) {
    return GetLoadAddress() < inInfo.GetLoadAddress();
  }

  // Sanity checking
  bool IsValid() {return GetVMSize() != 0;}

 private:
  DynamicImage(const DynamicImage&);
  DynamicImage& operator=(const DynamicImage&);

  friend class DynamicImages;
  template<typename MachBits>
  friend bool FindTextSection(DynamicImage& image);
  template<typename MachBits>
  friend uint32_t GetFileTypeFromHeader(DynamicImage& image);

  // Initializes vmaddr_, vmsize_, and slide_
  void CalculateMemoryAndVersionInfo();

  const vector<uint8_t>   header_;        // our local copy of the header
  size_t                  header_size_;    // mach_header plus load commands
  uint64_t                load_address_;   // base address image is mapped into
  mach_vm_address_t       vmaddr_;
  mach_vm_size_t          vmsize_;
  ptrdiff_t               slide_;
  uint32_t                version_;        // Dylib version
  string                  file_path_;     // path dyld used to load the image
  uintptr_t               file_mod_date_;  // time_t of image file

  mach_port_t             task_;
  cpu_type_t              cpu_type_;        // CPU type of task_
};

//==============================================================================
// DynamicImageRef is just a simple wrapper for a pointer to
// DynamicImage.  The reason we use it instead of a simple typedef is so
// that we can use stl::sort() on a vector of DynamicImageRefs
// and simple class pointers can't implement operator<().
//
class DynamicImageRef {
 public:
  explicit DynamicImageRef(DynamicImage* inP) : p(inP) {}
  // The copy constructor is required by STL
  DynamicImageRef(const DynamicImageRef& inRef) = default;
  DynamicImageRef& operator=(const DynamicImageRef& inRef) = default;

  bool operator<(const DynamicImageRef& inRef) const {
    return (*const_cast<DynamicImageRef*>(this)->p)
      < (*const_cast<DynamicImageRef&>(inRef).p);
  }

  bool operator==(const DynamicImageRef& inInfo) const {
    return (*const_cast<DynamicImageRef*>(this)->p).GetLoadAddress() ==
        (*const_cast<DynamicImageRef&>(inInfo)).GetLoadAddress();
  }

  // Be just like DynamicImage*
  DynamicImage* operator->() {return p;}
  operator DynamicImage*() {return p;}

 private:
  DynamicImage* p;
};

// Helper function to deal with 32-bit/64-bit Mach-O differences.
class DynamicImages;
template<typename MachBits>
void ReadImageInfo(DynamicImages& images, uint64_t image_list_address);

//==============================================================================
// An object of type DynamicImages may be created to allow introspection of
// an arbitrary task's dynamically loaded mach-o binaries.  This makes the
// assumption that the current task has send rights to the target task.
class DynamicImages {
 public:
  explicit DynamicImages(mach_port_t task);

  ~DynamicImages() {
    for (int i = 0; i < GetImageCount(); ++i) {
      delete image_list_[i];
    }
  }

  // Returns the number of dynamically loaded mach-o images.
  int GetImageCount() const {return static_cast<int>(image_list_.size());}

  // Returns an individual image.
  DynamicImage* GetImage(int i) {
    if (i < (int)image_list_.size()) {
      return image_list_[i];
    }
    return NULL;
  }

  // Returns the image corresponding to the main executable.
  DynamicImage* GetExecutableImage();
  int GetExecutableImageIndex();

  // Returns the task which we're looking at.
  mach_port_t GetTask() const {return task_;}

  // CPU type of the task
  cpu_type_t GetCPUType() {return cpu_type_;}

  // Return true if the task is a 64-bit architecture.
  bool Is64Bit() { return (GetCPUType() & CPU_ARCH_ABI64) == CPU_ARCH_ABI64; }

  // Determine the CPU type of the task being dumped.
  static cpu_type_t DetermineTaskCPUType(task_t task);

  // Get the native CPU type of this task.
  static cpu_type_t GetNativeCPUType() {
#if defined(__i386__)
    return CPU_TYPE_I386;
#elif defined(__x86_64__)
    return CPU_TYPE_X86_64;
#elif defined(__ppc__)
    return CPU_TYPE_POWERPC;
#elif defined(__ppc64__)
    return CPU_TYPE_POWERPC64;
#elif defined(__arm__)
    return CPU_TYPE_ARM;
#elif defined(__aarch64__)
    return CPU_TYPE_ARM64;
#else
#error "GetNativeCPUType not implemented for this architecture"
#endif
  }

 private:
  template<typename MachBits>
  friend void ReadImageInfo(DynamicImages& images, uint64_t image_list_address);

  bool IsOurTask() {return task_ == mach_task_self();}

  // Initialization
  void ReadImageInfoForTask();
  uint64_t GetDyldAllImageInfosPointer();

  mach_port_t              task_;
  cpu_type_t               cpu_type_;  // CPU type of task_
  vector<DynamicImageRef>  image_list_;
};

// Fill bytes with the contents of memory at a particular
// location in another task.
kern_return_t ReadTaskMemory(task_port_t target_task,
                             const uint64_t address,
                             size_t length,
                             vector<uint8_t>& bytes);

}   // namespace google_breakpad

#endif // CLIENT_MAC_HANDLER_DYNAMIC_IMAGES_H__
