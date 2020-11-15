// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "snapshot/mac/mach_o_image_segment_reader.h"

#include <mach-o/loader.h>
#include <string.h>

#include <utility>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "snapshot/mac/process_reader_mac.h"
#include "util/mac/checked_mach_address_range.h"
#include "util/mac/mac_util.h"
#include "util/stdlib/strnlen.h"

namespace crashpad {

namespace {

std::string SizeLimitedCString(const char* c_string, size_t max_length) {
  return std::string(c_string, strnlen(c_string, max_length));
}

}  // namespace

bool IsMalformedCLKernelsModule(uint32_t mach_o_file_type,
                                const std::string& module_name,
                                bool* has_timestamp) {
  if (mach_o_file_type != MH_BUNDLE) {
    return false;
  }

  if (module_name == "cl_kernels") {
    if (MacOSXMinorVersion() >= 10) {
      if (has_timestamp) {
        *has_timestamp = false;
      }
      return true;
    }
    return false;
  }

  static const char kCvmsObjectPathPrefix[] =
      "/private/var/db/CVMS/cvmsCodeSignObj";
  if (module_name.compare(
          0, strlen(kCvmsObjectPathPrefix), kCvmsObjectPathPrefix) == 0 &&
      MacOSXMinorVersion() >= 14) {
    if (has_timestamp) {
      *has_timestamp = true;
    }
    return true;
  }

  return false;
}

MachOImageSegmentReader::MachOImageSegmentReader()
    : segment_command_(),
      sections_(),
      section_map_(),
      slide_(0),
      initialized_(),
      initialized_slide_() {
}

MachOImageSegmentReader::~MachOImageSegmentReader() {
}

bool MachOImageSegmentReader::Initialize(ProcessReaderMac* process_reader,
                                         mach_vm_address_t load_command_address,
                                         const std::string& load_command_info,
                                         const std::string& module_name,
                                         uint32_t file_type) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!segment_command_.Read(process_reader, load_command_address)) {
    LOG(WARNING) << "could not read segment_command" << load_command_info;
    return false;
  }

  const uint32_t kExpectedSegmentCommand =
      process_reader->Is64Bit() ? LC_SEGMENT_64 : LC_SEGMENT;
  DCHECK_EQ(segment_command_.cmd, kExpectedSegmentCommand);
  DCHECK_GE(segment_command_.cmdsize, segment_command_.Size());
  const size_t kSectionStructSize =
      process_types::section::ExpectedSize(process_reader);
  const size_t kRequiredSize =
      segment_command_.Size() + segment_command_.nsects * kSectionStructSize;
  if (segment_command_.cmdsize < kRequiredSize) {
    LOG(WARNING) << base::StringPrintf(
                        "segment command cmdsize 0x%x insufficient for %u "
                        "section%s (0x%zx)",
                        segment_command_.cmdsize,
                        segment_command_.nsects,
                        segment_command_.nsects == 1 ? "" : "s",
                        kRequiredSize) << load_command_info;
    return false;
  }

  std::string segment_name = NameInternal();
  std::string segment_info = base::StringPrintf(
      ", segment %s%s", segment_name.c_str(), load_command_info.c_str());

  // This checks the unslid segment range. The slid range (as loaded into
  // memory) will be checked later by MachOImageReader.
  CheckedMachAddressRange segment_range(process_reader->Is64Bit(),
                                        segment_command_.vmaddr,
                                        segment_command_.vmsize);
  if (!segment_range.IsValid()) {
    LOG(WARNING) << base::StringPrintf("invalid segment range 0x%llx + 0x%llx",
                                       segment_command_.vmaddr,
                                       segment_command_.vmsize) << segment_info;
    return false;
  }

  sections_.resize(segment_command_.nsects);
  if (!sections_.empty() &&
      !process_types::section::ReadArrayInto(
          process_reader,
          load_command_address + segment_command_.Size(),
          segment_command_.nsects,
          &sections_[0])) {
    LOG(WARNING) << "could not read sections" << segment_info;
    return false;
  }

  for (size_t section_index = 0;
       section_index < sections_.size();
       ++section_index) {
    const process_types::section& section = sections_[section_index];
    std::string section_segment_name = SegmentNameString(section.segname);
    std::string section_name = SectionNameString(section.sectname);
    std::string section_full_name =
        SegmentAndSectionNameString(section.segname, section.sectname);

    std::string section_info = base::StringPrintf(", section %s %zu/%zu%s",
                                                  section_full_name.c_str(),
                                                  section_index,
                                                  sections_.size(),
                                                  load_command_info.c_str());

    // cl_kernels modules (for OpenCL) aren’t ld output, and they’re formatted
    // incorrectly on OS X 10.10 and later. Because at least one cl_kernels
    // module will commonly be found in a process, and sometimes more will be,
    // tolerate this quirk.
    //
    // https://openradar.appspot.com/20239912
    if (section_segment_name != segment_name &&
        !(IsMalformedCLKernelsModule(file_type, module_name, nullptr) &&
          segment_name == SEG_TEXT &&
          section_segment_name == "__LD" &&
          section_name == "__compact_unwind" &&
          (section.flags & S_ATTR_DEBUG))) {
      LOG(WARNING) << "section.segname incorrect in segment " << segment_name
                   << section_info;
      return false;
    }

    CheckedMachAddressRange section_range(
        process_reader->Is64Bit(), section.addr, section.size);
    if (!section_range.IsValid()) {
      LOG(WARNING) << base::StringPrintf(
                          "invalid section range 0x%llx + 0x%llx",
                          section.addr,
                          section.size) << section_info;
      return false;
    }

    if (!segment_range.ContainsRange(section_range)) {
      LOG(WARNING) << base::StringPrintf(
                          "section at 0x%llx + 0x%llx outside of segment at "
                          "0x%llx + 0x%llx",
                          section.addr,
                          section.size,
                          segment_command_.vmaddr,
                          segment_command_.vmsize) << section_info;
      return false;
    }

    const auto insert_result =
        section_map_.insert(std::make_pair(section_name, section_index));
    if (!insert_result.second) {
      LOG(WARNING) << base::StringPrintf("duplicate section name at %zu",
                                         insert_result.first->second)
                   << section_info;
      return false;
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

std::string MachOImageSegmentReader::Name() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return NameInternal();
}

mach_vm_address_t MachOImageSegmentReader::Address() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  INITIALIZATION_STATE_DCHECK_VALID(initialized_slide_);
  return vmaddr() + (SegmentSlides() ? slide_ : 0);
}

mach_vm_size_t MachOImageSegmentReader::Size() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  INITIALIZATION_STATE_DCHECK_VALID(initialized_slide_);
  return vmsize() + (SegmentSlides() ? 0 : slide_);
}

const process_types::section* MachOImageSegmentReader::GetSectionByName(
    const std::string& section_name,
    mach_vm_address_t* address) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  const auto& iterator = section_map_.find(section_name);
  if (iterator == section_map_.end()) {
    return nullptr;
  }

  return GetSectionAtIndex(iterator->second, address);
}

const process_types::section* MachOImageSegmentReader::GetSectionAtIndex(
    size_t index,
    mach_vm_address_t* address) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  CHECK_LT(index, sections_.size());

  const process_types::section* section = &sections_[index];

  if (address) {
    INITIALIZATION_STATE_DCHECK_VALID(initialized_slide_);
    *address = section->addr + (SegmentSlides() ? slide_ : 0);
  }

  return section;
}

bool MachOImageSegmentReader::SegmentSlides() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // These are the same rules that the kernel uses to identify __PAGEZERO. See
  // 10.9.4 xnu-2422.110.17/bsd/kern/mach_loader.c load_segment().
  return !(segment_command_.vmaddr == 0 && segment_command_.filesize == 0 &&
           segment_command_.vmsize != 0 &&
           (segment_command_.initprot & VM_PROT_ALL) == VM_PROT_NONE &&
           (segment_command_.maxprot & VM_PROT_ALL) == VM_PROT_NONE);
}

// static
std::string MachOImageSegmentReader::SegmentNameString(
    const char* segment_name_c) {
  // This is used to interpret the segname field of both the segment_command and
  // section structures, so be sure that they’re identical.
  static_assert(sizeof(process_types::segment_command::segname) ==
                    sizeof(process_types::section::segname),
                "sizes must be equal");

  return SizeLimitedCString(segment_name_c,
                            sizeof(process_types::segment_command::segname));
}

// static
std::string MachOImageSegmentReader::SectionNameString(
    const char* section_name_c) {
  return SizeLimitedCString(section_name_c,
                            sizeof(process_types::section::sectname));
}

// static
std::string MachOImageSegmentReader::SegmentAndSectionNameString(
    const char* segment_name_c,
    const char* section_name_c) {
  return base::StringPrintf("%s,%s",
                            SegmentNameString(segment_name_c).c_str(),
                            SectionNameString(section_name_c).c_str());
}

std::string MachOImageSegmentReader::NameInternal() const {
  return SegmentNameString(segment_command_.segname);
}

void MachOImageSegmentReader::SetSlide(mach_vm_size_t slide) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_slide_);
  slide_ = slide;
  INITIALIZATION_STATE_SET_VALID(initialized_slide_);
}

}  // namespace crashpad
