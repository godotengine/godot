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

#include "snapshot/mac/mach_o_image_reader.h"

#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <string.h>

#include <limits>
#include <utility>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "client/crashpad_info.h"
#include "snapshot/mac/mach_o_image_segment_reader.h"
#include "snapshot/mac/mach_o_image_symbol_table_reader.h"
#include "snapshot/mac/process_reader_mac.h"
#include "util/mac/checked_mach_address_range.h"
#include "util/misc/implicit_cast.h"

namespace {

constexpr uint32_t kInvalidSegmentIndex = std::numeric_limits<uint32_t>::max();

}  // namespace

namespace crashpad {

MachOImageReader::MachOImageReader()
    : segments_(),
      segment_map_(),
      module_name_(),
      module_info_(),
      dylinker_name_(),
      uuid_(),
      address_(0),
      size_(0),
      slide_(0),
      source_version_(0),
      symtab_command_(),
      dysymtab_command_(),
      symbol_table_(),
      id_dylib_command_(),
      process_reader_(nullptr),
      file_type_(0),
      initialized_(),
      symbol_table_initialized_() {
}

MachOImageReader::~MachOImageReader() {
}

bool MachOImageReader::Initialize(ProcessReaderMac* process_reader,
                                  mach_vm_address_t address,
                                  const std::string& name) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_reader_ = process_reader;
  address_ = address;
  module_name_ = name;

  module_info_ =
      base::StringPrintf(", module %s, address 0x%llx", name.c_str(), address);

  process_types::mach_header mach_header;
  if (!mach_header.Read(process_reader, address)) {
    LOG(WARNING) << "could not read mach_header" << module_info_;
    return false;
  }

  const bool is_64_bit = process_reader->Is64Bit();
  const uint32_t kExpectedMagic = is_64_bit ? MH_MAGIC_64 : MH_MAGIC;
  if (mach_header.magic != kExpectedMagic) {
    LOG(WARNING) << base::StringPrintf("unexpected mach_header::magic 0x%08x",
                                       mach_header.magic) << module_info_;
    return false;
  }

  switch (mach_header.filetype) {
    case MH_EXECUTE:
    case MH_DYLIB:
    case MH_DYLINKER:
    case MH_BUNDLE:
      file_type_ = mach_header.filetype;
      break;
    default:
      LOG(WARNING) << base::StringPrintf(
                          "unexpected mach_header::filetype 0x%08x",
                          mach_header.filetype) << module_info_;
      return false;
  }

  const uint32_t kExpectedSegmentCommand =
      is_64_bit ? LC_SEGMENT_64 : LC_SEGMENT;
  const uint32_t kUnexpectedSegmentCommand =
      is_64_bit ? LC_SEGMENT : LC_SEGMENT_64;

  const struct {
    // Which method to call when encountering a load command matching |command|.
    bool (MachOImageReader::*function)(mach_vm_address_t, const std::string&);

    // The minimum size that may be allotted to store the load command.
    size_t size;

    // The load command to match.
    uint32_t command;

    // True if the load command must not appear more than one time.
    bool singleton;
  } kLoadCommandReaders[] = {
    {
      &MachOImageReader::ReadSegmentCommand,
      process_types::segment_command::ExpectedSize(process_reader),
      kExpectedSegmentCommand,
      false,
    },
    {
      &MachOImageReader::ReadSymTabCommand,
      process_types::symtab_command::ExpectedSize(process_reader),
      LC_SYMTAB,
      true,
    },
    {
      &MachOImageReader::ReadDySymTabCommand,
      process_types::symtab_command::ExpectedSize(process_reader),
      LC_DYSYMTAB,
      true,
    },
    {
      &MachOImageReader::ReadIdDylibCommand,
      process_types::dylib_command::ExpectedSize(process_reader),
      LC_ID_DYLIB,
      true,
    },
    {
      &MachOImageReader::ReadDylinkerCommand,
      process_types::dylinker_command::ExpectedSize(process_reader),
      LC_LOAD_DYLINKER,
      true,
    },
    {
      &MachOImageReader::ReadDylinkerCommand,
      process_types::dylinker_command::ExpectedSize(process_reader),
      LC_ID_DYLINKER,
      true,
    },
    {
      &MachOImageReader::ReadUUIDCommand,
      process_types::uuid_command::ExpectedSize(process_reader),
      LC_UUID,
      true,
    },
    {
      &MachOImageReader::ReadSourceVersionCommand,
      process_types::source_version_command::ExpectedSize(process_reader),
      LC_SOURCE_VERSION,
      true,
    },

    // When reading a 64-bit process, no 32-bit segment commands should be
    // present, and vice-versa.
    {
      &MachOImageReader::ReadUnexpectedCommand,
      process_types::load_command::ExpectedSize(process_reader),
      kUnexpectedSegmentCommand,
      false,
    },
  };

  // This vector is parallel to the kLoadCommandReaders array, and tracks
  // whether a singleton load command matching the |command| field has been
  // found yet.
  std::vector<uint32_t> singleton_indices(arraysize(kLoadCommandReaders),
                                          kInvalidSegmentIndex);

  size_t offset = mach_header.Size();
  const mach_vm_address_t kLoadCommandAddressLimit =
      address + offset + mach_header.sizeofcmds;

  for (uint32_t load_command_index = 0;
       load_command_index < mach_header.ncmds;
       ++load_command_index) {
    mach_vm_address_t load_command_address = address + offset;
    std::string load_command_info = base::StringPrintf(", load command %u/%u%s",
                                                       load_command_index,
                                                       mach_header.ncmds,
                                                       module_info_.c_str());

    process_types::load_command load_command;

    // Make sure that the basic load command structure doesn’t overflow the
    // space allotted for load commands.
    if (load_command_address + load_command.ExpectedSize(process_reader) >
            kLoadCommandAddressLimit) {
      LOG(WARNING) << base::StringPrintf(
                          "load_command at 0x%llx exceeds sizeofcmds 0x%x",
                          load_command_address,
                          mach_header.sizeofcmds) << load_command_info;
      return false;
    }

    if (!load_command.Read(process_reader, load_command_address)) {
      LOG(WARNING) << "could not read load_command" << load_command_info;
      return false;
    }

    load_command_info = base::StringPrintf(", load command 0x%x %u/%u%s",
                                           load_command.cmd,
                                           load_command_index,
                                           mach_header.ncmds,
                                           module_info_.c_str());

    // Now that the load command’s stated size is known, make sure that it
    // doesn’t overflow the space allotted for load commands.
    if (load_command_address + load_command.cmdsize >
            kLoadCommandAddressLimit) {
      LOG(WARNING)
          << base::StringPrintf(
                 "load_command at 0x%llx cmdsize 0x%x exceeds sizeofcmds 0x%x",
                 load_command_address,
                 load_command.cmdsize,
                 mach_header.sizeofcmds) << load_command_info;
      return false;
    }

    for (size_t reader_index = 0;
         reader_index < arraysize(kLoadCommandReaders);
         ++reader_index) {
      if (load_command.cmd != kLoadCommandReaders[reader_index].command) {
        continue;
      }

      if (load_command.cmdsize < kLoadCommandReaders[reader_index].size) {
        LOG(WARNING) << base::StringPrintf(
                            "load command cmdsize 0x%x insufficient for 0x%zx",
                            load_command.cmdsize,
                            kLoadCommandReaders[reader_index].size)
                     << load_command_info;
        return false;
      }

      if (kLoadCommandReaders[reader_index].singleton) {
        if (singleton_indices[reader_index] != kInvalidSegmentIndex) {
          LOG(WARNING) << "duplicate load command at "
                       << singleton_indices[reader_index] << load_command_info;
          return false;
        }

        singleton_indices[reader_index] = load_command_index;
      }

      if (!((this)->*(kLoadCommandReaders[reader_index].function))(
              load_command_address, load_command_info)) {
        return false;
      }

      break;
    }

    offset += load_command.cmdsize;
  }

  // Now that the slide is known, push it into the segments.
  for (const auto& segment : segments_) {
    segment->SetSlide(slide_);

    // This was already checked for the unslid values while the segments were
    // read, but now it’s possible to check the slid values too. The individual
    // sections don’t need to be checked because they were verified to be
    // contained within their respective segments when the segments were read.
    mach_vm_address_t slid_segment_address = segment->Address();
    mach_vm_size_t slid_segment_size = segment->Size();
    CheckedMachAddressRange slid_segment_range(
        process_reader_->Is64Bit(), slid_segment_address, slid_segment_size);
    if (!slid_segment_range.IsValid()) {
      LOG(WARNING) << base::StringPrintf(
                          "invalid slid segment range 0x%llx + 0x%llx, "
                          "segment ",
                          slid_segment_address,
                          slid_segment_size) << segment->Name() << module_info_;
      return false;
    }
  }

  if (!segment_map_.count(SEG_TEXT)) {
    // The __TEXT segment is required. Even a module with no executable code
    // will have a __TEXT segment encompassing the Mach-O header and load
    // commands. Without a __TEXT segment, |size_| will not have been computed.
    LOG(WARNING) << "no " SEG_TEXT " segment" << module_info_;
    return false;
  }

  if (mach_header.filetype == MH_DYLIB && !id_dylib_command_) {
    // This doesn’t render a module unusable, it’s just weird and worth noting.
    LOG(INFO) << "no LC_ID_DYLIB" << module_info_;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

const MachOImageSegmentReader* MachOImageReader::GetSegmentByName(
    const std::string& segment_name) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  const auto& iterator = segment_map_.find(segment_name);
  if (iterator == segment_map_.end()) {
    return nullptr;
  }

  return segments_[iterator->second].get();
}

const process_types::section* MachOImageReader::GetSectionByName(
    const std::string& segment_name,
    const std::string& section_name,
    mach_vm_address_t* address) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  const MachOImageSegmentReader* segment = GetSegmentByName(segment_name);
  if (!segment) {
    return nullptr;
  }

  return segment->GetSectionByName(section_name, address);
}

const process_types::section* MachOImageReader::GetSectionAtIndex(
    size_t index,
    const MachOImageSegmentReader** containing_segment,
    mach_vm_address_t* address) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  static_assert(NO_SECT == 0, "NO_SECT must be zero");
  if (index == NO_SECT) {
    LOG(WARNING) << "section index " << index << " out of range";
    return nullptr;
  }

  // Switch to a more comfortable 0-based index.
  size_t local_index = index - 1;

  for (const auto& segment : segments_) {
    size_t nsects = segment->nsects();
    if (local_index < nsects) {
      const process_types::section* section =
          segment->GetSectionAtIndex(local_index, address);

      if (containing_segment) {
        *containing_segment = segment.get();
      }

      return section;
    }

    local_index -= nsects;
  }

  LOG(WARNING) << "section index " << index << " out of range";
  return nullptr;
}

bool MachOImageReader::LookUpExternalDefinedSymbol(
    const std::string& name,
    mach_vm_address_t* value) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (symbol_table_initialized_.is_uninitialized()) {
    InitializeSymbolTable();
  }

  if (!symbol_table_initialized_.is_valid() || !symbol_table_) {
    return false;
  }

  const MachOImageSymbolTableReader::SymbolInformation* symbol_info =
      symbol_table_->LookUpExternalDefinedSymbol(name);
  if (!symbol_info) {
    return false;
  }

  if (symbol_info->section == NO_SECT) {
    // This is an absolute (N_ABS) symbol, which requires no further validation
    // or processing.
    *value = symbol_info->value;
    return true;
  }

  // This is a symbol defined in a particular section, so make sure that it’s
  // valid for that section and fix it up for any “slide” as needed.

  mach_vm_address_t section_address;
  const MachOImageSegmentReader* segment;
  const process_types::section* section =
      GetSectionAtIndex(symbol_info->section, &segment, &section_address);
  if (!section) {
    return false;
  }

  mach_vm_address_t slid_value =
      symbol_info->value + (segment->SegmentSlides() ? slide_ : 0);

  // The __mh_execute_header (_MH_EXECUTE_SYM) symbol is weird. In
  // position-independent executables, it shows up in the symbol table as a
  // symbol in section 1, although it’s not really in that section. It points to
  // the mach_header[_64], which is the beginning of the __TEXT segment, and the
  // __text section normally begins after the load commands in the __TEXT
  // segment. The range check below will fail for this symbol, because it’s not
  // really in the section it claims to be in. See Xcode 5.1
  // ld64-236.3/src/ld/OutputFile.cpp ld::tool::OutputFile::buildSymbolTable().
  // There, ld takes symbols that refer to anything in the mach_header[_64] and
  // marks them as being in section 1. Here, section 1 is treated in this same
  // special way as long as it’s in the __TEXT segment that begins at the start
  // of the image, which is normally the case, and as long as the symbol’s value
  // is the base of the image.
  //
  // This only happens for PIE executables, because __mh_execute_header needs
  // to slide. In non-PIE executables, __mh_execute_header is an absolute
  // symbol.
  CheckedMachAddressRange section_range(
      process_reader_->Is64Bit(), section_address, section->size);
  if (!section_range.ContainsValue(slid_value) &&
      !(symbol_info->section == 1 && segment->Name() == SEG_TEXT &&
        slid_value == Address())) {
    std::string section_name_full =
        MachOImageSegmentReader::SegmentAndSectionNameString(section->segname,
                                                             section->sectname);
    LOG(WARNING) << base::StringPrintf(
                        "symbol %s (0x%llx) outside of section %s (0x%llx + "
                        "0x%llx)",
                        name.c_str(),
                        slid_value,
                        section_name_full.c_str(),
                        section_address,
                        section->size) << module_info_;
    return false;
  }

  *value = slid_value;
  return true;
}

uint32_t MachOImageReader::DylibVersion() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  DCHECK_EQ(FileType(), implicit_cast<uint32_t>(MH_DYLIB));

  if (id_dylib_command_) {
    return id_dylib_command_->dylib_current_version;
  }

  // In case this was a weird dylib without an LC_ID_DYLIB command.
  return 0;
}

void MachOImageReader::UUID(crashpad::UUID* uuid) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  memcpy(uuid, &uuid_, sizeof(uuid_));
}

bool MachOImageReader::GetCrashpadInfo(
    process_types::CrashpadInfo* crashpad_info) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  mach_vm_address_t crashpad_info_address;
  const process_types::section* crashpad_info_section =
      GetSectionByName(SEG_DATA, "crashpad_info", &crashpad_info_address);
  if (!crashpad_info_section) {
    return false;
  }

  if (crashpad_info_section->size <
      crashpad_info->MinimumSize(process_reader_)) {
    LOG(WARNING) << "small crashpad info section size "
                 << crashpad_info_section->size << module_info_;
    return false;
  }

  // This Read() will zero out anything beyond the structure’s declared size.
  if (!crashpad_info->Read(process_reader_, crashpad_info_address)) {
    LOG(WARNING) << "could not read crashpad info" << module_info_;
    return false;
  }

  if (crashpad_info->signature != CrashpadInfo::kSignature ||
      crashpad_info->version != 1) {
    LOG(WARNING) << base::StringPrintf(
        "unexpected crashpad info signature 0x%x, version %u%s",
        crashpad_info->signature,
        crashpad_info->version,
        module_info_.c_str());
    return false;
  }

  // Don’t require strict equality, to leave wiggle room for sloppy linkers.
  if (crashpad_info->size > crashpad_info_section->size) {
    LOG(WARNING) << "crashpad info struct size " << crashpad_info->size
                 << " large for section size " << crashpad_info_section->size
                 << module_info_;
    return false;
  }

  if (crashpad_info->size > crashpad_info->ExpectedSize(process_reader_)) {
    // This isn’t strictly a problem, because unknown fields will simply be
    // ignored, but it may be of diagnostic interest.
    LOG(INFO) << "large crashpad info size " << crashpad_info->size
              << module_info_;
  }

  return true;
}

template <typename T>
bool MachOImageReader::ReadLoadCommand(mach_vm_address_t load_command_address,
                                       const std::string& load_command_info,
                                       uint32_t expected_load_command_id,
                                       T* load_command) {
  if (!load_command->Read(process_reader_, load_command_address)) {
    LOG(WARNING) << "could not read load command" << load_command_info;
    return false;
  }

  DCHECK_GE(load_command->cmdsize, load_command->Size());
  DCHECK_EQ(load_command->cmd, expected_load_command_id);
  return true;
}

bool MachOImageReader::ReadSegmentCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  size_t segment_index = segments_.size();
  segments_.push_back(std::make_unique<MachOImageSegmentReader>());
  MachOImageSegmentReader* segment = segments_.back().get();

  if (!segment->Initialize(process_reader_,
                           load_command_address,
                           load_command_info,
                           module_name_,
                           file_type_)) {
    segments_.pop_back();
    return false;
  }

  // At this point, the segment itself is considered valid, but if one of the
  // next checks fails, it will render the module invalid. If any of the next
  // checks fail, this method should return false, but it doesn’t need to bother
  // removing the segment from segments_. The segment will be properly released
  // when the image is destroyed, and the image won’t be usable because
  // initialization won’t have completed. Most importantly, leaving the segment
  // in segments_ means that no other structures (such as perhaps segment_map_)
  // become inconsistent or require cleanup.

  const std::string segment_name = segment->Name();
  const auto insert_result =
      segment_map_.insert(std::make_pair(segment_name, segment_index));
  if (!insert_result.second) {
    LOG(WARNING) << base::StringPrintf("duplicate %s segment at %zu and %zu",
                                       segment_name.c_str(),
                                       insert_result.first->second,
                                       segment_index) << load_command_info;
    return false;
  }

  if (segment_name == SEG_TEXT) {
    mach_vm_size_t vmsize = segment->vmsize();

    if (vmsize == 0) {
      LOG(WARNING) << "zero-sized " SEG_TEXT " segment" << load_command_info;
      return false;
    }

    size_ = vmsize;

    // The slide is computed as the difference between the __TEXT segment’s
    // preferred and actual load addresses. This is the same way that dyld
    // computes slide. See 10.9.2 dyld-239.4/src/dyldInitialization.cpp
    // slideOfMainExecutable().
    slide_ = address_ - segment->vmaddr();
  }

  return true;
}

bool MachOImageReader::ReadSymTabCommand(mach_vm_address_t load_command_address,
                                         const std::string& load_command_info) {
  symtab_command_.reset(new process_types::symtab_command());
  return ReadLoadCommand(load_command_address,
                         load_command_info,
                         LC_SYMTAB,
                         symtab_command_.get());
}

bool MachOImageReader::ReadDySymTabCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  dysymtab_command_.reset(new process_types::dysymtab_command());
  return ReadLoadCommand(load_command_address,
                         load_command_info,
                         LC_DYSYMTAB,
                         dysymtab_command_.get());
}

bool MachOImageReader::ReadIdDylibCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  if (file_type_ != MH_DYLIB) {
    LOG(WARNING) << base::StringPrintf(
                        "LC_ID_DYLIB inappropriate in non-dylib file type 0x%x",
                        file_type_) << load_command_info;
    return false;
  }

  DCHECK(!id_dylib_command_);
  id_dylib_command_.reset(new process_types::dylib_command());
  return ReadLoadCommand(load_command_address,
                         load_command_info,
                         LC_ID_DYLIB,
                         id_dylib_command_.get());
}

bool MachOImageReader::ReadDylinkerCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  if (file_type_ != MH_EXECUTE && file_type_ != MH_DYLINKER) {
    LOG(WARNING) << base::StringPrintf(
                        "LC_LOAD_DYLINKER/LC_ID_DYLINKER inappropriate in file "
                        "type 0x%x",
                        file_type_) << load_command_info;
    return false;
  }

  const uint32_t kExpectedCommand =
      file_type_ == MH_DYLINKER ? LC_ID_DYLINKER : LC_LOAD_DYLINKER;
  process_types::dylinker_command dylinker_command;
  if (!ReadLoadCommand(load_command_address,
                       load_command_info,
                       kExpectedCommand,
                       &dylinker_command)) {
    return false;
  }

  if (!process_reader_->Memory()->ReadCStringSizeLimited(
          load_command_address + dylinker_command.name,
          dylinker_command.cmdsize - dylinker_command.name,
          &dylinker_name_)) {
    LOG(WARNING) << "could not read dylinker_command name" << load_command_info;
    return false;
  }

  return true;
}

bool MachOImageReader::ReadUUIDCommand(mach_vm_address_t load_command_address,
                                       const std::string& load_command_info) {
  process_types::uuid_command uuid_command;
  if (!ReadLoadCommand(
          load_command_address, load_command_info, LC_UUID, &uuid_command)) {
    return false;
  }

  uuid_.InitializeFromBytes(uuid_command.uuid);
  return true;
}

bool MachOImageReader::ReadSourceVersionCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  process_types::source_version_command source_version_command;
  if (!ReadLoadCommand(load_command_address,
                       load_command_info,
                       LC_SOURCE_VERSION,
                       &source_version_command)) {
    return false;
  }

  source_version_ = source_version_command.version;
  return true;
}

bool MachOImageReader::ReadUnexpectedCommand(
    mach_vm_address_t load_command_address,
    const std::string& load_command_info) {
  LOG(WARNING) << "unexpected load command" << load_command_info;
  return false;
}

void MachOImageReader::InitializeSymbolTable() const {
  DCHECK(symbol_table_initialized_.is_uninitialized());
  symbol_table_initialized_.set_invalid();

  if (!symtab_command_) {
    // It’s technically valid for there to be no LC_SYMTAB, and in that case,
    // any symbol lookups should fail. Mark the symbol table as valid, and
    // LookUpExternalDefinedSymbol() will understand what it means when this is
    // valid but symbol_table_ is not present.
    symbol_table_initialized_.set_valid();
    return;
  }

  // Find the __LINKEDIT segment. Technically, the symbol table can be in any
  // mapped segment, but by convention, it’s in the one named __LINKEDIT.
  const MachOImageSegmentReader* linkedit_segment =
      GetSegmentByName(SEG_LINKEDIT);
  if (!linkedit_segment) {
    LOG(WARNING) << "no " SEG_LINKEDIT " segment";
    return;
  }

  symbol_table_.reset(new MachOImageSymbolTableReader());
  if (!symbol_table_->Initialize(process_reader_,
                                 symtab_command_.get(),
                                 dysymtab_command_.get(),
                                 linkedit_segment,
                                 module_info_)) {
    symbol_table_.reset();
    return;
  }

  symbol_table_initialized_.set_valid();
}

}  // namespace crashpad
