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

#include <AvailabilityMacros.h>
#include <dlfcn.h>
#include <mach-o/dyld.h>
#include <mach-o/dyld_images.h>
#include <mach-o/getsect.h>
#include <mach-o/ldsyms.h>
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <stdint.h>

#include "base/strings/stringprintf.h"
#include "build/build_config.h"
#include "client/crashpad_info.h"
#include "gtest/gtest.h"
#include "snapshot/mac/mach_o_image_segment_reader.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/mac/process_types.h"
#include "test/mac/dyld.h"
#include "util/misc/from_pointer_cast.h"
#include "util/misc/implicit_cast.h"
#include "util/misc/uuid.h"

// This file is responsible for testing MachOImageReader,
// MachOImageSegmentReader, and MachOImageSymbolTableReader.

namespace crashpad {
namespace test {
namespace {

// Native types and constants, in cases where the 32-bit and 64-bit versions
// are different.
#if defined(ARCH_CPU_64_BITS)
using MachHeader = mach_header_64;
constexpr uint32_t kMachMagic = MH_MAGIC_64;
using SegmentCommand = segment_command_64;
constexpr uint32_t kSegmentCommand = LC_SEGMENT_64;
using Section = section_64;
using Nlist = nlist_64;
#else
using MachHeader = mach_header;
constexpr uint32_t kMachMagic = MH_MAGIC;
using SegmentCommand = segment_command;
constexpr uint32_t kSegmentCommand = LC_SEGMENT;
using Section = section;

// This needs to be called “struct nlist” because “nlist” without the struct
// refers to the nlist() function.
using Nlist = struct nlist;
#endif

#if defined(ARCH_CPU_X86_64)
constexpr int kCPUType = CPU_TYPE_X86_64;
#elif defined(ARCH_CPU_X86)
constexpr int kCPUType = CPU_TYPE_X86;
#endif

// Verifies that |expect_section| and |actual_section| agree.
void ExpectSection(const Section* expect_section,
                   const process_types::section* actual_section) {
  ASSERT_TRUE(expect_section);
  ASSERT_TRUE(actual_section);

  EXPECT_EQ(
      MachOImageSegmentReader::SectionNameString(actual_section->sectname),
      MachOImageSegmentReader::SectionNameString(expect_section->sectname));
  EXPECT_EQ(
      MachOImageSegmentReader::SegmentNameString(actual_section->segname),
      MachOImageSegmentReader::SegmentNameString(expect_section->segname));
  EXPECT_EQ(actual_section->addr, expect_section->addr);
  EXPECT_EQ(actual_section->size, expect_section->size);
  EXPECT_EQ(actual_section->offset, expect_section->offset);
  EXPECT_EQ(actual_section->align, expect_section->align);
  EXPECT_EQ(actual_section->reloff, expect_section->reloff);
  EXPECT_EQ(actual_section->nreloc, expect_section->nreloc);
  EXPECT_EQ(actual_section->flags, expect_section->flags);
  EXPECT_EQ(actual_section->reserved1, expect_section->reserved1);
  EXPECT_EQ(actual_section->reserved2, expect_section->reserved2);
}

// Verifies that |expect_segment| is a valid Mach-O segment load command for the
// current system by checking its |cmd| field. Then, verifies that the
// information in |actual_segment| matches that in |expect_segment|. The
// |segname|, |vmaddr|, |vmsize|, and |fileoff| fields are examined. Each
// section within the segment is also examined by calling ExpectSection().
// Access to each section via both MachOImageSegmentReader::GetSectionByName()
// and MachOImageReader::GetSectionByName() is verified, expecting that each
// call produces the same section. Segment and section data addresses are
// verified against data obtained by calling getsegmentdata() and
// getsectiondata(). The segment is checked to make sure that it behaves
// correctly when attempting to look up a nonexistent section by name.
// |section_index| is used to track the last-used section index in an image on
// entry, and is reset to the last-used section index on return after the
// sections are processed. This is used to test that
// MachOImageReader::GetSectionAtIndex() returns the correct result.
void ExpectSegmentCommand(const SegmentCommand* expect_segment,
                          const MachHeader* expect_image,
                          const MachOImageSegmentReader* actual_segment,
                          const MachOImageReader* actual_image,
                          size_t* section_index) {
  ASSERT_TRUE(expect_segment);
  ASSERT_TRUE(actual_segment);

  EXPECT_EQ(expect_segment->cmd, kSegmentCommand);

  std::string segment_name = actual_segment->Name();
  EXPECT_EQ(
      segment_name,
      MachOImageSegmentReader::SegmentNameString(expect_segment->segname));
  EXPECT_EQ(actual_segment->vmaddr(), expect_segment->vmaddr);
  EXPECT_EQ(actual_segment->vmsize(), expect_segment->vmsize);
  EXPECT_EQ(actual_segment->fileoff(), expect_segment->fileoff);

  if (actual_segment->SegmentSlides()) {
    EXPECT_EQ(actual_segment->vmaddr() + actual_image->Slide(),
              actual_segment->Address());

    unsigned long expect_segment_size;
    const uint8_t* expect_segment_data = getsegmentdata(
        expect_image, segment_name.c_str(), &expect_segment_size);
    mach_vm_address_t expect_segment_address =
        FromPointerCast<mach_vm_address_t>(expect_segment_data);
    EXPECT_EQ(actual_segment->Address(), expect_segment_address);
    EXPECT_EQ(actual_segment->vmsize(), expect_segment_size);
    EXPECT_EQ(actual_segment->Size(), actual_segment->vmsize());
  } else {
    // getsegmentdata() doesn’t return appropriate data for the __PAGEZERO
    // segment because getsegmentdata() always adjusts for slide, but the
    // __PAGEZERO segment never slides, it just grows. Skip the getsegmentdata()
    // check for that segment according to the same rules that the kernel uses
    // to identify __PAGEZERO. See 10.9.4 xnu-2422.110.17/bsd/kern/mach_loader.c
    // load_segment().
    EXPECT_EQ(actual_segment->vmaddr(), actual_segment->Address());
    EXPECT_EQ(actual_segment->Size(),
              actual_segment->vmsize() + actual_image->Slide());
  }

  ASSERT_EQ(actual_segment->nsects(), expect_segment->nsects);

  // Make sure that the expected load command is big enough for the number of
  // sections that it claims to have, and set up a pointer to its first section
  // structure.
  ASSERT_EQ(expect_segment->cmdsize,
            sizeof(*expect_segment) + expect_segment->nsects * sizeof(Section));
  const Section* expect_sections =
      reinterpret_cast<const Section*>(&expect_segment[1]);

  for (size_t index = 0; index < actual_segment->nsects(); ++index) {
    const Section* expect_section = &expect_sections[index];
    const process_types::section* actual_section =
        actual_segment->GetSectionAtIndex(index, nullptr);
    ASSERT_NO_FATAL_FAILURE(
        ExpectSection(&expect_sections[index], actual_section));

    // Make sure that the section is accessible by GetSectionByName as well.
    std::string section_name =
        MachOImageSegmentReader::SectionNameString(expect_section->sectname);
    const process_types::section* actual_section_by_name =
        actual_segment->GetSectionByName(section_name, nullptr);
    EXPECT_EQ(actual_section_by_name, actual_section);

    // Make sure that the section is accessible by the parent MachOImageReader’s
    // GetSectionByName.
    mach_vm_address_t actual_section_address;
    const process_types::section* actual_section_from_image_by_name =
        actual_image->GetSectionByName(
            segment_name, section_name, &actual_section_address);
    EXPECT_EQ(actual_section_from_image_by_name, actual_section);

    if (actual_segment->SegmentSlides()) {
      EXPECT_EQ(actual_section->addr + actual_image->Slide(),
                actual_section_address);

      unsigned long expect_section_size;
      const uint8_t* expect_section_data = getsectiondata(expect_image,
                                                          segment_name.c_str(),
                                                          section_name.c_str(),
                                                          &expect_section_size);
      mach_vm_address_t expect_section_address =
          FromPointerCast<mach_vm_address_t>(expect_section_data);
      EXPECT_EQ(actual_section_address, expect_section_address);
      EXPECT_EQ(actual_section->size, expect_section_size);
    } else {
      EXPECT_EQ(actual_section->addr, actual_section_address);
    }

    // Test the parent MachOImageReader’s GetSectionAtIndex as well.
    const MachOImageSegmentReader* containing_segment;
    mach_vm_address_t actual_section_address_at_index;
    const process_types::section* actual_section_from_image_at_index =
        actual_image->GetSectionAtIndex(++(*section_index),
                                        &containing_segment,
                                        &actual_section_address_at_index);
    EXPECT_EQ(actual_section_from_image_at_index, actual_section);
    EXPECT_EQ(containing_segment, actual_segment);
    EXPECT_EQ(actual_section_address_at_index, actual_section_address);
  }

  EXPECT_EQ(actual_segment->GetSectionByName("NoSuchSection", nullptr),
            nullptr);
}

// Walks through the load commands of |expect_image|, finding all of the
// expected segment commands. For each expected segment command, calls
// actual_image->GetSegmentByName() to obtain an actual segment command, and
// calls ExpectSegmentCommand() to compare the expected and actual segments. A
// series of by-name lookups is also performed on the segment to ensure that it
// behaves correctly when attempting to look up segment and section names that
// are not present. |test_section_indices| should be true to test
// MachOImageReader::GetSectionAtIndex() using out-of-range section indices.
// This should be tested for at least one module, but it’s very noisy in terms
// of logging output, so this knob is provided to suppress this portion of the
// test when looping over all modules.
void ExpectSegmentCommands(const MachHeader* expect_image,
                           const MachOImageReader* actual_image,
                           bool test_section_index_bounds) {
  ASSERT_TRUE(expect_image);
  ASSERT_TRUE(actual_image);

  // &expect_image[1] points right past the end of the mach_header[_64], to the
  // start of the load commands.
  const char* commands_base = reinterpret_cast<const char*>(&expect_image[1]);
  uint32_t position = 0;
  size_t section_index = 0;
  for (uint32_t index = 0; index < expect_image->ncmds; ++index) {
    ASSERT_LT(position, expect_image->sizeofcmds);
    const load_command* command =
        reinterpret_cast<const load_command*>(&commands_base[position]);
    ASSERT_LE(position + command->cmdsize, expect_image->sizeofcmds);
    if (command->cmd == kSegmentCommand) {
      ASSERT_GE(command->cmdsize, sizeof(SegmentCommand));
      const SegmentCommand* expect_segment =
          reinterpret_cast<const SegmentCommand*>(command);
      std::string segment_name =
          MachOImageSegmentReader::SegmentNameString(expect_segment->segname);
      const MachOImageSegmentReader* actual_segment =
          actual_image->GetSegmentByName(segment_name);
      ASSERT_NO_FATAL_FAILURE(ExpectSegmentCommand(expect_segment,
                                                   expect_image,
                                                   actual_segment,
                                                   actual_image,
                                                   &section_index));
    }
    position += command->cmdsize;
  }
  EXPECT_EQ(position, expect_image->sizeofcmds);

  if (test_section_index_bounds) {
    // GetSectionAtIndex uses a 1-based index. Make sure that the range is
    // correct.
    EXPECT_EQ(actual_image->GetSectionAtIndex(0, nullptr, nullptr), nullptr);
    EXPECT_EQ(
        actual_image->GetSectionAtIndex(section_index + 1, nullptr, nullptr),
        nullptr);
  }

  // Make sure that by-name lookups for names that don’t exist work properly:
  // they should return nullptr.
  EXPECT_FALSE(actual_image->GetSegmentByName("NoSuchSegment"));
  EXPECT_FALSE(actual_image->GetSectionByName(
      "NoSuchSegment", "NoSuchSection", nullptr));

  // Make sure that there’s a __TEXT segment so that this can do a valid test of
  // a section that doesn’t exist within a segment that does.
  EXPECT_TRUE(actual_image->GetSegmentByName(SEG_TEXT));
  EXPECT_FALSE(
      actual_image->GetSectionByName(SEG_TEXT, "NoSuchSection", nullptr));

  // Similarly, make sure that a section name that exists in one segment isn’t
  // accidentally found during a lookup for that section in a different segment.
  //
  // If the image has no sections (unexpected), then any section lookup should
  // fail, and these initial values of test_segment and test_section are fine
  // for the EXPECT_FALSE checks on GetSectionByName() below.
  std::string test_segment = SEG_DATA;
  std::string test_section = SECT_TEXT;

  const process_types::section* section =
      actual_image->GetSectionAtIndex(1, nullptr, nullptr);
  if (section) {
    // Use the name of the first section in the image as the section that
    // shouldn’t appear in a different segment. If the first section is in the
    // __TEXT segment (as it is normally), then a section by the same name
    // wouldn’t be expected in the __DATA segment. But if the first section is
    // in any other segment, then it wouldn’t be expected in the __TEXT segment.
    if (MachOImageSegmentReader::SegmentNameString(section->segname) ==
            SEG_TEXT) {
      test_segment = SEG_DATA;
    } else {
      test_segment = SEG_TEXT;
    }
    test_section =
        MachOImageSegmentReader::SectionNameString(section->sectname);

    // It should be possible to look up the first section by name.
    EXPECT_EQ(actual_image->GetSectionByName(
                  section->segname, section->sectname, nullptr),
              section);
  }
  EXPECT_FALSE(
      actual_image->GetSectionByName("NoSuchSegment", test_section, nullptr));
  EXPECT_FALSE(
      actual_image->GetSectionByName(test_segment, test_section, nullptr));

  // The __LINKEDIT segment normally does exist but doesn’t have any sections.
  EXPECT_FALSE(
      actual_image->GetSectionByName(SEG_LINKEDIT, "NoSuchSection", nullptr));
  EXPECT_FALSE(
      actual_image->GetSectionByName(SEG_LINKEDIT, SECT_TEXT, nullptr));
}

// In some cases, the expected slide value for an image is unknown, because no
// reasonable API to return it is provided. When this happens, use kSlideUnknown
// to avoid checking the actual slide value against anything.
constexpr mach_vm_size_t kSlideUnknown =
    std::numeric_limits<mach_vm_size_t>::max();

// Verifies that |expect_image| is a vaild Mach-O header for the current system
// by checking its |magic| and |cputype| fields. Then, verifies that the
// information in |actual_image| matches that in |expect_image|. The |filetype|
// field is examined, actual_image->Address() is compared to
// |expect_image_address|, and actual_image->Slide() is compared to
// |expect_image_slide|, unless |expect_image_slide| is kSlideUnknown. Various
// other attributes of |actual_image| are sanity-checked depending on the Mach-O
// file type. Finally, ExpectSegmentCommands() is called to verify all that all
// of the segments match; |test_section_index_bounds| is used as an argument to
// that function.
void ExpectMachImage(const MachHeader* expect_image,
                     mach_vm_address_t expect_image_address,
                     mach_vm_size_t expect_image_slide,
                     const MachOImageReader* actual_image,
                     bool test_section_index_bounds) {
  ASSERT_TRUE(expect_image);
  ASSERT_TRUE(actual_image);

  EXPECT_EQ(expect_image->magic, kMachMagic);
  EXPECT_EQ(expect_image->cputype, kCPUType);

  EXPECT_EQ(actual_image->FileType(), expect_image->filetype);
  EXPECT_EQ(actual_image->Address(), expect_image_address);
  if (expect_image_slide != kSlideUnknown) {
    EXPECT_EQ(actual_image->Slide(), expect_image_slide);
  }

  const MachOImageSegmentReader* actual_text_segment =
      actual_image->GetSegmentByName(SEG_TEXT);
  ASSERT_TRUE(actual_text_segment);
  EXPECT_EQ(actual_text_segment->Address(), expect_image_address);
  EXPECT_EQ(actual_text_segment->Size(), actual_image->Size());
  EXPECT_EQ(actual_image->Slide(),
            expect_image_address - actual_text_segment->vmaddr());

  uint32_t file_type = actual_image->FileType();
  EXPECT_TRUE(file_type == MH_EXECUTE || file_type == MH_DYLIB ||
              file_type == MH_DYLINKER || file_type == MH_BUNDLE);

  if (file_type == MH_EXECUTE || file_type == MH_DYLINKER) {
    EXPECT_EQ(actual_image->DylinkerName(), "/usr/lib/dyld");
  }

  // For these, just don’t crash or anything.
  if (file_type == MH_DYLIB) {
    actual_image->DylibVersion();
  }
  actual_image->SourceVersion();
  UUID uuid;
  actual_image->UUID(&uuid);

  ASSERT_NO_FATAL_FAILURE(ExpectSegmentCommands(
      expect_image, actual_image, test_section_index_bounds));
}

// Verifies the symbol whose Nlist structure is |entry| and whose name is |name|
// matches the value of a symbol by the same name looked up in |actual_image|.
// MachOImageReader::LookUpExternalDefinedSymbol() is used for this purpose.
// Only external defined symbols are considered, other types of symbols are
// excluded because LookUpExternalDefinedSymbol() only deals with external
// defined symbols.
void ExpectSymbol(const Nlist* entry,
                  const char* name,
                  const MachOImageReader* actual_image) {
  SCOPED_TRACE(name);

  uint32_t entry_type = entry->n_type & N_TYPE;
  if ((entry->n_type & N_STAB) == 0 && (entry->n_type & N_PEXT) == 0 &&
      (entry_type == N_ABS || entry_type == N_SECT) &&
      (entry->n_type & N_EXT) == 1) {
    mach_vm_address_t actual_address;
    ASSERT_TRUE(
        actual_image->LookUpExternalDefinedSymbol(name, &actual_address));

    // Since the nlist interface was used to read the symbol, use it to compute
    // the symbol address too. This isn’t perfect, and it should be possible in
    // theory to use dlsym() to get the expected address of a symbol. In
    // practice, dlsym() is difficult to use when only a MachHeader* is
    // available as in this function, as opposed to a void* opaque handle. It is
    // possible to get a void* handle by using dladdr() to find the file name
    // corresponding to the MachHeader*, and using dlopen() again on that name,
    // assuming it hasn’t changed on disk since being loaded. However, even with
    // that being done, dlsym() can only deal with symbols whose names begin
    // with an underscore (and requires that the leading underscore be trimmed).
    // dlsym() will also return different addresses for symbols that are
    // resolved via symbol resolver.
    mach_vm_address_t expect_address = entry->n_value;
    if (entry_type == N_SECT) {
      EXPECT_GE(entry->n_sect, 1u);
      expect_address += actual_image->Slide();
    } else {
      EXPECT_EQ(entry->n_sect, NO_SECT);
    }

    EXPECT_EQ(actual_address, expect_address);
  }

  // You’d think that it might be a good idea to verify that if the conditions
  // above weren’t met, that the symbol didn’t show up in actual_image’s symbol
  // table at all. Unfortunately, it’s possible for the same name to show up as
  // both an external defined symbol and as something else, so it’s not possible
  // to verify this reliably.
}

// Locates the symbol table in |expect_image| and verifies that all of the
// external defined symbols found there are also present and have the same
// values in |actual_image|. ExpectSymbol() is used to verify the actual symbol.
void ExpectSymbolTable(const MachHeader* expect_image,
                       const MachOImageReader* actual_image) {
  // This intentionally consults only LC_SYMTAB and not LC_DYSYMTAB so that it
  // can look at the larger set of all symbols. The actual implementation being
  // tested is free to consult LC_DYSYMTAB, but that’s considered an
  // optimization. It’s not necessary for the test, and it’s better for the test
  // to expose bugs in that optimization rather than duplicate them.
  const char* commands_base = reinterpret_cast<const char*>(&expect_image[1]);
  uint32_t position = 0;
  const symtab_command* symtab = nullptr;
  const SegmentCommand* linkedit = nullptr;
  for (uint32_t index = 0; index < expect_image->ncmds; ++index) {
    ASSERT_LT(position, expect_image->sizeofcmds);
    const load_command* command =
        reinterpret_cast<const load_command*>(&commands_base[position]);
    ASSERT_LE(position + command->cmdsize, expect_image->sizeofcmds);
    if (command->cmd == LC_SYMTAB) {
      ASSERT_FALSE(symtab);
      ASSERT_EQ(command->cmdsize, sizeof(symtab_command));
      symtab = reinterpret_cast<const symtab_command*>(command);
    } else if (command->cmd == kSegmentCommand) {
      ASSERT_GE(command->cmdsize, sizeof(SegmentCommand));
      const SegmentCommand* segment =
          reinterpret_cast<const SegmentCommand*>(command);
      std::string segment_name =
          MachOImageSegmentReader::SegmentNameString(segment->segname);
      if (segment_name == SEG_LINKEDIT) {
        ASSERT_FALSE(linkedit);
        linkedit = segment;
      }
    }
    position += command->cmdsize;
  }

  if (symtab) {
    ASSERT_TRUE(linkedit);

    const char* linkedit_base =
        reinterpret_cast<const char*>(linkedit->vmaddr + actual_image->Slide());
    const Nlist* nlist = reinterpret_cast<const Nlist*>(
        linkedit_base + symtab->symoff - linkedit->fileoff);
    const char* strtab = linkedit_base + symtab->stroff - linkedit->fileoff;

    for (uint32_t index = 0; index < symtab->nsyms; ++index) {
      const Nlist* entry = nlist + index;
      const char* name = strtab + entry->n_un.n_strx;
      ASSERT_NO_FATAL_FAILURE(ExpectSymbol(entry, name, actual_image));
    }
  }

  mach_vm_address_t ignore;
  EXPECT_FALSE(actual_image->LookUpExternalDefinedSymbol("", &ignore));
  EXPECT_FALSE(
      actual_image->LookUpExternalDefinedSymbol("NoSuchSymbolName", &ignore));
  EXPECT_FALSE(
      actual_image->LookUpExternalDefinedSymbol("_NoSuchSymbolName", &ignore));
}

TEST(MachOImageReader, Self_MainExecutable) {
  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

  const MachHeader* mh_execute_header =
      reinterpret_cast<MachHeader*>(dlsym(RTLD_MAIN_ONLY, MH_EXECUTE_SYM));
  ASSERT_NE(mh_execute_header, nullptr);
  mach_vm_address_t mh_execute_header_address =
      FromPointerCast<mach_vm_address_t>(mh_execute_header);

  MachOImageReader image_reader;
  ASSERT_TRUE(image_reader.Initialize(
      &process_reader, mh_execute_header_address, "executable"));

  EXPECT_EQ(image_reader.FileType(), implicit_cast<uint32_t>(MH_EXECUTE));

  // The main executable has image index 0.
  intptr_t image_slide = _dyld_get_image_vmaddr_slide(0);

  ASSERT_NO_FATAL_FAILURE(ExpectMachImage(mh_execute_header,
                                          mh_execute_header_address,
                                          image_slide,
                                          &image_reader,
                                          true));

  // This symbol, __mh_execute_header, is known to exist in all MH_EXECUTE
  // Mach-O files.
  mach_vm_address_t symbol_address;
  ASSERT_TRUE(image_reader.LookUpExternalDefinedSymbol(_MH_EXECUTE_SYM,
                                                       &symbol_address));
  EXPECT_EQ(symbol_address, mh_execute_header_address);

  ASSERT_NO_FATAL_FAILURE(ExpectSymbolTable(mh_execute_header, &image_reader));
}

TEST(MachOImageReader, Self_DyldImages) {
  ProcessReaderMac process_reader;
  ASSERT_TRUE(process_reader.Initialize(mach_task_self()));

  uint32_t count = _dyld_image_count();
  ASSERT_GE(count, 1u);

  size_t modules_with_crashpad_info = 0;

  for (uint32_t index = 0; index < count; ++index) {
    const char* image_name = _dyld_get_image_name(index);
    SCOPED_TRACE(base::StringPrintf("index %u, image %s", index, image_name));

    // _dyld_get_image_header() is poorly-declared: it’s declared as returning
    // const mach_header* in both 32-bit and 64-bit environments, but in the
    // 64-bit environment, it should be const mach_header_64*.
    const MachHeader* mach_header =
        reinterpret_cast<const MachHeader*>(_dyld_get_image_header(index));
    mach_vm_address_t image_address =
        FromPointerCast<mach_vm_address_t>(mach_header);

    MachOImageReader image_reader;
    ASSERT_TRUE(
        image_reader.Initialize(&process_reader, image_address, image_name));

    uint32_t file_type = image_reader.FileType();
    if (index == 0) {
      EXPECT_EQ(file_type, implicit_cast<uint32_t>(MH_EXECUTE));
    } else {
      EXPECT_TRUE(file_type == MH_DYLIB || file_type == MH_BUNDLE);
    }

    intptr_t image_slide = _dyld_get_image_vmaddr_slide(index);
    ASSERT_NO_FATAL_FAILURE(ExpectMachImage(
        mach_header, image_address, image_slide, &image_reader, false));

    ASSERT_NO_FATAL_FAILURE(ExpectSymbolTable(mach_header, &image_reader));

    process_types::CrashpadInfo crashpad_info;
    if (image_reader.GetCrashpadInfo(&crashpad_info)) {
      ++modules_with_crashpad_info;
    }
  }

  EXPECT_GE(modules_with_crashpad_info, 1u);

  // Now that all of the modules have been verified, make sure that dyld itself
  // can be read properly too.
  const dyld_all_image_infos* dyld_image_infos = DyldGetAllImageInfos();
  ASSERT_GE(dyld_image_infos->version, 1u);
  EXPECT_EQ(dyld_image_infos->infoArrayCount, count);

  if (dyld_image_infos->version >= 2) {
    SCOPED_TRACE("dyld");

    // dyld_all_image_infos::dyldImageLoadAddress is poorly-declared too.
    const MachHeader* mach_header = reinterpret_cast<const MachHeader*>(
        dyld_image_infos->dyldImageLoadAddress);
    mach_vm_address_t image_address =
        FromPointerCast<mach_vm_address_t>(mach_header);

    MachOImageReader image_reader;
    ASSERT_TRUE(
        image_reader.Initialize(&process_reader, image_address, "dyld"));

    EXPECT_EQ(image_reader.FileType(), implicit_cast<uint32_t>(MH_DYLINKER));

    // There’s no good API to get dyld’s slide, so don’t bother checking it.
    ASSERT_NO_FATAL_FAILURE(ExpectMachImage(
        mach_header, image_address, kSlideUnknown, &image_reader, false));

    ASSERT_NO_FATAL_FAILURE(ExpectSymbolTable(mach_header, &image_reader));
  }

#if MAC_OS_X_VERSION_MAX_ALLOWED >= MAC_OS_X_VERSION_10_7
  // If dyld is new enough to record UUIDs, check the UUID of any module that
  // it says has one. Note that dyld doesn’t record UUIDs of anything that
  // loaded out of the shared cache, but it should at least have a UUID for the
  // main executable if it has one.
  if (dyld_image_infos->version >= 8 && dyld_image_infos->uuidArray) {
    for (uint32_t index = 0;
         index < dyld_image_infos->uuidArrayCount;
         ++index) {
      const dyld_uuid_info* dyld_image = &dyld_image_infos->uuidArray[index];
      SCOPED_TRACE(base::StringPrintf("uuid index %u", index));

      // dyld_uuid_info::imageLoadAddress is poorly-declared too.
      const MachHeader* mach_header =
          reinterpret_cast<const MachHeader*>(dyld_image->imageLoadAddress);
      mach_vm_address_t image_address =
          FromPointerCast<mach_vm_address_t>(mach_header);

      MachOImageReader image_reader;
      ASSERT_TRUE(
          image_reader.Initialize(&process_reader, image_address, "uuid"));

      // There’s no good way to get the image’s slide here, although the image
      // should have already been checked along with its slide above, in the
      // loop through all images.
      ExpectMachImage(
          mach_header, image_address, kSlideUnknown, &image_reader, false);

      UUID expected_uuid;
      expected_uuid.InitializeFromBytes(dyld_image->imageUUID);
      UUID actual_uuid;
      image_reader.UUID(&actual_uuid);
      EXPECT_EQ(actual_uuid, expected_uuid);
    }
  }
#endif
}

}  // namespace
}  // namespace test
}  // namespace crashpad
