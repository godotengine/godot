// Copyright 2010 Google LLC
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

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// macho_reader.cc: Implementation of google_breakpad::Mach_O::FatReader and
// google_breakpad::Mach_O::Reader. See macho_reader.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/mac/macho_reader.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits>

// Unfortunately, CPU_TYPE_ARM is not define for 10.4.
#if !defined(CPU_TYPE_ARM)
#define CPU_TYPE_ARM 12
#endif

#if !defined(CPU_TYPE_ARM_64)
#define CPU_TYPE_ARM_64 16777228
#endif

namespace google_breakpad {
namespace mach_o {

// If NDEBUG is #defined, then the 'assert' macro doesn't evaluate its
// arguments, so you can't place expressions that do necessary work in
// the argument of an assert. Nor can you assign the result of the
// expression to a variable and assert that the variable's value is
// true: you'll get unused variable warnings when NDEBUG is #defined.
//
// ASSERT_ALWAYS_EVAL always evaluates its argument, and asserts that
// the result is true if NDEBUG is not #defined.
#if defined(NDEBUG)
#define ASSERT_ALWAYS_EVAL(x) (x)
#else
#define ASSERT_ALWAYS_EVAL(x) assert(x)
#endif

void FatReader::Reporter::BadHeader() {
  fprintf(stderr, "%s: file is neither a fat binary file"
          " nor a Mach-O object file\n", filename_.c_str());
}

void FatReader::Reporter::TooShort() {
  fprintf(stderr, "%s: file too short for the data it claims to contain\n",
          filename_.c_str());
}

void FatReader::Reporter::MisplacedObjectFile() {
  fprintf(stderr, "%s: file too short for the object files it claims"
          " to contain\n", filename_.c_str());
}

bool FatReader::Read(const uint8_t* buffer, size_t size) {
  buffer_.start = buffer;
  buffer_.end = buffer + size;
  ByteCursor cursor(&buffer_);

  // Fat binaries always use big-endian, so read the magic number in
  // that endianness. To recognize Mach-O magic numbers, which can use
  // either endianness, check for both the proper and reversed forms
  // of the magic numbers.
  cursor.set_big_endian(true);
  if (cursor >> magic_) {
    if (magic_ == FAT_MAGIC) {
      // How many object files does this fat binary contain?
      uint32_t object_files_count;
      if (!(cursor >> object_files_count)) {  // nfat_arch
        reporter_->TooShort();
        return false;
      }

      // Read the list of object files.
      object_files_.resize(object_files_count);
      for (size_t i = 0; i < object_files_count; i++) {
        struct fat_arch objfile;

        // Read this object file entry, byte-swapping as appropriate.
        cursor >> objfile.cputype
               >> objfile.cpusubtype
               >> objfile.offset
               >> objfile.size
               >> objfile.align;

        SuperFatArch super_fat_arch(objfile);
        object_files_[i] = super_fat_arch;

        if (!cursor) {
          reporter_->TooShort();
          return false;
        }
        // Does the file actually have the bytes this entry refers to?
        size_t fat_size = buffer_.Size();
        if (objfile.offset > fat_size ||
            objfile.size > fat_size - objfile.offset) {
          reporter_->MisplacedObjectFile();
          return false;
        }
      }

      return true;
    } else if (magic_ == MH_MAGIC || magic_ == MH_MAGIC_64 ||
               magic_ == MH_CIGAM || magic_ == MH_CIGAM_64) {
      // If this is a little-endian Mach-O file, fix the cursor's endianness.
      if (magic_ == MH_CIGAM || magic_ == MH_CIGAM_64)
        cursor.set_big_endian(false);
      // Record the entire file as a single entry in the object file list.
      object_files_.resize(1);

      // Get the cpu type and subtype from the Mach-O header.
      if (!(cursor >> object_files_[0].cputype
                   >> object_files_[0].cpusubtype)) {
        reporter_->TooShort();
        return false;
      }

      object_files_[0].offset = 0;
      object_files_[0].size = static_cast<uint64_t>(buffer_.Size());
      // This alignment is correct for 32 and 64-bit x86 and ppc.
      // See get_align in the lipo source for other architectures:
      // http://www.opensource.apple.com/source/cctools/cctools-773/misc/lipo.c
      object_files_[0].align = 12;  // 2^12 == 4096
      return true;
    }
  }
  reporter_->BadHeader();
  return false;
}

void Reader::Reporter::BadHeader() {
  fprintf(stderr, "%s: file is not a Mach-O object file\n", filename_.c_str());
}

void Reader::Reporter::CPUTypeMismatch(cpu_type_t cpu_type,
                                       cpu_subtype_t cpu_subtype,
                                       cpu_type_t expected_cpu_type,
                                       cpu_subtype_t expected_cpu_subtype) {
  fprintf(stderr, "%s: CPU type %d, subtype %d does not match expected"
          " type %d, subtype %d\n",
          filename_.c_str(), cpu_type, cpu_subtype,
          expected_cpu_type, expected_cpu_subtype);
}

void Reader::Reporter::HeaderTruncated() {
  fprintf(stderr, "%s: file does not contain a complete Mach-O header\n",
          filename_.c_str());
}

void Reader::Reporter::LoadCommandRegionTruncated() {
  fprintf(stderr, "%s: file too short to hold load command region"
          " given in Mach-O header\n", filename_.c_str());
}

void Reader::Reporter::LoadCommandsOverrun(size_t claimed, size_t i,
                                           LoadCommandType type) {
  fprintf(stderr, "%s: file's header claims there are %zu"
          " load commands, but load command #%zu",
          filename_.c_str(), claimed, i);
  if (type) fprintf(stderr, ", of type %d,", type);
  fprintf(stderr, " extends beyond the end of the load command region\n");
}

void Reader::Reporter::LoadCommandTooShort(size_t i, LoadCommandType type) {
  fprintf(stderr, "%s: the contents of load command #%zu, of type %d,"
          " extend beyond the size given in the load command's header\n",
          filename_.c_str(), i, type);
}

void Reader::Reporter::SectionsMissing(const string& name) {
  fprintf(stderr, "%s: the load command for segment '%s'"
          " is too short to hold the section headers it claims to have\n",
          filename_.c_str(), name.c_str());
}

void Reader::Reporter::MisplacedSegmentData(const string& name) {
  fprintf(stderr, "%s: the segment '%s' claims its contents lie beyond"
          " the end of the file\n", filename_.c_str(), name.c_str());
}

void Reader::Reporter::MisplacedSectionData(const string& section,
                                            const string& segment) {
  fprintf(stderr, "%s: the section '%s' in segment '%s'"
          " claims its contents lie outside the segment's contents\n",
          filename_.c_str(), section.c_str(), segment.c_str());
}

void Reader::Reporter::MisplacedSymbolTable() {
  fprintf(stderr, "%s: the LC_SYMTAB load command claims that the symbol"
          " table's contents are located beyond the end of the file\n",
          filename_.c_str());
}

void Reader::Reporter::UnsupportedCPUType(cpu_type_t cpu_type) {
  fprintf(stderr, "%s: CPU type %d is not supported\n",
          filename_.c_str(), cpu_type);
}

bool Reader::Read(const uint8_t* buffer,
                  size_t size,
                  cpu_type_t expected_cpu_type,
                  cpu_subtype_t expected_cpu_subtype) {
  assert(!buffer_.start);
  buffer_.start = buffer;
  buffer_.end = buffer + size;
  ByteCursor cursor(&buffer_, true);
  uint32_t magic;
  if (!(cursor >> magic)) {
    reporter_->HeaderTruncated();
    return false;
  }

  if (expected_cpu_type != CPU_TYPE_ANY) {
    uint32_t expected_magic;
    // validate that magic matches the expected cpu type
    switch (expected_cpu_type) {
      case CPU_TYPE_ARM:
      case CPU_TYPE_I386:
        expected_magic = MH_CIGAM;
        break;
      case CPU_TYPE_POWERPC:
        expected_magic = MH_MAGIC;
        break;
      case CPU_TYPE_ARM_64:
      case CPU_TYPE_X86_64:
        expected_magic = MH_CIGAM_64;
        break;
      case CPU_TYPE_POWERPC64:
        expected_magic = MH_MAGIC_64;
        break;
      default:
        reporter_->UnsupportedCPUType(expected_cpu_type);
        return false;
    }

    if (expected_magic != magic) {
      reporter_->BadHeader();
      return false;
    }
  }

  // Since the byte cursor is in big-endian mode, a reversed magic number
  // always indicates a little-endian file, regardless of our own endianness.
  switch (magic) {
    case MH_MAGIC:    big_endian_ = true;  bits_64_ = false; break;
    case MH_CIGAM:    big_endian_ = false; bits_64_ = false; break;
    case MH_MAGIC_64: big_endian_ = true;  bits_64_ = true;  break;
    case MH_CIGAM_64: big_endian_ = false; bits_64_ = true;  break;
    default:
      reporter_->BadHeader();
      return false;
  }
  cursor.set_big_endian(big_endian_);
  uint32_t commands_size, reserved;
  cursor >> cpu_type_ >> cpu_subtype_ >> file_type_ >> load_command_count_
         >> commands_size >> flags_;
  if (bits_64_)
    cursor >> reserved;
  if (!cursor) {
    reporter_->HeaderTruncated();
    return false;
  }

  if (expected_cpu_type != CPU_TYPE_ANY &&
      (expected_cpu_type != cpu_type_ ||
       expected_cpu_subtype != cpu_subtype_)) {
    reporter_->CPUTypeMismatch(cpu_type_, cpu_subtype_,
                              expected_cpu_type, expected_cpu_subtype);
    return false;
  }

  cursor
      .PointTo(&load_commands_.start, commands_size)
      .PointTo(&load_commands_.end, 0);
  if (!cursor) {
    reporter_->LoadCommandRegionTruncated();
    return false;
  }

  return true;
}

bool Reader::WalkLoadCommands(Reader::LoadCommandHandler* handler) const {
  ByteCursor list_cursor(&load_commands_, big_endian_);

  for (size_t index = 0; index < load_command_count_; ++index) {
    // command refers to this load command alone, so that cursor will
    // refuse to read past the load command's end. But since we haven't
    // read the size yet, let command initially refer to the entire
    // remainder of the load command series.
    ByteBuffer command(list_cursor.here(), list_cursor.Available());
    ByteCursor cursor(&command, big_endian_);

    // Read the command type and size --- fields common to all commands.
    uint32_t type, size;
    if (!(cursor >> type)) {
      reporter_->LoadCommandsOverrun(load_command_count_, index, 0);
      return false;
    }
    if (!(cursor >> size) || size > command.Size()) {
      reporter_->LoadCommandsOverrun(load_command_count_, index, type);
      return false;
    }

    // Now that we've read the length, restrict command's range to this
    // load command only.
    command.end = command.start + size;

    switch (type) {
      case LC_SEGMENT:
      case LC_SEGMENT_64: {
        Segment segment;
        segment.bits_64 = (type == LC_SEGMENT_64);
        size_t word_size = segment.bits_64 ? 8 : 4;
        cursor.CString(&segment.name, 16);
        cursor
            .Read(word_size, false, &segment.vmaddr)
            .Read(word_size, false, &segment.vmsize)
            .Read(word_size, false, &segment.fileoff)
            .Read(word_size, false, &segment.filesize);
        cursor >> segment.maxprot
               >> segment.initprot
               >> segment.nsects
               >> segment.flags;
        if (!cursor) {
          reporter_->LoadCommandTooShort(index, type);
          return false;
        }
        if (segment.fileoff > buffer_.Size() ||
            segment.filesize > buffer_.Size() - segment.fileoff) {
          reporter_->MisplacedSegmentData(segment.name);
          return false;
        }
        // Mach-O files in .dSYM bundles have the contents of the loaded
        // segments removed, and their file offsets and file sizes zeroed
        // out. To help us handle this special case properly, give such
        // segments' contents NULL starting and ending pointers.
        if (segment.fileoff == 0 && segment.filesize == 0) {
          segment.contents.start = segment.contents.end = NULL;
        } else {
          segment.contents.start = buffer_.start + segment.fileoff;
          segment.contents.end = segment.contents.start + segment.filesize;
        }
        // The section list occupies the remainder of this load command's space.
        segment.section_list.start = cursor.here();
        segment.section_list.end = command.end;

        if (!handler->SegmentCommand(segment))
          return false;
        break;
      }

      case LC_SYMTAB: {
        uint32_t symoff, nsyms, stroff, strsize;
        cursor >> symoff >> nsyms >> stroff >> strsize;
        if (!cursor) {
          reporter_->LoadCommandTooShort(index, type);
          return false;
        }
        // How big are the entries in the symbol table?
        // sizeof(struct nlist_64) : sizeof(struct nlist),
        // but be paranoid about alignment vs. target architecture.
        size_t symbol_size = bits_64_ ? 16 : 12;
        // How big is the entire symbol array?
        size_t symbols_size = nsyms * symbol_size;
        if (symoff > buffer_.Size() || symbols_size > buffer_.Size() - symoff ||
            stroff > buffer_.Size() || strsize > buffer_.Size() - stroff) {
          reporter_->MisplacedSymbolTable();
          return false;
        }
        ByteBuffer entries(buffer_.start + symoff, symbols_size);
        ByteBuffer names(buffer_.start + stroff, strsize);
        if (!handler->SymtabCommand(entries, names))
          return false;
        break;
      }

      default: {
        if (!handler->UnknownCommand(type, command))
          return false;
        break;
      }
    }

    list_cursor.set_here(command.end);
  }

  return true;
}

// A load command handler that looks for a segment of a given name.
class Reader::SegmentFinder : public LoadCommandHandler {
 public:
  // Create a load command handler that looks for a segment named NAME,
  // and sets SEGMENT to describe it if found.
  SegmentFinder(const string& name, Segment* segment)
      : name_(name), segment_(segment), found_() { }

  // Return true if the traversal found the segment, false otherwise.
  bool found() const { return found_; }

  bool SegmentCommand(const Segment& segment) {
    if (segment.name == name_) {
      *segment_ = segment;
      found_ = true;
      return false;
    }
    return true;
  }

 private:
  // The name of the segment our creator is looking for.
  const string& name_;

  // Where we should store the segment if found. (WEAK)
  Segment* segment_;

  // True if we found the segment.
  bool found_;
};

bool Reader::FindSegment(const string& name, Segment* segment) const {
  SegmentFinder finder(name, segment);
  WalkLoadCommands(&finder);
  return finder.found();
}

bool Reader::WalkSegmentSections(const Segment& segment,
                                 SectionHandler* handler) const {
  size_t word_size = segment.bits_64 ? 8 : 4;
  ByteCursor cursor(&segment.section_list, big_endian_);

  for (size_t i = 0; i < segment.nsects; i++) {
    Section section;
    section.bits_64 = segment.bits_64;
    uint64_t size, offset;
    uint32_t dummy32;
    cursor
        .CString(&section.section_name, 16)
        .CString(&section.segment_name, 16)
        .Read(word_size, false, &section.address)
        .Read(word_size, false, &size)
        .Read(sizeof(uint32_t), false, &offset)  // clears high bits of |offset|
        >> section.align
        >> dummy32
        >> dummy32
        >> section.flags
        >> dummy32
        >> dummy32;
    if (section.bits_64)
      cursor >> dummy32;
    if (!cursor) {
      reporter_->SectionsMissing(segment.name);
      return false;
    }

    // Even 64-bit Mach-O isn’t a true 64-bit format in that it doesn’t handle
    // 64-bit file offsets gracefully. Segment load commands do contain 64-bit
    // file offsets, but sections within do not. Because segments load
    // contiguously, recompute each section’s file offset on the basis of its
    // containing segment’s file offset and the difference between the section’s
    // and segment’s load addresses. If truncation is detected, honor the
    // recomputed offset.
    if (segment.bits_64 &&
        segment.fileoff + segment.filesize >
            std::numeric_limits<uint32_t>::max()) {
      const uint64_t section_offset_recomputed =
          segment.fileoff + section.address - segment.vmaddr;
      if (offset == static_cast<uint32_t>(section_offset_recomputed)) {
        offset = section_offset_recomputed;
      }
    }

    const uint32_t section_type = section.flags & SECTION_TYPE;
    if (section_type == S_ZEROFILL || section_type == S_THREAD_LOCAL_ZEROFILL ||
            section_type == S_GB_ZEROFILL) {
      // Zero-fill sections have a size, but no contents.
      section.contents.start = section.contents.end = NULL;
    } else if (segment.contents.start == NULL &&
               segment.contents.end == NULL) {
      // Mach-O files in .dSYM bundles have the contents of the loaded
      // segments removed, and their file offsets and file sizes zeroed
      // out.  However, the sections within those segments still have
      // non-zero sizes.  There's no reason to call MisplacedSectionData in
      // this case; the caller may just need the section's load
      // address. But do set the contents' limits to NULL, for safety.
      section.contents.start = section.contents.end = NULL;
    } else {
      if (offset < size_t(segment.contents.start - buffer_.start) ||
          offset > size_t(segment.contents.end - buffer_.start) ||
          size > size_t(segment.contents.end - buffer_.start - offset)) {
        if (offset > 0) {
          reporter_->MisplacedSectionData(section.section_name,
                                          section.segment_name);
          return false;
        } else {
          // Mach-O files in .dSYM bundles have the contents of the loaded
          // segments partially removed. The removed sections will have zero as
          // their offset. MisplacedSectionData should not be called in this
          // case.
          section.contents.start = section.contents.end = NULL;
        }
      } else {
        section.contents.start = buffer_.start + offset;
        section.contents.end = section.contents.start + size;
      }
    }
    if (!handler->HandleSection(section))
      return false;
  }
  return true;
}

// A SectionHandler that builds a SectionMap for the sections within a
// given segment.
class Reader::SectionMapper: public SectionHandler {
 public:
  // Create a SectionHandler that populates MAP with an entry for
  // each section it is given.
  SectionMapper(SectionMap* map) : map_(map) { }
  bool HandleSection(const Section& section) {
    (*map_)[section.section_name] = section;
    return true;
  }
 private:
  // The map under construction. (WEAK)
  SectionMap* map_;
};

bool Reader::MapSegmentSections(const Segment& segment,
                                SectionMap* section_map) const {
  section_map->clear();
  SectionMapper mapper(section_map);
  return WalkSegmentSections(segment, &mapper);
}

}  // namespace mach_o
}  // namespace google_breakpad
