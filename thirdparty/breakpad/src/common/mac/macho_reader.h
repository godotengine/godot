// -*- mode: C++ -*-

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

// macho_reader.h: A class for parsing Mach-O files.

#ifndef BREAKPAD_COMMON_MAC_MACHO_READER_H_
#define BREAKPAD_COMMON_MAC_MACHO_READER_H_

#include <mach-o/loader.h>
#include <mach-o/fat.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>

#include <map>
#include <string>
#include <vector>

#include "common/byte_cursor.h"
#include "common/mac/super_fat_arch.h"

namespace google_breakpad {
namespace mach_o {

using std::map;
using std::string;
using std::vector;

// The Mac headers don't specify particular types for these groups of
// constants, but defining them here provides some documentation
// value.  We also give them the same width as the fields in which
// they appear, which makes them a bit easier to use with ByteCursors.
typedef uint32_t Magic;
typedef uint32_t FileType;
typedef uint32_t FileFlags;
typedef uint32_t LoadCommandType;
typedef uint32_t SegmentFlags;
typedef uint32_t SectionFlags;

// A parser for fat binary files, used to store universal binaries.
// When applied to a (non-fat) Mach-O file, this behaves as if the
// file were a fat file containing a single object file.
class FatReader {
 public:

  // A class for reporting errors found while parsing fat binary files. The
  // default definitions of these methods print messages to stderr.
  class Reporter {
   public:
    // Create a reporter that attributes problems to |filename|.
    explicit Reporter(const string& filename) : filename_(filename) { }

    virtual ~Reporter() { }

    // The data does not begin with a fat binary or Mach-O magic number.
    // This is a fatal error.
    virtual void BadHeader();

    // The Mach-O fat binary file ends abruptly, without enough space
    // to contain an object file it claims is present.
    virtual void MisplacedObjectFile();

    // The file ends abruptly: either it is not large enough to hold a
    // complete header, or the header implies that contents are present
    // beyond the actual end of the file.
    virtual void TooShort();

   private:
    // The filename to which the reader should attribute problems.
    string filename_;
  };

  // Create a fat binary file reader that uses |reporter| to report problems.
  explicit FatReader(Reporter* reporter) : reporter_(reporter) { }

  // Read the |size| bytes at |buffer| as a fat binary file. On success,
  // return true; on failure, report the problem to reporter_ and return
  // false.
  //
  // If the data is a plain Mach-O file, rather than a fat binary file,
  // then the reader behaves as if it had found a fat binary file whose
  // single object file is the Mach-O file.
  bool Read(const uint8_t* buffer, size_t size);

  // Return an array of 'SuperFatArch' structures describing the
  // object files present in this fat binary file. Set |size| to the
  // number of elements in the array.
  //
  // Assuming Read returned true, the entries are validated: it is safe to
  // assume that the offsets and sizes in each SuperFatArch refer to subranges
  // of the bytes passed to Read.
  //
  // If there are no object files in this fat binary, then this
  // function can return NULL.
  //
  // The array is owned by this FatReader instance; it will be freed when
  // this FatReader is destroyed.
  //
  // This function returns a C-style array instead of a vector to make it
  // possible to use the result with OS X functions like NXFindBestFatArch,
  // so that the symbol dumper will behave consistently with other OS X
  // utilities that work with fat binaries.
  const SuperFatArch* object_files(size_t* count) const {
    *count = object_files_.size();
    if (object_files_.size() > 0)
      return &object_files_[0];
    return NULL;
  }

 private:
  // We use this to report problems parsing the file's contents. (WEAK)
  Reporter* reporter_;

  // The contents of the fat binary or Mach-O file we're parsing. We do not
  // own the storage it refers to.
  ByteBuffer buffer_;

  // The magic number of this binary, in host byte order.
  Magic magic_;

  // The list of object files in this binary.
  // object_files_.size() == fat_header.nfat_arch
  vector<SuperFatArch> object_files_;
};

// A segment in a Mach-O file. All these fields have been byte-swapped as
// appropriate for use by the executing architecture.
struct Segment {
  // The ByteBuffers below point into the bytes passed to the Reader that
  // created this Segment.

  ByteBuffer section_list;    // This segment's section list.
  ByteBuffer contents;        // This segment's contents.

  // This segment's name.
  string name;

  // The address at which this segment should be loaded in memory. If
  // bits_64 is false, only the bottom 32 bits of this value are valid.
  uint64_t vmaddr;

  // The size of this segment when loaded into memory. This may be larger
  // than contents.Size(), in which case the extra area will be
  // initialized with zeros. If bits_64 is false, only the bottom 32 bits
  // of this value are valid.
  uint64_t vmsize;

  // The file offset and size of the segment in the Mach-O image.
  uint64_t fileoff;
  uint64_t filesize;

  // The maximum and initial VM protection of this segment's contents.
  uint32_t maxprot;
  uint32_t initprot;

  // The number of sections in section_list.
  uint32_t nsects;

  // Flags describing this segment, from SegmentFlags.
  uint32_t flags;

  // True if this is a 64-bit section; false if it is a 32-bit section.
  bool bits_64;
};

// A section in a Mach-O file. All these fields have been byte-swapped as
// appropriate for use by the executing architecture.
struct Section {
  // This section's contents. This points into the bytes passed to the
  // Reader that created this Section.
  ByteBuffer contents;

  // This section's name.
  string section_name;  // section[_64].sectname
  // The name of the segment this section belongs to.
  string segment_name;  // section[_64].segname

  // The address at which this section's contents should be loaded in
  // memory. If bits_64 is false, only the bottom 32 bits of this value
  // are valid.
  uint64_t address;

  // The contents of this section should be loaded into memory at an
  // address which is a multiple of (two raised to this power).
  uint32_t align;

  // Flags from SectionFlags describing the section's contents.
  uint32_t flags;

  // We don't support reading relocations yet.

  // True if this is a 64-bit section; false if it is a 32-bit section.
  bool bits_64;
};

// A map from section names to Sections.
typedef map<string, Section> SectionMap;

// A reader for a Mach-O file.
//
// This does not handle fat binaries; see FatReader above. FatReader
// provides a friendly interface for parsing data that could be either a
// fat binary or a Mach-O file.
class Reader {
 public:

  // A class for reporting errors found while parsing Mach-O files. The
  // default definitions of these member functions print messages to
  // stderr.
  class Reporter {
   public:
    // Create a reporter that attributes problems to |filename|.
    explicit Reporter(const string& filename) : filename_(filename) { }
    virtual ~Reporter() { }

    // Reporter functions for fatal errors return void; the reader will
    // definitely return an error to its caller after calling them

    // The data does not begin with a Mach-O magic number, or the magic
    // number does not match the expected value for the cpu architecture.
    // This is a fatal error.
    virtual void BadHeader();

    // The data contained in a Mach-O fat binary (|cpu_type|, |cpu_subtype|)
    // does not match the expected CPU architecture
    // (|expected_cpu_type|, |expected_cpu_subtype|).
    virtual void CPUTypeMismatch(cpu_type_t cpu_type,
                                 cpu_subtype_t cpu_subtype,
                                 cpu_type_t expected_cpu_type,
                                 cpu_subtype_t expected_cpu_subtype);

    // The file ends abruptly: either it is not large enough to hold a
    // complete header, or the header implies that contents are present
    // beyond the actual end of the file.
    virtual void HeaderTruncated();

    // The file's load command region, as given in the Mach-O header, is
    // too large for the file.
    virtual void LoadCommandRegionTruncated();

    // The file's Mach-O header claims the file contains |claimed| load
    // commands, but the I'th load command, of type |type|, extends beyond
    // the end of the load command region, as given by the Mach-O header.
    // If |type| is zero, the command's type was unreadable.
    virtual void LoadCommandsOverrun(size_t claimed, size_t i,
                                     LoadCommandType type);

    // The contents of the |i|'th load command, of type |type|, extend beyond
    // the size given in the load command's header.
    virtual void LoadCommandTooShort(size_t i, LoadCommandType type);

    // The LC_SEGMENT or LC_SEGMENT_64 load command for the segment named
    // |name| is too short to hold the sections that its header says it does.
    // (This more specific than LoadCommandTooShort.)
    virtual void SectionsMissing(const string& name);

    // The segment named |name| claims that its contents lie beyond the end
    // of the file.
    virtual void MisplacedSegmentData(const string& name);

    // The section named |section| in the segment named |segment| claims that
    // its contents do not lie entirely within the segment.
    virtual void MisplacedSectionData(const string& section,
                                      const string& segment);

    // The LC_SYMTAB command claims that symbol table contents are located
    // beyond the end of the file.
    virtual void MisplacedSymbolTable();

    // An attempt was made to read a Mach-O file of the unsupported
    // CPU architecture |cpu_type|.
    virtual void UnsupportedCPUType(cpu_type_t cpu_type);

   private:
    string filename_;
  };

  // A handler for sections parsed from a segment. The WalkSegmentSections
  // member function accepts an instance of this class, and applies it to
  // each section defined in a given segment.
  class SectionHandler {
   public:
    virtual ~SectionHandler() { }

    // Called to report that the segment's section list contains |section|.
    // This should return true if the iteration should continue, or false
    // if it should stop.
    virtual bool HandleSection(const Section& section) = 0;
  };

  // A handler for the load commands in a Mach-O file.
  class LoadCommandHandler {
   public:
    LoadCommandHandler() { }
    virtual ~LoadCommandHandler() { }

    // When called from WalkLoadCommands, the following handler functions
    // should return true if they wish to continue iterating over the load
    // command list, or false if they wish to stop iterating.
    //
    // When called from LoadCommandIterator::Handle or Reader::Handle,
    // these functions' return values are simply passed through to Handle's
    // caller.
    //
    // The definitions provided by this base class simply return true; the
    // default is to silently ignore sections whose member functions the
    // subclass doesn't override.

    // COMMAND is load command we don't recognize. We provide only the
    // command type and a ByteBuffer enclosing the command's data (If we
    // cannot parse the command type or its size, we call
    // reporter_->IncompleteLoadCommand instead.)
    virtual bool UnknownCommand(LoadCommandType type,
                                const ByteBuffer& contents) {
      return true;
    }

    // The load command is LC_SEGMENT or LC_SEGMENT_64, defining a segment
    // with the properties given in |segment|.
    virtual bool SegmentCommand(const Segment& segment) {
      return true;
    }

    // The load command is LC_SYMTAB. |entries| holds the array of nlist
    // entries, and |names| holds the strings the entries refer to.
    virtual bool SymtabCommand(const ByteBuffer& entries,
                               const ByteBuffer& names) {
      return true;
    }

    // Add handler functions for more load commands here as needed.
  };

  // Create a Mach-O file reader that reports problems to |reporter|.
  explicit Reader(Reporter* reporter)
      : reporter_(reporter) { }

  // Read the given data as a Mach-O file. The reader retains pointers
  // into the data passed, so the data should live as long as the reader
  // does. On success, return true; on failure, return false.
  //
  // At most one of these functions should be invoked once on each Reader
  // instance.
  bool Read(const uint8_t* buffer,
            size_t size,
            cpu_type_t expected_cpu_type,
            cpu_subtype_t expected_cpu_subtype);
  bool Read(const ByteBuffer& buffer,
            cpu_type_t expected_cpu_type,
            cpu_subtype_t expected_cpu_subtype) {
    return Read(buffer.start,
                buffer.Size(),
                expected_cpu_type,
                expected_cpu_subtype);
  }

  // Return this file's characteristics, as found in the Mach-O header.
  cpu_type_t    cpu_type()    const { return cpu_type_; }
  cpu_subtype_t cpu_subtype() const { return cpu_subtype_; }
  FileType      file_type()   const { return file_type_; }
  FileFlags     flags()       const { return flags_; }

  // Return true if this is a 64-bit Mach-O file, false if it is a 32-bit
  // Mach-O file.
  bool bits_64() const { return bits_64_; }

  // Return true if this is a big-endian Mach-O file, false if it is
  // little-endian.
  bool big_endian() const { return big_endian_; }

  // Apply |handler| to each load command in this Mach-O file, stopping when
  // a handler function returns false. If we encounter a malformed load
  // command, report it via reporter_ and return false. Return true if all
  // load commands were parseable and all handlers returned true.
  bool WalkLoadCommands(LoadCommandHandler* handler) const;

  // Set |segment| to describe the segment named |name|, if present. If
  // found, |segment|'s byte buffers refer to a subregion of the bytes
  // passed to Read. If we find the section, return true; otherwise,
  // return false.
  bool FindSegment(const string& name, Segment* segment) const;

  // Apply |handler| to each section defined in |segment|. If |handler| returns
  // false, stop iterating and return false. If all calls to |handler| return
  // true and we reach the end of the section list, return true.
  bool WalkSegmentSections(const Segment& segment, SectionHandler* handler)
    const;

  // Clear |section_map| and then populate it with a map of the sections
  // in |segment|, from section names to Section structures.
  // Each Section's contents refer to bytes in |segment|'s contents.
  // On success, return true; if a problem occurs, report it and return false.
  bool MapSegmentSections(const Segment& segment, SectionMap* section_map)
    const;

 private:
  // Used internally.
  class SegmentFinder;
  class SectionMapper;

  // We use this to report problems parsing the file's contents. (WEAK)
  Reporter* reporter_;

  // The contents of the Mach-O file we're parsing. We do not own the
  // storage it refers to.
  ByteBuffer buffer_;

  // True if this file is big-endian.
  bool big_endian_;

  // True if this file is a 64-bit Mach-O file.
  bool bits_64_;

  // This file's cpu type and subtype.
  cpu_type_t cpu_type_;        // mach_header[_64].cputype
  cpu_subtype_t cpu_subtype_;  // mach_header[_64].cpusubtype

  // This file's type.
  FileType file_type_;         // mach_header[_64].filetype

  // The region of buffer_ occupied by load commands.
  ByteBuffer load_commands_;

  // The number of load commands in load_commands_.
  uint32_t load_command_count_;  // mach_header[_64].ncmds

  // This file's header flags.
  FileFlags flags_;
};

}  // namespace mach_o
}  // namespace google_breakpad

#endif  // BREAKPAD_COMMON_MAC_MACHO_READER_H_
