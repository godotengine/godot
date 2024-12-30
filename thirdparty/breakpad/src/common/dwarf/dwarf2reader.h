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

// CFI reader author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// This file contains definitions related to the DWARF2/3 reader and
// it's handler interfaces.
// The DWARF2/3 specification can be found at
// http://dwarf.freestandards.org and should be considered required
// reading if you wish to modify the implementation.
// Only a cursory attempt is made to explain terminology that is
// used here, as it is much better explained in the standard documents
#ifndef COMMON_DWARF_DWARF2READER_H__
#define COMMON_DWARF_DWARF2READER_H__

#include <assert.h>
#include <stdint.h>

#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "common/dwarf/bytereader.h"
#include "common/dwarf/dwarf2enums.h"
#include "common/dwarf/types.h"
#include "common/using_std_string.h"
#include "common/dwarf/elf_reader.h"

namespace google_breakpad {
struct LineStateMachine;
class Dwarf2Handler;
class LineInfoHandler;
class DwpReader;

// This maps from a string naming a section to a pair containing a
// the data for the section, and the size of the section.
typedef std::map<string, std::pair<const uint8_t*, uint64_t> > SectionMap;

// Abstract away the difference between elf and mach-o section names.
// Elf-names use ".section_name, mach-o uses "__section_name".  Pass "name" in
// the elf form, ".section_name".
const SectionMap::const_iterator GetSectionByName(const SectionMap&
                                                  sections, const char* name);

// Most of the time, this struct functions as a simple attribute and form pair.
// However, Dwarf5 DW_FORM_implicit_const means that a form may have its value
// in line in the abbrev table, and that value must be associated with the
// pair until the attr's value is needed.
struct AttrForm {
  AttrForm(enum DwarfAttribute attr, enum DwarfForm form, uint64_t value) :
      attr_(attr), form_(form), value_(value) { }

  enum DwarfAttribute attr_;
  enum DwarfForm form_;
  uint64_t value_;
};
typedef std::list<AttrForm> AttributeList;
typedef AttributeList::iterator AttributeIterator;
typedef AttributeList::const_iterator ConstAttributeIterator;

struct LineInfoHeader {
  uint64_t total_length;
  uint16_t version;
  uint64_t prologue_length;
  uint8_t min_insn_length; // insn stands for instructin
  bool default_is_stmt; // stmt stands for statement
  int8_t line_base;
  uint8_t line_range;
  uint8_t opcode_base;
  // Use a pointer so that signalsafe_addr2line is able to use this structure
  // without heap allocation problem.
  std::vector<unsigned char>* std_opcode_lengths;
};

class LineInfo {
 public:

  // Initializes a .debug_line reader. Buffer and buffer length point
  // to the beginning and length of the line information to read.
  // Reader is a ByteReader class that has the endianness set
  // properly.
  LineInfo(const uint8_t* buffer, uint64_t buffer_length,
           ByteReader* reader, const uint8_t* string_buffer,
           size_t string_buffer_length, const uint8_t* line_string_buffer,
           size_t line_string_buffer_length, LineInfoHandler* handler);

  virtual ~LineInfo() {
    if (header_.std_opcode_lengths) {
      delete header_.std_opcode_lengths;
    }
  }

  // Start processing line info, and calling callbacks in the handler.
  // Consumes the line number information for a single compilation unit.
  // Returns the number of bytes processed.
  uint64_t Start();

  // Process a single line info opcode at START using the state
  // machine at LSM.  Return true if we should define a line using the
  // current state of the line state machine.  Place the length of the
  // opcode in LEN.
  // If LSM_PASSES_PC is non-NULL, this function also checks if the lsm
  // passes the address of PC. In other words, LSM_PASSES_PC will be
  // set to true, if the following condition is met.
  //
  // lsm's old address < PC <= lsm's new address
  static bool ProcessOneOpcode(ByteReader* reader,
                               LineInfoHandler* handler,
                               const struct LineInfoHeader& header,
                               const uint8_t* start,
                               struct LineStateMachine* lsm,
                               size_t* len,
                               uintptr pc,
                               bool* lsm_passes_pc);

 private:
  // Reads the DWARF2/3 header for this line info.
  void ReadHeader();

  // Reads the DWARF2/3 line information
  void ReadLines();

  // Read the DWARF5 types and forms for the file and directory tables.
  void ReadTypesAndForms(const uint8_t** lineptr, uint32_t* content_types,
                         uint32_t* content_forms, uint32_t max_types,
                         uint32_t* format_count);

  // Read a row from the dwarf5 LineInfo file table.
  void ReadFileRow(const uint8_t** lineptr, const uint32_t* content_types,
                   const uint32_t* content_forms, uint32_t row,
                   uint32_t format_count);

  // Read and return the data at *lineptr according to form. Advance
  // *lineptr appropriately.
  uint64_t ReadUnsignedData(uint32_t form, const uint8_t** lineptr);

  // Read and return the data at *lineptr according to form. Advance
  // *lineptr appropriately.
  const char* ReadStringForm(uint32_t form, const uint8_t** lineptr);

  // The associated handler to call processing functions in
  LineInfoHandler* handler_;

  // The associated ByteReader that handles endianness issues for us
  ByteReader* reader_;

  // A DWARF line info header.  This is not the same size as in the actual file,
  // as the one in the file may have a 32 bit or 64 bit lengths

  struct LineInfoHeader header_;

  // buffer is the buffer for our line info, starting at exactly where
  // the line info to read is.  after_header is the place right after
  // the end of the line information header.
  const uint8_t* buffer_;
#ifndef NDEBUG
  uint64_t buffer_length_;
#endif
  // Convenience pointers into .debug_str and .debug_line_str. These exactly
  // correspond to those in the compilation unit.
  const uint8_t* string_buffer_;
#ifndef NDEBUG
  uint64_t string_buffer_length_;
#endif
  const uint8_t* line_string_buffer_;
#ifndef NDEBUG
  uint64_t line_string_buffer_length_;
#endif

  const uint8_t* after_header_;
};

// This class is the main interface between the line info reader and
// the client.  The virtual functions inside this get called for
// interesting events that happen during line info reading.  The
// default implementation does nothing

class LineInfoHandler {
 public:
  LineInfoHandler() { }

  virtual ~LineInfoHandler() { }

  // Called when we define a directory.  NAME is the directory name,
  // DIR_NUM is the directory number
  virtual void DefineDir(const string& name, uint32_t dir_num) { }

  // Called when we define a filename. NAME is the filename, FILE_NUM
  // is the file number which is -1 if the file index is the next
  // index after the last numbered index (this happens when files are
  // dynamically defined by the line program), DIR_NUM is the
  // directory index for the directory name of this file, MOD_TIME is
  // the modification time of the file, and LENGTH is the length of
  // the file
  virtual void DefineFile(const string& name, int32_t file_num,
                          uint32_t dir_num, uint64_t mod_time,
                          uint64_t length) { }

  // Called when the line info reader has a new line, address pair
  // ready for us. ADDRESS is the address of the code, LENGTH is the
  // length of its machine code in bytes, FILE_NUM is the file number
  // containing the code, LINE_NUM is the line number in that file for
  // the code, and COLUMN_NUM is the column number the code starts at,
  // if we know it (0 otherwise).
  virtual void AddLine(uint64_t address, uint64_t length,
                       uint32_t file_num, uint32_t line_num, uint32_t column_num) { }
};

class RangeListHandler {
 public:
  RangeListHandler() { }

  virtual ~RangeListHandler() { }

  // Add a range.
  virtual void AddRange(uint64_t begin, uint64_t end) { };

  // Finish processing the range list.
  virtual void Finish() { };
};

class RangeListReader {
 public:
  // Reading a range list requires quite a bit of information
  // from the compilation unit. Package it conveniently.
  struct CURangesInfo {
    CURangesInfo() :
        version_(0), base_address_(0), ranges_base_(0),
        buffer_(nullptr), size_(0), addr_buffer_(nullptr),
        addr_buffer_size_(0), addr_base_(0) { }

    uint16_t version_;
    // Ranges base address. Ordinarily the CU's low_pc.
    uint64_t base_address_;
    // Offset into .debug_rnglists for this CU's rangelists.
    uint64_t ranges_base_;
    // Contents of either .debug_ranges or .debug_rnglists.
    const uint8_t* buffer_;
    uint64_t size_;
    // Contents of .debug_addr. This cu's contribution starts at
    // addr_base_
    const uint8_t* addr_buffer_;
    uint64_t addr_buffer_size_;
    uint64_t addr_base_;
  };

  RangeListReader(ByteReader* reader, CURangesInfo* cu_info,
                  RangeListHandler* handler) :
      reader_(reader), cu_info_(cu_info), handler_(handler),
      offset_array_(0) { }

  // Read ranges from cu_info as specified by form and data.
  bool ReadRanges(enum DwarfForm form, uint64_t data);

 private:
  // Read dwarf4 .debug_ranges at offset.
  bool ReadDebugRanges(uint64_t offset);
  // Read dwarf5 .debug_rngslist at offset.
  bool ReadDebugRngList(uint64_t offset);

  // Convenience functions to handle the mechanics of reading entries in the
  // ranges section.
  uint64_t ReadULEB(uint64_t offset, uint64_t* value) {
    size_t len;
    *value = reader_->ReadUnsignedLEB128(cu_info_->buffer_ + offset, &len);
    return len;
  }

  uint64_t ReadAddress(uint64_t offset, uint64_t* value) {
    *value = reader_->ReadAddress(cu_info_->buffer_ + offset);
    return reader_->AddressSize();
  }

  // Read the address at this CU's addr_index in the .debug_addr section.
  uint64_t GetAddressAtIndex(uint64_t addr_index) {
    assert(cu_info_->addr_buffer_ != nullptr);
    uint64_t offset =
        cu_info_->addr_base_ + addr_index * reader_->AddressSize();
    assert(offset < cu_info_->addr_buffer_size_);
    return reader_->ReadAddress(cu_info_->addr_buffer_ + offset);
  }

  ByteReader* reader_;
  CURangesInfo* cu_info_;
  RangeListHandler* handler_;
  uint64_t offset_array_;
};

// This class is the main interface between the reader and the
// client.  The virtual functions inside this get called for
// interesting events that happen during DWARF2 reading.
// The default implementation skips everything.
class Dwarf2Handler {
 public:
  Dwarf2Handler() { }

  virtual ~Dwarf2Handler() { }

  // Start to process a compilation unit at OFFSET from the beginning of the
  // .debug_info section. Return false if you would like to skip this
  // compilation unit.
  virtual bool StartCompilationUnit(uint64_t offset, uint8_t address_size,
                                    uint8_t offset_size, uint64_t cu_length,
                                    uint8_t dwarf_version) { return false; }

  // When processing a skeleton compilation unit, resulting from a split
  // DWARF compilation, once the skeleton debug info has been read,
  // the reader will call this function to ask the client if it needs
  // the full debug info from the .dwo or .dwp file.  Return true if
  // you need it, or false to skip processing the split debug info.
  virtual bool NeedSplitDebugInfo() { return true; }

  // Start to process a split compilation unit at OFFSET from the beginning of
  // the debug_info section in the .dwp/.dwo file.  Return false if you would
  // like to skip this compilation unit.
  virtual bool StartSplitCompilationUnit(uint64_t offset,
                                         uint64_t cu_length) { return false; }

  // Start to process a DIE at OFFSET from the beginning of the .debug_info
  // section. Return false if you would like to skip this DIE.
  virtual bool StartDIE(uint64_t offset, enum DwarfTag tag) { return false; }

  // Called when we have an attribute with unsigned data to give to our
  // handler. The attribute is for the DIE at OFFSET from the beginning of the
  // .debug_info section. Its name is ATTR, its form is FORM, and its value is
  // DATA.
  virtual void ProcessAttributeUnsigned(uint64_t offset,
                                        enum DwarfAttribute attr,
                                        enum DwarfForm form,
                                        uint64_t data) { }

  // Called when we have an attribute with signed data to give to our handler.
  // The attribute is for the DIE at OFFSET from the beginning of the
  // .debug_info section. Its name is ATTR, its form is FORM, and its value is
  // DATA.
  virtual void ProcessAttributeSigned(uint64_t offset,
                                      enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      int64_t data) { }

  // Called when we have an attribute whose value is a reference to
  // another DIE. The attribute belongs to the DIE at OFFSET from the
  // beginning of the .debug_info section. Its name is ATTR, its form
  // is FORM, and the offset of the DIE being referred to from the
  // beginning of the .debug_info section is DATA.
  virtual void ProcessAttributeReference(uint64_t offset,
                                         enum DwarfAttribute attr,
                                         enum DwarfForm form,
                                         uint64_t data) { }

  // Called when we have an attribute with a buffer of data to give to our
  // handler. The attribute is for the DIE at OFFSET from the beginning of the
  // .debug_info section. Its name is ATTR, its form is FORM, DATA points to
  // the buffer's contents, and its length in bytes is LENGTH. The buffer is
  // owned by the caller, not the callee, and may not persist for very long.
  // If you want the data to be available later, it needs to be copied.
  virtual void ProcessAttributeBuffer(uint64_t offset,
                                      enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      const uint8_t* data,
                                      uint64_t len) { }

  // Called when we have an attribute with string data to give to our handler.
  // The attribute is for the DIE at OFFSET from the beginning of the
  // .debug_info section. Its name is ATTR, its form is FORM, and its value is
  // DATA.
  virtual void ProcessAttributeString(uint64_t offset,
                                      enum DwarfAttribute attr,
                                      enum DwarfForm form,
                                      const string& data) { }

  // Called when we have an attribute whose value is the 64-bit signature
  // of a type unit in the .debug_types section. OFFSET is the offset of
  // the DIE whose attribute we're reporting. ATTR and FORM are the
  // attribute's name and form. SIGNATURE is the type unit's signature.
  virtual void ProcessAttributeSignature(uint64_t offset,
                                         enum DwarfAttribute attr,
                                         enum DwarfForm form,
                                         uint64_t signature) { }

  // Called when finished processing the DIE at OFFSET.
  // Because DWARF2/3 specifies a tree of DIEs, you may get starts
  // before ends of the previous DIE, as we process children before
  // ending the parent.
  virtual void EndDIE(uint64_t offset) { }

};

// The base of DWARF2/3 debug info is a DIE (Debugging Information
// Entry.
// DWARF groups DIE's into a tree and calls the root of this tree a
// "compilation unit".  Most of the time, there is one compilation
// unit in the .debug_info section for each file that had debug info
// generated.
// Each DIE consists of

// 1. a tag specifying a thing that is being described (ie
// DW_TAG_subprogram for functions, DW_TAG_variable for variables, etc
// 2. attributes (such as DW_AT_location for location in memory,
// DW_AT_name for name), and data for each attribute.
// 3. A flag saying whether the DIE has children or not

// In order to gain some amount of compression, the format of
// each DIE (tag name, attributes and data forms for the attributes)
// are stored in a separate table called the "abbreviation table".
// This is done because a large number of DIEs have the exact same tag
// and list of attributes, but different data for those attributes.
// As a result, the .debug_info section is just a stream of data, and
// requires reading of the .debug_abbrev section to say what the data
// means.

// As a warning to the user, it should be noted that the reason for
// using absolute offsets from the beginning of .debug_info is that
// DWARF2/3 supports referencing DIE's from other DIE's by their offset
// from either the current compilation unit start, *or* the beginning
// of the .debug_info section.  This means it is possible to reference
// a DIE in one compilation unit from a DIE in another compilation
// unit.  This style of reference is usually used to eliminate
// duplicated information that occurs across compilation
// units, such as base types, etc.  GCC 3.4+ support this with
// -feliminate-dwarf2-dups.  Other toolchains will sometimes do
// duplicate elimination in the linker.

class CompilationUnit {
 public:

  // Initialize a compilation unit.  This requires a map of sections,
  // the offset of this compilation unit in the .debug_info section, a
  // ByteReader, and a Dwarf2Handler class to call callbacks in.
  CompilationUnit(const string& path, const SectionMap& sections,
                  uint64_t offset, ByteReader* reader, Dwarf2Handler* handler);
  virtual ~CompilationUnit() {
    if (abbrevs_) delete abbrevs_;
  }

  // Initialize a compilation unit from a .dwo or .dwp file.
  // In this case, we need the .debug_addr section from the
  // executable file that contains the corresponding skeleton
  // compilation unit.  We also inherit the Dwarf2Handler from
  // the executable file, and call it as if we were still
  // processing the original compilation unit.
  void SetSplitDwarf(uint64_t addr_base, uint64_t dwo_id);

  // Begin reading a Dwarf2 compilation unit, and calling the
  // callbacks in the Dwarf2Handler

  // Return the full length of the compilation unit, including
  // headers. This plus the starting offset passed to the constructor
  // is the offset of the end of the compilation unit --- and the
  // start of the next compilation unit, if there is one.
  uint64_t Start();

  // Process the actual debug information in a split DWARF file.
  bool ProcessSplitDwarf(std::string& split_file,
                         SectionMap& sections,
                         ByteReader& split_byte_reader,
                         uint64_t& cu_offset);

  const uint8_t* GetAddrBuffer() { return addr_buffer_; }

  uint64_t GetAddrBufferLen() { return addr_buffer_length_; }

  uint64_t GetAddrBase() { return addr_base_; }

  uint64_t GetLowPC() { return low_pc_; }

  uint64_t GetDWOID() { return dwo_id_; }

  const uint8_t* GetLineBuffer() { return line_buffer_; }

  uint64_t GetLineBufferLen() { return line_buffer_length_; }

  const uint8_t* GetLineStrBuffer() { return line_string_buffer_; }

  uint64_t GetLineStrBufferLen() { return line_string_buffer_length_; }

  bool HasSourceLineInfo() { return has_source_line_info_; }

  uint64_t GetSourceLineOffset() { return source_line_offset_; }

  bool ShouldProcessSplitDwarf() { return should_process_split_dwarf_; }

 private:

  // This struct represents a single DWARF2/3 abbreviation
  // The abbreviation tells how to read a DWARF2/3 DIE, and consist of a
  // tag and a list of attributes, as well as the data form of each attribute.
  struct Abbrev {
    uint64_t number;
    enum DwarfTag tag;
    bool has_children;
    AttributeList attributes;
  };

  // A DWARF2/3 compilation unit header.  This is not the same size as
  // in the actual file, as the one in the file may have a 32 bit or
  // 64 bit length.
  struct CompilationUnitHeader {
    uint64_t length;
    uint16_t version;
    uint64_t abbrev_offset;
    uint8_t address_size;
  } header_;

  // Reads the DWARF2/3 header for this compilation unit.
  void ReadHeader();

  // Reads the DWARF2/3 abbreviations for this compilation unit
  void ReadAbbrevs();

  // Read the abbreviation offset for this compilation unit
  size_t ReadAbbrevOffset(const uint8_t* headerptr);

  // Read the address size for this compilation unit
  size_t ReadAddressSize(const uint8_t* headerptr);

  // Read the DWO id from a split or skeleton compilation unit header
  size_t ReadDwoId(const uint8_t* headerptr);

  // Read the type signature from a type or split type compilation unit header
  size_t ReadTypeSignature(const uint8_t* headerptr);

  // Read the DWO id from a split or skeleton compilation unit header
  size_t ReadTypeOffset(const uint8_t* headerptr);

  // Processes a single DIE for this compilation unit and return a new
  // pointer just past the end of it
  const uint8_t* ProcessDIE(uint64_t dieoffset,
                            const uint8_t* start,
                            const Abbrev& abbrev);

  // Processes a single attribute and return a new pointer just past the
  // end of it
  const uint8_t* ProcessAttribute(uint64_t dieoffset,
                                  const uint8_t* start,
                                  enum DwarfAttribute attr,
                                  enum DwarfForm form,
                                  uint64_t implicit_const);

  // Special version of ProcessAttribute, for finding str_offsets_base and
  // DW_AT_addr_base in DW_TAG_compile_unit, for DWARF v5.
  const uint8_t* ProcessOffsetBaseAttribute(uint64_t dieoffset,
                                            const uint8_t* start,
                                            enum DwarfAttribute attr,
                                            enum DwarfForm form,
                                            uint64_t implicit_const);

  // Called when we have an attribute with unsigned data to give to
  // our handler.  The attribute is for the DIE at OFFSET from the
  // beginning of compilation unit, has a name of ATTR, a form of
  // FORM, and the actual data of the attribute is in DATA.
  // If we see a DW_AT_GNU_dwo_id attribute, save the value so that
  // we can find the debug info in a .dwo or .dwp file.
  void ProcessAttributeUnsigned(uint64_t offset,
                                enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data) {
    if (attr == DW_AT_GNU_dwo_id) {
      dwo_id_ = data;
    }
    else if (attr == DW_AT_GNU_addr_base || attr == DW_AT_addr_base) {
      addr_base_ = data;
    }
    else if (attr == DW_AT_str_offsets_base) {
      str_offsets_base_ = data;
    }
    else if (attr == DW_AT_low_pc) {
      low_pc_ = data;
    }
    else if (attr == DW_AT_stmt_list) {
      has_source_line_info_ = true;
      source_line_offset_ = data;
    }
    handler_->ProcessAttributeUnsigned(offset, attr, form, data);
  }

  // Called when we have an attribute with signed data to give to
  // our handler.  The attribute is for the DIE at OFFSET from the
  // beginning of compilation unit, has a name of ATTR, a form of
  // FORM, and the actual data of the attribute is in DATA.
  void ProcessAttributeSigned(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              int64_t data) {
    handler_->ProcessAttributeSigned(offset, attr, form, data);
  }

  // Called when we have an attribute with a buffer of data to give to
  // our handler.  The attribute is for the DIE at OFFSET from the
  // beginning of compilation unit, has a name of ATTR, a form of
  // FORM, and the actual data of the attribute is in DATA, and the
  // length of the buffer is LENGTH.
  void ProcessAttributeBuffer(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const uint8_t* data,
                              uint64_t len) {
    handler_->ProcessAttributeBuffer(offset, attr, form, data, len);
  }

  // Handles the common parts of DW_FORM_GNU_str_index, DW_FORM_strx,
  // DW_FORM_strx1, DW_FORM_strx2, DW_FORM_strx3, and DW_FORM_strx4.
  // Retrieves the data and calls through to ProcessAttributeString.
  void ProcessFormStringIndex(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              uint64_t str_index);

  // Called when we have an attribute with string data to give to
  // our handler.  The attribute is for the DIE at OFFSET from the
  // beginning of compilation unit, has a name of ATTR, a form of
  // FORM, and the actual data of the attribute is in DATA.
  // If we see a DW_AT_GNU_dwo_name attribute, save the value so
  // that we can find the debug info in a .dwo or .dwp file.
  void ProcessAttributeString(uint64_t offset,
                              enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const char* data) {
    if (attr == DW_AT_GNU_dwo_name || attr == DW_AT_dwo_name)
      dwo_name_ = data;
    handler_->ProcessAttributeString(offset, attr, form, data);
  }

  // Called to handle common portions of DW_FORM_addrx and variations, as well
  // as DW_FORM_GNU_addr_index.
  void ProcessAttributeAddrIndex(uint64_t offset,
                                 enum DwarfAttribute attr,
                                 enum DwarfForm form,
                                 uint64_t addr_index) {
    const uint8_t* addr_ptr =
        addr_buffer_ + addr_base_ + addr_index * reader_->AddressSize();
    ProcessAttributeUnsigned(
        offset, attr, form, reader_->ReadAddress(addr_ptr));
  }

  // Processes all DIEs for this compilation unit
  bool ProcessDIEs();

  // Skips the die with attributes specified in ABBREV starting at
  // START, and return the new place to position the stream to.
  const uint8_t* SkipDIE(const uint8_t* start, const Abbrev& abbrev);

  // Skips the attribute starting at START, with FORM, and return the
  // new place to position the stream to.
  const uint8_t* SkipAttribute(const uint8_t* start, enum DwarfForm form);

  // Read the debug sections from a .dwo file.
  void ReadDebugSectionsFromDwo(ElfReader* elf_reader,
                                SectionMap* sections);

  // Path of the file containing the debug information.
  const string path_;

  // Offset from section start is the offset of this compilation unit
  // from the beginning of the .debug_info/.debug_info.dwo section.
  uint64_t offset_from_section_start_;

  // buffer is the buffer for our CU, starting at .debug_info + offset
  // passed in from constructor.
  // after_header points to right after the compilation unit header.
  const uint8_t* buffer_;
  uint64_t buffer_length_;
  const uint8_t* after_header_;

  // The associated ByteReader that handles endianness issues for us
  ByteReader* reader_;

  // The map of sections in our file to buffers containing their data
  const SectionMap& sections_;

  // The associated handler to call processing functions in
  Dwarf2Handler* handler_;

  // Set of DWARF2/3 abbreviations for this compilation unit.  Indexed
  // by abbreviation number, which means that abbrevs_[0] is not
  // valid.
  std::vector<Abbrev>* abbrevs_;

  // String section buffer and length, if we have a string section.
  // This is here to avoid doing a section lookup for strings in
  // ProcessAttribute, which is in the hot path for DWARF2 reading.
  const uint8_t* string_buffer_;
  uint64_t string_buffer_length_;

  // Similarly for .debug_line_str.
  const uint8_t* line_string_buffer_;
  uint64_t line_string_buffer_length_;

  // String offsets section buffer and length, if we have a string offsets
  // section (.debug_str_offsets or .debug_str_offsets.dwo).
  const uint8_t* str_offsets_buffer_;
  uint64_t str_offsets_buffer_length_;

  // Address section buffer and length, if we have an address section
  // (.debug_addr).
  const uint8_t* addr_buffer_;
  uint64_t addr_buffer_length_;

  // .debug_line section buffer and length.
  const uint8_t* line_buffer_;
  uint64_t line_buffer_length_;

  // Flag indicating whether this compilation unit is part of a .dwo
  // or .dwp file.  If true, we are reading this unit because a
  // skeleton compilation unit in an executable file had a
  // DW_AT_GNU_dwo_name or DW_AT_GNU_dwo_id attribute.
  // In a .dwo file, we expect the string offsets section to
  // have a ".dwo" suffix, and we will use the ".debug_addr" section
  // associated with the skeleton compilation unit.
  bool is_split_dwarf_;

  // Flag indicating if it's a Type Unit (only applicable to DWARF v5).
  bool is_type_unit_;

  // The value of the DW_AT_GNU_dwo_id attribute, if any.
  uint64_t dwo_id_;

  // The value of the DW_AT_GNU_type_signature attribute, if any.
  uint64_t type_signature_;

  // The value of the DW_AT_GNU_type_offset attribute, if any.
  size_t type_offset_;

  // The value of the DW_AT_GNU_dwo_name attribute, if any.
  const char* dwo_name_;

  // If this is a split DWARF CU, the value of the DW_AT_GNU_dwo_id attribute
  // from the skeleton CU.
  uint64_t skeleton_dwo_id_;

  // The value of the DW_AT_GNU_addr_base attribute, if any.
  uint64_t addr_base_;

  // The value of DW_AT_str_offsets_base attribute, if any.
  uint64_t str_offsets_base_;

  // True if we have already looked for a .dwp file.
  bool have_checked_for_dwp_;

  // ElfReader for the dwo/dwo file.
  std::unique_ptr<ElfReader> split_elf_reader_;

  // DWP reader.
  std::unique_ptr<DwpReader> dwp_reader_;

  bool should_process_split_dwarf_;

  // The value of the DW_AT_low_pc attribute, if any.
  uint64_t low_pc_;

  // The value of DW_AT_stmt_list attribute if any.
  bool has_source_line_info_;
  uint64_t source_line_offset_;
};

// A Reader for a .dwp file.  Supports the fetching of DWARF debug
// info for a given dwo_id.
//
// There are two versions of .dwp files.  In both versions, the
// .dwp file is an ELF file containing only debug sections.
// In Version 1, the file contains many copies of each debug
// section, one for each .dwo file that is packaged in the .dwp
// file, and the .debug_cu_index section maps from the dwo_id
// to a set of section indexes.  In Version 2, the file contains
// one of each debug section, and the .debug_cu_index section
// maps from the dwo_id to a set of offsets and lengths that
// identify each .dwo file's contribution to the larger sections.

class DwpReader {
 public:
  DwpReader(const ByteReader& byte_reader, ElfReader* elf_reader);

  // Read the CU index and initialize data members.
  void Initialize();

  // Read the debug sections for the given dwo_id.
  void ReadDebugSectionsForCU(uint64_t dwo_id, SectionMap* sections);

 private:
  // Search a v1 hash table for "dwo_id".  Returns the slot index
  // where the dwo_id was found, or -1 if it was not found.
  int LookupCU(uint64_t dwo_id);

  // Search a v2 hash table for "dwo_id".  Returns the row index
  // in the offsets and sizes tables, or 0 if it was not found.
  uint32_t LookupCUv2(uint64_t dwo_id);

  // The ELF reader for the .dwp file.
  ElfReader* elf_reader_;

  // The ByteReader for the .dwp file.
  const ByteReader& byte_reader_;

  // Pointer to the .debug_cu_index section.
  const char* cu_index_;

  // Size of the .debug_cu_index section.
  size_t cu_index_size_;

  // Pointer to the .debug_str.dwo section.
  const char* string_buffer_;

  // Size of the .debug_str.dwo section.
  size_t string_buffer_size_;

  // Version of the .dwp file.  We support versions 1 and 2 currently.
  int version_;

  // Number of columns in the section tables (version 2).
  unsigned int ncolumns_;

  // Number of units in the section tables (version 2).
  unsigned int nunits_;

  // Number of slots in the hash table.
  unsigned int nslots_;

  // Pointer to the beginning of the hash table.
  const char* phash_;

  // Pointer to the beginning of the index table.
  const char* pindex_;

  // Pointer to the beginning of the section index pool (version 1).
  const char* shndx_pool_;

  // Pointer to the beginning of the section offset table (version 2).
  const char* offset_table_;

  // Pointer to the beginning of the section size table (version 2).
  const char* size_table_;

  // Contents of the sections of interest (version 2).
  const char* abbrev_data_;
  size_t abbrev_size_;
  const char* info_data_;
  size_t info_size_;
  const char* str_offsets_data_;
  size_t str_offsets_size_;
  const char* rnglist_data_;
  size_t rnglist_size_;
};

// This class is a reader for DWARF's Call Frame Information.  CFI
// describes how to unwind stack frames --- even for functions that do
// not follow fixed conventions for saving registers, whose frame size
// varies as they execute, etc.
//
// CFI describes, at each machine instruction, how to compute the
// stack frame's base address, how to find the return address, and
// where to find the saved values of the caller's registers (if the
// callee has stashed them somewhere to free up the registers for its
// own use).
//
// For example, suppose we have a function whose machine code looks
// like this (imagine an assembly language that looks like C, for a
// machine with 32-bit registers, and a stack that grows towards lower
// addresses):
//
// func:                                ; entry point; return address at sp
// func+0:      sp = sp - 16            ; allocate space for stack frame
// func+1:      sp[12] = r0             ; save r0 at sp+12
// ...                                  ; other code, not frame-related
// func+10:     sp -= 4; *sp = x        ; push some x on the stack
// ...                                  ; other code, not frame-related
// func+20:     r0 = sp[16]             ; restore saved r0
// func+21:     sp += 20                ; pop whole stack frame
// func+22:     pc = *sp; sp += 4       ; pop return address and jump to it
//
// DWARF CFI is (a very compressed representation of) a table with a
// row for each machine instruction address and a column for each
// register showing how to restore it, if possible.
//
// A special column named "CFA", for "Canonical Frame Address", tells how
// to compute the base address of the frame; registers' entries may
// refer to the CFA in describing where the registers are saved.
//
// Another special column, named "RA", represents the return address.
//
// For example, here is a complete (uncompressed) table describing the
// function above:
//
//     insn      cfa    r0      r1 ...  ra
//     =======================================
//     func+0:   sp                     cfa[0]
//     func+1:   sp+16                  cfa[0]
//     func+2:   sp+16  cfa[-4]         cfa[0]
//     func+11:  sp+20  cfa[-4]         cfa[0]
//     func+21:  sp+20                  cfa[0]
//     func+22:  sp                     cfa[0]
//
// Some things to note here:
//
// - Each row describes the state of affairs *before* executing the
//   instruction at the given address.  Thus, the row for func+0
//   describes the state before we allocate the stack frame.  In the
//   next row, the formula for computing the CFA has changed,
//   reflecting that allocation.
//
// - The other entries are written in terms of the CFA; this allows
//   them to remain unchanged as the stack pointer gets bumped around.
//   For example, the rule for recovering the return address (the "ra"
//   column) remains unchanged throughout the function, even as the
//   stack pointer takes on three different offsets from the return
//   address.
//
// - Although we haven't shown it, most calling conventions designate
//   "callee-saves" and "caller-saves" registers. The callee must
//   preserve the values of callee-saves registers; if it uses them,
//   it must save their original values somewhere, and restore them
//   before it returns. In contrast, the callee is free to trash
//   caller-saves registers; if the callee uses these, it will
//   probably not bother to save them anywhere, and the CFI will
//   probably mark their values as "unrecoverable".
//
//   (However, since the caller cannot assume the callee was going to
//   save them, caller-saves registers are probably dead in the caller
//   anyway, so compilers usually don't generate CFA for caller-saves
//   registers.)
//
// - Exactly where the CFA points is a matter of convention that
//   depends on the architecture and ABI in use. In the example, the
//   CFA is the value the stack pointer had upon entry to the
//   function, pointing at the saved return address. But on the x86,
//   the call frame information generated by GCC follows the
//   convention that the CFA is the address *after* the saved return
//   address.
//
//   But by definition, the CFA remains constant throughout the
//   lifetime of the frame. This makes it a useful value for other
//   columns to refer to. It is also gives debuggers a useful handle
//   for identifying a frame.
//
// If you look at the table above, you'll notice that a given entry is
// often the same as the one immediately above it: most instructions
// change only one or two aspects of the stack frame, if they affect
// it at all. The DWARF format takes advantage of this fact, and
// reduces the size of the data by mentioning only the addresses and
// columns at which changes take place. So for the above, DWARF CFI
// data would only actually mention the following:
//
//     insn      cfa    r0      r1 ...  ra
//     =======================================
//     func+0:   sp                     cfa[0]
//     func+1:   sp+16
//     func+2:          cfa[-4]
//     func+11:  sp+20
//     func+21:         r0
//     func+22:  sp
//
// In fact, this is the way the parser reports CFI to the consumer: as
// a series of statements of the form, "At address X, column Y changed
// to Z," and related conventions for describing the initial state.
//
// Naturally, it would be impractical to have to scan the entire
// program's CFI, noting changes as we go, just to recover the
// unwinding rules in effect at one particular instruction. To avoid
// this, CFI data is grouped into "entries", each of which covers a
// specified range of addresses and begins with a complete statement
// of the rules for all recoverable registers at that starting
// address. Each entry typically covers a single function.
//
// Thus, to compute the contents of a given row of the table --- that
// is, rules for recovering the CFA, RA, and registers at a given
// instruction --- the consumer should find the entry that covers that
// instruction's address, start with the initial state supplied at the
// beginning of the entry, and work forward until it has processed all
// the changes up to and including those for the present instruction.
//
// There are seven kinds of rules that can appear in an entry of the
// table:
//
// - "undefined": The given register is not preserved by the callee;
//   its value cannot be recovered.
//
// - "same value": This register has the same value it did in the callee.
//
// - offset(N): The register is saved at offset N from the CFA.
//
// - val_offset(N): The value the register had in the caller is the
//   CFA plus offset N. (This is usually only useful for describing
//   the stack pointer.)
//
// - register(R): The register's value was saved in another register R.
//
// - expression(E): Evaluating the DWARF expression E using the
//   current frame's registers' values yields the address at which the
//   register was saved.
//
// - val_expression(E): Evaluating the DWARF expression E using the
//   current frame's registers' values yields the value the register
//   had in the caller.

class CallFrameInfo {
 public:
  // The different kinds of entries one finds in CFI. Used internally,
  // and for error reporting.
  enum EntryKind { kUnknown, kCIE, kFDE, kTerminator };

  // The handler class to which the parser hands the parsed call frame
  // information.  Defined below.
  class Handler;

  // A reporter class, which CallFrameInfo uses to report errors
  // encountered while parsing call frame information.  Defined below.
  class Reporter;

  // Create a DWARF CFI parser. BUFFER points to the contents of the
  // .debug_frame section to parse; BUFFER_LENGTH is its length in bytes.
  // REPORTER is an error reporter the parser should use to report
  // problems. READER is a ByteReader instance that has the endianness and
  // address size set properly. Report the data we find to HANDLER.
  //
  // This class can also parse Linux C++ exception handling data, as found
  // in '.eh_frame' sections. This data is a variant of DWARF CFI that is
  // placed in loadable segments so that it is present in the program's
  // address space, and is interpreted by the C++ runtime to search the
  // call stack for a handler interested in the exception being thrown,
  // actually pop the frames, and find cleanup code to run.
  //
  // There are two differences between the call frame information described
  // in the DWARF standard and the exception handling data Linux places in
  // the .eh_frame section:
  //
  // - Exception handling data uses uses a different format for call frame
  //   information entry headers. The distinguished CIE id, the way FDEs
  //   refer to their CIEs, and the way the end of the series of entries is
  //   determined are all slightly different.
  //
  //   If the constructor's EH_FRAME argument is true, then the
  //   CallFrameInfo parses the entry headers as Linux C++ exception
  //   handling data. If EH_FRAME is false or omitted, the CallFrameInfo
  //   parses standard DWARF call frame information.
  //
  // - Linux C++ exception handling data uses CIE augmentation strings
  //   beginning with 'z' to specify the presence of additional data after
  //   the CIE and FDE headers and special encodings used for addresses in
  //   frame description entries.
  //
  //   CallFrameInfo can handle 'z' augmentations in either DWARF CFI or
  //   exception handling data if you have supplied READER with the base
  //   addresses needed to interpret the pointer encodings that 'z'
  //   augmentations can specify. See the ByteReader interface for details
  //   about the base addresses. See the CallFrameInfo::Handler interface
  //   for details about the additional information one might find in
  //   'z'-augmented data.
  //
  // Thus:
  //
  // - If you are parsing standard DWARF CFI, as found in a .debug_frame
  //   section, you should pass false for the EH_FRAME argument, or omit
  //   it, and you need not worry about providing READER with the
  //   additional base addresses.
  //
  // - If you want to parse Linux C++ exception handling data from a
  //   .eh_frame section, you should pass EH_FRAME as true, and call
  //   READER's Set*Base member functions before calling our Start method.
  //
  // - If you want to parse DWARF CFI that uses the 'z' augmentations
  //   (although I don't think any toolchain ever emits such data), you
  //   could pass false for EH_FRAME, but call READER's Set*Base members.
  //
  // The extensions the Linux C++ ABI makes to DWARF for exception
  // handling are described here, rather poorly:
  // http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/dwarfext.html
  // http://refspecs.linux-foundation.org/LSB_4.0.0/LSB-Core-generic/LSB-Core-generic/ehframechpt.html
  //
  // The mechanics of C++ exception handling, personality routines,
  // and language-specific data areas are described here, rather nicely:
  // http://www.codesourcery.com/public/cxx-abi/abi-eh.html
  CallFrameInfo(const uint8_t* buffer, size_t buffer_length,
                ByteReader* reader, Handler* handler, Reporter* reporter,
                bool eh_frame = false)
      : buffer_(buffer), buffer_length_(buffer_length),
        reader_(reader), handler_(handler), reporter_(reporter),
        eh_frame_(eh_frame) { }

  ~CallFrameInfo() { }

  // Parse the entries in BUFFER, reporting what we find to HANDLER.
  // Return true if we reach the end of the section successfully, or
  // false if we encounter an error.
  bool Start();

  // Return the textual name of KIND. For error reporting.
  static const char* KindName(EntryKind kind);

 private:

  struct CIE;

  // A CFI entry, either an FDE or a CIE.
  struct Entry {
    // The starting offset of the entry in the section, for error
    // reporting.
    size_t offset;

    // The start of this entry in the buffer.
    const uint8_t* start;

    // Which kind of entry this is.
    //
    // We want to be able to use this for error reporting even while we're
    // in the midst of parsing. Error reporting code may assume that kind,
    // offset, and start fields are valid, although kind may be kUnknown.
    EntryKind kind;

    // The end of this entry's common prologue (initial length and id), and
    // the start of this entry's kind-specific fields.
    const uint8_t* fields;

    // The start of this entry's instructions.
    const uint8_t* instructions;

    // The address past the entry's last byte in the buffer. (Note that
    // since offset points to the entry's initial length field, and the
    // length field is the number of bytes after that field, this is not
    // simply buffer_ + offset + length.)
    const uint8_t* end;

    // For both DWARF CFI and .eh_frame sections, this is the CIE id in a
    // CIE, and the offset of the associated CIE in an FDE.
    uint64_t id;

    // The CIE that applies to this entry, if we've parsed it. If this is a
    // CIE, then this field points to this structure.
    CIE* cie;
  };

  // A common information entry (CIE).
  struct CIE: public Entry {
    uint8_t version;                      // CFI data version number
    string augmentation;                // vendor format extension markers
    uint64_t code_alignment_factor;       // scale for code address adjustments
    int data_alignment_factor;          // scale for stack pointer adjustments
    unsigned return_address_register;   // which register holds the return addr

    // True if this CIE includes Linux C++ ABI 'z' augmentation data.
    bool has_z_augmentation;

    // Parsed 'z' augmentation data. These are meaningful only if
    // has_z_augmentation is true.
    bool has_z_lsda;                    // The 'z' augmentation included 'L'.
    bool has_z_personality;             // The 'z' augmentation included 'P'.
    bool has_z_signal_frame;            // The 'z' augmentation included 'S'.

    // If has_z_lsda is true, this is the encoding to be used for language-
    // specific data area pointers in FDEs.
    DwarfPointerEncoding lsda_encoding;

    // If has_z_personality is true, this is the encoding used for the
    // personality routine pointer in the augmentation data.
    DwarfPointerEncoding personality_encoding;

    // If has_z_personality is true, this is the address of the personality
    // routine --- or, if personality_encoding & DW_EH_PE_indirect, the
    // address where the personality routine's address is stored.
    uint64_t personality_address;

    // This is the encoding used for addresses in the FDE header and
    // in DW_CFA_set_loc instructions. This is always valid, whether
    // or not we saw a 'z' augmentation string; its default value is
    // DW_EH_PE_absptr, which is what normal DWARF CFI uses.
    DwarfPointerEncoding pointer_encoding;

    // These were only introduced in DWARF4, so will not be set in older
    // versions.
    uint8_t address_size;
    uint8_t segment_size;
  };

  // A frame description entry (FDE).
  struct FDE: public Entry {
    uint64_t address;                     // start address of described code
    uint64_t size;                        // size of described code, in bytes

    // If cie->has_z_lsda is true, then this is the language-specific data
    // area's address --- or its address's address, if cie->lsda_encoding
    // has the DW_EH_PE_indirect bit set.
    uint64_t lsda_address;
  };

  // Internal use.
  class Rule;
  class UndefinedRule;
  class SameValueRule;
  class OffsetRule;
  class ValOffsetRule;
  class RegisterRule;
  class ExpressionRule;
  class ValExpressionRule;
  class RuleMap;
  class State;

  // Parse the initial length and id of a CFI entry, either a CIE, an FDE,
  // or a .eh_frame end-of-data mark. CURSOR points to the beginning of the
  // data to parse. On success, populate ENTRY as appropriate, and return
  // true. On failure, report the problem, and return false. Even if we
  // return false, set ENTRY->end to the first byte after the entry if we
  // were able to figure that out, or NULL if we weren't.
  bool ReadEntryPrologue(const uint8_t* cursor, Entry* entry);

  // Parse the fields of a CIE after the entry prologue, including any 'z'
  // augmentation data. Assume that the 'Entry' fields of CIE are
  // populated; use CIE->fields and CIE->end as the start and limit for
  // parsing. On success, populate the rest of *CIE, and return true; on
  // failure, report the problem and return false.
  bool ReadCIEFields(CIE* cie);

  // Parse the fields of an FDE after the entry prologue, including any 'z'
  // augmentation data. Assume that the 'Entry' fields of *FDE are
  // initialized; use FDE->fields and FDE->end as the start and limit for
  // parsing. Assume that FDE->cie is fully initialized. On success,
  // populate the rest of *FDE, and return true; on failure, report the
  // problem and return false.
  bool ReadFDEFields(FDE* fde);

  // Report that ENTRY is incomplete, and return false. This is just a
  // trivial wrapper for invoking reporter_->Incomplete; it provides a
  // little brevity.
  bool ReportIncomplete(Entry* entry);

  // Return true if ENCODING has the DW_EH_PE_indirect bit set.
  static bool IsIndirectEncoding(DwarfPointerEncoding encoding) {
    return encoding & DW_EH_PE_indirect;
  }

  // The contents of the DWARF .debug_info section we're parsing.
  const uint8_t* buffer_;
  size_t buffer_length_;

  // For reading multi-byte values with the appropriate endianness.
  ByteReader* reader_;

  // The handler to which we should report the data we find.
  Handler* handler_;

  // For reporting problems in the info we're parsing.
  Reporter* reporter_;

  // True if we are processing .eh_frame-format data.
  bool eh_frame_;
};

// The handler class for CallFrameInfo.  The a CFI parser calls the
// member functions of a handler object to report the data it finds.
class CallFrameInfo::Handler {
 public:
  // The pseudo-register number for the canonical frame address.
  enum { kCFARegister = -1 };

  Handler() { }
  virtual ~Handler() { }

  // The parser has found CFI for the machine code at ADDRESS,
  // extending for LENGTH bytes. OFFSET is the offset of the frame
  // description entry in the section, for use in error messages.
  // VERSION is the version number of the CFI format. AUGMENTATION is
  // a string describing any producer-specific extensions present in
  // the data. RETURN_ADDRESS is the number of the register that holds
  // the address to which the function should return.
  //
  // Entry should return true to process this CFI, or false to skip to
  // the next entry.
  //
  // The parser invokes Entry for each Frame Description Entry (FDE)
  // it finds.  The parser doesn't report Common Information Entries
  // to the handler explicitly; instead, if the handler elects to
  // process a given FDE, the parser reiterates the appropriate CIE's
  // contents at the beginning of the FDE's rules.
  virtual bool Entry(size_t offset, uint64_t address, uint64_t length,
                     uint8_t version, const string& augmentation,
                     unsigned return_address) = 0;

  // When the Entry function returns true, the parser calls these
  // handler functions repeatedly to describe the rules for recovering
  // registers at each instruction in the given range of machine code.
  // Immediately after a call to Entry, the handler should assume that
  // the rule for each callee-saves register is "unchanged" --- that
  // is, that the register still has the value it had in the caller.
  //
  // If a *Rule function returns true, we continue processing this entry's
  // instructions. If a *Rule function returns false, we stop evaluating
  // instructions, and skip to the next entry. Either way, we call End
  // before going on to the next entry.
  //
  // In all of these functions, if the REG parameter is kCFARegister, then
  // the rule describes how to find the canonical frame address.
  // kCFARegister may be passed as a BASE_REGISTER argument, meaning that
  // the canonical frame address should be used as the base address for the
  // computation. All other REG values will be positive.

  // At ADDRESS, register REG's value is not recoverable.
  virtual bool UndefinedRule(uint64_t address, int reg) = 0;

  // At ADDRESS, register REG's value is the same as that it had in
  // the caller.
  virtual bool SameValueRule(uint64_t address, int reg) = 0;

  // At ADDRESS, register REG has been saved at offset OFFSET from
  // BASE_REGISTER.
  virtual bool OffsetRule(uint64_t address, int reg,
                          int base_register, long offset) = 0;

  // At ADDRESS, the caller's value of register REG is the current
  // value of BASE_REGISTER plus OFFSET. (This rule doesn't provide an
  // address at which the register's value is saved.)
  virtual bool ValOffsetRule(uint64_t address, int reg,
                             int base_register, long offset) = 0;

  // At ADDRESS, register REG has been saved in BASE_REGISTER. This differs
  // from ValOffsetRule(ADDRESS, REG, BASE_REGISTER, 0), in that
  // BASE_REGISTER is the "home" for REG's saved value: if you want to
  // assign to a variable whose home is REG in the calling frame, you
  // should put the value in BASE_REGISTER.
  virtual bool RegisterRule(uint64_t address, int reg, int base_register) = 0;

  // At ADDRESS, the DWARF expression EXPRESSION yields the address at
  // which REG was saved.
  virtual bool ExpressionRule(uint64_t address, int reg,
                              const string& expression) = 0;

  // At ADDRESS, the DWARF expression EXPRESSION yields the caller's
  // value for REG. (This rule doesn't provide an address at which the
  // register's value is saved.)
  virtual bool ValExpressionRule(uint64_t address, int reg,
                                 const string& expression) = 0;

  // Indicate that the rules for the address range reported by the
  // last call to Entry are complete.  End should return true if
  // everything is okay, or false if an error has occurred and parsing
  // should stop.
  virtual bool End() = 0;

  // The target architecture for the data.
  virtual string Architecture() = 0;

  // Handler functions for Linux C++ exception handling data. These are
  // only called if the data includes 'z' augmentation strings.

  // The Linux C++ ABI uses an extension of the DWARF CFI format to
  // walk the stack to propagate exceptions from the throw to the
  // appropriate catch, and do the appropriate cleanups along the way.
  // CFI entries used for exception handling have two additional data
  // associated with them:
  //
  // - The "language-specific data area" describes which exception
  //   types the function has 'catch' clauses for, and indicates how
  //   to go about re-entering the function at the appropriate catch
  //   clause. If the exception is not caught, it describes the
  //   destructors that must run before the frame is popped.
  //
  // - The "personality routine" is responsible for interpreting the
  //   language-specific data area's contents, and deciding whether
  //   the exception should continue to propagate down the stack,
  //   perhaps after doing some cleanup for this frame, or whether the
  //   exception will be caught here.
  //
  // In principle, the language-specific data area is opaque to
  // everybody but the personality routine. In practice, these values
  // may be useful or interesting to readers with extra context, and
  // we have to at least skip them anyway, so we might as well report
  // them to the handler.

  // This entry's exception handling personality routine's address is
  // ADDRESS. If INDIRECT is true, then ADDRESS is the address at
  // which the routine's address is stored. The default definition for
  // this handler function simply returns true, allowing parsing of
  // the entry to continue.
  virtual bool PersonalityRoutine(uint64_t address, bool indirect) {
    return true;
  }

  // This entry's language-specific data area (LSDA) is located at
  // ADDRESS. If INDIRECT is true, then ADDRESS is the address at
  // which the area's address is stored. The default definition for
  // this handler function simply returns true, allowing parsing of
  // the entry to continue.
  virtual bool LanguageSpecificDataArea(uint64_t address, bool indirect) {
    return true;
  }

  // This entry describes a signal trampoline --- this frame is the
  // caller of a signal handler. The default definition for this
  // handler function simply returns true, allowing parsing of the
  // entry to continue.
  //
  // The best description of the rationale for and meaning of signal
  // trampoline CFI entries seems to be in the GCC bug database:
  // http://gcc.gnu.org/bugzilla/show_bug.cgi?id=26208
  virtual bool SignalHandler() { return true; }
};

// The CallFrameInfo class makes calls on an instance of this class to
// report errors or warn about problems in the data it is parsing. The
// default definitions of these methods print a message to stderr, but
// you can make a derived class that overrides them.
class CallFrameInfo::Reporter {
 public:
  // Create an error reporter which attributes troubles to the section
  // named SECTION in FILENAME.
  //
  // Normally SECTION would be .debug_frame, but the Mac puts CFI data
  // in a Mach-O section named __debug_frame. If we support
  // Linux-style exception handling data, we could be reading an
  // .eh_frame section.
  Reporter(const string& filename,
           const string& section = ".debug_frame")
      : filename_(filename), section_(section) { }
  virtual ~Reporter() { }

  // The CFI entry at OFFSET ends too early to be well-formed. KIND
  // indicates what kind of entry it is; KIND can be kUnknown if we
  // haven't parsed enough of the entry to tell yet.
  virtual void Incomplete(uint64_t offset, CallFrameInfo::EntryKind kind);

  // The .eh_frame data has a four-byte zero at OFFSET where the next
  // entry's length would be; this is a terminator. However, the buffer
  // length as given to the CallFrameInfo constructor says there should be
  // more data.
  virtual void EarlyEHTerminator(uint64_t offset);

  // The FDE at OFFSET refers to the CIE at CIE_OFFSET, but the
  // section is not that large.
  virtual void CIEPointerOutOfRange(uint64_t offset, uint64_t cie_offset);

  // The FDE at OFFSET refers to the CIE at CIE_OFFSET, but the entry
  // there is not a CIE.
  virtual void BadCIEId(uint64_t offset, uint64_t cie_offset);

  // The FDE at OFFSET refers to a CIE with an address size we don't know how
  // to handle.
  virtual void UnexpectedAddressSize(uint64_t offset, uint8_t address_size);

  // The FDE at OFFSET refers to a CIE with an segment descriptor size we
  // don't know how to handle.
  virtual void UnexpectedSegmentSize(uint64_t offset, uint8_t segment_size);

  // The FDE at OFFSET refers to a CIE with version number VERSION,
  // which we don't recognize. We cannot parse DWARF CFI if it uses
  // a version number we don't recognize.
  virtual void UnrecognizedVersion(uint64_t offset, int version);

  // The FDE at OFFSET refers to a CIE with augmentation AUGMENTATION,
  // which we don't recognize. We cannot parse DWARF CFI if it uses
  // augmentations we don't recognize.
  virtual void UnrecognizedAugmentation(uint64_t offset,
                                        const string& augmentation);

  // The pointer encoding ENCODING, specified by the CIE at OFFSET, is not
  // a valid encoding.
  virtual void InvalidPointerEncoding(uint64_t offset, uint8_t encoding);

  // The pointer encoding ENCODING, specified by the CIE at OFFSET, depends
  // on a base address which has not been supplied.
  virtual void UnusablePointerEncoding(uint64_t offset, uint8_t encoding);

  // The CIE at OFFSET contains a DW_CFA_restore instruction at
  // INSN_OFFSET, which may not appear in a CIE.
  virtual void RestoreInCIE(uint64_t offset, uint64_t insn_offset);

  // The entry at OFFSET, of kind KIND, has an unrecognized
  // instruction at INSN_OFFSET.
  virtual void BadInstruction(uint64_t offset, CallFrameInfo::EntryKind kind,
                              uint64_t insn_offset);

  // The instruction at INSN_OFFSET in the entry at OFFSET, of kind
  // KIND, establishes a rule that cites the CFA, but we have not
  // established a CFA rule yet.
  virtual void NoCFARule(uint64_t offset, CallFrameInfo::EntryKind kind,
                         uint64_t insn_offset);

  // The instruction at INSN_OFFSET in the entry at OFFSET, of kind
  // KIND, is a DW_CFA_restore_state instruction, but the stack of
  // saved states is empty.
  virtual void EmptyStateStack(uint64_t offset, CallFrameInfo::EntryKind kind,
                               uint64_t insn_offset);

  // The DW_CFA_remember_state instruction at INSN_OFFSET in the entry
  // at OFFSET, of kind KIND, would restore a state that has no CFA
  // rule, whereas the current state does have a CFA rule. This is
  // bogus input, which the CallFrameInfo::Handler interface doesn't
  // (and shouldn't) have any way to report.
  virtual void ClearingCFARule(uint64_t offset, CallFrameInfo::EntryKind kind,
                               uint64_t insn_offset);

 protected:
  // The name of the file whose CFI we're reading.
  string filename_;

  // The name of the CFI section in that file.
  string section_;
};

}  // namespace google_breakpad

#endif  // UTIL_DEBUGINFO_DWARF2READER_H__
