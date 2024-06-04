// -*- mode: c++ -*-

// Copyright (c) 2010 Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

// Add DWARF debugging information to a Breakpad symbol file. This
// file defines the DwarfCUToModule class, which accepts parsed DWARF
// data and populates a google_breakpad::Module with the results; the
// Module can then write its contents as a Breakpad symbol file.

#ifndef COMMON_LINUX_DWARF_CU_TO_MODULE_H__
#define COMMON_LINUX_DWARF_CU_TO_MODULE_H__

#include <stdint.h>

#include <string>

#include "common/language.h"
#include "common/module.h"
#include "common/dwarf/dwarf2diehandler.h"
#include "common/dwarf/dwarf2reader.h"
#include "common/scoped_ptr.h"
#include "common/using_std_string.h"

namespace google_breakpad {

// Populate a google_breakpad::Module with DWARF debugging information.
//
// An instance of this class can be provided as a handler to a
// DIEDispatcher, which can in turn be a handler for a
// CompilationUnit DWARF parser. The handler uses the results
// of parsing to populate a google_breakpad::Module with source file,
// function, and source line information.
class DwarfCUToModule: public RootDIEHandler {
  struct FilePrivate;
 public:
  // Information global to the DWARF-bearing file we are processing,
  // for use by DwarfCUToModule. Each DwarfCUToModule instance deals
  // with a single compilation unit within the file, but information
  // global to the whole file is held here. The client is responsible
  // for filling it in appropriately (except for the 'file_private'
  // field, which the constructor and destructor take care of), and
  // then providing it to the DwarfCUToModule instance for each
  // compilation unit we process in that file. Set HANDLE_INTER_CU_REFS
  // to true to handle debugging symbols with DW_FORM_ref_addr entries.
  class FileContext {
   public:
    FileContext(const string& filename,
                Module* module,
                bool handle_inter_cu_refs);
    ~FileContext();

    // Add CONTENTS of size LENGTH to the section map as NAME.
    void AddSectionToSectionMap(const string& name,
                                const uint8_t* contents,
                                uint64_t length);

    // Clear the section map for testing.
    void ClearSectionMapForTest();

    const SectionMap& section_map() const;

   private:
    friend class DwarfCUToModule;

    // Clears all the Specifications if HANDLE_INTER_CU_REFS_ is false.
    void ClearSpecifications();

    // Given an OFFSET and a CU that starts at COMPILATION_UNIT_START, returns
    // true if this is an inter-compilation unit reference that is not being
    // handled.
    bool IsUnhandledInterCUReference(uint64_t offset,
                                     uint64_t compilation_unit_start) const;

    // The name of this file, for use in error messages.
    const string filename_;

    // A map of this file's sections, used for finding other DWARF
    // sections that the .debug_info section may refer to.
    SectionMap section_map_;

    // The Module to which we're contributing definitions.
    Module* module_;

    // True if we are handling references between compilation units.
    const bool handle_inter_cu_refs_;

    // Inter-compilation unit data used internally by the handlers.
    scoped_ptr<FilePrivate> file_private_;
  };

  // An abstract base class for handlers that handle DWARF range lists for
  // DwarfCUToModule.
  class RangesHandler {
   public:
    RangesHandler() { }
    virtual ~RangesHandler() { }

    // Called when finishing a function to populate the function's ranges.
    // The entries are read according to the form and data.
    virtual bool ReadRanges(
        enum DwarfForm form, uint64_t data,
        RangeListReader::CURangesInfo* cu_info,
        vector<Module::Range>* ranges) = 0;
  };

  // An abstract base class for handlers that handle DWARF line data
  // for DwarfCUToModule. DwarfCUToModule could certainly just use
  // LineInfo itself directly, but decoupling things
  // this way makes unit testing a little easier.
  class LineToModuleHandler {
   public:
    LineToModuleHandler() { }
    virtual ~LineToModuleHandler() { }

    // Called at the beginning of a new compilation unit, prior to calling
    // ReadProgram(). compilation_dir will indicate the path that the
    // current compilation unit was compiled in, consistent with the
    // DW_AT_comp_dir DIE.
    virtual void StartCompilationUnit(const string& compilation_dir) = 0;

    // Populate MODULE and LINES with source file names and code/line
    // mappings, given a pointer to some DWARF line number data
    // PROGRAM, and an overestimate of its size. Add no zero-length
    // lines to LINES.
    virtual void ReadProgram(const uint8_t* program, uint64_t length,
                             const uint8_t* string_section,
                             uint64_t string_section_length,
                             const uint8_t* line_string_section,
                             uint64_t line_string_length,
                             Module* module, vector<Module::Line>* lines,
                             map<uint32_t, Module::File*>* files) = 0;
  };

  // The interface DwarfCUToModule uses to report warnings. The member
  // function definitions for this class write messages to stderr, but
  // you can override them if you'd like to detect or report these
  // conditions yourself.
  class WarningReporter {
   public:
    // Warn about problems in the DWARF file FILENAME, in the
    // compilation unit at OFFSET.
    WarningReporter(const string& filename, uint64_t cu_offset)
        : filename_(filename), cu_offset_(cu_offset), printed_cu_header_(false),
          printed_unpaired_header_(false),
          uncovered_warnings_enabled_(false) { }
    virtual ~WarningReporter() { }

    // Set the name of the compilation unit we're processing to NAME.
    virtual void SetCUName(const string& name) { cu_name_ = name; }

    // Accessor and setter for uncovered_warnings_enabled_.
    // UncoveredFunction and UncoveredLine only report a problem if that is
    // true. By default, these warnings are disabled, because those
    // conditions occur occasionally in healthy code.
    virtual bool uncovered_warnings_enabled() const {
      return uncovered_warnings_enabled_;
    }
    virtual void set_uncovered_warnings_enabled(bool value) {
      uncovered_warnings_enabled_ = value;
    }

    // A DW_AT_specification in the DIE at OFFSET refers to a DIE we
    // haven't processed yet, or that wasn't marked as a declaration,
    // at TARGET.
    virtual void UnknownSpecification(uint64_t offset, uint64_t target);

    // A DW_AT_abstract_origin in the DIE at OFFSET refers to a DIE we
    // haven't processed yet, or that wasn't marked as inline, at TARGET.
    virtual void UnknownAbstractOrigin(uint64_t offset, uint64_t target);

    // We were unable to find the DWARF section named SECTION_NAME.
    virtual void MissingSection(const string& section_name);

    // The CU's DW_AT_stmt_list offset OFFSET is bogus.
    virtual void BadLineInfoOffset(uint64_t offset);

    // FUNCTION includes code covered by no line number data.
    virtual void UncoveredFunction(const Module::Function& function);

    // Line number NUMBER in LINE_FILE, of length LENGTH, includes code
    // covered by no function.
    virtual void UncoveredLine(const Module::Line& line);

    // The DW_TAG_subprogram DIE at OFFSET has no name specified directly
    // in the DIE, nor via a DW_AT_specification or DW_AT_abstract_origin
    // link.
    virtual void UnnamedFunction(uint64_t offset);

    // __cxa_demangle() failed to demangle INPUT.
    virtual void DemangleError(const string& input);

    // The DW_FORM_ref_addr at OFFSET to TARGET was not handled because
    // FilePrivate did not retain the inter-CU specification data.
    virtual void UnhandledInterCUReference(uint64_t offset, uint64_t target);

    // The DW_AT_ranges at offset is malformed (truncated or outside of the
    // .debug_ranges section's bound).
    virtual void MalformedRangeList(uint64_t offset);

    // A DW_AT_ranges attribute was encountered but the no .debug_ranges
    // section was found.
    virtual void MissingRanges();

    uint64_t cu_offset() const {
      return cu_offset_;
    }

   protected:
    const string filename_;
    const uint64_t cu_offset_;
    string cu_name_;
    bool printed_cu_header_;
    bool printed_unpaired_header_;
    bool uncovered_warnings_enabled_;

   private:
    // Print a per-CU heading, once.
    void CUHeading();
    // Print an unpaired function/line heading, once.
    void UncoveredHeading();
  };

  // Create a DWARF debugging info handler for a compilation unit
  // within FILE_CONTEXT. This uses information received from the
  // CompilationUnit DWARF parser to populate
  // FILE_CONTEXT->module. Use LINE_READER to handle the compilation
  // unit's line number data. Use REPORTER to report problems with the
  // data we find.
  DwarfCUToModule(FileContext* file_context,
                  LineToModuleHandler* line_reader,
                  RangesHandler* ranges_handler,
                  WarningReporter* reporter,
                  bool handle_inline = false);
  ~DwarfCUToModule();

  void ProcessAttributeSigned(enum DwarfAttribute attr,
                              enum DwarfForm form,
                              int64_t data);
  void ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data);
  void ProcessAttributeString(enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const string& data);
  bool EndAttributes();
  DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag);

  // Assign all our source Lines to the Functions that cover their
  // addresses, and then add them to module_.
  void Finish();

  bool StartCompilationUnit(uint64_t offset, uint8_t address_size,
                            uint8_t offset_size, uint64_t cu_length,
                            uint8_t dwarf_version);
  bool StartRootDIE(uint64_t offset, enum DwarfTag tag);

 private:
  // Used internally by the handler. Full definitions are in
  // dwarf_cu_to_module.cc.
  struct CUContext;
  struct DIEContext;
  struct Specification;
  class GenericDIEHandler;
  class FuncHandler;
  class InlineHandler;
  class NamedScopeHandler;

  // A map from section offsets to specifications.
  typedef map<uint64_t, Specification> SpecificationByOffset;

  // Set this compilation unit's source language to LANGUAGE.
  void SetLanguage(DwarfLanguage language);

  // Read source line information at OFFSET in the .debug_line
  // section.  Record source files in module_, but record source lines
  // in lines_; we apportion them to functions in
  // AssignLinesToFunctions.
  void ReadSourceLines(uint64_t offset);

  // Assign the lines in lines_ to the individual line lists of the
  // functions in functions_.  (DWARF line information maps an entire
  // compilation unit at a time, and gives no indication of which
  // lines belong to which functions, beyond their addresses.)
  void AssignLinesToFunctions();

  void AssignFilesToInlines();

  // The only reason cu_context_ and child_context_ are pointers is
  // that we want to keep their definitions private to
  // dwarf_cu_to_module.cc, instead of listing them all here. They are
  // owned by this DwarfCUToModule: the constructor sets them, and the
  // destructor deletes them.

  // The handler to use to handle line number data.
  LineToModuleHandler* line_reader_;

  // This compilation unit's context.
  scoped_ptr<CUContext> cu_context_;

  // A context for our children.
  scoped_ptr<DIEContext> child_context_;

  // True if this compilation unit has source line information.
  bool has_source_line_info_;

  // The offset of this compilation unit's line number information in
  // the .debug_line section.
  uint64_t source_line_offset_;

  // The line numbers we have seen thus far.  We accumulate these here
  // during parsing.  Then, in Finish, we call AssignLinesToFunctions
  // to dole them out to the appropriate functions.
  vector<Module::Line> lines_;

  // The map from file index to File* in this CU.
  std::map<uint32_t, Module::File*> files_;
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_DWARF_CU_TO_MODULE_H__
