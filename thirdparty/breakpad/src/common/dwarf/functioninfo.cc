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

// This is a client for the dwarf2reader to extract function and line
// information from the debug info.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <assert.h>
#include <limits.h>
#include <stdio.h>

#include <map>
#include <queue>
#include <vector>

#include "common/dwarf/functioninfo.h"
#include "common/dwarf/bytereader.h"
#include "common/scoped_ptr.h"
#include "common/using_std_string.h"

namespace google_breakpad {

CULineInfoHandler::CULineInfoHandler(std::vector<SourceFileInfo>* files,
                                     std::vector<string>* dirs,
                                     LineMap* linemap):linemap_(linemap),
                                                       files_(files),
                                                       dirs_(dirs) {
  // In dwarf4, the dirs and files are 1 indexed, and in dwarf5 they are zero
  // indexed. This is handled in the LineInfo reader, so empty files are not
  // needed here.
}

void CULineInfoHandler::DefineDir(const string& name, uint32_t dir_num) {
  // These should never come out of order, actually
  assert(dir_num == dirs_->size());
  dirs_->push_back(name);
}

void CULineInfoHandler::DefineFile(const string& name,
                                   int32 file_num, uint32_t dir_num,
                                   uint64_t mod_time, uint64_t length) {
  assert(dir_num >= 0);
  assert(dir_num < dirs_->size());

  // These should never come out of order, actually.
  if (file_num == (int32)files_->size() || file_num == -1) {
    string dir = dirs_->at(dir_num);

    SourceFileInfo s;
    s.lowpc = ULLONG_MAX;

    if (dir == "") {
      s.name = name;
    } else {
      s.name = dir + "/" + name;
    }

    files_->push_back(s);
  } else {
    fprintf(stderr, "error in DefineFile");
  }
}

void CULineInfoHandler::AddLine(uint64_t address, uint64_t length,
                                uint32_t file_num, uint32_t line_num,
                                uint32_t column_num) {
  if (file_num < files_->size()) {
    linemap_->insert(
        std::make_pair(address,
                       std::make_pair(files_->at(file_num).name.c_str(),
                                      line_num)));

    if (address < files_->at(file_num).lowpc) {
      files_->at(file_num).lowpc = address;
    }
  } else {
    fprintf(stderr, "error in AddLine");
  }
}

bool CUFunctionInfoHandler::StartCompilationUnit(uint64_t offset,
                                                 uint8_t address_size,
                                                 uint8_t offset_size,
                                                 uint64_t cu_length,
                                                 uint8_t dwarf_version) {
  current_compilation_unit_offset_ = offset;
  return true;
}


// For function info, we only care about subprograms and inlined
// subroutines. For line info, the DW_AT_stmt_list lives in the
// compile unit tag.

bool CUFunctionInfoHandler::StartDIE(uint64_t offset, enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_subprogram:
    case DW_TAG_inlined_subroutine: {
      current_function_info_ = new FunctionInfo;
      current_function_info_->lowpc = current_function_info_->highpc = 0;
      current_function_info_->name = "";
      current_function_info_->line = 0;
      current_function_info_->file = "";
      offset_to_funcinfo_->insert(std::make_pair(offset,
                                                 current_function_info_));
    };
      // FALLTHROUGH
    case DW_TAG_compile_unit:
      return true;
    default:
      return false;
  }
  return false;
}

// Only care about the name attribute for functions

void CUFunctionInfoHandler::ProcessAttributeString(uint64_t offset,
                                                   enum DwarfAttribute attr,
                                                   enum DwarfForm form,
                                                   const string& data) {
  if (current_function_info_) {
    if (attr == DW_AT_name)
      current_function_info_->name = data;
    else if (attr == DW_AT_MIPS_linkage_name)
      current_function_info_->mangled_name = data;
  }
}

void CUFunctionInfoHandler::ProcessAttributeUnsigned(uint64_t offset,
                                                     enum DwarfAttribute attr,
                                                     enum DwarfForm form,
                                                     uint64_t data) {
  if (attr == DW_AT_stmt_list) {
    SectionMap::const_iterator iter =
        GetSectionByName(sections_, ".debug_line");
    assert(iter != sections_.end());

    scoped_ptr<LineInfo> lireader(new LineInfo(iter->second.first + data,
                                               iter->second.second  - data,
                                               reader_, linehandler_));
    lireader->Start();
  } else if (current_function_info_) {
    switch (attr) {
      case DW_AT_low_pc:
        current_function_info_->lowpc = data;
        break;
      case DW_AT_high_pc:
        current_function_info_->highpc = data;
        break;
      case DW_AT_decl_line:
        current_function_info_->line = data;
        break;
      case DW_AT_decl_file:
        current_function_info_->file = files_->at(data).name;
        break;
      case DW_AT_ranges:
        current_function_info_->ranges = data;
        break;
      default:
        break;
    }
  }
}

void CUFunctionInfoHandler::ProcessAttributeReference(uint64_t offset,
                                                      enum DwarfAttribute attr,
                                                      enum DwarfForm form,
                                                      uint64_t data) {
  if (current_function_info_) {
    switch (attr) {
      case DW_AT_specification: {
        // Some functions have a "specification" attribute
        // which means they were defined elsewhere. The name
        // attribute is not repeated, and must be taken from
        // the specification DIE. Here we'll assume that
        // any DIE referenced in this manner will already have
        // been seen, but that's not really required by the spec.
        FunctionMap::iterator iter = offset_to_funcinfo_->find(data);
        if (iter != offset_to_funcinfo_->end()) {
          current_function_info_->name = iter->second->name;
          current_function_info_->mangled_name = iter->second->mangled_name;
        } else {
          // If you hit this, this code probably needs to be rewritten.
          fprintf(stderr,
                  "Error: DW_AT_specification was seen before the referenced "
                  "DIE! (Looking for DIE at offset %08llx, in DIE at "
                  "offset %08llx)\n", data, offset);
        }
        break;
      }
      default:
        break;
    }
  }
}

void CUFunctionInfoHandler::EndDIE(uint64_t offset) {
  if (current_function_info_ && current_function_info_->lowpc)
    address_to_funcinfo_->insert(std::make_pair(current_function_info_->lowpc,
                                                current_function_info_));
}

}  // namespace google_breakpad
