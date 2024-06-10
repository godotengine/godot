// Copyright 2006 Google LLC
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

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/windows/pdb_source_line_writer.h"

#include <windows.h>
#include <winnt.h>
#include <atlbase.h>
#include <dia2.h>
#include <diacreate.h>
#include <ImageHlp.h>
#include <stdio.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <utility>

#include "common/windows/dia_util.h"
#include "common/windows/guid_string.h"
#include "common/windows/pe_util.h"
#include "common/windows/string_utils-inl.h"

// This constant may be missing from DbgHelp.h.  See the documentation for
// IDiaSymbol::get_undecoratedNameEx.
#ifndef UNDNAME_NO_ECSU
#define UNDNAME_NO_ECSU 0x8000  // Suppresses enum/class/struct/union.
#endif  // UNDNAME_NO_ECSU

namespace google_breakpad {

namespace {

using std::set;
using std::unique_ptr;
using std::vector;

// The symbol (among possibly many) selected to represent an rva.
struct SelectedSymbol {
  SelectedSymbol(const CComPtr<IDiaSymbol>& symbol, bool is_public)
      : symbol(symbol), is_public(is_public), is_multiple(false) {}

  // The symbol to use for an rva.
  CComPtr<IDiaSymbol> symbol;
  // Whether this is a public or function symbol.
  bool is_public;
  // Whether the rva has multiple associated symbols. An rva will correspond to
  // multiple symbols in the case of linker identical symbol folding.
  bool is_multiple;
};

// Maps rva to the symbol to use for that address.
typedef std::map<DWORD, SelectedSymbol> SymbolMap;

// Record this in the map as the selected symbol for the rva if it satisfies the
// necessary conditions.
void MaybeRecordSymbol(DWORD rva,
                       const CComPtr<IDiaSymbol> symbol,
                       bool is_public,
                       SymbolMap* map) {
  SymbolMap::iterator loc = map->find(rva);
  if (loc == map->end()) {
    map->insert(std::make_pair(rva, SelectedSymbol(symbol, is_public)));
    return;
  }

  // Prefer function symbols to public symbols.
  if (is_public && !loc->second.is_public) {
    return;
  }

  loc->second.is_multiple = true;

  // Take the 'least' symbol by lexicographical order of the decorated name. We
  // use the decorated rather than undecorated name because computing the latter
  // is expensive.
  BSTR current_name, new_name;
  loc->second.symbol->get_name(&current_name);
  symbol->get_name(&new_name);
  if (wcscmp(new_name, current_name) < 0) {
    loc->second.symbol = symbol;
    loc->second.is_public = is_public;
  }
}



bool SymbolsMatch(IDiaSymbol* a, IDiaSymbol* b) {
  DWORD a_section, a_offset, b_section, b_offset;
  if (FAILED(a->get_addressSection(&a_section)) ||
      FAILED(a->get_addressOffset(&a_offset)) ||
      FAILED(b->get_addressSection(&b_section)) ||
      FAILED(b->get_addressOffset(&b_offset)))
    return false;
  return a_section == b_section && a_offset == b_offset;
}

bool CreateDiaDataSourceInstance(CComPtr<IDiaDataSource>& data_source) {
  if (SUCCEEDED(data_source.CoCreateInstance(CLSID_DiaSource))) {
    return true;
  }

  class DECLSPEC_UUID("B86AE24D-BF2F-4ac9-B5A2-34B14E4CE11D") DiaSource100;
  class DECLSPEC_UUID("761D3BCD-1304-41D5-94E8-EAC54E4AC172") DiaSource110;
  class DECLSPEC_UUID("3BFCEA48-620F-4B6B-81F7-B9AF75454C7D") DiaSource120;
  class DECLSPEC_UUID("E6756135-1E65-4D17-8576-610761398C3C") DiaSource140;

  // If the CoCreateInstance call above failed, msdia*.dll is not registered.
  // We can try loading the DLL corresponding to the #included DIA SDK, but
  // the DIA headers don't provide a version. Lets try to figure out which DIA
  // version we're compiling against by comparing CLSIDs.
  const wchar_t* msdia_dll = nullptr;
  if (CLSID_DiaSource == _uuidof(DiaSource100)) {
    msdia_dll = L"msdia100.dll";
  } else if (CLSID_DiaSource == _uuidof(DiaSource110)) {
    msdia_dll = L"msdia110.dll";
  } else if (CLSID_DiaSource == _uuidof(DiaSource120)) {
    msdia_dll = L"msdia120.dll";
  } else if (CLSID_DiaSource == _uuidof(DiaSource140)) {
    msdia_dll = L"msdia140.dll";
  }

  if (msdia_dll &&
      SUCCEEDED(NoRegCoCreate(msdia_dll, CLSID_DiaSource, IID_IDiaDataSource,
                              reinterpret_cast<void**>(&data_source)))) {
    return true;
  }

  return false;
}

const DWORD kUndecorateOptions = UNDNAME_NO_MS_KEYWORDS |
                                 UNDNAME_NO_FUNCTION_RETURNS |
                                 UNDNAME_NO_ALLOCATION_MODEL |
                                 UNDNAME_NO_ALLOCATION_LANGUAGE |
                                 UNDNAME_NO_THISTYPE |
                                 UNDNAME_NO_ACCESS_SPECIFIERS |
                                 UNDNAME_NO_THROW_SIGNATURES |
                                 UNDNAME_NO_MEMBER_TYPE |
                                 UNDNAME_NO_RETURN_UDT_MODEL |
                                 UNDNAME_NO_ECSU;

#define arraysize(f) (sizeof(f) / sizeof(*f))

void StripLlvmSuffixAndUndecorate(BSTR* name) {
  // LLVM sometimes puts a suffix on symbols to give them a globally unique
  // name. The suffix is either some string preceded by a period (like in the
  // Itanium ABI; also on Windows this is safe since periods are otherwise
  // never part of mangled names), or a dollar sign followed by a 32-char hex
  // string (this should go away in future LLVM versions). Strip such suffixes
  // and try demangling again.
  //
  //
  // Example symbol names with such suffixes:
  //
  //   ?foo@@YAXXZ$5520c83448162c04f2b239db4b5a2c61
  //   ?foo@@YAXXZ.llvm.13040715209719948753

  if (**name != L'?')
    return;  // The name is already demangled.

  for (size_t i = 0, len = wcslen(*name); i < len; i++) {
    wchar_t c = (*name)[i];

    if (c == L'.' || (c == L'$' && len - i == 32 + 1)) {
      (*name)[i] = L'\0';
      wchar_t undecorated[1024];
      DWORD res = UnDecorateSymbolNameW(*name, undecorated,
                                        arraysize(undecorated),
                                        kUndecorateOptions);
      if (res == 0 || undecorated[0] == L'?') {
        // Demangling failed; restore the symbol name and return.
        (*name)[i] = c;
        return;
      }

      SysFreeString(*name);
      *name = SysAllocString(undecorated);
      return;
    }
  }
}

// Prints the error message related to the error code as seen in
// Microsoft's MSVS documentation for loadDataFromPdb and loadDataForExe.
void PrintOpenError(HRESULT hr, const char* fn_name, const wchar_t* file) {
  switch (hr) {
    case E_PDB_NOT_FOUND:
      fprintf(stderr, "%s: Failed to open %ws, or the file has an "
              "invalid format.\n", fn_name, file);
      break;
    case E_PDB_FORMAT:
      fprintf(stderr, "%s: Attempted to access %ws with an obsolete "
              "format.\n", fn_name, file);
      break;
    case E_PDB_INVALID_SIG:
      fprintf(stderr, "%s: Signature does not match for %ws.\n", fn_name,
              file);
      break;
    case E_PDB_INVALID_AGE:
      fprintf(stderr, "%s: Age does not match for %ws.\n", fn_name, file);
      break;
    case E_INVALIDARG:
      fprintf(stderr, "%s: Invalid parameter for %ws.\n", fn_name, file);
      break;
    case E_UNEXPECTED:
      fprintf(stderr, "%s: Data source has already been prepared for %ws.\n",
              fn_name, file);
      break;
    default:
      fprintf(stderr, "%s: Unexpected error 0x%lx, file: %ws.\n",
              fn_name, hr, file);
      break;
  }
}

}  // namespace

PDBSourceLineWriter::Inline::Inline(int inline_nest_level)
    : inline_nest_level_(inline_nest_level) {}

void PDBSourceLineWriter::Inline::SetOriginId(int origin_id) {
  origin_id_ = origin_id;
}

void PDBSourceLineWriter::Inline::ExtendRanges(const Line& line) {
  if (ranges_.empty()) {
    ranges_[line.rva] = line.length;
    return;
  }
  auto iter = ranges_.lower_bound(line.rva);
  // There is no overlap if this function is called with inlinee lines from
  // the same callsite.
  if (iter == ranges_.begin()) {
    return;
  }
  if (line.rva + line.length == iter->first) {
    // If they are connected, merge their ranges into one.
    DWORD length = line.length + iter->second;
    ranges_.erase(iter);
    ranges_[line.rva] = length;
  } else {
    --iter;
    if (iter->first + iter->second == line.rva) {
      ranges_[iter->first] = iter->second + line.length;
    } else {
      ranges_[line.rva] = line.length;
    }
  }
}

void PDBSourceLineWriter::Inline::SetCallSiteLine(DWORD call_site_line) {
  call_site_line_ = call_site_line;
}

void PDBSourceLineWriter::Inline::SetCallSiteFileId(DWORD call_site_file_id) {
  call_site_file_id_ = call_site_file_id;
}

void PDBSourceLineWriter::Inline::SetChildInlines(
    vector<unique_ptr<Inline>> child_inlines) {
  child_inlines_ = std::move(child_inlines);
}

void PDBSourceLineWriter::Inline::Print(FILE* output) const {
  // Ignore INLINE record that doesn't have any range.
  if (ranges_.empty())
    return;
  fprintf(output, "INLINE %d %lu %lu %d", inline_nest_level_, call_site_line_,
          call_site_file_id_, origin_id_);
  for (const auto& r : ranges_) {
    fprintf(output, " %lx %lx", r.first, r.second);
  }
  fprintf(output, "\n");
  for (const unique_ptr<Inline>& in : child_inlines_) {
    in->Print(output);
  }
}

const PDBSourceLineWriter::Line* PDBSourceLineWriter::Lines::GetLine(
    DWORD rva) const {
  auto iter = line_map_.find(rva);
  if (iter == line_map_.end()) {
    // If not found exact rva, check if it's within any range.
    iter = line_map_.lower_bound(rva);
    if (iter == line_map_.begin())
      return nullptr;
    --iter;
    auto l = iter->second;
    // This happens when there is no top level lines cover this rva (e.g. empty
    // lines found for the function). Then we don't know the call site line
    // number for this inlined function.
    if (rva >= l.rva + l.length)
      return nullptr;
  }
  return &iter->second;
}

DWORD PDBSourceLineWriter::Lines::GetLineNum(DWORD rva) const {
  const Line* line = GetLine(rva);
  return line ? line->line_num : 0;
}

DWORD PDBSourceLineWriter::Lines::GetFileId(DWORD rva) const {
  const Line* line = GetLine(rva);
  return line ? line->file_id : 0;
}

void PDBSourceLineWriter::Lines::AddLine(const Line& line) {
  if (line_map_.empty()) {
    line_map_[line.rva] = line;
    return;
  }

  // Given an existing line in line_map_, remove it from line_map_ if it
  // overlaps with the line and add a new line for the non-overlap range. Return
  // true if there is an overlap.
  auto intercept = [&](Line old_line) {
    DWORD end = old_line.rva + old_line.length;
    // No overlap.
    if (old_line.rva >= line.rva + line.length || line.rva >= end)
      return false;
    // old_line is within the line.
    if (old_line.rva >= line.rva && end <= line.rva + line.length) {
      line_map_.erase(old_line.rva);
      return true;
    }
    // Then there is a overlap.
    if (old_line.rva < line.rva) {
      old_line.length -= end - line.rva;
      if (end > line.rva + line.length) {
        Line new_line = old_line;
        new_line.rva = line.rva + line.length;
        new_line.length = end - new_line.rva;
        line_map_[new_line.rva] = new_line;
      }
    } else {
      line_map_.erase(old_line.rva);
      old_line.length -= line.rva + line.length - old_line.rva;
      old_line.rva = line.rva + line.length;
    }
    line_map_[old_line.rva] = old_line;
    return true;
  };

  bool is_intercept;
  // Use a loop in cases that there are multiple lines within the given line.
  do {
    auto iter = line_map_.lower_bound(line.rva);
    if (iter == line_map_.end()) {
      if (!line_map_.empty()) {
        --iter;
        intercept(iter->second);
      }
      break;
    }
    is_intercept = false;
    if (iter != line_map_.begin()) {
      // Check if the given line overlaps a line with smaller in the map.
      auto prev = line_map_.lower_bound(line.rva);
      --prev;
      is_intercept = intercept(prev->second);
    }
    // Check if the given line overlaps a line with greater or equal rva in the
    // map. Using operator |= here since it's possible that there are multiple
    // lines with greater rva in the map overlap with the given line.
    is_intercept |= intercept(iter->second);
  } while (is_intercept);
  line_map_[line.rva] = line;
}

PDBSourceLineWriter::PDBSourceLineWriter(bool handle_inline)
    : output_(NULL), handle_inline_(handle_inline) {}

PDBSourceLineWriter::~PDBSourceLineWriter() {
  Close();
}

bool PDBSourceLineWriter::SetCodeFile(const wstring& exe_file) {
  if (code_file_.empty()) {
    code_file_ = exe_file;
    return true;
  }
  // Setting a different code file path is an error.  It is success only if the
  // file paths are the same.
  return exe_file == code_file_;
}

bool PDBSourceLineWriter::Open(const wstring& file, FileFormat format) {
  Close();
  code_file_.clear();

  if (FAILED(CoInitialize(NULL))) {
    fprintf(stderr, "CoInitialize failed\n");
    return false;
  }

  CComPtr<IDiaDataSource> data_source;
  if (!CreateDiaDataSourceInstance(data_source)) {
    const int kGuidSize = 64;
    wchar_t classid[kGuidSize] = {0};
    StringFromGUID2(CLSID_DiaSource, classid, kGuidSize);
    fprintf(stderr, "CoCreateInstance CLSID_DiaSource %S failed "
            "(msdia*.dll unregistered?)\n", classid);
    return false;
  }

  HRESULT from_pdb_result;
  HRESULT for_exe_result;
  const wchar_t* file_name = file.c_str();
  switch (format) {
    case PDB_FILE:
      from_pdb_result = data_source->loadDataFromPdb(file_name);
      if (FAILED(from_pdb_result)) {
        PrintOpenError(from_pdb_result, "loadDataFromPdb", file_name);
        return false;
      }
      break;
    case EXE_FILE:
      for_exe_result = data_source->loadDataForExe(file_name, NULL, NULL);
      if (FAILED(for_exe_result)) {
        PrintOpenError(for_exe_result, "loadDataForExe", file_name);
        return false;
      }
      code_file_ = file;
      break;
    case ANY_FILE:
      from_pdb_result = data_source->loadDataFromPdb(file_name);
      if (FAILED(from_pdb_result)) {
        for_exe_result = data_source->loadDataForExe(file_name, NULL, NULL);
        if (FAILED(for_exe_result)) {
          PrintOpenError(from_pdb_result, "loadDataFromPdb", file_name);
          PrintOpenError(for_exe_result, "loadDataForExe", file_name);
          return false;
        }
        code_file_ = file;
      }
      break;
    default:
      fprintf(stderr, "Unknown file format\n");
      return false;
  }

  if (FAILED(data_source->openSession(&session_))) {
    fprintf(stderr, "openSession failed\n");
  }

  return true;
}

bool PDBSourceLineWriter::GetLine(IDiaLineNumber* dia_line, Line* line) const {
  if (FAILED(dia_line->get_relativeVirtualAddress(&line->rva))) {
    fprintf(stderr, "failed to get line rva\n");
    return false;
  }

  if (FAILED(dia_line->get_length(&line->length))) {
    fprintf(stderr, "failed to get line code length\n");
    return false;
  }

  DWORD dia_source_id;
  if (FAILED(dia_line->get_sourceFileId(&dia_source_id))) {
    fprintf(stderr, "failed to get line source file id\n");
    return false;
  }
  // duplicate file names are coalesced to share one ID
  line->file_id = GetRealFileID(dia_source_id);

  if (FAILED(dia_line->get_lineNumber(&line->line_num))) {
    fprintf(stderr, "failed to get line number\n");
    return false;
  }
  return true;
}

bool PDBSourceLineWriter::GetLines(IDiaEnumLineNumbers* lines,
                                   Lines* line_list) const {
  CComPtr<IDiaLineNumber> line;
  ULONG count;

  while (SUCCEEDED(lines->Next(1, &line, &count)) && count == 1) {
    Line l;
    if (!GetLine(line, &l))
      return false;
    // Silently ignore zero-length lines.
    if (l.length != 0)
      line_list->AddLine(l);
    line.Release();
  }
  return true;
}

void PDBSourceLineWriter::PrintLines(const Lines& lines) const {
  // The line number format is:
  // <rva> <line number> <source file id>
  for (const auto& kv : lines.GetLineMap()) {
    const Line& l = kv.second;
    AddressRangeVector ranges;
    MapAddressRange(image_map_, AddressRange(l.rva, l.length), &ranges);
    for (auto& range : ranges) {
      fprintf(output_, "%lx %lx %lu %lu\n", range.rva, range.length, l.line_num,
              l.file_id);
    }
  }
}

bool PDBSourceLineWriter::PrintFunction(IDiaSymbol* function,
                                        IDiaSymbol* block,
                                        bool has_multiple_symbols) {
  // The function format is:
  // FUNC <address> <length> <param_stack_size> <function>
  DWORD rva;
  if (FAILED(block->get_relativeVirtualAddress(&rva))) {
    fprintf(stderr, "couldn't get rva\n");
    return false;
  }

  ULONGLONG length;
  if (FAILED(block->get_length(&length))) {
    fprintf(stderr, "failed to get function length\n");
    return false;
  }

  if (length == 0) {
    // Silently ignore zero-length functions, which can infrequently pop up.
    return true;
  }

  CComBSTR name;
  int stack_param_size;
  if (!GetSymbolFunctionName(function, &name, &stack_param_size)) {
    return false;
  }

  // If the decorated name didn't give the parameter size, try to
  // calculate it.
  if (stack_param_size < 0) {
    stack_param_size = GetFunctionStackParamSize(function);
  }

  AddressRangeVector ranges;
  MapAddressRange(image_map_, AddressRange(rva, static_cast<DWORD>(length)),
                  &ranges);
  for (size_t i = 0; i < ranges.size(); ++i) {
    const char* optional_multiple_field = has_multiple_symbols ? "m " : "";
    fprintf(output_, "FUNC %s%lx %lx %x %ws\n", optional_multiple_field,
            ranges[i].rva, ranges[i].length, stack_param_size, name.m_str);
  }

  CComPtr<IDiaEnumLineNumbers> lines;
  if (FAILED(session_->findLinesByRVA(rva, DWORD(length), &lines))) {
    return false;
  }

  // Get top level lines first, which later may be split into multiple smaller
  // lines if any inline exists in their ranges if we want to handle inline.
  Lines line_list;
  if (!GetLines(lines, &line_list)) {
    return false;
  }
  if (handle_inline_) {
    vector<unique_ptr<Inline>> inlines;
    if (!GetInlines(block, &line_list, 0, &inlines)) {
      return false;
    }
    PrintInlines(inlines);
  }
  PrintLines(line_list);
  return true;
}

bool PDBSourceLineWriter::PrintSourceFiles() {
  CComPtr<IDiaSymbol> global;
  if (FAILED(session_->get_globalScope(&global))) {
    fprintf(stderr, "get_globalScope failed\n");
    return false;
  }

  CComPtr<IDiaEnumSymbols> compilands;
  if (FAILED(global->findChildren(SymTagCompiland, NULL,
                                  nsNone, &compilands))) {
    fprintf(stderr, "findChildren failed\n");
    return false;
  }

  // Print a dummy file with id equals 0 to represent unknown file, because
  // inline records might have unknown call site.
  fwprintf(output_, L"FILE %d unknown file\n", 0);

  CComPtr<IDiaSymbol> compiland;
  ULONG count;
  while (SUCCEEDED(compilands->Next(1, &compiland, &count)) && count == 1) {
    CComPtr<IDiaEnumSourceFiles> source_files;
    if (FAILED(session_->findFile(compiland, NULL, nsNone, &source_files))) {
      return false;
    }
    CComPtr<IDiaSourceFile> file;
    while (SUCCEEDED(source_files->Next(1, &file, &count)) && count == 1) {
      DWORD file_id;
      if (FAILED(file->get_uniqueId(&file_id))) {
        return false;
      }

      CComBSTR file_name;
      if (FAILED(file->get_fileName(&file_name))) {
        return false;
      }

      wstring file_name_string(file_name);
      if (!FileIDIsCached(file_name_string)) {
        // this is a new file name, cache it and output a FILE line.
        CacheFileID(file_name_string, file_id);
        fwprintf(output_, L"FILE %d %ws\n", file_id, file_name_string.c_str());
      } else {
        // this file name has already been seen, just save this
        // ID for later lookup.
        StoreDuplicateFileID(file_name_string, file_id);
      }
      file.Release();
    }
    compiland.Release();
  }
  return true;
}

bool PDBSourceLineWriter::PrintFunctions() {
  ULONG count = 0;
  DWORD rva = 0;
  CComPtr<IDiaSymbol> global;
  HRESULT hr;

  if (FAILED(session_->get_globalScope(&global))) {
    fprintf(stderr, "get_globalScope failed\n");
    return false;
  }

  CComPtr<IDiaEnumSymbols> symbols = NULL;

  // Find all function symbols first.
  SymbolMap rva_symbol;
  hr = global->findChildren(SymTagFunction, NULL, nsNone, &symbols);

  if (SUCCEEDED(hr)) {
    CComPtr<IDiaSymbol> symbol = NULL;

    while (SUCCEEDED(symbols->Next(1, &symbol, &count)) && count == 1) {
      if (SUCCEEDED(symbol->get_relativeVirtualAddress(&rva))) {
        // Potentially record this as the canonical symbol for this rva.
        MaybeRecordSymbol(rva, symbol, false, &rva_symbol);
      } else {
        fprintf(stderr, "get_relativeVirtualAddress failed on the symbol\n");
        return false;
      }

      symbol.Release();
    }

    symbols.Release();
  }

  // Find all public symbols and record public symbols that are not also private
  // symbols.
  hr = global->findChildren(SymTagPublicSymbol, NULL, nsNone, &symbols);

  if (SUCCEEDED(hr)) {
    CComPtr<IDiaSymbol> symbol = NULL;

    while (SUCCEEDED(symbols->Next(1, &symbol, &count)) && count == 1) {
      if (SUCCEEDED(symbol->get_relativeVirtualAddress(&rva))) {
        // Potentially record this as the canonical symbol for this rva.
        MaybeRecordSymbol(rva, symbol, true, &rva_symbol);
      } else {
        fprintf(stderr, "get_relativeVirtualAddress failed on the symbol\n");
        return false;
      }

      symbol.Release();
    }

    symbols.Release();
  }

  // For each rva, dump the selected symbol at the address.
  SymbolMap::iterator it;
  for (it = rva_symbol.begin(); it != rva_symbol.end(); ++it) {
    CComPtr<IDiaSymbol> symbol = it->second.symbol;
    // Only print public symbols if there is no function symbol for the address.
    if (!it->second.is_public) {
      if (!PrintFunction(symbol, symbol, it->second.is_multiple))
        return false;
    } else {
      if (!PrintCodePublicSymbol(symbol, it->second.is_multiple))
        return false;
    }
  }

  // When building with PGO, the compiler can split functions into
  // "hot" and "cold" blocks, and move the "cold" blocks out to separate
  // pages, so the function can be noncontiguous. To find these blocks,
  // we have to iterate over all the compilands, and then find blocks
  // that are children of them. We can then find the lexical parents
  // of those blocks and print out an extra FUNC line for blocks
  // that are not contained in their parent functions.
  CComPtr<IDiaEnumSymbols> compilands;
  if (FAILED(global->findChildren(SymTagCompiland, NULL,
                                  nsNone, &compilands))) {
    fprintf(stderr, "findChildren failed on the global\n");
    return false;
  }

  CComPtr<IDiaSymbol> compiland;
  while (SUCCEEDED(compilands->Next(1, &compiland, &count)) && count == 1) {
    CComPtr<IDiaEnumSymbols> blocks;
    if (FAILED(compiland->findChildren(SymTagBlock, NULL,
                                       nsNone, &blocks))) {
      fprintf(stderr, "findChildren failed on a compiland\n");
      return false;
    }

    CComPtr<IDiaSymbol> block;
    while (SUCCEEDED(blocks->Next(1, &block, &count)) && count == 1) {
      // find this block's lexical parent function
      CComPtr<IDiaSymbol> parent;
      DWORD tag;
      if (SUCCEEDED(block->get_lexicalParent(&parent)) &&
          SUCCEEDED(parent->get_symTag(&tag)) &&
          tag == SymTagFunction) {
        // now get the block's offset and the function's offset and size,
        // and determine if the block is outside of the function
        DWORD func_rva, block_rva;
        ULONGLONG func_length;
        if (SUCCEEDED(block->get_relativeVirtualAddress(&block_rva)) &&
            SUCCEEDED(parent->get_relativeVirtualAddress(&func_rva)) &&
            SUCCEEDED(parent->get_length(&func_length))) {
          if (block_rva < func_rva || block_rva > (func_rva + func_length)) {
            if (!PrintFunction(parent, block, false)) {
              return false;
            }
          }
        }
      }
      parent.Release();
      block.Release();
    }
    blocks.Release();
    compiland.Release();
  }

  global.Release();
  return true;
}

void PDBSourceLineWriter::PrintInlineOrigins() const {
  struct OriginCompare {
    bool operator()(const InlineOrigin lhs, const InlineOrigin rhs) const {
      return lhs.id < rhs.id;
    }
  };
  set<InlineOrigin, OriginCompare> origins;
  // Sort by origin id.
  for (auto const& origin : inline_origins_)
    origins.insert(origin.second);
  for (auto o : origins) {
    fprintf(output_, "INLINE_ORIGIN %d %ls\n", o.id, o.name.c_str());
  }
}

bool PDBSourceLineWriter::GetInlines(IDiaSymbol* block,
                                     Lines* line_list,
                                     int inline_nest_level,
                                     vector<unique_ptr<Inline>>* inlines) {
  CComPtr<IDiaEnumSymbols> inline_callsites;
  if (FAILED(block->findChildrenEx(SymTagInlineSite, nullptr, nsNone,
                                   &inline_callsites))) {
    return false;
  }
  ULONG count;
  CComPtr<IDiaSymbol> callsite;
  while (SUCCEEDED(inline_callsites->Next(1, &callsite, &count)) &&
         count == 1) {
    unique_ptr<Inline> new_inline(new Inline(inline_nest_level));
    CComPtr<IDiaEnumLineNumbers> lines;
    // All inlinee lines have the same file id.
    DWORD file_id = 0;
    DWORD call_site_line = 0;
    if (FAILED(session_->findInlineeLines(callsite, &lines))) {
      return false;
    }
    CComPtr<IDiaLineNumber> dia_line;
    while (SUCCEEDED(lines->Next(1, &dia_line, &count)) && count == 1) {
      Line line;
      if (!GetLine(dia_line, &line)) {
        return false;
      }
      // Silently ignore zero-length lines.
      if (line.length != 0) {
        // Use the first line num and file id at rva as this inline's call site
        // line number, because after adding lines it may be changed to inner
        // line number and inner file id.
        if (call_site_line == 0)
          call_site_line = line_list->GetLineNum(line.rva);
        if (file_id == 0)
          file_id = line_list->GetFileId(line.rva);
        line_list->AddLine(line);
        new_inline->ExtendRanges(line);
      }
      dia_line.Release();
    }
    BSTR name;
    callsite->get_name(&name);
    if (SysStringLen(name) == 0) {
      name = SysAllocString(L"<name omitted>");
    }
    auto iter = inline_origins_.find(name);
    if (iter == inline_origins_.end()) {
      InlineOrigin origin;
      origin.id = inline_origins_.size();
      origin.name = name;
      inline_origins_[name] = origin;
    }
    new_inline->SetOriginId(inline_origins_[name].id);
    new_inline->SetCallSiteLine(call_site_line);
    new_inline->SetCallSiteFileId(file_id);
    // Go to next level.
    vector<unique_ptr<Inline>> child_inlines;
    if (!GetInlines(callsite, line_list, inline_nest_level + 1,
                    &child_inlines)) {
      return false;
    }
    new_inline->SetChildInlines(std::move(child_inlines));
    inlines->push_back(std::move(new_inline));
    callsite.Release();
  }
  return true;
}

void PDBSourceLineWriter::PrintInlines(
    const vector<unique_ptr<Inline>>& inlines) const {
  for (const unique_ptr<Inline>& in : inlines) {
    in->Print(output_);
  }
}

#undef max

bool PDBSourceLineWriter::PrintFrameDataUsingPDB() {
  // It would be nice if it were possible to output frame data alongside the
  // associated function, as is done with line numbers, but the DIA API
  // doesn't make it possible to get the frame data in that way.

  CComPtr<IDiaEnumFrameData> frame_data_enum;
  if (!FindTable(session_, &frame_data_enum))
    return false;

  DWORD last_type = std::numeric_limits<DWORD>::max();
  DWORD last_rva = std::numeric_limits<DWORD>::max();
  DWORD last_code_size = 0;
  DWORD last_prolog_size = std::numeric_limits<DWORD>::max();

  CComPtr<IDiaFrameData> frame_data;
  ULONG count = 0;
  while (SUCCEEDED(frame_data_enum->Next(1, &frame_data, &count)) &&
         count == 1) {
    DWORD type;
    if (FAILED(frame_data->get_type(&type)))
      return false;

    DWORD rva;
    if (FAILED(frame_data->get_relativeVirtualAddress(&rva)))
      return false;

    DWORD code_size;
    if (FAILED(frame_data->get_lengthBlock(&code_size)))
      return false;

    DWORD prolog_size;
    if (FAILED(frame_data->get_lengthProlog(&prolog_size)))
      return false;

    // parameter_size is the size of parameters passed on the stack.  If any
    // parameters are not passed on the stack (such as in registers), their
    // sizes will not be included in parameter_size.
    DWORD parameter_size;
    if (FAILED(frame_data->get_lengthParams(&parameter_size)))
      return false;

    DWORD saved_register_size;
    if (FAILED(frame_data->get_lengthSavedRegisters(&saved_register_size)))
      return false;

    DWORD local_size;
    if (FAILED(frame_data->get_lengthLocals(&local_size)))
      return false;

    // get_maxStack can return S_FALSE, just use 0 in that case.
    DWORD max_stack_size = 0;
    if (FAILED(frame_data->get_maxStack(&max_stack_size)))
      return false;

    // get_programString can return S_FALSE, indicating that there is no
    // program string.  In that case, check whether %ebp is used.
    HRESULT program_string_result;
    CComBSTR program_string;
    if (FAILED(program_string_result = frame_data->get_program(
        &program_string))) {
      return false;
    }

    // get_allocatesBasePointer can return S_FALSE, treat that as though
    // %ebp is not used.
    BOOL allocates_base_pointer = FALSE;
    if (program_string_result != S_OK) {
      if (FAILED(frame_data->get_allocatesBasePointer(
          &allocates_base_pointer))) {
        return false;
      }
    }

    // Only print out a line if type, rva, code_size, or prolog_size have
    // changed from the last line.  It is surprisingly common (especially in
    // system library PDBs) for DIA to return a series of identical
    // IDiaFrameData objects.  For kernel32.pdb from Windows XP SP2 on x86,
    // this check reduces the size of the dumped symbol file by a third.
    if (type != last_type || rva != last_rva || code_size != last_code_size ||
        prolog_size != last_prolog_size) {
      // The prolog and the code portions of the frame have to be treated
      // independently as they may have independently changed in size, or may
      // even have been split.
      // NOTE: If epilog size is ever non-zero, we have to do something
      //     similar with it.

      // Figure out where the prolog bytes have landed.
      AddressRangeVector prolog_ranges;
      if (prolog_size > 0) {
        MapAddressRange(image_map_, AddressRange(rva, prolog_size),
                        &prolog_ranges);
      }

      // And figure out where the code bytes have landed.
      AddressRangeVector code_ranges;
      MapAddressRange(image_map_,
                      AddressRange(rva + prolog_size,
                                   code_size - prolog_size),
                      &code_ranges);

      struct FrameInfo {
        DWORD rva;
        DWORD code_size;
        DWORD prolog_size;
      };
      std::vector<FrameInfo> frame_infos;

      // Special case: The prolog and the code bytes remain contiguous. This is
      // only done for compactness of the symbol file, and we could actually
      // be outputting independent frame info for the prolog and code portions.
      if (prolog_ranges.size() == 1 && code_ranges.size() == 1 &&
          prolog_ranges[0].end() == code_ranges[0].rva) {
        FrameInfo fi = { prolog_ranges[0].rva,
                         prolog_ranges[0].length + code_ranges[0].length,
                         prolog_ranges[0].length };
        frame_infos.push_back(fi);
      } else {
        // Otherwise we output the prolog and code frame info independently.
        for (size_t i = 0; i < prolog_ranges.size(); ++i) {
          FrameInfo fi = { prolog_ranges[i].rva,
                           prolog_ranges[i].length,
                           prolog_ranges[i].length };
          frame_infos.push_back(fi);
        }
        for (size_t i = 0; i < code_ranges.size(); ++i) {
          FrameInfo fi = { code_ranges[i].rva, code_ranges[i].length, 0 };
          frame_infos.push_back(fi);
        }
      }

      for (size_t i = 0; i < frame_infos.size(); ++i) {
        const FrameInfo& fi(frame_infos[i]);
        fprintf(output_, "STACK WIN %lx %lx %lx %lx %x %lx %lx %lx %lx %d ",
                type, fi.rva, fi.code_size, fi.prolog_size,
                0 /* epilog_size */, parameter_size, saved_register_size,
                local_size, max_stack_size, program_string_result == S_OK);
        if (program_string_result == S_OK) {
          fprintf(output_, "%ws\n", program_string.m_str);
        } else {
          fprintf(output_, "%d\n", allocates_base_pointer);
        }
      }

      last_type = type;
      last_rva = rva;
      last_code_size = code_size;
      last_prolog_size = prolog_size;
    }

    frame_data.Release();
  }

  return true;
}

bool PDBSourceLineWriter::PrintFrameDataUsingEXE() {
  if (code_file_.empty() && !FindPEFile()) {
    fprintf(stderr, "Couldn't locate EXE or DLL file.\n");
    return false;
  }

  return PrintPEFrameData(code_file_, output_);
}

bool PDBSourceLineWriter::PrintFrameData() {
  PDBModuleInfo info;
  if (GetModuleInfo(&info) && info.cpu == L"x86_64") {
    return PrintFrameDataUsingEXE();
  }
  return PrintFrameDataUsingPDB();
}

bool PDBSourceLineWriter::PrintCodePublicSymbol(IDiaSymbol* symbol,
                                                bool has_multiple_symbols) {
  BOOL is_code;
  if (FAILED(symbol->get_code(&is_code))) {
    return false;
  }
  if (!is_code) {
    return true;
  }

  DWORD rva;
  if (FAILED(symbol->get_relativeVirtualAddress(&rva))) {
    return false;
  }

  CComBSTR name;
  int stack_param_size;
  if (!GetSymbolFunctionName(symbol, &name, &stack_param_size)) {
    return false;
  }

  AddressRangeVector ranges;
  MapAddressRange(image_map_, AddressRange(rva, 1), &ranges);
  for (size_t i = 0; i < ranges.size(); ++i) {
    const char* optional_multiple_field = has_multiple_symbols ? "m " : "";
    fprintf(output_, "PUBLIC %s%lx %x %ws\n", optional_multiple_field,
            ranges[i].rva, stack_param_size > 0 ? stack_param_size : 0,
            name.m_str);
  }

  // Now walk the function in the original untranslated space, asking DIA
  // what function is at that location, stepping through OMAP blocks. If
  // we're still in the same function, emit another entry, because the
  // symbol could have been split into multiple pieces. If we've gotten to
  // another symbol in the original address space, then we're done for
  // this symbol. See https://crbug.com/678874.
  for (;;) {
    // This steps to the next block in the original image. Simply doing
    // rva++ would also be correct, but would emit tons of unnecessary
    // entries.
    rva = image_map_.subsequent_rva_block[rva];
    if (rva == 0)
      break;

    CComPtr<IDiaSymbol> next_sym = NULL;
    LONG displacement;
    if (FAILED(session_->findSymbolByRVAEx(rva, SymTagPublicSymbol, &next_sym,
                                           &displacement))) {
      break;
    }

    if (!SymbolsMatch(symbol, next_sym))
      break;

    AddressRangeVector next_ranges;
    MapAddressRange(image_map_, AddressRange(rva, 1), &next_ranges);
    for (size_t i = 0; i < next_ranges.size(); ++i) {
      fprintf(output_, "PUBLIC %lx %x %ws\n", next_ranges[i].rva,
              stack_param_size > 0 ? stack_param_size : 0, name.m_str);
    }
  }

  return true;
}

bool PDBSourceLineWriter::PrintPDBInfo() {
  PDBModuleInfo info;
  if (!GetModuleInfo(&info)) {
    return false;
  }

  // Hard-code "windows" for the OS because that's the only thing that makes
  // sense for PDB files.  (This might not be strictly correct for Windows CE
  // support, but we don't care about that at the moment.)
  fprintf(output_, "MODULE windows %ws %ws %ws\n",
          info.cpu.c_str(), info.debug_identifier.c_str(),
          info.debug_file.c_str());

  return true;
}

bool PDBSourceLineWriter::PrintPEInfo() {
  PEModuleInfo info;
  if (!GetPEInfo(&info)) {
    return false;
  }

  fprintf(output_, "INFO CODE_ID %ws %ws\n",
          info.code_identifier.c_str(),
          info.code_file.c_str());
  return true;
}

// wcstol_positive_strict is sort of like wcstol, but much stricter.  string
// should be a buffer pointing to a null-terminated string containing only
// decimal digits.  If the entire string can be converted to an integer
// without overflowing, and there are no non-digit characters before the
// result is set to the value and this function returns true.  Otherwise,
// this function returns false.  This is an alternative to the strtol, atoi,
// and scanf families, which are not as strict about input and in some cases
// don't provide a good way for the caller to determine if a conversion was
// successful.
static bool wcstol_positive_strict(wchar_t* string, int* result) {
  int value = 0;
  for (wchar_t* c = string; *c != '\0'; ++c) {
    int last_value = value;
    value *= 10;
    // Detect overflow.
    if (value / 10 != last_value || value < 0) {
      return false;
    }
    if (*c < '0' || *c > '9') {
      return false;
    }
    unsigned int c_value = *c - '0';
    last_value = value;
    value += c_value;
    // Detect overflow.
    if (value < last_value) {
      return false;
    }
    // Forbid leading zeroes unless the string is just "0".
    if (value == 0 && *(c+1) != '\0') {
      return false;
    }
  }
  *result = value;
  return true;
}

bool PDBSourceLineWriter::FindPEFile() {
  CComPtr<IDiaSymbol> global;
  if (FAILED(session_->get_globalScope(&global))) {
    fprintf(stderr, "get_globalScope failed\n");
    return false;
  }

  CComBSTR symbols_file;
  if (SUCCEEDED(global->get_symbolsFileName(&symbols_file))) {
    wstring file(symbols_file);

    // Look for an EXE or DLL file.
    const wchar_t* extensions[] = { L"exe", L"dll" };
    for (size_t i = 0; i < sizeof(extensions) / sizeof(extensions[0]); i++) {
      size_t dot_pos = file.find_last_of(L".");
      if (dot_pos != wstring::npos) {
        file.replace(dot_pos + 1, wstring::npos, extensions[i]);
        // Check if this file exists.
        if (GetFileAttributesW(file.c_str()) != INVALID_FILE_ATTRIBUTES) {
          code_file_ = file;
          return true;
        }
      }
    }
  }

  return false;
}

// static
bool PDBSourceLineWriter::GetSymbolFunctionName(IDiaSymbol* function,
                                                BSTR* name,
                                                int* stack_param_size) {
  *stack_param_size = -1;

  // Use get_undecoratedNameEx to get readable C++ names with arguments.
  if (function->get_undecoratedNameEx(kUndecorateOptions, name) != S_OK) {
    if (function->get_name(name) != S_OK) {
      fprintf(stderr, "failed to get function name\n");
      return false;
    }

    // It's possible for get_name to return an empty string, so
    // special-case that.
    if (wcscmp(*name, L"") == 0) {
      SysFreeString(*name);
      // dwarf_cu_to_module.cc uses "<name omitted>", so match that.
      *name = SysAllocString(L"<name omitted>");
      return true;
    }

    // If a name comes from get_name because no undecorated form existed,
    // it's already formatted properly to be used as output.  Don't do any
    // additional processing.
    //
    // MSVC7's DIA seems to not undecorate names in as many cases as MSVC8's.
    // This will result in calling get_name for some C++ symbols, so
    // all of the parameter and return type information may not be included in
    // the name string.
  } else {
    StripLlvmSuffixAndUndecorate(name);

    // C++ uses a bogus "void" argument for functions and methods that don't
    // take any parameters.  Take it out of the undecorated name because it's
    // ugly and unnecessary.
    const wchar_t* replace_string = L"(void)";
    const size_t replace_length = wcslen(replace_string);
    const wchar_t* replacement_string = L"()";
    size_t length = wcslen(*name);
    if (length >= replace_length) {
      wchar_t* name_end = *name + length - replace_length;
      if (wcscmp(name_end, replace_string) == 0) {
        WindowsStringUtils::safe_wcscpy(name_end, replace_length,
                                        replacement_string);
        length = wcslen(*name);
      }
    }

    // Undecorate names used for stdcall and fastcall.  These names prefix
    // the identifier with '_' (stdcall) or '@' (fastcall) and suffix it
    // with '@' followed by the number of bytes of parameters, in decimal.
    // If such a name is found, take note of the size and undecorate it.
    // Only do this for names that aren't C++, which is determined based on
    // whether the undecorated name contains any ':' or '(' characters.
    if (!wcschr(*name, ':') && !wcschr(*name, '(') &&
        (*name[0] == '_' || *name[0] == '@')) {
      wchar_t* last_at = wcsrchr(*name + 1, '@');
      if (last_at && wcstol_positive_strict(last_at + 1, stack_param_size)) {
        // If this function adheres to the fastcall convention, it accepts up
        // to the first 8 bytes of parameters in registers (%ecx and %edx).
        // We're only interested in the stack space used for parameters, so
        // so subtract 8 and don't let the size go below 0.
        if (*name[0] == '@') {
          if (*stack_param_size > 8) {
            *stack_param_size -= 8;
          } else {
            *stack_param_size = 0;
          }
        }

        // Undecorate the name by moving it one character to the left in its
        // buffer, and terminating it where the last '@' had been.
        WindowsStringUtils::safe_wcsncpy(*name, length,
                                         *name + 1, last_at - *name - 1);
     } else if (*name[0] == '_') {
        // This symbol's name is encoded according to the cdecl rules.  The
        // name doesn't end in a '@' character followed by a decimal positive
        // integer, so it's not a stdcall name.  Strip off the leading
        // underscore.
        WindowsStringUtils::safe_wcsncpy(*name, length, *name + 1, length);
      }
    }
  }

  return true;
}

// static
int PDBSourceLineWriter::GetFunctionStackParamSize(IDiaSymbol* function) {
  // This implementation is highly x86-specific.

  // Gather the symbols corresponding to data.
  CComPtr<IDiaEnumSymbols> data_children;
  if (FAILED(function->findChildren(SymTagData, NULL, nsNone,
                                    &data_children))) {
    return 0;
  }

  // lowest_base is the lowest %ebp-relative byte offset used for a parameter.
  // highest_end is one greater than the highest offset (i.e. base + length).
  // Stack parameters are assumed to be contiguous, because in reality, they
  // are.
  int lowest_base = INT_MAX;
  int highest_end = INT_MIN;

  CComPtr<IDiaSymbol> child;
  DWORD count;
  while (SUCCEEDED(data_children->Next(1, &child, &count)) && count == 1) {
    // If any operation fails at this point, just proceed to the next child.
    // Use the next_child label instead of continue because child needs to
    // be released before it's reused.  Declare constructable/destructable
    // types early to avoid gotos that cross initializations.
    CComPtr<IDiaSymbol> child_type;

    // DataIsObjectPtr is only used for |this|.  Because |this| can be passed
    // as a stack parameter, look for it in addition to traditional
    // parameters.
    DWORD child_kind;
    if (FAILED(child->get_dataKind(&child_kind)) ||
        (child_kind != DataIsParam && child_kind != DataIsObjectPtr)) {
      goto next_child;
    }

    // Only concentrate on register-relative parameters.  Parameters may also
    // be enregistered (passed directly in a register), but those don't
    // consume any stack space, so they're not of interest.
    DWORD child_location_type;
    if (FAILED(child->get_locationType(&child_location_type)) ||
        child_location_type != LocIsRegRel) {
      goto next_child;
    }

    // Of register-relative parameters, the only ones that make any sense are
    // %ebp- or %esp-relative.  Note that MSVC's debugging information always
    // gives parameters as %ebp-relative even when a function doesn't use a
    // traditional frame pointer and stack parameters are accessed relative to
    // %esp, so just look for %ebp-relative parameters.  If you wanted to
    // access parameters, you'd probably want to treat these %ebp-relative
    // offsets as if they were relative to %esp before a function's prolog
    // executed.
    DWORD child_register;
    if (FAILED(child->get_registerId(&child_register)) ||
        child_register != CV_REG_EBP) {
      goto next_child;
    }

    LONG child_register_offset;
    if (FAILED(child->get_offset(&child_register_offset))) {
      goto next_child;
    }

    // IDiaSymbol::get_type can succeed but still pass back a NULL value.
    if (FAILED(child->get_type(&child_type)) || !child_type) {
      goto next_child;
    }

    ULONGLONG child_length;
    if (FAILED(child_type->get_length(&child_length))) {
      goto next_child;
    }

    // Extra scope to avoid goto jumping over variable initialization
    {
      int child_end = child_register_offset + static_cast<ULONG>(child_length);
      if (child_register_offset < lowest_base) {
        lowest_base = child_register_offset;
      }
      if (child_end > highest_end) {
        highest_end = child_end;
      }
    }

next_child:
    child.Release();
  }

  int param_size = 0;
  // Make sure lowest_base isn't less than 4, because [%esp+4] is the lowest
  // possible address to find a stack parameter before executing a function's
  // prolog (see above).  Some optimizations cause parameter offsets to be
  // lower than 4, but we're not concerned with those because we're only
  // looking for parameters contained in addresses higher than where the
  // return address is stored.
  if (lowest_base < 4) {
    lowest_base = 4;
  }
  if (highest_end > lowest_base) {
    // All stack parameters are pushed as at least 4-byte quantities.  If the
    // last type was narrower than 4 bytes, promote it.  This assumes that all
    // parameters' offsets are 4-byte-aligned, which is always the case.  Only
    // worry about the last type, because we're not summing the type sizes,
    // just looking at the lowest and highest offsets.
    int remainder = highest_end % 4;
    if (remainder) {
      highest_end += 4 - remainder;
    }

    param_size = highest_end - lowest_base;
  }

  return param_size;
}

bool PDBSourceLineWriter::WriteSymbols(FILE* symbol_file) {
  output_ = symbol_file;

  // Load the OMAP information, and disable auto-translation of addresses in
  // preference of doing it ourselves.
  OmapData omap_data;
  if (!GetOmapDataAndDisableTranslation(session_, &omap_data))
    return false;
  BuildImageMap(omap_data, &image_map_);

  bool ret = PrintPDBInfo();
  // This is not a critical piece of the symbol file.
  PrintPEInfo();
  ret = ret && PrintSourceFiles() && PrintFunctions() && PrintFrameData();
  PrintInlineOrigins();

  output_ = NULL;
  return ret;
}

void PDBSourceLineWriter::Close() {
  if (session_ != nullptr) {
    session_.Release();
  }
}

bool PDBSourceLineWriter::GetModuleInfo(PDBModuleInfo* info) {
  if (!info) {
    return false;
  }

  info->debug_file.clear();
  info->debug_identifier.clear();
  info->cpu.clear();

  CComPtr<IDiaSymbol> global;
  if (FAILED(session_->get_globalScope(&global))) {
    return false;
  }

  DWORD machine_type;
  // get_machineType can return S_FALSE.
  if (global->get_machineType(&machine_type) == S_OK) {
    // The documentation claims that get_machineType returns a value from
    // the CV_CPU_TYPE_e enumeration, but that's not the case.
    // Instead, it returns one of the IMAGE_FILE_MACHINE values as
    // defined here:
    // http://msdn.microsoft.com/en-us/library/ms680313%28VS.85%29.aspx
    info->cpu = FileHeaderMachineToCpuString(static_cast<WORD>(machine_type));
  } else {
    // Unexpected, but handle gracefully.
    info->cpu = L"unknown";
  }

  // DWORD* and int* are not compatible.  This is clean and avoids a cast.
  DWORD age;
  if (FAILED(global->get_age(&age))) {
    return false;
  }

  bool uses_guid;
  if (!UsesGUID(&uses_guid)) {
    return false;
  }

  if (uses_guid) {
    GUID guid;
    if (FAILED(global->get_guid(&guid))) {
      return false;
    }

    info->debug_identifier = GenerateDebugIdentifier(age, guid);
  } else {
    DWORD signature;
    if (FAILED(global->get_signature(&signature))) {
      return false;
    }

    info->debug_identifier = GenerateDebugIdentifier(age, signature);
  }

  CComBSTR debug_file_string;
  if (FAILED(global->get_symbolsFileName(&debug_file_string))) {
    return false;
  }
  info->debug_file =
      WindowsStringUtils::GetBaseName(wstring(debug_file_string));

  return true;
}

bool PDBSourceLineWriter::GetPEInfo(PEModuleInfo* info) {
  if (!info) {
    return false;
  }

  if (code_file_.empty() && !FindPEFile()) {
    fprintf(stderr, "Couldn't locate EXE or DLL file.\n");
    return false;
  }

  return ReadPEInfo(code_file_, info);
}

bool PDBSourceLineWriter::UsesGUID(bool* uses_guid) {
  if (!uses_guid)
    return false;

  CComPtr<IDiaSymbol> global;
  if (FAILED(session_->get_globalScope(&global)))
    return false;

  GUID guid;
  if (FAILED(global->get_guid(&guid)))
    return false;

  DWORD signature;
  if (FAILED(global->get_signature(&signature)))
    return false;

  // There are two possibilities for guid: either it's a real 128-bit GUID
  // as identified in a code module by a new-style CodeView record, or it's
  // a 32-bit signature (timestamp) as identified by an old-style record.
  // See MDCVInfoPDB70 and MDCVInfoPDB20 in minidump_format.h.
  //
  // Because DIA doesn't provide a way to directly determine whether a module
  // uses a GUID or a 32-bit signature, this code checks whether the first 32
  // bits of guid are the same as the signature, and if the rest of guid is
  // zero.  If so, then with a pretty high degree of certainty, there's an
  // old-style CodeView record in use.  This method will only falsely find an
  // an old-style CodeView record if a real 128-bit GUID has its first 32
  // bits set the same as the module's signature (timestamp) and the rest of
  // the GUID is set to 0.  This is highly unlikely.

  GUID signature_guid = {signature};  // 0-initializes other members
  *uses_guid = !IsEqualGUID(guid, signature_guid);
  return true;
}

}  // namespace google_breakpad
