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

// PDBSourceLineWriter uses a pdb file produced by Visual C++ to output
// a line/address map for use with BasicSourceLineResolver.

#ifndef COMMON_WINDOWS_PDB_SOURCE_LINE_WRITER_H_
#define COMMON_WINDOWS_PDB_SOURCE_LINE_WRITER_H_

#include <atlcomcli.h>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/windows/module_info.h"
#include "common/windows/omap.h"

struct IDiaEnumLineNumbers;
struct IDiaSession;
struct IDiaSymbol;

namespace google_breakpad {

using std::map;
using std::vector;
using std::wstring;
using std::unordered_map;

class PDBSourceLineWriter {
 public:
  enum FileFormat {
    PDB_FILE,  // a .pdb file containing debug symbols
    EXE_FILE,  // a .exe or .dll file
    ANY_FILE   // try PDB_FILE and then EXE_FILE
  };

  explicit PDBSourceLineWriter(bool handle_inline = false);
  ~PDBSourceLineWriter();

  // Opens the given file.  For executable files, the corresponding pdb
  // file must be available; Open will be if it is not.
  // If there is already a pdb file open, it is automatically closed.
  // Returns true on success.
  bool Open(const wstring& file, FileFormat format);

  // Closes the current pdb file and its associated resources.
  void Close();

  // Sets the code file full path.  This is optional for 32-bit modules.  It is
  // also optional for 64-bit modules when there is an executable file stored
  // in the same directory as the PDB file.  It is only required for 64-bit
  // modules when the executable file is not in the same location as the PDB
  // file and it must be called after Open() and before WriteMap().
  // If Open() was called for an executable file, then it is an error to call
  // SetCodeFile() with a different file path and it will return false.
  bool SetCodeFile(const wstring& exe_file);

  // Writes a Breakpad symbol file from the current pdb file to |symbol_file|.
  // Returns true on success.
  bool WriteSymbols(FILE *symbol_file);

  // Retrieves information about the module's debugging file.  Returns
  // true on success and false on failure.
  bool GetModuleInfo(PDBModuleInfo *info);

  // Retrieves information about the module's PE file.  Returns
  // true on success and false on failure.
  bool GetPEInfo(PEModuleInfo *info);

  // Sets uses_guid to true if the opened file uses a new-style CodeView
  // record with a 128-bit GUID, or false if the opened file uses an old-style
  // CodeView record.  When no GUID is available, a 32-bit signature should be
  // used to identify the module instead.  If the information cannot be
  // determined, this method returns false.
  bool UsesGUID(bool *uses_guid);

 private:
  // InlineOrigin represents INLINE_ORIGIN record in a symbol file. It's an
  // inlined function.
  struct InlineOrigin {
    // The unique id for an InlineOrigin.
    int id;
    // The name of the inlined function.
    wstring name;
  };

  // Line represents LINE record in a symbol file. It represents a source code
  // line.
  struct Line {
    // The relative address of a line.
    DWORD rva;
    // The number bytes this line has.
    DWORD length;
    // The source line number.
    DWORD line_num;
    // The source file id where the source line is located at.
    DWORD file_id;
  };

  // Inline represents INLINE record in a symbol file.
  class Inline {
   public:
    explicit Inline(int inline_nest_level);

    void SetOriginId(int origin_id);

    // Adding inlinee line's range into ranges. If line is adjacent with any
    // existing lines, extend the range. Otherwise, add line as a new range.
    void ExtendRanges(const Line& line);

    void SetCallSiteLine(DWORD call_site_line);

    void SetCallSiteFileId(DWORD call_site_file_id);

    void SetChildInlines(std::vector<std::unique_ptr<Inline>> child_inlines);

    void Print(FILE* output) const;

   private:
    // The nest level of this inline record.
    int inline_nest_level_;
    // The source line number at where this inlined function is called.
    DWORD call_site_line_ = 0;
    // The call site file id at where this inlined function is called.
    DWORD call_site_file_id_ = 0;
    // The id used for referring to an InlineOrigin.
    int origin_id_ = 0;
    // A map from rva to length. This is the address ranges covered by this
    // Inline.
    map<DWORD, DWORD> ranges_;
    // The list of direct Inlines inlined inside this Inline.
    vector<std::unique_ptr<Inline>> child_inlines_;
  };

  // Lines represents a map of lines inside a function with rva as the key.
  // AddLine function adds a line into the map and ensures that there is no
  // overlap between any two lines in the map.
  class Lines {
   public:
    const map<DWORD, Line>& GetLineMap() const { return line_map_; }

    // Finds the line from line_map_ that contains the given rva returns its
    // line_num. If not found, return 0.
    DWORD GetLineNum(DWORD rva) const;

    // Finds the line from line_map_ that contains the given rva returns its
    // file_id. If not found, return 0.
    DWORD GetFileId(DWORD rva) const;

    // Add the `line` into line_map_. If the `line` overlaps with existing
    // lines, truncate the existing lines and add the given line. It ensures
    // that all lines in line_map_ do not overlap with each other. For example,
    // suppose there is a line A in the map and we call AddLine with Line B.
    // Line A: rva: 100, length: 20, line_num: 10, file_id: 1
    // Line B: rva: 105, length: 10, line_num: 4, file_id: 2
    // After calling AddLine with Line B, we will have the following lines:
    // Line 1: rva: 100, length: 5, line_num: 10, file_id: 1
    // Line 2: rva: 105, length: 10, line_num: 4, file_id: 2
    // Line 3: rva: 115, length: 5, line_num: 10, file_id: 1
    void AddLine(const Line& line);

   private:
    // Finds the line from line_map_ that contains the given rva. If not found,
    // return nullptr.
    const Line* GetLine(DWORD rva) const;
    // The key is rva. AddLine function ensures that any two lines in the map do
    // not overlap.
    map<DWORD, Line> line_map_;
  };

  // Construct Line from IDiaLineNumber. The output Line is stored at line.
  // Return true on success.
  bool GetLine(IDiaLineNumber* dia_line, Line* line) const;

  // Construct Lines from IDiaEnumLineNumbers. The list of Lines are stored at
  // line_list.
  // Returns true on success.
  bool GetLines(IDiaEnumLineNumbers* lines, Lines* line_list) const;

  // Outputs the line/address pairs for each line in the enumerator.
  void PrintLines(const Lines& lines) const;

  // Outputs a function address and name, followed by its source line list.
  // block can be the same object as function, or it can be a reference to a
  // code block that is lexically part of this function, but resides at a
  // separate address. If has_multiple_symbols is true, this function's
  // instructions correspond to multiple symbols. Returns true on success.
  bool PrintFunction(IDiaSymbol *function, IDiaSymbol *block,
                     bool has_multiple_symbols);

  // Outputs all functions as described above.  Returns true on success.
  bool PrintFunctions();

  // Outputs all of the source files in the session's pdb file.
  // Returns true on success.
  bool PrintSourceFiles();

  // Output all inline origins.
  void PrintInlineOrigins() const;

  // Retrieve inlines inside the given block. It also adds inlinee lines to
  // `line_list` since inner lines are more precise source location. If the
  // block has children wih SymTagInlineSite Tag, it will recursively (DFS) call
  // itself with each child as first argument. Returns true on success.
  // `block`: the IDiaSymbol that may have inline sites.
  // `line_list`: the list of lines inside current function.
  // `inline_nest_level`: the nest level of block's Inlines.
  // `inlines`: the vector to store the list of inlines for the block.
  bool GetInlines(IDiaSymbol* block,
                  Lines* line_list,
                  int inline_nest_level,
                  vector<std::unique_ptr<Inline>>* inlines);

  // Outputs all inlines.
  void PrintInlines(const vector<std::unique_ptr<Inline>>& inlines) const;

  // Outputs all of the frame information necessary to construct stack
  // backtraces in the absence of frame pointers. For x86 data stored in
  // .pdb files. Returns true on success.
  bool PrintFrameDataUsingPDB();

  // Outputs all of the frame information necessary to construct stack
  // backtraces in the absence of frame pointers. For x64 data stored in
  // .exe, .dll files. Returns true on success.
  bool PrintFrameDataUsingEXE();

  // Outputs all of the frame information necessary to construct stack
  // backtraces in the absence of frame pointers.  Returns true on success.
  bool PrintFrameData();

  // Outputs a single public symbol address and name, if the symbol corresponds
  // to a code address.  Returns true on success.  If symbol is does not
  // correspond to code, returns true without outputting anything. If
  // has_multiple_symbols is true, the symbol corresponds to a code address and
  // the instructions correspond to multiple symbols.
  bool PrintCodePublicSymbol(IDiaSymbol *symbol, bool has_multiple_symbols);

  // Outputs a line identifying the PDB file that is being dumped, along with
  // its uuid and age.
  bool PrintPDBInfo();

  // Outputs a line identifying the PE file corresponding to the PDB
  // file that is being dumped, along with its code identifier,
  // which consists of its timestamp and file size.
  bool PrintPEInfo();

  // Returns true if this filename has already been seen,
  // and an ID is stored for it, or false if it has not.
  bool FileIDIsCached(const wstring& file) {
    return unique_files_.find(file) != unique_files_.end();
  }

  // Cache this filename and ID for later reuse.
  void CacheFileID(const wstring& file, DWORD id) {
    unique_files_[file] = id;
  }

  // Store this ID in the cache as a duplicate for this filename.
  void StoreDuplicateFileID(const wstring& file, DWORD id) {
    unordered_map<wstring, DWORD>::iterator iter = unique_files_.find(file);
    if (iter != unique_files_.end()) {
      // map this id to the previously seen one
      file_ids_[id] = iter->second;
    }
  }

  // Given a file's unique ID, return the ID that should be used to
  // reference it. There may be multiple files with identical filenames
  // but different unique IDs. The cache attempts to coalesce these into
  // one ID per unique filename.
  DWORD GetRealFileID(DWORD id) const {
    unordered_map<DWORD, DWORD>::const_iterator iter = file_ids_.find(id);
    if (iter == file_ids_.end())
      return id;
    return iter->second;
  }

  // Find the PE file corresponding to the loaded PDB file, and
  // set the code_file_ member. Returns false on failure.
  bool FindPEFile();

  // Returns the function name for a symbol.  If possible, the name is
  // undecorated.  If the symbol's decorated form indicates the size of
  // parameters on the stack, this information is returned in stack_param_size.
  // Returns true on success.  If the symbol doesn't encode parameter size
  // information, stack_param_size is set to -1.
  static bool GetSymbolFunctionName(IDiaSymbol *function, BSTR *name,
                                    int *stack_param_size);

  // Returns the number of bytes of stack space used for a function's
  // parameters.  function must have the tag SymTagFunction.  In the event of
  // a failure, returns 0, which is also a valid number of bytes.
  static int GetFunctionStackParamSize(IDiaSymbol *function);

  // The filename of the PE file corresponding to the currently-open
  // pdb file.
  wstring code_file_;

  // The session for the currently-open pdb file.
  CComPtr<IDiaSession> session_;

  // The current output file for this WriteMap invocation.
  FILE *output_;

  // There may be many duplicate filenames with different IDs.
  // This maps from the DIA "unique ID" to a single ID per unique
  // filename.
  unordered_map<DWORD, DWORD> file_ids_;
  // This maps unique filenames to file IDs.
  unordered_map<wstring, DWORD> unique_files_;

  // The INLINE_ORIGINS records. The key is the function name.
  std::map<wstring, InlineOrigin> inline_origins_;

  // This is used for calculating post-transform symbol addresses and lengths.
  ImageMap image_map_;

  // If we should output INLINE/INLINE_ORIGIN records
  bool handle_inline_;

  // Disallow copy ctor and operator=
  PDBSourceLineWriter(const PDBSourceLineWriter&);
  void operator=(const PDBSourceLineWriter&);
};

}  // namespace google_breakpad

#endif  // COMMON_WINDOWS_PDB_SOURCE_LINE_WRITER_H_
