// -*- mode: c++ -*-

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

// module.h: Define google_breakpad::Module. A Module holds debugging
// information, and can write that information out as a Breakpad
// symbol file.

#ifndef COMMON_LINUX_MODULE_H__
#define COMMON_LINUX_MODULE_H__

#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "common/string_view.h"
#include "common/symbol_data.h"
#include "common/unordered.h"
#include "common/using_std_string.h"
#include "google_breakpad/common/breakpad_types.h"

namespace google_breakpad {

using std::set;
using std::vector;
using std::map;

// A Module represents the contents of a module, and supports methods
// for adding information produced by parsing STABS or DWARF data
// --- possibly both from the same file --- and then writing out the
// unified contents as a Breakpad-format symbol file.
class Module {
 public:
  // The type of addresses and sizes in a symbol table.
  typedef uint64_t Address;
  static constexpr uint64_t kMaxAddress = std::numeric_limits<Address>::max();
  struct File;
  struct Function;
  struct InlineOrigin;
  struct Inline;
  struct Line;
  struct Extern;

  // Addresses appearing in File, Function, and Line structures are
  // absolute, not relative to the the module's load address.  That
  // is, if the module were loaded at its nominal load address, the
  // addresses would be correct.

  // A source file.
  struct File {
    explicit File(const string& name_input) : name(name_input), source_id(0) {}

    // The name of the source file.
    const string name;

    // The file's source id.  The Write member function clears this
    // field and assigns source ids a fresh, so any value placed here
    // before calling Write will be lost.
    int source_id;
  };

  // An address range.
  struct Range {
    Range(const Address address_input, const Address size_input) :
        address(address_input), size(size_input) { }

    Address address;
    Address size;
  };

  // A function.
  struct Function {
    Function(StringView name_input, const Address& address_input) :
        name(name_input), address(address_input), parameter_size(0) {}

    // For sorting by address.  (Not style-guide compliant, but it's
    // stupid not to put this in the struct.)
    static bool CompareByAddress(const Function* x, const Function* y) {
      return x->address < y->address;
    }

    // The function's name.
    StringView name;

    // The start address and the address ranges covered by the function.
    const Address address;
    vector<Range> ranges;

    // The function's parameter size.
    Address parameter_size;

    // Source lines belonging to this function, sorted by increasing
    // address.
    vector<Line> lines;

    // Inlined call sites belonging to this functions.
    vector<std::unique_ptr<Inline>> inlines;

    // If this symbol has been folded with other symbols in the linked binary.
    bool is_multiple = false;

    // If the function's name should be filled out from a matching Extern,
    // should they not match.
    bool prefer_extern_name = false;
  };

  struct InlineOrigin {
    explicit InlineOrigin(StringView name) : id(-1), name(name) {}

    // A unique id for each InlineOrigin object. INLINE records use the id to
    // refer to its INLINE_ORIGIN record.
    int id;

    // The inlined function's name.
    StringView name;
  };

  // A inlined call site.
  struct Inline {
    Inline(InlineOrigin* origin,
           const vector<Range>& ranges,
           int call_site_line,
           int call_site_file_id,
           int inline_nest_level,
           vector<std::unique_ptr<Inline>> child_inlines)
        : origin(origin),
          ranges(ranges),
          call_site_line(call_site_line),
          call_site_file_id(call_site_file_id),
          call_site_file(nullptr),
          inline_nest_level(inline_nest_level),
          child_inlines(std::move(child_inlines)) {}

    InlineOrigin* origin;

    // The list of addresses and sizes.
    vector<Range> ranges;

    int call_site_line;

    // The id is only meanful inside a CU. It's only used for looking up real
    // File* after scanning a CU.
    int call_site_file_id;

    File* call_site_file;

    int inline_nest_level;

    // A list of inlines which are children of this inline.
    vector<std::unique_ptr<Inline>> child_inlines;

    int getCallSiteFileID() const {
      return call_site_file ? call_site_file->source_id : -1;
    }

    static void InlineDFS(
        vector<std::unique_ptr<Module::Inline>>& inlines,
        std::function<void(std::unique_ptr<Module::Inline>&)> const& forEach) {
      for (std::unique_ptr<Module::Inline>& in : inlines) {
        forEach(in);
        InlineDFS(in->child_inlines, forEach);
      }
    }
  };

  typedef map<uint64_t, InlineOrigin*> InlineOriginByOffset;

  class InlineOriginMap {
   public:
    // Add INLINE ORIGIN to the module. Return a pointer to origin .
    InlineOrigin* GetOrCreateInlineOrigin(uint64_t offset, StringView name);

    // offset is the offset of a DW_TAG_subprogram. specification_offset is the
    // value of its DW_AT_specification or equals to offset if
    // DW_AT_specification doesn't exist in that DIE.
    void SetReference(uint64_t offset, uint64_t specification_offset);

    ~InlineOriginMap() {
      for (const auto& iter : inline_origins_) {
        delete iter.second;
      }
    }

   private:
    // A map from a DW_TAG_subprogram's offset to the DW_TAG_subprogram.
    InlineOriginByOffset inline_origins_;

    // A map from a DW_TAG_subprogram's offset to the offset of its
    // specification or abstract origin subprogram. The set of values in this
    // map should always be the same set of keys in inline_origins_.
    map<uint64_t, uint64_t> references_;
  };

  map<std::string, InlineOriginMap> inline_origin_maps;

  // A source line.
  struct Line {
    // For sorting by address.  (Not style-guide compliant, but it's
    // stupid not to put this in the struct.)
    static bool CompareByAddress(const Module::Line& x, const Module::Line& y) {
      return x.address < y.address;
    }

    Address address, size;    // The address and size of the line's code.
    File* file;                // The source file.
    int number;                // The source line number.
  };

  // An exported symbol.
  struct Extern {
    explicit Extern(const Address& address_input) : address(address_input) {}
    const Address address;
    string name;
    // If this symbol has been folded with other symbols in the linked binary.
    bool is_multiple = false;
  };

  // A map from register names to postfix expressions that recover
  // their their values. This can represent a complete set of rules to
  // follow at some address, or a set of changes to be applied to an
  // extant set of rules.
  typedef map<string, string> RuleMap;

  // A map from addresses to RuleMaps, representing changes that take
  // effect at given addresses.
  typedef map<Address, RuleMap> RuleChangeMap;

  // A range of 'STACK CFI' stack walking information. An instance of
  // this structure corresponds to a 'STACK CFI INIT' record and the
  // subsequent 'STACK CFI' records that fall within its range.
  struct StackFrameEntry {
    // The starting address and number of bytes of machine code this
    // entry covers.
    Address address, size;

    // The initial register recovery rules, in force at the starting
    // address.
    RuleMap initial_rules;

    // A map from addresses to rule changes. To find the rules in
    // force at a given address, start with initial_rules, and then
    // apply the changes given in this map for all addresses up to and
    // including the address you're interested in.
    RuleChangeMap rule_changes;
  };

  struct FunctionCompare {
    bool operator() (const Function* lhs, const Function* rhs) const {
      if (lhs->address == rhs->address)
        return lhs->name < rhs->name;
      return lhs->address < rhs->address;
    }
  };

  struct InlineOriginCompare {
    bool operator()(const InlineOrigin* lhs, const InlineOrigin* rhs) const {
      return lhs->name < rhs->name;
    }
  };

  struct ExternCompare {
    // Defining is_transparent allows
    // std::set<std::unique_ptr<Extern>, ExternCompare>::find() to be called
    // with an Extern* and have set use the overloads below.
    using is_transparent = void;
    bool operator() (const std::unique_ptr<Extern>& lhs,
                     const std::unique_ptr<Extern>& rhs) const {
      return lhs->address < rhs->address;
    }
    bool operator() (const Extern* lhs, const std::unique_ptr<Extern>& rhs) const {
      return lhs->address < rhs->address;
    }
    bool operator() (const std::unique_ptr<Extern>& lhs, const Extern* rhs) const {
      return lhs->address < rhs->address;
    }
  };

  // Create a new module with the given name, operating system,
  // architecture, and ID string.
  // NB: `enable_multiple_field` is temporary while transitioning to enabling
  // writing the multiple field permanently.
  Module(const string& name,
         const string& os,
         const string& architecture,
         const string& id,
         const string& code_id = "",
         bool enable_multiple_field = false,
         bool prefer_extern_name = false);
  ~Module();

  // Set the module's load address to LOAD_ADDRESS; addresses given
  // for functions and lines will be written to the Breakpad symbol
  // file as offsets from this address.  Construction initializes this
  // module's load address to zero: addresses written to the symbol
  // file will be the same as they appear in the Function, Line, and
  // StackFrameEntry structures.
  //
  // Note that this member function has no effect on addresses stored
  // in the data added to this module; the Write member function
  // simply subtracts off the load address from addresses before it
  // prints them. Only the last load address given before calling
  // Write is used.
  void SetLoadAddress(Address load_address);

  // Sets address filtering on elements added to the module.  This allows
  // libraries with extraneous debug symbols to generate symbol files containing
  // only relevant symbols.  For example, an LLD-generated partition library may
  // contain debug information pertaining to all partitions derived from a
  // single "combined" library.  Filtering applies only to elements added after
  // this method is called.
  void SetAddressRanges(const vector<Range>& ranges);

  // Add FUNCTION to the module. FUNCTION's name must not be empty.
  // This module owns all Function objects added with this function:
  // destroying the module destroys them as well.
  // Return false if the function is duplicate and needs to be freed.
  bool AddFunction(Function* function);

  // Add STACK_FRAME_ENTRY to the module.
  // This module owns all StackFrameEntry objects added with this
  // function: destroying the module destroys them as well.
  void AddStackFrameEntry(std::unique_ptr<StackFrameEntry> stack_frame_entry);

  // Add PUBLIC to the module.
  // This module owns all Extern objects added with this function:
  // destroying the module destroys them as well.
  void AddExtern(std::unique_ptr<Extern> ext);

  // If this module has a file named NAME, return a pointer to it. If
  // it has none, then create one and return a pointer to the new
  // file. This module owns all File objects created using these
  // functions; destroying the module destroys them as well.
  File* FindFile(const string& name);
  File* FindFile(const char* name);

  // If this module has a file named NAME, return a pointer to it.
  // Otherwise, return NULL.
  File* FindExistingFile(const string& name);

  // Insert pointers to the functions added to this module at I in
  // VEC. The pointed-to Functions are still owned by this module.
  // (Since this is effectively a copy of the function list, this is
  // mostly useful for testing; other uses should probably get a more
  // appropriate interface.)
  void GetFunctions(vector<Function*>* vec, vector<Function*>::iterator i);

  // Insert pointers to the externs added to this module at I in
  // VEC. The pointed-to Externs are still owned by this module.
  // (Since this is effectively a copy of the extern list, this is
  // mostly useful for testing; other uses should probably get a more
  // appropriate interface.)
  void GetExterns(vector<Extern*>* vec, vector<Extern*>::iterator i);

  // Clear VEC and fill it with pointers to the Files added to this
  // module, sorted by name. The pointed-to Files are still owned by
  // this module. (Since this is effectively a copy of the file list,
  // this is mostly useful for testing; other uses should probably get
  // a more appropriate interface.)
  void GetFiles(vector<File*>* vec);

  // Clear VEC and fill it with pointers to the StackFrameEntry
  // objects that have been added to this module. (Since this is
  // effectively a copy of the stack frame entry list, this is mostly
  // useful for testing; other uses should probably get
  // a more appropriate interface.)
  void GetStackFrameEntries(vector<StackFrameEntry*>* vec) const;

  // Find those files in this module that are actually referred to by
  // functions' line number data, and assign them source id numbers.
  // Set the source id numbers for all other files --- unused by the
  // source line data --- to -1.  We do this before writing out the
  // symbol file, at which point we omit any unused files.
  void AssignSourceIds();

  // This function should be called before AssignSourceIds() to get the set of
  // valid InlineOrigins*.
  void CreateInlineOrigins(
      set<InlineOrigin*, InlineOriginCompare>& inline_origins);

  // Call AssignSourceIds, and write this module to STREAM in the
  // breakpad symbol format. Return true if all goes well, or false if
  // an error occurs. This method writes out:
  // - a header based on the values given to the constructor,
  // If symbol_data is not CFI then:
  // - the source files added via FindFile,
  // - the functions added via AddFunctions, each with its lines,
  // - all public records,
  // If symbol_data is CFI then:
  // - all CFI records.
  // Addresses in the output are all relative to the load address
  // established by SetLoadAddress, unless preserve_load_address
  // is equal to true, in which case each address will remain unchanged.
  bool Write(std::ostream& stream, SymbolData symbol_data, bool preserve_load_address = false);

  // Place the name in the global set of strings. Return a StringView points to
  // a string inside the pool.
  StringView AddStringToPool(const string& str) {
    auto result = common_strings_.insert(str);
    return *(result.first);
  }

  string name() const { return name_; }
  string os() const { return os_; }
  string architecture() const { return architecture_; }
  string identifier() const { return id_; }
  string code_identifier() const { return code_id_; }

 private:
  // Report an error that has occurred writing the symbol file, using
  // errno to find the appropriate cause.  Return false.
  static bool ReportError();

  // Write RULE_MAP to STREAM, in the form appropriate for 'STACK CFI'
  // records, without a final newline. Return true if all goes well;
  // if an error occurs, return false, and leave errno set.
  static bool WriteRuleMap(const RuleMap& rule_map, std::ostream& stream);

  // Returns true of the specified address resides with an specified address
  // range, or if no ranges have been specified.
  bool AddressIsInModule(Address address) const;

  // Module header entries.
  string name_, os_, architecture_, id_, code_id_;

  // The module's nominal load address.  Addresses for functions and
  // lines are absolute, assuming the module is loaded at this
  // address.
  Address load_address_;

  // The set of valid address ranges of the module.  If specified, attempts to
  // add elements residing outside these ranges will be silently filtered.
  vector<Range> address_ranges_;

  // Relation for maps whose keys are strings shared with some other
  // structure.
  struct CompareStringPtrs {
    bool operator()(const string* x, const string* y) const { return *x < *y; }
  };

  // A map from filenames to File structures.  The map's keys are
  // pointers to the Files' names.
  typedef map<const string*, File*, CompareStringPtrs> FileByNameMap;

  // A set containing Function structures, sorted by address.
  typedef set<Function*, FunctionCompare> FunctionSet;

  // A set containing Extern structures, sorted by address.
  typedef set<std::unique_ptr<Extern>, ExternCompare> ExternSet;

  // The module owns all the files and functions that have been added
  // to it; destroying the module frees the Files and Functions these
  // point to.
  FileByNameMap files_;    // This module's source files.
  FunctionSet functions_;  // This module's functions.
  // Used to quickly look up whether a function exists at a particular address.
  unordered_set<Address> function_addresses_;

  // The module owns all the call frame info entries that have been
  // added to it.
  vector<std::unique_ptr<StackFrameEntry>> stack_frame_entries_;

  // The module owns all the externs that have been added to it;
  // destroying the module frees the Externs these point to.
  ExternSet externs_;

  unordered_set<string> common_strings_;

  // Whether symbols sharing an address should be collapsed into a single entry
  // and marked with an `m` in the output. See
  // https://bugs.chromium.org/p/google-breakpad/issues/detail?id=751 and docs
  // at
  // https://chromium.googlesource.com/breakpad/breakpad/+/master/docs/symbol_files.md#records-3
  bool enable_multiple_field_;

  // If a Function and an Extern share the same address but have a different
  // name, prefer the name of the Extern.
  //
  // Use this when dumping Mach-O .dSYMs built with -gmlt (Minimum Line Tables),
  // as the Function's fully-qualified name will only be present in the STABS
  // (which are placed in the Extern), not in the DWARF symbols (which are
  // placed in the Function).
  bool prefer_extern_name_;
};

}  // namespace google_breakpad

#endif  // COMMON_LINUX_MODULE_H__
