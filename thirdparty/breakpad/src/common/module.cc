// Copyright 2011 Google LLC
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

// module.cc: Implement google_breakpad::Module.  See module.h.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/module.h"
#include "common/string_view.h"

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <functional>
#include <iostream>
#include <memory>
#include <utility>

namespace google_breakpad {

using std::dec;
using std::hex;
using std::unique_ptr;

Module::InlineOrigin* Module::InlineOriginMap::GetOrCreateInlineOrigin(
    uint64_t offset,
    StringView name) {
  uint64_t specification_offset = references_[offset];
  // Find the root offset.
  auto iter = references_.find(specification_offset);
  while (iter != references_.end() &&
         specification_offset != references_[specification_offset]) {
    specification_offset = references_[specification_offset];
    iter = references_.find(specification_offset);
  }
  if (inline_origins_.find(specification_offset) != inline_origins_.end()) {
    if (inline_origins_[specification_offset]->name == "<name omitted>") {
      inline_origins_[specification_offset]->name = name;
    }
    return inline_origins_[specification_offset];
  }
  inline_origins_[specification_offset] = new Module::InlineOrigin(name);
  return inline_origins_[specification_offset];
}

void Module::InlineOriginMap::SetReference(uint64_t offset,
                                           uint64_t specification_offset) {
  // If we haven't seen this doesn't exist in reference map, always add it.
  if (references_.find(offset) == references_.end()) {
    references_[offset] = specification_offset;
    return;
  }
  // If offset equals specification_offset and offset exists in
  // references_, there is no need to update the references_ map.
  // This early return is necessary because the call to erase in following if
  // will remove the entry of specification_offset in inline_origins_. If
  // specification_offset equals to references_[offset], it might be
  // duplicate debug info.
  if (offset == specification_offset ||
      specification_offset == references_[offset])
    return;

  // Fix up mapping in inline_origins_.
  auto remove = inline_origins_.find(references_[offset]);
  if (remove != inline_origins_.end()) {
    inline_origins_[specification_offset] = std::move(remove->second);
    inline_origins_.erase(remove);
  }
  references_[offset] = specification_offset;
}

Module::Module(const string& name,
               const string& os,
               const string& architecture,
               const string& id,
               const string& code_id /* = "" */,
               bool enable_multiple_field /* = false*/,
               bool prefer_extern_name /* = false*/)
    : name_(name),
      os_(os),
      architecture_(architecture),
      id_(id),
      code_id_(code_id),
      load_address_(0),
      enable_multiple_field_(enable_multiple_field),
      prefer_extern_name_(prefer_extern_name) {}

Module::~Module() {
  for (FileByNameMap::iterator it = files_.begin(); it != files_.end(); ++it)
    delete it->second;
  for (FunctionSet::iterator it = functions_.begin();
       it != functions_.end(); ++it) {
    delete *it;
  }
}

void Module::SetLoadAddress(Address address) {
  load_address_ = address;
}

void Module::SetAddressRanges(const vector<Range>& ranges) {
  address_ranges_ = ranges;
}

bool Module::AddFunction(Function* function) {
  // FUNC lines must not hold an empty name, so catch the problem early if
  // callers try to add one.
  assert(!function->name.empty());

  if (!AddressIsInModule(function->address)) {
    return false;
  }

  // FUNCs are better than PUBLICs as they come with sizes, so remove an extern
  // with the same address if present.
  Extern ext(function->address);
  ExternSet::iterator it_ext = externs_.find(&ext);
  if (it_ext == externs_.end() &&
      architecture_ == "arm" &&
      (function->address & 0x1) == 0) {
    // ARM THUMB functions have bit 0 set. ARM64 does not have THUMB.
    Extern arm_thumb_ext(function->address | 0x1);
    it_ext = externs_.find(&arm_thumb_ext);
  }
  if (it_ext != externs_.end()) {
    Extern* found_ext = it_ext->get();
    bool name_mismatch = found_ext->name != function->name;
    if (enable_multiple_field_) {
      bool is_multiple_based_on_name;
      // In the case of a .dSYM built with -gmlt, the external name will be the
      // fully-qualified symbol name, but the function name will be the partial
      // name (or omitted).
      //
      // Don't mark multiple in this case.
      if (name_mismatch &&
          (function->name == "<name omitted>" ||
           found_ext->name.find(function->name.str()) != string::npos)) {
        is_multiple_based_on_name = false;
      } else {
        is_multiple_based_on_name = name_mismatch;
      }
      // If the PUBLIC is for the same symbol as the FUNC, don't mark multiple.
      function->is_multiple |=
          is_multiple_based_on_name || found_ext->is_multiple;
    }
    if (name_mismatch && prefer_extern_name_) {
      function->name = AddStringToPool(it_ext->get()->name);
    }
    externs_.erase(it_ext);
  }
#if _DEBUG
  {
    // There should be no other PUBLIC symbols that overlap with the function.
    for (const Range& range : function->ranges) {
      Extern debug_ext(range.address);
      ExternSet::iterator it_debug = externs_.lower_bound(&ext);
      assert(it_debug == externs_.end() ||
             (*it_debug)->address >= range.address + range.size);
    }
  }
#endif
  if (enable_multiple_field_ && function_addresses_.count(function->address)) {
    FunctionSet::iterator existing_function = std::find_if(
        functions_.begin(), functions_.end(),
        [&](Function* other) { return other->address == function->address; });
    assert(existing_function != functions_.end());
    (*existing_function)->is_multiple = true;
    // Free the duplicate that was not inserted because this Module
    // now owns it.
    return false;
  }
  function_addresses_.emplace(function->address);
  std::pair<FunctionSet::iterator, bool> ret = functions_.insert(function);
  if (!ret.second && (*ret.first != function)) {
    // Free the duplicate that was not inserted because this Module
    // now owns it.
    return false;
  }
  return true;
}

void Module::AddStackFrameEntry(std::unique_ptr<StackFrameEntry> stack_frame_entry) {
  if (!AddressIsInModule(stack_frame_entry->address)) {
    return;
  }

  stack_frame_entries_.push_back(std::move(stack_frame_entry));
}

void Module::AddExtern(std::unique_ptr<Extern> ext) {
  if (!AddressIsInModule(ext->address)) {
    return;
  }

  std::pair<ExternSet::iterator,bool> ret = externs_.emplace(std::move(ext));
  if (!ret.second && enable_multiple_field_) {
    (*ret.first)->is_multiple = true;
  }
}

void Module::GetFunctions(vector<Function*>* vec,
                          vector<Function*>::iterator i) {
  vec->insert(i, functions_.begin(), functions_.end());
}

void Module::GetExterns(vector<Extern*>* vec,
                        vector<Extern*>::iterator i) {
  auto pos = vec->insert(i, externs_.size(), nullptr);
  for (const std::unique_ptr<Extern>& ext : externs_) {
    *pos = ext.get();
    ++pos;
  }
}

Module::File* Module::FindFile(const string& name) {
  // A tricky bit here.  The key of each map entry needs to be a
  // pointer to the entry's File's name string.  This means that we
  // can't do the initial lookup with any operation that would create
  // an empty entry for us if the name isn't found (like, say,
  // operator[] or insert do), because such a created entry's key will
  // be a pointer the string passed as our argument.  Since the key of
  // a map's value type is const, we can't fix it up once we've
  // created our file.  lower_bound does the lookup without doing an
  // insertion, and returns a good hint iterator to pass to insert.
  // Our "destiny" is where we belong, whether we're there or not now.
  FileByNameMap::iterator destiny = files_.lower_bound(&name);
  if (destiny == files_.end()
      || *destiny->first != name) {  // Repeated string comparison, boo hoo.
    File* file = new File(name);
    file->source_id = -1;
    destiny = files_.insert(destiny,
                            FileByNameMap::value_type(&file->name, file));
  }
  return destiny->second;
}

Module::File* Module::FindFile(const char* name) {
  string name_string = name;
  return FindFile(name_string);
}

Module::File* Module::FindExistingFile(const string& name) {
  FileByNameMap::iterator it = files_.find(&name);
  return (it == files_.end()) ? NULL : it->second;
}

void Module::GetFiles(vector<File*>* vec) {
  vec->clear();
  for (FileByNameMap::iterator it = files_.begin(); it != files_.end(); ++it)
    vec->push_back(it->second);
}

void Module::GetStackFrameEntries(vector<StackFrameEntry*>* vec) const {
  vec->clear();
  vec->reserve(stack_frame_entries_.size());
  for (const auto& ent : stack_frame_entries_) {
    vec->push_back(ent.get());
  }
}

void Module::AssignSourceIds() {
  // First, give every source file an id of -1.
  for (FileByNameMap::iterator file_it = files_.begin();
       file_it != files_.end(); ++file_it) {
    file_it->second->source_id = -1;
  }

  // Next, mark all files actually cited by our functions' line number
  // info, by setting each one's source id to zero.
  for (FunctionSet::const_iterator func_it = functions_.begin();
       func_it != functions_.end(); ++func_it) {
    Function* func = *func_it;
    for (vector<Line>::iterator line_it = func->lines.begin();
         line_it != func->lines.end(); ++line_it)
      line_it->file->source_id = 0;
  }

  // Also mark all files cited by inline callsite by setting each one's source
  // id to zero.
  auto markInlineFiles = [](unique_ptr<Inline>& in) {
    // There are some artificial inline functions which don't belong to
    // any file. Those will have file id -1.
    if (in->call_site_file) {
      in->call_site_file->source_id = 0;
    }
  };
  for (auto func : functions_) {
    Inline::InlineDFS(func->inlines, markInlineFiles);
  }

  // Finally, assign source ids to those files that have been marked.
  // We could have just assigned source id numbers while traversing
  // the line numbers, but doing it this way numbers the files in
  // lexicographical order by name, which is neat.
  int next_source_id = 0;
  for (FileByNameMap::iterator file_it = files_.begin();
       file_it != files_.end(); ++file_it) {
    if (!file_it->second->source_id)
      file_it->second->source_id = next_source_id++;
  }
}

void Module::CreateInlineOrigins(
    set<InlineOrigin*, InlineOriginCompare>& inline_origins) {
  // Only add origins that have file and deduplicate origins with same name and
  // file id by doing a DFS.
  auto addInlineOrigins = [&](unique_ptr<Inline>& in) {
    auto it = inline_origins.find(in->origin);
    if (it == inline_origins.end())
      inline_origins.insert(in->origin);
    else
      in->origin = *it;
  };
  for (Function* func : functions_)
    Module::Inline::InlineDFS(func->inlines, addInlineOrigins);
  int next_id = 0;
  for (InlineOrigin* origin : inline_origins) {
    origin->id = next_id++;
  }
}

bool Module::ReportError() {
  fprintf(stderr, "error writing symbol file: %s\n",
          strerror(errno));
  return false;
}

bool Module::WriteRuleMap(const RuleMap& rule_map, std::ostream& stream) {
  for (RuleMap::const_iterator it = rule_map.begin();
       it != rule_map.end(); ++it) {
    if (it != rule_map.begin())
      stream << ' ';
    stream << it->first << ": " << it->second;
  }
  return stream.good();
}

bool Module::AddressIsInModule(Address address) const {
  if (address_ranges_.empty()) {
    return true;
  }
  for (const auto& segment : address_ranges_) {
    if (address >= segment.address &&
        address < segment.address + segment.size) {
      return true;
    }
  }
  return false;
}

bool Module::Write(std::ostream& stream, SymbolData symbol_data, bool preserve_load_address) {
  stream << "MODULE " << os_ << " " << architecture_ << " "
         << id_ << " " << name_ << "\n";
  if (!stream.good())
    return ReportError();

  if (!code_id_.empty()) {
    stream << "INFO CODE_ID " << code_id_ << "\n";
  }

  // load_address is subtracted from each line. If we use zero instead, we
  // preserve the original addresses present in the ELF binary.
  Address load_offset = load_address_;
  if (preserve_load_address) {
    load_offset = 0;
  }

  if (symbol_data & SYMBOLS_AND_FILES) {
    // Get all referenced inline origins.
    set<InlineOrigin*, InlineOriginCompare> inline_origins;
    CreateInlineOrigins(inline_origins);
    AssignSourceIds();

    // Write out files.
    for (FileByNameMap::iterator file_it = files_.begin();
         file_it != files_.end(); ++file_it) {
      File* file = file_it->second;
      if (file->source_id >= 0) {
        stream << "FILE " << file->source_id << " " <<  file->name << "\n";
        if (!stream.good())
          return ReportError();
      }
    }

    // Write out inline origins.
    for (InlineOrigin* origin : inline_origins) {
      stream << "INLINE_ORIGIN " << origin->id << " " << origin->name << "\n";
      if (!stream.good())
        return ReportError();
    }
    // Write out functions and their inlines and lines.
    for (FunctionSet::const_iterator func_it = functions_.begin();
         func_it != functions_.end(); ++func_it) {
      Function* func = *func_it;
      vector<Line>::iterator line_it = func->lines.begin();
      for (auto range_it = func->ranges.cbegin();
           range_it != func->ranges.cend(); ++range_it) {
        stream << "FUNC " << (func->is_multiple ? "m " : "") << hex
               << (range_it->address - load_offset) << " " << range_it->size
               << " " << func->parameter_size << " " << func->name << dec
               << "\n";

        if (!stream.good())
          return ReportError();

        // Write out inlines.
        auto write_inline = [&](unique_ptr<Inline>& in) {
          stream << "INLINE ";
          stream << in->inline_nest_level << " " << in->call_site_line << " "
                 << in->getCallSiteFileID() << " " << in->origin->id << hex;
          for (const Range& r : in->ranges)
            stream << " " << (r.address - load_offset) << " " << r.size;
          stream << dec << "\n";
        };
        Module::Inline::InlineDFS(func->inlines, write_inline);
        if (!stream.good())
          return ReportError();

        while ((line_it != func->lines.end()) &&
               (line_it->address >= range_it->address) &&
               (line_it->address < (range_it->address + range_it->size))) {
          stream << hex
                 << (line_it->address - load_offset) << " "
                 << line_it->size << " "
                 << dec
                 << line_it->number << " "
                 << line_it->file->source_id << "\n";

          if (!stream.good())
            return ReportError();

          ++line_it;
        }
      }
    }

    // Write out 'PUBLIC' records.
    for (ExternSet::const_iterator extern_it = externs_.begin();
         extern_it != externs_.end(); ++extern_it) {
      Extern* ext = extern_it->get();
      stream << "PUBLIC " << (ext->is_multiple ? "m " : "") << hex
             << (ext->address - load_offset) << " 0 " << ext->name << dec
             << "\n";
    }
  }

  if (symbol_data & CFI) {
    // Write out 'STACK CFI INIT' and 'STACK CFI' records.
    for (auto frame_it = stack_frame_entries_.begin();
         frame_it != stack_frame_entries_.end(); ++frame_it) {
      StackFrameEntry* entry = frame_it->get();
      stream << "STACK CFI INIT " << hex
             << (entry->address - load_offset) << " "
             << entry->size << " " << dec;
      if (!stream.good()
          || !WriteRuleMap(entry->initial_rules, stream))
        return ReportError();

      stream << "\n";

      // Write out this entry's delta rules as 'STACK CFI' records.
      for (RuleChangeMap::const_iterator delta_it = entry->rule_changes.begin();
           delta_it != entry->rule_changes.end(); ++delta_it) {
        stream << "STACK CFI " << hex
               << (delta_it->first - load_offset) << " " << dec;
        if (!stream.good()
            || !WriteRuleMap(delta_it->second, stream))
          return ReportError();

        stream << "\n";
      }
    }
  }

  return true;
}

}  // namespace google_breakpad
