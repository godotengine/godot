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

// Implement the DwarfCUToModule class; see dwarf_cu_to_module.h.

// For <inttypes.h> PRI* macros, before anything else might #include it.
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif  /* __STDC_FORMAT_MACROS */

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/dwarf_cu_to_module.h"

#include <assert.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#include "common/string_view.h"
#include "common/dwarf_line_to_module.h"
#include "google_breakpad/common/breakpad_types.h"

namespace google_breakpad {

using std::accumulate;
using std::map;
using std::pair;
using std::sort;
using std::vector;
using std::unique_ptr;

// Data provided by a DWARF specification DIE.
//
// In DWARF, the DIE for a definition may contain a DW_AT_specification
// attribute giving the offset of the corresponding declaration DIE, and
// the definition DIE may omit information given in the declaration. For
// example, it's common for a function's address range to appear only in
// its definition DIE, but its name to appear only in its declaration
// DIE.
//
// The dumper needs to be able to follow DW_AT_specification links to
// bring all this information together in a FUNC record. Conveniently,
// DIEs that are the target of such links have a DW_AT_declaration flag
// set, so we can identify them when we first see them, and record their
// contents for later reference.
//
// A Specification holds information gathered from a declaration DIE that
// we may need if we find a DW_AT_specification link pointing to it.
struct DwarfCUToModule::Specification {
  // The qualified name that can be found by demangling DW_AT_MIPS_linkage_name.
  StringView qualified_name;

  // The name of the enclosing scope, or the empty string if there is none.
  StringView enclosing_name;

  // The name for the specification DIE itself, without any enclosing
  // name components.
  StringView unqualified_name;
};

// An abstract origin -- base definition of an inline function.
struct AbstractOrigin {
  explicit AbstractOrigin(StringView name) : name(name) {}

  StringView name;
};

typedef map<uint64_t, AbstractOrigin> AbstractOriginByOffset;

// Data global to the DWARF-bearing file that is private to the
// DWARF-to-Module process.
struct DwarfCUToModule::FilePrivate {
  // A map from offsets of DIEs within the .debug_info section to
  // Specifications describing those DIEs. Specification references can
  // cross compilation unit boundaries.
  SpecificationByOffset specifications;

  AbstractOriginByOffset origins;

  // Keep a list of forward references from DW_AT_abstract_origin and
  // DW_AT_specification attributes so names can be fixed up.
  std::map<uint64_t, Module::Function*> forward_ref_die_to_func;
};

DwarfCUToModule::FileContext::FileContext(const string& filename,
                                          Module* module,
                                          bool handle_inter_cu_refs)
    : filename_(filename),
      module_(module),
      handle_inter_cu_refs_(handle_inter_cu_refs),
      file_private_(new FilePrivate()) {
}

DwarfCUToModule::FileContext::~FileContext() {
  for (std::vector<uint8_t *>::iterator i = uncompressed_sections_.begin();
        i != uncompressed_sections_.end(); ++i) {
    delete[] *i;
  }
}

void DwarfCUToModule::FileContext::AddSectionToSectionMap(
    const string& name, const uint8_t* contents, uint64_t length) {
  section_map_[name] = std::make_pair(contents, length);
}

void DwarfCUToModule::FileContext::AddManagedSectionToSectionMap(
    const string& name, uint8_t* contents, uint64_t length) {
  section_map_[name] = std::make_pair(contents, length);
  uncompressed_sections_.push_back(contents);
}

void DwarfCUToModule::FileContext::ClearSectionMapForTest() {
  section_map_.clear();
}

const SectionMap&
DwarfCUToModule::FileContext::section_map() const {
  return section_map_;
}

void DwarfCUToModule::FileContext::ClearSpecifications() {
  if (!handle_inter_cu_refs_)
    file_private_->specifications.clear();
}

bool DwarfCUToModule::FileContext::IsUnhandledInterCUReference(
    uint64_t offset, uint64_t compilation_unit_start) const {
  if (handle_inter_cu_refs_)
    return false;
  return offset < compilation_unit_start;
}

// Information global to the particular compilation unit we're
// parsing. This is for data shared across the CU's entire DIE tree,
// and parameters from the code invoking the CU parser.
struct DwarfCUToModule::CUContext {
  CUContext(FileContext* file_context_arg,
            WarningReporter* reporter_arg,
            RangesHandler* ranges_handler_arg,
            uint64_t low_pc,
            uint64_t addr_base)
      : version(0),
        file_context(file_context_arg),
        reporter(reporter_arg),
        ranges_handler(ranges_handler_arg),
        language(Language::CPlusPlus),
        low_pc(low_pc),
        high_pc(0),
        ranges_form(DW_FORM_sec_offset),
        ranges_data(0),
        ranges_base(0),
        addr_base(addr_base),
        str_offsets_base(0) {}

  ~CUContext() {
    for (vector<Module::Function*>::iterator it = functions.begin();
         it != functions.end(); ++it) {
      delete *it;
    }
  };

  // Dwarf version of the source CU.
  uint8_t version;

  // The DWARF-bearing file into which this CU was incorporated.
  FileContext* file_context;

  // For printing error messages.
  WarningReporter* reporter;

  // For reading ranges from the .debug_ranges section
  RangesHandler* ranges_handler;

  // The source language of this compilation unit.
  const Language* language;

  // Addresses covered by this CU. If high_pc_ is non-zero then the CU covers
  // low_pc to high_pc, otherwise ranges_data is non-zero and low_pc represents
  // the base address of the ranges covered by the CU. ranges_data will define
  // the CU's actual ranges.
  uint64_t low_pc;
  uint64_t high_pc;

  // Ranges for this CU are read according to this form.
  enum DwarfForm ranges_form;
  uint64_t ranges_data;

  // Offset into .debug_rngslists where this CU's ranges are stored.
  // Data in DW_FORM_rnglistx is relative to this offset.
  uint64_t ranges_base;

  // Offset into .debug_addr where this CU's addresses are stored. Data in
  // form DW_FORM_addrxX is relative to this offset.
  uint64_t addr_base;

  // Offset into this CU's contribution to .debug_str_offsets.
  uint64_t str_offsets_base;

  // Collect all the data from the CU that a RangeListReader needs to read a
  // range.
  bool AssembleRangeListInfo(
      RangeListReader::CURangesInfo* info) {
    const SectionMap& section_map
        = file_context->section_map();
    info->version_ = version;
    info->base_address_ = low_pc;
    info->ranges_base_ = ranges_base;
    const char* section_name = (version <= 4 ?
                                ".debug_ranges" : ".debug_rnglists");
    SectionMap::const_iterator map_entry
        = GetSectionByName(section_map, section_name);
    if (map_entry == section_map.end()) {
      return false;
    }
    info->buffer_ = map_entry->second.first;
    info->size_ = map_entry->second.second;
    if (version > 4) {
      SectionMap::const_iterator map_entry
          = GetSectionByName(section_map, ".debug_addr");
      if (map_entry == section_map.end()) {
        return false;
      }
      info->addr_buffer_ = map_entry->second.first;
      info->addr_buffer_size_ = map_entry->second.second;
      info->addr_base_ = addr_base;
    }
    return true;
  }

  // The functions defined in this compilation unit. We accumulate
  // them here during parsing. Then, in DwarfCUToModule::Finish, we
  // assign them lines and add them to file_context->module.
  //
  // Destroying this destroys all the functions this vector points to.
  vector<Module::Function*> functions;

  // A map of function pointers to the its forward specification DIE's offset.
  map<Module::Function*, uint64_t> spec_function_offsets;
};

// Information about the context of a particular DIE. This is for
// information that changes as we descend the tree towards the leaves:
// the containing classes/namespaces, etc.
struct DwarfCUToModule::DIEContext {
  // The fully-qualified name of the context. For example, for a
  // tree like:
  //
  // DW_TAG_namespace Foo
  //   DW_TAG_class Bar
  //     DW_TAG_subprogram Baz
  //
  // in a C++ compilation unit, the DIEContext's name for the
  // DW_TAG_subprogram DIE would be "Foo::Bar". The DIEContext's
  // name for the DW_TAG_namespace DIE would be "".
  StringView name;
};

// An abstract base class for all the dumper's DIE handlers.
class DwarfCUToModule::GenericDIEHandler: public DIEHandler {
 public:
  // Create a handler for the DIE at OFFSET whose compilation unit is
  // described by CU_CONTEXT, and whose immediate context is described
  // by PARENT_CONTEXT.
  GenericDIEHandler(CUContext* cu_context, DIEContext* parent_context,
                    uint64_t offset)
      : cu_context_(cu_context),
        parent_context_(parent_context),
        offset_(offset),
        declaration_(false),
        specification_(NULL),
        no_specification(false),
        abstract_origin_(NULL),
        forward_ref_die_offset_(0), specification_offset_(0) { }

  // Derived classes' ProcessAttributeUnsigned can defer to this to
  // handle DW_AT_declaration, or simply not override it.
  void ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data);

  // Derived classes' ProcessAttributeReference can defer to this to
  // handle DW_AT_specification, or simply not override it.
  void ProcessAttributeReference(enum DwarfAttribute attr,
                                 enum DwarfForm form,
                                 uint64_t data);

  // Derived classes' ProcessAttributeReference can defer to this to
  // handle DW_AT_specification, or simply not override it.
  void ProcessAttributeString(enum DwarfAttribute attr,
                              enum DwarfForm form,
                              const string& data);

 protected:
  // Compute and return the fully-qualified name of the DIE. If this
  // DIE is a declaration DIE, to be cited by other DIEs'
  // DW_AT_specification attributes, record its enclosing name and
  // unqualified name in the specification table.
  //
  // Use this from EndAttributes member functions, not ProcessAttribute*
  // functions; only the former can be sure that all the DIE's attributes
  // have been seen.
  //
  // On return, if has_qualified_name is non-NULL, *has_qualified_name is set to
  // true if the DIE includes a fully-qualified name, false otherwise.
  StringView ComputeQualifiedName(bool* has_qualified_name);

  CUContext* cu_context_;
  DIEContext* parent_context_;
  uint64_t offset_;

  // If this DIE has a DW_AT_declaration attribute, this is its value.
  // It is false on DIEs with no DW_AT_declaration attribute.
  bool declaration_;

  // If this DIE has a DW_AT_specification attribute, this is the
  // Specification structure for the DIE the attribute refers to.
  // Otherwise, this is NULL.
  Specification* specification_;

  // If this DIE has DW_AT_specification with offset smaller than this DIE and
  // we can't find that in the specification map.
  bool no_specification;

  // If this DIE has a DW_AT_abstract_origin attribute, this is the
  // AbstractOrigin structure for the DIE the attribute refers to.
  // Otherwise, this is NULL.
  const AbstractOrigin* abstract_origin_;

  // If this DIE has a DW_AT_specification or DW_AT_abstract_origin and it is a
  // forward reference, no Specification will be available. Track the reference
  // to be fixed up when the DIE is parsed.
  uint64_t forward_ref_die_offset_;

  // The root offset of Specification or abstract origin.
  uint64_t specification_offset_;

  // The value of the DW_AT_name attribute, or the empty string if the
  // DIE has no such attribute.
  StringView name_attribute_;

  // The demangled value of the DW_AT_MIPS_linkage_name attribute, or the empty
  // string if the DIE has no such attribute or its content could not be
  // demangled.
  StringView demangled_name_;

  // The non-demangled value of the DW_AT_MIPS_linkage_name attribute,
  // it its content count not be demangled.
  StringView raw_name_;
};

void DwarfCUToModule::GenericDIEHandler::ProcessAttributeUnsigned(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    uint64_t data) {
  switch (attr) {
    case DW_AT_declaration: declaration_ = (data != 0); break;
    default: break;
  }
}

void DwarfCUToModule::GenericDIEHandler::ProcessAttributeReference(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    uint64_t data) {
  switch (attr) {
    case DW_AT_specification: {
      FileContext* file_context = cu_context_->file_context;
      if (file_context->IsUnhandledInterCUReference(
              data, cu_context_->reporter->cu_offset())) {
        cu_context_->reporter->UnhandledInterCUReference(offset_, data);
        break;
      }
      // Find the Specification to which this attribute refers, and
      // set specification_ appropriately. We could do more processing
      // here, but it's better to leave the real work to our
      // EndAttribute member function, at which point we know we have
      // seen all the DIE's attributes.
      SpecificationByOffset* specifications =
          &file_context->file_private_->specifications;
      SpecificationByOffset::iterator spec = specifications->find(data);
      if (spec != specifications->end()) {
        specification_ = &spec->second;
      } else if (data > offset_) {
        forward_ref_die_offset_ = data;
      } else {
        no_specification = true;
      }
      specification_offset_ = data;
      break;
    }
    case DW_AT_abstract_origin: {
      const AbstractOriginByOffset& origins =
          cu_context_->file_context->file_private_->origins;
      AbstractOriginByOffset::const_iterator origin = origins.find(data);
      if (origin != origins.end()) {
        abstract_origin_ = &(origin->second);
      } else if (data > offset_) {
        forward_ref_die_offset_ = data;
      }
      specification_offset_ = data;
      break;
    }
    default: break;
  }
}

void DwarfCUToModule::GenericDIEHandler::ProcessAttributeString(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    const string& data) {
  switch (attr) {
    case DW_AT_name:
      name_attribute_ =
          cu_context_->file_context->module_->AddStringToPool(data);
      break;
    case DW_AT_MIPS_linkage_name:
    case DW_AT_linkage_name: {
      string demangled;
      Language::DemangleResult result =
          cu_context_->language->DemangleName(data, &demangled);
      switch (result) {
        case Language::kDemangleSuccess:
          demangled_name_ =
              cu_context_->file_context->module_->AddStringToPool(demangled);
          break;

        case Language::kDemangleFailure:
          cu_context_->reporter->DemangleError(data);
          // fallthrough
        case Language::kDontDemangle:
          demangled_name_ = StringView();
          raw_name_ = cu_context_->file_context->module_->AddStringToPool(data);
          break;
      }
      break;
    }
    default: break;
  }
}

StringView DwarfCUToModule::GenericDIEHandler::ComputeQualifiedName(
    bool* has_qualified_name) {
  // Use the demangled name, if one is available. Demangled names are
  // preferable to those inferred from the DWARF structure because they
  // include argument types.
  StringView* qualified_name = nullptr;
  if (!demangled_name_.empty()) {
    // Found it is this DIE.
    qualified_name = &demangled_name_;
  } else if (specification_ && !specification_->qualified_name.empty()) {
    // Found it on the specification.
    qualified_name = &specification_->qualified_name;
  }

  StringView* unqualified_name = nullptr;
  StringView* enclosing_name = nullptr;
  if (!qualified_name) {
    if (has_qualified_name) {
      // dSYMs built with -gmlt do not include the DW_AT_linkage_name
      // with the unmangled symbol, but rather include it in the
      // LC_SYMTAB STABS, which end up in the externs of the module.
      //
      // Remember this so the Module can copy over the extern name later.
      *has_qualified_name = false;
    }

    // Find the unqualified name. If the DIE has its own DW_AT_name
    // attribute, then use that; otherwise, check the specification.
    if (!name_attribute_.empty()) {
      unqualified_name = &name_attribute_;
    } else if (specification_) {
      unqualified_name = &specification_->unqualified_name;
    } else if (!raw_name_.empty()) {
      unqualified_name = &raw_name_;
    }

    // Find the name of the enclosing context. If this DIE has a
    // specification, it's the specification's enclosing context that
    // counts; otherwise, use this DIE's context.
    if (specification_) {
      enclosing_name = &specification_->enclosing_name;
    } else if (parent_context_) {
      enclosing_name = &parent_context_->name;
    }
  } else {
    if (has_qualified_name) {
      *has_qualified_name = true;
    }
  }

  // Prepare the return value before upcoming mutations possibly invalidate the
  // existing pointers.
  string return_value;
  if (qualified_name) {
    return_value = qualified_name->str();
  } else if (unqualified_name && enclosing_name) {
    // Combine the enclosing name and unqualified name to produce our
    // own fully-qualified name.
    return_value = cu_context_->language->MakeQualifiedName(
        enclosing_name->str(), unqualified_name->str());
  }

  // If this DIE was marked as a declaration, record its names in the
  // specification table.
  if ((declaration_ && qualified_name) ||
      (unqualified_name && enclosing_name)) {
    Specification spec;
    if (qualified_name) {
      spec.qualified_name = *qualified_name;
    } else {
      spec.enclosing_name = *enclosing_name;
      spec.unqualified_name = *unqualified_name;
    }
    cu_context_->file_context->file_private_->specifications[offset_] = spec;
  }

  return cu_context_->file_context->module_->AddStringToPool(return_value);
}

static bool IsEmptyRange(const vector<Module::Range>& ranges) {
  uint64_t size = accumulate(ranges.cbegin(), ranges.cend(), 0,
    [](uint64_t total, Module::Range entry) {
      return total + entry.size;
    }
  );

  return size == 0;
}

// A handler for DW_TAG_lexical_block DIEs.
class DwarfCUToModule::LexicalBlockHandler : public GenericDIEHandler {
 public:
  LexicalBlockHandler(CUContext* cu_context,
                      uint64_t offset,
                      int inline_nest_level,
                      vector<unique_ptr<Module::Inline>>& inlines)
      : GenericDIEHandler(cu_context, nullptr, offset),
        inline_nest_level_(inline_nest_level),
        inlines_(inlines) {}

  DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag);
  bool EndAttributes() { return true; }
  void Finish();

 private:
  int inline_nest_level_;
  // A vector of inlines in the same nest level. It's owned by its parent
  // function/inline. At Finish(), add this inline into the vector.
  vector<unique_ptr<Module::Inline>>& inlines_;
  // A vector of child inlines.
  vector<unique_ptr<Module::Inline>> child_inlines_;
};

// A handler for DW_TAG_inlined_subroutine DIEs.
class DwarfCUToModule::InlineHandler : public GenericDIEHandler {
 public:
  InlineHandler(CUContext* cu_context,
                DIEContext* parent_context,
                uint64_t offset,
                int inline_nest_level,
                vector<unique_ptr<Module::Inline>>& inlines)
      : GenericDIEHandler(cu_context, parent_context, offset),
        low_pc_(0),
        high_pc_(0),
        high_pc_form_(DW_FORM_addr),
        ranges_form_(DW_FORM_sec_offset),
        ranges_data_(0),
        call_site_line_(0),
        inline_nest_level_(inline_nest_level),
        has_range_data_(false),
        inlines_(inlines) {}

  void ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data);
  DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag);
  bool EndAttributes();
  void Finish();

 private:
  // The fully-qualified name, as derived from name_attribute_,
  // specification_, parent_context_. Computed in EndAttributes.
  StringView name_;
  uint64_t low_pc_;            // DW_AT_low_pc
  uint64_t high_pc_;           // DW_AT_high_pc
  DwarfForm high_pc_form_;     // DW_AT_high_pc can be length or address.
  DwarfForm ranges_form_;      // DW_FORM_sec_offset or DW_FORM_rnglistx
  uint64_t ranges_data_;       // DW_AT_ranges
  int call_site_line_;         // DW_AT_call_line
  int call_site_file_id_;      // DW_AT_call_file
  int inline_nest_level_;
  bool has_range_data_;
  // A vector of inlines in the same nest level. It's owned by its parent
  // function/inline. At Finish(), add this inline into the vector.
  vector<unique_ptr<Module::Inline>>& inlines_;
  // A vector of child inlines.
  vector<unique_ptr<Module::Inline>> child_inlines_;
};

void DwarfCUToModule::InlineHandler::ProcessAttributeUnsigned(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    uint64_t data) {
  switch (attr) {
    case DW_AT_low_pc:
      low_pc_ = data;
      break;
    case DW_AT_high_pc:
      high_pc_form_ = form;
      high_pc_ = data;
      break;
    case DW_AT_ranges:
      has_range_data_ = true;
      ranges_data_ = data;
      ranges_form_ = form;
      break;
    case DW_AT_call_line:
      call_site_line_ = data;
      break;
    case DW_AT_call_file:
      call_site_file_id_ = data;
      break;
    default:
      GenericDIEHandler::ProcessAttributeUnsigned(attr, form, data);
      break;
  }
}

DIEHandler* DwarfCUToModule::InlineHandler::FindChildHandler(
    uint64_t offset,
    enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_inlined_subroutine:
      return new InlineHandler(cu_context_, nullptr, offset,
                               inline_nest_level_ + 1, child_inlines_);
    case DW_TAG_lexical_block:
      return new LexicalBlockHandler(cu_context_, offset,
                                     inline_nest_level_ + 1, child_inlines_);
    default:
      return NULL;
  }
}

bool DwarfCUToModule::InlineHandler::EndAttributes() {
  if (abstract_origin_)
    name_ = abstract_origin_->name;
  if (name_.empty()) {
    // We haven't seen the abstract origin yet, which might appears later and we
    // will fix the name after calling
    // InlineOriginMap::GetOrCreateInlineOrigin with right name.
    name_ =
        cu_context_->file_context->module_->AddStringToPool("<name omitted>");
  }
  return true;
}

void DwarfCUToModule::InlineHandler::Finish() {
  vector<Module::Range> ranges;

  if (!has_range_data_) {
    if (high_pc_form_ != DW_FORM_addr &&
        high_pc_form_ != DW_FORM_GNU_addr_index &&
        high_pc_form_ != DW_FORM_addrx &&
        high_pc_form_ != DW_FORM_addrx1 &&
        high_pc_form_ != DW_FORM_addrx2 &&
        high_pc_form_ != DW_FORM_addrx3 &&
        high_pc_form_ != DW_FORM_addrx4) {
      high_pc_ += low_pc_;
    }

    Module::Range range(low_pc_, high_pc_ - low_pc_);
    ranges.push_back(range);
  } else {
    RangesHandler* ranges_handler = cu_context_->ranges_handler;
    if (ranges_handler) {
      RangeListReader::CURangesInfo cu_info;
      if (cu_context_->AssembleRangeListInfo(&cu_info)) {
        if (!ranges_handler->ReadRanges(ranges_form_, ranges_data_,
                                        &cu_info, &ranges)) {
          ranges.clear();
          cu_context_->reporter->MalformedRangeList(ranges_data_);
        }
      } else {
        cu_context_->reporter->MissingRanges();
      }
    }
  }

  // Ignore DW_TAG_inlined_subroutine with empty range.
  if (ranges.empty()) {
    return;
  }

  // Every DW_TAG_inlined_subroutine should have a DW_AT_abstract_origin.
  assert(specification_offset_ != 0);

  Module::InlineOriginMap& inline_origin_map =
      cu_context_->file_context->module_
          ->inline_origin_maps[cu_context_->file_context->filename_];
  inline_origin_map.SetReference(specification_offset_, specification_offset_);
  Module::InlineOrigin* origin =
      inline_origin_map.GetOrCreateInlineOrigin(specification_offset_, name_);
  unique_ptr<Module::Inline> in(
      new Module::Inline(origin, ranges, call_site_line_, call_site_file_id_,
                         inline_nest_level_, std::move(child_inlines_)));
  inlines_.push_back(std::move(in));
}

DIEHandler* DwarfCUToModule::LexicalBlockHandler::FindChildHandler(
    uint64_t offset,
    enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_inlined_subroutine:
      return new InlineHandler(cu_context_, nullptr, offset, inline_nest_level_,
                               child_inlines_);
    case DW_TAG_lexical_block:
      return new LexicalBlockHandler(cu_context_, offset, inline_nest_level_,
                                     child_inlines_);
    default:
      return nullptr;
  }
}

void DwarfCUToModule::LexicalBlockHandler::Finish() {
  // Insert child inlines inside the lexical block into the inline vector from
  // parent as if the block does not exit.
  inlines_.insert(inlines_.end(),
                  std::make_move_iterator(child_inlines_.begin()),
                  std::make_move_iterator(child_inlines_.end()));
}

// A handler for DIEs that contain functions and contribute a
// component to their names: namespaces, classes, etc.
class DwarfCUToModule::NamedScopeHandler: public GenericDIEHandler {
 public:
  NamedScopeHandler(CUContext* cu_context,
                    DIEContext* parent_context,
                    uint64_t offset,
                    bool handle_inline)
      : GenericDIEHandler(cu_context, parent_context, offset),
        handle_inline_(handle_inline) {}
  bool EndAttributes();
  DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag);

 private:
  DIEContext child_context_; // A context for our children.
  bool handle_inline_;
};

// A handler class for DW_TAG_subprogram DIEs.
class DwarfCUToModule::FuncHandler: public GenericDIEHandler {
 public:
  FuncHandler(CUContext* cu_context,
              DIEContext* parent_context,
              uint64_t offset,
              bool handle_inline)
      : GenericDIEHandler(cu_context, parent_context, offset),
        low_pc_(0),
        high_pc_(0),
        high_pc_form_(DW_FORM_addr),
        ranges_form_(DW_FORM_sec_offset),
        ranges_data_(0),
        inline_(false),
        handle_inline_(handle_inline),
        has_qualified_name_(false),
        has_range_data_(false) {}

  void ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                enum DwarfForm form,
                                uint64_t data);
  void ProcessAttributeSigned(enum DwarfAttribute attr,
                              enum DwarfForm form,
                              int64_t data);
  DIEHandler* FindChildHandler(uint64_t offset, enum DwarfTag tag);
  bool EndAttributes();
  void Finish();

 private:
  // The fully-qualified name, as derived from name_attribute_,
  // specification_, parent_context_.  Computed in EndAttributes.
  StringView name_;
  uint64_t low_pc_, high_pc_; // DW_AT_low_pc, DW_AT_high_pc
  DwarfForm high_pc_form_; // DW_AT_high_pc can be length or address.
  DwarfForm ranges_form_; // DW_FORM_sec_offset or DW_FORM_rnglistx
  uint64_t ranges_data_;  // DW_AT_ranges
  bool inline_;
  vector<unique_ptr<Module::Inline>> child_inlines_;
  bool handle_inline_;
  bool has_qualified_name_;
  bool has_range_data_;
  DIEContext child_context_; // A context for our children.
};

void DwarfCUToModule::FuncHandler::ProcessAttributeUnsigned(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    uint64_t data) {
  switch (attr) {
    // If this attribute is present at all --- even if its value is
    // DW_INL_not_inlined --- then GCC may cite it as someone else's
    // DW_AT_abstract_origin attribute.
    case DW_AT_inline:      inline_  = true; break;

    case DW_AT_low_pc:      low_pc_  = data; break;
    case DW_AT_high_pc:
      high_pc_form_ = form;
      high_pc_ = data;
      break;
    case DW_AT_ranges:
      has_range_data_ = true;
      ranges_data_ = data;
      ranges_form_ = form;
      break;
    default:
      GenericDIEHandler::ProcessAttributeUnsigned(attr, form, data);
      break;
  }
}

void DwarfCUToModule::FuncHandler::ProcessAttributeSigned(
    enum DwarfAttribute attr,
    enum DwarfForm form,
    int64_t data) {
  switch (attr) {
    // If this attribute is present at all --- even if its value is
    // DW_INL_not_inlined --- then GCC may cite it as someone else's
    // DW_AT_abstract_origin attribute.
    case DW_AT_inline:      inline_  = true; break;

    default:
      break;
  }
}

DIEHandler* DwarfCUToModule::FuncHandler::FindChildHandler(
    uint64_t offset,
    enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_inlined_subroutine:
      if (handle_inline_)
        return new InlineHandler(cu_context_, nullptr, offset, 0,
                                 child_inlines_);
    case DW_TAG_class_type:
    case DW_TAG_structure_type:
    case DW_TAG_union_type:
      return new NamedScopeHandler(cu_context_, &child_context_, offset,
                                   handle_inline_);
    case DW_TAG_lexical_block:
      if (handle_inline_)
        return new LexicalBlockHandler(cu_context_, offset, 0, child_inlines_);
    default:
      return NULL;
  }
}

bool DwarfCUToModule::FuncHandler::EndAttributes() {
  // Compute our name, and record a specification, if appropriate.
  name_ = ComputeQualifiedName(&has_qualified_name_);
  if (name_.empty() && abstract_origin_) {
    name_ = abstract_origin_->name;
  }
  child_context_.name = name_;
  if (name_.empty() && no_specification) {
    cu_context_->reporter->UnknownSpecification(offset_, specification_offset_);
  }
  return true;
}

void DwarfCUToModule::FuncHandler::Finish() {
  vector<Module::Range> ranges;

  // Check if this DIE was one of the forward references that was not able
  // to be processed, and fix up the name of the appropriate Module::Function.
  // "name_" will have already been fixed up in EndAttributes().
  if (!name_.empty()) {
    auto iter =
        cu_context_->file_context->file_private_->forward_ref_die_to_func.find(
            offset_);
    if (iter !=
        cu_context_->file_context->file_private_->forward_ref_die_to_func.end())
      iter->second->name = name_;
  }

  if (!has_range_data_) {
    // Make high_pc_ an address, if it isn't already.
    if (high_pc_form_ != DW_FORM_addr &&
        high_pc_form_ != DW_FORM_GNU_addr_index &&
        high_pc_form_ != DW_FORM_addrx &&
        high_pc_form_ != DW_FORM_addrx1 &&
        high_pc_form_ != DW_FORM_addrx2 &&
        high_pc_form_ != DW_FORM_addrx3 &&
        high_pc_form_ != DW_FORM_addrx4) {
      high_pc_ += low_pc_;
    }

    Module::Range range(low_pc_, high_pc_ - low_pc_);
    ranges.push_back(range);
  } else {
    RangesHandler* ranges_handler = cu_context_->ranges_handler;
    if (ranges_handler) {
      RangeListReader::CURangesInfo cu_info;
      if (cu_context_->AssembleRangeListInfo(&cu_info)) {
        if (!ranges_handler->ReadRanges(ranges_form_, ranges_data_,
                                        &cu_info, &ranges)) {
          ranges.clear();
          cu_context_->reporter->MalformedRangeList(ranges_data_);
        }
      } else {
        cu_context_->reporter->MissingRanges();
      }
    }
  }

  StringView name_omitted =
      cu_context_->file_context->module_->AddStringToPool("<name omitted>");
  bool empty_range = IsEmptyRange(ranges);
  // Did we collect the information we need?  Not all DWARF function
  // entries are non-empty (for example, inlined functions that were never
  // used), but all the ones we're interested in cover a non-empty range of
  // bytes.
  if (!empty_range) {
    low_pc_ = ranges.front().address;
    // Malformed DWARF may omit the name, but all Module::Functions must
    // have names.
    StringView name = name_.empty() ? name_omitted : name_;
    // Create a Module::Function based on the data we've gathered, and
    // add it to the functions_ list.
    scoped_ptr<Module::Function> func(new Module::Function(name, low_pc_));
    func->ranges = ranges;
    func->parameter_size = 0;
    // If the name was unqualified, prefer the Extern name if there's a mismatch
    // (the Extern name will be fully-qualified in that case).
    func->prefer_extern_name = !has_qualified_name_;
    if (func->address) {
      // If the function address is zero this is a sign that this function
      // description is just empty debug data and should just be discarded.
      cu_context_->functions.push_back(func.release());
      if (forward_ref_die_offset_ != 0) {
        cu_context_->file_context->file_private_
            ->forward_ref_die_to_func[forward_ref_die_offset_] =
            cu_context_->functions.back();

        cu_context_->spec_function_offsets[cu_context_->functions.back()] =
            forward_ref_die_offset_;
      }

      cu_context_->functions.back()->inlines.swap(child_inlines_);
    }
  } else if (inline_) {
    AbstractOrigin origin(name_);
    cu_context_->file_context->file_private_->origins.insert({offset_, origin});
  }

  // Only keep track of DW_TAG_subprogram which have the attributes we are
  // interested.
  if (handle_inline_ && (!empty_range || inline_)) {
    StringView name = name_.empty() ? name_omitted : name_;
    uint64_t offset =
        specification_offset_ != 0 ? specification_offset_ : offset_;
    Module::InlineOriginMap& inline_origin_map =
        cu_context_->file_context->module_
            ->inline_origin_maps[cu_context_->file_context->filename_];
    inline_origin_map.SetReference(offset_, offset);
    inline_origin_map.GetOrCreateInlineOrigin(offset_, name);
  }
}

bool DwarfCUToModule::NamedScopeHandler::EndAttributes() {
  child_context_.name = ComputeQualifiedName(NULL);
  if (child_context_.name.empty() && no_specification) {
    cu_context_->reporter->UnknownSpecification(offset_, specification_offset_);
  }
  return true;
}

DIEHandler* DwarfCUToModule::NamedScopeHandler::FindChildHandler(
    uint64_t offset,
    enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_subprogram:
      return new FuncHandler(cu_context_, &child_context_, offset,
                             handle_inline_);
    case DW_TAG_namespace:
    case DW_TAG_class_type:
    case DW_TAG_structure_type:
    case DW_TAG_union_type:
      return new NamedScopeHandler(cu_context_, &child_context_, offset,
                                   handle_inline_);
    default:
      return NULL;
  }
}

void DwarfCUToModule::WarningReporter::CUHeading() {
  if (printed_cu_header_)
    return;
  fprintf(stderr, "%s: in compilation unit '%s' (offset 0x%" PRIx64 "):\n",
          filename_.c_str(), cu_name_.c_str(), cu_offset_);
  printed_cu_header_ = true;
}

void DwarfCUToModule::WarningReporter::UnknownSpecification(uint64_t offset,
                                                            uint64_t target) {
  CUHeading();
  fprintf(stderr, "%s: the DIE at offset 0x%" PRIx64 " has a "
          "DW_AT_specification attribute referring to the DIE at offset 0x%"
          PRIx64 ", which was not marked as a declaration\n",
          filename_.c_str(), offset, target);
}

void DwarfCUToModule::WarningReporter::UnknownAbstractOrigin(uint64_t offset,
                                                             uint64_t target) {
  CUHeading();
  fprintf(stderr, "%s: the DIE at offset 0x%" PRIx64 " has a "
          "DW_AT_abstract_origin attribute referring to the DIE at offset 0x%"
          PRIx64 ", which was not marked as an inline\n",
          filename_.c_str(), offset, target);
}

void DwarfCUToModule::WarningReporter::MissingSection(const string& name) {
  CUHeading();
  fprintf(stderr, "%s: warning: couldn't find DWARF '%s' section\n",
          filename_.c_str(), name.c_str());
}

void DwarfCUToModule::WarningReporter::BadLineInfoOffset(uint64_t offset) {
  CUHeading();
  fprintf(stderr, "%s: warning: line number data offset beyond end"
          " of '.debug_line' section\n",
          filename_.c_str());
}

void DwarfCUToModule::WarningReporter::UncoveredHeading() {
  if (printed_unpaired_header_)
    return;
  CUHeading();
  fprintf(stderr, "%s: warning: skipping unpaired lines/functions:\n",
          filename_.c_str());
  printed_unpaired_header_ = true;
}

void DwarfCUToModule::WarningReporter::UncoveredFunction(
    const Module::Function& function) {
  if (!uncovered_warnings_enabled_)
    return;
  UncoveredHeading();
  fprintf(stderr, "    function%s: %s\n",
          IsEmptyRange(function.ranges) ? " (zero-length)" : "",
          function.name.str().c_str());
}

void DwarfCUToModule::WarningReporter::UncoveredLine(const Module::Line& line) {
  if (!uncovered_warnings_enabled_)
    return;
  UncoveredHeading();
  fprintf(stderr, "    line%s: %s:%d at 0x%" PRIx64 "\n",
          (line.size == 0 ? " (zero-length)" : ""),
          line.file->name.c_str(), line.number, line.address);
}

void DwarfCUToModule::WarningReporter::UnnamedFunction(uint64_t offset) {
  CUHeading();
  fprintf(stderr, "%s: warning: function at offset 0x%" PRIx64 " has no name\n",
          filename_.c_str(), offset);
}

void DwarfCUToModule::WarningReporter::DemangleError(const string& input) {
  CUHeading();
  fprintf(stderr, "%s: warning: failed to demangle %s\n",
          filename_.c_str(), input.c_str());
}

void DwarfCUToModule::WarningReporter::UnhandledInterCUReference(
    uint64_t offset, uint64_t target) {
  CUHeading();
  fprintf(stderr, "%s: warning: the DIE at offset 0x%" PRIx64 " has a "
                  "DW_FORM_ref_addr attribute with an inter-CU reference to "
                  "0x%" PRIx64 ", but inter-CU reference handling is turned "
                  " off.\n", filename_.c_str(), offset, target);
}

void DwarfCUToModule::WarningReporter::MalformedRangeList(uint64_t offset) {
  CUHeading();
  fprintf(stderr, "%s: warning: the range list at offset 0x%" PRIx64 " falls "
                  " out of the .debug_ranges section.\n",
                  filename_.c_str(), offset);
}

void DwarfCUToModule::WarningReporter::MissingRanges() {
  CUHeading();
  fprintf(stderr, "%s: warning: A DW_AT_ranges attribute was encountered but "
                  "the .debug_ranges section is missing.\n", filename_.c_str());
}

DwarfCUToModule::DwarfCUToModule(FileContext* file_context,
                                 LineToModuleHandler* line_reader,
                                 RangesHandler* ranges_handler,
                                 WarningReporter* reporter,
                                 bool handle_inline,
                                 uint64_t low_pc,
                                 uint64_t addr_base,
                                 bool has_source_line_info,
                                 uint64_t source_line_offset)
    : RootDIEHandler(handle_inline),
      line_reader_(line_reader),
      cu_context_(new CUContext(file_context,
                                reporter,
                                ranges_handler,
                                low_pc,
                                addr_base)),
      child_context_(new DIEContext()),
      has_source_line_info_(has_source_line_info),
      source_line_offset_(source_line_offset) {}

DwarfCUToModule::~DwarfCUToModule() {
}

void DwarfCUToModule::ProcessAttributeSigned(enum DwarfAttribute attr,
                                             enum DwarfForm form,
                                             int64_t data) {
  switch (attr) {
    case DW_AT_language: // source language of this CU
      SetLanguage(static_cast<DwarfLanguage>(data));
      break;
    default:
      break;
  }
}

void DwarfCUToModule::ProcessAttributeUnsigned(enum DwarfAttribute attr,
                                               enum DwarfForm form,
                                               uint64_t data) {
  switch (attr) {
    case DW_AT_stmt_list: // Line number information.
      has_source_line_info_ = true;
      source_line_offset_ = data;
      break;
    case DW_AT_language: // source language of this CU
      SetLanguage(static_cast<DwarfLanguage>(data));
      break;
    case DW_AT_low_pc:
      cu_context_->low_pc  = data;
      break;
    case DW_AT_high_pc:
      cu_context_->high_pc  = data;
      break;
    case DW_AT_ranges:
      cu_context_->ranges_data = data;
      cu_context_->ranges_form = form;
      break;
    case DW_AT_rnglists_base:
      cu_context_->ranges_base = data;
      break;
    case DW_AT_addr_base:
    case DW_AT_GNU_addr_base:
      cu_context_->addr_base = data;
      break;
    case DW_AT_str_offsets_base:
      cu_context_->str_offsets_base = data;
      break;
    default:
      break;
  }
}

void DwarfCUToModule::ProcessAttributeString(enum DwarfAttribute attr,
                                             enum DwarfForm form,
                                            const string& data) {
  switch (attr) {
    case DW_AT_name:
      cu_context_->reporter->SetCUName(data);
      break;
    case DW_AT_comp_dir:
      line_reader_->StartCompilationUnit(data);
      break;
    default:
      break;
  }
}

bool DwarfCUToModule::EndAttributes() {
  return true;
}

DIEHandler* DwarfCUToModule::FindChildHandler(
    uint64_t offset,
    enum DwarfTag tag) {
  switch (tag) {
    case DW_TAG_subprogram:
      return new FuncHandler(cu_context_.get(), child_context_.get(), offset,
                             handle_inline);
    case DW_TAG_namespace:
    case DW_TAG_class_type:
    case DW_TAG_structure_type:
    case DW_TAG_union_type:
    case DW_TAG_module:
      return new NamedScopeHandler(cu_context_.get(), child_context_.get(),
                                   offset, handle_inline);
    default:
      return NULL;
  }
}

void DwarfCUToModule::SetLanguage(DwarfLanguage language) {
  switch (language) {
    case DW_LANG_Java:
      cu_context_->language = Language::Java;
      break;

    case DW_LANG_Swift:
      cu_context_->language = Language::Swift;
      break;

    case DW_LANG_Rust:
      cu_context_->language = Language::Rust;
      break;

    // DWARF has no generic language code for assembly language; this is
    // what the GNU toolchain uses.
    case DW_LANG_Mips_Assembler:
      cu_context_->language = Language::Assembler;
      break;

    // C++ covers so many cases that it probably has some way to cope
    // with whatever the other languages throw at us. So make it the
    // default.
    //
    // Objective C and Objective C++ seem to create entries for
    // methods whose DW_AT_name values are already fully-qualified:
    // "-[Classname method:]".  These appear at the top level.
    //
    // DWARF data for C should never include namespaces or functions
    // nested in struct types, but if it ever does, then C++'s
    // notation is probably not a bad choice for that.
    default:
    case DW_LANG_ObjC:
    case DW_LANG_ObjC_plus_plus:
    case DW_LANG_C:
    case DW_LANG_C89:
    case DW_LANG_C99:
    case DW_LANG_C_plus_plus:
      cu_context_->language = Language::CPlusPlus;
      break;
  }
}

void DwarfCUToModule::ReadSourceLines(uint64_t offset) {
  const SectionMap& section_map
      = cu_context_->file_context->section_map();
  SectionMap::const_iterator map_entry
      = GetSectionByName(section_map, ".debug_line");
  if (map_entry == section_map.end()) {
    cu_context_->reporter->MissingSection(".debug_line");
    return;
  }
  const uint8_t* line_section_start = map_entry->second.first + offset;
  uint64_t line_section_length = map_entry->second.second;
  if (offset >= line_section_length) {
    cu_context_->reporter->BadLineInfoOffset(offset);
    return;
  }
  line_section_length -= offset;
  // When reading line tables, string sections are never needed for dwarf4, and
  // may or may not be needed by dwarf5, so no error if they are missing.
  const uint8_t* string_section_start = nullptr;
  uint64_t string_section_length = 0;
  map_entry = GetSectionByName(section_map, ".debug_str");
  if (map_entry != section_map.end()) {
    string_section_start = map_entry->second.first;
    string_section_length = map_entry->second.second;
  }
  const uint8_t* line_string_section_start = nullptr;
  uint64_t line_string_section_length = 0;
  map_entry = GetSectionByName(section_map, ".debug_line_str");
  if (map_entry != section_map.end()) {
    line_string_section_start = map_entry->second.first;
    line_string_section_length = map_entry->second.second;
  }
  line_reader_->ReadProgram(
      line_section_start, line_section_length,
      string_section_start, string_section_length,
      line_string_section_start, line_string_section_length,
      cu_context_->file_context->module_, &lines_, &files_);
}

namespace {
class FunctionRange {
 public:
  FunctionRange(const Module::Range& range, Module::Function* function) :
      address(range.address), size(range.size), function(function) { }

  void AddLine(Module::Line& line) {
    function->lines.push_back(line);
  }

  Module::Address address;
  Module::Address size;
  Module::Function* function;
};

// Fills an array of ranges with pointers to the functions which owns
// them. The array is sorted in ascending order and the ranges are non
// empty and non-overlapping.

static void FillSortedFunctionRanges(vector<FunctionRange>& dest_ranges,
                                     vector<Module::Function*>* functions) {
  for (vector<Module::Function*>::const_iterator func_it = functions->cbegin();
       func_it != functions->cend();
       func_it++)
  {
    Module::Function* func = *func_it;
    vector<Module::Range>& ranges = func->ranges;
    for (vector<Module::Range>::const_iterator ranges_it = ranges.cbegin();
         ranges_it != ranges.cend();
         ++ranges_it) {
      FunctionRange range(*ranges_it, func);
      if (range.size != 0) {
          dest_ranges.push_back(range);
      }
    }
  }

  sort(dest_ranges.begin(), dest_ranges.end(),
    [](const FunctionRange& fr1, const FunctionRange& fr2) {
      return fr1.address < fr2.address;
    }
  );
}

// Return true if ADDRESS falls within the range of ITEM.
template <class T>
inline bool within(const T& item, Module::Address address) {
  // Because Module::Address is unsigned, and unsigned arithmetic
  // wraps around, this will be false if ADDRESS falls before the
  // start of ITEM, or if it falls after ITEM's end.
  return address - item.address < item.size;
}
}

void DwarfCUToModule::AssignLinesToFunctions() {
  vector<Module::Function*>* functions = &cu_context_->functions;
  WarningReporter* reporter = cu_context_->reporter;

  // This would be simpler if we assumed that source line entries
  // don't cross function boundaries.  However, there's no real reason
  // to assume that (say) a series of function definitions on the same
  // line wouldn't get coalesced into one line number entry.  The
  // DWARF spec certainly makes no such promises.
  //
  // So treat the functions and lines as peers, and take the trouble
  // to compute their ranges' intersections precisely.  In any case,
  // the hair here is a constant factor for performance; the
  // complexity from here on out is linear.

  // Put both our functions and lines in order by address.
  std::sort(functions->begin(), functions->end(),
            Module::Function::CompareByAddress);
  std::sort(lines_.begin(), lines_.end(), Module::Line::CompareByAddress);

  // The last line that we used any piece of.  We use this only for
  // generating warnings.
  const Module::Line* last_line_used = NULL;

  // The last function and line we warned about --- so we can avoid
  // doing so more than once.
  const Module::Function* last_function_cited = NULL;
  const Module::Line* last_line_cited = NULL;

  // Prepare a sorted list of ranges with range-to-function mapping
  vector<FunctionRange> sorted_ranges;
  FillSortedFunctionRanges(sorted_ranges, functions);

  // Make a single pass through both the range and line vectors from lower to
  // higher addresses, populating each range's function lines vector with lines
  // from our lines_ vector that fall within the range.
  vector<FunctionRange>::iterator range_it = sorted_ranges.begin();
  vector<Module::Line>::const_iterator line_it = lines_.begin();

  Module::Address current;

  // Pointers to the referents of func_it and line_it, or NULL if the
  // iterator is at the end of the sequence.
  FunctionRange* range;
  const Module::Line* line;

  // Start current at the beginning of the first line or function,
  // whichever is earlier.
  if (range_it != sorted_ranges.end() && line_it != lines_.end()) {
    range = &*range_it;
    line = &*line_it;
    current = std::min(range->address, line->address);
  } else if (line_it != lines_.end()) {
    range = NULL;
    line = &*line_it;
    current = line->address;
  } else if (range_it != sorted_ranges.end()) {
    range = &*range_it;
    line = NULL;
    current = range->address;
  } else {
    return;
  }

  // Some dwarf producers handle linker-removed functions by using -1 as a
  // tombstone in the line table. So the end marker can be -1.
  if (current == Module::kMaxAddress)
    return;

  while (range || line) {
    // This loop has two invariants that hold at the top.
    //
    // First, at least one of the iterators is not at the end of its
    // sequence, and those that are not refer to the earliest
    // range or line that contains or starts after CURRENT.
    //
    // Note that every byte is in one of four states: it is covered
    // or not covered by a range, and, independently, it is
    // covered or not covered by a line.
    //
    // The second invariant is that CURRENT refers to a byte whose
    // state is different from its predecessor, or it refers to the
    // first byte in the address space. In other words, CURRENT is
    // always the address of a transition.
    //
    // Note that, although each iteration advances CURRENT from one
    // transition address to the next in each iteration, it might
    // not advance the iterators. Suppose we have a range that
    // starts with a line, has a gap, and then a second line, and
    // suppose that we enter an iteration with CURRENT at the end of
    // the first line. The next transition address is the start of
    // the second line, after the gap, so the iteration should
    // advance CURRENT to that point. At the head of that iteration,
    // the invariants require that the line iterator be pointing at
    // the second line. But this is also true at the head of the
    // next. And clearly, the iteration must not change the range
    // iterator. So neither iterator moves.

    // Assert the first invariant (see above).
    assert(!range || current < range->address || within(*range, current));
    assert(!line || current < line->address || within(*line, current));

    // The next transition after CURRENT.
    Module::Address next_transition;

    // Figure out which state we're in, add lines or warn, and compute
    // the next transition address.
    if (range && current >= range->address) {
      if (line && current >= line->address) {
        // Covered by both a line and a range.
        Module::Address range_left = range->size - (current - range->address);
        Module::Address line_left = line->size - (current - line->address);
        // This may overflow, but things work out.
        next_transition = current + std::min(range_left, line_left);
        Module::Line l = *line;
        l.address = current;
        l.size = next_transition - current;
        range->AddLine(l);
        last_line_used = line;
      } else {
        // Covered by a range, but no line.
        if (range->function != last_function_cited) {
          reporter->UncoveredFunction(*(range->function));
          last_function_cited = range->function;
        }
        if (line && within(*range, line->address))
          next_transition = line->address;
        else
          // If this overflows, we'll catch it below.
          next_transition = range->address + range->size;
      }
    } else {
      if (line && current >= line->address) {
        // Covered by a line, but no range.
        //
        // If GCC emits padding after one function to align the start
        // of the next, then it will attribute the padding
        // instructions to the last source line of function (to reduce
        // the size of the line number info), but omit it from the
        // DW_AT_{low,high}_pc range given in .debug_info (since it
        // costs nothing to be precise there). If we did use at least
        // some of the line we're about to skip, and it ends at the
        // start of the next function, then assume this is what
        // happened, and don't warn.
        if (line != last_line_cited
            && !(range
                 && line == last_line_used
                 && range->address - line->address == line->size)) {
          reporter->UncoveredLine(*line);
          last_line_cited = line;
        }
        if (range && within(*line, range->address))
          next_transition = range->address;
        else
          // If this overflows, we'll catch it below.
          next_transition = line->address + line->size;
      } else {
        // Covered by neither a range nor a line. By the invariant,
        // both range and line begin after CURRENT. The next transition
        // is the start of the next range or next line, whichever
        // is earliest.
        assert(range || line);
        if (range && line)
          next_transition = std::min(range->address, line->address);
        else if (range)
          next_transition = range->address;
        else
          next_transition = line->address;
      }
    }

    // If a function or line abuts the end of the address space, then
    // next_transition may end up being zero, in which case we've completed
    // our pass. Handle that here, instead of trying to deal with it in
    // each place we compute next_transition.

    // Some dwarf producers handle linker-removed functions by using -1 as a
    // tombstone in the line table. So the end marker can be -1.
    if (!next_transition || next_transition == Module::kMaxAddress)
      break;

    // Advance iterators as needed. If lines overlap or functions overlap,
    // then we could go around more than once. We don't worry too much
    // about what result we produce in that case, just as long as we don't
    // hang or crash.
    while (range_it != sorted_ranges.end()
           && next_transition >= range_it->address
           && !within(*range_it, next_transition))
      range_it++;
    range = (range_it != sorted_ranges.end()) ? &(*range_it) : NULL;
    while (line_it != lines_.end()
           && next_transition >= line_it->address
           && !within(*line_it, next_transition))
      line_it++;
    line = (line_it != lines_.end()) ? &*line_it : NULL;

    // We must make progress.
    assert(next_transition > current);
    current = next_transition;
  }
}

void DwarfCUToModule::AssignFilesToInlines() {
  // Assign File* to Inlines inside this CU.
  auto assignFile = [this](unique_ptr<Module::Inline>& in) {
    in->call_site_file = files_[in->call_site_file_id];
  };
  for (auto func : cu_context_->functions) {
    Module::Inline::InlineDFS(func->inlines, assignFile);
  }
}

void DwarfCUToModule::Finish() {
  // Assembly language files have no function data, and that gives us
  // no place to store our line numbers (even though the GNU toolchain
  // will happily produce source line info for assembly language
  // files).  To avoid spurious warnings about lines we can't assign
  // to functions, skip CUs in languages that lack functions.
  if (!cu_context_->language->HasFunctions())
    return;

  // Read source line info, if we have any.
  if (has_source_line_info_)
    ReadSourceLines(source_line_offset_);

  vector<Module::Function*>* functions = &cu_context_->functions;

  // Dole out lines to the appropriate functions.
  AssignLinesToFunctions();

  AssignFilesToInlines();

  // Add our functions, which now have source lines assigned to them,
  // to module_, and remove duplicate functions.
  for (Module::Function* func : *functions)
    if (!cu_context_->file_context->module_->AddFunction(func)) {
      auto iter = cu_context_->spec_function_offsets.find(func);
      if (iter != cu_context_->spec_function_offsets.end())
        cu_context_->file_context->file_private_->forward_ref_die_to_func.erase(
            iter->second);
      delete func;
    }

  // Ownership of the function objects has shifted from cu_context to
  // the Module.
  functions->clear();

  cu_context_->file_context->ClearSpecifications();
}

bool DwarfCUToModule::StartCompilationUnit(uint64_t offset,
                                           uint8_t address_size,
                                           uint8_t offset_size,
                                           uint64_t cu_length,
                                           uint8_t dwarf_version) {
  cu_context_->version = dwarf_version;
  return dwarf_version >= 2;
}

bool DwarfCUToModule::StartRootDIE(uint64_t offset, enum DwarfTag tag) {
  // We don't deal with partial compilation units (the only other tag
  // likely to be used for root DIE).
  return (tag == DW_TAG_compile_unit
	  || tag == DW_TAG_skeleton_unit);
}

} // namespace google_breakpad
