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

// Implementation of google_breakpad::DwarfCFIToModule.
// See dwarf_cfi_to_module.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <memory>
#include <sstream>
#include <utility>

#include "common/dwarf_cfi_to_module.h"

namespace google_breakpad {

using std::ostringstream;

vector<string> DwarfCFIToModule::RegisterNames::MakeVector(
    const char * const *strings,
    size_t size) {
  vector<string> names(strings, strings + size);
  return names;
}

vector<string> DwarfCFIToModule::RegisterNames::I386() {
  static const char *const names[] = {
    "$eax", "$ecx", "$edx", "$ebx", "$esp", "$ebp", "$esi", "$edi",
    "$eip", "$eflags", "$unused1",
    "$st0", "$st1", "$st2", "$st3", "$st4", "$st5", "$st6", "$st7",
    "$unused2", "$unused3",
    "$xmm0", "$xmm1", "$xmm2", "$xmm3", "$xmm4", "$xmm5", "$xmm6", "$xmm7",
    "$mm0", "$mm1", "$mm2", "$mm3", "$mm4", "$mm5", "$mm6", "$mm7",
    "$fcw", "$fsw", "$mxcsr",
    "$es", "$cs", "$ss", "$ds", "$fs", "$gs", "$unused4", "$unused5",
    "$tr", "$ldtr"
  };

  return MakeVector(names, sizeof(names) / sizeof(names[0]));
}

vector<string> DwarfCFIToModule::RegisterNames::X86_64() {
  static const char *const names[] = {
    "$rax", "$rdx", "$rcx", "$rbx", "$rsi", "$rdi", "$rbp", "$rsp",
    "$r8",  "$r9",  "$r10", "$r11", "$r12", "$r13", "$r14", "$r15",
    "$rip",
    "$xmm0","$xmm1","$xmm2", "$xmm3", "$xmm4", "$xmm5", "$xmm6", "$xmm7",
    "$xmm8","$xmm9","$xmm10","$xmm11","$xmm12","$xmm13","$xmm14","$xmm15",
    "$st0", "$st1", "$st2", "$st3", "$st4", "$st5", "$st6", "$st7",
    "$mm0", "$mm1", "$mm2", "$mm3", "$mm4", "$mm5", "$mm6", "$mm7",
    "$rflags",
    "$es", "$cs", "$ss", "$ds", "$fs", "$gs", "$unused1", "$unused2",
    "$fs.base", "$gs.base", "$unused3", "$unused4",
    "$tr", "$ldtr",
    "$mxcsr", "$fcw", "$fsw"
  };

  return MakeVector(names, sizeof(names) / sizeof(names[0]));
}

// Per ARM IHI 0040A, section 3.1
vector<string> DwarfCFIToModule::RegisterNames::ARM() {
  static const char *const names[] = {
    "r0",  "r1",  "r2",  "r3",  "r4",  "r5",  "r6",  "r7",
    "r8",  "r9",  "r10", "r11", "r12", "sp",  "lr",  "pc",
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
    "fps", "cpsr", "",   "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "s0",  "s1",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "s12", "s13", "s14", "s15",
    "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
    "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31",
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7"
  };

  return MakeVector(names, sizeof(names) / sizeof(names[0]));
}

// Per ARM IHI 0057A, section 3.1
vector<string> DwarfCFIToModule::RegisterNames::ARM64() {
  static const char *const names[] = {
    "x0",  "x1",  "x2",  "x3",  "x4",  "x5",  "x6",  "x7",
    "x8",  "x9",  "x10", "x11", "x12", "x13", "x14", "x15",
    "x16", "x17", "x18", "x19", "x20", "x21", "x22", "x23",
    "x24", "x25", "x26", "x27", "x28", "x29", "x30", "sp",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
    "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
  };

  return MakeVector(names, sizeof(names) / sizeof(names[0]));
}

vector<string> DwarfCFIToModule::RegisterNames::MIPS() {
  static const char* const kRegisterNames[] = {
    "$zero", "$at",  "$v0",  "$v1",  "$a0",   "$a1",  "$a2",  "$a3",
    "$t0",   "$t1",  "$t2",  "$t3",  "$t4",   "$t5",  "$t6",  "$t7",
    "$s0",   "$s1",  "$s2",  "$s3",  "$s4",   "$s5",  "$s6",  "$s7",
    "$t8",   "$t9",  "$k0",  "$k1",  "$gp",   "$sp",  "$fp",  "$ra",
    "$lo",   "$hi",  "$pc",  "$f0",  "$f2",   "$f3",  "$f4",  "$f5",
    "$f6",   "$f7",  "$f8",  "$f9",  "$f10",  "$f11", "$f12", "$f13",
    "$f14",  "$f15", "$f16", "$f17", "$f18",  "$f19", "$f20",
    "$f21",  "$f22", "$f23", "$f24", "$f25",  "$f26", "$f27",
    "$f28",  "$f29", "$f30", "$f31", "$fcsr", "$fir"
  };

  return MakeVector(kRegisterNames,
                    sizeof(kRegisterNames) / sizeof(kRegisterNames[0]));
}

vector<string> DwarfCFIToModule::RegisterNames::RISCV() {
  static const char *const names[] = {
    "pc",  "ra",  "sp",  "gp",  "tp",  "t0",  "t1",  "t2",
    "s0",  "s1",  "a0",  "a1",  "a2",  "a3",  "a4",  "a5",
    "a6",  "a7",  "s2",  "s3",  "s4",  "s5",  "s6",  "s7",
    "s8",  "s9",  "s10", "s11", "t3",  "t4",  "t5",  "t6",
    "f0",  "f1",  "f2",  "f3",  "f4",  "f5",  "f6",  "f7",
    "f8",  "f9",  "f10", "f11", "f12", "f13", "f14", "f15",
    "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23",
    "f24", "f25", "f26", "f27", "f28", "f29", "f30", "f31",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "",    "",    "",    "",    "",    "",    "",    "",
    "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
    "v8",  "v9",  "v10", "v11", "v12", "v13", "v14", "v15",
    "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
    "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"
  };

  return MakeVector(names, sizeof(names) / sizeof(names[0]));
}

bool DwarfCFIToModule::Entry(size_t offset, uint64_t address, uint64_t length,
                             uint8_t version, const string& augmentation,
                             unsigned return_address) {
  assert(!entry_);

  // If CallFrameInfo can handle this version and
  // augmentation, then we should be okay with that, so there's no
  // need to check them here.

  // Get ready to collect entries.
  entry_ = std::make_unique<Module::StackFrameEntry>();
  entry_->address = address;
  entry_->size = length;
  entry_offset_ = offset;
  return_address_ = return_address;

  // Breakpad STACK CFI records must provide a .ra rule, but DWARF CFI
  // may not establish any rule for .ra if the return address column
  // is an ordinary register, and that register holds the return
  // address on entry to the function. So establish an initial .ra
  // rule citing the return address register.
  if (return_address_ < register_names_.size())
    entry_->initial_rules[ra_name_] = register_names_[return_address_];

  return true;
}

string DwarfCFIToModule::RegisterName(int i) {
  assert(entry_);
  if (i < 0) {
    assert(i == kCFARegister);
    return cfa_name_;
  }
  unsigned reg = i;
  if (reg == return_address_)
    return ra_name_;

  // Ensure that a non-empty name exists for this register value.
  if (reg < register_names_.size() && !register_names_[reg].empty())
    return register_names_[reg];

  reporter_->UnnamedRegister(entry_offset_, reg);
  return string("unnamed_register") + std::to_string(reg);
}

void DwarfCFIToModule::Record(Module::Address address, int reg,
                              const string& rule) {
  assert(entry_);

  // Place the name in our global set of strings, and then use the string
  // from the set. Even though the assignment looks like a copy, all the
  // major string implementations use reference counting internally,
  // so the effect is to have all our data structures share copies of rules
  // whenever possible. Since register names are drawn from a
  // vector<string>, register names are already shared.
  string shared_rule = *common_strings_.insert(rule).first;

  // Is this one of this entry's initial rules?
  if (address == entry_->address)
    entry_->initial_rules[RegisterName(reg)] = shared_rule;
  // File it under the appropriate address.
  else
    entry_->rule_changes[address][RegisterName(reg)] = shared_rule;
}

bool DwarfCFIToModule::UndefinedRule(uint64_t address, int reg) {
  reporter_->UndefinedNotSupported(entry_offset_, RegisterName(reg));
  // Treat this as a non-fatal error.
  return true;
}

bool DwarfCFIToModule::SameValueRule(uint64_t address, int reg) {
  ostringstream s;
  s << RegisterName(reg);
  Record(address, reg, s.str());
  return true;
}

bool DwarfCFIToModule::OffsetRule(uint64_t address, int reg,
                                  int base_register, long offset) {
  ostringstream s;
  s << RegisterName(base_register) << " " << offset << " + ^";
  Record(address, reg, s.str());
  return true;
}

bool DwarfCFIToModule::ValOffsetRule(uint64_t address, int reg,
                                     int base_register, long offset) {
  ostringstream s;
  s << RegisterName(base_register) << " " << offset << " +";
  Record(address, reg, s.str());
  return true;
}

bool DwarfCFIToModule::RegisterRule(uint64_t address, int reg,
                                    int base_register) {
  ostringstream s;
  s << RegisterName(base_register);
  Record(address, reg, s.str());
  return true;
}

bool DwarfCFIToModule::ExpressionRule(uint64_t address, int reg,
                                      const string& expression) {
  reporter_->ExpressionsNotSupported(entry_offset_, RegisterName(reg));
  // Treat this as a non-fatal error.
  return true;
}

bool DwarfCFIToModule::ValExpressionRule(uint64_t address, int reg,
                                         const string& expression) {
  reporter_->ExpressionsNotSupported(entry_offset_, RegisterName(reg));
  // Treat this as a non-fatal error.
  return true;
}

bool DwarfCFIToModule::End() {
  module_->AddStackFrameEntry(std::move(entry_));
  return true;
}

string DwarfCFIToModule::Architecture() {
  return module_->architecture();
}

void DwarfCFIToModule::Reporter::UnnamedRegister(size_t offset, int reg) {
  fprintf(stderr, "%s, section '%s': "
          "the call frame entry at offset 0x%zx refers to register %d,"
          " whose name we don't know\n",
          file_.c_str(), section_.c_str(), offset, reg);
}

void DwarfCFIToModule::Reporter::UndefinedNotSupported(size_t offset,
                                                       const string& reg) {
  fprintf(stderr, "%s, section '%s': "
          "the call frame entry at offset 0x%zx sets the rule for "
          "register '%s' to 'undefined', but the Breakpad symbol file format"
          " cannot express this\n",
          file_.c_str(), section_.c_str(), offset, reg.c_str());
}

void DwarfCFIToModule::Reporter::ExpressionsNotSupported(size_t offset,
                                                         const string& reg) {
  fprintf(stderr, "%s, section '%s': "
          "the call frame entry at offset 0x%zx uses a DWARF expression to"
          " describe how to recover register '%s', "
          " but this translator cannot yet translate DWARF expressions to"
          " Breakpad postfix expressions\n",
          file_.c_str(), section_.c_str(), offset, reg.c_str());
}

} // namespace google_breakpad
