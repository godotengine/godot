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

// language.h: Define google_breakpad::Language. Instances of
// subclasses of this class provide language-appropriate operations
// for the Breakpad symbol dumper.

#ifndef COMMON_LINUX_LANGUAGE_H__
#define COMMON_LINUX_LANGUAGE_H__

#include <string>

#include "common/using_std_string.h"

namespace google_breakpad {

// An abstract base class for language-specific operations. We choose
// an instance of a subclass of this when we find the CU's language.
// This class's definitions are appropriate for CUs with no specified
// language.
class Language {
 public:
  // A base class destructor should be either public and virtual,
  // or protected and nonvirtual.
  virtual ~Language() {}

  // Return true if this language has functions to which we can assign
  // line numbers. (Debugging info for assembly language, for example,
  // can have source location information, but does not have functions
  // recorded using DW_TAG_subprogram DIEs.)
  virtual bool HasFunctions() const { return true; }

  // Construct a fully-qualified, language-appropriate form of NAME,
  // given that PARENT_NAME is the name of the construct enclosing
  // NAME. If PARENT_NAME is the empty string, then NAME is a
  // top-level name.
  //
  // This API sort of assumes that a fully-qualified name is always
  // some simple textual composition of the unqualified name and its
  // parent's name, and that we don't need to know anything else about
  // the parent or the child (say, their DIEs' tags) to do the job.
  // This is true for the languages we support at the moment, and
  // keeps things concrete. Perhaps a more refined operation would
  // take into account the parent and child DIE types, allow languages
  // to use their own data type for complex parent names, etc. But if
  // C++ doesn't need all that, who would?
  virtual string MakeQualifiedName (const string& parent_name,
                                    const string& name) const = 0;

  enum DemangleResult {
    // Demangling was not performed because it’s not appropriate to attempt.
    kDontDemangle = -1,

    kDemangleSuccess,
    kDemangleFailure,
  };

  // Wraps abi::__cxa_demangle() or similar for languages where appropriate.
  virtual DemangleResult DemangleName(const string& mangled,
                                      string* demangled) const {
    demangled->clear();
    return kDontDemangle;
  }

  // Instances for specific languages.
  static const Language * const CPlusPlus,
                        * const Java,
                        * const Swift,
                        * const Rust,
                        * const Assembler;
};

} // namespace google_breakpad

#endif  // COMMON_LINUX_LANGUAGE_H__
