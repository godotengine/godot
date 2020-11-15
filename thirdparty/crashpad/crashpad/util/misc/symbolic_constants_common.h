// Copyright 2014 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef CRASHPAD_UTIL_MISC_SYMBOLIC_CONSTANTS_COMMON_H_
#define CRASHPAD_UTIL_MISC_SYMBOLIC_CONSTANTS_COMMON_H_

//! \file
//!
//! \anchor symbolic_constant_terminology
//! Symbolic constant terminology
//! =============================
//! <dl>
//!   <dt>Family</dt>
//!   <dd>A group of related symbolic constants. Typically, within a single
//!       family, one function will be used to transform a numeric value to a
//!       string equivalent, and another will perform the inverse operation.
//!       Families include POSIX signals and Mach exception masks.</dd>
//!   <dt>Full name</dt>
//!   <dd>The normal symbolic name used for a constant. For example, in the
//!       family of POSIX signals, the strings `"SIGHUP"` and `"SIGSEGV"` are
//!       full names.</dd>
//!   <dt>Short name</dt>
//!   <dd>An abbreviated form of symbolic name used for a constant. Short names
//!       vary between families, but are commonly constructed by removing a
//!       common prefix from full names. For example, in the family of POSIX
//!       signals, the prefix is `SIG`, and short names include `"HUP"` and
//!       `"SEGV"`.</dd>
//!   <dt>Numeric string</dt>
//!   <dd>A string that does not contain a full or short name, but contains a
//!       numeric value that can be interpreted as a symbolic constant. For
//!       example, in the family of POSIX signals, `SIGKILL` generally has value
//!       `9`, so the numeric string `"9"` would be interpreted equivalently to
//!       `"SIGKILL"`.</dd>
//! </dl>

namespace crashpad {

//! \brief Options for various `*ToString` functions in `symbolic_constants_*`
//!     files.
//!
//! \sa \ref symbolic_constant_terminology "Symbolic constant terminology"
enum SymbolicConstantToStringOptionBits {
  //! \brief Return the full name for a given constant.
  //!
  //! \attention API consumers should provide this value when desired, but
  //!     should provide only one of kUseFullName and ::kUseShortName. Because
  //!     kUseFullName is valueless, implementers should check for the absence
  //!     of ::kUseShortName instead.
  kUseFullName = 0 << 0,

  //! \brief Return the short name for a given constant.
  kUseShortName = 1 << 0,

  //! \brief If no symbolic name is known for a given constant, return an empty
  //!     string.
  //!
  //! \attention API consumers should provide this value when desired, but
  //!     should provide only one of kUnknownIsEmpty and ::kUnknownIsNumeric.
  //!     Because kUnknownIsEmpty is valueless, implementers should check for
  //!     the absence of ::kUnknownIsNumeric instead.
  kUnknownIsEmpty = 0 << 1,

  //! \brief If no symbolic name is known for a given constant, return a numeric
  //!     string.
  //!
  //! The numeric format used will vary by family, but will be appropriate to
  //! the family. Families whose values are typically constructed as bitfields
  //! will generally use a hexadecimal format, and other families will generally
  //! use a signed or unsigned decimal format.
  kUnknownIsNumeric = 1 << 1,

  //! \brief Use `|` to combine values in a bitfield.
  //!
  //! For families whose values may be constructed as bitfields, allow
  //! conversion to strings containing multiple individual components treated as
  //! being combined by a bitwise “or” operation. An example family of constants
  //! that behaves this way is the suite of Mach exception masks. For constants
  //! that are not constructed as bitfields, or constants that are only
  //! partially constructed as bitfields, this option has no effect.
  kUseOr = 1 << 2,
};

//! \brief A bitfield containing values of #SymbolicConstantToStringOptionBits.
using SymbolicConstantToStringOptions = unsigned int;

//! \brief Options for various `StringTo*` functions in `symbolic_constants_*`
//!     files.
//!
//! Not every `StringTo*` function will implement each of these options. See
//! function-specific documentation for details.
//!
//! \sa \ref symbolic_constant_terminology "Symbolic constant terminology"
enum StringToSymbolicConstantOptionBits {
  //! \brief Allow conversion from a string containing a symbolic constant by
  //!     its full name.
  kAllowFullName = 1 << 0,

  //! \brief Allow conversion from a string containing a symbolic constant by
  //!     its short name.
  kAllowShortName = 1 << 1,

  //! \brief Allow conversion from a numeric string.
  kAllowNumber = 1 << 2,

  //! \brief Allow `|` to combine values in a bitfield.
  //!
  //! For families whose values may be constructed as bitfields, allow
  //! conversion of strings containing multiple individual components treated as
  //! being combined by a bitwise “or” operation. An example family of constants
  //! that behaves this way is the suite of Mach exception masks. For constants
  //! that are not constructed as bitfields, or constants that are only
  //! partially constructed as bitfields, this option has no effect.
  kAllowOr = 1 << 3,
};

//! \brief A bitfield containing values of #StringToSymbolicConstantOptionBits.
using StringToSymbolicConstantOptions = unsigned int;

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_SYMBOLIC_CONSTANTS_COMMON_H_
