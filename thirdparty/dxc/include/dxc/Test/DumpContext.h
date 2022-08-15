///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DumpContext.h                                                             //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// Context for dumping structured data, enums, and flags for use in tests.   //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "dxc/Support/Global.h"
#include <string>
#include <ostream>
#include <sstream>
#include <iomanip>
#include <unordered_set>

namespace hlsl {
namespace dump {

template<typename _T>
struct EnumValue {
public:
  EnumValue(const _T &e) : eValue(e) {}
  _T eValue;
};

template<typename _T, typename _StoreT = uint32_t>
struct FlagsValue {
public:
  FlagsValue(const _StoreT &f) : Flags(f) {}
  _StoreT Flags;
};

struct QuotedStringValue {
public:
  QuotedStringValue(const char *str) : Str(str) {}
  const char *Str;
};

class DumpContext {
private:
  std::ostream &m_out;
  unsigned m_indent = 0;
  bool m_bCheckByName = false;
  std::unordered_set<size_t> m_visited;

  std::ostream &DoIndent() {
    return m_out << std::setfill(' ')
      << std::setw((m_indent > 16) ? 32 : m_indent * 2)
      << "";
  }

public:
  DumpContext(std::ostream &outStream) : m_out(outStream) {}

  void Indent() { if (m_indent < (1 << 30)) m_indent++; }
  void Dedent() { if (m_indent > 0) m_indent--; }

  template<typename _T>
  std::ostream &Write(_T t) {
    return Write(m_out, t);
  }
  template<typename _T, typename... Args>
  std::ostream &Write(_T t, Args... args) {
    return Write(Write(m_out, t), args...);
  }
  template<typename _T>
  std::ostream &Write(std::ostream &out, _T t) {
    return out << t;
  }
  template<>
  std::ostream &Write<uint8_t>(std::ostream &out, uint8_t t) {
    return out << (unsigned)t;
  }
  template<typename _T, typename... Args>
  std::ostream &Write(std::ostream &out, _T t, Args... args) {
    return Write(Write(out, t), args...);
  }

  template<typename _T>
  std::ostream &WriteLn(_T t) {
    return Write(DoIndent(), t) << std::endl
      << std::resetiosflags(std::ios_base::basefield | std::ios_base::showbase);
  }
  template<typename _T, typename... Args>
  std::ostream &WriteLn(_T t, Args... args) {
    return Write(Write(DoIndent(), t), args...) << std::endl
      << std::resetiosflags(std::ios_base::basefield | std::ios_base::showbase);
  }

  template <typename _T>
  std::ostream &WriteEnumValue(_T eValue) {
    const char *szValue = ToString(eValue);
    if (szValue)
      return Write(szValue);
    else
      return Write("<unknown: ", std::hex, std::showbase, (UINT)eValue, ">");
  }

  template<typename _T>
  void DumpEnum(const char *Name, _T eValue) {
    WriteLn(Name, ": ", EnumValue<_T>(eValue));
  }
  template<typename _T, typename _StoreT = uint32_t>
  void DumpFlags(const char *Name, _StoreT Flags) {
    WriteLn(Name, ": ", FlagsValue<_T, _StoreT>(Flags));
  }

  template<typename... Args>
  void Failure(Args... args) {
    WriteLn("Failed: ", args...);
  }

  // Return true if ptr has not yet been visited, prevents recursive dumping
  bool Visit(size_t value) { return m_visited.insert(value).second; }
  bool Visit(const void *ptr) { return Visit((size_t)ptr); }
  void VisitReset() { m_visited.clear(); }
};

// Copied from llvm/ADT/StringExtras.h
inline char hexdigit(unsigned X, bool LowerCase = false) {
  const char HexChar = LowerCase ? 'a' : 'A';
  return X < 10 ? '0' + X : HexChar + X - 10;
}
// Copied from lib/IR/AsmWriter.cpp
// EscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
inline std::string EscapedString(const char *text) {
  std::ostringstream ss;
  size_t size = strlen(text);
  for (unsigned i = 0, e = size; i != e; ++i) {
    unsigned char C = text[i];
    if (isprint(C) && C != '\\' && C != '"')
      ss << C;
    else
      ss << '\\' << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
  return ss.str();
}

template<typename _T>
std::ostream& operator<<(std::ostream& out, const EnumValue<_T> &obj) {
  if (const char *szValue = ToString(obj.eValue))
    return out << szValue;
  else
    return out << "<unknown: " << std::hex << std::showbase << (UINT)obj.eValue << ">";
}

template<typename _T, typename _StoreT>
std::ostream& operator<<(std::ostream& out, const FlagsValue<_T, _StoreT> &obj) {
  _StoreT Flags = obj.Flags;
  if (!Flags) {
    const char *szValue = ToString((_T)0);
    if (szValue)
      return out << "0 (" << szValue << ")";
    else
      return out << "0";
  }
  uint32_t flag = 0;
  out << "(";
  while (Flags) {
    if (flag)
      out << " | ";
    flag = (Flags & ~(Flags - 1));
    Flags ^= flag;
    out << EnumValue<_T>((_T)flag);
  }
  out << ")";
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const QuotedStringValue &obj) {
  if (!obj.Str)
    return out << "<null string pointer>";
  return out << "\"" << EscapedString(obj.Str) << "\"";
}

} // namespace dump
} // namespace hlsl
