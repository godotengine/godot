//===-- StringTableBuilder.cpp - String table building utility ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/StringTableBuilder.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/COFF.h"
#include "llvm/Support/Endian.h"

using namespace llvm;

static bool compareBySuffix(StringRef a, StringRef b) {
  size_t sizeA = a.size();
  size_t sizeB = b.size();
  size_t len = std::min(sizeA, sizeB);
  for (size_t i = 0; i < len; ++i) {
    char ca = a[sizeA - i - 1];
    char cb = b[sizeB - i - 1];
    if (ca != cb)
      return ca > cb;
  }
  return sizeA > sizeB;
}

void StringTableBuilder::finalize(Kind kind) {
  SmallVector<StringRef, 8> Strings;
  Strings.reserve(StringIndexMap.size());

  for (auto i = StringIndexMap.begin(), e = StringIndexMap.end(); i != e; ++i)
    Strings.push_back(i->getKey());

  std::sort(Strings.begin(), Strings.end(), compareBySuffix);

  switch (kind) {
  case ELF:
  case MachO:
    // Start the table with a NUL byte.
    StringTable += '\x00';
    break;
  case WinCOFF:
    // Make room to write the table size later.
    StringTable.append(4, '\x00');
    break;
  }

  StringRef Previous;
  for (StringRef s : Strings) {
    if (kind == WinCOFF)
      assert(s.size() > COFF::NameSize && "Short string in COFF string table!");

    if (Previous.endswith(s)) {
      StringIndexMap[s] = StringTable.size() - 1 - s.size();
      continue;
    }

    StringIndexMap[s] = StringTable.size();
    StringTable += s;
    StringTable += '\x00';
    Previous = s;
  }

  switch (kind) {
  case ELF:
    break;
  case MachO:
    // Pad to multiple of 4.
    while (StringTable.size() % 4)
      StringTable += '\x00';
    break;
  case WinCOFF:
    // Write the table size in the first word.
    assert(StringTable.size() <= std::numeric_limits<uint32_t>::max());
    uint32_t size = static_cast<uint32_t>(StringTable.size());
    support::endian::write<uint32_t, support::little, support::unaligned>(
        StringTable.data(), size);
    break;
  }
}

void StringTableBuilder::clear() {
  StringTable.clear();
  StringIndexMap.clear();
}
