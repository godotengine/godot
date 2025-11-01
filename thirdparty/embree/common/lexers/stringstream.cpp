// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "stringstream.h"

namespace embree
{
  static const std::string stringChars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 _.,+-=:/*\\";
  
  /* creates map for fast categorization of characters */
  static void createCharMap(bool map[256], const std::string& chrs) {
    for (size_t i=0; i<256; i++) map[i] = false;
    for (size_t i=0; i<chrs.size(); i++) map[uint8_t(chrs[i])] = true;
  }

  /* simple tokenizer */
  StringStream::StringStream(const Ref<Stream<int> >& cin, const std::string& seps, const std::string& endl, bool multiLine)
    : cin(cin), endl(endl), multiLine(multiLine)
  {
    createCharMap(isSepMap,seps);
    createCharMap(isValidCharMap,stringChars);
  }

  std::string StringStream::next()
  {
    /* skip separators */
    while (cin->peek() != EOF) {
      if (endl != "" && cin->peek() == '\n') { cin->drop(); return endl; }
      if (multiLine && cin->peek() == '\\') {
        cin->drop();
        if (cin->peek() == '\n') { cin->drop(); continue; }
        cin->unget();
      }
      if (!isSeparator(cin->peek())) break;
      cin->drop();
    }

    /* parse everything until the next separator */
    std::vector<char> str; str.reserve(64);
    while (cin->peek() != EOF && !isSeparator(cin->peek())) {
      int c = cin->get();
      if (!isValidChar(c)) abort(); //throw std::runtime_error("invalid character "+std::string(1,c)+" in input");
      str.push_back((char)c);
    }
    str.push_back(0);
    return std::string(str.data());
  }
}
