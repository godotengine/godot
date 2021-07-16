// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stream.h"

namespace embree
{
  /*! simple tokenizer that produces a string stream */
  class StringStream : public Stream<std::string>
  {
  public:
    StringStream(const Ref<Stream<int> >& cin, const std::string& seps = "\n\t\r ",
                 const std::string& endl = "", bool multiLine = false);
  public:
    ParseLocation location() { return cin->loc(); }
    std::string next();
  private:
    __forceinline bool isSeparator(unsigned int c) const { return c<256 && isSepMap[c]; }
    __forceinline bool isValidChar(unsigned int c) const { return c<256 && isValidCharMap[c]; }
  private:
    Ref<Stream<int> > cin; /*! source character stream */
    bool isSepMap[256];    /*! map for fast classification of separators */
    bool isValidCharMap[256];  /*! map for valid characters */
    std::string endl;      /*! the token of the end of line */
    bool multiLine;        /*! whether to parse lines wrapped with \ */
  };
}
