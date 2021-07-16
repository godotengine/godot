// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stream.h"

namespace embree
{
  /* removes all line comments from a stream */
  class LineCommentFilter : public Stream<int>
  {
  public:
    LineCommentFilter (const FileName& fileName, const std::string& lineComment)
      : cin(new FileStream(fileName)), lineComment(lineComment) {}
    LineCommentFilter (Ref<Stream<int> > cin, const std::string& lineComment)
      : cin(cin), lineComment(lineComment) {}

    ParseLocation location() { return cin->loc(); }

    int next()
    {
      /* look if the line comment starts here */
      for (size_t j=0; j<lineComment.size(); j++) {
        if (cin->peek() != lineComment[j]) { cin->unget(j); goto not_found; }
        cin->get();
      }
      /* eat all characters until the end of the line (or file) */
      while (cin->peek() != '\n' && cin->peek() != EOF) cin->get();

    not_found:
      return cin->get();
    }

  private:
    Ref<Stream<int> > cin;
    std::string lineComment;
  };
}
