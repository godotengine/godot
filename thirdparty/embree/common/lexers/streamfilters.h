// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
