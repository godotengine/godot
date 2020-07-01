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
    __forceinline bool isSeparator(int c) const { return c<256 && isSepMap[c]; }
  private:
    Ref<Stream<int> > cin; /*! source character stream */
    bool isSepMap[256];    /*! map for fast classification of separators */
    std::string endl;      /*! the token of the end of line */
    bool multiLine;        /*! whether to parse lines wrapped with \ */
  };
}
