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

#include "stringstream.h"

namespace embree
{
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
    while (cin->peek() != EOF && !isSeparator(cin->peek()))
      str.push_back((char)cin->get());
    str.push_back(0);
    return std::string(str.data());
  }
}
