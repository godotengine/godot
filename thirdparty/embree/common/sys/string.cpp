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

#include "string.h"

#include <algorithm>
#include <ctype.h>

namespace embree
{
  char to_lower(char c) { return char(tolower(int(c))); }
  char to_upper(char c) { return char(toupper(int(c))); }
  std::string toLowerCase(const std::string& s) { std::string dst(s); std::transform(dst.begin(), dst.end(), dst.begin(), to_lower); return dst; }
  std::string toUpperCase(const std::string& s) { std::string dst(s); std::transform(dst.begin(), dst.end(), dst.begin(), to_upper); return dst; }

  Vec2f string_to_Vec2f ( std::string str )
  {
    size_t next = 0;
    const float x = std::stof(str,&next); str = str.substr(next+1);
    const float y = std::stof(str,&next);
    return Vec2f(x,y);
  }
  
  Vec3f string_to_Vec3f ( std::string str )
  {
    size_t next = 0;
    const float x = std::stof(str,&next); str = str.substr(next+1);
    const float y = std::stof(str,&next); str = str.substr(next+1);
    const float z = std::stof(str,&next); 
    return Vec3f(x,y,z);
  }
}
