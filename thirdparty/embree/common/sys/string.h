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

#include "platform.h"
#include "../math/vec2.h"
#include "../math/vec3.h"

namespace embree
{
  class IOStreamStateRestorer 
  {
  public:
    IOStreamStateRestorer(std::ostream& iostream)
      : iostream(iostream), flags(iostream.flags()), precision(iostream.precision()) {
    }

    ~IOStreamStateRestorer() {
      iostream.flags(flags);
      iostream.precision(precision);
    }
    
  private:
    std::ostream& iostream;
    std::ios::fmtflags flags;
    std::streamsize precision;
  };

  std::string toLowerCase(const std::string& s);
  std::string toUpperCase(const std::string& s);

  Vec2f string_to_Vec2f ( std::string str );
  Vec3f string_to_Vec3f ( std::string str );
}
