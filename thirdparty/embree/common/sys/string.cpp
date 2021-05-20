// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

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
  
  Vec4f string_to_Vec4f ( std::string str )
  {
    size_t next = 0;
    const float x = std::stof(str,&next); str = str.substr(next+1);
    const float y = std::stof(str,&next); str = str.substr(next+1);
    const float z = std::stof(str,&next); str = str.substr(next+1);
    const float w = std::stof(str,&next);
    return Vec4f(x,y,z,w);
  }
}
