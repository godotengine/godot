// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"
#include "../math/vec2.h"
#include "../math/vec3.h"
#include "../math/vec4.h"

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
  Vec4f string_to_Vec4f ( std::string str );
}
