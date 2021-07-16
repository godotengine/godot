// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "stringstream.h"
#include "../sys/filename.h"
#include "../math/vec2.h"
#include "../math/vec3.h"
#include "../math/col3.h"
#include "../math/color.h"

namespace embree
{
  /*! helper class for simple command line parsing */
  class ParseStream : public Stream<std::string>
  {
  public:
    ParseStream (const Ref<Stream<std::string> >& cin) : cin(cin) {}

    ParseStream (const Ref<Stream<int> >& cin, const std::string& seps = "\n\t\r ",
                 const std::string& endl = "", bool multiLine = false)
      : cin(new StringStream(cin,seps,endl,multiLine)) {}

  public:
    ParseLocation location() { return cin->loc(); }
    std::string next() { return cin->get(); }

    void force(const std::string& next) {
      std::string token = getString();
      if (token != next)
        THROW_RUNTIME_ERROR("token \""+next+"\" expected but token \""+token+"\" found");
    }

    std::string getString() {
      return get();
    }

    FileName getFileName()  {
      return FileName(get());
    }

    int   getInt  () {
      return atoi(get().c_str());
    }

    Vec2i getVec2i() {
      int x = atoi(get().c_str());
      int y = atoi(get().c_str());
      return Vec2i(x,y);
    }

    Vec3ia getVec3ia() {
      int x = atoi(get().c_str());
      int y = atoi(get().c_str());
      int z = atoi(get().c_str());
      return Vec3ia(x,y,z);
    }

    float getFloat() {
      return (float)atof(get().c_str());
    }

    Vec2f getVec2f() {
      float x = (float)atof(get().c_str());
      float y = (float)atof(get().c_str());
      return Vec2f(x,y);
    }

    Vec3f getVec3f() {
      float x = (float)atof(get().c_str());
      float y = (float)atof(get().c_str());
      float z = (float)atof(get().c_str());
      return Vec3f(x,y,z);
    }

    Vec3fa getVec3fa() {
      float x = (float)atof(get().c_str());
      float y = (float)atof(get().c_str());
      float z = (float)atof(get().c_str());
      return Vec3fa(x,y,z);
    }

    Col3f getCol3f() {
      float x = (float)atof(get().c_str());
      float y = (float)atof(get().c_str());
      float z = (float)atof(get().c_str());
      return Col3f(x,y,z);
    }

    Color getColor() {
      float r = (float)atof(get().c_str());
      float g = (float)atof(get().c_str());
      float b = (float)atof(get().c_str());
      return Color(r,g,b);
    }

  private:
    Ref<Stream<std::string> > cin;
  };
}
