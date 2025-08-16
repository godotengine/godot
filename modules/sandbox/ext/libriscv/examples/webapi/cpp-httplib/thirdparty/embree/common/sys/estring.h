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

  struct IndentOStream : public std::streambuf
  {
    explicit IndentOStream(std::ostream &ostream, int indent = 2)
      : streambuf(ostream.rdbuf())
      , start_of_line(true)
      , ident_str(indent, ' ')
      , stream(&ostream)
    {
      // set streambuf of ostream to this and save original streambuf
      stream->rdbuf(this);
    }

    virtual ~IndentOStream()
    {
      if (stream != NULL) {
        // restore old streambuf
        stream->rdbuf(streambuf);
      }
    }

  protected:
    virtual int overflow(int ch) {
      if (start_of_line && ch != '\n') {
        streambuf->sputn(ident_str.data(), ident_str.size());
      }
      start_of_line = ch == '\n';
      return streambuf->sputc(ch);
    }

  private:
    std::streambuf *streambuf;
    bool start_of_line;
    std::string ident_str;
    std::ostream *stream;
  };

  std::string toLowerCase(const std::string& s);
  std::string toUpperCase(const std::string& s);

  Vec2f string_to_Vec2f ( std::string str );
  Vec3f string_to_Vec3f ( std::string str );
  Vec4f string_to_Vec4f ( std::string str );
}
