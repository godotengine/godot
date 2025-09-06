
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#pragma once

#include <ostream>
#include <streambuf>

// from https://stackoverflow.com/a/8244052
class NullStreambuf : public std::streambuf {
  char dummyBuffer[64];

protected:
  virtual int overflow(int c) final;
};

class NullOStream final : private NullStreambuf, public std::ostream {
public:
  NullOStream() : std::ostream(this) {}
  NullStreambuf *rdbuf() { return this; }
  virtual void avoidOutOfLineVirtualCompilerWarning();
};

