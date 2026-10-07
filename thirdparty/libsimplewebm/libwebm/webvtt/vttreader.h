// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef WEBVTT_VTTREADER_H_
#define WEBVTT_VTTREADER_H_

#include <cstdio>

#include "./webvttparser.h"

namespace libwebvtt {

class VttReader : public libwebvtt::Reader {
 public:
  VttReader();
  virtual ~VttReader();

  // Open the file identified by |filename| in read-only mode, as a
  // binary stream of bytes.  Returns 0 on success, negative if error.
  int Open(const char* filename);

  // Closes the file stream.  Note that the stream is automatically
  // closed when the VttReader object is destroyed.
  void Close();

  // Reads the next character in the file stream, as per the semantics
  // of Reader::GetChar.  Returns negative if error, 0 on success, and
  // positive if end-of-stream has been reached.
  virtual int GetChar(char* c);

 private:
  FILE* file_;

  VttReader(const VttReader&);
  VttReader& operator=(const VttReader&);
};

}  // namespace libwebvtt

#endif  // WEBVTT_VTTREADER_H_
