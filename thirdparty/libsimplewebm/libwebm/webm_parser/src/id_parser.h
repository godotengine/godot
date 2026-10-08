// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_ID_PARSER_H_
#define SRC_ID_PARSER_H_

#include <cstdint>

#include "src/parser.h"
#include "webm/callback.h"
#include "webm/id.h"
#include "webm/reader.h"
#include "webm/status.h"

namespace webm {

// Parses an EBML ID from a byte stream.
class IdParser : public Parser {
 public:
  IdParser() = default;
  IdParser(IdParser&&) = default;
  IdParser& operator=(IdParser&&) = default;

  IdParser(const IdParser&) = delete;
  IdParser& operator=(const IdParser&) = delete;

  Status Feed(Callback* callback, Reader* reader,
              std::uint64_t* num_bytes_read) override;

  // Gets the parsed ID. This must not be called until the parse had been
  // successfully completed.
  Id id() const;

 private:
  int num_bytes_remaining_ = -1;
  Id id_;
};

}  // namespace webm

#endif  // SRC_ID_PARSER_H_
