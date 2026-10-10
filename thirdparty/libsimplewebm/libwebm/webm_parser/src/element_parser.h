// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef SRC_ELEMENT_PARSER_H_
#define SRC_ELEMENT_PARSER_H_

#include <cassert>
#include <cstdint>

#include "src/ancestory.h"
#include "src/parser.h"
#include "webm/callback.h"
#include "webm/element.h"

namespace webm {

// Parses an element from a WebM byte stream. Objects that implement this
// interface are expected to be used as follows in order to parse the specific
// WebM element that they are designed to handle.
//
// Reader* reader = ...;               // Create some Reader.
// Callback* callback = ...;           // Create some Callback.
//
// ElementMetadata metadata = {
//     id,             // Element parsed from the reader.
//     header_size,    // The number of bytes used to encode the id and size.
//     size_in_bytes,  // The number of bytes in the element body.
//     position,       // The position of the element (starting at the ID).
// };
//
// std::uint64_t max_size = ...;  // Some upper bound on this element's size.
// ElementParser* parser = ...;   // Create some parser capable of handling
//                                // elements that match id.
//
// Status status = parser->Init(metadata, max_size);
// if (!status.completed_ok()) {
//   // An error occurred. See status.code for the reason.
// } else {
//   do {
//     std::uint64_t num_bytes_read = 0;
//     status = parser->Feed(callback, reader, &num_bytes_read);
//   } while (status.code == Status::kOkPartial);
//
//   if (status.completed_ok()) {
//     // Parsing successfully completed.
//   } else {
//     // An error occurred. If status.code is a parsing error (see status.h for
//     // errors that are considered parsing errors), do not call Feed again;
//     // parsing has already failed and further progress can't be made. If
//     // status.code is not a parsing error (i.e. Status::kWouldBlock), then
//     // Feed may be called again to attempt resuming parsing.
//   }
// }
class ElementParser : public Parser {
 public:
  // Initializes the parser and prepares it for parsing its element. Returns
  // Status::kOkCompleted if successful. Must not return Status::kOkPartial (it
  // is not resumable). metadata is the metadata associated with this element.
  // max_size must be <= metadata.size (unless metadata.size is
  // kUnknownElementSize).
  virtual Status Init(const ElementMetadata& metadata,
                      std::uint64_t max_size) = 0;

  // Initializes the parser after a seek was done and prepares it for parsing.
  // The reader is now at the position of the child element indicated by
  // child_metadata, whose ancestory is child_ancestory. The child element for
  // this parser is the first element in child_ancestory, or if that is empty,
  // then child_metadata itself. If the child is not a valid child of this
  // parser, then a debug assertion is made (because that indicates a bug).
  virtual void InitAfterSeek(const Ancestory& /* child_ancestory */,
                             const ElementMetadata& /* child_metadata */) {
    assert(false);
  }

  // Returns true and sets metadata if this parser read too far and read the
  // element metadata for an element that is not its child. This may happen, for
  // example, when an element with unknown size is being read (because its end
  // is considered the first element that is not a valid child, so it must read
  // further to detect this). If this did not happen and false is returned, then
  // metadata will not be modified. metadata must not be null.
  virtual bool GetCachedMetadata(ElementMetadata* metadata) {
    assert(metadata != nullptr);

    return false;
  }

  // Returns true if this parser skipped the element instead of fully parsing
  // it. This will be true if the user requested a kSkip action from the
  // Callback in Feed(). This method should only be called after Feed() has
  // returned kOkCompleted. If the element was skipped, do not try to access its
  // value; it has no meaningful value and doing so will likely result in an
  // assertion failing.
  virtual bool WasSkipped() const { return false; }
};

}  // namespace webm

#endif  // SRC_ELEMENT_PARSER_H_
