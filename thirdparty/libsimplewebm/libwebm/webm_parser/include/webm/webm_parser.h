// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_WEBM_PARSER_H_
#define INCLUDE_WEBM_WEBM_PARSER_H_

#include <memory>

#include "./callback.h"
#include "./reader.h"
#include "./status.h"

/**
 \file
 The main parser class for parsing WebM files.
 */

namespace webm {

/**
 \defgroup PUBLIC_API Public API
 Public types and header files intended for use by users.
 */

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 Incrementally parses a WebM document encoded in a byte stream.

 It is expected that the parsing will begin at the start of the WebM
 file/document. Otherwise, the `DidSeek()` method should be used to inform the
 parser that reading may not necessarily begin at the beginning of the document
 and the parser should be prepared to handle data at an arbitrary point in the
 document.

 WebM files are mostly a subset of Matroska, with a few small modifications.
 Matroska is a format built on top of EBML.
 */
// Spec references:
// http://www.webmproject.org/docs/container/
// https://matroska.org/technical/specs/index.html
// https://github.com/Matroska-Org/ebml-specification/blob/master/specification.markdown
class WebmParser {
 public:
  /**
   Constructs the object and prepares it for parsing.
   */
  WebmParser();

  /*
   Cleans up the parser. No further cleanup is required from the user.
   */
  ~WebmParser();

  // Non-copyable and non-movable.
  WebmParser(const WebmParser&) = delete;
  WebmParser& operator=(const WebmParser&) = delete;

  /**
   Resets the parser after a seek was performed on the reader, which prepares
   the parser for starting at an arbitrary point in the stream.

   The seek must be to the start of an element (that is, it can't be any random
   byte) or parsing will fail.
   */
  void DidSeek();

  /**
   Feeds data into the parser from the provided reader.

   If a parsing error code has been returned (which indicates a malformed
   document), calling Feed() again will just result in the same error; parsing
   has already failed at that point and further progress can't be made.

   \param callback The callback which receives parsing events. No internal
   references are maintained to `callback`, so it may be modified or freed after
   this method returns.
   \param reader The reader the parser will use to request data. No internal
   references are maintained to `reader`, so it may be modified or freed after
   this method returns.
   \return `Status::kOkCompleted` when parsing completes successfully.
   `Status::kOkPartial` or another non-parsing error code (see status.h for
   error codes classified as parsing errors) will be returned if parsing has
   only partially completed, and Feed() should be called again to resume
   parsing.
   */
  Status Feed(Callback* callback, Reader* reader);

  /**
   Swaps this parser and the provided parser.

   \param other The parser to swap values/states with. Must not be null.
   */
  void Swap(WebmParser* other);

 private:
  // This header can only rely on sources in the public API.
  class DocumentParser;

  // The internal implementation of the parser.
  std::unique_ptr<DocumentParser> parser_;

  // The status of the parser, used to prevent further progress if an
  // unrecoverable parsing error has already been encountered.
  Status parsing_status_ = Status(Status::kOkPartial);
};

/**
 Swaps the two parsers.

 This is provided so code can use argument dependent lookup in an idiomatic way
 when swapping (especially since `std::swap` won't work since the parser is
 non-movable).

 \param left The first parser to swap.
 \param right The second parser to swap.
 */
void swap(WebmParser& left, WebmParser& right);

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_WEBM_PARSER_H_
