// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_CALLBACK_H_
#define INCLUDE_WEBM_CALLBACK_H_

#include <cstdint>

#include "./dom_types.h"
#include "./reader.h"
#include "./status.h"

/**
 \file
 The main callback type that receives parsing events.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 The action to be performed when parsing an element.
 */
enum class Action {
  /**
   Read and parse the element.
   */
  kRead,

  /**
   Skip the element. Skipped elements are not parsed or stored, and the callback
   is not given any further notifications regarding the element.
   */
  kSkip,
};

/**
 A callback that receives parsing events.

 Every method that returns a `Status` should return `Status::kOkCompleted` when
 the method has completed and parsing should continue. Returning any other value
 will cause parsing to stop. Parsing may be resumed if the returned status was
 not a parsing error (see `Status::is_parsing_error()`). When parsing is
 resumed, the same `Callback` method will be called again.

 Methods that take a `Reader` expect the implementation to consume (either via
 `Reader::Read()` or `Reader::Skip()`) the specified number of bytes before
 returning `Status::kOkCompleted`. Default implementations will call
 `Reader::Skip()` to skip the specified number of bytes and the resulting
 `Status` will be returned (unless it's `Status::kOkPartial`, in which case
 `Reader::Skip()` will be called again to skip more data).

 Throwing an exception from the member functions is permitted, though if the
 exception will be caught and parsing resumed, then the reader should not
 advance its position (for methods that take a `Reader`) before the exception is
 thrown. When parsing is resumed, the same `Callback` method will be called
 again.

 Users should derive from this class and override member methods as needed.
 */
class Callback {
 public:
  virtual ~Callback() = default;

  /**
   Called when the parser starts a new element. This is called after the
   elements ID and size has been parsed, but before any of its body has been
   read (or validated).

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element that has just been encountered.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnElementBegin(const ElementMetadata& metadata,
                                Action* action);

  /**
   Called when the parser encounters an unknown element.

   Defaults to calling (and returning the result of) `Reader::Skip()`.

   \param metadata Metadata about the element.
   \param reader The reader that should be used to consume data. Will not be
   null.
   \param[in,out] bytes_remaining The number of remaining bytes that need to be
   consumed for the element. Will not be null.
   \return `Status::kOkCompleted` when the element has been fully consumed and
   `bytes_remaining` is now zero.
   */
  virtual Status OnUnknownElement(const ElementMetadata& metadata,
                                  Reader* reader,
                                  std::uint64_t* bytes_remaining);

  /**
   Called when the parser encounters an `Id::kEbml` element and it has been
   fully parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param ebml The parsed element.
   */
  virtual Status OnEbml(const ElementMetadata& metadata, const Ebml& ebml);

  /**
   Called when the parser encounters an Id::kVoid element.

   Defaults to calling (and returning the result of) Reader::Skip.

   \param metadata Metadata about the element.
   \param reader The reader that should be used to consume data. Will not be
   null.
   \param[in,out] bytes_remaining The number of remaining bytes that need to be
   consumed for the element. Will not be null.
   \return `Status::kOkCompleted` when the element has been fully consumed and
   `bytes_remaining` is now zero.
   */
  virtual Status OnVoid(const ElementMetadata& metadata, Reader* reader,
                        std::uint64_t* bytes_remaining);

  /**
   Called when the parser starts an `Id::kSegment` element.

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element that has just been encountered.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnSegmentBegin(const ElementMetadata& metadata,
                                Action* action);

  /**
   Called when the parser encounters an `Id::kSeek` element and it has been
   fully parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param seek The parsed element.
   */
  virtual Status OnSeek(const ElementMetadata& metadata, const Seek& seek);

  /**
   Called when the parser encounters an `Id::kInfo` element and it has been
   fully parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param info The parsed element.
   */
  virtual Status OnInfo(const ElementMetadata& metadata, const Info& info);

  /**
   Called when the parser starts an `Id::kCluster` element.

   Because Cluster elements should start with a Timecode (and optionally
   PrevSize) child, this method is not invoked until a child BlockGroup or
   SimpleBlock element is encountered (or the Cluster ends if no such child
   exists).

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param cluster The element, as it has currently been parsed.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnClusterBegin(const ElementMetadata& metadata,
                                const Cluster& cluster, Action* action);

  /**
   Called when the parser starts an `Id::kSimpleBlock` element.

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param simple_block The parsed SimpleBlock header.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnSimpleBlockBegin(const ElementMetadata& metadata,
                                    const SimpleBlock& simple_block,
                                    Action* action);

  /**
   Called when the parser finishes an `Id::kSimpleBlock` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param simple_block The parsed SimpleBlock header.
   */
  virtual Status OnSimpleBlockEnd(const ElementMetadata& metadata,
                                  const SimpleBlock& simple_block);

  /**
   Called when the parser starts an `Id::kBlockGroup` element.

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnBlockGroupBegin(const ElementMetadata& metadata,
                                   Action* action);

  /**
   Called when the parser starts an `Id::kBlock` element.

   Defaults to `Action::kRead` and returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param block The parsed Block header.
   \param[out] action The action that should be taken when handling this
   element. Will not be null.
   */
  virtual Status OnBlockBegin(const ElementMetadata& metadata,
                              const Block& block, Action* action);

  /**
   Called when the parser finishes an `Id::Block` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param block The parsed Block header.
   */
  virtual Status OnBlockEnd(const ElementMetadata& metadata,
                            const Block& block);

  /**
   Called when the parser finishes an `Id::kBlockGroup` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param block_group The parsed element.
   */
  virtual Status OnBlockGroupEnd(const ElementMetadata& metadata,
                                 const BlockGroup& block_group);

  /**
   Called when the parser encounters a frame within a `Id::kBlock` or
   `Id::kSimpleBlock` element.

   Defaults to calling (and returning the result of) `Reader::Skip`.

   \param metadata Metadata about the frame.
   \param reader The reader that should be used to consume data. Will not be
   null.
   \param[in,out] bytes_remaining The number of remaining bytes that need to be
   consumed for the frame. Will not be null.
   \return `Status::kOkCompleted` when the frame has been fully consumed and
   `bytes_remaining` is now zero.
   */
  virtual Status OnFrame(const FrameMetadata& metadata, Reader* reader,
                         std::uint64_t* bytes_remaining);

  /**
   Called when the parser finishes an `Id::kCluster` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param cluster The parsed element.
   */
  virtual Status OnClusterEnd(const ElementMetadata& metadata,
                              const Cluster& cluster);

  /**
   Called when the parser starts an `Id::kTrackEntry` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param track_entry The parsed element.
   */
  virtual Status OnTrackEntry(const ElementMetadata& metadata,
                              const TrackEntry& track_entry);

  /**
   Called when the parser encounters an `Id::kCuePoint` element and it has been
   fully parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param cue_point The parsed element.
   */
  virtual Status OnCuePoint(const ElementMetadata& metadata,
                            const CuePoint& cue_point);

  /**
   Called when the parser encounters an `Id::kEditionEntry` element and it has
   been fully parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param edition_entry The parsed element.
   */
  virtual Status OnEditionEntry(const ElementMetadata& metadata,
                                const EditionEntry& edition_entry);

  /**
   Called when the parser encounters an `Id::kTag` element and it has been fully
   parsed.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   \param tag The parsed element.
   */
  virtual Status OnTag(const ElementMetadata& metadata, const Tag& tag);

  /**
   Called when the parser finishes an `Id::kSegment` element.

   Defaults to returning `Status::kOkCompleted`.

   \param metadata Metadata about the element.
   */
  virtual Status OnSegmentEnd(const ElementMetadata& metadata);

 protected:
  /**
   Calls (and returns the result of) `Reader::Skip()`, skipping (up to) the
   requested number of bytes.

   Unlike `Reader::Skip()`, this method may be called with `*bytes_remaining ==
   0`, which will result in `Status::kOkCompleted`. `Reader::Skip()` will be
   called multiple times if it returns `Status::kOkPartial` until it returns a
   different status (indicating the requested number of bytes has been fully
   skipped or some error occurred).

   \param reader The reader that should be used to skip data. Must not be null.
   \param[in,out] bytes_remaining The number of remaining bytes that need to be
   skipped. Must not be null. May be zero.
   \return The result of `Reader::Skip()`.
   */
  static Status Skip(Reader* reader, std::uint64_t* bytes_remaining);
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_CALLBACK_H_
