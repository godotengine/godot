// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "webm/callback.h"

#include <cassert>

namespace webm {

Status Callback::OnElementBegin(const ElementMetadata& /* metadata */,
                                Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnUnknownElement(const ElementMetadata& /* metadata */,
                                  Reader* reader,
                                  std::uint64_t* bytes_remaining) {
  assert(reader != nullptr);
  assert(bytes_remaining != nullptr);
  return Skip(reader, bytes_remaining);
}

Status Callback::OnEbml(const ElementMetadata& /* metadata */,
                        const Ebml& /* ebml */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnVoid(const ElementMetadata& /* metadata */, Reader* reader,
                        std::uint64_t* bytes_remaining) {
  assert(reader != nullptr);
  assert(bytes_remaining != nullptr);
  return Skip(reader, bytes_remaining);
}

Status Callback::OnSegmentBegin(const ElementMetadata& /* metadata */,
                                Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnSeek(const ElementMetadata& /* metadata */,
                        const Seek& /* seek */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnInfo(const ElementMetadata& /* metadata */,
                        const Info& /* info */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnClusterBegin(const ElementMetadata& /* metadata */,
                                const Cluster& /* cluster */, Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnSimpleBlockBegin(const ElementMetadata& /* metadata */,
                                    const SimpleBlock& /* simple_block */,
                                    Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnSimpleBlockEnd(const ElementMetadata& /* metadata */,
                                  const SimpleBlock& /* simple_block */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnBlockGroupBegin(const ElementMetadata& /* metadata */,
                                   Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnBlockBegin(const ElementMetadata& /* metadata */,
                              const Block& /* block */, Action* action) {
  assert(action != nullptr);
  *action = Action::kRead;
  return Status(Status::kOkCompleted);
}

Status Callback::OnBlockEnd(const ElementMetadata& /* metadata */,
                            const Block& /* block */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnBlockGroupEnd(const ElementMetadata& /* metadata */,
                                 const BlockGroup& /* block_group */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnFrame(const FrameMetadata& /* metadata */, Reader* reader,
                         std::uint64_t* bytes_remaining) {
  assert(reader != nullptr);
  assert(bytes_remaining != nullptr);
  return Skip(reader, bytes_remaining);
}

Status Callback::OnClusterEnd(const ElementMetadata& /* metadata */,
                              const Cluster& /* cluster */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnTrackEntry(const ElementMetadata& /* metadata */,
                              const TrackEntry& /* track_entry */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnCuePoint(const ElementMetadata& /* metadata */,
                            const CuePoint& /* cue_point */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnEditionEntry(const ElementMetadata& /* metadata */,
                                const EditionEntry& /* edition_entry */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnTag(const ElementMetadata& /* metadata */,
                       const Tag& /* tag */) {
  return Status(Status::kOkCompleted);
}

Status Callback::OnSegmentEnd(const ElementMetadata& /* metadata */) {
  return Status(Status::kOkCompleted);
}

Status Callback::Skip(Reader* reader, std::uint64_t* bytes_remaining) {
  assert(reader != nullptr);
  assert(bytes_remaining != nullptr);

  if (*bytes_remaining == 0)
    return Status(Status::kOkCompleted);

  Status status;
  do {
    std::uint64_t num_actually_skipped;
    status = reader->Skip(*bytes_remaining, &num_actually_skipped);
    *bytes_remaining -= num_actually_skipped;
  } while (status.code == Status::kOkPartial);

  return status;
}

}  // namespace webm
