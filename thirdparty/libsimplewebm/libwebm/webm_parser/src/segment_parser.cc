// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/segment_parser.h"

#include "src/chapters_parser.h"
#include "src/cluster_parser.h"
#include "src/cues_parser.h"
#include "src/info_parser.h"
#include "src/seek_head_parser.h"
#include "src/skip_callback.h"
#include "src/tags_parser.h"
#include "src/tracks_parser.h"
#include "webm/id.h"

namespace webm {

SegmentParser::SegmentParser()
    : MasterParser(MakeChild<ChaptersParser>(Id::kChapters),
                   MakeChild<ClusterParser>(Id::kCluster),
                   MakeChild<CuesParser>(Id::kCues),
                   MakeChild<InfoParser>(Id::kInfo),
                   MakeChild<SeekHeadParser>(Id::kSeekHead),
                   MakeChild<TagsParser>(Id::kTags),
                   MakeChild<TracksParser>(Id::kTracks)) {}

Status SegmentParser::Init(const ElementMetadata& metadata,
                           std::uint64_t max_size) {
  assert(metadata.size == kUnknownElementSize || metadata.size <= max_size);

  begin_done_ = false;
  parse_completed_ = false;
  return MasterParser::Init(metadata, max_size);
}

void SegmentParser::InitAfterSeek(const Ancestory& child_ancestory,
                                  const ElementMetadata& child_metadata) {
  MasterParser::InitAfterSeek(child_ancestory, child_metadata);

  begin_done_ = true;
  parse_completed_ = false;
  action_ = Action::kRead;
}

Status SegmentParser::Feed(Callback* callback, Reader* reader,
                           std::uint64_t* num_bytes_read) {
  assert(callback != nullptr);
  assert(reader != nullptr);
  assert(num_bytes_read != nullptr);

  *num_bytes_read = 0;

  if (!begin_done_) {
    const ElementMetadata metadata{Id::kSegment, header_size(), size(),
                                   position()};
    const Status status = callback->OnSegmentBegin(metadata, &action_);
    if (!status.completed_ok()) {
      return status;
    }
    begin_done_ = true;
  }

  SkipCallback skip_callback;
  if (action_ == Action::kSkip) {
    callback = &skip_callback;
  }

  if (!parse_completed_) {
    const Status status = MasterParser::Feed(callback, reader, num_bytes_read);
    if (!status.completed_ok()) {
      return status;
    }
    parse_completed_ = true;
  }

  return callback->OnSegmentEnd(
      {Id::kSegment, header_size(), size(), position()});
}

bool SegmentParser::WasSkipped() const { return action_ == Action::kSkip; }

}  // namespace webm
