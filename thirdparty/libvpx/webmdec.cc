/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./webmdec.h"

#include <cstring>
#include <cstdio>

#include "third_party/libwebm/mkvparser/mkvparser.h"
#include "third_party/libwebm/mkvparser/mkvreader.h"

namespace {

void reset(struct WebmInputContext *const webm_ctx) {
  if (webm_ctx->reader != nullptr) {
    mkvparser::MkvReader *const reader =
        reinterpret_cast<mkvparser::MkvReader *>(webm_ctx->reader);
    delete reader;
  }
  if (webm_ctx->segment != nullptr) {
    mkvparser::Segment *const segment =
        reinterpret_cast<mkvparser::Segment *>(webm_ctx->segment);
    delete segment;
  }
  if (webm_ctx->buffer != nullptr) {
    delete[] webm_ctx->buffer;
  }
  webm_ctx->reader = nullptr;
  webm_ctx->segment = nullptr;
  webm_ctx->buffer = nullptr;
  webm_ctx->cluster = nullptr;
  webm_ctx->block_entry = nullptr;
  webm_ctx->block = nullptr;
  webm_ctx->block_frame_index = 0;
  webm_ctx->video_track_index = 0;
  webm_ctx->timestamp_ns = 0;
  webm_ctx->is_key_frame = false;
}

void get_first_cluster(struct WebmInputContext *const webm_ctx) {
  mkvparser::Segment *const segment =
      reinterpret_cast<mkvparser::Segment *>(webm_ctx->segment);
  const mkvparser::Cluster *const cluster = segment->GetFirst();
  webm_ctx->cluster = cluster;
}

void rewind_and_reset(struct WebmInputContext *const webm_ctx,
                      struct VpxInputContext *const vpx_ctx) {
  rewind(vpx_ctx->file);
  reset(webm_ctx);
}

}  // namespace

int file_is_webm(struct WebmInputContext *webm_ctx,
                 struct VpxInputContext *vpx_ctx) {
  mkvparser::MkvReader *const reader = new mkvparser::MkvReader(vpx_ctx->file);
  webm_ctx->reader = reader;
  webm_ctx->reached_eos = 0;

  mkvparser::EBMLHeader header;
  long long pos = 0;
  if (header.Parse(reader, pos) < 0) {
    rewind_and_reset(webm_ctx, vpx_ctx);
    return 0;
  }

  mkvparser::Segment *segment;
  if (mkvparser::Segment::CreateInstance(reader, pos, segment)) {
    rewind_and_reset(webm_ctx, vpx_ctx);
    return 0;
  }
  webm_ctx->segment = segment;
  if (segment->Load() < 0) {
    rewind_and_reset(webm_ctx, vpx_ctx);
    return 0;
  }

  const mkvparser::Tracks *const tracks = segment->GetTracks();
  const mkvparser::VideoTrack *video_track = nullptr;
  for (unsigned long i = 0; i < tracks->GetTracksCount(); ++i) {
    const mkvparser::Track *const track = tracks->GetTrackByIndex(i);
    if (track->GetType() == mkvparser::Track::kVideo) {
      video_track = static_cast<const mkvparser::VideoTrack *>(track);
      webm_ctx->video_track_index = static_cast<int>(track->GetNumber());
      break;
    }
  }

  if (video_track == nullptr || video_track->GetCodecId() == nullptr) {
    rewind_and_reset(webm_ctx, vpx_ctx);
    return 0;
  }

  if (!strncmp(video_track->GetCodecId(), "V_VP8", 5)) {
    vpx_ctx->fourcc = VP8_FOURCC;
  } else if (!strncmp(video_track->GetCodecId(), "V_VP9", 5)) {
    vpx_ctx->fourcc = VP9_FOURCC;
  } else {
    rewind_and_reset(webm_ctx, vpx_ctx);
    return 0;
  }

  vpx_ctx->framerate.denominator = 0;
  vpx_ctx->framerate.numerator = 0;
  vpx_ctx->width = static_cast<uint32_t>(video_track->GetWidth());
  vpx_ctx->height = static_cast<uint32_t>(video_track->GetHeight());

  get_first_cluster(webm_ctx);

  return 1;
}

int webm_read_frame(struct WebmInputContext *webm_ctx, uint8_t **buffer,
                    size_t *buffer_size) {
  // This check is needed for frame parallel decoding, in which case this
  // function could be called even after it has reached end of input stream.
  if (webm_ctx->reached_eos) {
    return 1;
  }
  mkvparser::Segment *const segment =
      reinterpret_cast<mkvparser::Segment *>(webm_ctx->segment);
  const mkvparser::Cluster *cluster =
      reinterpret_cast<const mkvparser::Cluster *>(webm_ctx->cluster);
  const mkvparser::Block *block =
      reinterpret_cast<const mkvparser::Block *>(webm_ctx->block);
  const mkvparser::BlockEntry *block_entry =
      reinterpret_cast<const mkvparser::BlockEntry *>(webm_ctx->block_entry);
  bool block_entry_eos = false;
  do {
    long status = 0;
    bool get_new_block = false;
    if (block_entry == nullptr && !block_entry_eos) {
      status = cluster->GetFirst(block_entry);
      get_new_block = true;
    } else if (block_entry_eos || block_entry->EOS()) {
      cluster = segment->GetNext(cluster);
      if (cluster == nullptr || cluster->EOS()) {
        *buffer_size = 0;
        webm_ctx->reached_eos = 1;
        return 1;
      }
      status = cluster->GetFirst(block_entry);
      block_entry_eos = false;
      get_new_block = true;
    } else if (block == nullptr ||
               webm_ctx->block_frame_index == block->GetFrameCount() ||
               block->GetTrackNumber() != webm_ctx->video_track_index) {
      status = cluster->GetNext(block_entry, block_entry);
      if (block_entry == nullptr || block_entry->EOS()) {
        block_entry_eos = true;
        continue;
      }
      get_new_block = true;
    }
    if (status || block_entry == nullptr) {
      return -1;
    }
    if (get_new_block) {
      block = block_entry->GetBlock();
      if (block == nullptr) return -1;
      webm_ctx->block_frame_index = 0;
    }
  } while (block_entry_eos ||
           block->GetTrackNumber() != webm_ctx->video_track_index);

  webm_ctx->cluster = cluster;
  webm_ctx->block_entry = block_entry;
  webm_ctx->block = block;

  const mkvparser::Block::Frame &frame =
      block->GetFrame(webm_ctx->block_frame_index);
  ++webm_ctx->block_frame_index;
  if (frame.len > static_cast<long>(*buffer_size)) {
    delete[] *buffer;
    *buffer = new uint8_t[frame.len];
    if (*buffer == nullptr) {
      return -1;
    }
    webm_ctx->buffer = *buffer;
  }
  *buffer_size = frame.len;
  webm_ctx->timestamp_ns = block->GetTime(cluster);
  webm_ctx->is_key_frame = block->IsKey();

  mkvparser::MkvReader *const reader =
      reinterpret_cast<mkvparser::MkvReader *>(webm_ctx->reader);
  return frame.Read(reader, *buffer) ? -1 : 0;
}

int webm_guess_framerate(struct WebmInputContext *webm_ctx,
                         struct VpxInputContext *vpx_ctx) {
  uint32_t i = 0;
  uint8_t *buffer = nullptr;
  size_t buffer_size = 0;
  while (webm_ctx->timestamp_ns < 1000000000 && i < 50) {
    if (webm_read_frame(webm_ctx, &buffer, &buffer_size)) {
      break;
    }
    ++i;
  }
  vpx_ctx->framerate.numerator = (i - 1) * 1000000;
  vpx_ctx->framerate.denominator =
      static_cast<int>(webm_ctx->timestamp_ns / 1000);
  delete[] buffer;
  // webm_ctx->buffer is assigned to the buffer pointer in webm_read_frame().
  webm_ctx->buffer = nullptr;

  get_first_cluster(webm_ctx);
  webm_ctx->block = nullptr;
  webm_ctx->block_entry = nullptr;
  webm_ctx->block_frame_index = 0;
  webm_ctx->timestamp_ns = 0;
  webm_ctx->reached_eos = 0;

  return 0;
}

void webm_free(struct WebmInputContext *webm_ctx) { reset(webm_ctx); }
