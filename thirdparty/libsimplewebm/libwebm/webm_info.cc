// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include <inttypes.h>
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "common/hdr_util.h"
#include "common/indent.h"
#include "common/vp9_header_parser.h"
#include "common/vp9_level_stats.h"
#include "common/webm_constants.h"
#include "common/webm_endian.h"
#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"

namespace {

using libwebm::Indent;
using libwebm::kNanosecondsPerSecond;
using libwebm::kNanosecondsPerSecondi;
using mkvparser::ContentEncoding;
using std::string;
using std::wstring;

const char VERSION_STRING[] = "1.0.4.5";

struct Options {
  Options();

  // Returns true if |value| matches -|option| or -no|option|.
  static bool MatchesBooleanOption(const string& option, const string& value);

  // Set all of the member variables to |value|.
  void SetAll(bool value);

  bool output_video;
  bool output_audio;
  bool output_size;
  bool output_offset;
  bool output_seconds;
  bool output_ebml_header;
  bool output_segment;
  bool output_seekhead;
  bool output_segment_info;
  bool output_tracks;
  bool output_clusters;
  bool output_blocks;
  bool output_codec_info;
  bool output_clusters_size;
  bool output_encrypted_info;
  bool output_cues;
  bool output_frame_stats;
  bool output_vp9_level;
};

Options::Options()
    : output_video(true),
      output_audio(true),
      output_size(false),
      output_offset(false),
      output_seconds(true),
      output_ebml_header(true),
      output_segment(true),
      output_seekhead(false),
      output_segment_info(true),
      output_tracks(true),
      output_clusters(false),
      output_blocks(false),
      output_codec_info(false),
      output_clusters_size(false),
      output_encrypted_info(false),
      output_cues(false),
      output_frame_stats(false),
      output_vp9_level(false) {}

void Options::SetAll(bool value) {
  output_video = value;
  output_audio = value;
  output_size = value;
  output_offset = value;
  output_ebml_header = value;
  output_seconds = value;
  output_segment = value;
  output_segment_info = value;
  output_tracks = value;
  output_clusters = value;
  output_blocks = value;
  output_codec_info = value;
  output_clusters_size = value;
  output_encrypted_info = value;
  output_cues = value;
  output_frame_stats = value;
  output_vp9_level = value;
}

bool Options::MatchesBooleanOption(const string& option, const string& value) {
  const string opt = "-" + option;
  const string noopt = "-no" + option;
  return value == opt || value == noopt;
}

struct FrameStats {
  FrameStats()
      : frames(0),
        displayed_frames(0),
        first_altref(true),
        frames_since_last_altref(0),
        minimum_altref_distance(std::numeric_limits<int>::max()),
        min_altref_end_ns(0),
        max_window_size(0),
        max_window_end_ns(0) {}

  int frames;
  int displayed_frames;

  bool first_altref;
  int frames_since_last_altref;
  int minimum_altref_distance;
  int64_t min_altref_end_ns;

  std::queue<int64_t> window;
  int64_t max_window_size;
  int64_t max_window_end_ns;
};

void Usage() {
  printf("Usage: webm_info [options] -i input\n");
  printf("\n");
  printf("Main options:\n");
  printf("  -h | -?               show help\n");
  printf("  -v                    show version\n");
  printf("  -all                  Enable all output options.\n");
  printf("  -video                Output video tracks (true)\n");
  printf("  -audio                Output audio tracks (true)\n");
  printf("  -size                 Output element sizes (false)\n");
  printf("  -offset               Output element offsets (false)\n");
  printf("  -times_seconds        Output times as seconds (true)\n");
  printf("  -ebml_header          Output EBML header (true)\n");
  printf("  -segment              Output Segment (true)\n");
  printf("  -seekhead             Output SeekHead (false)\n");
  printf("  -segment_info         Output SegmentInfo (true)\n");
  printf("  -tracks               Output Tracks (true)\n");
  printf("  -clusters             Output Clusters (false)\n");
  printf("  -blocks               Output Blocks (false)\n");
  printf("  -codec_info           Output video codec information (false)\n");
  printf("  -clusters_size        Output Total Clusters size (false)\n");
  printf("  -encrypted_info       Output encrypted frame info (false)\n");
  printf("  -cues                 Output Cues entries (false)\n");
  printf("  -frame_stats          Output frame stats (VP9)(false)\n");
  printf("  -vp9_level            Output VP9 level(false)\n");
  printf("\nOutput options may be negated by prefixing 'no'.\n");
}

// TODO(fgalligan): Add support for non-ascii.
wstring UTF8ToWideString(const char* str) {
  wstring wstr;

  if (str == NULL)
    return wstr;

  string temp_str(str, strlen(str));
  wstr.assign(temp_str.begin(), temp_str.end());

  return wstr;
}

string ToString(const char* str) { return string((str == NULL) ? "" : str); }

void OutputEBMLHeader(const mkvparser::EBMLHeader& ebml, FILE* o,
                      Indent* indent) {
  fprintf(o, "EBML Header:\n");
  indent->Adjust(libwebm::kIncreaseIndent);
  fprintf(o, "%sEBMLVersion       : %lld\n", indent->indent_str().c_str(),
          ebml.m_version);
  fprintf(o, "%sEBMLReadVersion   : %lld\n", indent->indent_str().c_str(),
          ebml.m_readVersion);
  fprintf(o, "%sEBMLMaxIDLength   : %lld\n", indent->indent_str().c_str(),
          ebml.m_maxIdLength);
  fprintf(o, "%sEBMLMaxSizeLength : %lld\n", indent->indent_str().c_str(),
          ebml.m_maxSizeLength);
  fprintf(o, "%sDoc Type          : %s\n", indent->indent_str().c_str(),
          ebml.m_docType);
  fprintf(o, "%sDocTypeVersion    : %lld\n", indent->indent_str().c_str(),
          ebml.m_docTypeVersion);
  fprintf(o, "%sDocTypeReadVersion: %lld\n", indent->indent_str().c_str(),
          ebml.m_docTypeReadVersion);
  indent->Adjust(libwebm::kDecreaseIndent);
}

void OutputSegment(const mkvparser::Segment& segment, const Options& options,
                   FILE* o) {
  fprintf(o, "Segment:");
  if (options.output_offset)
    fprintf(o, "  @: %lld", segment.m_element_start);
  if (options.output_size)
    fprintf(o, "  size: %lld",
            segment.m_size + segment.m_start - segment.m_element_start);
  fprintf(o, "\n");
}

bool OutputSeekHead(const mkvparser::Segment& segment, const Options& options,
                    FILE* o, Indent* indent) {
  const mkvparser::SeekHead* const seekhead = segment.GetSeekHead();
  if (!seekhead) {
    // SeekHeads are optional.
    return true;
  }

  fprintf(o, "%sSeekHead:", indent->indent_str().c_str());
  if (options.output_offset)
    fprintf(o, "  @: %lld", seekhead->m_element_start);
  if (options.output_size)
    fprintf(o, "  size: %lld", seekhead->m_element_size);
  fprintf(o, "\n");

  indent->Adjust(libwebm::kIncreaseIndent);

  for (int i = 0; i < seekhead->GetCount(); ++i) {
    const mkvparser::SeekHead::Entry* const entry = seekhead->GetEntry(i);
    if (!entry) {
      fprintf(stderr, "Error retrieving SeekHead entry #%d\n", i);
      return false;
    }

    fprintf(o, "%sEntry[%d]", indent->indent_str().c_str(), i);
    if (options.output_offset)
      fprintf(o, "  @: %lld", entry->element_start);
    if (options.output_size)
      fprintf(o, "  size: %lld", entry->element_size);
    fprintf(o, "\n");

    indent->Adjust(libwebm::kIncreaseIndent);
    std::string entry_indent = indent->indent_str();
    // TODO(jzern): 1) known ids could be stringified. 2) ids could be
    // reencoded to EBML for ease of lookup.
    fprintf(o, "%sSeek ID       : %llx\n", entry_indent.c_str(), entry->id);
    fprintf(o, "%sSeek position : %lld\n", entry_indent.c_str(), entry->pos);
    indent->Adjust(libwebm::kDecreaseIndent);
  }

  for (int i = 0; i < seekhead->GetVoidElementCount(); ++i) {
    const mkvparser::SeekHead::VoidElement* const entry =
        seekhead->GetVoidElement(i);
    if (!entry) {
      fprintf(stderr, "Error retrieving SeekHead void element #%d\n", i);
      return false;
    }

    fprintf(o, "%sVoid element[%d]", indent->indent_str().c_str(), i);
    if (options.output_offset)
      fprintf(o, "  @: %lld", entry->element_start);
    if (options.output_size)
      fprintf(o, "  size: %lld", entry->element_size);
    fprintf(o, "\n");
  }

  indent->Adjust(libwebm::kDecreaseIndent);
  return true;
}

bool OutputSegmentInfo(const mkvparser::Segment& segment,
                       const Options& options, FILE* o, Indent* indent) {
  const mkvparser::SegmentInfo* const segment_info = segment.GetInfo();
  if (!segment_info) {
    fprintf(stderr, "SegmentInfo was NULL.\n");
    return false;
  }

  const int64_t timecode_scale = segment_info->GetTimeCodeScale();
  const int64_t duration_ns = segment_info->GetDuration();
  const wstring title = UTF8ToWideString(segment_info->GetTitleAsUTF8());
  const wstring muxing_app =
      UTF8ToWideString(segment_info->GetMuxingAppAsUTF8());
  const wstring writing_app =
      UTF8ToWideString(segment_info->GetWritingAppAsUTF8());

  fprintf(o, "%sSegmentInfo:", indent->indent_str().c_str());
  if (options.output_offset)
    fprintf(o, "  @: %lld", segment_info->m_element_start);
  if (options.output_size)
    fprintf(o, "  size: %lld", segment_info->m_element_size);
  fprintf(o, "\n");

  indent->Adjust(libwebm::kIncreaseIndent);
  fprintf(o, "%sTimecodeScale : %" PRId64 " \n", indent->indent_str().c_str(),
          timecode_scale);
  if (options.output_seconds)
    fprintf(o, "%sDuration(secs): %g\n", indent->indent_str().c_str(),
            duration_ns / kNanosecondsPerSecond);
  else
    fprintf(o, "%sDuration(nano): %" PRId64 "\n", indent->indent_str().c_str(),
            duration_ns);

  if (!title.empty())
    fprintf(o, "%sTitle         : %ls\n", indent->indent_str().c_str(),
            title.c_str());
  if (!muxing_app.empty())
    fprintf(o, "%sMuxingApp     : %ls\n", indent->indent_str().c_str(),
            muxing_app.c_str());
  if (!writing_app.empty())
    fprintf(o, "%sWritingApp    : %ls\n", indent->indent_str().c_str(),
            writing_app.c_str());
  indent->Adjust(libwebm::kDecreaseIndent);
  return true;
}

bool OutputTracks(const mkvparser::Segment& segment, const Options& options,
                  FILE* o, Indent* indent) {
  const mkvparser::Tracks* const tracks = segment.GetTracks();
  if (!tracks) {
    fprintf(stderr, "Tracks was NULL.\n");
    return false;
  }

  fprintf(o, "%sTracks:", indent->indent_str().c_str());
  if (options.output_offset)
    fprintf(o, "  @: %lld", tracks->m_element_start);
  if (options.output_size)
    fprintf(o, "  size: %lld", tracks->m_element_size);
  fprintf(o, "\n");

  unsigned int i = 0;
  const unsigned long j = tracks->GetTracksCount();
  while (i != j) {
    const mkvparser::Track* const track = tracks->GetTrackByIndex(i++);
    if (track == NULL)
      continue;

    indent->Adjust(libwebm::kIncreaseIndent);
    fprintf(o, "%sTrack:", indent->indent_str().c_str());
    if (options.output_offset)
      fprintf(o, "  @: %lld", track->m_element_start);
    if (options.output_size)
      fprintf(o, "  size: %lld", track->m_element_size);
    fprintf(o, "\n");

    const int64_t track_type = track->GetType();
    const int64_t track_number = track->GetNumber();
    const wstring track_name = UTF8ToWideString(track->GetNameAsUTF8());

    indent->Adjust(libwebm::kIncreaseIndent);
    fprintf(o, "%sTrackType   : %" PRId64 "\n", indent->indent_str().c_str(),
            track_type);
    fprintf(o, "%sTrackNumber : %" PRId64 "\n", indent->indent_str().c_str(),
            track_number);
    if (!track_name.empty())
      fprintf(o, "%sName        : %ls\n", indent->indent_str().c_str(),
              track_name.c_str());

    const char* const codec_id = track->GetCodecId();
    if (codec_id)
      fprintf(o, "%sCodecID     : %s\n", indent->indent_str().c_str(),
              codec_id);

    const wstring codec_name = UTF8ToWideString(track->GetCodecNameAsUTF8());
    if (!codec_name.empty())
      fprintf(o, "%sCodecName   : %ls\n", indent->indent_str().c_str(),
              codec_name.c_str());

    size_t private_size;
    const unsigned char* const private_data =
        track->GetCodecPrivate(private_size);
    if (private_data) {
      fprintf(o, "%sPrivateData(size): %d\n", indent->indent_str().c_str(),
              static_cast<int>(private_size));

      if (track_type == mkvparser::Track::kVideo) {
        const std::string codec_id = ToString(track->GetCodecId());
        const std::string v_vp9 = "V_VP9";
        if (codec_id == v_vp9) {
          libwebm::Vp9CodecFeatures features;
          if (!libwebm::ParseVpxCodecPrivate(private_data,
                                             static_cast<int32_t>(private_size),
                                             &features)) {
            fprintf(stderr, "Error parsing VpxCodecPrivate.\n");
            return false;
          }
          if (features.profile != -1)
            fprintf(o, "%sVP9 profile            : %d\n",
                    indent->indent_str().c_str(), features.profile);
          if (features.level != -1)
            fprintf(o, "%sVP9 level              : %d\n",
                    indent->indent_str().c_str(), features.level);
          if (features.bit_depth != -1)
            fprintf(o, "%sVP9 bit_depth          : %d\n",
                    indent->indent_str().c_str(), features.bit_depth);
          if (features.chroma_subsampling != -1)
            fprintf(o, "%sVP9 chroma subsampling : %d\n",
                    indent->indent_str().c_str(), features.chroma_subsampling);
        }
      }
    }

    const uint64_t default_duration = track->GetDefaultDuration();
    if (default_duration > 0)
      fprintf(o, "%sDefaultDuration: %" PRIu64 "\n",
              indent->indent_str().c_str(), default_duration);

    if (track->GetContentEncodingCount() > 0) {
      // Only check the first content encoding.
      const ContentEncoding* const encoding =
          track->GetContentEncodingByIndex(0);
      if (!encoding) {
        printf("Could not get first ContentEncoding.\n");
        return false;
      }

      fprintf(o, "%sContentEncodingOrder : %lld\n",
              indent->indent_str().c_str(), encoding->encoding_order());
      fprintf(o, "%sContentEncodingScope : %lld\n",
              indent->indent_str().c_str(), encoding->encoding_scope());
      fprintf(o, "%sContentEncodingType  : %lld\n",
              indent->indent_str().c_str(), encoding->encoding_type());

      if (encoding->GetEncryptionCount() > 0) {
        // Only check the first encryption.
        const ContentEncoding::ContentEncryption* const encryption =
            encoding->GetEncryptionByIndex(0);
        if (!encryption) {
          printf("Could not get first ContentEncryption.\n");
          return false;
        }

        fprintf(o, "%sContentEncAlgo       : %lld\n",
                indent->indent_str().c_str(), encryption->algo);

        if (encryption->key_id_len > 0) {
          fprintf(o, "%sContentEncKeyID      : ", indent->indent_str().c_str());
          for (int k = 0; k < encryption->key_id_len; ++k) {
            fprintf(o, "0x%02x, ", encryption->key_id[k]);
          }
          fprintf(o, "\n");
        }

        if (encryption->signature_len > 0) {
          fprintf(o, "%sContentSignature     : 0x",
                  indent->indent_str().c_str());
          for (int k = 0; k < encryption->signature_len; ++k) {
            fprintf(o, "%x", encryption->signature[k]);
          }
          fprintf(o, "\n");
        }

        if (encryption->sig_key_id_len > 0) {
          fprintf(o, "%sContentSigKeyID      : 0x",
                  indent->indent_str().c_str());
          for (int k = 0; k < encryption->sig_key_id_len; ++k) {
            fprintf(o, "%x", encryption->sig_key_id[k]);
          }
          fprintf(o, "\n");
        }

        fprintf(o, "%sContentSigAlgo       : %lld\n",
                indent->indent_str().c_str(), encryption->sig_algo);
        fprintf(o, "%sContentSigHashAlgo   : %lld\n",
                indent->indent_str().c_str(), encryption->sig_hash_algo);

        const ContentEncoding::ContentEncAESSettings& aes =
            encryption->aes_settings;
        fprintf(o, "%sCipherMode           : %lld\n",
                indent->indent_str().c_str(), aes.cipher_mode);
      }
    }

    if (track_type == mkvparser::Track::kVideo) {
      const mkvparser::VideoTrack* const video_track =
          static_cast<const mkvparser::VideoTrack*>(track);
      const int64_t width = video_track->GetWidth();
      const int64_t height = video_track->GetHeight();
      const int64_t display_width = video_track->GetDisplayWidth();
      const int64_t display_height = video_track->GetDisplayHeight();
      const int64_t display_unit = video_track->GetDisplayUnit();
      const double frame_rate = video_track->GetFrameRate();
      fprintf(o, "%sPixelWidth  : %" PRId64 "\n", indent->indent_str().c_str(),
              width);
      fprintf(o, "%sPixelHeight : %" PRId64 "\n", indent->indent_str().c_str(),
              height);
      if (frame_rate > 0.0)
        fprintf(o, "%sFrameRate   : %g\n", indent->indent_str().c_str(),
                video_track->GetFrameRate());
      if (display_unit > 0 || display_width != width ||
          display_height != height) {
        fprintf(o, "%sDisplayWidth  : %" PRId64 "\n",
                indent->indent_str().c_str(), display_width);
        fprintf(o, "%sDisplayHeight : %" PRId64 "\n",
                indent->indent_str().c_str(), display_height);
        fprintf(o, "%sDisplayUnit   : %" PRId64 "\n",
                indent->indent_str().c_str(), display_unit);
      }

      const mkvparser::Colour* const colour = video_track->GetColour();
      if (colour) {
        // TODO(fgalligan): Add support for Colour's address and size.
        fprintf(o, "%sColour:\n", indent->indent_str().c_str());
        indent->Adjust(libwebm::kIncreaseIndent);

        const int64_t matrix_coefficients = colour->matrix_coefficients;
        const int64_t bits_per_channel = colour->bits_per_channel;
        const int64_t chroma_subsampling_horz = colour->chroma_subsampling_horz;
        const int64_t chroma_subsampling_vert = colour->chroma_subsampling_vert;
        const int64_t cb_subsampling_horz = colour->cb_subsampling_horz;
        const int64_t cb_subsampling_vert = colour->cb_subsampling_vert;
        const int64_t chroma_siting_horz = colour->chroma_siting_horz;
        const int64_t chroma_siting_vert = colour->chroma_siting_vert;
        const int64_t range = colour->range;
        const int64_t transfer_characteristics =
            colour->transfer_characteristics;
        const int64_t primaries = colour->primaries;
        const int64_t max_cll = colour->max_cll;
        const int64_t max_fall = colour->max_fall;
        if (matrix_coefficients != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sMatrixCoefficients      : %" PRId64 "\n",
                  indent->indent_str().c_str(), matrix_coefficients);
        if (bits_per_channel != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sBitsPerChannel          : %" PRId64 "\n",
                  indent->indent_str().c_str(), bits_per_channel);
        if (chroma_subsampling_horz != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sChromaSubsamplingHorz   : %" PRId64 "\n",
                  indent->indent_str().c_str(), chroma_subsampling_horz);
        if (chroma_subsampling_vert != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sChromaSubsamplingVert   : %" PRId64 "\n",
                  indent->indent_str().c_str(), chroma_subsampling_vert);
        if (cb_subsampling_horz != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sCbSubsamplingHorz       : %" PRId64 "\n",
                  indent->indent_str().c_str(), cb_subsampling_horz);
        if (cb_subsampling_vert != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sCbSubsamplingVert       : %" PRId64 "\n",
                  indent->indent_str().c_str(), cb_subsampling_vert);
        if (chroma_siting_horz != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sChromaSitingHorz        : %" PRId64 "\n",
                  indent->indent_str().c_str(), chroma_siting_horz);
        if (chroma_siting_vert != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sChromaSitingVert        : %" PRId64 "\n",
                  indent->indent_str().c_str(), chroma_siting_vert);
        if (range != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sRange                   : %" PRId64 "\n",
                  indent->indent_str().c_str(), range);
        if (transfer_characteristics != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sTransferCharacteristics : %" PRId64 "\n",
                  indent->indent_str().c_str(), transfer_characteristics);
        if (primaries != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sPrimaries               : %" PRId64 "\n",
                  indent->indent_str().c_str(), primaries);
        if (max_cll != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sMaxCLL                  : %" PRId64 "\n",
                  indent->indent_str().c_str(), max_cll);
        if (max_fall != mkvparser::Colour::kValueNotPresent)
          fprintf(o, "%sMaxFALL                 : %" PRId64 "\n",
                  indent->indent_str().c_str(), max_fall);

        const mkvparser::MasteringMetadata* const metadata =
            colour->mastering_metadata;
        if (metadata) {
          // TODO(fgalligan): Add support for MasteringMetadata's address and
          // size.
          fprintf(o, "%sMasteringMetadata:\n", indent->indent_str().c_str());
          indent->Adjust(libwebm::kIncreaseIndent);

          const mkvparser::PrimaryChromaticity* const red = metadata->r;
          const mkvparser::PrimaryChromaticity* const green = metadata->g;
          const mkvparser::PrimaryChromaticity* const blue = metadata->b;
          const mkvparser::PrimaryChromaticity* const white =
              metadata->white_point;
          const float max = metadata->luminance_max;
          const float min = metadata->luminance_min;
          if (red) {
            fprintf(o, "%sPrimaryRChromaticityX   : %g\n",
                    indent->indent_str().c_str(), red->x);
            fprintf(o, "%sPrimaryRChromaticityY   : %g\n",
                    indent->indent_str().c_str(), red->y);
          }
          if (green) {
            fprintf(o, "%sPrimaryGChromaticityX   : %g\n",
                    indent->indent_str().c_str(), green->x);
            fprintf(o, "%sPrimaryGChromaticityY   : %g\n",
                    indent->indent_str().c_str(), green->y);
          }
          if (blue) {
            fprintf(o, "%sPrimaryBChromaticityX   : %g\n",
                    indent->indent_str().c_str(), blue->x);
            fprintf(o, "%sPrimaryBChromaticityY   : %g\n",
                    indent->indent_str().c_str(), blue->y);
          }
          if (white) {
            fprintf(o, "%sWhitePointChromaticityX : %g\n",
                    indent->indent_str().c_str(), white->x);
            fprintf(o, "%sWhitePointChromaticityY : %g\n",
                    indent->indent_str().c_str(), white->y);
          }
          if (max != mkvparser::MasteringMetadata::kValueNotPresent)
            fprintf(o, "%sLuminanceMax            : %g\n",
                    indent->indent_str().c_str(), max);
          if (min != mkvparser::MasteringMetadata::kValueNotPresent)
            fprintf(o, "%sLuminanceMin            : %g\n",
                    indent->indent_str().c_str(), min);
          indent->Adjust(libwebm::kDecreaseIndent);
        }
        indent->Adjust(libwebm::kDecreaseIndent);
      }

      const mkvparser::Projection* const projection =
          video_track->GetProjection();
      if (projection) {
        fprintf(o, "%sProjection:\n", indent->indent_str().c_str());
        indent->Adjust(libwebm::kIncreaseIndent);

        const int projection_type = static_cast<int>(projection->type);
        const int kTypeNotPresent =
            static_cast<int>(mkvparser::Projection::kTypeNotPresent);
        const float kValueNotPresent = mkvparser::Projection::kValueNotPresent;
        if (projection_type != kTypeNotPresent)
          fprintf(o, "%sProjectionType            : %d\n",
                  indent->indent_str().c_str(), projection_type);
        if (projection->private_data)
          fprintf(o, "%sProjectionPrivate(size)   : %d\n",
                  indent->indent_str().c_str(),
                  static_cast<int>(projection->private_data_length));
        if (projection->pose_yaw != kValueNotPresent)
          fprintf(o, "%sProjectionPoseYaw         : %g\n",
                  indent->indent_str().c_str(), projection->pose_yaw);
        if (projection->pose_pitch != kValueNotPresent)
          fprintf(o, "%sProjectionPosePitch       : %g\n",
                  indent->indent_str().c_str(), projection->pose_pitch);
        if (projection->pose_roll != kValueNotPresent)
          fprintf(o, "%sProjectionPoseRoll         : %g\n",
                  indent->indent_str().c_str(), projection->pose_roll);
        indent->Adjust(libwebm::kDecreaseIndent);
      }
    } else if (track_type == mkvparser::Track::kAudio) {
      const mkvparser::AudioTrack* const audio_track =
          static_cast<const mkvparser::AudioTrack*>(track);
      const int64_t channels = audio_track->GetChannels();
      const int64_t bit_depth = audio_track->GetBitDepth();
      const uint64_t codec_delay = audio_track->GetCodecDelay();
      const uint64_t seek_preroll = audio_track->GetSeekPreRoll();
      fprintf(o, "%sChannels         : %" PRId64 "\n",
              indent->indent_str().c_str(), channels);
      if (bit_depth > 0)
        fprintf(o, "%sBitDepth         : %" PRId64 "\n",
                indent->indent_str().c_str(), bit_depth);
      fprintf(o, "%sSamplingFrequency: %g\n", indent->indent_str().c_str(),
              audio_track->GetSamplingRate());
      if (codec_delay)
        fprintf(o, "%sCodecDelay       : %" PRIu64 "\n",
                indent->indent_str().c_str(), codec_delay);
      if (seek_preroll)
        fprintf(o, "%sSeekPreRoll      : %" PRIu64 "\n",
                indent->indent_str().c_str(), seek_preroll);
    }
    indent->Adjust(libwebm::kDecreaseIndent * 2);
  }

  return true;
}

// libvpx reference: vp9/vp9_dx_iface.c
void ParseSuperframeIndex(const uint8_t* data, size_t data_sz,
                          uint32_t sizes[8], int* count) {
  const uint8_t marker = data[data_sz - 1];
  *count = 0;

  if ((marker & 0xe0) == 0xc0) {
    const int frames = (marker & 0x7) + 1;
    const int mag = ((marker >> 3) & 0x3) + 1;
    const size_t index_sz = 2 + mag * frames;

    if (data_sz >= index_sz && data[data_sz - index_sz] == marker) {
      // found a valid superframe index
      const uint8_t* x = data + data_sz - index_sz + 1;

      for (int i = 0; i < frames; ++i) {
        uint32_t this_sz = 0;

        for (int j = 0; j < mag; ++j) {
          this_sz |= (*x++) << (j * 8);
        }
        sizes[i] = this_sz;
      }
      *count = frames;
    }
  }
}

void PrintVP9Info(const uint8_t* data, int size, FILE* o, int64_t time_ns,
                  FrameStats* stats, vp9_parser::Vp9HeaderParser* parser,
                  vp9_parser::Vp9LevelStats* level_stats) {
  if (size < 1)
    return;

  uint32_t sizes[8];
  int i = 0, count = 0;
  ParseSuperframeIndex(data, size, sizes, &count);

  // Remove all frames that are less than window size.
  while (!stats->window.empty() &&
         stats->window.front() < (time_ns - (kNanosecondsPerSecondi - 1)))
    stats->window.pop();

  do {
    const size_t frame_length = (count > 0) ? sizes[i] : size;
    if (frame_length > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        static_cast<int>(frame_length) > size) {
      fprintf(o, " invalid VP9 frame size (%u)\n",
              static_cast<uint32_t>(frame_length));
      return;
    }
    if (!parser->ParseUncompressedHeader(data, frame_length))
      return;
    level_stats->AddFrame(*parser, time_ns);

    // const int frame_marker = (data[0] >> 6) & 0x3;
    const int version = parser->profile();
    const int key = parser->key();
    const int altref_frame = parser->altref();
    const int error_resilient_mode = parser->error_resilient_mode();
    const int row_tiles = parser->row_tiles();
    const int column_tiles = parser->column_tiles();
    const int frame_parallel_mode = parser->frame_parallel_mode();

    if (key &&
        !(size >= 4 && data[1] == 0x49 && data[2] == 0x83 && data[3] == 0x42)) {
      fprintf(o, " invalid VP9 signature");
      return;
    }

    stats->window.push(time_ns);
    ++stats->frames;

    if (altref_frame) {
      const int delta_altref = stats->frames_since_last_altref;
      if (stats->first_altref) {
        stats->first_altref = false;
      } else if (delta_altref < stats->minimum_altref_distance) {
        stats->minimum_altref_distance = delta_altref;
        stats->min_altref_end_ns = time_ns;
      }
      stats->frames_since_last_altref = 0;
    } else {
      ++stats->frames_since_last_altref;
      ++stats->displayed_frames;
    }

    if (count > 0) {
      fprintf(o, " packed [%d]: {", i);
    }

    fprintf(o, " key:%d v:%d altref:%d errm:%d rt:%d ct:%d fpm:%d", key,
            version, altref_frame, error_resilient_mode, row_tiles,
            column_tiles, frame_parallel_mode);

    if (key) {
      if (size > 4) {
        fprintf(o, " cs:%d", parser->color_space());
      }
      if (parser->display_width() != parser->width() ||
          parser->display_height() != parser->height()) {
        fprintf(o, " dw:%d dh:%d", parser->display_width(),
                parser->display_height());
      }
    }

    if (count > 0) {
      fprintf(o, " size: %u }", sizes[i]);
      data += sizes[i];
      size -= sizes[i];
    }
    ++i;
  } while (i < count);

  if (stats->max_window_size < static_cast<int64_t>(stats->window.size())) {
    stats->max_window_size = stats->window.size();
    stats->max_window_end_ns = time_ns;
  }
}

void PrintVP8Info(const uint8_t* data, int size, FILE* o) {
  if (size < 3)
    return;

  const uint32_t bits = data[0] | (data[1] << 8) | (data[2] << 16);
  const int key = !(bits & 0x1);
  const int altref_frame = !((bits >> 4) & 0x1);
  const int version = (bits >> 1) & 0x7;
  const int partition_length = (bits >> 5) & 0x7FFFF;
  if (key &&
      !(size >= 6 && data[3] == 0x9d && data[4] == 0x01 && data[5] == 0x2a)) {
    fprintf(o, " invalid VP8 signature");
    return;
  }
  fprintf(o, " key:%d v:%d altref:%d partition_length:%d", key, version,
          altref_frame, partition_length);
}

// Prints the partition offsets of the sub-sample encryption. |data| must point
// to an encrypted frame just after the signal byte. Returns the number of
// bytes read from the sub-sample partition information.
int PrintSubSampleEncryption(const uint8_t* data, int size, FILE* o) {
  int read_end = sizeof(uint64_t);

  // Skip past IV.
  if (size < read_end)
    return 0;
  data += sizeof(uint64_t);

  // Read number of partitions.
  read_end += sizeof(uint8_t);
  if (size < read_end)
    return 0;
  const int num_partitions = data[0];
  data += sizeof(uint8_t);

  // Read partitions.
  for (int i = 0; i < num_partitions; ++i) {
    read_end += sizeof(uint32_t);
    if (size < read_end)
      return 0;
    uint32_t partition_offset;
    memcpy(&partition_offset, data, sizeof(partition_offset));
    partition_offset = libwebm::bigendian_to_host(partition_offset);
    fprintf(o, " off[%d]:%u", i, partition_offset);
    data += sizeof(uint32_t);
  }

  return read_end;
}

bool OutputCluster(const mkvparser::Cluster& cluster,
                   const mkvparser::Tracks& tracks, const Options& options,
                   FILE* o, mkvparser::MkvReader* reader, Indent* indent,
                   int64_t* clusters_size, FrameStats* stats,
                   vp9_parser::Vp9HeaderParser* parser,
                   vp9_parser::Vp9LevelStats* level_stats) {
  if (clusters_size) {
    // Load the Cluster.
    const mkvparser::BlockEntry* block_entry;
    long status = cluster.GetFirst(block_entry);
    if (status) {
      fprintf(stderr, "Could not get first Block of Cluster.\n");
      return false;
    }

    *clusters_size += cluster.GetElementSize();
  }

  if (options.output_clusters) {
    const int64_t time_ns = cluster.GetTime();
    const int64_t duration_ns = cluster.GetLastTime() - cluster.GetFirstTime();

    fprintf(o, "%sCluster:", indent->indent_str().c_str());
    if (options.output_offset)
      fprintf(o, "  @: %lld", cluster.m_element_start);
    if (options.output_size)
      fprintf(o, "  size: %lld", cluster.GetElementSize());
    fprintf(o, "\n");
    indent->Adjust(libwebm::kIncreaseIndent);
    if (options.output_seconds)
      fprintf(o, "%sTimecode (sec) : %g\n", indent->indent_str().c_str(),
              time_ns / kNanosecondsPerSecond);
    else
      fprintf(o, "%sTimecode (nano): %" PRId64 "\n",
              indent->indent_str().c_str(), time_ns);
    if (options.output_seconds)
      fprintf(o, "%sDuration (sec) : %g\n", indent->indent_str().c_str(),
              duration_ns / kNanosecondsPerSecond);
    else
      fprintf(o, "%sDuration (nano): %" PRId64 "\n",
              indent->indent_str().c_str(), duration_ns);

    fprintf(o, "%s# Blocks       : %ld\n", indent->indent_str().c_str(),
            cluster.GetEntryCount());
  }

  if (options.output_blocks) {
    const mkvparser::BlockEntry* block_entry;
    long status = cluster.GetFirst(block_entry);
    if (status) {
      fprintf(stderr, "Could not get first Block of Cluster.\n");
      return false;
    }

    std::vector<unsigned char> vector_data;
    while (block_entry != NULL && !block_entry->EOS()) {
      const mkvparser::Block* const block = block_entry->GetBlock();
      if (!block) {
        fprintf(stderr, "Could not getblock entry.\n");
        return false;
      }

      const unsigned int track_number =
          static_cast<unsigned int>(block->GetTrackNumber());
      const mkvparser::Track* track = tracks.GetTrackByNumber(track_number);
      if (!track) {
        fprintf(stderr, "Could not get Track.\n");
        return false;
      }

      const int64_t track_type = track->GetType();
      if ((track_type == mkvparser::Track::kVideo && options.output_video) ||
          (track_type == mkvparser::Track::kAudio && options.output_audio)) {
        const int64_t time_ns = block->GetTime(&cluster);
        const bool is_key = block->IsKey();

        if (block_entry->GetKind() == mkvparser::BlockEntry::kBlockGroup) {
          fprintf(o, "%sBlockGroup:\n", indent->indent_str().c_str());
          indent->Adjust(libwebm::kIncreaseIndent);
        }

        fprintf(o, "%sBlock: type:%s frame:%s", indent->indent_str().c_str(),
                track_type == mkvparser::Track::kVideo ? "V" : "A",
                is_key ? "I" : "P");
        if (options.output_seconds)
          fprintf(o, " secs:%5g", time_ns / kNanosecondsPerSecond);
        else
          fprintf(o, " nano:%10" PRId64, time_ns);

        if (options.output_offset)
          fprintf(o, " @_payload: %lld", block->m_start);
        if (options.output_size)
          fprintf(o, " size_payload: %lld", block->m_size);

        const uint8_t KEncryptedBit = 0x1;
        const uint8_t kSubSampleBit = 0x2;
        const int kSignalByteSize = 1;
        bool encrypted_stream = false;
        if (options.output_encrypted_info) {
          if (track->GetContentEncodingCount() > 0) {
            // Only check the first content encoding.
            const ContentEncoding* const encoding =
                track->GetContentEncodingByIndex(0);
            if (encoding) {
              if (encoding->GetEncryptionCount() > 0) {
                const ContentEncoding::ContentEncryption* const encryption =
                    encoding->GetEncryptionByIndex(0);
                if (encryption) {
                  const ContentEncoding::ContentEncAESSettings& aes =
                      encryption->aes_settings;
                  if (aes.cipher_mode == 1) {
                    encrypted_stream = true;
                  }
                }
              }
            }
          }

          if (encrypted_stream) {
            const mkvparser::Block::Frame& frame = block->GetFrame(0);
            if (frame.len > static_cast<int>(vector_data.size())) {
              vector_data.resize(frame.len + 1024);
            }

            unsigned char* data = &vector_data[0];
            if (frame.Read(reader, data) < 0) {
              fprintf(stderr, "Could not read frame.\n");
              return false;
            }

            const bool encrypted_frame = !!(data[0] & KEncryptedBit);
            const bool sub_sample_encrypt = !!(data[0] & kSubSampleBit);
            fprintf(o, " enc: %d", encrypted_frame ? 1 : 0);
            fprintf(o, " sub: %d", sub_sample_encrypt ? 1 : 0);

            if (encrypted_frame) {
              uint64_t iv;
              memcpy(&iv, data + kSignalByteSize, sizeof(iv));
              fprintf(o, " iv: %" PRIx64, iv);
            }
          }
        }

        if (options.output_codec_info) {
          const int frame_count = block->GetFrameCount();

          if (frame_count > 1) {
            fprintf(o, "\n");
            indent->Adjust(libwebm::kIncreaseIndent);
          }

          for (int i = 0; i < frame_count; ++i) {
            if (track_type == mkvparser::Track::kVideo) {
              const mkvparser::Block::Frame& frame = block->GetFrame(i);
              if (frame.len > static_cast<int>(vector_data.size())) {
                vector_data.resize(frame.len + 1024);
              }

              unsigned char* data = &vector_data[0];
              if (frame.Read(reader, data) < 0) {
                fprintf(stderr, "Could not read frame.\n");
                return false;
              }

              if (frame_count > 1)
                fprintf(o, "\n%sVP8 data     :", indent->indent_str().c_str());

              bool encrypted_frame = false;
              bool sub_sample_encrypt = false;
              int frame_size = static_cast<int>(frame.len);

              int frame_offset = 0;
              if (encrypted_stream) {
                if (data[0] & KEncryptedBit) {
                  encrypted_frame = true;
                  if (data[0] & kSubSampleBit) {
                    sub_sample_encrypt = true;
                    data += kSignalByteSize;
                    frame_size -= kSignalByteSize;
                    frame_offset =
                        PrintSubSampleEncryption(data, frame_size, o);
                  }
                } else {
                  frame_offset = kSignalByteSize;
                }
              }

              if (!encrypted_frame || sub_sample_encrypt) {
                data += frame_offset;
                frame_size -= frame_offset;

                const string codec_id = ToString(track->GetCodecId());
                if (codec_id == "V_VP8") {
                  PrintVP8Info(data, frame_size, o);
                } else if (codec_id == "V_VP9") {
                  PrintVP9Info(data, frame_size, o, time_ns, stats, parser,
                               level_stats);
                }
              }
            }
          }

          if (frame_count > 1)
            indent->Adjust(libwebm::kDecreaseIndent);
        }

        if (block_entry->GetKind() == mkvparser::BlockEntry::kBlockGroup) {
          const int64_t discard_padding = block->GetDiscardPadding();
          if (discard_padding != 0) {
            fprintf(o, "\n%sDiscardPadding: %10" PRId64,
                    indent->indent_str().c_str(), discard_padding);
          }
          indent->Adjust(libwebm::kDecreaseIndent);
        }

        fprintf(o, "\n");
      }

      status = cluster.GetNext(block_entry, block_entry);
      if (status) {
        printf("\n Could not get next block of cluster.\n");
        return false;
      }
    }
  }

  if (options.output_clusters)
    indent->Adjust(libwebm::kDecreaseIndent);

  return true;
}

bool OutputCues(const mkvparser::Segment& segment,
                const mkvparser::Tracks& tracks, const Options& options,
                FILE* o, Indent* indent) {
  const mkvparser::Cues* const cues = segment.GetCues();
  if (cues == NULL)
    return true;

  // Load all of the cue points.
  while (!cues->DoneParsing())
    cues->LoadCuePoint();

  // Confirm that the input has cue points.
  const mkvparser::CuePoint* const first_cue = cues->GetFirst();
  if (first_cue == NULL) {
    fprintf(o, "%sNo cue points.\n", indent->indent_str().c_str());
    return true;
  }

  // Input has cue points, dump them:
  fprintf(o, "%sCues:", indent->indent_str().c_str());
  if (options.output_offset)
    fprintf(o, " @:%lld", cues->m_element_start);
  if (options.output_size)
    fprintf(o, " size:%lld", cues->m_element_size);
  fprintf(o, "\n");

  const mkvparser::CuePoint* cue_point = first_cue;
  int cue_point_num = 1;
  const int num_tracks = static_cast<int>(tracks.GetTracksCount());
  indent->Adjust(libwebm::kIncreaseIndent);

  do {
    for (int track_num = 0; track_num < num_tracks; ++track_num) {
      const mkvparser::Track* const track = tracks.GetTrackByIndex(track_num);
      const mkvparser::CuePoint::TrackPosition* const track_pos =
          cue_point->Find(track);

      if (track_pos != NULL) {
        const char track_type =
            (track->GetType() == mkvparser::Track::kVideo) ? 'V' : 'A';
        fprintf(o, "%sCue Point:%d type:%c track:%d",
                indent->indent_str().c_str(), cue_point_num, track_type,
                static_cast<int>(track->GetNumber()));

        if (options.output_seconds) {
          fprintf(o, " secs:%g",
                  cue_point->GetTime(&segment) / kNanosecondsPerSecond);
        } else {
          fprintf(o, " nano:%lld", cue_point->GetTime(&segment));
        }

        if (options.output_blocks)
          fprintf(o, " block:%lld", track_pos->m_block);

        if (options.output_offset)
          fprintf(o, " @:%lld", track_pos->m_pos);

        fprintf(o, "\n");
      }
    }

    cue_point = cues->GetNext(cue_point);
    ++cue_point_num;
  } while (cue_point != NULL);

  indent->Adjust(libwebm::kDecreaseIndent);
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  string input;
  Options options;

  const int argc_check = argc - 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp("-h", argv[i]) || !strcmp("-?", argv[i])) {
      Usage();
      return EXIT_SUCCESS;
    } else if (!strcmp("-v", argv[i])) {
      printf("version: %s\n", VERSION_STRING);
    } else if (!strcmp("-i", argv[i]) && i < argc_check) {
      input = argv[++i];
    } else if (!strcmp("-all", argv[i])) {
      options.SetAll(true);
    } else if (Options::MatchesBooleanOption("video", argv[i])) {
      options.output_video = !strcmp("-video", argv[i]);
    } else if (Options::MatchesBooleanOption("audio", argv[i])) {
      options.output_audio = !strcmp("-audio", argv[i]);
    } else if (Options::MatchesBooleanOption("size", argv[i])) {
      options.output_size = !strcmp("-size", argv[i]);
    } else if (Options::MatchesBooleanOption("offset", argv[i])) {
      options.output_offset = !strcmp("-offset", argv[i]);
    } else if (Options::MatchesBooleanOption("times_seconds", argv[i])) {
      options.output_seconds = !strcmp("-times_seconds", argv[i]);
    } else if (Options::MatchesBooleanOption("ebml_header", argv[i])) {
      options.output_ebml_header = !strcmp("-ebml_header", argv[i]);
    } else if (Options::MatchesBooleanOption("segment", argv[i])) {
      options.output_segment = !strcmp("-segment", argv[i]);
    } else if (Options::MatchesBooleanOption("seekhead", argv[i])) {
      options.output_seekhead = !strcmp("-seekhead", argv[i]);
    } else if (Options::MatchesBooleanOption("segment_info", argv[i])) {
      options.output_segment_info = !strcmp("-segment_info", argv[i]);
    } else if (Options::MatchesBooleanOption("tracks", argv[i])) {
      options.output_tracks = !strcmp("-tracks", argv[i]);
    } else if (Options::MatchesBooleanOption("clusters", argv[i])) {
      options.output_clusters = !strcmp("-clusters", argv[i]);
    } else if (Options::MatchesBooleanOption("blocks", argv[i])) {
      options.output_blocks = !strcmp("-blocks", argv[i]);
    } else if (Options::MatchesBooleanOption("codec_info", argv[i])) {
      options.output_codec_info = !strcmp("-codec_info", argv[i]);
    } else if (Options::MatchesBooleanOption("clusters_size", argv[i])) {
      options.output_clusters_size = !strcmp("-clusters_size", argv[i]);
    } else if (Options::MatchesBooleanOption("encrypted_info", argv[i])) {
      options.output_encrypted_info = !strcmp("-encrypted_info", argv[i]);
    } else if (Options::MatchesBooleanOption("cues", argv[i])) {
      options.output_cues = !strcmp("-cues", argv[i]);
    } else if (Options::MatchesBooleanOption("frame_stats", argv[i])) {
      options.output_frame_stats = !strcmp("-frame_stats", argv[i]);
    } else if (Options::MatchesBooleanOption("vp9_level", argv[i])) {
      options.output_vp9_level = !strcmp("-vp9_level", argv[i]);
    }
  }

  if (argc < 3 || input.empty()) {
    Usage();
    return EXIT_FAILURE;
  }

  std::unique_ptr<mkvparser::MkvReader> reader(
      new (std::nothrow) mkvparser::MkvReader());  // NOLINT
  if (!reader || reader->Open(input.c_str())) {
    fprintf(stderr, "Error opening file:%s\n", input.c_str());
    return EXIT_FAILURE;
  }

  long long int pos = 0;
  std::unique_ptr<mkvparser::EBMLHeader> ebml_header(
      new (std::nothrow) mkvparser::EBMLHeader());  // NOLINT
  if (!ebml_header || ebml_header->Parse(reader.get(), pos) < 0) {
    fprintf(stderr, "Error parsing EBML header.\n");
    return EXIT_FAILURE;
  }

  Indent indent(0);
  FILE* out = stdout;

  if (options.output_ebml_header)
    OutputEBMLHeader(*ebml_header.get(), out, &indent);

  mkvparser::Segment* temp_segment;
  if (mkvparser::Segment::CreateInstance(reader.get(), pos, temp_segment)) {
    fprintf(stderr, "Segment::CreateInstance() failed.\n");
    return EXIT_FAILURE;
  }
  std::unique_ptr<mkvparser::Segment> segment(temp_segment);

  if (segment->Load() < 0) {
    fprintf(stderr, "Segment::Load() failed.\n");
    return EXIT_FAILURE;
  }

  if (options.output_segment) {
    OutputSegment(*(segment.get()), options, out);
    indent.Adjust(libwebm::kIncreaseIndent);
  }

  if (options.output_seekhead)
    if (!OutputSeekHead(*(segment.get()), options, out, &indent))
      return EXIT_FAILURE;

  if (options.output_segment_info)
    if (!OutputSegmentInfo(*(segment.get()), options, out, &indent))
      return EXIT_FAILURE;

  if (options.output_tracks)
    if (!OutputTracks(*(segment.get()), options, out, &indent))
      return EXIT_FAILURE;

  const mkvparser::Tracks* const tracks = segment->GetTracks();
  if (!tracks) {
    fprintf(stderr, "Could not get Tracks.\n");
    return EXIT_FAILURE;
  }

  // If Cues are before the clusters output them first.
  if (options.output_cues) {
    const mkvparser::Cluster* cluster = segment->GetFirst();
    const mkvparser::Cues* const cues = segment->GetCues();
    if (cluster != NULL && cues != NULL) {
      if (cues->m_element_start < cluster->m_element_start) {
        if (!OutputCues(*segment, *tracks, options, out, &indent)) {
          return EXIT_FAILURE;
        }
        options.output_cues = false;
      }
    }
  }

  if (options.output_clusters)
    fprintf(out, "%sClusters (count):%ld\n", indent.indent_str().c_str(),
            segment->GetCount());

  int64_t clusters_size = 0;
  FrameStats stats;
  vp9_parser::Vp9HeaderParser parser;
  vp9_parser::Vp9LevelStats level_stats;
  const mkvparser::Cluster* cluster = segment->GetFirst();
  while (cluster != NULL && !cluster->EOS()) {
    if (!OutputCluster(*cluster, *tracks, options, out, reader.get(), &indent,
                       &clusters_size, &stats, &parser, &level_stats))
      return EXIT_FAILURE;
    cluster = segment->GetNext(cluster);
  }

  if (options.output_clusters_size)
    fprintf(out, "%sClusters (size):%" PRId64 "\n", indent.indent_str().c_str(),
            clusters_size);

  if (options.output_cues)
    if (!OutputCues(*segment, *tracks, options, out, &indent))
      return EXIT_FAILURE;

  // TODO(fgalligan): Add support for VP8.
  if (options.output_frame_stats &&
      stats.minimum_altref_distance != std::numeric_limits<int>::max()) {
    const double actual_fps =
        stats.frames /
        (segment->GetInfo()->GetDuration() / kNanosecondsPerSecond);
    const double displayed_fps =
        stats.displayed_frames /
        (segment->GetInfo()->GetDuration() / kNanosecondsPerSecond);
    fprintf(out, "\nActual fps:%g  Displayed fps:%g\n", actual_fps,
            displayed_fps);

    fprintf(out, "Minimum Altref Distance:%d  at:%g seconds\n",
            stats.minimum_altref_distance,
            stats.min_altref_end_ns / kNanosecondsPerSecond);

    // TODO(fgalligan): Add support for window duration other than 1 second.
    const double sec_end = stats.max_window_end_ns / kNanosecondsPerSecond;
    const double sec_start =
        stats.max_window_end_ns > kNanosecondsPerSecondi ? sec_end - 1.0 : 0.0;
    fprintf(out, "Maximum Window:%g-%g seconds  Window fps:%" PRId64 "\n",
            sec_start, sec_end, stats.max_window_size);
  }

  if (options.output_vp9_level) {
    level_stats.set_duration(segment->GetInfo()->GetDuration());
    const vp9_parser::Vp9Level level = level_stats.GetLevel();
    fprintf(out, "VP9 Level:%d\n", level);
    fprintf(
        out,
        "mlsr:%" PRId64 " mlps:%" PRId64 " mlpb:%" PRId64
        " abr:%g mcs:%g cr:%g mct:%d"
        " mad:%d mrf:%d\n",
        level_stats.GetMaxLumaSampleRate(), level_stats.GetMaxLumaPictureSize(),
        level_stats.GetMaxLumaPictureBreadth(), level_stats.GetAverageBitRate(),
        level_stats.GetMaxCpbSize(), level_stats.GetCompressionRatio(),
        level_stats.GetMaxColumnTiles(), level_stats.GetMinimumAltrefDistance(),
        level_stats.GetMaxReferenceFrames());
  }
  return EXIT_SUCCESS;
}
