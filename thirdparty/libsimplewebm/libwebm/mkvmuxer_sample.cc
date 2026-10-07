// Copyright (c) 2011 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <new>
#include <string>

// libwebm common includes.
#include "common/file_util.h"
#include "common/hdr_util.h"

// libwebm mkvparser includes
#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"

// libwebm mkvmuxer includes
#include "mkvmuxer/mkvmuxer.h"
#include "mkvmuxer/mkvmuxertypes.h"
#include "mkvmuxer/mkvwriter.h"
#include "sample_muxer_metadata.h"

namespace {

void Usage() {
  printf("Usage: mkvmuxer_sample -i input -o output [options]\n");
  printf("\n");
  printf("Main options:\n");
  printf("  -h | -?                     show help\n");
  printf("  -video <int>                >0 outputs video\n");
  printf("  -audio <int>                >0 outputs audio\n");
  printf("  -live <int>                 >0 puts the muxer into live mode\n");
  printf("                              0 puts the muxer into file mode\n");
  printf("  -output_cues <int>          >0 outputs cues element\n");
  printf("  -cues_on_video_track <int>  >0 outputs cues on video track\n");
  printf("  -cues_on_audio_track <int>  >0 outputs cues on audio track\n");
  printf("  -max_cluster_duration <double> in seconds\n");
  printf("  -max_cluster_size <int>     in bytes\n");
  printf("  -switch_tracks <int>        >0 switches tracks in output\n");
  printf("  -audio_track_number <int>   >0 Changes the audio track number\n");
  printf("  -video_track_number <int>   >0 Changes the video track number\n");
  printf("  -chunking <string>          Chunk output\n");
  printf("  -copy_tags <int>            >0 Copies the tags\n");
  printf("  -accurate_cluster_duration <int> ");
  printf(">0 Writes the last frame in each cluster with Duration\n");
  printf("  -fixed_size_cluster_timecode <int> ");
  printf(">0 Writes the cluster timecode using exactly 8 bytes\n");
  printf("  -copy_input_duration        >0 Copies the input duration\n");
  printf("\n");
  printf("Video options:\n");
  printf("  -display_width <int>           Display width in pixels\n");
  printf("  -display_height <int>          Display height in pixels\n");
  printf("  -pixel_width <int>             Override pixel width\n");
  printf("  -pixel_height <int>            Override pixel height\n");
  printf("  -projection_type <int>         Set/override projection type:\n");
  printf("                                   0: Rectangular\n");
  printf("                                   1: Equirectangular\n");
  printf("                                   2: Cube map\n");
  printf("                                   3: Mesh\n");
  printf("  -projection_file <string>      Override projection private data\n");
  printf("                                 with contents of this file\n");
  printf("  -projection_pose_yaw <float>   Projection pose yaw\n");
  printf("  -projection_pose_pitch <float> Projection pose pitch\n");
  printf("  -projection_pose_roll <float>  Projection pose roll\n");
  printf("  -stereo_mode <int>             3D video mode\n");
  printf("\n");
  printf("VP9 options:\n");
  printf("  -profile <int>              VP9 profile\n");
  printf("  -level <int>                VP9 level\n");
  printf("\n");
  printf("Cues options:\n");
  printf("  -output_cues_block_number <int> >0 outputs cue block number\n");
  printf("  -cues_before_clusters <int> >0 puts Cues before Clusters\n");
  printf("\n");
  printf("Metadata options:\n");
  printf("  -webvtt-subtitles <vttfile>    ");
  printf("add WebVTT subtitles as metadata track\n");
  printf("  -webvtt-captions <vttfile>     ");
  printf("add WebVTT captions as metadata track\n");
  printf("  -webvtt-descriptions <vttfile> ");
  printf("add WebVTT descriptions as metadata track\n");
  printf("  -webvtt-metadata <vttfile>     ");
  printf("add WebVTT subtitles as metadata track\n");
  printf("  -webvtt-chapters <vttfile>     ");
  printf("add WebVTT chapters as MKV chapters element\n");
}

struct MetadataFile {
  const char* name;
  SampleMuxerMetadata::Kind kind;
};

typedef std::list<MetadataFile> metadata_files_t;

// Cache the WebVTT filenames specified as command-line args.
bool LoadMetadataFiles(const metadata_files_t& files,
                       SampleMuxerMetadata* metadata) {
  typedef metadata_files_t::const_iterator iter_t;

  iter_t i = files.begin();
  const iter_t j = files.end();

  while (i != j) {
    const metadata_files_t::value_type& v = *i++;

    if (!metadata->Load(v.name, v.kind))
      return false;
  }

  return true;
}

int ParseArgWebVTT(char* argv[], int* argv_index, int argc_check,
                   metadata_files_t* metadata_files) {
  int& i = *argv_index;

  enum { kCount = 5 };
  struct Arg {
    const char* name;
    SampleMuxerMetadata::Kind kind;
  };
  const Arg args[kCount] = {
      {"-webvtt-subtitles", SampleMuxerMetadata::kSubtitles},
      {"-webvtt-captions", SampleMuxerMetadata::kCaptions},
      {"-webvtt-descriptions", SampleMuxerMetadata::kDescriptions},
      {"-webvtt-metadata", SampleMuxerMetadata::kMetadata},
      {"-webvtt-chapters", SampleMuxerMetadata::kChapters}};

  for (int idx = 0; idx < kCount; ++idx) {
    const Arg& arg = args[idx];

    if (strcmp(arg.name, argv[i]) != 0)  // no match
      continue;

    ++i;  // consume arg name here

    if (i > argc_check) {
      printf("missing value for %s\n", arg.name);
      return -1;  // error
    }

    MetadataFile f;
    f.name = argv[i];  // arg value is consumed via caller's loop idx
    f.kind = arg.kind;

    metadata_files->push_back(f);
    return 1;  // successfully parsed WebVTT arg
  }

  return 0;  // not a WebVTT arg
}

bool CopyVideoProjection(const mkvparser::Projection& parser_projection,
                         mkvmuxer::Projection* muxer_projection) {
  typedef mkvmuxer::Projection::ProjectionType MuxerProjType;
  const int kTypeNotPresent = mkvparser::Projection::kTypeNotPresent;
  if (parser_projection.type != kTypeNotPresent) {
    muxer_projection->set_type(
        static_cast<MuxerProjType>(parser_projection.type));
  }
  if (parser_projection.private_data &&
      parser_projection.private_data_length > 0) {
    if (!muxer_projection->SetProjectionPrivate(
            parser_projection.private_data,
            parser_projection.private_data_length)) {
      return false;
    }
  }

  const float kValueNotPresent = mkvparser::Projection::kValueNotPresent;
  if (parser_projection.pose_yaw != kValueNotPresent)
    muxer_projection->set_pose_yaw(parser_projection.pose_yaw);
  if (parser_projection.pose_pitch != kValueNotPresent)
    muxer_projection->set_pose_pitch(parser_projection.pose_pitch);
  if (parser_projection.pose_roll != kValueNotPresent)
    muxer_projection->set_pose_roll(parser_projection.pose_roll);
  return true;
}
}  // end namespace

int main(int argc, char* argv[]) {
  char* input = NULL;
  char* output = NULL;

  // Segment variables
  bool output_video = true;
  bool output_audio = true;
  bool live_mode = false;
  bool output_cues = true;
  bool cues_before_clusters = false;
  bool cues_on_video_track = true;
  bool cues_on_audio_track = false;
  uint64_t max_cluster_duration = 0;
  uint64_t max_cluster_size = 0;
  bool switch_tracks = false;
  int audio_track_number = 0;  // 0 tells muxer to decide.
  int video_track_number = 0;  // 0 tells muxer to decide.
  bool chunking = false;
  bool copy_tags = false;
  const char* chunk_name = NULL;
  bool accurate_cluster_duration = false;
  bool fixed_size_cluster_timecode = false;
  bool copy_input_duration = false;

  bool output_cues_block_number = true;

  uint64_t display_width = 0;
  uint64_t display_height = 0;
  uint64_t pixel_width = 0;
  uint64_t pixel_height = 0;
  uint64_t stereo_mode = 0;
  const char* projection_file = 0;
  int64_t projection_type = mkvparser::Projection::kTypeNotPresent;
  float projection_pose_roll = mkvparser::Projection::kValueNotPresent;
  float projection_pose_pitch = mkvparser::Projection::kValueNotPresent;
  float projection_pose_yaw = mkvparser::Projection::kValueNotPresent;
  int vp9_profile = -1;  // No profile set.
  int vp9_level = -1;  // No level set.

  metadata_files_t metadata_files;

  const int argc_check = argc - 1;
  for (int i = 1; i < argc; ++i) {
    char* end;

    if (!strcmp("-h", argv[i]) || !strcmp("-?", argv[i])) {
      Usage();
      return EXIT_SUCCESS;
    } else if (!strcmp("-i", argv[i]) && i < argc_check) {
      input = argv[++i];
    } else if (!strcmp("-o", argv[i]) && i < argc_check) {
      output = argv[++i];
    } else if (!strcmp("-video", argv[i]) && i < argc_check) {
      output_video = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-audio", argv[i]) && i < argc_check) {
      output_audio = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-live", argv[i]) && i < argc_check) {
      live_mode = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-output_cues", argv[i]) && i < argc_check) {
      output_cues = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-cues_before_clusters", argv[i]) && i < argc_check) {
      cues_before_clusters = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-cues_on_video_track", argv[i]) && i < argc_check) {
      cues_on_video_track = strtol(argv[++i], &end, 10) == 0 ? false : true;
      if (cues_on_video_track)
        cues_on_audio_track = false;
    } else if (!strcmp("-cues_on_audio_track", argv[i]) && i < argc_check) {
      cues_on_audio_track = strtol(argv[++i], &end, 10) == 0 ? false : true;
      if (cues_on_audio_track)
        cues_on_video_track = false;
    } else if (!strcmp("-max_cluster_duration", argv[i]) && i < argc_check) {
      const double seconds = strtod(argv[++i], &end);
      max_cluster_duration = static_cast<uint64_t>(seconds * 1000000000.0);
    } else if (!strcmp("-max_cluster_size", argv[i]) && i < argc_check) {
      max_cluster_size = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-switch_tracks", argv[i]) && i < argc_check) {
      switch_tracks = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-audio_track_number", argv[i]) && i < argc_check) {
      audio_track_number = static_cast<int>(strtol(argv[++i], &end, 10));
    } else if (!strcmp("-video_track_number", argv[i]) && i < argc_check) {
      video_track_number = static_cast<int>(strtol(argv[++i], &end, 10));
    } else if (!strcmp("-chunking", argv[i]) && i < argc_check) {
      chunking = true;
      chunk_name = argv[++i];
    } else if (!strcmp("-copy_tags", argv[i]) && i < argc_check) {
      copy_tags = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-accurate_cluster_duration", argv[i]) &&
               i < argc_check) {
      accurate_cluster_duration =
          strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-fixed_size_cluster_timecode", argv[i]) &&
               i < argc_check) {
      fixed_size_cluster_timecode =
          strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-copy_input_duration", argv[i]) && i < argc_check) {
      copy_input_duration = strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (!strcmp("-display_width", argv[i]) && i < argc_check) {
      display_width = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-display_height", argv[i]) && i < argc_check) {
      display_height = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-pixel_width", argv[i]) && i < argc_check) {
      pixel_width = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-pixel_height", argv[i]) && i < argc_check) {
      pixel_height = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-stereo_mode", argv[i]) && i < argc_check) {
      stereo_mode = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-projection_type", argv[i]) && i < argc_check) {
      projection_type = strtol(argv[++i], &end, 10);
    } else if (!strcmp("-projection_file", argv[i]) && i < argc_check) {
      projection_file = argv[++i];
    } else if (!strcmp("-projection_pose_roll", argv[i]) && i < argc_check) {
      projection_pose_roll = strtof(argv[++i], &end);
    } else if (!strcmp("-projection_pose_pitch", argv[i]) && i < argc_check) {
      projection_pose_pitch = strtof(argv[++i], &end);
    } else if (!strcmp("-projection_pose_yaw", argv[i]) && i < argc_check) {
      projection_pose_yaw = strtof(argv[++i], &end);
    } else if (!strcmp("-profile", argv[i]) && i < argc_check) {
      vp9_profile = static_cast<int>(strtol(argv[++i], &end, 10));
    } else if (!strcmp("-level", argv[i]) && i < argc_check) {
      vp9_level = static_cast<int>(strtol(argv[++i], &end, 10));
    } else if (!strcmp("-output_cues_block_number", argv[i]) &&
               i < argc_check) {
      output_cues_block_number =
          strtol(argv[++i], &end, 10) == 0 ? false : true;
    } else if (int e = ParseArgWebVTT(argv, &i, argc_check, &metadata_files)) {
      if (e < 0)
        return EXIT_FAILURE;
    }
  }

  if (input == NULL || output == NULL) {
    Usage();
    return EXIT_FAILURE;
  }

  // Get parser header info
  mkvparser::MkvReader reader;

  if (reader.Open(input)) {
    printf("\n Filename is invalid or error while opening.\n");
    return EXIT_FAILURE;
  }

  long long pos = 0;
  mkvparser::EBMLHeader ebml_header;
  long long ret = ebml_header.Parse(&reader, pos);
  if (ret) {
    printf("\n EBMLHeader::Parse() failed.");
    return EXIT_FAILURE;
  }

  mkvparser::Segment* parser_segment_;
  ret = mkvparser::Segment::CreateInstance(&reader, pos, parser_segment_);
  if (ret) {
    printf("\n Segment::CreateInstance() failed.");
    return EXIT_FAILURE;
  }

  const std::unique_ptr<mkvparser::Segment> parser_segment(parser_segment_);
  ret = parser_segment->Load();
  if (ret < 0) {
    printf("\n Segment::Load() failed.");
    return EXIT_FAILURE;
  }

  const mkvparser::SegmentInfo* const segment_info = parser_segment->GetInfo();
  if (segment_info == NULL) {
    printf("\n Segment::GetInfo() failed.");
    return EXIT_FAILURE;
  }
  const long long timeCodeScale = segment_info->GetTimeCodeScale();

  // Set muxer header info
  mkvmuxer::MkvWriter writer;

  const std::string temp_file =
      cues_before_clusters ? libwebm::GetTempFileName() : output;
  if (!writer.Open(temp_file.c_str())) {
    printf("\n Filename is invalid or error while opening.\n");
    return EXIT_FAILURE;
  }

  // Set Segment element attributes
  mkvmuxer::Segment muxer_segment;

  if (!muxer_segment.Init(&writer)) {
    printf("\n Could not initialize muxer segment!\n");
    return EXIT_FAILURE;
  }

  muxer_segment.AccurateClusterDuration(accurate_cluster_duration);
  muxer_segment.UseFixedSizeClusterTimecode(fixed_size_cluster_timecode);

  if (live_mode)
    muxer_segment.set_mode(mkvmuxer::Segment::kLive);
  else
    muxer_segment.set_mode(mkvmuxer::Segment::kFile);

  if (chunking)
    muxer_segment.SetChunking(true, chunk_name);

  if (max_cluster_duration > 0)
    muxer_segment.set_max_cluster_duration(max_cluster_duration);
  if (max_cluster_size > 0)
    muxer_segment.set_max_cluster_size(max_cluster_size);
  muxer_segment.OutputCues(output_cues);

  // Set SegmentInfo element attributes
  mkvmuxer::SegmentInfo* const info = muxer_segment.GetSegmentInfo();
  info->set_timecode_scale(timeCodeScale);
  info->set_writing_app("mkvmuxer_sample");

  const mkvparser::Tags* const tags = parser_segment->GetTags();
  if (copy_tags && tags) {
    for (int i = 0; i < tags->GetTagCount(); i++) {
      const mkvparser::Tags::Tag* const tag = tags->GetTag(i);
      mkvmuxer::Tag* muxer_tag = muxer_segment.AddTag();

      for (int j = 0; j < tag->GetSimpleTagCount(); j++) {
        const mkvparser::Tags::SimpleTag* const simple_tag =
            tag->GetSimpleTag(j);
        muxer_tag->add_simple_tag(simple_tag->GetTagName(),
                                  simple_tag->GetTagString());
      }
    }
  }

  // Set Tracks element attributes
  const mkvparser::Tracks* const parser_tracks = parser_segment->GetTracks();
  unsigned long i = 0;
  uint64_t vid_track = 0;  // no track added
  uint64_t aud_track = 0;  // no track added

  using mkvparser::Track;

  while (i != parser_tracks->GetTracksCount()) {
    unsigned long track_num = i++;
    if (switch_tracks)
      track_num = i % parser_tracks->GetTracksCount();

    const Track* const parser_track = parser_tracks->GetTrackByIndex(track_num);

    if (parser_track == NULL)
      continue;

    // TODO(fgalligan): Add support for language to parser.
    const char* const track_name = parser_track->GetNameAsUTF8();

    const long long track_type = parser_track->GetType();

    if (track_type == Track::kVideo && output_video) {
      // Get the video track from the parser
      const mkvparser::VideoTrack* const pVideoTrack =
          static_cast<const mkvparser::VideoTrack*>(parser_track);
      const long long width = pVideoTrack->GetWidth();
      const long long height = pVideoTrack->GetHeight();

      // Add the video track to the muxer
      vid_track = muxer_segment.AddVideoTrack(static_cast<int>(width),
                                              static_cast<int>(height),
                                              video_track_number);
      if (!vid_track) {
        printf("\n Could not add video track.\n");
        return EXIT_FAILURE;
      }

      mkvmuxer::VideoTrack* const video = static_cast<mkvmuxer::VideoTrack*>(
          muxer_segment.GetTrackByNumber(vid_track));
      if (!video) {
        printf("\n Could not get video track.\n");
        return EXIT_FAILURE;
      }

      if (pVideoTrack->GetColour()) {
        mkvmuxer::Colour muxer_colour;
        if (!libwebm::CopyColour(*pVideoTrack->GetColour(), &muxer_colour))
          return EXIT_FAILURE;
        if (!video->SetColour(muxer_colour))
          return EXIT_FAILURE;
      }

      if (pVideoTrack->GetProjection() ||
          projection_type != mkvparser::Projection::kTypeNotPresent) {
        mkvmuxer::Projection muxer_projection;
        const mkvparser::Projection* const parser_projection =
            pVideoTrack->GetProjection();
        typedef mkvmuxer::Projection::ProjectionType MuxerProjType;
        if (parser_projection &&
            !CopyVideoProjection(*parser_projection, &muxer_projection)) {
          printf("\n Unable to copy video projection.\n");
          return EXIT_FAILURE;
        }
        // Override the values that came from parser if set on command line.
        if (projection_type != mkvparser::Projection::kTypeNotPresent) {
          muxer_projection.set_type(
              static_cast<MuxerProjType>(projection_type));
          if (projection_type == mkvparser::Projection::kRectangular &&
              projection_file != NULL) {
            printf("\n Rectangular projection must not have private data.\n");
            return EXIT_FAILURE;
          } else if ((projection_type == mkvparser::Projection::kCubeMap ||
                      projection_type == mkvparser::Projection::kMesh) &&
                     projection_file == NULL) {
            printf("\n Mesh or CubeMap projection must have private data.\n");
            return EXIT_FAILURE;
          }
          if (projection_file != NULL) {
            std::string contents;
            if (!libwebm::GetFileContents(projection_file, &contents) ||
                contents.size() == 0) {
              printf("\n Failed to read file \"%s\" or file is empty\n",
                     projection_file);
              return EXIT_FAILURE;
            }
            if (!muxer_projection.SetProjectionPrivate(
                    reinterpret_cast<uint8_t*>(&contents[0]),
                    contents.size())) {
              printf("\n Failed to SetProjectionPrivate of length %zu.\n",
                     contents.size());
              return EXIT_FAILURE;
            }
          }
        }
        const float kValueNotPresent = mkvparser::Projection::kValueNotPresent;
        if (projection_pose_yaw != kValueNotPresent)
          muxer_projection.set_pose_yaw(projection_pose_yaw);
        if (projection_pose_pitch != kValueNotPresent)
          muxer_projection.set_pose_pitch(projection_pose_pitch);
        if (projection_pose_roll != kValueNotPresent)
          muxer_projection.set_pose_roll(projection_pose_roll);

        if (!video->SetProjection(muxer_projection))
          return EXIT_FAILURE;
      }

      if (track_name)
        video->set_name(track_name);

      video->set_codec_id(pVideoTrack->GetCodecId());

      if (display_width > 0)
        video->set_display_width(display_width);
      if (display_height > 0)
        video->set_display_height(display_height);
      if (pixel_width > 0)
        video->set_pixel_width(pixel_width);
      if (pixel_height > 0)
        video->set_pixel_height(pixel_height);
      if (stereo_mode > 0)
        video->SetStereoMode(stereo_mode);

      const double rate = pVideoTrack->GetFrameRate();
      if (rate > 0.0) {
        video->set_frame_rate(rate);
      }

      size_t parser_private_size;
      const unsigned char* const parser_private_data =
          pVideoTrack->GetCodecPrivate(parser_private_size);

      if (!strcmp(video->codec_id(), mkvmuxer::Tracks::kAv1CodecId)) {
        if (parser_private_data == NULL || parser_private_size == 0) {
          printf("AV1 input track has no CodecPrivate. %s is invalid.", input);
          return EXIT_FAILURE;
        }
      }

      if (!strcmp(video->codec_id(), mkvmuxer::Tracks::kVp9CodecId) &&
          (vp9_profile >= 0 || vp9_level >= 0)) {
        const int kMaxVp9PrivateSize = 6;
        unsigned char vp9_private_data[kMaxVp9PrivateSize];
        int vp9_private_size = 0;
        if (vp9_profile >= 0) {
          if (vp9_profile < 0 || vp9_profile > 3) {
            printf("\n VP9 profile(%d) is not valid.\n", vp9_profile);
            return EXIT_FAILURE;
          }
          const uint8_t kVp9ProfileId = 1;
          const uint8_t kVp9ProfileIdLength = 1;
          vp9_private_data[vp9_private_size++] = kVp9ProfileId;
          vp9_private_data[vp9_private_size++] = kVp9ProfileIdLength;
          vp9_private_data[vp9_private_size++] = vp9_profile;
        }

        if (vp9_level >= 0) {
          const int kNumLevels = 14;
          const int levels[kNumLevels] = {10, 11, 20, 21, 30, 31, 40,
                                          41, 50, 51, 52, 60, 61, 62};
          bool level_is_valid = false;
          for (int i = 0; i < kNumLevels; ++i) {
            if (vp9_level == levels[i]) {
              level_is_valid = true;
              break;
            }
          }
          if (!level_is_valid) {
            printf("\n VP9 level(%d) is not valid.\n", vp9_level);
            return EXIT_FAILURE;
          }
          const uint8_t kVp9LevelId = 2;
          const uint8_t kVp9LevelIdLength = 1;
          vp9_private_data[vp9_private_size++] = kVp9LevelId;
          vp9_private_data[vp9_private_size++] = kVp9LevelIdLength;
          vp9_private_data[vp9_private_size++] = vp9_level;
        }
        if (!video->SetCodecPrivate(vp9_private_data, vp9_private_size)) {
          printf("\n Could not add video private data.\n");
          return EXIT_FAILURE;
        }
      } else if (parser_private_data && parser_private_size > 0) {
        if (!video->SetCodecPrivate(parser_private_data, parser_private_size)) {
          printf("\n Could not add video private data.\n");
          return EXIT_FAILURE;
        }
      }
    } else if (track_type == Track::kAudio && output_audio) {
      // Get the audio track from the parser
      const mkvparser::AudioTrack* const pAudioTrack =
          static_cast<const mkvparser::AudioTrack*>(parser_track);
      const long long channels = pAudioTrack->GetChannels();
      const double sample_rate = pAudioTrack->GetSamplingRate();

      // Add the audio track to the muxer
      aud_track = muxer_segment.AddAudioTrack(static_cast<int>(sample_rate),
                                              static_cast<int>(channels),
                                              audio_track_number);
      if (!aud_track) {
        printf("\n Could not add audio track.\n");
        return EXIT_FAILURE;
      }

      mkvmuxer::AudioTrack* const audio = static_cast<mkvmuxer::AudioTrack*>(
          muxer_segment.GetTrackByNumber(aud_track));
      if (!audio) {
        printf("\n Could not get audio track.\n");
        return EXIT_FAILURE;
      }

      if (track_name)
        audio->set_name(track_name);

      audio->set_codec_id(pAudioTrack->GetCodecId());

      size_t private_size;
      const unsigned char* const private_data =
          pAudioTrack->GetCodecPrivate(private_size);
      if (private_size > 0) {
        if (!audio->SetCodecPrivate(private_data, private_size)) {
          printf("\n Could not add audio private data.\n");
          return EXIT_FAILURE;
        }
      }

      const long long bit_depth = pAudioTrack->GetBitDepth();
      if (bit_depth > 0)
        audio->set_bit_depth(bit_depth);

      if (pAudioTrack->GetCodecDelay())
        audio->set_codec_delay(pAudioTrack->GetCodecDelay());
      if (pAudioTrack->GetSeekPreRoll())
        audio->set_seek_pre_roll(pAudioTrack->GetSeekPreRoll());
    }
  }

  // We have created all the video and audio tracks.  If any WebVTT
  // files were specified as command-line args, then parse them and
  // add a track to the output file corresponding to each metadata
  // input file.

  SampleMuxerMetadata metadata;

  if (!metadata.Init(&muxer_segment)) {
    printf("\n Could not initialize metadata cache.\n");
    return EXIT_FAILURE;
  }

  if (!LoadMetadataFiles(metadata_files, &metadata))
    return EXIT_FAILURE;

  if (!metadata.AddChapters())
    return EXIT_FAILURE;

  // Set Cues element attributes
  mkvmuxer::Cues* const cues = muxer_segment.GetCues();
  cues->set_output_block_number(output_cues_block_number);
  if (cues_on_video_track && vid_track)
    muxer_segment.CuesTrack(vid_track);
  if (cues_on_audio_track && aud_track)
    muxer_segment.CuesTrack(aud_track);

  // Write clusters
  unsigned char* data = NULL;
  long data_len = 0;

  const mkvparser::Cluster* cluster = parser_segment->GetFirst();

  while (cluster != NULL && !cluster->EOS()) {
    const mkvparser::BlockEntry* block_entry;

    long status = cluster->GetFirst(block_entry);

    if (status) {
      printf("\n Could not get first block of cluster.\n");
      return EXIT_FAILURE;
    }

    while (block_entry != NULL && !block_entry->EOS()) {
      const mkvparser::Block* const block = block_entry->GetBlock();
      const long long trackNum = block->GetTrackNumber();
      const mkvparser::Track* const parser_track =
          parser_tracks->GetTrackByNumber(static_cast<unsigned long>(trackNum));

      // When |parser_track| is NULL, it means that the track number in the
      // Block is invalid (i.e.) the was no TrackEntry corresponding to the
      // track number. So we reject the file.
      if (!parser_track) {
        return EXIT_FAILURE;
      }

      const long long track_type = parser_track->GetType();
      const long long time_ns = block->GetTime(cluster);

      // Flush any metadata frames to the output file, before we write
      // the current block.
      if (!metadata.Write(time_ns))
        return EXIT_FAILURE;

      if ((track_type == Track::kAudio && output_audio) ||
          (track_type == Track::kVideo && output_video)) {
        const int frame_count = block->GetFrameCount();

        for (int i = 0; i < frame_count; ++i) {
          const mkvparser::Block::Frame& frame = block->GetFrame(i);

          if (frame.len > data_len) {
            delete[] data;
            data = new (std::nothrow) unsigned char[frame.len];
            if (!data)
              return EXIT_FAILURE;
            data_len = frame.len;
          }

          if (frame.Read(&reader, data))
            return EXIT_FAILURE;

          mkvmuxer::Frame muxer_frame;
          if (!muxer_frame.Init(data, frame.len))
            return EXIT_FAILURE;
          muxer_frame.set_track_number(track_type == Track::kAudio ? aud_track
                                                                   : vid_track);
          if (block->GetDiscardPadding())
            muxer_frame.set_discard_padding(block->GetDiscardPadding());
          muxer_frame.set_timestamp(time_ns);
          muxer_frame.set_is_key(block->IsKey());
          if (!muxer_segment.AddGenericFrame(&muxer_frame)) {
            printf("\n Could not add frame.\n");
            return EXIT_FAILURE;
          }
        }
      }

      status = cluster->GetNext(block_entry, block_entry);

      if (status) {
        printf("\n Could not get next block of cluster.\n");
        return EXIT_FAILURE;
      }
    }

    cluster = parser_segment->GetNext(cluster);
  }

  // We have exhausted all video and audio frames in the input file.
  // Flush any remaining metadata frames to the output file.
  if (!metadata.Write(-1))
    return EXIT_FAILURE;

  if (copy_input_duration) {
    const double input_duration =
        static_cast<double>(segment_info->GetDuration()) / timeCodeScale;
    muxer_segment.set_duration(input_duration);
  }

  if (!muxer_segment.Finalize()) {
    printf("Finalization of segment failed.\n");
    return EXIT_FAILURE;
  }

  reader.Close();
  writer.Close();

  if (cues_before_clusters) {
    if (reader.Open(temp_file.c_str())) {
      printf("\n Filename is invalid or error while opening.\n");
      return EXIT_FAILURE;
    }
    if (!writer.Open(output)) {
      printf("\n Filename is invalid or error while opening.\n");
      return EXIT_FAILURE;
    }
    if (!muxer_segment.CopyAndMoveCuesBeforeClusters(&reader, &writer)) {
      printf("\n Unable to copy and move cues before clusters.\n");
      return EXIT_FAILURE;
    }
    reader.Close();
    writer.Close();
    remove(temp_file.c_str());
  }

  delete[] data;

  return EXIT_SUCCESS;
}
