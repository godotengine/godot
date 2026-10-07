// Copyright (c) 2010 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
//
// This sample application demonstrates how to use the Matroska parser
// library, which allows clients to handle a Matroska format file.
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>

#include "mkvparser/mkvparser.h"
#include "mkvparser/mkvreader.h"

namespace {
const wchar_t* utf8towcs(const char* str) {
  if (str == NULL)
    return NULL;

  // TODO: this probably requires that the locale be
  // configured somehow:

  const size_t size = mbstowcs(NULL, str, 0);

  if (size == 0 || size == static_cast<size_t>(-1))
    return NULL;

  wchar_t* const val = new (std::nothrow) wchar_t[size + 1];
  if (val == NULL)
    return NULL;

  mbstowcs(val, str, size);
  val[size] = L'\0';

  return val;
}

bool InputHasCues(const mkvparser::Segment* const segment) {
  const mkvparser::Cues* const cues = segment->GetCues();
  if (cues == NULL)
    return false;

  while (!cues->DoneParsing())
    cues->LoadCuePoint();

  const mkvparser::CuePoint* const cue_point = cues->GetFirst();
  if (cue_point == NULL)
    return false;

  return true;
}

bool MasteringMetadataValuePresent(double value) {
  return value != mkvparser::MasteringMetadata::kValueNotPresent;
}

bool ColourValuePresent(long long value) {
  return value != mkvparser::Colour::kValueNotPresent;
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 1) {
    printf("Mkv Parser Sample Application\n");
    printf("  Usage: %s <input file> \n", argv[0]);
    return EXIT_FAILURE;
  }

  mkvparser::MkvReader reader;

  if (reader.Open(argv[1])) {
    printf("\n Filename is invalid or error while opening.\n");
    return EXIT_FAILURE;
  }

  int maj, min, build, rev;

  mkvparser::GetVersion(maj, min, build, rev);
  printf("\t\t libwebm version: %d.%d.%d.%d\n", maj, min, build, rev);

  long long pos = 0;

  mkvparser::EBMLHeader ebmlHeader;

  long long ret = ebmlHeader.Parse(&reader, pos);
  if (ret < 0) {
    printf("\n EBMLHeader::Parse() failed.");
    return EXIT_FAILURE;
  }

  printf("\t\t\t    EBML Header\n");
  printf("\t\tEBML Version\t\t: %lld\n", ebmlHeader.m_version);
  printf("\t\tEBML MaxIDLength\t: %lld\n", ebmlHeader.m_maxIdLength);
  printf("\t\tEBML MaxSizeLength\t: %lld\n", ebmlHeader.m_maxSizeLength);
  printf("\t\tDoc Type\t\t: %s\n", ebmlHeader.m_docType);
  printf("\t\tPos\t\t\t: %lld\n", pos);

  typedef mkvparser::Segment seg_t;
  seg_t* pSegment_;

  ret = seg_t::CreateInstance(&reader, pos, pSegment_);
  if (ret) {
    printf("\n Segment::CreateInstance() failed.");
    return EXIT_FAILURE;
  }

  const std::unique_ptr<seg_t> pSegment(pSegment_);

  ret = pSegment->Load();
  if (ret < 0) {
    printf("\n Segment::Load() failed.");
    return EXIT_FAILURE;
  }

  const mkvparser::SegmentInfo* const pSegmentInfo = pSegment->GetInfo();
  if (pSegmentInfo == NULL) {
    printf("\n Segment::GetInfo() failed.");
    return EXIT_FAILURE;
  }

  const long long timeCodeScale = pSegmentInfo->GetTimeCodeScale();
  const long long duration_ns = pSegmentInfo->GetDuration();

  const char* const pTitle_ = pSegmentInfo->GetTitleAsUTF8();
  const wchar_t* const pTitle = utf8towcs(pTitle_);

  const char* const pMuxingApp_ = pSegmentInfo->GetMuxingAppAsUTF8();
  const wchar_t* const pMuxingApp = utf8towcs(pMuxingApp_);

  const char* const pWritingApp_ = pSegmentInfo->GetWritingAppAsUTF8();
  const wchar_t* const pWritingApp = utf8towcs(pWritingApp_);

  printf("\n");
  printf("\t\t\t   Segment Info\n");
  printf("\t\tTimeCodeScale\t\t: %lld \n", timeCodeScale);
  printf("\t\tDuration\t\t: %lld\n", duration_ns);

  const double duration_sec = double(duration_ns) / 1000000000;
  printf("\t\tDuration(secs)\t\t: %7.3lf\n", duration_sec);

  if (pTitle == NULL)
    printf("\t\tTrack Name\t\t: NULL\n");
  else {
    printf("\t\tTrack Name\t\t: %ls\n", pTitle);
    delete[] pTitle;
  }

  if (pMuxingApp == NULL)
    printf("\t\tMuxing App\t\t: NULL\n");
  else {
    printf("\t\tMuxing App\t\t: %ls\n", pMuxingApp);
    delete[] pMuxingApp;
  }

  if (pWritingApp == NULL)
    printf("\t\tWriting App\t\t: NULL\n");
  else {
    printf("\t\tWriting App\t\t: %ls\n", pWritingApp);
    delete[] pWritingApp;
  }

  // pos of segment payload
  printf("\t\tPosition(Segment)\t: %lld\n", pSegment->m_start);

  // size of segment payload
  printf("\t\tSize(Segment)\t\t: %lld\n", pSegment->m_size);

  const mkvparser::Tracks* pTracks = pSegment->GetTracks();

  unsigned long track_num = 0;
  const unsigned long num_tracks = pTracks->GetTracksCount();

  printf("\n\t\t\t   Track Info\n");

  while (track_num != num_tracks) {
    const mkvparser::Track* const pTrack =
        pTracks->GetTrackByIndex(track_num++);

    if (pTrack == NULL)
      continue;

    const long trackType = pTrack->GetType();
    const long trackNumber = pTrack->GetNumber();
    const unsigned long long trackUid = pTrack->GetUid();
    const wchar_t* const pTrackName = utf8towcs(pTrack->GetNameAsUTF8());

    printf("\t\tTrack Type\t\t: %ld\n", trackType);
    printf("\t\tTrack Number\t\t: %ld\n", trackNumber);
    printf("\t\tTrack Uid\t\t: %lld\n", trackUid);

    if (pTrackName == NULL)
      printf("\t\tTrack Name\t\t: NULL\n");
    else {
      printf("\t\tTrack Name\t\t: %ls \n", pTrackName);
      delete[] pTrackName;
    }

    const char* const pCodecId = pTrack->GetCodecId();

    if (pCodecId == NULL)
      printf("\t\tCodec Id\t\t: NULL\n");
    else
      printf("\t\tCodec Id\t\t: %s\n", pCodecId);

    size_t codec_private_size = 0;
    if (pTrack->GetCodecPrivate(codec_private_size)) {
      printf("\t\tCodec private length: %u bytes\n",
             static_cast<unsigned int>(codec_private_size));
    }

    const char* const pCodecName_ = pTrack->GetCodecNameAsUTF8();
    const wchar_t* const pCodecName = utf8towcs(pCodecName_);

    if (pCodecName == NULL)
      printf("\t\tCodec Name\t\t: NULL\n");
    else {
      printf("\t\tCodec Name\t\t: %ls\n", pCodecName);
      delete[] pCodecName;
    }

    if (trackType == mkvparser::Track::kVideo) {
      const mkvparser::VideoTrack* const pVideoTrack =
          static_cast<const mkvparser::VideoTrack*>(pTrack);

      const long long width = pVideoTrack->GetWidth();
      printf("\t\tVideo Width\t\t: %lld\n", width);

      const long long height = pVideoTrack->GetHeight();
      printf("\t\tVideo Height\t\t: %lld\n", height);

      const double rate = pVideoTrack->GetFrameRate();
      printf("\t\tVideo Rate\t\t: %f\n", rate);

      const mkvparser::Colour* const colour = pVideoTrack->GetColour();
      if (colour) {
        printf("\t\tVideo Colour:\n");
        if (ColourValuePresent(colour->matrix_coefficients))
          printf("\t\t\tMatrixCoefficients: %lld\n",
                 colour->matrix_coefficients);
        if (ColourValuePresent(colour->bits_per_channel))
          printf("\t\t\tBitsPerChannel: %lld\n", colour->bits_per_channel);
        if (ColourValuePresent(colour->chroma_subsampling_horz))
          printf("\t\t\tChromaSubsamplingHorz: %lld\n",
                 colour->chroma_subsampling_horz);
        if (ColourValuePresent(colour->chroma_subsampling_vert))
          printf("\t\t\tChromaSubsamplingVert: %lld\n",
                 colour->chroma_subsampling_vert);
        if (ColourValuePresent(colour->cb_subsampling_horz))
          printf("\t\t\tCbSubsamplingHorz: %lld\n",
                 colour->cb_subsampling_horz);
        if (ColourValuePresent(colour->cb_subsampling_vert))
          printf("\t\t\tCbSubsamplingVert: %lld\n",
                 colour->cb_subsampling_vert);
        if (ColourValuePresent(colour->chroma_siting_horz))
          printf("\t\t\tChromaSitingHorz: %lld\n", colour->chroma_siting_horz);
        if (ColourValuePresent(colour->chroma_siting_vert))
          printf("\t\t\tChromaSitingVert: %lld\n", colour->chroma_siting_vert);
        if (ColourValuePresent(colour->range))
          printf("\t\t\tRange: %lld\n", colour->range);
        if (ColourValuePresent(colour->transfer_characteristics))
          printf("\t\t\tTransferCharacteristics: %lld\n",
                 colour->transfer_characteristics);
        if (ColourValuePresent(colour->primaries))
          printf("\t\t\tPrimaries: %lld\n", colour->primaries);
        if (ColourValuePresent(colour->max_cll))
          printf("\t\t\tMaxCLL: %lld\n", colour->max_cll);
        if (ColourValuePresent(colour->max_fall))
          printf("\t\t\tMaxFALL: %lld\n", colour->max_fall);
        if (colour->mastering_metadata) {
          const mkvparser::MasteringMetadata* const mm =
              colour->mastering_metadata;
          printf("\t\t\tMastering Metadata:\n");
          if (MasteringMetadataValuePresent(mm->luminance_max))
            printf("\t\t\t\tLuminanceMax: %f\n", mm->luminance_max);
          if (MasteringMetadataValuePresent(mm->luminance_min))
            printf("\t\t\t\tLuminanceMin: %f\n", mm->luminance_min);
          if (mm->r) {
            printf("\t\t\t\t\tPrimaryRChromaticityX: %f\n", mm->r->x);
            printf("\t\t\t\t\tPrimaryRChromaticityY: %f\n", mm->r->y);
          }
          if (mm->g) {
            printf("\t\t\t\t\tPrimaryGChromaticityX: %f\n", mm->g->x);
            printf("\t\t\t\t\tPrimaryGChromaticityY: %f\n", mm->g->y);
          }
          if (mm->b) {
            printf("\t\t\t\t\tPrimaryBChromaticityX: %f\n", mm->b->x);
            printf("\t\t\t\t\tPrimaryBChromaticityY: %f\n", mm->b->y);
          }
          if (mm->white_point) {
            printf("\t\t\t\t\tWhitePointChromaticityX: %f\n",
                   mm->white_point->x);
            printf("\t\t\t\t\tWhitePointChromaticityY: %f\n",
                   mm->white_point->y);
          }
        }
      }

      const mkvparser::Projection* const projection =
          pVideoTrack->GetProjection();
      if (projection) {
        printf("\t\tVideo Projection:\n");
        if (projection->type != mkvparser::Projection::kTypeNotPresent)
          printf("\t\t\tProjectionType: %d\n",
                 static_cast<int>(projection->type));
        if (projection->private_data) {
          printf("\t\t\tProjectionPrivate: %u bytes\n",
                 static_cast<unsigned int>(projection->private_data_length));
        }
        if (projection->pose_yaw != mkvparser::Projection::kValueNotPresent)
          printf("\t\t\tProjectionPoseYaw: %f\n", projection->pose_yaw);
        if (projection->pose_pitch != mkvparser::Projection::kValueNotPresent)
          printf("\t\t\tProjectionPosePitch: %f\n", projection->pose_pitch);
        if (projection->pose_roll != mkvparser::Projection::kValueNotPresent)
          printf("\t\t\tProjectionPosePitch: %f\n", projection->pose_roll);
      }
    }

    if (trackType == mkvparser::Track::kAudio) {
      const mkvparser::AudioTrack* const pAudioTrack =
          static_cast<const mkvparser::AudioTrack*>(pTrack);

      const long long channels = pAudioTrack->GetChannels();
      printf("\t\tAudio Channels\t\t: %lld\n", channels);

      const long long bitDepth = pAudioTrack->GetBitDepth();
      printf("\t\tAudio BitDepth\t\t: %lld\n", bitDepth);

      const double sampleRate = pAudioTrack->GetSamplingRate();
      printf("\t\tAddio Sample Rate\t: %.3f\n", sampleRate);

      const long long codecDelay = pAudioTrack->GetCodecDelay();
      printf("\t\tAudio Codec Delay\t\t: %lld\n", codecDelay);

      const long long seekPreRoll = pAudioTrack->GetSeekPreRoll();
      printf("\t\tAudio Seek Pre Roll\t\t: %lld\n", seekPreRoll);
    }
  }

  printf("\n\n\t\t\t   Cluster Info\n");
  const unsigned long clusterCount = pSegment->GetCount();

  printf("\t\tCluster Count\t: %ld\n\n", clusterCount);

  if (clusterCount == 0) {
    printf("\t\tSegment has no clusters.\n");
    return EXIT_FAILURE;
  }

  const mkvparser::Cluster* pCluster = pSegment->GetFirst();

  while (pCluster != NULL && !pCluster->EOS()) {
    const long long timeCode = pCluster->GetTimeCode();
    printf("\t\tCluster Time Code\t: %lld\n", timeCode);

    const long long time_ns = pCluster->GetTime();
    printf("\t\tCluster Time (ns)\t: %lld\n", time_ns);

    const mkvparser::BlockEntry* pBlockEntry;

    long status = pCluster->GetFirst(pBlockEntry);

    if (status < 0)  // error
    {
      printf("\t\tError parsing first block of cluster\n");
      fflush(stdout);
      return EXIT_FAILURE;
    }

    while (pBlockEntry != NULL && !pBlockEntry->EOS()) {
      const mkvparser::Block* const pBlock = pBlockEntry->GetBlock();
      const long long trackNum = pBlock->GetTrackNumber();
      const unsigned long tn = static_cast<unsigned long>(trackNum);
      const mkvparser::Track* const pTrack = pTracks->GetTrackByNumber(tn);

      if (pTrack == NULL)
        printf("\t\t\tBlock\t\t:UNKNOWN TRACK TYPE\n");
      else {
        const long long trackType = pTrack->GetType();
        const int frameCount = pBlock->GetFrameCount();
        const long long time_ns = pBlock->GetTime(pCluster);
        const long long discard_padding = pBlock->GetDiscardPadding();

        printf("\t\t\tBlock\t\t:%s,%s,%15lld,%lld\n",
               (trackType == mkvparser::Track::kVideo) ? "V" : "A",
               pBlock->IsKey() ? "I" : "P", time_ns, discard_padding);

        for (int i = 0; i < frameCount; ++i) {
          const mkvparser::Block::Frame& theFrame = pBlock->GetFrame(i);
          const long size = theFrame.len;
          const long long offset = theFrame.pos;
          printf("\t\t\t %15ld,%15llx\n", size, offset);
        }
      }

      status = pCluster->GetNext(pBlockEntry, pBlockEntry);

      if (status < 0) {
        printf("\t\t\tError parsing next block of cluster\n");
        fflush(stdout);
        return EXIT_FAILURE;
      }
    }

    pCluster = pSegment->GetNext(pCluster);
  }

  if (InputHasCues(pSegment.get())) {
    // Walk them.
    const mkvparser::Cues* const cues = pSegment->GetCues();
    const mkvparser::CuePoint* cue = cues->GetFirst();
    int cue_point_num = 1;

    printf("\t\tCues\n");
    do {
      for (track_num = 0; track_num < num_tracks; ++track_num) {
        const mkvparser::Track* const track =
            pTracks->GetTrackByIndex(track_num);
        const mkvparser::CuePoint::TrackPosition* const track_pos =
            cue->Find(track);

        if (track_pos != NULL) {
          const char track_type =
              (track->GetType() == mkvparser::Track::kVideo) ? 'V' : 'A';
          printf(
              "\t\t\tCue Point %4d Track %3lu(%c) Time %14lld "
              "Block %4lld Pos %8llx\n",
              cue_point_num, track->GetNumber(), track_type,
              cue->GetTime(pSegment.get()), track_pos->m_block,
              track_pos->m_pos);
        }
      }

      cue = cues->GetNext(cue);
      ++cue_point_num;
    } while (cue != NULL);
  }

  const mkvparser::Tags* const tags = pSegment->GetTags();
  if (tags && tags->GetTagCount() > 0) {
    printf("\t\tTags\n");
    for (int i = 0; i < tags->GetTagCount(); ++i) {
      const mkvparser::Tags::Tag* const tag = tags->GetTag(i);
      printf("\t\t\tTag\n");
      for (int j = 0; j < tag->GetSimpleTagCount(); j++) {
        const mkvparser::Tags::SimpleTag* const simple_tag =
            tag->GetSimpleTag(j);
        printf("\t\t\t\tSimple Tag \"%s\" Value \"%s\"\n",
               simple_tag->GetTagName(), simple_tag->GetTagString());
      }
    }
  }

  fflush(stdout);
  return EXIT_SUCCESS;
}
