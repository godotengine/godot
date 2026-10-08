// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#include "src/ancestory.h"

#include "webm/id.h"

namespace webm {

bool Ancestory::ById(Id id, Ancestory* ancestory) {
  // These lists of IDs were generated and must match the switch statement and
  // have static storage duration. They were generated as follows:
  //
  // 1. List all the master elements:
  //      kEbml
  //      kSegment
  //      kSeekHead
  //      kSeek
  //      kInfo
  //      kCluster
  //      kBlockGroup
  //      kBlockAdditions
  //      kBlockMore
  //      kSlices
  //      kTimeSlice
  //      etc.
  //
  // 2. Now prefix each entry with its full ancestory:
  //      kEbml
  //      kSegment
  //      kSegment, kSeekHead
  //      kSegment, kSeekHead, kSeek
  //      kSegment, kInfo
  //      kSegment, kCluster
  //      kSegment, kCluster, kBlockGroup
  //      kSegment, kCluster, kBlockGroup, kBlockAdditions
  //      kSegment, kCluster, kBlockGroup, kBlockAdditions, kBlockMore
  //      kSegment, kCluster, kBlockGroup, kSlices
  //      kSegment, kCluster, kBlockGroup, kSlices, kTimeSlice
  //      etc.
  //
  // 3. Now remove entries that are just subsets of others:
  //      kEbml
  //      kSegment, kSeekHead, kSeek
  //      kSegment, kInfo
  //      kSegment, kCluster, kBlockGroup, kBlockAdditions, kBlockMore
  //      kSegment, kCluster, kBlockGroup, kSlices, kTimeSlice
  //      etc.
  static constexpr Id kEbmlAncestory[] = {
      Id::kEbml,
  };
  static constexpr Id kSeekAncestory[] = {
      Id::kSegment,
      Id::kSeekHead,
      Id::kSeek,
  };
  static constexpr Id kInfoAncestory[] = {
      Id::kSegment,
      Id::kInfo,
  };
  static constexpr Id kBlockMoreAncestory[] = {
      Id::kSegment,        Id::kCluster,   Id::kBlockGroup,
      Id::kBlockAdditions, Id::kBlockMore,
  };
  static constexpr Id kTimeSliceAncestory[] = {
      Id::kSegment, Id::kCluster, Id::kBlockGroup, Id::kSlices, Id::kTimeSlice,
  };
  static constexpr Id kVideoAncestory[] = {
      Id::kSegment,
      Id::kTracks,
      Id::kTrackEntry,
      Id::kVideo,
  };
  static constexpr Id kAudioAncestory[] = {
      Id::kSegment,
      Id::kTracks,
      Id::kTrackEntry,
      Id::kAudio,
  };
  static constexpr Id kContentEncAesSettingsAncestory[] = {
      Id::kSegment,
      Id::kTracks,
      Id::kTrackEntry,
      Id::kContentEncodings,
      Id::kContentEncoding,
      Id::kContentEncryption,
      Id::kContentEncAesSettings,
  };
  static constexpr Id kCueTrackPositionsAncestory[] = {
      Id::kSegment,
      Id::kCues,
      Id::kCuePoint,
      Id::kCueTrackPositions,
  };
  static constexpr Id kChapterDisplayAncestory[] = {
      Id::kSegment,     Id::kChapters,       Id::kEditionEntry,
      Id::kChapterAtom, Id::kChapterDisplay,
  };
  static constexpr Id kTargetsAncestory[] = {
      Id::kSegment,
      Id::kTags,
      Id::kTag,
      Id::kTargets,
  };
  static constexpr Id kSimpleTagAncestory[] = {
      Id::kSegment,
      Id::kTags,
      Id::kTag,
      Id::kSimpleTag,
  };

  switch (id) {
    case Id::kEbmlVersion:
    case Id::kEbmlReadVersion:
    case Id::kEbmlMaxIdLength:
    case Id::kEbmlMaxSizeLength:
    case Id::kDocType:
    case Id::kDocTypeVersion:
    case Id::kDocTypeReadVersion:
      *ancestory = Ancestory(kEbmlAncestory, 1);
      return true;

    case Id::kSeekHead:
    case Id::kInfo:
    case Id::kCluster:
    case Id::kTracks:
    case Id::kCues:
    case Id::kChapters:
    case Id::kTags:
      *ancestory = Ancestory(kSeekAncestory, 1);
      return true;

    case Id::kSeek:
      *ancestory = Ancestory(kSeekAncestory, 2);
      return true;

    case Id::kSeekId:
    case Id::kSeekPosition:
      *ancestory = Ancestory(kSeekAncestory, 3);
      return true;

    case Id::kTimecodeScale:
    case Id::kDuration:
    case Id::kDateUtc:
    case Id::kTitle:
    case Id::kMuxingApp:
    case Id::kWritingApp:
      *ancestory = Ancestory(kInfoAncestory, 2);
      return true;

    case Id::kTimecode:
    case Id::kPrevSize:
    case Id::kSimpleBlock:
    case Id::kBlockGroup:
      *ancestory = Ancestory(kBlockMoreAncestory, 2);
      return true;

    case Id::kBlock:
    case Id::kBlockVirtual:
    case Id::kBlockAdditions:
    case Id::kBlockDuration:
    case Id::kReferenceBlock:
    case Id::kDiscardPadding:
    case Id::kSlices:
      *ancestory = Ancestory(kBlockMoreAncestory, 3);
      return true;

    case Id::kBlockMore:
      *ancestory = Ancestory(kBlockMoreAncestory, 4);
      return true;

    case Id::kBlockAddId:
    case Id::kBlockAdditional:
      *ancestory = Ancestory(kBlockMoreAncestory, 5);
      return true;

    case Id::kTimeSlice:
      *ancestory = Ancestory(kTimeSliceAncestory, 4);
      return true;

    case Id::kLaceNumber:
      *ancestory = Ancestory(kTimeSliceAncestory, 5);
      return true;

    case Id::kTrackEntry:
      *ancestory = Ancestory(kVideoAncestory, 2);
      return true;

    case Id::kTrackNumber:
    case Id::kTrackUid:
    case Id::kTrackType:
    case Id::kFlagEnabled:
    case Id::kFlagDefault:
    case Id::kFlagForced:
    case Id::kFlagLacing:
    case Id::kDefaultDuration:
    case Id::kName:
    case Id::kLanguage:
    case Id::kCodecId:
    case Id::kCodecPrivate:
    case Id::kCodecName:
    case Id::kCodecDelay:
    case Id::kSeekPreRoll:
    case Id::kVideo:
    case Id::kAudio:
    case Id::kContentEncodings:
      *ancestory = Ancestory(kVideoAncestory, 3);
      return true;

    case Id::kFlagInterlaced:
    case Id::kStereoMode:
    case Id::kAlphaMode:
    case Id::kPixelWidth:
    case Id::kPixelHeight:
    case Id::kPixelCropBottom:
    case Id::kPixelCropTop:
    case Id::kPixelCropLeft:
    case Id::kPixelCropRight:
    case Id::kDisplayWidth:
    case Id::kDisplayHeight:
    case Id::kDisplayUnit:
    case Id::kAspectRatioType:
    case Id::kFrameRate:
      *ancestory = Ancestory(kVideoAncestory, 4);
      return true;

    case Id::kSamplingFrequency:
    case Id::kOutputSamplingFrequency:
    case Id::kChannels:
    case Id::kBitDepth:
      *ancestory = Ancestory(kAudioAncestory, 4);
      return true;

    case Id::kContentEncoding:
      *ancestory = Ancestory(kContentEncAesSettingsAncestory, 4);
      return true;

    case Id::kContentEncodingOrder:
    case Id::kContentEncodingScope:
    case Id::kContentEncodingType:
    case Id::kContentEncryption:
      *ancestory = Ancestory(kContentEncAesSettingsAncestory, 5);
      return true;

    case Id::kContentEncAlgo:
    case Id::kContentEncKeyId:
    case Id::kContentEncAesSettings:
      *ancestory = Ancestory(kContentEncAesSettingsAncestory, 6);
      return true;

    case Id::kAesSettingsCipherMode:
      *ancestory = Ancestory(kContentEncAesSettingsAncestory, 7);
      return true;

    case Id::kCuePoint:
      *ancestory = Ancestory(kCueTrackPositionsAncestory, 2);
      return true;

    case Id::kCueTime:
    case Id::kCueTrackPositions:
      *ancestory = Ancestory(kCueTrackPositionsAncestory, 3);
      return true;

    case Id::kCueTrack:
    case Id::kCueClusterPosition:
    case Id::kCueRelativePosition:
    case Id::kCueDuration:
    case Id::kCueBlockNumber:
      *ancestory = Ancestory(kCueTrackPositionsAncestory, 4);
      return true;

    case Id::kEditionEntry:
      *ancestory = Ancestory(kChapterDisplayAncestory, 2);
      return true;

    case Id::kChapterAtom:
      *ancestory = Ancestory(kChapterDisplayAncestory, 3);
      return true;

    case Id::kChapterUid:
    case Id::kChapterStringUid:
    case Id::kChapterTimeStart:
    case Id::kChapterTimeEnd:
    case Id::kChapterDisplay:
      *ancestory = Ancestory(kChapterDisplayAncestory, 4);
      return true;

    case Id::kChapString:
    case Id::kChapLanguage:
    case Id::kChapCountry:
      *ancestory = Ancestory(kChapterDisplayAncestory, 5);
      return true;

    case Id::kTag:
      *ancestory = Ancestory(kTargetsAncestory, 2);
      return true;

    case Id::kTargets:
    case Id::kSimpleTag:
      *ancestory = Ancestory(kTargetsAncestory, 3);
      return true;

    case Id::kTargetTypeValue:
    case Id::kTargetType:
    case Id::kTagTrackUid:
      *ancestory = Ancestory(kTargetsAncestory, 4);
      return true;

    case Id::kTagName:
    case Id::kTagLanguage:
    case Id::kTagDefault:
    case Id::kTagString:
    case Id::kTagBinary:
      *ancestory = Ancestory(kSimpleTagAncestory, 4);
      return true;

    case Id::kEbml:
    case Id::kSegment:
      *ancestory = {};
      return true;

    default:
      // This is an unknown element or a global element (i.e. Void); its
      // ancestory cannot be deduced.
      *ancestory = {};
      return false;
  }
}

}  // namespace webm
