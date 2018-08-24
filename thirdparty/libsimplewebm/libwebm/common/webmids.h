// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef COMMON_WEBMIDS_H_
#define COMMON_WEBMIDS_H_

namespace libwebm {

enum MkvId {
  kMkvEBML = 0x1A45DFA3,
  kMkvEBMLVersion = 0x4286,
  kMkvEBMLReadVersion = 0x42F7,
  kMkvEBMLMaxIDLength = 0x42F2,
  kMkvEBMLMaxSizeLength = 0x42F3,
  kMkvDocType = 0x4282,
  kMkvDocTypeVersion = 0x4287,
  kMkvDocTypeReadVersion = 0x4285,
  kMkvVoid = 0xEC,
  kMkvSignatureSlot = 0x1B538667,
  kMkvSignatureAlgo = 0x7E8A,
  kMkvSignatureHash = 0x7E9A,
  kMkvSignaturePublicKey = 0x7EA5,
  kMkvSignature = 0x7EB5,
  kMkvSignatureElements = 0x7E5B,
  kMkvSignatureElementList = 0x7E7B,
  kMkvSignedElement = 0x6532,
  // segment
  kMkvSegment = 0x18538067,
  // Meta Seek Information
  kMkvSeekHead = 0x114D9B74,
  kMkvSeek = 0x4DBB,
  kMkvSeekID = 0x53AB,
  kMkvSeekPosition = 0x53AC,
  // Segment Information
  kMkvInfo = 0x1549A966,
  kMkvTimecodeScale = 0x2AD7B1,
  kMkvDuration = 0x4489,
  kMkvDateUTC = 0x4461,
  kMkvTitle = 0x7BA9,
  kMkvMuxingApp = 0x4D80,
  kMkvWritingApp = 0x5741,
  // Cluster
  kMkvCluster = 0x1F43B675,
  kMkvTimecode = 0xE7,
  kMkvPrevSize = 0xAB,
  kMkvBlockGroup = 0xA0,
  kMkvBlock = 0xA1,
  kMkvBlockDuration = 0x9B,
  kMkvReferenceBlock = 0xFB,
  kMkvLaceNumber = 0xCC,
  kMkvSimpleBlock = 0xA3,
  kMkvBlockAdditions = 0x75A1,
  kMkvBlockMore = 0xA6,
  kMkvBlockAddID = 0xEE,
  kMkvBlockAdditional = 0xA5,
  kMkvDiscardPadding = 0x75A2,
  // Track
  kMkvTracks = 0x1654AE6B,
  kMkvTrackEntry = 0xAE,
  kMkvTrackNumber = 0xD7,
  kMkvTrackUID = 0x73C5,
  kMkvTrackType = 0x83,
  kMkvFlagEnabled = 0xB9,
  kMkvFlagDefault = 0x88,
  kMkvFlagForced = 0x55AA,
  kMkvFlagLacing = 0x9C,
  kMkvDefaultDuration = 0x23E383,
  kMkvMaxBlockAdditionID = 0x55EE,
  kMkvName = 0x536E,
  kMkvLanguage = 0x22B59C,
  kMkvCodecID = 0x86,
  kMkvCodecPrivate = 0x63A2,
  kMkvCodecName = 0x258688,
  kMkvCodecDelay = 0x56AA,
  kMkvSeekPreRoll = 0x56BB,
  // video
  kMkvVideo = 0xE0,
  kMkvFlagInterlaced = 0x9A,
  kMkvStereoMode = 0x53B8,
  kMkvAlphaMode = 0x53C0,
  kMkvPixelWidth = 0xB0,
  kMkvPixelHeight = 0xBA,
  kMkvPixelCropBottom = 0x54AA,
  kMkvPixelCropTop = 0x54BB,
  kMkvPixelCropLeft = 0x54CC,
  kMkvPixelCropRight = 0x54DD,
  kMkvDisplayWidth = 0x54B0,
  kMkvDisplayHeight = 0x54BA,
  kMkvDisplayUnit = 0x54B2,
  kMkvAspectRatioType = 0x54B3,
  kMkvFrameRate = 0x2383E3,
  // end video
  // colour
  kMkvColour = 0x55B0,
  kMkvMatrixCoefficients = 0x55B1,
  kMkvBitsPerChannel = 0x55B2,
  kMkvChromaSubsamplingHorz = 0x55B3,
  kMkvChromaSubsamplingVert = 0x55B4,
  kMkvCbSubsamplingHorz = 0x55B5,
  kMkvCbSubsamplingVert = 0x55B6,
  kMkvChromaSitingHorz = 0x55B7,
  kMkvChromaSitingVert = 0x55B8,
  kMkvRange = 0x55B9,
  kMkvTransferCharacteristics = 0x55BA,
  kMkvPrimaries = 0x55BB,
  kMkvMaxCLL = 0x55BC,
  kMkvMaxFALL = 0x55BD,
  // mastering metadata
  kMkvMasteringMetadata = 0x55D0,
  kMkvPrimaryRChromaticityX = 0x55D1,
  kMkvPrimaryRChromaticityY = 0x55D2,
  kMkvPrimaryGChromaticityX = 0x55D3,
  kMkvPrimaryGChromaticityY = 0x55D4,
  kMkvPrimaryBChromaticityX = 0x55D5,
  kMkvPrimaryBChromaticityY = 0x55D6,
  kMkvWhitePointChromaticityX = 0x55D7,
  kMkvWhitePointChromaticityY = 0x55D8,
  kMkvLuminanceMax = 0x55D9,
  kMkvLuminanceMin = 0x55DA,
  // end mastering metadata
  // end colour
  // projection
  kMkvProjection = 0x7670,
  kMkvProjectionType = 0x7671,
  kMkvProjectionPrivate = 0x7672,
  kMkvProjectionPoseYaw = 0x7673,
  kMkvProjectionPosePitch = 0x7674,
  kMkvProjectionPoseRoll = 0x7675,
  // end projection
  // audio
  kMkvAudio = 0xE1,
  kMkvSamplingFrequency = 0xB5,
  kMkvOutputSamplingFrequency = 0x78B5,
  kMkvChannels = 0x9F,
  kMkvBitDepth = 0x6264,
  // end audio
  // ContentEncodings
  kMkvContentEncodings = 0x6D80,
  kMkvContentEncoding = 0x6240,
  kMkvContentEncodingOrder = 0x5031,
  kMkvContentEncodingScope = 0x5032,
  kMkvContentEncodingType = 0x5033,
  kMkvContentCompression = 0x5034,
  kMkvContentCompAlgo = 0x4254,
  kMkvContentCompSettings = 0x4255,
  kMkvContentEncryption = 0x5035,
  kMkvContentEncAlgo = 0x47E1,
  kMkvContentEncKeyID = 0x47E2,
  kMkvContentSignature = 0x47E3,
  kMkvContentSigKeyID = 0x47E4,
  kMkvContentSigAlgo = 0x47E5,
  kMkvContentSigHashAlgo = 0x47E6,
  kMkvContentEncAESSettings = 0x47E7,
  kMkvAESSettingsCipherMode = 0x47E8,
  kMkvAESSettingsCipherInitData = 0x47E9,
  // end ContentEncodings
  // Cueing Data
  kMkvCues = 0x1C53BB6B,
  kMkvCuePoint = 0xBB,
  kMkvCueTime = 0xB3,
  kMkvCueTrackPositions = 0xB7,
  kMkvCueTrack = 0xF7,
  kMkvCueClusterPosition = 0xF1,
  kMkvCueBlockNumber = 0x5378,
  // Chapters
  kMkvChapters = 0x1043A770,
  kMkvEditionEntry = 0x45B9,
  kMkvChapterAtom = 0xB6,
  kMkvChapterUID = 0x73C4,
  kMkvChapterStringUID = 0x5654,
  kMkvChapterTimeStart = 0x91,
  kMkvChapterTimeEnd = 0x92,
  kMkvChapterDisplay = 0x80,
  kMkvChapString = 0x85,
  kMkvChapLanguage = 0x437C,
  kMkvChapCountry = 0x437E,
  // Tags
  kMkvTags = 0x1254C367,
  kMkvTag = 0x7373,
  kMkvSimpleTag = 0x67C8,
  kMkvTagName = 0x45A3,
  kMkvTagString = 0x4487
};

}  // namespace libwebm

#endif  // COMMON_WEBMIDS_H_
