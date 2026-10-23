// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_ID_H_
#define INCLUDE_WEBM_ID_H_

#include <cstdint>

/**
 \file
 A full enumeration of WebM's EBML IDs.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 An EBML ID for a WebM element.

 The enum names correspond to the element names from the Matroska and WebM
 specifications. See those specifications for further information on each
 element.
 */
// For the WebM spec and element info, see:
// http://www.webmproject.org/docs/container/
// http://www.webmproject.org/docs/webm-encryption/#42-new-matroskawebm-elements
// http://matroska.org/technical/specs/index.html
enum class Id : std::uint32_t {
  // The MatroskaID alias links to the WebM and Matroska specifications.
  // The WebMID alias links to the WebM specification.
  // The WebMTable alias produces a table given the following arguments:
  //   Type, Level, Mandatory, Multiple, Recursive, Value range, Default value

  /**
   \MatroskaID{EBML} element ID.

   \WebMTable{Master, 0, Yes, Yes, No, , }
   */
  kEbml = 0x1A45DFA3,

  /**
   \MatroskaID{EBMLVersion} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 1}
   */
  kEbmlVersion = 0x4286,

  /**
   \MatroskaID{EBMLReadVersion} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 1}
   */
  kEbmlReadVersion = 0x42F7,

  /**
   \MatroskaID{EBMLMaxIDLength} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 4}
   */
  kEbmlMaxIdLength = 0x42F2,

  /**
   \MatroskaID{EBMLMaxSizeLength} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 8}
   */
  kEbmlMaxSizeLength = 0x42F3,

  /**
   \MatroskaID{DocType} element ID.

   \WebMTable{ASCII string, 1, Yes, No, No, , matroska}
   */
  kDocType = 0x4282,

  /**
   \MatroskaID{DocTypeVersion} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 1}
   */
  kDocTypeVersion = 0x4287,

  /**
   \MatroskaID{DocTypeReadVersion} element ID.

   \WebMTable{Unsigned integer, 1, Yes, No, No, , 1}
   */
  kDocTypeReadVersion = 0x4285,

  /**
   \MatroskaID{Void} element ID.

   \WebMTable{Binary, g, No, No, No, , }
   */
  kVoid = 0xEC,

  /**
   \MatroskaID{Segment} element ID.

   \WebMTable{Master, 0, Yes, Yes, No, , }
   */
  kSegment = 0x18538067,

  /**
   \MatroskaID{SeekHead} element ID.

   \WebMTable{Master, 1, No, Yes, No, , }
   */
  kSeekHead = 0x114D9B74,

  /**
   \MatroskaID{Seek} element ID.

   \WebMTable{Master, 2, Yes, Yes, No, , }
   */
  kSeek = 0x4DBB,

  /**
   \MatroskaID{SeekID} element ID.

   \WebMTable{Binary, 3, Yes, No, No, , }
   */
  kSeekId = 0x53AB,

  /**
   \MatroskaID{SeekPosition} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, , 0}
   */
  kSeekPosition = 0x53AC,

  /**
   \MatroskaID{Info} element ID.

   \WebMTable{Master, 1, Yes, Yes, No, , }
   */
  kInfo = 0x1549A966,

  /**
   \MatroskaID{TimecodeScale} element ID.

   \WebMTable{Unsigned integer, 2, Yes, No, No, , 1000000}
   */
  kTimecodeScale = 0x2AD7B1,

  /**
   \MatroskaID{Duration} element ID.

   \WebMTable{Float, 2, No, No, No, > 0, 0}
   */
  kDuration = 0x4489,

  /**
   \MatroskaID{DateUTC} element ID.

   \WebMTable{Date, 2, No, No, No, , 0}
   */
  kDateUtc = 0x4461,

  /**
   \MatroskaID{Title} element ID.

   \WebMTable{UTF-8 string, 2, No, No, No, , }
   */
  kTitle = 0x7BA9,

  /**
   \MatroskaID{MuxingApp} element ID.

   \WebMTable{UTF-8 string, 2, Yes, No, No, , }
   */
  kMuxingApp = 0x4D80,

  /**
   \MatroskaID{WritingApp} element ID.

   \WebMTable{UTF-8 string, 2, Yes, No, No, , }
   */
  kWritingApp = 0x5741,

  /**
   \MatroskaID{Cluster} element ID.

   \WebMTable{Master, 1, No, Yes, No, , }
   */
  kCluster = 0x1F43B675,

  /**
   \MatroskaID{Timecode} element ID.

   \WebMTable{Unsigned integer, 2, Yes, No, No, , 0}
   */
  kTimecode = 0xE7,

  /**
   \MatroskaID{PrevSize} element ID.

   \WebMTable{Unsigned integer, 2, No, No, No, , 0}
   */
  kPrevSize = 0xAB,

  /**
   \MatroskaID{SimpleBlock} element ID.

   \WebMTable{Binary, 2, No, Yes, No, , }
   */
  kSimpleBlock = 0xA3,

  /**
   \MatroskaID{BlockGroup} element ID.

   \WebMTable{Master, 2, No, Yes, No, , }
   */
  kBlockGroup = 0xA0,

  /**
   \MatroskaID{Block} element ID.

   \WebMTable{Binary, 3, Yes, No, No, , }
   */
  kBlock = 0xA1,

  /**
   \MatroskaID{BlockVirtual} (deprecated) element ID.

   \WebMTable{Binary, 3, No, No, No, , }
   */
  kBlockVirtual = 0xA2,

  /**
   \MatroskaID{BlockAdditions} element ID.

   \WebMTable{Master, 3, No, No, No, , }
   */
  kBlockAdditions = 0x75A1,

  /**
   \MatroskaID{BlockMore} element ID.

   \WebMTable{Master, 4, Yes, Yes, No, , }
   */
  kBlockMore = 0xA6,

  /**
   \MatroskaID{BlockAddID} element ID.

   \WebMTable{Unsigned integer, 5, Yes, No, No, Not 0, 1}
   */
  kBlockAddId = 0xEE,

  /**
   \MatroskaID{BlockAdditional} element ID.

   \WebMTable{Binary, 5, Yes, No, No, , }
   */
  kBlockAdditional = 0xA5,

  /**
   \MatroskaID{BlockDuration} element ID.

   \WebMTable{Unsigned integer, 3, No, No, No, , DefaultDuration}
   */
  kBlockDuration = 0x9B,

  /**
   \MatroskaID{ReferenceBlock} element ID.

   \WebMTable{Signed integer, 3, No, Yes, No, , 0}
   */
  kReferenceBlock = 0xFB,

  /**
   \MatroskaID{DiscardPadding} element ID.

   \WebMTable{Signed integer, 3, No, No, No, , 0}
   */
  kDiscardPadding = 0x75A2,

  /**
   \MatroskaID{Slices} (deprecated).

   \WebMTable{Master, 3, No, No, No, , }
   */
  kSlices = 0x8E,

  /**
   \MatroskaID{TimeSlice} (deprecated) element ID.

   \WebMTable{Master, 4, No, Yes, No, , }
   */
  kTimeSlice = 0xE8,

  /**
   \MatroskaID{LaceNumber} (deprecated) element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kLaceNumber = 0xCC,

  /**
   \MatroskaID{Tracks} element ID.

   \WebMTable{Master, 1, No, Yes, No, , }
   */
  kTracks = 0x1654AE6B,

  /**
   \MatroskaID{TrackEntry} element ID.

   \WebMTable{Master, 2, Yes, Yes, No, , }
   */
  kTrackEntry = 0xAE,

  /**
   \MatroskaID{TrackNumber} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, Not 0, 0}
   */
  kTrackNumber = 0xD7,

  /**
   \MatroskaID{TrackUID} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, Not 0, 0}
   */
  kTrackUid = 0x73C5,

  /**
   \MatroskaID{TrackType} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, 1-254, 0}
   */
  kTrackType = 0x83,

  /**
   \MatroskaID{FlagEnabled} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, 0-1, 1}
   */
  kFlagEnabled = 0xB9,

  /**
   \MatroskaID{FlagDefault} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, 0-1, 1}
   */
  kFlagDefault = 0x88,

  /**
   \MatroskaID{FlagForced} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, 0-1, 0}
   */
  kFlagForced = 0x55AA,

  /**
   \MatroskaID{FlagLacing} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, 0-1, 1}
   */
  kFlagLacing = 0x9C,

  /**
   \MatroskaID{DefaultDuration} element ID.

   \WebMTable{Unsigned integer, 3, No, No, No, Not 0, 0}
   */
  kDefaultDuration = 0x23E383,

  /**
   \MatroskaID{Name} element ID.

   \WebMTable{UTF-8 string, 3, No, No, No, , }
   */
  kName = 0x536E,

  /**
   \MatroskaID{Language} element ID.

   \WebMTable{ASCII string, 3, No, No, No, , eng}
   */
  kLanguage = 0x22B59C,

  /**
   \MatroskaID{CodecID} element ID.

   \WebMTable{ASCII string, 3, Yes, No, No, , }
   */
  kCodecId = 0x86,

  /**
   \MatroskaID{CodecPrivate} element ID.

   \WebMTable{Binary, 3, No, No, No, , }
   */
  kCodecPrivate = 0x63A2,

  /**
   \MatroskaID{CodecName} element ID.

   \WebMTable{UTF-8 string, 3, No, No, No, , }
   */
  kCodecName = 0x258688,

  /**
   \MatroskaID{CodecDelay} element ID.

   \WebMTable{Unsigned integer, 3, No, No, No, , 0}
   */
  kCodecDelay = 0x56AA,

  /**
   \MatroskaID{SeekPreRoll} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, , 0}
   */
  kSeekPreRoll = 0x56BB,

  /**
   \MatroskaID{Video} element ID.

   \WebMTable{Master, 3, No, No, No, , }
   */
  kVideo = 0xE0,

  /**
   \MatroskaID{FlagInterlaced} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, 0-1, 0}
   */
  kFlagInterlaced = 0x9A,

  /**
   \MatroskaID{StereoMode} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kStereoMode = 0x53B8,

  /**
   \MatroskaID{AlphaMode} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kAlphaMode = 0x53C0,

  /**
   \MatroskaID{PixelWidth} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, Not 0, 0}
   */
  kPixelWidth = 0xB0,

  /**
   \MatroskaID{PixelHeight} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, Not 0, 0}
   */
  kPixelHeight = 0xBA,

  /**
   \MatroskaID{PixelCropBottom} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kPixelCropBottom = 0x54AA,

  /**
   \MatroskaID{PixelCropTop} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kPixelCropTop = 0x54BB,

  /**
   \MatroskaID{PixelCropLeft} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kPixelCropLeft = 0x54CC,

  /**
   \MatroskaID{PixelCropRight} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kPixelCropRight = 0x54DD,

  /**
   \MatroskaID{DisplayWidth} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, Not 0, PixelWidth}
   */
  kDisplayWidth = 0x54B0,

  /**
   \MatroskaID{DisplayHeight} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, Not 0, PixelHeight}
   */
  kDisplayHeight = 0x54BA,

  /**
   \MatroskaID{DisplayUnit} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kDisplayUnit = 0x54B2,

  /**
   \MatroskaID{AspectRatioType} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kAspectRatioType = 0x54B3,

  /**
   \MatroskaID{FrameRate} (deprecated) element ID.

   \WebMTable{Float, 4, No, No, No, > 0, 0}
   */
  kFrameRate = 0x2383E3,

  /**
   \MatroskaID{Colour} element ID.

   \WebMTable{Master, 4, No, No, No, , }
   */
  kColour = 0x55B0,

  /**
   \MatroskaID{MatrixCoefficients} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 2}
   */
  kMatrixCoefficients = 0x55B1,

  /**
   \MatroskaID{BitsPerChannel} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kBitsPerChannel = 0x55B2,

  /**
   \MatroskaID{ChromaSubsamplingHorz} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kChromaSubsamplingHorz = 0x55B3,

  /**
   \MatroskaID{ChromaSubsamplingVert} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kChromaSubsamplingVert = 0x55B4,

  /**
   \MatroskaID{CbSubsamplingHorz} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kCbSubsamplingHorz = 0x55B5,

  /**
   \MatroskaID{CbSubsamplingVert} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kCbSubsamplingVert = 0x55B6,

  /**
   \MatroskaID{ChromaSitingHorz} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kChromaSitingHorz = 0x55B7,

  /**
   \MatroskaID{ChromaSitingVert} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kChromaSitingVert = 0x55B8,

  /**
   \MatroskaID{Range} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kRange = 0x55B9,

  /**
   \MatroskaID{TransferCharacteristics} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 2}
   */
  kTransferCharacteristics = 0x55BA,

  /**
   \MatroskaID{Primaries} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 2}
   */
  kPrimaries = 0x55BB,

  /**
   \MatroskaID{MaxCLL} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kMaxCll = 0x55BC,

  /**
   \MatroskaID{MaxFALL} element ID.

   \WebMTable{Unsigned integer, 5, No, No, No, , 0}
   */
  kMaxFall = 0x55BD,

  /**
   \MatroskaID{MasteringMetadata} element ID.

   \WebMTable{Master, 5, No, No, No, , }
   */
  kMasteringMetadata = 0x55D0,

  /**
   \MatroskaID{PrimaryRChromaticityX} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryRChromaticityX = 0x55D1,

  /**
   \MatroskaID{PrimaryRChromaticityY} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryRChromaticityY = 0x55D2,

  /**
   \MatroskaID{PrimaryGChromaticityX} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryGChromaticityX = 0x55D3,

  /**
   \MatroskaID{PrimaryGChromaticityY} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryGChromaticityY = 0x55D4,

  /**
   \MatroskaID{PrimaryBChromaticityX} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryBChromaticityX = 0x55D5,

  /**
   \MatroskaID{PrimaryBChromaticityY} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kPrimaryBChromaticityY = 0x55D6,

  /**
   \MatroskaID{WhitePointChromaticityX} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kWhitePointChromaticityX = 0x55D7,

  /**
   \MatroskaID{WhitePointChromaticityY} element ID.

   \WebMTable{Float, 6, No, No, No, 0-1, 0}
   */
  kWhitePointChromaticityY = 0x55D8,

  /**
   \MatroskaID{LuminanceMax} element ID.

   \WebMTable{Float, 6, No, No, No, 0-9999.99, 0}
   */
  kLuminanceMax = 0x55D9,

  /**
   \MatroskaID{LuminanceMin} element ID.

   \WebMTable{Float, 6, No, No, No, 0-999.9999, 0}
   */
  kLuminanceMin = 0x55DA,

  /**
   \WebMID{Projection} element ID.

   \WebMTable{Master, 5, No, No, No, , }
   */
  kProjection = 0x7670,

  /**
   \WebMID{ProjectionType} element ID.

   \WebMTable{Unsigned integer, 6, Yes, No, No, , 0}
   */
  kProjectionType = 0x7671,

  /**
   \WebMID{ProjectionPrivate} element ID.

   \WebMTable{Binary, 6, No, No, No, , }
   */
  kProjectionPrivate = 0x7672,

  /**
   \WebMID{ProjectionPoseYaw} element ID.

   \WebMTable{Float, 6, Yes, No, No, , 0}
   */
  kProjectionPoseYaw = 0x7673,

  /**
   \WebMID{ProjectionPosePitch} element ID.

   \WebMTable{Float, 6, Yes, No, No, , 0}
   */
  kProjectionPosePitch = 0x7674,

  /**
   \WebMID{ProjectionPoseRoll} element ID.

   \WebMTable{Float, 6, Yes, No, No, , 0}
   */
  kProjectionPoseRoll = 0x7675,

  /**
   \MatroskaID{Audio} element ID.

   \WebMTable{Master, 3, No, No, No, , }
   */
  kAudio = 0xE1,

  /**
   \MatroskaID{SamplingFrequency} element ID.

   \WebMTable{Float, 4, Yes, No, No, > 0, 8000}
   */
  kSamplingFrequency = 0xB5,

  /**
   \MatroskaID{OutputSamplingFrequency} element ID.

   \WebMTable{Float, 4, No, No, No, > 0, SamplingFrequency}
   */
  kOutputSamplingFrequency = 0x78B5,

  /**
   \MatroskaID{Channels} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, Not 0, 1}
   */
  kChannels = 0x9F,

  /**
   \MatroskaID{BitDepth} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, Not 0, 0}
   */
  kBitDepth = 0x6264,

  /**
   \MatroskaID{ContentEncodings} element ID.

   \WebMTable{Master, 3, No, No, No, , }
   */
  kContentEncodings = 0x6D80,

  /**
   \MatroskaID{ContentEncoding} element ID.

   \WebMTable{Master, 4, Yes, Yes, No, , }
   */
  kContentEncoding = 0x6240,

  /**
   \MatroskaID{ContentEncodingOrder} element ID.

   \WebMTable{Unsigned integer, 5, Yes, No, No, , 0}
   */
  kContentEncodingOrder = 0x5031,

  /**
   \MatroskaID{ContentEncodingScope} element ID.

   \WebMTable{Unsigned integer, 5, Yes, No, No, Not 0, 1}
   */
  kContentEncodingScope = 0x5032,

  /**
   \MatroskaID{ContentEncodingType} element ID.

   \WebMTable{Unsigned integer, 5, Yes, No, No, , 0}
   */
  kContentEncodingType = 0x5033,

  /**
   \MatroskaID{ContentEncryption} element ID.

   \WebMTable{Master, 5, No, No, No, , }
   */
  kContentEncryption = 0x5035,

  /**
   \MatroskaID{ContentEncAlgo} element ID.

   \WebMTable{Unsigned integer, 6, No, No, No, , 0}
   */
  kContentEncAlgo = 0x47E1,

  /**
   \MatroskaID{ContentEncKeyID} element ID.

   \WebMTable{Binary, 6, No, No, No, , }
   */
  kContentEncKeyId = 0x47E2,

  /**
   \WebMID{ContentEncAESSettings} element ID.

   \WebMTable{Master, 6, No, No, No, , }
   */
  kContentEncAesSettings = 0x47E7,

  /**
   \WebMID{AESSettingsCipherMode} element ID.

   \WebMTable{Unsigned integer, 7, Yes, No, No, 1, 1}
   */
  kAesSettingsCipherMode = 0x47E8,

  /**
   \MatroskaID{Cues} element ID.

   \WebMTable{Master, 1, No, No, No, , }
   */
  kCues = 0x1C53BB6B,

  /**
   \MatroskaID{CuePoint} element ID.

   \WebMTable{Master, 2, Yes, Yes, No, , }
   */
  kCuePoint = 0xBB,

  /**
   \MatroskaID{CueTime} element ID.

   \WebMTable{Unsigned integer, 3, Yes, No, No, , 0}
   */
  kCueTime = 0xB3,

  /**
   \MatroskaID{CueTrackPositions} element ID.

   \WebMTable{Master, 3, Yes, Yes, No, , }
   */
  kCueTrackPositions = 0xB7,

  /**
   \MatroskaID{CueTrack} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, Not 0, 0}
   */
  kCueTrack = 0xF7,

  /**
   \MatroskaID{CueClusterPosition} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, , 0}
   */
  kCueClusterPosition = 0xF1,

  /**
   \MatroskaID{CueRelativePosition} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kCueRelativePosition = 0xF0,

  /**
   \MatroskaID{CueDuration} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kCueDuration = 0xB2,

  /**
   \MatroskaID{CueBlockNumber} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, Not 0, 1}
   */
  kCueBlockNumber = 0x5378,

  /**
   \MatroskaID{Chapters} element ID.

   \WebMTable{Master, 1, No, No, No, , }
   */
  kChapters = 0x1043A770,

  /**
   \MatroskaID{EditionEntry} element ID.

   \WebMTable{Master, 2, Yes, Yes, No, , }
   */
  kEditionEntry = 0x45B9,

  /**
   \MatroskaID{ChapterAtom} element ID.

   \WebMTable{Master, 3, Yes, Yes, Yes, , }
   */
  kChapterAtom = 0xB6,

  /**
   \MatroskaID{ChapterUID} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, Not 0, 0}
   */
  kChapterUid = 0x73C4,

  /**
   \MatroskaID{ChapterStringUID} element ID.

   \WebMTable{UTF-8 string, 4, No, No, No, , }
   */
  kChapterStringUid = 0x5654,

  /**
   \MatroskaID{ChapterTimeStart} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, , 0}
   */
  kChapterTimeStart = 0x91,

  /**
   \MatroskaID{ChapterTimeEnd} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 0}
   */
  kChapterTimeEnd = 0x92,

  /**
   \MatroskaID{ChapterDisplay} element ID.

   \WebMTable{Master, 4, No, Yes, No, , }
   */
  kChapterDisplay = 0x80,

  /**
   \MatroskaID{ChapString} element ID.

   \WebMTable{UTF-8 string, 5, Yes, No, No, , }
   */
  kChapString = 0x85,

  /**
   \MatroskaID{ChapLanguage} element ID.

   \WebMTable{ASCII string, 5, Yes, Yes, No, , eng}
   */
  kChapLanguage = 0x437C,

  /**
   \MatroskaID{ChapCountry} element ID.

   \WebMTable{ASCII string, 5, No, Yes, No, , }
   */
  kChapCountry = 0x437E,

  /**
   \MatroskaID{Tags} element ID.

   \WebMTable{Master, 1, No, Yes, No, , }
   */
  kTags = 0x1254C367,

  /**
   \MatroskaID{Tag} element ID.

   \WebMTable{Master, 2, Yes, Yes, No, , }
   */
  kTag = 0x7373,

  /**
   \MatroskaID{Targets} element ID.

   \WebMTable{Master, 3, Yes, No, No, , }
   */
  kTargets = 0x63C0,

  /**
   \MatroskaID{TargetTypeValue} element ID.

   \WebMTable{Unsigned integer, 4, No, No, No, , 50}
   */
  kTargetTypeValue = 0x68CA,

  /**
   \MatroskaID{TargetType} element ID.

   \WebMTable{ASCII string, 4, No, No, No, , }
   */
  kTargetType = 0x63CA,

  /**
   \MatroskaID{TagTrackUID} element ID.

   \WebMTable{Unsigned integer, 4, No, Yes, No, , 0}
   */
  kTagTrackUid = 0x63C5,

  /**
   \MatroskaID{SimpleTag} element ID.

   \WebMTable{Master, 3, Yes, Yes, Yes, , }
   */
  kSimpleTag = 0x67C8,

  /**
   \MatroskaID{TagName} element ID.

   \WebMTable{UTF-8 string, 4, Yes, No, No, , }
   */
  kTagName = 0x45A3,

  /**
   \MatroskaID{TagLanguage} element ID.

   \WebMTable{ASCII string, 4, Yes, No, No, , und}
   */
  kTagLanguage = 0x447A,

  /**
   \MatroskaID{TagDefault} element ID.

   \WebMTable{Unsigned integer, 4, Yes, No, No, 0-1, 1}
   */
  kTagDefault = 0x4484,

  /**
   \MatroskaID{TagString} element ID.

   \WebMTable{UTF-8 string, 4, No, No, No, , }
   */
  kTagString = 0x4487,

  /**
   \MatroskaID{TagBinary} element ID.

   \WebMTable{Binary, 4, No, No, No, , }
   */
  kTagBinary = 0x4485,
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_ID_H_
