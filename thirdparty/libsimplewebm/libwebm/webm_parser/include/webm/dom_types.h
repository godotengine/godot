// Copyright (c) 2016 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
#ifndef INCLUDE_WEBM_DOM_TYPES_H_
#define INCLUDE_WEBM_DOM_TYPES_H_

#include <cstdint>
#include <string>
#include <vector>

#include "./element.h"
#include "./id.h"

/**
 \file
 Data structures representing parsed DOM objects.

 For more information on each type and member, see the WebM specification for
 the element that each type/member represents.
 */

namespace webm {

/**
 \addtogroup PUBLIC_API
 @{
 */

/**
 Metadata for a single frame.
 */
struct FrameMetadata {
  /**
   Metadata for the EBML element (\WebMID{Block} or \WebMID{SimpleBlock}) that
   contains this frame.
   */
  ElementMetadata parent_element;

  /**
   Absolute byte position (from the beginning of the byte stream/file) of the
   frame start.
   */
  std::uint64_t position;

  /**
   Size (in bytes) of the frame.
   */
  std::uint64_t size;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const FrameMetadata& other) const {
    return parent_element == other.parent_element &&
           position == other.position && size == other.size;
  }
};

/**
 A parsed \WebMID{BlockMore} element.
 */
struct BlockMore {
  /**
   A parsed \WebMID{BlockAddID} element.
   */
  Element<std::uint64_t> id{1};

  /**
   A parsed \WebMID{BlockAdditional} element.
   */
  Element<std::vector<std::uint8_t>> data;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const BlockMore& other) const {
    return id == other.id && data == other.data;
  }
};

/**
 A parsed \WebMID{BlockAdditions} element.
 */
struct BlockAdditions {
  /**
   Parsed \WebMID{BlockMore} elements.
   */
  std::vector<Element<BlockMore>> block_mores;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const BlockAdditions& other) const {
    return block_mores == other.block_mores;
  }
};

/**
 A parsed \WebMID{TimeSlice} element (deprecated).
 */
struct TimeSlice {
  /**
   A parsed \WebMID{LaceNumber} element (deprecated).
   */
  Element<std::uint64_t> lace_number;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const TimeSlice& other) const {
    return lace_number == other.lace_number;
  }
};

/**
 A parsed \WebMID{Slices} element (deprecated).
 */
struct Slices {
  /**
   Parsed \WebMID{TimeSlice} elements (deprecated).
   */
  std::vector<Element<TimeSlice>> slices;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Slices& other) const { return slices == other.slices; }
};

/**
 A parsed \WebMID{BlockVirtual} element.
 */
struct VirtualBlock {
  /**
   The virtual block's track number.
   */
  std::uint64_t track_number;

  /**
   The timecode of the virtual block.
   */
  std::int16_t timecode;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const VirtualBlock& other) const {
    return track_number == other.track_number && timecode == other.timecode;
  }
};

/**
 The frame lacing used in a block.
 */
enum class Lacing : std::uint8_t {
  /**
   No lacing is used.
   */
  kNone = 0x00,

  /**
   Xiph-style lacing is used.
   */
  kXiph = 0x02,

  /**
   Fixed-lacing is used, where each frame has the same, fixed size.
   */
  kFixed = 0x04,

  /**
   EBML-style lacing is used.
   */
  kEbml = 0x06,
};

/**
 A parsed \WebMID{Block} element.
 */
struct Block {
  /**
   The block's track number.
   */
  std::uint64_t track_number;

  /**
   The number of frames in the block.
   */
  int num_frames;

  /**
   The timecode of the block (relative to the containing \WebMID{Cluster}'s
   timecode).
   */
  std::int16_t timecode;

  /**
   The lacing used to store frames in the block.
   */
  Lacing lacing;

  /**
   True if the frames are visible, false if they are invisible.
   */
  bool is_visible;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Block& other) const {
    return track_number == other.track_number &&
           num_frames == other.num_frames && timecode == other.timecode &&
           lacing == other.lacing && is_visible == other.is_visible;
  }
};

/**
 A parsed \WebMID{SimpleBlock} element.
 */
struct SimpleBlock : public Block {
  /**
   True if the frames are all key frames.
   */
  bool is_key_frame;

  /**
   True if frames can be discarded during playback if needed.
   */
  bool is_discardable;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const SimpleBlock& other) const {
    return Block::operator==(other) && is_key_frame == other.is_key_frame &&
           is_discardable == other.is_discardable;
  }
};

/**
 A parsed \WebMID{BlockGroup} element.
 */
struct BlockGroup {
  /**
   A parsed \WebMID{Block} element.
   */
  Element<Block> block;

  /**
   A parsed \WebMID{BlockVirtual} element.
   */
  Element<VirtualBlock> virtual_block;

  /**
   A parsed \WebMID{BlockAdditions} element.
   */
  Element<BlockAdditions> additions;

  /**
   A parsed \WebMID{BlockDuration} element.
   */
  Element<std::uint64_t> duration;

  /**
   Parsed \WebMID{ReferenceBlock} elements.
   */
  std::vector<Element<std::int64_t>> references;

  /**
   A parsed \WebMID{DiscardPadding} element.
   */
  Element<std::int64_t> discard_padding;

  /**
   A parsed \WebMID{Slices} element (deprecated).
   */
  Element<Slices> slices;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const BlockGroup& other) const {
    return block == other.block && virtual_block == other.virtual_block &&
           additions == other.additions && duration == other.duration &&
           references == other.references &&
           discard_padding == other.discard_padding && slices == other.slices;
  }
};

/**
 A parsed \WebMID{Cluster} element.
 */
struct Cluster {
  /**
   A parsed \WebMID{Timecode} element.
   */
  Element<std::uint64_t> timecode;

  /**
   A parsed \WebMID{PrevSize} element.
   */
  Element<std::uint64_t> previous_size;

  /**
   Parsed \WebMID{SimpleBlock} elements.
   */
  std::vector<Element<SimpleBlock>> simple_blocks;

  /**
   Parsed \WebMID{BlockGroup} elements.
   */
  std::vector<Element<BlockGroup>> block_groups;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Cluster& other) const {
    return timecode == other.timecode && previous_size == other.previous_size &&
           simple_blocks == other.simple_blocks &&
           block_groups == other.block_groups;
  }
};

/**
 A parsed \WebMID{EBML} element.
 */
struct Ebml {
  /**
   A parsed \WebMID{EBMLVersion} element.
   */
  Element<std::uint64_t> ebml_version{1};

  /**
   A parsed \WebMID{EBMLReadVersion} element.
   */
  Element<std::uint64_t> ebml_read_version{1};

  /**
   A parsed \WebMID{EBMLMaxIDLength} element.
   */
  Element<std::uint64_t> ebml_max_id_length{4};

  /**
   A parsed \WebMID{EBMLMaxSizeLength} element.
   */
  Element<std::uint64_t> ebml_max_size_length{8};

  /**
   A parsed \WebMID{DocType} element.
   */
  Element<std::string> doc_type{"matroska"};

  /**
   A parsed \WebMID{DocTypeVersion} element.
   */
  Element<std::uint64_t> doc_type_version{1};

  /**
   A parsed \WebMID{DocTypeReadVersion} element.
   */
  Element<std::uint64_t> doc_type_read_version{1};

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Ebml& other) const {
    return ebml_version == other.ebml_version &&
           ebml_read_version == other.ebml_read_version &&
           ebml_max_id_length == other.ebml_max_id_length &&
           ebml_max_size_length == other.ebml_max_size_length &&
           doc_type == other.doc_type &&
           doc_type_version == other.doc_type_version &&
           doc_type_read_version == other.doc_type_read_version;
  }
};

/**
 A parsed \WebMID{Info} element.
 */
struct Info {
  /**
   A parsed \WebMID{TimecodeScale} element.
   */
  Element<std::uint64_t> timecode_scale{1000000};

  /**
   A parsed \WebMID{Duration} element.
   */
  Element<double> duration;

  /**
   A parsed \WebMID{DateUTC} element.
   */
  Element<std::int64_t> date_utc;

  /**
   A parsed \WebMID{Title} element.
   */
  Element<std::string> title;

  /**
   A parsed \WebMID{MuxingApp} element.
   */
  Element<std::string> muxing_app;

  /**
   A parsed \WebMID{WritingApp} element.
   */
  Element<std::string> writing_app;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Info& other) const {
    return timecode_scale == other.timecode_scale &&
           duration == other.duration && date_utc == other.date_utc &&
           title == other.title && muxing_app == other.muxing_app &&
           writing_app == other.writing_app;
  }
};

/**
 A parsed \WebMID{Seek} element.
 */
struct Seek {
  /**
   A parsed \WebMID{SeekID} element.
   */
  Element<Id> id;

  /**
   A parsed \WebMID{SeekPosition} element.
   */
  Element<std::uint64_t> position;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Seek& other) const {
    return id == other.id && position == other.position;
  }
};

/**
 A parsed \WebMID{Audio} element.
 */
struct Audio {
  /**
   A parsed \WebMID{SamplingFrequency} element.
   */
  Element<double> sampling_frequency{8000};

  /**
   A parsed \WebMID{OutputSamplingFrequency} element.
   */
  Element<double> output_frequency{8000};

  /**
   A parsed \WebMID{Channels} element.
   */
  Element<std::uint64_t> channels{1};

  /**
   A parsed \WebMID{BitDepth} element.
   */
  Element<std::uint64_t> bit_depth;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Audio& other) const {
    return sampling_frequency == other.sampling_frequency &&
           output_frequency == other.output_frequency &&
           channels == other.channels && bit_depth == other.bit_depth;
  }
};

/**
 A parsed \WebMID{MasteringMetadata} element.
 */
struct MasteringMetadata {
  /**
   A parsed \WebMID{PrimaryRChromaticityX} element.
   */
  Element<double> primary_r_chromaticity_x;

  /**
   A parsed \WebMID{PrimaryRChromaticityY} element.
   */
  Element<double> primary_r_chromaticity_y;

  /**
   A parsed \WebMID{PrimaryGChromaticityX} element.
   */
  Element<double> primary_g_chromaticity_x;

  /**
   A parsed \WebMID{PrimaryGChromaticityY} element.
   */
  Element<double> primary_g_chromaticity_y;

  /**
   A parsed \WebMID{PrimaryBChromaticityX} element.
   */
  Element<double> primary_b_chromaticity_x;

  /**
   A parsed \WebMID{PrimaryBChromaticityY} element.
   */
  Element<double> primary_b_chromaticity_y;

  /**
   A parsed \WebMID{WhitePointChromaticityX} element.
   */
  Element<double> white_point_chromaticity_x;

  /**
   A parsed \WebMID{WhitePointChromaticityY} element.
   */
  Element<double> white_point_chromaticity_y;

  /**
   A parsed \WebMID{LuminanceMax} element.
   */
  Element<double> luminance_max;

  /**
   A parsed \WebMID{LuminanceMin} element.
   */
  Element<double> luminance_min;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const MasteringMetadata& other) const {
    return primary_r_chromaticity_x == other.primary_r_chromaticity_x &&
           primary_r_chromaticity_y == other.primary_r_chromaticity_y &&
           primary_g_chromaticity_x == other.primary_g_chromaticity_x &&
           primary_g_chromaticity_y == other.primary_g_chromaticity_y &&
           primary_b_chromaticity_x == other.primary_b_chromaticity_x &&
           primary_b_chromaticity_y == other.primary_b_chromaticity_y &&
           white_point_chromaticity_x == other.white_point_chromaticity_x &&
           white_point_chromaticity_y == other.white_point_chromaticity_y &&
           luminance_max == other.luminance_max &&
           luminance_min == other.luminance_min;
  }
};

/**
 A parsed \WebMID{MatrixCoefficients} element.

 Matroska/WebM adopted these values from Table 4 of ISO/IEC 23001-8:2013/DCOR1.
 See that document for further details.
 */
enum class MatrixCoefficients : std::uint64_t {
  /**
   The identity matrix.

   Typically used for GBR (often referred to as RGB); however, may also be used
   for YZX (often referred to as XYZ).
   */
  kRgb = 0,

  /**
   Rec. ITU-R BT.709-5.
   */
  kBt709 = 1,

  /**
   Image characteristics are unknown or are determined by the application.
   */
  kUnspecified = 2,

  /**
   United States Federal Communications Commission Title 47 Code of Federal
   Regulations (2003) 73.682 (a) (20).
   */
  kFcc = 4,

  /**
   Rec. ITU-R BT.470‑6 System B, G (historical).
   */
  kBt470Bg = 5,

  /**
   Society of Motion Picture and Television Engineers 170M (2004).
   */
  kSmpte170M = 6,

  /**
   Society of Motion Picture and Television Engineers 240M (1999).
   */
  kSmpte240M = 7,

  /**
   YCgCo.
   */
  kYCgCo = 8,

  /**
   Rec. ITU-R BT.2020 (non-constant luminance).
   */
  kBt2020NonconstantLuminance = 9,

  /**
   Rec. ITU-R BT.2020 (constant luminance).
   */
  kBt2020ConstantLuminance = 10,
};

/**
 A parsed \WebMID{Range} element.
 */
enum class Range : std::uint64_t {
  /**
   Unspecified.
   */
  kUnspecified = 0,

  /**
   Broadcast range.
   */
  kBroadcast = 1,

  /**
   Full range (no clipping).
   */
  kFull = 2,

  /**
   Defined by MatrixCoefficients/TransferCharacteristics.
   */
  kDerived = 3,
};

/**
 A parsed \WebMID{TransferCharacteristics} element.

 Matroska/WebM adopted these values from Table 3 of ISO/IEC 23001-8:2013/DCOR1.
 See that document for further details.
 */
enum class TransferCharacteristics : std::uint64_t {
  /**
   Rec. ITU-R BT.709-6.
   */
  kBt709 = 1,

  /**
   Image characteristics are unknown or are determined by the application.
   */
  kUnspecified = 2,

  /**
   Rec. ITU‑R BT.470‑6 System M (historical) with assumed display gamma 2.2.
   */
  kGamma22curve = 4,

  /**
   Rec. ITU‑R BT.470-6 System B, G (historical) with assumed display gamma 2.8.
   */
  kGamma28curve = 5,

  /**
   Society of Motion Picture and Television Engineers 170M (2004).
   */
  kSmpte170M = 6,

  /**
   Society of Motion Picture and Television Engineers 240M (1999).
   */
  kSmpte240M = 7,

  /**
   Linear transfer characteristics.
   */
  kLinear = 8,

  /**
   Logarithmic transfer characteristic (100:1 range).
   */
  kLog = 9,

  /**
   Logarithmic transfer characteristic (100 * Sqrt(10) : 1 range).
   */
  kLogSqrt = 10,

  /**
   IEC 61966-2-4.
   */
  kIec6196624 = 11,

  /**
   Rec. ITU‑R BT.1361-0 extended colour gamut system (historical).
   */
  kBt1361ExtendedColourGamut = 12,

  /**
   IEC 61966-2-1 sRGB or sYCC.
   */
  kIec6196621 = 13,

  /**
   Rec. ITU-R BT.2020-2 (10-bit system).
   */
  k10BitBt2020 = 14,

  /**
   Rec. ITU-R BT.2020-2 (12-bit system).
   */
  k12BitBt2020 = 15,

  /**
   Society of Motion Picture and Television Engineers ST 2084.
   */
  kSmpteSt2084 = 16,

  /**
   Society of Motion Picture and Television Engineers ST 428-1.
   */
  kSmpteSt4281 = 17,

  /**
   Association of Radio Industries and Businesses (ARIB) STD-B67.
   */
  kAribStdB67Hlg = 18,
};

/**
 A parsed \WebMID{Primaries} element.

 Matroska/WebM adopted these values from Table 2 of ISO/IEC 23001-8:2013/DCOR1.
 See that document for further details.
 */
enum class Primaries : std::uint64_t {
  /**
   Rec. ITU‑R BT.709-6.
   */
  kBt709 = 1,

  /**
   Image characteristics are unknown or are determined by the application.
   */
  kUnspecified = 2,

  /**
   Rec. ITU‑R BT.470‑6 System M (historical).
   */
  kBt470M = 4,

  /**
   Rec. ITU‑R BT.470‑6 System B, G (historical).
   */
  kBt470Bg = 5,

  /**
   Society of Motion Picture and Television Engineers 170M (2004).
   */
  kSmpte170M = 6,

  /**
   Society of Motion Picture and Television Engineers 240M (1999).
   */
  kSmpte240M = 7,

  /**
   Generic film.
   */
  kFilm = 8,

  /**
   Rec. ITU-R BT.2020-2.
   */
  kBt2020 = 9,

  /**
   Society of Motion Picture and Television Engineers ST 428-1.
   */
  kSmpteSt4281 = 10,

  /**
   Society of Motion Picture and Television Engineers RP 431-2 (a.k.a. DCI-P3).
   */
  kSmpteRp431 = 11,

  /**
   Society of Motion Picture and Television Engineers EG 432-1
   (a.k.a. DCI-P3 D65).
   */
  kSmpteEg432 = 12,

  /**
   JEDEC P22 phosphors/EBU Tech. 3213-E (1975).
   */
  kJedecP22Phosphors = 22,
};

/**
 A parsed \WebMID{Colour} element.
 */
struct Colour {
  /**
   A parsed \WebMID{MatrixCoefficients} element.
   */
  Element<MatrixCoefficients> matrix_coefficients{
      MatrixCoefficients::kUnspecified};

  /**
   A parsed \WebMID{BitsPerChannel} element.
   */
  Element<std::uint64_t> bits_per_channel{0};

  /**
   A parsed \WebMID{ChromaSubsamplingHorz} element.
   */
  Element<std::uint64_t> chroma_subsampling_x;

  /**
   A parsed \WebMID{ChromaSubsamplingVert} element.
   */
  Element<std::uint64_t> chroma_subsampling_y;

  /**
   A parsed \WebMID{CbSubsamplingHorz} element.
   */
  Element<std::uint64_t> cb_subsampling_x;

  /**
   A parsed \WebMID{CbSubsamplingVert} element.
   */
  Element<std::uint64_t> cb_subsampling_y;

  /**
   A parsed \WebMID{ChromaSitingHorz} element.
   */
  Element<std::uint64_t> chroma_siting_x{0};

  /**
   A parsed \WebMID{ChromaSitingVert} element.
   */
  Element<std::uint64_t> chroma_siting_y{0};

  /**
   A parsed \WebMID{Range} element.
   */
  Element<Range> range{Range::kUnspecified};

  /**
   A parsed \WebMID{TransferCharacteristics} element.
   */
  Element<TransferCharacteristics> transfer_characteristics{
      TransferCharacteristics::kUnspecified};

  /**
   A parsed \WebMID{Primaries} element.
   */
  Element<Primaries> primaries{Primaries::kUnspecified};

  /**
   A parsed \WebMID{MaxCLL} element.
   */
  Element<std::uint64_t> max_cll;

  /**
   A parsed \WebMID{MaxFALL} element.
   */
  Element<std::uint64_t> max_fall;

  /**
   A parsed \WebMID{MasteringMetadata} element.
   */
  Element<MasteringMetadata> mastering_metadata;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Colour& other) const {
    return matrix_coefficients == other.matrix_coefficients &&
           bits_per_channel == other.bits_per_channel &&
           chroma_subsampling_x == other.chroma_subsampling_x &&
           chroma_subsampling_y == other.chroma_subsampling_y &&
           cb_subsampling_x == other.cb_subsampling_x &&
           cb_subsampling_y == other.cb_subsampling_y &&
           chroma_siting_x == other.chroma_siting_x &&
           chroma_siting_y == other.chroma_siting_y && range == other.range &&
           transfer_characteristics == other.transfer_characteristics &&
           primaries == other.primaries && max_cll == other.max_cll &&
           max_fall == other.max_fall &&
           mastering_metadata == other.mastering_metadata;
  }
};

/**
 A parsed \WebMID{ProjectionType} element.
 */
enum class ProjectionType : std::uint64_t {
  /**
   Rectangular.
   */
  kRectangular = 0,

  /**
   Equirectangular.
   */
  kEquirectangular = 1,

  /**
   Cube map.
   */
  kCubeMap = 2,

  /**
   Mesh.
   */
  kMesh = 3,
};

/**
 A parsed \WebMID{Projection} element.
 */
struct Projection {
  /**
   A parsed \WebMID{ProjectionType} element.
   */
  Element<ProjectionType> type;

  /**
   A parsed \WebMID{ProjectionPrivate} element.
   */
  Element<std::vector<std::uint8_t>> projection_private;

  /**
   A parsed \WebMID{ProjectionPoseYaw} element.
   */
  Element<double> pose_yaw;

  /**
   A parsed \WebMID{ProjectionPosePitch} element.
   */
  Element<double> pose_pitch;

  /**
   A parsed \WebMID{ProjectionPoseRoll} element.
   */
  Element<double> pose_roll;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Projection& other) const {
    return type == other.type &&
           projection_private == other.projection_private &&
           pose_yaw == other.pose_yaw && pose_pitch == other.pose_pitch &&
           pose_roll == other.pose_roll;
  }
};

/**
 A parsed \WebMID{FlagInterlaced} element.
 */
enum class FlagInterlaced : std::uint64_t {
  /**
   Unspecified.
   */
  kUnspecified = 0,

  /**
   Interlaced.
   */
  kInterlaced = 1,

  /**
   Progressive.
   */
  kProgressive = 2,
};

/**
 A parsed \WebMID{StereoMode} element.
 */
enum class StereoMode : std::uint64_t {
  /**
   Mono.
   */
  kMono = 0,

  /**
   Side-by-side, left eye is first.
   */
  kSideBySideLeftFirst = 1,

  /**
   Top-bottom, right eye is first.
   */
  kTopBottomRightFirst = 2,

  /**
   Top-bottom, left eye is first.
   */
  kTopBottomLeftFirst = 3,

  /**
   Checkboard, right eye is first.
   */
  kCheckboardRightFirst = 4,

  /**
   Checkboard, left eye is first.
   */
  kCheckboardLeftFirst = 5,

  /**
   Row interleaved, right eye is first.
   */
  kRowInterleavedRightFirst = 6,

  /**
   Row interleaved, left eye is first.
   */
  kRowInterleavedLeftFirst = 7,

  /**
   Column interleaved, right eye is first.
   */
  kColumnInterleavedRightFirst = 8,

  /**
   Column interleaved, left eye is first.
   */
  kColumnInterleavedLeftFirst = 9,

  /**
   Anaglyph (cyan/read).
   */
  kAnaglyphCyanRed = 10,

  /**
   Side-by-side, right eye is first.
   */
  kSideBySideRightFirst = 11,

  /**
   Anaglyph (green/magenta).
   */
  kAnaglyphGreenMagenta = 12,

  /**
   Both eyes are laced in one block, left eye is first.
   */
  kBlockLacedLeftFirst = 13,

  /**
   Both eyes are laced in one block, right eye is first.
   */
  kBlockLacedRightFirst = 14,

  /**
   Stereo, but the layout for the left and right eyes is application dependent
   and must be determined from other data (like the ProjectionPrivate element).
   */
  kStereoCustom = 15,
};

/**
 A parsed \WebMID{DisplayUnit} element.

 Note that WebM only supports pixel display units. Centimeters, inches, and
 display aspect ratio aren't supported.
 */
enum class DisplayUnit : std::uint64_t {
  // The only value legal value in WebM is 0 (pixels).
  /**
   Pixels.

   This is the only option supported by WebM.
   */
  kPixels = 0,

  /**
   Centimeters.
   */
  kCentimeters = 1,

  /**
   Inches.
   */
  kInches = 2,

  /**
   Display aspect ratio.
   */
  kDisplayAspectRatio = 3,
};

/**
 A parsed \WebMID{AspectRatioType} element.
 */
enum class AspectRatioType : std::uint64_t {
  /**
   Free resizing.
   */
  kFreeResizing = 0,

  /**
   Keep aspect ratio.
   */
  kKeep = 1,

  /**
   Fixed aspect ratio.
   */
  kFixed = 2,
};

/**
 A parsed \WebMID{Video} element.
 */
struct Video {
  /**
   A parsed \WebMID{FlagInterlaced} element.
   */
  Element<FlagInterlaced> interlaced{FlagInterlaced::kUnspecified};

  /**
   A parsed \WebMID{StereoMode} element.
   */
  Element<StereoMode> stereo_mode{StereoMode::kMono};

  /**
   A parsed \WebMID{AlphaMode} element.
   */
  Element<std::uint64_t> alpha_mode{0};

  /**
   A parsed \WebMID{PixelWidth} element.
   */
  Element<std::uint64_t> pixel_width;

  /**
   A parsed \WebMID{PixelHeight} element.
   */
  Element<std::uint64_t> pixel_height;

  /**
   A parsed \WebMID{PixelCropBottom} element.
   */
  Element<std::uint64_t> pixel_crop_bottom{0};

  /**
   A parsed \WebMID{PixelCropTop} element.
   */
  Element<std::uint64_t> pixel_crop_top{0};

  /**
   A parsed \WebMID{PixelCropLeft} element.
   */
  Element<std::uint64_t> pixel_crop_left{0};

  /**
   A parsed \WebMID{PixelCropRight} element.
   */
  Element<std::uint64_t> pixel_crop_right{0};

  /**
   A parsed \WebMID{DisplayWidth} element.
   */
  Element<std::uint64_t> display_width;

  /**
   A parsed \WebMID{DisplayHeight} element.
   */
  Element<std::uint64_t> display_height;

  /**
   A parsed \WebMID{DisplayUnit} element.
   */
  Element<DisplayUnit> display_unit{DisplayUnit::kPixels};

  /**
   A parsed \WebMID{AspectRatioType} element.
   */
  Element<AspectRatioType> aspect_ratio_type{AspectRatioType::kFreeResizing};

  /**
   A parsed \WebMID{FrameRate} element (deprecated).
   */
  Element<double> frame_rate;

  /**
   A parsed \WebMID{Colour} element.
   */
  Element<Colour> colour;

  /**
   A parsed \WebMID{Projection} element.
   */
  Element<Projection> projection;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Video& other) const {
    return interlaced == other.interlaced && stereo_mode == other.stereo_mode &&
           alpha_mode == other.alpha_mode && pixel_width == other.pixel_width &&
           pixel_height == other.pixel_height &&
           pixel_crop_bottom == other.pixel_crop_bottom &&
           pixel_crop_top == other.pixel_crop_top &&
           pixel_crop_left == other.pixel_crop_left &&
           pixel_crop_right == other.pixel_crop_right &&
           display_width == other.display_width &&
           display_height == other.display_height &&
           display_unit == other.display_unit &&
           aspect_ratio_type == other.aspect_ratio_type &&
           frame_rate == other.frame_rate && colour == other.colour &&
           projection == other.projection;
  }
};

/**
 A parsed \WebMID{AESSettingsCipherMode} element.
 */
enum class AesSettingsCipherMode : std::uint64_t {
  /**
   Counter (CTR).
   */
  kCtr = 1,
};

/**
 A parsed \WebMID{ContentEncAESSettings} element.
 */
struct ContentEncAesSettings {
  /**
   A parsed \WebMID{AESSettingsCipherMode} element.
   */
  Element<AesSettingsCipherMode> aes_settings_cipher_mode{
      AesSettingsCipherMode::kCtr};

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ContentEncAesSettings& other) const {
    return aes_settings_cipher_mode == other.aes_settings_cipher_mode;
  }
};

/**
 A parsed \WebMID{ContentEncAlgo} element.
 */
enum class ContentEncAlgo : std::uint64_t {
  /**
   The contents have been signed but not encrypted.
   */
  kOnlySigned = 0,

  /**
   DES encryption.
   */
  kDes = 1,

  /**
   3DES encryption.
   */
  k3Des = 2,

  /**
   Twofish encryption.
   */
  kTwofish = 3,

  /**
   Blowfish encryption.
   */
  kBlowfish = 4,

  /**
   AES encryption.
   */
  kAes = 5,
};

/**
 A parsed \WebMID{ContentEncryption} element.
 */
struct ContentEncryption {
  /**
   A parsed \WebMID{ContentEncAlgo} element.
   */
  Element<ContentEncAlgo> algorithm{ContentEncAlgo::kOnlySigned};

  /**
   A parsed \WebMID{ContentEncKeyID} element.
   */
  Element<std::vector<std::uint8_t>> key_id;

  /**
   A parsed \WebMID{ContentEncAESSettings} element.
   */
  Element<ContentEncAesSettings> aes_settings;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ContentEncryption& other) const {
    return algorithm == other.algorithm && key_id == other.key_id &&
           aes_settings == other.aes_settings;
  }
};

/**
 A parsed \WebMID{ContentEncodingType} element.
 */
enum class ContentEncodingType : std::uint64_t {
  /**
   Compression.
   */
  kCompression = 0,

  /**
   Encryption.
   */
  kEncryption = 1,
};

/**
 A parsed \WebMID{ContentEncoding} element.
 */
struct ContentEncoding {
  /**
   A parsed \WebMID{ContentEncodingOrder} element.
   */
  Element<std::uint64_t> order{0};

  /**
   A parsed \WebMID{ContentEncodingScope} element.
   */
  Element<std::uint64_t> scope{1};

  /**
   A parsed \WebMID{ContentEncodingType} element.
   */
  Element<ContentEncodingType> type{ContentEncodingType::kCompression};

  /**
   A parsed \WebMID{ContentEncryption} element.
   */
  Element<ContentEncryption> encryption;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ContentEncoding& other) const {
    return order == other.order && scope == other.scope && type == other.type &&
           encryption == other.encryption;
  }
};

/**
 A parsed \WebMID{ContentEncodings} element.
 */
struct ContentEncodings {
  /**
   Parsed \WebMID{ContentEncoding} elements.
   */
  std::vector<Element<ContentEncoding>> encodings;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ContentEncodings& other) const {
    return encodings == other.encodings;
  }
};

/**
 A parsed \WebMID{TrackType} element.
 */
enum class TrackType : std::uint64_t {
  /**
   Video.
   */
  kVideo = 0x01,

  /**
   Audio.
   */
  kAudio = 0x02,

  /**
   Complex.
   */
  kComplex = 0x03,

  /**
   Logo.
   */
  kLogo = 0x10,

  /**
   Subtitle.
   */
  kSubtitle = 0x11,

  /**
   Buttons.
   */
  kButtons = 0x12,

  /**
   Control.
   */
  kControl = 0x20,
};

/**
 A parsed \WebMID{TrackEntry} element.
 */
struct TrackEntry {
  /**
   A parsed \WebMID{TrackNumber} element.
   */
  Element<std::uint64_t> track_number;

  /**
   A parsed \WebMID{TrackUID} element.
   */
  Element<std::uint64_t> track_uid;

  /**
   A parsed \WebMID{TrackType} element.
   */
  Element<TrackType> track_type;

  /**
   A parsed \WebMID{FlagEnabled} element.
   */
  Element<bool> is_enabled{true};

  /**
   A parsed \WebMID{FlagDefault} element.
   */
  Element<bool> is_default{true};

  /**
   A parsed \WebMID{FlagForced} element.
   */
  Element<bool> is_forced{false};

  /**
   A parsed \WebMID{FlagLacing} element.
   */
  Element<bool> uses_lacing{true};

  /**
   A parsed \WebMID{DefaultDuration} element.
   */
  Element<std::uint64_t> default_duration;

  /**
   A parsed \WebMID{Name} element.
   */
  Element<std::string> name;

  /**
   A parsed \WebMID{Language} element.
   */
  Element<std::string> language{"eng"};

  /**
   A parsed \WebMID{CodecID} element.
   */
  Element<std::string> codec_id;

  /**
   A parsed \WebMID{CodecPrivate} element.
   */
  Element<std::vector<std::uint8_t>> codec_private;

  /**
   A parsed \WebMID{CodecName} element.
   */
  Element<std::string> codec_name;

  /**
   A parsed \WebMID{CodecDelay} element.
   */
  Element<std::uint64_t> codec_delay{0};

  /**
   A parsed \WebMID{SeekPreRoll} element.
   */
  Element<std::uint64_t> seek_pre_roll{0};

  /**
   A parsed \WebMID{Video} element.
   */
  Element<Video> video;

  /**
   A parsed \WebMID{Audio} element.
   */
  Element<Audio> audio;

  /**
   A parsed \WebMID{ContentEncodings} element.
   */
  Element<ContentEncodings> content_encodings;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const TrackEntry& other) const {
    return track_number == other.track_number && track_uid == other.track_uid &&
           track_type == other.track_type && is_enabled == other.is_enabled &&
           is_default == other.is_default && is_forced == other.is_forced &&
           uses_lacing == other.uses_lacing &&
           default_duration == other.default_duration && name == other.name &&
           language == other.language && codec_id == other.codec_id &&
           codec_private == other.codec_private &&
           codec_name == other.codec_name && codec_delay == other.codec_delay &&
           seek_pre_roll == other.seek_pre_roll && video == other.video &&
           audio == other.audio && content_encodings == other.content_encodings;
  }
};

/**
 A parsed \WebMID{CueTrackPositions} element.
 */
struct CueTrackPositions {
  /**
   A parsed \WebMID{CueTrack} element.
   */
  Element<std::uint64_t> track;

  /**
   A parsed \WebMID{CueClusterPosition} element.
   */
  Element<std::uint64_t> cluster_position;

  /**
   A parsed \WebMID{CueRelativePosition} element.
   */
  Element<std::uint64_t> relative_position;

  /**
   A parsed \WebMID{CueDuration} element.
   */
  Element<std::uint64_t> duration;

  /**
   A parsed \WebMID{CueBlockNumber} element.
   */
  Element<std::uint64_t> block_number{1};

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const CueTrackPositions& other) const {
    return track == other.track && cluster_position == other.cluster_position &&
           relative_position == other.relative_position &&
           duration == other.duration && block_number == other.block_number;
  }
};

/**
 A parsed \WebMID{CuePoint} element.
 */
struct CuePoint {
  /**
   A parsed \WebMID{CueTime} element.
   */
  Element<std::uint64_t> time;

  /**
   Parsed \WebMID{CueTrackPositions} elements.
   */
  std::vector<Element<CueTrackPositions>> cue_track_positions;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const CuePoint& other) const {
    return time == other.time &&
           cue_track_positions == other.cue_track_positions;
  }
};

/**
 A parsed \WebMID{ChapterDisplay} element.
 */
struct ChapterDisplay {
  /**
   A parsed \WebMID{ChapString} element.
   */
  Element<std::string> string;

  /**
   Parsed \WebMID{ChapLanguage} elements.
   */
  std::vector<Element<std::string>> languages{Element<std::string>{"eng"}};

  /**
   Parsed \WebMID{ChapCountry} elements.
   */
  std::vector<Element<std::string>> countries;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ChapterDisplay& other) const {
    return string == other.string && languages == other.languages &&
           countries == other.countries;
  }
};

/**
 A parsed \WebMID{ChapterAtom} element.
 */
struct ChapterAtom {
  /**
   A parsed \WebMID{ChapterUID} element.
   */
  Element<std::uint64_t> uid;

  /**
   A parsed \WebMID{ChapterStringUID} element.
   */
  Element<std::string> string_uid;

  /**
   A parsed \WebMID{ChapterTimeStart} element.
   */
  Element<std::uint64_t> time_start;

  /**
   A parsed \WebMID{ChapterTimeEnd} element.
   */
  Element<std::uint64_t> time_end;

  /**
   Parsed \WebMID{ChapterDisplay} elements.
   */
  std::vector<Element<ChapterDisplay>> displays;

  /**
   Parsed \WebMID{ChapterAtom} elements.
   */
  std::vector<Element<ChapterAtom>> atoms;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const ChapterAtom& other) const {
    return uid == other.uid && string_uid == other.string_uid &&
           time_start == other.time_start && time_end == other.time_end &&
           displays == other.displays && atoms == other.atoms;
  }
};

/**
 A parsed \WebMID{EditionEntry} element.
 */
struct EditionEntry {
  /**
   Parsed \WebMID{ChapterAtom} elements.
   */
  std::vector<Element<ChapterAtom>> atoms;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const EditionEntry& other) const {
    return atoms == other.atoms;
  }
};

/**
 A parsed \WebMID{SimpleTag} element.
 */
struct SimpleTag {
  /**
   A parsed \WebMID{TagName} element.
   */
  Element<std::string> name;

  /**
   A parsed \WebMID{TagLanguage} element.
   */
  Element<std::string> language{"und"};

  /**
   A parsed \WebMID{TagDefault} element.
   */
  Element<bool> is_default{true};

  /**
   A parsed \WebMID{TagString} element.
   */
  Element<std::string> string;

  /**
   A parsed \WebMID{TagBinary} element.
   */
  Element<std::vector<std::uint8_t>> binary;

  /**
   Parsed \WebMID{SimpleTag} elements.
   */
  std::vector<Element<SimpleTag>> tags;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const SimpleTag& other) const {
    return name == other.name && language == other.language &&
           is_default == other.is_default && string == other.string &&
           binary == other.binary && tags == other.tags;
  }
};

/**
 A parsed \WebMID{Targets} element.
 */
struct Targets {
  /**
   A parsed \WebMID{TargetTypeValue} element.
   */
  Element<std::uint64_t> type_value{50};

  /**
   A parsed \WebMID{TargetType} element.
   */
  Element<std::string> type;

  /**
   Parsed \WebMID{TagTrackUID} elements.
   */
  std::vector<Element<std::uint64_t>> track_uids;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Targets& other) const {
    return type_value == other.type_value && type == other.type &&
           track_uids == other.track_uids;
  }
};

/**
 A parsed \WebMID{Tag} element.
 */
struct Tag {
  /**
   A parsed \WebMID{Targets} element.
   */
  Element<Targets> targets;

  /**
   Parsed \WebMID{SimpleTag} elements.
   */
  std::vector<Element<SimpleTag>> tags;

  /**
   Returns true if every member within the two objects are equal.
   */
  bool operator==(const Tag& other) const {
    return targets == other.targets && tags == other.tags;
  }
};

/**
 @}
 */

}  // namespace webm

#endif  // INCLUDE_WEBM_DOM_TYPES_H_
