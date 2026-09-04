// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#ifndef MKVMUXER_MKVMUXER_H_
#define MKVMUXER_MKVMUXER_H_

#include <stdint.h>

#include <cstddef>
#include <list>
#include <map>

#include "common/webmids.h"
#include "mkvmuxer/mkvmuxertypes.h"

// For a description of the WebM elements see
// http://www.webmproject.org/code/specs/container/.

namespace mkvparser {
class IMkvReader;
}  // namespace mkvparser

namespace mkvmuxer {

class MkvWriter;
class Segment;

const uint64_t kMaxTrackNumber = 126;

///////////////////////////////////////////////////////////////
// Interface used by the mkvmuxer to write out the Mkv data.
class IMkvWriter {
 public:
  // Writes out |len| bytes of |buf|. Returns 0 on success.
  virtual int32 Write(const void* buf, uint32 len) = 0;

  // Returns the offset of the output position from the beginning of the
  // output.
  virtual int64 Position() const = 0;

  // Set the current File position. Returns 0 on success.
  virtual int32 Position(int64 position) = 0;

  // Returns true if the writer is seekable.
  virtual bool Seekable() const = 0;

  // Element start notification. Called whenever an element identifier is about
  // to be written to the stream. |element_id| is the element identifier, and
  // |position| is the location in the WebM stream where the first octet of the
  // element identifier will be written.
  // Note: the |MkvId| enumeration in webmids.hpp defines element values.
  virtual void ElementStartNotify(uint64 element_id, int64 position) = 0;

 protected:
  IMkvWriter();
  virtual ~IMkvWriter();

 private:
  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(IMkvWriter);
};

// Writes out the EBML header for a WebM file, but allows caller to specify
// DocType. This function must be called before any other libwebm writing
// functions are called.
bool WriteEbmlHeader(IMkvWriter* writer, uint64_t doc_type_version,
                     const char* const doc_type);

// Writes out the EBML header for a WebM file. This function must be called
// before any other libwebm writing functions are called.
bool WriteEbmlHeader(IMkvWriter* writer, uint64_t doc_type_version);

// Deprecated. Writes out EBML header with doc_type_version as
// kDefaultDocTypeVersion. Exists for backward compatibility.
bool WriteEbmlHeader(IMkvWriter* writer);

// Copies in Chunk from source to destination between the given byte positions
bool ChunkedCopy(mkvparser::IMkvReader* source, IMkvWriter* dst, int64_t start,
                 int64_t size);

///////////////////////////////////////////////////////////////
// Class to hold data the will be written to a block.
class Frame {
 public:
  Frame();
  ~Frame();

  // Sets this frame's contents based on |frame|. Returns true on success. On
  // failure, this frame's existing contents may be lost.
  bool CopyFrom(const Frame& frame);

  // Copies |frame| data into |frame_|. Returns true on success.
  bool Init(const uint8_t* frame, uint64_t length);

  // Copies |additional| data into |additional_|. Returns true on success.
  bool AddAdditionalData(const uint8_t* additional, uint64_t length,
                         uint64_t add_id);

  // Returns true if the frame has valid parameters.
  bool IsValid() const;

  // Returns true if the frame can be written as a SimpleBlock based on current
  // parameters.
  bool CanBeSimpleBlock() const;

  uint64_t add_id() const { return add_id_; }
  const uint8_t* additional() const { return additional_; }
  uint64_t additional_length() const { return additional_length_; }
  void set_duration(uint64_t duration);
  uint64_t duration() const { return duration_; }
  bool duration_set() const { return duration_set_; }
  const uint8_t* frame() const { return frame_; }
  void set_is_key(bool key) { is_key_ = key; }
  bool is_key() const { return is_key_; }
  uint64_t length() const { return length_; }
  void set_track_number(uint64_t track_number) { track_number_ = track_number; }
  uint64_t track_number() const { return track_number_; }
  void set_timestamp(uint64_t timestamp) { timestamp_ = timestamp; }
  uint64_t timestamp() const { return timestamp_; }
  void set_discard_padding(int64_t discard_padding) {
    discard_padding_ = discard_padding;
  }
  int64_t discard_padding() const { return discard_padding_; }
  void set_reference_block_timestamp(int64_t reference_block_timestamp);
  int64_t reference_block_timestamp() const {
    return reference_block_timestamp_;
  }
  bool reference_block_timestamp_set() const {
    return reference_block_timestamp_set_;
  }

 private:
  // Id of the Additional data.
  uint64_t add_id_;

  // Pointer to additional data. Owned by this class.
  uint8_t* additional_;

  // Length of the additional data.
  uint64_t additional_length_;

  // Duration of the frame in nanoseconds.
  uint64_t duration_;

  // Flag indicating that |duration_| has been set. Setting duration causes the
  // frame to be written out as a Block with BlockDuration instead of as a
  // SimpleBlock.
  bool duration_set_;

  // Pointer to the data. Owned by this class.
  uint8_t* frame_;

  // Flag telling if the data should set the key flag of a block.
  bool is_key_;

  // Length of the data.
  uint64_t length_;

  // Mkv track number the data is associated with.
  uint64_t track_number_;

  // Timestamp of the data in nanoseconds.
  uint64_t timestamp_;

  // Discard padding for the frame.
  int64_t discard_padding_;

  // Reference block timestamp.
  int64_t reference_block_timestamp_;

  // Flag indicating if |reference_block_timestamp_| has been set.
  bool reference_block_timestamp_set_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Frame);
};

///////////////////////////////////////////////////////////////
// Class to hold one cue point in a Cues element.
class CuePoint {
 public:
  CuePoint();
  ~CuePoint();

  // Returns the size in bytes for the entire CuePoint element.
  uint64_t Size() const;

  // Output the CuePoint element to the writer. Returns true on success.
  bool Write(IMkvWriter* writer) const;

  void set_time(uint64_t time) { time_ = time; }
  uint64_t time() const { return time_; }
  void set_track(uint64_t track) { track_ = track; }
  uint64_t track() const { return track_; }
  void set_cluster_pos(uint64_t cluster_pos) { cluster_pos_ = cluster_pos; }
  uint64_t cluster_pos() const { return cluster_pos_; }
  void set_block_number(uint64_t block_number) { block_number_ = block_number; }
  uint64_t block_number() const { return block_number_; }
  void set_output_block_number(bool output_block_number) {
    output_block_number_ = output_block_number;
  }
  bool output_block_number() const { return output_block_number_; }

 private:
  // Returns the size in bytes for the payload of the CuePoint element.
  uint64_t PayloadSize() const;

  // Absolute timecode according to the segment time base.
  uint64_t time_;

  // The Track element associated with the CuePoint.
  uint64_t track_;

  // The position of the Cluster containing the Block.
  uint64_t cluster_pos_;

  // Number of the Block within the Cluster, starting from 1.
  uint64_t block_number_;

  // If true the muxer will write out the block number for the cue if the
  // block number is different than the default of 1. Default is set to true.
  bool output_block_number_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(CuePoint);
};

///////////////////////////////////////////////////////////////
// Cues element.
class Cues {
 public:
  Cues();
  ~Cues();

  // Adds a cue point to the Cues element. Returns true on success.
  bool AddCue(CuePoint* cue);

  // Returns the cue point by index. Returns NULL if there is no cue point
  // match.
  CuePoint* GetCueByIndex(int32_t index) const;

  // Returns the total size of the Cues element
  uint64_t Size();

  // Output the Cues element to the writer. Returns true on success.
  bool Write(IMkvWriter* writer) const;

  int32_t cue_entries_size() const { return cue_entries_size_; }
  void set_output_block_number(bool output_block_number) {
    output_block_number_ = output_block_number;
  }
  bool output_block_number() const { return output_block_number_; }

 private:
  // Number of allocated elements in |cue_entries_|.
  int32_t cue_entries_capacity_;

  // Number of CuePoints in |cue_entries_|.
  int32_t cue_entries_size_;

  // CuePoint list.
  CuePoint** cue_entries_;

  // If true the muxer will write out the block number for the cue if the
  // block number is different than the default of 1. Default is set to true.
  bool output_block_number_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Cues);
};

///////////////////////////////////////////////////////////////
// ContentEncAESSettings element
class ContentEncAESSettings {
 public:
  enum { kCTR = 1 };

  ContentEncAESSettings();
  ~ContentEncAESSettings() {}

  // Returns the size in bytes for the ContentEncAESSettings element.
  uint64_t Size() const;

  // Writes out the ContentEncAESSettings element to |writer|. Returns true on
  // success.
  bool Write(IMkvWriter* writer) const;

  uint64_t cipher_mode() const { return cipher_mode_; }

 private:
  // Returns the size in bytes for the payload of the ContentEncAESSettings
  // element.
  uint64_t PayloadSize() const;

  // Sub elements
  uint64_t cipher_mode_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(ContentEncAESSettings);
};

///////////////////////////////////////////////////////////////
// ContentEncoding element
// Elements used to describe if the track data has been encrypted or
// compressed with zlib or header stripping.
// Currently only whole frames can be encrypted with AES. This dictates that
// ContentEncodingOrder will be 0, ContentEncodingScope will be 1,
// ContentEncodingType will be 1, and ContentEncAlgo will be 5.
class ContentEncoding {
 public:
  ContentEncoding();
  ~ContentEncoding();

  // Sets the content encryption id. Copies |length| bytes from |id| to
  // |enc_key_id_|. Returns true on success.
  bool SetEncryptionID(const uint8_t* id, uint64_t length);

  // Returns the size in bytes for the ContentEncoding element.
  uint64_t Size() const;

  // Writes out the ContentEncoding element to |writer|. Returns true on
  // success.
  bool Write(IMkvWriter* writer) const;

  uint64_t enc_algo() const { return enc_algo_; }
  uint64_t encoding_order() const { return encoding_order_; }
  uint64_t encoding_scope() const { return encoding_scope_; }
  uint64_t encoding_type() const { return encoding_type_; }
  ContentEncAESSettings* enc_aes_settings() { return &enc_aes_settings_; }

 private:
  // Returns the size in bytes for the encoding elements.
  uint64_t EncodingSize(uint64_t compression_size,
                        uint64_t encryption_size) const;

  // Returns the size in bytes for the encryption elements.
  uint64_t EncryptionSize() const;

  // Track element names
  uint64_t enc_algo_;
  uint8_t* enc_key_id_;
  uint64_t encoding_order_;
  uint64_t encoding_scope_;
  uint64_t encoding_type_;

  // ContentEncAESSettings element.
  ContentEncAESSettings enc_aes_settings_;

  // Size of the ContentEncKeyID data in bytes.
  uint64_t enc_key_id_length_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(ContentEncoding);
};

///////////////////////////////////////////////////////////////
// Colour element.
class PrimaryChromaticity {
 public:
  static const float kChromaticityMin;
  static const float kChromaticityMax;

  PrimaryChromaticity(float x_val, float y_val) : x_(x_val), y_(y_val) {}
  PrimaryChromaticity() : x_(0), y_(0) {}
  ~PrimaryChromaticity() {}

  // Returns sum of |x_id| and |y_id| element id sizes and payload sizes.
  uint64_t PrimaryChromaticitySize(libwebm::MkvId x_id,
                                   libwebm::MkvId y_id) const;
  bool Valid() const;
  bool Write(IMkvWriter* writer, libwebm::MkvId x_id,
             libwebm::MkvId y_id) const;

  float x() const { return x_; }
  void set_x(float new_x) { x_ = new_x; }
  float y() const { return y_; }
  void set_y(float new_y) { y_ = new_y; }

 private:
  float x_;
  float y_;
};

class MasteringMetadata {
 public:
  static const float kValueNotPresent;
  static const float kMinLuminance;
  static const float kMinLuminanceMax;
  static const float kMaxLuminanceMax;

  MasteringMetadata()
      : luminance_max_(kValueNotPresent),
        luminance_min_(kValueNotPresent),
        r_(NULL),
        g_(NULL),
        b_(NULL),
        white_point_(NULL) {}
  ~MasteringMetadata() {
    delete r_;
    delete g_;
    delete b_;
    delete white_point_;
  }

  // Returns total size of the MasteringMetadata element.
  uint64_t MasteringMetadataSize() const;
  bool Valid() const;
  bool Write(IMkvWriter* writer) const;

  // Copies non-null chromaticity.
  bool SetChromaticity(const PrimaryChromaticity* r,
                       const PrimaryChromaticity* g,
                       const PrimaryChromaticity* b,
                       const PrimaryChromaticity* white_point);
  const PrimaryChromaticity* r() const { return r_; }
  const PrimaryChromaticity* g() const { return g_; }
  const PrimaryChromaticity* b() const { return b_; }
  const PrimaryChromaticity* white_point() const { return white_point_; }

  float luminance_max() const { return luminance_max_; }
  void set_luminance_max(float luminance_max) {
    luminance_max_ = luminance_max;
  }
  float luminance_min() const { return luminance_min_; }
  void set_luminance_min(float luminance_min) {
    luminance_min_ = luminance_min;
  }

 private:
  // Returns size of MasteringMetadata child elements.
  uint64_t PayloadSize() const;

  float luminance_max_;
  float luminance_min_;
  PrimaryChromaticity* r_;
  PrimaryChromaticity* g_;
  PrimaryChromaticity* b_;
  PrimaryChromaticity* white_point_;
};

class Colour {
 public:
  enum MatrixCoefficients {
    kGbr = 0,
    kBt709 = 1,
    kUnspecifiedMc = 2,
    kReserved = 3,
    kFcc = 4,
    kBt470bg = 5,
    kSmpte170MMc = 6,
    kSmpte240MMc = 7,
    kYcocg = 8,
    kBt2020NonConstantLuminance = 9,
    kBt2020ConstantLuminance = 10,
  };
  enum ChromaSitingHorz {
    kUnspecifiedCsh = 0,
    kLeftCollocated = 1,
    kHalfCsh = 2,
  };
  enum ChromaSitingVert {
    kUnspecifiedCsv = 0,
    kTopCollocated = 1,
    kHalfCsv = 2,
  };
  enum Range {
    kUnspecifiedCr = 0,
    kBroadcastRange = 1,
    kFullRange = 2,
    kMcTcDefined = 3,  // Defined by MatrixCoefficients/TransferCharacteristics.
  };
  enum TransferCharacteristics {
    kIturBt709Tc = 1,
    kUnspecifiedTc = 2,
    kReservedTc = 3,
    kGamma22Curve = 4,
    kGamma28Curve = 5,
    kSmpte170MTc = 6,
    kSmpte240MTc = 7,
    kLinear = 8,
    kLog = 9,
    kLogSqrt = 10,
    kIec6196624 = 11,
    kIturBt1361ExtendedColourGamut = 12,
    kIec6196621 = 13,
    kIturBt202010bit = 14,
    kIturBt202012bit = 15,
    kSmpteSt2084 = 16,
    kSmpteSt4281Tc = 17,
    kAribStdB67Hlg = 18,
  };
  enum Primaries {
    kReservedP0 = 0,
    kIturBt709P = 1,
    kUnspecifiedP = 2,
    kReservedP3 = 3,
    kIturBt470M = 4,
    kIturBt470Bg = 5,
    kSmpte170MP = 6,
    kSmpte240MP = 7,
    kFilm = 8,
    kIturBt2020 = 9,
    kSmpteSt4281P = 10,
    kJedecP22Phosphors = 22,
  };
  static const uint64_t kValueNotPresent;
  Colour()
      : matrix_coefficients_(kValueNotPresent),
        bits_per_channel_(kValueNotPresent),
        chroma_subsampling_horz_(kValueNotPresent),
        chroma_subsampling_vert_(kValueNotPresent),
        cb_subsampling_horz_(kValueNotPresent),
        cb_subsampling_vert_(kValueNotPresent),
        chroma_siting_horz_(kValueNotPresent),
        chroma_siting_vert_(kValueNotPresent),
        range_(kValueNotPresent),
        transfer_characteristics_(kValueNotPresent),
        primaries_(kValueNotPresent),
        max_cll_(kValueNotPresent),
        max_fall_(kValueNotPresent),
        mastering_metadata_(NULL) {}
  ~Colour() { delete mastering_metadata_; }

  // Returns total size of the Colour element.
  uint64_t ColourSize() const;
  bool Valid() const;
  bool Write(IMkvWriter* writer) const;

  // Deep copies |mastering_metadata|.
  bool SetMasteringMetadata(const MasteringMetadata& mastering_metadata);

  const MasteringMetadata* mastering_metadata() const {
    return mastering_metadata_;
  }

  uint64_t matrix_coefficients() const { return matrix_coefficients_; }
  void set_matrix_coefficients(uint64_t matrix_coefficients) {
    matrix_coefficients_ = matrix_coefficients;
  }
  uint64_t bits_per_channel() const { return bits_per_channel_; }
  void set_bits_per_channel(uint64_t bits_per_channel) {
    bits_per_channel_ = bits_per_channel;
  }
  uint64_t chroma_subsampling_horz() const { return chroma_subsampling_horz_; }
  void set_chroma_subsampling_horz(uint64_t chroma_subsampling_horz) {
    chroma_subsampling_horz_ = chroma_subsampling_horz;
  }
  uint64_t chroma_subsampling_vert() const { return chroma_subsampling_vert_; }
  void set_chroma_subsampling_vert(uint64_t chroma_subsampling_vert) {
    chroma_subsampling_vert_ = chroma_subsampling_vert;
  }
  uint64_t cb_subsampling_horz() const { return cb_subsampling_horz_; }
  void set_cb_subsampling_horz(uint64_t cb_subsampling_horz) {
    cb_subsampling_horz_ = cb_subsampling_horz;
  }
  uint64_t cb_subsampling_vert() const { return cb_subsampling_vert_; }
  void set_cb_subsampling_vert(uint64_t cb_subsampling_vert) {
    cb_subsampling_vert_ = cb_subsampling_vert;
  }
  uint64_t chroma_siting_horz() const { return chroma_siting_horz_; }
  void set_chroma_siting_horz(uint64_t chroma_siting_horz) {
    chroma_siting_horz_ = chroma_siting_horz;
  }
  uint64_t chroma_siting_vert() const { return chroma_siting_vert_; }
  void set_chroma_siting_vert(uint64_t chroma_siting_vert) {
    chroma_siting_vert_ = chroma_siting_vert;
  }
  uint64_t range() const { return range_; }
  void set_range(uint64_t range) { range_ = range; }
  uint64_t transfer_characteristics() const {
    return transfer_characteristics_;
  }
  void set_transfer_characteristics(uint64_t transfer_characteristics) {
    transfer_characteristics_ = transfer_characteristics;
  }
  uint64_t primaries() const { return primaries_; }
  void set_primaries(uint64_t primaries) { primaries_ = primaries; }
  uint64_t max_cll() const { return max_cll_; }
  void set_max_cll(uint64_t max_cll) { max_cll_ = max_cll; }
  uint64_t max_fall() const { return max_fall_; }
  void set_max_fall(uint64_t max_fall) { max_fall_ = max_fall; }

 private:
  // Returns size of Colour child elements.
  uint64_t PayloadSize() const;

  uint64_t matrix_coefficients_;
  uint64_t bits_per_channel_;
  uint64_t chroma_subsampling_horz_;
  uint64_t chroma_subsampling_vert_;
  uint64_t cb_subsampling_horz_;
  uint64_t cb_subsampling_vert_;
  uint64_t chroma_siting_horz_;
  uint64_t chroma_siting_vert_;
  uint64_t range_;
  uint64_t transfer_characteristics_;
  uint64_t primaries_;
  uint64_t max_cll_;
  uint64_t max_fall_;

  MasteringMetadata* mastering_metadata_;
};

///////////////////////////////////////////////////////////////
// Projection element.
class Projection {
 public:
  enum ProjectionType {
    kTypeNotPresent = -1,
    kRectangular = 0,
    kEquirectangular = 1,
    kCubeMap = 2,
    kMesh = 3,
  };
  static const uint64_t kValueNotPresent;
  Projection()
      : type_(kRectangular),
        pose_yaw_(0.0),
        pose_pitch_(0.0),
        pose_roll_(0.0),
        private_data_(NULL),
        private_data_length_(0) {}
  ~Projection() { delete[] private_data_; }

  uint64_t ProjectionSize() const;
  bool Write(IMkvWriter* writer) const;

  bool SetProjectionPrivate(const uint8_t* private_data,
                            uint64_t private_data_length);

  ProjectionType type() const { return type_; }
  void set_type(ProjectionType type) { type_ = type; }
  float pose_yaw() const { return pose_yaw_; }
  void set_pose_yaw(float pose_yaw) { pose_yaw_ = pose_yaw; }
  float pose_pitch() const { return pose_pitch_; }
  void set_pose_pitch(float pose_pitch) { pose_pitch_ = pose_pitch; }
  float pose_roll() const { return pose_roll_; }
  void set_pose_roll(float pose_roll) { pose_roll_ = pose_roll; }
  uint8_t* private_data() const { return private_data_; }
  uint64_t private_data_length() const { return private_data_length_; }

 private:
  // Returns size of VideoProjection child elements.
  uint64_t PayloadSize() const;

  ProjectionType type_;
  float pose_yaw_;
  float pose_pitch_;
  float pose_roll_;
  uint8_t* private_data_;
  uint64_t private_data_length_;
};

///////////////////////////////////////////////////////////////
// Track element.
class Track {
 public:
  // The |seed| parameter is used to synthesize a UID for the track.
  explicit Track(unsigned int* seed);
  virtual ~Track();

  // Adds a ContentEncoding element to the Track. Returns true on success.
  virtual bool AddContentEncoding();

  // Returns the ContentEncoding by index. Returns NULL if there is no
  // ContentEncoding match.
  ContentEncoding* GetContentEncodingByIndex(uint32_t index) const;

  // Returns the size in bytes for the payload of the Track element.
  virtual uint64_t PayloadSize() const;

  // Returns the size in bytes of the Track element.
  virtual uint64_t Size() const;

  // Output the Track element to the writer. Returns true on success.
  virtual bool Write(IMkvWriter* writer) const;

  // Sets the CodecPrivate element of the Track element. Copies |length|
  // bytes from |codec_private| to |codec_private_|. Returns true on success.
  bool SetCodecPrivate(const uint8_t* codec_private, uint64_t length);

  void set_codec_id(const char* codec_id);
  const char* codec_id() const { return codec_id_; }
  const uint8_t* codec_private() const { return codec_private_; }
  void set_language(const char* language);
  const char* language() const { return language_; }
  void set_max_block_additional_id(uint64_t max_block_additional_id) {
    max_block_additional_id_ = max_block_additional_id;
  }
  uint64_t max_block_additional_id() const { return max_block_additional_id_; }
  void set_name(const char* name);
  const char* name() const { return name_; }
  void set_number(uint64_t number) { number_ = number; }
  uint64_t number() const { return number_; }
  void set_type(uint64_t type) { type_ = type; }
  uint64_t type() const { return type_; }
  void set_uid(uint64_t uid) { uid_ = uid; }
  uint64_t uid() const { return uid_; }
  void set_codec_delay(uint64_t codec_delay) { codec_delay_ = codec_delay; }
  uint64_t codec_delay() const { return codec_delay_; }
  void set_seek_pre_roll(uint64_t seek_pre_roll) {
    seek_pre_roll_ = seek_pre_roll;
  }
  uint64_t seek_pre_roll() const { return seek_pre_roll_; }
  void set_default_duration(uint64_t default_duration) {
    default_duration_ = default_duration;
  }
  uint64_t default_duration() const { return default_duration_; }

  uint64_t codec_private_length() const { return codec_private_length_; }
  uint32_t content_encoding_entries_size() const {
    return content_encoding_entries_size_;
  }

 private:
  // Track element names.
  char* codec_id_;
  uint8_t* codec_private_;
  char* language_;
  uint64_t max_block_additional_id_;
  char* name_;
  uint64_t number_;
  uint64_t type_;
  uint64_t uid_;
  uint64_t codec_delay_;
  uint64_t seek_pre_roll_;
  uint64_t default_duration_;

  // Size of the CodecPrivate data in bytes.
  uint64_t codec_private_length_;

  // ContentEncoding element list.
  ContentEncoding** content_encoding_entries_;

  // Number of ContentEncoding elements added.
  uint32_t content_encoding_entries_size_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Track);
};

///////////////////////////////////////////////////////////////
// Track that has video specific elements.
class VideoTrack : public Track {
 public:
  // Supported modes for stereo 3D.
  enum StereoMode {
    kMono = 0,
    kSideBySideLeftIsFirst = 1,
    kTopBottomRightIsFirst = 2,
    kTopBottomLeftIsFirst = 3,
    kSideBySideRightIsFirst = 11
  };

  enum AlphaMode { kNoAlpha = 0, kAlpha = 1 };

  // The |seed| parameter is used to synthesize a UID for the track.
  explicit VideoTrack(unsigned int* seed);
  virtual ~VideoTrack();

  // Returns the size in bytes for the payload of the Track element plus the
  // video specific elements.
  virtual uint64_t PayloadSize() const;

  // Output the VideoTrack element to the writer. Returns true on success.
  virtual bool Write(IMkvWriter* writer) const;

  // Sets the video's stereo mode. Returns true on success.
  bool SetStereoMode(uint64_t stereo_mode);

  // Sets the video's alpha mode. Returns true on success.
  bool SetAlphaMode(uint64_t alpha_mode);

  void set_display_height(uint64_t height) { display_height_ = height; }
  uint64_t display_height() const { return display_height_; }
  void set_display_width(uint64_t width) { display_width_ = width; }
  uint64_t display_width() const { return display_width_; }
  void set_pixel_height(uint64_t height) { pixel_height_ = height; }
  uint64_t pixel_height() const { return pixel_height_; }
  void set_pixel_width(uint64_t width) { pixel_width_ = width; }
  uint64_t pixel_width() const { return pixel_width_; }

  void set_crop_left(uint64_t crop_left) { crop_left_ = crop_left; }
  uint64_t crop_left() const { return crop_left_; }
  void set_crop_right(uint64_t crop_right) { crop_right_ = crop_right; }
  uint64_t crop_right() const { return crop_right_; }
  void set_crop_top(uint64_t crop_top) { crop_top_ = crop_top; }
  uint64_t crop_top() const { return crop_top_; }
  void set_crop_bottom(uint64_t crop_bottom) { crop_bottom_ = crop_bottom; }
  uint64_t crop_bottom() const { return crop_bottom_; }

  void set_frame_rate(double frame_rate) { frame_rate_ = frame_rate; }
  double frame_rate() const { return frame_rate_; }
  void set_height(uint64_t height) { height_ = height; }
  uint64_t height() const { return height_; }
  uint64_t stereo_mode() { return stereo_mode_; }
  uint64_t alpha_mode() { return alpha_mode_; }
  void set_width(uint64_t width) { width_ = width; }
  uint64_t width() const { return width_; }
  void set_colour_space(const char* colour_space);
  const char* colour_space() const { return colour_space_; }

  Colour* colour() { return colour_; }

  // Deep copies |colour|.
  bool SetColour(const Colour& colour);

  Projection* projection() { return projection_; }

  // Deep copies |projection|.
  bool SetProjection(const Projection& projection);

 private:
  // Returns the size in bytes of the Video element.
  uint64_t VideoPayloadSize() const;

  // Video track element names.
  uint64_t display_height_;
  uint64_t display_width_;
  uint64_t pixel_height_;
  uint64_t pixel_width_;
  uint64_t crop_left_;
  uint64_t crop_right_;
  uint64_t crop_top_;
  uint64_t crop_bottom_;
  double frame_rate_;
  uint64_t height_;
  uint64_t stereo_mode_;
  uint64_t alpha_mode_;
  uint64_t width_;
  char* colour_space_;

  Colour* colour_;
  Projection* projection_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(VideoTrack);
};

///////////////////////////////////////////////////////////////
// Track that has audio specific elements.
class AudioTrack : public Track {
 public:
  // The |seed| parameter is used to synthesize a UID for the track.
  explicit AudioTrack(unsigned int* seed);
  virtual ~AudioTrack();

  // Returns the size in bytes for the payload of the Track element plus the
  // audio specific elements.
  virtual uint64_t PayloadSize() const;

  // Output the AudioTrack element to the writer. Returns true on success.
  virtual bool Write(IMkvWriter* writer) const;

  void set_bit_depth(uint64_t bit_depth) { bit_depth_ = bit_depth; }
  uint64_t bit_depth() const { return bit_depth_; }
  void set_channels(uint64_t channels) { channels_ = channels; }
  uint64_t channels() const { return channels_; }
  void set_sample_rate(double sample_rate) { sample_rate_ = sample_rate; }
  double sample_rate() const { return sample_rate_; }

 private:
  // Audio track element names.
  uint64_t bit_depth_;
  uint64_t channels_;
  double sample_rate_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(AudioTrack);
};

///////////////////////////////////////////////////////////////
// Tracks element
class Tracks {
 public:
  // Audio and video type defined by the Matroska specs.
  enum { kVideo = 0x1, kAudio = 0x2 };

  static const char kOpusCodecId[];
  static const char kVorbisCodecId[];
  static const char kAv1CodecId[];
  static const char kVp8CodecId[];
  static const char kVp9CodecId[];
  static const char kWebVttCaptionsId[];
  static const char kWebVttDescriptionsId[];
  static const char kWebVttMetadataId[];
  static const char kWebVttSubtitlesId[];

  Tracks();
  ~Tracks();

  // Adds a Track element to the Tracks object. |track| will be owned and
  // deleted by the Tracks object. Returns true on success. |number| is the
  // number to use for the track. |number| must be >= 0. If |number| == 0
  // then the muxer will decide on the track number.
  bool AddTrack(Track* track, int32_t number);

  // Returns the track by index. Returns NULL if there is no track match.
  const Track* GetTrackByIndex(uint32_t idx) const;

  // Search the Tracks and return the track that matches |tn|. Returns NULL
  // if there is no track match.
  Track* GetTrackByNumber(uint64_t track_number) const;

  // Returns true if the track number is an audio track.
  bool TrackIsAudio(uint64_t track_number) const;

  // Returns true if the track number is a video track.
  bool TrackIsVideo(uint64_t track_number) const;

  // Output the Tracks element to the writer. Returns true on success.
  bool Write(IMkvWriter* writer) const;

  uint32_t track_entries_size() const { return track_entries_size_; }

 private:
  // Track element list.
  Track** track_entries_;

  // Number of Track elements added.
  uint32_t track_entries_size_;

  // Whether or not Tracks element has already been written via IMkvWriter.
  mutable bool wrote_tracks_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Tracks);
};

///////////////////////////////////////////////////////////////
// Chapter element
//
class Chapter {
 public:
  // Set the identifier for this chapter.  (This corresponds to the
  // Cue Identifier line in WebVTT.)
  // TODO(matthewjheaney): the actual serialization of this item in
  // MKV is pending.
  bool set_id(const char* id);

  // Converts the nanosecond start and stop times of this chapter to
  // their corresponding timecode values, and stores them that way.
  void set_time(const Segment& segment, uint64_t start_time_ns,
                uint64_t end_time_ns);

  // Sets the uid for this chapter. Primarily used to enable
  // deterministic output from the muxer.
  void set_uid(const uint64_t uid) { uid_ = uid; }

  // Add a title string to this chapter, per the semantics described
  // here:
  //  http://www.matroska.org/technical/specs/index.html
  //
  // The title ("chapter string") is a UTF-8 string.
  //
  // The language has ISO 639-2 representation, described here:
  //  http://www.loc.gov/standards/iso639-2/englangn.html
  //  http://www.loc.gov/standards/iso639-2/php/English_list.php
  // If you specify NULL as the language value, this implies
  // English ("eng").
  //
  // The country value corresponds to the codes listed here:
  //  http://www.iana.org/domains/root/db/
  //
  // The function returns false if the string could not be allocated.
  bool add_string(const char* title, const char* language, const char* country);

 private:
  friend class Chapters;

  // For storage of chapter titles that differ by language.
  class Display {
   public:
    // Establish representation invariant for new Display object.
    void Init();

    // Reclaim resources, in anticipation of destruction.
    void Clear();

    // Copies the title to the |title_| member.  Returns false on
    // error.
    bool set_title(const char* title);

    // Copies the language to the |language_| member.  Returns false
    // on error.
    bool set_language(const char* language);

    // Copies the country to the |country_| member.  Returns false on
    // error.
    bool set_country(const char* country);

    // If |writer| is non-NULL, serialize the Display sub-element of
    // the Atom into the stream.  Returns the Display element size on
    // success, 0 if error.
    uint64_t WriteDisplay(IMkvWriter* writer) const;

   private:
    char* title_;
    char* language_;
    char* country_;
  };

  Chapter();
  ~Chapter();

  // Establish the representation invariant for a newly-created
  // Chapter object.  The |seed| parameter is used to create the UID
  // for this chapter atom.
  void Init(unsigned int* seed);

  // Copies this Chapter object to a different one.  This is used when
  // expanding a plain array of Chapter objects (see Chapters).
  void ShallowCopy(Chapter* dst) const;

  // Reclaim resources used by this Chapter object, pending its
  // destruction.
  void Clear();

  // If there is no storage remaining on the |displays_| array for a
  // new display object, creates a new, longer array and copies the
  // existing Display objects to the new array.  Returns false if the
  // array cannot be expanded.
  bool ExpandDisplaysArray();

  // If |writer| is non-NULL, serialize the Atom sub-element into the
  // stream.  Returns the total size of the element on success, 0 if
  // error.
  uint64_t WriteAtom(IMkvWriter* writer) const;

  // The string identifier for this chapter (corresponds to WebVTT cue
  // identifier).
  char* id_;

  // Start timecode of the chapter.
  uint64_t start_timecode_;

  // Stop timecode of the chapter.
  uint64_t end_timecode_;

  // The binary identifier for this chapter.
  uint64_t uid_;

  // The Atom element can contain multiple Display sub-elements, as
  // the same logical title can be rendered in different languages.
  Display* displays_;

  // The physical length (total size) of the |displays_| array.
  int displays_size_;

  // The logical length (number of active elements) on the |displays_|
  // array.
  int displays_count_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Chapter);
};

///////////////////////////////////////////////////////////////
// Chapters element
//
class Chapters {
 public:
  Chapters();
  ~Chapters();

  Chapter* AddChapter(unsigned int* seed);

  // Returns the number of chapters that have been added.
  int Count() const;

  // Output the Chapters element to the writer. Returns true on success.
  bool Write(IMkvWriter* writer) const;

 private:
  // Expands the chapters_ array if there is not enough space to contain
  // another chapter object.  Returns true on success.
  bool ExpandChaptersArray();

  // If |writer| is non-NULL, serialize the Edition sub-element of the
  // Chapters element into the stream.  Returns the Edition element
  // size on success, 0 if error.
  uint64_t WriteEdition(IMkvWriter* writer) const;

  // Total length of the chapters_ array.
  int chapters_size_;

  // Number of active chapters on the chapters_ array.
  int chapters_count_;

  // Array for storage of chapter objects.
  Chapter* chapters_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Chapters);
};

///////////////////////////////////////////////////////////////
// Tag element
//
class Tag {
 public:
  bool add_simple_tag(const char* tag_name, const char* tag_string);

 private:
  // Tags calls Clear and the destructor of Tag
  friend class Tags;

  // For storage of simple tags
  class SimpleTag {
   public:
    // Establish representation invariant for new SimpleTag object.
    void Init();

    // Reclaim resources, in anticipation of destruction.
    void Clear();

    // Copies the title to the |tag_name_| member.  Returns false on
    // error.
    bool set_tag_name(const char* tag_name);

    // Copies the language to the |tag_string_| member.  Returns false
    // on error.
    bool set_tag_string(const char* tag_string);

    // If |writer| is non-NULL, serialize the SimpleTag sub-element of
    // the Atom into the stream.  Returns the SimpleTag element size on
    // success, 0 if error.
    uint64_t Write(IMkvWriter* writer) const;

   private:
    char* tag_name_;
    char* tag_string_;
  };

  Tag();
  ~Tag();

  // Copies this Tag object to a different one.  This is used when
  // expanding a plain array of Tag objects (see Tags).
  void ShallowCopy(Tag* dst) const;

  // Reclaim resources used by this Tag object, pending its
  // destruction.
  void Clear();

  // If there is no storage remaining on the |simple_tags_| array for a
  // new display object, creates a new, longer array and copies the
  // existing SimpleTag objects to the new array.  Returns false if the
  // array cannot be expanded.
  bool ExpandSimpleTagsArray();

  // If |writer| is non-NULL, serialize the Tag sub-element into the
  // stream.  Returns the total size of the element on success, 0 if
  // error.
  uint64_t Write(IMkvWriter* writer) const;

  // The Atom element can contain multiple SimpleTag sub-elements
  SimpleTag* simple_tags_;

  // The physical length (total size) of the |simple_tags_| array.
  int simple_tags_size_;

  // The logical length (number of active elements) on the |simple_tags_|
  // array.
  int simple_tags_count_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Tag);
};

///////////////////////////////////////////////////////////////
// Tags element
//
class Tags {
 public:
  Tags();
  ~Tags();

  Tag* AddTag();

  // Returns the number of tags that have been added.
  int Count() const;

  // Output the Tags element to the writer. Returns true on success.
  bool Write(IMkvWriter* writer) const;

 private:
  // Expands the tags_ array if there is not enough space to contain
  // another tag object.  Returns true on success.
  bool ExpandTagsArray();

  // Total length of the tags_ array.
  int tags_size_;

  // Number of active tags on the tags_ array.
  int tags_count_;

  // Array for storage of tag objects.
  Tag* tags_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Tags);
};

///////////////////////////////////////////////////////////////
// Cluster element
//
// Notes:
//  |Init| must be called before any other method in this class.
class Cluster {
 public:
  // |timecode| is the absolute timecode of the cluster. |cues_pos| is the
  // position for the cluster within the segment that should be written in
  // the cues element. |timecode_scale| is the timecode scale of the segment.
  Cluster(uint64_t timecode, int64_t cues_pos, uint64_t timecode_scale,
          bool write_last_frame_with_duration = false,
          bool fixed_size_timecode = false);
  ~Cluster();

  bool Init(IMkvWriter* ptr_writer);

  // Adds a frame to be output in the file. The frame is written out through
  // |writer_| if successful. Returns true on success.
  bool AddFrame(const Frame* frame);

  // Adds a frame to be output in the file. The frame is written out through
  // |writer_| if successful. Returns true on success.
  // Inputs:
  //   data: Pointer to the data
  //   length: Length of the data
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.  The range of allowed values is [1, 126].
  //   timecode:     Absolute (not relative to cluster) timestamp of the
  //                 frame, expressed in timecode units.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrame(const uint8_t* data, uint64_t length, uint64_t track_number,
                uint64_t timecode,  // timecode units (absolute)
                bool is_key);

  // Adds a frame to be output in the file. The frame is written out through
  // |writer_| if successful. Returns true on success.
  // Inputs:
  //   data: Pointer to the data
  //   length: Length of the data
  //   additional: Pointer to the additional data
  //   additional_length: Length of the additional data
  //   add_id: Value of BlockAddID element
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.  The range of allowed values is [1, 126].
  //   abs_timecode: Absolute (not relative to cluster) timestamp of the
  //                 frame, expressed in timecode units.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrameWithAdditional(const uint8_t* data, uint64_t length,
                              const uint8_t* additional,
                              uint64_t additional_length, uint64_t add_id,
                              uint64_t track_number, uint64_t abs_timecode,
                              bool is_key);

  // Adds a frame to be output in the file. The frame is written out through
  // |writer_| if successful. Returns true on success.
  // Inputs:
  //   data: Pointer to the data.
  //   length: Length of the data.
  //   discard_padding: DiscardPadding element value.
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.  The range of allowed values is [1, 126].
  //   abs_timecode: Absolute (not relative to cluster) timestamp of the
  //                 frame, expressed in timecode units.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrameWithDiscardPadding(const uint8_t* data, uint64_t length,
                                  int64_t discard_padding,
                                  uint64_t track_number, uint64_t abs_timecode,
                                  bool is_key);

  // Writes a frame of metadata to the output medium; returns true on
  // success.
  // Inputs:
  //   data: Pointer to the data
  //   length: Length of the data
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.  The range of allowed values is [1, 126].
  //   timecode:     Absolute (not relative to cluster) timestamp of the
  //                 metadata frame, expressed in timecode units.
  //   duration:     Duration of metadata frame, in timecode units.
  //
  // The metadata frame is written as a block group, with a duration
  // sub-element but no reference time sub-elements (indicating that
  // it is considered a keyframe, per Matroska semantics).
  bool AddMetadata(const uint8_t* data, uint64_t length, uint64_t track_number,
                   uint64_t timecode, uint64_t duration);

  // Increments the size of the cluster's data in bytes.
  void AddPayloadSize(uint64_t size);

  // Closes the cluster so no more data can be written to it. Will update the
  // cluster's size if |writer_| is seekable. Returns true on success. This
  // variant of Finalize() fails when |write_last_frame_with_duration_| is set
  // to true.
  bool Finalize();

  // Closes the cluster so no more data can be written to it. Will update the
  // cluster's size if |writer_| is seekable. Returns true on success.
  // Inputs:
  //   set_last_frame_duration: Boolean indicating whether or not the duration
  //                            of the last frame should be set. If set to
  //                            false, the |duration| value is ignored and
  //                            |write_last_frame_with_duration_| will not be
  //                            honored.
  //   duration: Duration of the Cluster in timecode scale.
  bool Finalize(bool set_last_frame_duration, uint64_t duration);

  // Returns the size in bytes for the entire Cluster element.
  uint64_t Size() const;

  // Given |abs_timecode|, calculates timecode relative to most recent timecode.
  // Returns -1 on failure, or a relative timecode.
  int64_t GetRelativeTimecode(int64_t abs_timecode) const;

  int64_t size_position() const { return size_position_; }
  int32_t blocks_added() const { return blocks_added_; }
  uint64_t payload_size() const { return payload_size_; }
  int64_t position_for_cues() const { return position_for_cues_; }
  uint64_t timecode() const { return timecode_; }
  uint64_t timecode_scale() const { return timecode_scale_; }
  void set_write_last_frame_with_duration(bool write_last_frame_with_duration) {
    write_last_frame_with_duration_ = write_last_frame_with_duration;
  }
  bool write_last_frame_with_duration() const {
    return write_last_frame_with_duration_;
  }

 private:
  // Iterator type for the |stored_frames_| map.
  typedef std::map<uint64_t, std::list<Frame*> >::iterator FrameMapIterator;

  // Utility method that confirms that blocks can still be added, and that the
  // cluster header has been written. Used by |DoWriteFrame*|. Returns true
  // when successful.
  bool PreWriteBlock();

  // Utility method used by the |DoWriteFrame*| methods that handles the book
  // keeping required after each block is written.
  void PostWriteBlock(uint64_t element_size);

  // Does some verification and calls WriteFrame.
  bool DoWriteFrame(const Frame* const frame);

  // Either holds back the given frame, or writes it out depending on whether or
  // not |write_last_frame_with_duration_| is set.
  bool QueueOrWriteFrame(const Frame* const frame);

  // Outputs the Cluster header to |writer_|. Returns true on success.
  bool WriteClusterHeader();

  // Number of blocks added to the cluster.
  int32_t blocks_added_;

  // Flag telling if the cluster has been closed.
  bool finalized_;

  // Flag indicating whether the cluster's timecode will always be written out
  // using 8 bytes.
  bool fixed_size_timecode_;

  // Flag telling if the cluster's header has been written.
  bool header_written_;

  // The size of the cluster elements in bytes.
  uint64_t payload_size_;

  // The file position used for cue points.
  const int64_t position_for_cues_;

  // The file position of the cluster's size element.
  int64_t size_position_;

  // The absolute timecode of the cluster.
  const uint64_t timecode_;

  // The timecode scale of the Segment containing the cluster.
  const uint64_t timecode_scale_;

  // Flag indicating whether the last frame of the cluster should be written as
  // a Block with Duration. If set to true, then it will result in holding back
  // of frames and the parameterized version of Finalize() must be called to
  // finish writing the Cluster.
  bool write_last_frame_with_duration_;

  // Map used to hold back frames, if required. Track number is the key.
  std::map<uint64_t, std::list<Frame*> > stored_frames_;

  // Map from track number to the timestamp of the last block written for that
  // track.
  std::map<uint64_t, uint64_t> last_block_timestamp_;

  // Pointer to the writer object. Not owned by this class.
  IMkvWriter* writer_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Cluster);
};

///////////////////////////////////////////////////////////////
// SeekHead element
class SeekHead {
 public:
  SeekHead();
  ~SeekHead();

  // TODO(fgalligan): Change this to reserve a certain size. Then check how
  // big the seek entry to be added is as not every seek entry will be the
  // maximum size it could be.
  // Adds a seek entry to be written out when the element is finalized. |id|
  // must be the coded mkv element id. |pos| is the file position of the
  // element. Returns true on success.
  bool AddSeekEntry(uint32_t id, uint64_t pos);

  // Writes out SeekHead and SeekEntry elements. Returns true on success.
  bool Finalize(IMkvWriter* writer) const;

  // Returns the id of the Seek Entry at the given index. Returns -1 if index is
  // out of range.
  uint32_t GetId(int index) const;

  // Returns the position of the Seek Entry at the given index. Returns -1 if
  // index is out of range.
  uint64_t GetPosition(int index) const;

  // Sets the Seek Entry id and position at given index.
  // Returns true on success.
  bool SetSeekEntry(int index, uint32_t id, uint64_t position);

  // Reserves space by writing out a Void element which will be updated with
  // a SeekHead element later. Returns true on success.
  bool Write(IMkvWriter* writer);

  // We are going to put a cap on the number of Seek Entries.
  constexpr static int32_t kSeekEntryCount = 5;

 private:
  // Returns the maximum size in bytes of one seek entry.
  uint64_t MaxEntrySize() const;

  // Seek entry id element list.
  uint32_t seek_entry_id_[kSeekEntryCount];

  // Seek entry pos element list.
  uint64_t seek_entry_pos_[kSeekEntryCount];

  // The file position of SeekHead element.
  int64_t start_pos_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(SeekHead);
};

///////////////////////////////////////////////////////////////
// Segment Information element
class SegmentInfo {
 public:
  SegmentInfo();
  ~SegmentInfo();

  // Will update the duration if |duration_| is > 0.0. Returns true on success.
  bool Finalize(IMkvWriter* writer) const;

  // Sets |muxing_app_| and |writing_app_|.
  bool Init();

  // Output the Segment Information element to the writer. Returns true on
  // success.
  bool Write(IMkvWriter* writer);

  void set_duration(double duration) { duration_ = duration; }
  double duration() const { return duration_; }
  void set_muxing_app(const char* app);
  const char* muxing_app() const { return muxing_app_; }
  void set_timecode_scale(uint64_t scale) { timecode_scale_ = scale; }
  uint64_t timecode_scale() const { return timecode_scale_; }
  void set_writing_app(const char* app);
  const char* writing_app() const { return writing_app_; }
  void set_date_utc(int64_t date_utc) { date_utc_ = date_utc; }
  int64_t date_utc() const { return date_utc_; }

 private:
  // Segment Information element names.
  // Initially set to -1 to signify that a duration has not been set and should
  // not be written out.
  double duration_;
  // Set to libwebm-%d.%d.%d.%d, major, minor, build, revision.
  char* muxing_app_;
  uint64_t timecode_scale_;
  // Initially set to libwebm-%d.%d.%d.%d, major, minor, build, revision.
  char* writing_app_;
  // INT64_MIN when DateUTC is not set.
  int64_t date_utc_;

  // The file position of the duration element.
  int64_t duration_pos_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(SegmentInfo);
};

///////////////////////////////////////////////////////////////
// This class represents the main segment in a WebM file. Currently only
// supports one Segment element.
//
// Notes:
//  |Init| must be called before any other method in this class.
class Segment {
 public:
  enum Mode { kLive = 0x1, kFile = 0x2 };

  enum CuesPosition {
    kAfterClusters = 0x0,  // Position Cues after Clusters - Default
    kBeforeClusters = 0x1  // Position Cues before Clusters
  };

  static constexpr uint32_t kDefaultDocTypeVersion = 4;
  static constexpr uint64_t kDefaultMaxClusterDuration = 30000000000ULL;

  Segment();
  ~Segment();

  // Initializes |SegmentInfo| and returns result. Always returns false when
  // |ptr_writer| is NULL.
  bool Init(IMkvWriter* ptr_writer);

  // Adds a generic track to the segment.  Returns the newly-allocated
  // track object (which is owned by the segment) on success, NULL on
  // error. |number| is the number to use for the track.  |number|
  // must be >= 0. If |number| == 0 then the muxer will decide on the
  // track number.
  Track* AddTrack(int32_t number);

  // Adds a Vorbis audio track to the segment. Returns the number of the track
  // on success, 0 on error. |number| is the number to use for the audio track.
  // |number| must be >= 0. If |number| == 0 then the muxer will decide on
  // the track number.
  uint64_t AddAudioTrack(int32_t sample_rate, int32_t channels, int32_t number);

  // Adds an empty chapter to the chapters of this segment.  Returns
  // non-NULL on success.  After adding the chapter, the caller should
  // populate its fields via the Chapter member functions.
  Chapter* AddChapter();

  // Adds an empty tag to the tags of this segment.  Returns
  // non-NULL on success.  After adding the tag, the caller should
  // populate its fields via the Tag member functions.
  Tag* AddTag();

  // Adds a cue point to the Cues element. |timestamp| is the time in
  // nanoseconds of the cue's time. |track| is the Track of the Cue. This
  // function must be called after AddFrame to calculate the correct
  // BlockNumber for the CuePoint. Returns true on success.
  bool AddCuePoint(uint64_t timestamp, uint64_t track);

  // Adds a frame to be output in the file. Returns true on success.
  // Inputs:
  //   data: Pointer to the data
  //   length: Length of the data
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.
  //   timestamp:    Timestamp of the frame in nanoseconds from 0.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrame(const uint8_t* data, uint64_t length, uint64_t track_number,
                uint64_t timestamp_ns, bool is_key);

  // Writes a frame of metadata to the output medium; returns true on
  // success.
  // Inputs:
  //   data: Pointer to the data
  //   length: Length of the data
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.
  //   timecode:     Absolute timestamp of the metadata frame, expressed
  //                 in nanosecond units.
  //   duration:     Duration of metadata frame, in nanosecond units.
  //
  // The metadata frame is written as a block group, with a duration
  // sub-element but no reference time sub-elements (indicating that
  // it is considered a keyframe, per Matroska semantics).
  bool AddMetadata(const uint8_t* data, uint64_t length, uint64_t track_number,
                   uint64_t timestamp_ns, uint64_t duration_ns);

  // Writes a frame with additional data to the output medium; returns true on
  // success.
  // Inputs:
  //   data: Pointer to the data.
  //   length: Length of the data.
  //   additional: Pointer to additional data.
  //   additional_length: Length of additional data.
  //   add_id: Additional ID which identifies the type of additional data.
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.
  //   timestamp:    Absolute timestamp of the frame, expressed in nanosecond
  //                 units.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrameWithAdditional(const uint8_t* data, uint64_t length,
                              const uint8_t* additional,
                              uint64_t additional_length, uint64_t add_id,
                              uint64_t track_number, uint64_t timestamp,
                              bool is_key);

  // Writes a frame with DiscardPadding to the output medium; returns true on
  // success.
  // Inputs:
  //   data: Pointer to the data.
  //   length: Length of the data.
  //   discard_padding: DiscardPadding element value.
  //   track_number: Track to add the data to. Value returned by Add track
  //                 functions.
  //   timestamp:    Absolute timestamp of the frame, expressed in nanosecond
  //                 units.
  //   is_key:       Flag telling whether or not this frame is a key frame.
  bool AddFrameWithDiscardPadding(const uint8_t* data, uint64_t length,
                                  int64_t discard_padding,
                                  uint64_t track_number, uint64_t timestamp,
                                  bool is_key);

  // Writes a Frame to the output medium. Chooses the correct way of writing
  // the frame (Block vs SimpleBlock) based on the parameters passed.
  // Inputs:
  //   frame: frame object
  bool AddGenericFrame(const Frame* frame);

  // Adds a VP8 video track to the segment. Returns the number of the track on
  // success, 0 on error. |number| is the number to use for the video track.
  // |number| must be >= 0. If |number| == 0 then the muxer will decide on
  // the track number.
  uint64_t AddVideoTrack(int32_t width, int32_t height, int32_t number);

  // This function must be called after Finalize() if you need a copy of the
  // output with Cues written before the Clusters. It will return false if the
  // writer is not seekable of if chunking is set to true.
  // Input parameters:
  // reader - an IMkvReader object created with the same underlying file of the
  //          current writer object. Make sure to close the existing writer
  //          object before creating this so that all the data is properly
  //          flushed and available for reading.
  // writer - an IMkvWriter object pointing to a *different* file than the one
  //          pointed by the current writer object. This file will contain the
  //          Cues element before the Clusters.
  bool CopyAndMoveCuesBeforeClusters(mkvparser::IMkvReader* reader,
                                     IMkvWriter* writer);

  // Sets which track to use for the Cues element. Must have added the track
  // before calling this function. Returns true on success. |track_number| is
  // returned by the Add track functions.
  bool CuesTrack(uint64_t track_number);

  // This will force the muxer to create a new Cluster when the next frame is
  // added.
  void ForceNewClusterOnNextFrame();

  // Writes out any frames that have not been written out. Finalizes the last
  // cluster. May update the size and duration of the segment. May output the
  // Cues element. May finalize the SeekHead element. Returns true on success.
  bool Finalize();

  // Returns the Cues object.
  Cues* GetCues() { return &cues_; }

  // Returns the Segment Information object.
  const SegmentInfo* GetSegmentInfo() const { return &segment_info_; }
  SegmentInfo* GetSegmentInfo() { return &segment_info_; }

  // Search the Tracks and return the track that matches |track_number|.
  // Returns NULL if there is no track match.
  Track* GetTrackByNumber(uint64_t track_number) const;

  // Toggles whether to output a cues element.
  void OutputCues(bool output_cues);

  // Toggles whether to write the last frame in each Cluster with Duration.
  void AccurateClusterDuration(bool accurate_cluster_duration);

  // Toggles whether to write the Cluster Timecode using exactly 8 bytes.
  void UseFixedSizeClusterTimecode(bool fixed_size_cluster_timecode);

  // Sets if the muxer will output files in chunks or not. |chunking| is a
  // flag telling whether or not to turn on chunking. |filename| is the base
  // filename for the chunk files. The header chunk file will be named
  // |filename|.hdr and the data chunks will be named
  // |filename|_XXXXXX.chk. Chunking implies that the muxer will be writing
  // to files so the muxer will use the default MkvWriter class to control
  // what data is written to what files. Returns true on success.
  // TODO: Should we change the IMkvWriter Interface to add Open and Close?
  // That will force the interface to be dependent on files.
  bool SetChunking(bool chunking, const char* filename);

  bool chunking() const { return chunking_; }
  uint64_t cues_track() const { return cues_track_; }
  void set_max_cluster_duration(uint64_t max_cluster_duration) {
    max_cluster_duration_ = max_cluster_duration;
  }
  uint64_t max_cluster_duration() const { return max_cluster_duration_; }
  void set_max_cluster_size(uint64_t max_cluster_size) {
    max_cluster_size_ = max_cluster_size;
  }
  uint64_t max_cluster_size() const { return max_cluster_size_; }
  void set_mode(Mode mode) { mode_ = mode; }
  Mode mode() const { return mode_; }
  CuesPosition cues_position() const { return cues_position_; }
  bool output_cues() const { return output_cues_; }
  void set_estimate_file_duration(bool estimate_duration) {
    estimate_file_duration_ = estimate_duration;
  }
  bool estimate_file_duration() const { return estimate_file_duration_; }
  const SegmentInfo* segment_info() const { return &segment_info_; }
  void set_duration(double duration) { duration_ = duration; }
  double duration() const { return duration_; }

  // Returns true when codec IDs are valid for WebM.
  bool DocTypeIsWebm() const;

 private:
  // Checks if header information has been output and initialized. If not it
  // will output the Segment element and initialize the SeekHead elment and
  // Cues elements.
  bool CheckHeaderInfo();

  // Sets |doc_type_version_| based on the current element requirements.
  void UpdateDocTypeVersion();

  // Sets |name| according to how many chunks have been written. |ext| is the
  // file extension. |name| must be deleted by the calling app. Returns true
  // on success.
  bool UpdateChunkName(const char* ext, char** name) const;

  // Returns the maximum offset within the segment's payload. When chunking
  // this function is needed to determine offsets of elements within the
  // chunked files. Returns -1 on error.
  int64_t MaxOffset();

  // Adds the frame to our frame array.
  bool QueueFrame(Frame* frame);

  // Output all frames that are queued. Returns -1 on error, otherwise
  // it returns the number of frames written.
  int WriteFramesAll();

  // Output all frames that are queued that have an end time that is less
  // then |timestamp|. Returns true on success and if there are no frames
  // queued.
  bool WriteFramesLessThan(uint64_t timestamp);

  // Outputs the segment header, Segment Information element, SeekHead element,
  // and Tracks element to |writer_|.
  bool WriteSegmentHeader();

  // Given a frame with the specified timestamp (nanosecond units) and
  // keyframe status, determine whether a new cluster should be
  // created, before writing enqueued frames and the frame itself. The
  // function returns one of the following values:
  //  -1 = error: an out-of-order frame was detected
  //  0 = do not create a new cluster, and write frame to the existing cluster
  //  1 = create a new cluster, and write frame to that new cluster
  //  2 = create a new cluster, and re-run test
  int TestFrame(uint64_t track_num, uint64_t timestamp_ns, bool key) const;

  // Create a new cluster, using the earlier of the first enqueued
  // frame, or the indicated time. Returns true on success.
  bool MakeNewCluster(uint64_t timestamp_ns);

  // Checks whether a new cluster needs to be created, and if so
  // creates a new cluster. Returns false if creation of a new cluster
  // was necessary but creation was not successful.
  bool DoNewClusterProcessing(uint64_t track_num, uint64_t timestamp_ns,
                              bool key);

  // Adjusts Cue Point values (to place Cues before Clusters) so that they
  // reflect the correct offsets.
  void MoveCuesBeforeClusters();

  // This function recursively computes the correct cluster offsets (this is
  // done to move the Cues before Clusters). It recursively updates the change
  // in size (which indicates a change in cluster offset) until no sizes change.
  // Parameters:
  // diff - indicates the difference in size of the Cues element that needs to
  //        accounted for.
  // index - index in the list of Cues which is currently being adjusted.
  // cue_size - sum of size of all the CuePoint elements.
  void MoveCuesBeforeClustersHelper(uint64_t diff, int index,
                                    uint64_t* cue_size);

  // Seeds the random number generator used to make UIDs.
  unsigned int seed_;

  // WebM elements
  Cues cues_;
  SeekHead seek_head_;
  SegmentInfo segment_info_;
  Tracks tracks_;
  Chapters chapters_;
  Tags tags_;

  // Number of chunks written.
  int chunk_count_;

  // Current chunk filename.
  char* chunk_name_;

  // Default MkvWriter object created by this class used for writing clusters
  // out in separate files.
  MkvWriter* chunk_writer_cluster_;

  // Default MkvWriter object created by this class used for writing Cues
  // element out to a file.
  MkvWriter* chunk_writer_cues_;

  // Default MkvWriter object created by this class used for writing the
  // Matroska header out to a file.
  MkvWriter* chunk_writer_header_;

  // Flag telling whether or not the muxer is chunking output to multiple
  // files.
  bool chunking_;

  // Base filename for the chunked files.
  char* chunking_base_name_;

  // File position offset where the Clusters end.
  int64_t cluster_end_offset_;

  // List of clusters.
  Cluster** cluster_list_;

  // Number of cluster pointers allocated in the cluster list.
  int32_t cluster_list_capacity_;

  // Number of clusters in the cluster list.
  int32_t cluster_list_size_;

  // Indicates whether Cues should be written before or after Clusters
  CuesPosition cues_position_;

  // Track number that is associated with the cues element for this segment.
  uint64_t cues_track_;

  // Tells the muxer to force a new cluster on the next Block.
  bool force_new_cluster_;

  // List of stored audio frames. These variables are used to store frames so
  // the muxer can follow the guideline "Audio blocks that contain the video
  // key frame's timecode should be in the same cluster as the video key frame
  // block."
  Frame** frames_;

  // Number of frame pointers allocated in the frame list.
  int32_t frames_capacity_;

  // Number of frames in the frame list.
  int32_t frames_size_;

  // Flag telling if a video track has been added to the segment.
  bool has_video_;

  // Flag telling if the segment's header has been written.
  bool header_written_;

  // Duration of the last block in nanoseconds.
  uint64_t last_block_duration_;

  // Last timestamp in nanoseconds added to a cluster.
  uint64_t last_timestamp_;

  // Last timestamp in nanoseconds by track number added to a cluster.
  uint64_t last_track_timestamp_[kMaxTrackNumber];

  // Number of frames written per track.
  uint64_t track_frames_written_[kMaxTrackNumber];

  // Maximum time in nanoseconds for a cluster duration. This variable is a
  // guideline and some clusters may have a longer duration. Default is 30
  // seconds.
  uint64_t max_cluster_duration_;

  // Maximum size in bytes for a cluster. This variable is a guideline and
  // some clusters may have a larger size. Default is 0 which signifies that
  // the muxer will decide the size.
  uint64_t max_cluster_size_;

  // The mode that segment is in. If set to |kLive| the writer must not
  // seek backwards.
  Mode mode_;

  // Flag telling the muxer that a new cue point should be added.
  bool new_cuepoint_;

  // TODO(fgalligan): Should we add support for more than one Cues element?
  // Flag whether or not the muxer should output a Cues element.
  bool output_cues_;

  // Flag whether or not the last frame in each Cluster will have a Duration
  // element in it.
  bool accurate_cluster_duration_;

  // Flag whether or not to write the Cluster Timecode using exactly 8 bytes.
  bool fixed_size_cluster_timecode_;

  // Flag whether or not to estimate the file duration.
  bool estimate_file_duration_;

  // The size of the EBML header, used to validate the header if
  // WriteEbmlHeader() is called more than once.
  int32_t ebml_header_size_;

  // The file position of the segment's payload.
  int64_t payload_pos_;

  // The file position of the element's size.
  int64_t size_position_;

  // Current DocTypeVersion (|doc_type_version_|) and that written in
  // WriteSegmentHeader().
  // WriteEbmlHeader() will be called from Finalize() if |doc_type_version_|
  // differs from |doc_type_version_written_|.
  uint32_t doc_type_version_;
  uint32_t doc_type_version_written_;

  // If |duration_| is > 0, then explicitly set the duration of the segment.
  double duration_;

  // Pointer to the writer objects. Not owned by this class.
  IMkvWriter* writer_cluster_;
  IMkvWriter* writer_cues_;
  IMkvWriter* writer_header_;

  LIBWEBM_DISALLOW_COPY_AND_ASSIGN(Segment);
};

}  // namespace mkvmuxer

#endif  // MKVMUXER_MKVMUXER_H_
