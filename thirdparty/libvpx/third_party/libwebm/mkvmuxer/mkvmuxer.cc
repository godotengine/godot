// Copyright (c) 2012 The WebM project authors. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the LICENSE file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS.  All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.

#include "mkvmuxer/mkvmuxer.h"

#include <stdint.h>

#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "common/webmids.h"
#include "mkvmuxer/mkvmuxerutil.h"
#include "mkvmuxer/mkvwriter.h"
#include "mkvparser/mkvparser.h"

namespace mkvmuxer {

const float PrimaryChromaticity::kChromaticityMin = 0.0f;
const float PrimaryChromaticity::kChromaticityMax = 1.0f;
const float MasteringMetadata::kMinLuminance = 0.0f;
const float MasteringMetadata::kMinLuminanceMax = 999.99f;
const float MasteringMetadata::kMaxLuminanceMax = 9999.99f;
const float MasteringMetadata::kValueNotPresent = FLT_MAX;
const uint64_t Colour::kValueNotPresent = UINT64_MAX;

namespace {

const char kDocTypeWebm[] = "webm";
const char kDocTypeMatroska[] = "matroska";

// Deallocate the string designated by |dst|, and then copy the |src|
// string to |dst|.  The caller owns both the |src| string and the
// |dst| copy (hence the caller is responsible for eventually
// deallocating the strings, either directly, or indirectly via
// StrCpy).  Returns true if the source string was successfully copied
// to the destination.
bool StrCpy(const char* src, char** dst_ptr) {
  if (dst_ptr == NULL)
    return false;

  char*& dst = *dst_ptr;

  delete[] dst;
  dst = NULL;

  if (src == NULL)
    return true;

  const size_t size = strlen(src) + 1;

  dst = new (std::nothrow) char[size];  // NOLINT
  if (dst == NULL)
    return false;

  memcpy(dst, src, size - 1);
  dst[size - 1] = '\0';
  return true;
}

typedef std::unique_ptr<PrimaryChromaticity> PrimaryChromaticityPtr;
bool CopyChromaticity(const PrimaryChromaticity* src,
                      PrimaryChromaticityPtr* dst) {
  if (!dst)
    return false;

  dst->reset(new (std::nothrow) PrimaryChromaticity(src->x(), src->y()));
  if (!dst->get())
    return false;

  return true;
}

}  // namespace

///////////////////////////////////////////////////////////////
//
// IMkvWriter Class

IMkvWriter::IMkvWriter() {}

IMkvWriter::~IMkvWriter() {}

bool WriteEbmlHeader(IMkvWriter* writer, uint64_t doc_type_version,
                     const char* const doc_type) {
  // Level 0
  uint64_t size =
      EbmlElementSize(libwebm::kMkvEBMLVersion, static_cast<uint64>(1));
  size += EbmlElementSize(libwebm::kMkvEBMLReadVersion, static_cast<uint64>(1));
  size += EbmlElementSize(libwebm::kMkvEBMLMaxIDLength, static_cast<uint64>(4));
  size +=
      EbmlElementSize(libwebm::kMkvEBMLMaxSizeLength, static_cast<uint64>(8));
  size += EbmlElementSize(libwebm::kMkvDocType, doc_type);
  size += EbmlElementSize(libwebm::kMkvDocTypeVersion,
                          static_cast<uint64>(doc_type_version));
  size +=
      EbmlElementSize(libwebm::kMkvDocTypeReadVersion, static_cast<uint64>(2));

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvEBML, size))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvEBMLVersion,
                        static_cast<uint64>(1))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvEBMLReadVersion,
                        static_cast<uint64>(1))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvEBMLMaxIDLength,
                        static_cast<uint64>(4))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvEBMLMaxSizeLength,
                        static_cast<uint64>(8))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvDocType, doc_type))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvDocTypeVersion,
                        static_cast<uint64>(doc_type_version))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvDocTypeReadVersion,
                        static_cast<uint64>(2))) {
    return false;
  }

  return true;
}

bool WriteEbmlHeader(IMkvWriter* writer, uint64_t doc_type_version) {
  return WriteEbmlHeader(writer, doc_type_version, kDocTypeWebm);
}

bool WriteEbmlHeader(IMkvWriter* writer) {
  return WriteEbmlHeader(writer, mkvmuxer::Segment::kDefaultDocTypeVersion);
}

bool ChunkedCopy(mkvparser::IMkvReader* source, mkvmuxer::IMkvWriter* dst,
                 int64_t start, int64_t size) {
  // TODO(vigneshv): Check if this is a reasonable value.
  const uint32_t kBufSize = 2048;
  uint8_t* buf = new uint8_t[kBufSize];
  int64_t offset = start;
  while (size > 0) {
    const int64_t read_len = (size > kBufSize) ? kBufSize : size;
    if (source->Read(offset, static_cast<long>(read_len), buf))
      return false;
    dst->Write(buf, static_cast<uint32_t>(read_len));
    offset += read_len;
    size -= read_len;
  }
  delete[] buf;
  return true;
}

///////////////////////////////////////////////////////////////
//
// Frame Class

Frame::Frame()
    : add_id_(0),
      additional_(NULL),
      additional_length_(0),
      duration_(0),
      duration_set_(false),
      frame_(NULL),
      is_key_(false),
      length_(0),
      track_number_(0),
      timestamp_(0),
      discard_padding_(0),
      reference_block_timestamp_(0),
      reference_block_timestamp_set_(false) {}

Frame::~Frame() {
  delete[] frame_;
  delete[] additional_;
}

bool Frame::CopyFrom(const Frame& frame) {
  delete[] frame_;
  frame_ = NULL;
  length_ = 0;
  if (frame.length() > 0 && frame.frame() != NULL &&
      !Init(frame.frame(), frame.length())) {
    return false;
  }
  add_id_ = 0;
  delete[] additional_;
  additional_ = NULL;
  additional_length_ = 0;
  if (frame.additional_length() > 0 && frame.additional() != NULL &&
      !AddAdditionalData(frame.additional(), frame.additional_length(),
                         frame.add_id())) {
    return false;
  }
  duration_ = frame.duration();
  duration_set_ = frame.duration_set();
  is_key_ = frame.is_key();
  track_number_ = frame.track_number();
  timestamp_ = frame.timestamp();
  discard_padding_ = frame.discard_padding();
  reference_block_timestamp_ = frame.reference_block_timestamp();
  reference_block_timestamp_set_ = frame.reference_block_timestamp_set();
  return true;
}

bool Frame::Init(const uint8_t* frame, uint64_t length) {
  uint8_t* const data =
      new (std::nothrow) uint8_t[static_cast<size_t>(length)];  // NOLINT
  if (!data)
    return false;

  delete[] frame_;
  frame_ = data;
  length_ = length;

  memcpy(frame_, frame, static_cast<size_t>(length_));
  return true;
}

bool Frame::AddAdditionalData(const uint8_t* additional, uint64_t length,
                              uint64_t add_id) {
  uint8_t* const data =
      new (std::nothrow) uint8_t[static_cast<size_t>(length)];  // NOLINT
  if (!data)
    return false;

  delete[] additional_;
  additional_ = data;
  additional_length_ = length;
  add_id_ = add_id;

  memcpy(additional_, additional, static_cast<size_t>(additional_length_));
  return true;
}

bool Frame::IsValid() const {
  if (length_ == 0 || !frame_) {
    return false;
  }
  if ((additional_length_ != 0 && !additional_) ||
      (additional_ != NULL && additional_length_ == 0)) {
    return false;
  }
  if (track_number_ == 0 || track_number_ > kMaxTrackNumber) {
    return false;
  }
  if (!CanBeSimpleBlock() && !is_key_ && !reference_block_timestamp_set_) {
    return false;
  }
  return true;
}

bool Frame::CanBeSimpleBlock() const {
  return additional_ == NULL && discard_padding_ == 0 && duration_ == 0;
}

void Frame::set_duration(uint64_t duration) {
  duration_ = duration;
  duration_set_ = true;
}

void Frame::set_reference_block_timestamp(int64_t reference_block_timestamp) {
  reference_block_timestamp_ = reference_block_timestamp;
  reference_block_timestamp_set_ = true;
}

///////////////////////////////////////////////////////////////
//
// CuePoint Class

CuePoint::CuePoint()
    : time_(0),
      track_(0),
      cluster_pos_(0),
      block_number_(1),
      output_block_number_(true) {}

CuePoint::~CuePoint() {}

bool CuePoint::Write(IMkvWriter* writer) const {
  if (!writer || track_ < 1 || cluster_pos_ < 1)
    return false;

  uint64_t size = EbmlElementSize(libwebm::kMkvCueClusterPosition,
                                  static_cast<uint64>(cluster_pos_));
  size += EbmlElementSize(libwebm::kMkvCueTrack, static_cast<uint64>(track_));
  if (output_block_number_ && block_number_ > 1)
    size += EbmlElementSize(libwebm::kMkvCueBlockNumber,
                            static_cast<uint64>(block_number_));
  const uint64_t track_pos_size =
      EbmlMasterElementSize(libwebm::kMkvCueTrackPositions, size) + size;
  const uint64_t payload_size =
      EbmlElementSize(libwebm::kMkvCueTime, static_cast<uint64>(time_)) +
      track_pos_size;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvCuePoint, payload_size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvCueTime,
                        static_cast<uint64>(time_))) {
    return false;
  }

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvCueTrackPositions, size))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvCueTrack,
                        static_cast<uint64>(track_))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvCueClusterPosition,
                        static_cast<uint64>(cluster_pos_))) {
    return false;
  }
  if (output_block_number_ && block_number_ > 1) {
    if (!WriteEbmlElement(writer, libwebm::kMkvCueBlockNumber,
                          static_cast<uint64>(block_number_))) {
      return false;
    }
  }

  const int64_t stop_position = writer->Position();
  if (stop_position < 0)
    return false;

  if (stop_position - payload_position != static_cast<int64_t>(payload_size))
    return false;

  return true;
}

uint64_t CuePoint::PayloadSize() const {
  uint64_t size = EbmlElementSize(libwebm::kMkvCueClusterPosition,
                                  static_cast<uint64>(cluster_pos_));
  size += EbmlElementSize(libwebm::kMkvCueTrack, static_cast<uint64>(track_));
  if (output_block_number_ && block_number_ > 1)
    size += EbmlElementSize(libwebm::kMkvCueBlockNumber,
                            static_cast<uint64>(block_number_));
  const uint64_t track_pos_size =
      EbmlMasterElementSize(libwebm::kMkvCueTrackPositions, size) + size;
  const uint64_t payload_size =
      EbmlElementSize(libwebm::kMkvCueTime, static_cast<uint64>(time_)) +
      track_pos_size;

  return payload_size;
}

uint64_t CuePoint::Size() const {
  const uint64_t payload_size = PayloadSize();
  return EbmlMasterElementSize(libwebm::kMkvCuePoint, payload_size) +
         payload_size;
}

///////////////////////////////////////////////////////////////
//
// Cues Class

Cues::Cues()
    : cue_entries_capacity_(0),
      cue_entries_size_(0),
      cue_entries_(NULL),
      output_block_number_(true) {}

Cues::~Cues() {
  if (cue_entries_) {
    for (int32_t i = 0; i < cue_entries_size_; ++i) {
      CuePoint* const cue = cue_entries_[i];
      delete cue;
    }
    delete[] cue_entries_;
  }
}

bool Cues::AddCue(CuePoint* cue) {
  if (!cue)
    return false;

  if ((cue_entries_size_ + 1) > cue_entries_capacity_) {
    // Add more CuePoints.
    const int32_t new_capacity =
        (!cue_entries_capacity_) ? 2 : cue_entries_capacity_ * 2;

    if (new_capacity < 1)
      return false;

    CuePoint** const cues =
        new (std::nothrow) CuePoint*[new_capacity];  // NOLINT
    if (!cues)
      return false;

    for (int32_t i = 0; i < cue_entries_size_; ++i) {
      cues[i] = cue_entries_[i];
    }

    delete[] cue_entries_;

    cue_entries_ = cues;
    cue_entries_capacity_ = new_capacity;
  }

  cue->set_output_block_number(output_block_number_);
  cue_entries_[cue_entries_size_++] = cue;
  return true;
}

CuePoint* Cues::GetCueByIndex(int32_t index) const {
  if (cue_entries_ == NULL)
    return NULL;

  if (index >= cue_entries_size_)
    return NULL;

  return cue_entries_[index];
}

uint64_t Cues::Size() {
  uint64_t size = 0;
  for (int32_t i = 0; i < cue_entries_size_; ++i)
    size += GetCueByIndex(i)->Size();
  size += EbmlMasterElementSize(libwebm::kMkvCues, size);
  return size;
}

bool Cues::Write(IMkvWriter* writer) const {
  if (!writer)
    return false;

  uint64_t size = 0;
  for (int32_t i = 0; i < cue_entries_size_; ++i) {
    const CuePoint* const cue = GetCueByIndex(i);

    if (!cue)
      return false;

    size += cue->Size();
  }

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvCues, size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  for (int32_t i = 0; i < cue_entries_size_; ++i) {
    const CuePoint* const cue = GetCueByIndex(i);

    if (!cue->Write(writer))
      return false;
  }

  const int64_t stop_position = writer->Position();
  if (stop_position < 0)
    return false;

  if (stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  return true;
}

///////////////////////////////////////////////////////////////
//
// ContentEncAESSettings Class

ContentEncAESSettings::ContentEncAESSettings() : cipher_mode_(kCTR) {}

uint64_t ContentEncAESSettings::Size() const {
  const uint64_t payload = PayloadSize();
  const uint64_t size =
      EbmlMasterElementSize(libwebm::kMkvContentEncAESSettings, payload) +
      payload;
  return size;
}

bool ContentEncAESSettings::Write(IMkvWriter* writer) const {
  const uint64_t payload = PayloadSize();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvContentEncAESSettings,
                              payload))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvAESSettingsCipherMode,
                        static_cast<uint64>(cipher_mode_))) {
    return false;
  }

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(payload))
    return false;

  return true;
}

uint64_t ContentEncAESSettings::PayloadSize() const {
  uint64_t size = EbmlElementSize(libwebm::kMkvAESSettingsCipherMode,
                                  static_cast<uint64>(cipher_mode_));
  return size;
}

///////////////////////////////////////////////////////////////
//
// ContentEncoding Class

ContentEncoding::ContentEncoding()
    : enc_algo_(5),
      enc_key_id_(NULL),
      encoding_order_(0),
      encoding_scope_(1),
      encoding_type_(1),
      enc_key_id_length_(0) {}

ContentEncoding::~ContentEncoding() { delete[] enc_key_id_; }

bool ContentEncoding::SetEncryptionID(const uint8_t* id, uint64_t length) {
  if (!id || length < 1)
    return false;

  delete[] enc_key_id_;

  enc_key_id_ =
      new (std::nothrow) uint8_t[static_cast<size_t>(length)];  // NOLINT
  if (!enc_key_id_)
    return false;

  memcpy(enc_key_id_, id, static_cast<size_t>(length));
  enc_key_id_length_ = length;

  return true;
}

uint64_t ContentEncoding::Size() const {
  const uint64_t encryption_size = EncryptionSize();
  const uint64_t encoding_size = EncodingSize(0, encryption_size);
  const uint64_t encodings_size =
      EbmlMasterElementSize(libwebm::kMkvContentEncoding, encoding_size) +
      encoding_size;

  return encodings_size;
}

bool ContentEncoding::Write(IMkvWriter* writer) const {
  const uint64_t encryption_size = EncryptionSize();
  const uint64_t encoding_size = EncodingSize(0, encryption_size);
  const uint64_t size =
      EbmlMasterElementSize(libwebm::kMkvContentEncoding, encoding_size) +
      encoding_size;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvContentEncoding,
                              encoding_size))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvContentEncodingOrder,
                        static_cast<uint64>(encoding_order_)))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvContentEncodingScope,
                        static_cast<uint64>(encoding_scope_)))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvContentEncodingType,
                        static_cast<uint64>(encoding_type_)))
    return false;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvContentEncryption,
                              encryption_size))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvContentEncAlgo,
                        static_cast<uint64>(enc_algo_))) {
    return false;
  }
  if (!WriteEbmlElement(writer, libwebm::kMkvContentEncKeyID, enc_key_id_,
                        enc_key_id_length_))
    return false;

  if (!enc_aes_settings_.Write(writer))
    return false;

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  return true;
}

uint64_t ContentEncoding::EncodingSize(uint64_t compression_size,
                                       uint64_t encryption_size) const {
  // TODO(fgalligan): Add support for compression settings.
  if (compression_size != 0)
    return 0;

  uint64_t encoding_size = 0;

  if (encryption_size > 0) {
    encoding_size +=
        EbmlMasterElementSize(libwebm::kMkvContentEncryption, encryption_size) +
        encryption_size;
  }
  encoding_size += EbmlElementSize(libwebm::kMkvContentEncodingType,
                                   static_cast<uint64>(encoding_type_));
  encoding_size += EbmlElementSize(libwebm::kMkvContentEncodingScope,
                                   static_cast<uint64>(encoding_scope_));
  encoding_size += EbmlElementSize(libwebm::kMkvContentEncodingOrder,
                                   static_cast<uint64>(encoding_order_));

  return encoding_size;
}

uint64_t ContentEncoding::EncryptionSize() const {
  const uint64_t aes_size = enc_aes_settings_.Size();

  uint64_t encryption_size = EbmlElementSize(libwebm::kMkvContentEncKeyID,
                                             enc_key_id_, enc_key_id_length_);
  encryption_size += EbmlElementSize(libwebm::kMkvContentEncAlgo,
                                     static_cast<uint64>(enc_algo_));

  return encryption_size + aes_size;
}

///////////////////////////////////////////////////////////////
//
// Track Class

Track::Track(unsigned int* seed)
    : codec_id_(NULL),
      codec_private_(NULL),
      language_(NULL),
      max_block_additional_id_(0),
      name_(NULL),
      number_(0),
      type_(0),
      uid_(MakeUID(seed)),
      codec_delay_(0),
      seek_pre_roll_(0),
      default_duration_(0),
      codec_private_length_(0),
      content_encoding_entries_(NULL),
      content_encoding_entries_size_(0) {}

Track::~Track() {
  delete[] codec_id_;
  delete[] codec_private_;
  delete[] language_;
  delete[] name_;

  if (content_encoding_entries_) {
    for (uint32_t i = 0; i < content_encoding_entries_size_; ++i) {
      ContentEncoding* const encoding = content_encoding_entries_[i];
      delete encoding;
    }
    delete[] content_encoding_entries_;
  }
}

bool Track::AddContentEncoding() {
  const uint32_t count = content_encoding_entries_size_ + 1;

  ContentEncoding** const content_encoding_entries =
      new (std::nothrow) ContentEncoding*[count];  // NOLINT
  if (!content_encoding_entries)
    return false;

  ContentEncoding* const content_encoding =
      new (std::nothrow) ContentEncoding();  // NOLINT
  if (!content_encoding) {
    delete[] content_encoding_entries;
    return false;
  }

  for (uint32_t i = 0; i < content_encoding_entries_size_; ++i) {
    content_encoding_entries[i] = content_encoding_entries_[i];
  }

  delete[] content_encoding_entries_;

  content_encoding_entries_ = content_encoding_entries;
  content_encoding_entries_[content_encoding_entries_size_] = content_encoding;
  content_encoding_entries_size_ = count;
  return true;
}

ContentEncoding* Track::GetContentEncodingByIndex(uint32_t index) const {
  if (content_encoding_entries_ == NULL)
    return NULL;

  if (index >= content_encoding_entries_size_)
    return NULL;

  return content_encoding_entries_[index];
}

uint64_t Track::PayloadSize() const {
  uint64_t size =
      EbmlElementSize(libwebm::kMkvTrackNumber, static_cast<uint64>(number_));
  size += EbmlElementSize(libwebm::kMkvTrackUID, static_cast<uint64>(uid_));
  size += EbmlElementSize(libwebm::kMkvTrackType, static_cast<uint64>(type_));
  if (codec_id_)
    size += EbmlElementSize(libwebm::kMkvCodecID, codec_id_);
  if (codec_private_)
    size += EbmlElementSize(libwebm::kMkvCodecPrivate, codec_private_,
                            codec_private_length_);
  if (language_)
    size += EbmlElementSize(libwebm::kMkvLanguage, language_);
  if (name_)
    size += EbmlElementSize(libwebm::kMkvName, name_);
  if (max_block_additional_id_) {
    size += EbmlElementSize(libwebm::kMkvMaxBlockAdditionID,
                            static_cast<uint64>(max_block_additional_id_));
  }
  if (codec_delay_) {
    size += EbmlElementSize(libwebm::kMkvCodecDelay,
                            static_cast<uint64>(codec_delay_));
  }
  if (seek_pre_roll_) {
    size += EbmlElementSize(libwebm::kMkvSeekPreRoll,
                            static_cast<uint64>(seek_pre_roll_));
  }
  if (default_duration_) {
    size += EbmlElementSize(libwebm::kMkvDefaultDuration,
                            static_cast<uint64>(default_duration_));
  }

  if (content_encoding_entries_size_ > 0) {
    uint64_t content_encodings_size = 0;
    for (uint32_t i = 0; i < content_encoding_entries_size_; ++i) {
      ContentEncoding* const encoding = content_encoding_entries_[i];
      content_encodings_size += encoding->Size();
    }

    size += EbmlMasterElementSize(libwebm::kMkvContentEncodings,
                                  content_encodings_size) +
            content_encodings_size;
  }

  return size;
}

uint64_t Track::Size() const {
  uint64_t size = PayloadSize();
  size += EbmlMasterElementSize(libwebm::kMkvTrackEntry, size);
  return size;
}

bool Track::Write(IMkvWriter* writer) const {
  if (!writer)
    return false;

  // mandatory elements without a default value.
  if (!type_ || !codec_id_)
    return false;

  // AV1 tracks require a CodecPrivate. See
  // https://github.com/ietf-wg-cellar/matroska-specification/blob/HEAD/codec/av1.md
  // TODO(tomfinegan): Update the above link to the AV1 Matroska mappings to
  // point to a stable version once it is finalized, or our own WebM mappings
  // page on webmproject.org should we decide to release them.
  if (!strcmp(codec_id_, Tracks::kAv1CodecId) && !codec_private_)
    return false;

  // |size| may be bigger than what is written out in this function because
  // derived classes may write out more data in the Track element.
  const uint64_t payload_size = PayloadSize();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvTrackEntry, payload_size))
    return false;

  uint64_t size =
      EbmlElementSize(libwebm::kMkvTrackNumber, static_cast<uint64>(number_));
  size += EbmlElementSize(libwebm::kMkvTrackUID, static_cast<uint64>(uid_));
  size += EbmlElementSize(libwebm::kMkvTrackType, static_cast<uint64>(type_));
  if (codec_id_)
    size += EbmlElementSize(libwebm::kMkvCodecID, codec_id_);
  if (codec_private_)
    size += EbmlElementSize(libwebm::kMkvCodecPrivate, codec_private_,
                            static_cast<uint64>(codec_private_length_));
  if (language_)
    size += EbmlElementSize(libwebm::kMkvLanguage, language_);
  if (name_)
    size += EbmlElementSize(libwebm::kMkvName, name_);
  if (max_block_additional_id_)
    size += EbmlElementSize(libwebm::kMkvMaxBlockAdditionID,
                            static_cast<uint64>(max_block_additional_id_));
  if (codec_delay_)
    size += EbmlElementSize(libwebm::kMkvCodecDelay,
                            static_cast<uint64>(codec_delay_));
  if (seek_pre_roll_)
    size += EbmlElementSize(libwebm::kMkvSeekPreRoll,
                            static_cast<uint64>(seek_pre_roll_));
  if (default_duration_)
    size += EbmlElementSize(libwebm::kMkvDefaultDuration,
                            static_cast<uint64>(default_duration_));

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvTrackNumber,
                        static_cast<uint64>(number_)))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvTrackUID,
                        static_cast<uint64>(uid_)))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvTrackType,
                        static_cast<uint64>(type_)))
    return false;
  if (max_block_additional_id_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvMaxBlockAdditionID,
                          static_cast<uint64>(max_block_additional_id_))) {
      return false;
    }
  }
  if (codec_delay_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvCodecDelay,
                          static_cast<uint64>(codec_delay_)))
      return false;
  }
  if (seek_pre_roll_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvSeekPreRoll,
                          static_cast<uint64>(seek_pre_roll_)))
      return false;
  }
  if (default_duration_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvDefaultDuration,
                          static_cast<uint64>(default_duration_)))
      return false;
  }
  if (codec_id_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvCodecID, codec_id_))
      return false;
  }
  if (codec_private_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvCodecPrivate, codec_private_,
                          static_cast<uint64>(codec_private_length_)))
      return false;
  }
  if (language_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvLanguage, language_))
      return false;
  }
  if (name_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvName, name_))
      return false;
  }

  int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  if (content_encoding_entries_size_ > 0) {
    uint64_t content_encodings_size = 0;
    for (uint32_t i = 0; i < content_encoding_entries_size_; ++i) {
      ContentEncoding* const encoding = content_encoding_entries_[i];
      content_encodings_size += encoding->Size();
    }

    if (!WriteEbmlMasterElement(writer, libwebm::kMkvContentEncodings,
                                content_encodings_size))
      return false;

    for (uint32_t i = 0; i < content_encoding_entries_size_; ++i) {
      ContentEncoding* const encoding = content_encoding_entries_[i];
      if (!encoding->Write(writer))
        return false;
    }
  }

  stop_position = writer->Position();
  if (stop_position < 0)
    return false;
  return true;
}

bool Track::SetCodecPrivate(const uint8_t* codec_private, uint64_t length) {
  if (!codec_private || length < 1)
    return false;

  delete[] codec_private_;

  codec_private_ =
      new (std::nothrow) uint8_t[static_cast<size_t>(length)];  // NOLINT
  if (!codec_private_)
    return false;

  memcpy(codec_private_, codec_private, static_cast<size_t>(length));
  codec_private_length_ = length;

  return true;
}

void Track::set_codec_id(const char* codec_id) {
  if (codec_id) {
    delete[] codec_id_;

    const size_t length = strlen(codec_id) + 1;
    codec_id_ = new (std::nothrow) char[length];  // NOLINT
    if (codec_id_) {
      memcpy(codec_id_, codec_id, length - 1);
      codec_id_[length - 1] = '\0';
    }
  }
}

// TODO(fgalligan): Vet the language parameter.
void Track::set_language(const char* language) {
  if (language) {
    delete[] language_;

    const size_t length = strlen(language) + 1;
    language_ = new (std::nothrow) char[length];  // NOLINT
    if (language_) {
      memcpy(language_, language, length - 1);
      language_[length - 1] = '\0';
    }
  }
}

void Track::set_name(const char* name) {
  if (name) {
    delete[] name_;

    const size_t length = strlen(name) + 1;
    name_ = new (std::nothrow) char[length];  // NOLINT
    if (name_) {
      memcpy(name_, name, length - 1);
      name_[length - 1] = '\0';
    }
  }
}

///////////////////////////////////////////////////////////////
//
// Colour and its child elements

uint64_t PrimaryChromaticity::PrimaryChromaticitySize(
    libwebm::MkvId x_id, libwebm::MkvId y_id) const {
  return EbmlElementSize(x_id, x_) + EbmlElementSize(y_id, y_);
}

bool PrimaryChromaticity::Write(IMkvWriter* writer, libwebm::MkvId x_id,
                                libwebm::MkvId y_id) const {
  if (!Valid()) {
    return false;
  }
  return WriteEbmlElement(writer, x_id, x_) &&
         WriteEbmlElement(writer, y_id, y_);
}

bool PrimaryChromaticity::Valid() const {
  return (x_ >= kChromaticityMin && x_ <= kChromaticityMax &&
          y_ >= kChromaticityMin && y_ <= kChromaticityMax);
}

uint64_t MasteringMetadata::MasteringMetadataSize() const {
  uint64_t size = PayloadSize();

  if (size > 0)
    size += EbmlMasterElementSize(libwebm::kMkvMasteringMetadata, size);

  return size;
}

bool MasteringMetadata::Valid() const {
  if (luminance_min_ != kValueNotPresent) {
    if (luminance_min_ < kMinLuminance || luminance_min_ > kMinLuminanceMax ||
        luminance_min_ > luminance_max_) {
      return false;
    }
  }
  if (luminance_max_ != kValueNotPresent) {
    if (luminance_max_ < kMinLuminance || luminance_max_ > kMaxLuminanceMax ||
        luminance_max_ < luminance_min_) {
      return false;
    }
  }
  if (r_ && !r_->Valid())
    return false;
  if (g_ && !g_->Valid())
    return false;
  if (b_ && !b_->Valid())
    return false;
  if (white_point_ && !white_point_->Valid())
    return false;

  return true;
}

bool MasteringMetadata::Write(IMkvWriter* writer) const {
  const uint64_t size = PayloadSize();

  // Don't write an empty element.
  if (size == 0)
    return true;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvMasteringMetadata, size))
    return false;
  if (luminance_max_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvLuminanceMax, luminance_max_)) {
    return false;
  }
  if (luminance_min_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvLuminanceMin, luminance_min_)) {
    return false;
  }
  if (r_ && !r_->Write(writer, libwebm::kMkvPrimaryRChromaticityX,
                       libwebm::kMkvPrimaryRChromaticityY)) {
    return false;
  }
  if (g_ && !g_->Write(writer, libwebm::kMkvPrimaryGChromaticityX,
                       libwebm::kMkvPrimaryGChromaticityY)) {
    return false;
  }
  if (b_ && !b_->Write(writer, libwebm::kMkvPrimaryBChromaticityX,
                       libwebm::kMkvPrimaryBChromaticityY)) {
    return false;
  }
  if (white_point_ &&
      !white_point_->Write(writer, libwebm::kMkvWhitePointChromaticityX,
                           libwebm::kMkvWhitePointChromaticityY)) {
    return false;
  }

  return true;
}

bool MasteringMetadata::SetChromaticity(
    const PrimaryChromaticity* r, const PrimaryChromaticity* g,
    const PrimaryChromaticity* b, const PrimaryChromaticity* white_point) {
  PrimaryChromaticityPtr r_ptr(nullptr);
  if (r) {
    if (!CopyChromaticity(r, &r_ptr))
      return false;
  }
  PrimaryChromaticityPtr g_ptr(nullptr);
  if (g) {
    if (!CopyChromaticity(g, &g_ptr))
      return false;
  }
  PrimaryChromaticityPtr b_ptr(nullptr);
  if (b) {
    if (!CopyChromaticity(b, &b_ptr))
      return false;
  }
  PrimaryChromaticityPtr wp_ptr(nullptr);
  if (white_point) {
    if (!CopyChromaticity(white_point, &wp_ptr))
      return false;
  }

  r_ = r_ptr.release();
  g_ = g_ptr.release();
  b_ = b_ptr.release();
  white_point_ = wp_ptr.release();
  return true;
}

uint64_t MasteringMetadata::PayloadSize() const {
  uint64_t size = 0;

  if (luminance_max_ != kValueNotPresent)
    size += EbmlElementSize(libwebm::kMkvLuminanceMax, luminance_max_);
  if (luminance_min_ != kValueNotPresent)
    size += EbmlElementSize(libwebm::kMkvLuminanceMin, luminance_min_);

  if (r_) {
    size += r_->PrimaryChromaticitySize(libwebm::kMkvPrimaryRChromaticityX,
                                        libwebm::kMkvPrimaryRChromaticityY);
  }
  if (g_) {
    size += g_->PrimaryChromaticitySize(libwebm::kMkvPrimaryGChromaticityX,
                                        libwebm::kMkvPrimaryGChromaticityY);
  }
  if (b_) {
    size += b_->PrimaryChromaticitySize(libwebm::kMkvPrimaryBChromaticityX,
                                        libwebm::kMkvPrimaryBChromaticityY);
  }
  if (white_point_) {
    size += white_point_->PrimaryChromaticitySize(
        libwebm::kMkvWhitePointChromaticityX,
        libwebm::kMkvWhitePointChromaticityY);
  }

  return size;
}

uint64_t Colour::ColourSize() const {
  uint64_t size = PayloadSize();

  if (size > 0)
    size += EbmlMasterElementSize(libwebm::kMkvColour, size);

  return size;
}

bool Colour::Valid() const {
  if (mastering_metadata_ && !mastering_metadata_->Valid())
    return false;
  if (matrix_coefficients_ != kValueNotPresent &&
      !IsMatrixCoefficientsValueValid(matrix_coefficients_)) {
    return false;
  }
  if (chroma_siting_horz_ != kValueNotPresent &&
      !IsChromaSitingHorzValueValid(chroma_siting_horz_)) {
    return false;
  }
  if (chroma_siting_vert_ != kValueNotPresent &&
      !IsChromaSitingVertValueValid(chroma_siting_vert_)) {
    return false;
  }
  if (range_ != kValueNotPresent && !IsColourRangeValueValid(range_))
    return false;
  if (transfer_characteristics_ != kValueNotPresent &&
      !IsTransferCharacteristicsValueValid(transfer_characteristics_)) {
    return false;
  }
  if (primaries_ != kValueNotPresent && !IsPrimariesValueValid(primaries_))
    return false;

  return true;
}

bool Colour::Write(IMkvWriter* writer) const {
  const uint64_t size = PayloadSize();

  // Don't write an empty element.
  if (size == 0)
    return true;

  // Don't write an invalid element.
  if (!Valid())
    return false;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvColour, size))
    return false;

  if (matrix_coefficients_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvMatrixCoefficients,
                        static_cast<uint64>(matrix_coefficients_))) {
    return false;
  }
  if (bits_per_channel_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvBitsPerChannel,
                        static_cast<uint64>(bits_per_channel_))) {
    return false;
  }
  if (chroma_subsampling_horz_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvChromaSubsamplingHorz,
                        static_cast<uint64>(chroma_subsampling_horz_))) {
    return false;
  }
  if (chroma_subsampling_vert_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvChromaSubsamplingVert,
                        static_cast<uint64>(chroma_subsampling_vert_))) {
    return false;
  }

  if (cb_subsampling_horz_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvCbSubsamplingHorz,
                        static_cast<uint64>(cb_subsampling_horz_))) {
    return false;
  }
  if (cb_subsampling_vert_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvCbSubsamplingVert,
                        static_cast<uint64>(cb_subsampling_vert_))) {
    return false;
  }
  if (chroma_siting_horz_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvChromaSitingHorz,
                        static_cast<uint64>(chroma_siting_horz_))) {
    return false;
  }
  if (chroma_siting_vert_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvChromaSitingVert,
                        static_cast<uint64>(chroma_siting_vert_))) {
    return false;
  }
  if (range_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvRange,
                        static_cast<uint64>(range_))) {
    return false;
  }
  if (transfer_characteristics_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvTransferCharacteristics,
                        static_cast<uint64>(transfer_characteristics_))) {
    return false;
  }
  if (primaries_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvPrimaries,
                        static_cast<uint64>(primaries_))) {
    return false;
  }
  if (max_cll_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvMaxCLL,
                        static_cast<uint64>(max_cll_))) {
    return false;
  }
  if (max_fall_ != kValueNotPresent &&
      !WriteEbmlElement(writer, libwebm::kMkvMaxFALL,
                        static_cast<uint64>(max_fall_))) {
    return false;
  }

  if (mastering_metadata_ && !mastering_metadata_->Write(writer))
    return false;

  return true;
}

bool Colour::SetMasteringMetadata(const MasteringMetadata& mastering_metadata) {
  std::unique_ptr<MasteringMetadata> mm_ptr(new MasteringMetadata());
  if (!mm_ptr.get())
    return false;

  mm_ptr->set_luminance_max(mastering_metadata.luminance_max());
  mm_ptr->set_luminance_min(mastering_metadata.luminance_min());

  if (!mm_ptr->SetChromaticity(mastering_metadata.r(), mastering_metadata.g(),
                               mastering_metadata.b(),
                               mastering_metadata.white_point())) {
    return false;
  }

  delete mastering_metadata_;
  mastering_metadata_ = mm_ptr.release();
  return true;
}

uint64_t Colour::PayloadSize() const {
  uint64_t size = 0;

  if (matrix_coefficients_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvMatrixCoefficients,
                            static_cast<uint64>(matrix_coefficients_));
  }
  if (bits_per_channel_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvBitsPerChannel,
                            static_cast<uint64>(bits_per_channel_));
  }
  if (chroma_subsampling_horz_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvChromaSubsamplingHorz,
                            static_cast<uint64>(chroma_subsampling_horz_));
  }
  if (chroma_subsampling_vert_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvChromaSubsamplingVert,
                            static_cast<uint64>(chroma_subsampling_vert_));
  }
  if (cb_subsampling_horz_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvCbSubsamplingHorz,
                            static_cast<uint64>(cb_subsampling_horz_));
  }
  if (cb_subsampling_vert_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvCbSubsamplingVert,
                            static_cast<uint64>(cb_subsampling_vert_));
  }
  if (chroma_siting_horz_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvChromaSitingHorz,
                            static_cast<uint64>(chroma_siting_horz_));
  }
  if (chroma_siting_vert_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvChromaSitingVert,
                            static_cast<uint64>(chroma_siting_vert_));
  }
  if (range_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvRange, static_cast<uint64>(range_));
  }
  if (transfer_characteristics_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvTransferCharacteristics,
                            static_cast<uint64>(transfer_characteristics_));
  }
  if (primaries_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvPrimaries,
                            static_cast<uint64>(primaries_));
  }
  if (max_cll_ != kValueNotPresent) {
    size += EbmlElementSize(libwebm::kMkvMaxCLL, static_cast<uint64>(max_cll_));
  }
  if (max_fall_ != kValueNotPresent) {
    size +=
        EbmlElementSize(libwebm::kMkvMaxFALL, static_cast<uint64>(max_fall_));
  }

  if (mastering_metadata_)
    size += mastering_metadata_->MasteringMetadataSize();

  return size;
}

///////////////////////////////////////////////////////////////
//
// Projection element

uint64_t Projection::ProjectionSize() const {
  uint64_t size = PayloadSize();

  if (size > 0)
    size += EbmlMasterElementSize(libwebm::kMkvProjection, size);

  return size;
}

bool Projection::Write(IMkvWriter* writer) const {
  const uint64_t size = PayloadSize();

  // Don't write an empty element.
  if (size == 0)
    return true;

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvProjection, size))
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvProjectionType,
                        static_cast<uint64>(type_))) {
    return false;
  }

  if (private_data_length_ > 0 && private_data_ != NULL &&
      !WriteEbmlElement(writer, libwebm::kMkvProjectionPrivate, private_data_,
                        private_data_length_)) {
    return false;
  }

  if (!WriteEbmlElement(writer, libwebm::kMkvProjectionPoseYaw, pose_yaw_))
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvProjectionPosePitch,
                        pose_pitch_)) {
    return false;
  }

  if (!WriteEbmlElement(writer, libwebm::kMkvProjectionPoseRoll, pose_roll_)) {
    return false;
  }

  return true;
}

bool Projection::SetProjectionPrivate(const uint8_t* data,
                                      uint64_t data_length) {
  if (data == NULL || data_length == 0) {
    return false;
  }

  if (data_length != static_cast<size_t>(data_length)) {
    return false;
  }

  uint8_t* new_private_data =
      new (std::nothrow) uint8_t[static_cast<size_t>(data_length)];
  if (new_private_data == NULL) {
    return false;
  }

  delete[] private_data_;
  private_data_ = new_private_data;
  private_data_length_ = data_length;
  memcpy(private_data_, data, static_cast<size_t>(data_length));

  return true;
}

uint64_t Projection::PayloadSize() const {
  uint64_t size =
      EbmlElementSize(libwebm::kMkvProjection, static_cast<uint64>(type_));

  if (private_data_length_ > 0 && private_data_ != NULL) {
    size += EbmlElementSize(libwebm::kMkvProjectionPrivate, private_data_,
                            private_data_length_);
  }

  size += EbmlElementSize(libwebm::kMkvProjectionPoseYaw, pose_yaw_);
  size += EbmlElementSize(libwebm::kMkvProjectionPosePitch, pose_pitch_);
  size += EbmlElementSize(libwebm::kMkvProjectionPoseRoll, pose_roll_);

  return size;
}

///////////////////////////////////////////////////////////////
//
// VideoTrack Class

VideoTrack::VideoTrack(unsigned int* seed)
    : Track(seed),
      display_height_(0),
      display_width_(0),
      pixel_height_(0),
      pixel_width_(0),
      crop_left_(0),
      crop_right_(0),
      crop_top_(0),
      crop_bottom_(0),
      frame_rate_(0.0),
      height_(0),
      stereo_mode_(0),
      alpha_mode_(0),
      width_(0),
      colour_space_(NULL),
      colour_(NULL),
      projection_(NULL) {}

VideoTrack::~VideoTrack() {
  delete colour_;
  delete projection_;
}

bool VideoTrack::SetStereoMode(uint64_t stereo_mode) {
  if (stereo_mode != kMono && stereo_mode != kSideBySideLeftIsFirst &&
      stereo_mode != kTopBottomRightIsFirst &&
      stereo_mode != kTopBottomLeftIsFirst &&
      stereo_mode != kSideBySideRightIsFirst)
    return false;

  stereo_mode_ = stereo_mode;
  return true;
}

bool VideoTrack::SetAlphaMode(uint64_t alpha_mode) {
  if (alpha_mode != kNoAlpha && alpha_mode != kAlpha)
    return false;

  alpha_mode_ = alpha_mode;
  return true;
}

uint64_t VideoTrack::PayloadSize() const {
  const uint64_t parent_size = Track::PayloadSize();

  uint64_t size = VideoPayloadSize();
  size += EbmlMasterElementSize(libwebm::kMkvVideo, size);

  return parent_size + size;
}

bool VideoTrack::Write(IMkvWriter* writer) const {
  if (!Track::Write(writer))
    return false;

  const uint64_t size = VideoPayloadSize();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvVideo, size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(
          writer, libwebm::kMkvPixelWidth,
          static_cast<uint64>((pixel_width_ > 0) ? pixel_width_ : width_)))
    return false;
  if (!WriteEbmlElement(
          writer, libwebm::kMkvPixelHeight,
          static_cast<uint64>((pixel_height_ > 0) ? pixel_height_ : height_)))
    return false;
  if (display_width_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvDisplayWidth,
                          static_cast<uint64>(display_width_)))
      return false;
  }
  if (display_height_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvDisplayHeight,
                          static_cast<uint64>(display_height_)))
      return false;
  }
  if (crop_left_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvPixelCropLeft,
                          static_cast<uint64>(crop_left_)))
      return false;
  }
  if (crop_right_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvPixelCropRight,
                          static_cast<uint64>(crop_right_)))
      return false;
  }
  if (crop_top_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvPixelCropTop,
                          static_cast<uint64>(crop_top_)))
      return false;
  }
  if (crop_bottom_ > 0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvPixelCropBottom,
                          static_cast<uint64>(crop_bottom_)))
      return false;
  }
  if (stereo_mode_ > kMono) {
    if (!WriteEbmlElement(writer, libwebm::kMkvStereoMode,
                          static_cast<uint64>(stereo_mode_)))
      return false;
  }
  if (alpha_mode_ > kNoAlpha) {
    if (!WriteEbmlElement(writer, libwebm::kMkvAlphaMode,
                          static_cast<uint64>(alpha_mode_)))
      return false;
  }
  if (colour_space_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvColourSpace, colour_space_))
      return false;
  }
  if (frame_rate_ > 0.0) {
    if (!WriteEbmlElement(writer, libwebm::kMkvFrameRate,
                          static_cast<float>(frame_rate_))) {
      return false;
    }
  }
  if (colour_) {
    if (!colour_->Write(writer))
      return false;
  }
  if (projection_) {
    if (!projection_->Write(writer))
      return false;
  }

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size)) {
    return false;
  }

  return true;
}

void VideoTrack::set_colour_space(const char* colour_space) {
  if (colour_space) {
    delete[] colour_space_;

    const size_t length = strlen(colour_space) + 1;
    colour_space_ = new (std::nothrow) char[length];  // NOLINT
    if (colour_space_) {
      memcpy(colour_space_, colour_space, length - 1);
      colour_space_[length - 1] = '\0';
    }
  }
}

bool VideoTrack::SetColour(const Colour& colour) {
  std::unique_ptr<Colour> colour_ptr(new Colour());
  if (!colour_ptr.get())
    return false;

  if (colour.mastering_metadata()) {
    if (!colour_ptr->SetMasteringMetadata(*colour.mastering_metadata()))
      return false;
  }

  colour_ptr->set_matrix_coefficients(colour.matrix_coefficients());
  colour_ptr->set_bits_per_channel(colour.bits_per_channel());
  colour_ptr->set_chroma_subsampling_horz(colour.chroma_subsampling_horz());
  colour_ptr->set_chroma_subsampling_vert(colour.chroma_subsampling_vert());
  colour_ptr->set_cb_subsampling_horz(colour.cb_subsampling_horz());
  colour_ptr->set_cb_subsampling_vert(colour.cb_subsampling_vert());
  colour_ptr->set_chroma_siting_horz(colour.chroma_siting_horz());
  colour_ptr->set_chroma_siting_vert(colour.chroma_siting_vert());
  colour_ptr->set_range(colour.range());
  colour_ptr->set_transfer_characteristics(colour.transfer_characteristics());
  colour_ptr->set_primaries(colour.primaries());
  colour_ptr->set_max_cll(colour.max_cll());
  colour_ptr->set_max_fall(colour.max_fall());
  delete colour_;
  colour_ = colour_ptr.release();
  return true;
}

bool VideoTrack::SetProjection(const Projection& projection) {
  std::unique_ptr<Projection> projection_ptr(new Projection());
  if (!projection_ptr.get())
    return false;

  if (projection.private_data()) {
    if (!projection_ptr->SetProjectionPrivate(
            projection.private_data(), projection.private_data_length())) {
      return false;
    }
  }

  projection_ptr->set_type(projection.type());
  projection_ptr->set_pose_yaw(projection.pose_yaw());
  projection_ptr->set_pose_pitch(projection.pose_pitch());
  projection_ptr->set_pose_roll(projection.pose_roll());
  delete projection_;
  projection_ = projection_ptr.release();
  return true;
}

uint64_t VideoTrack::VideoPayloadSize() const {
  uint64_t size = EbmlElementSize(
      libwebm::kMkvPixelWidth,
      static_cast<uint64>((pixel_width_ > 0) ? pixel_width_ : width_));
  size += EbmlElementSize(
      libwebm::kMkvPixelHeight,
      static_cast<uint64>((pixel_height_ > 0) ? pixel_height_ : height_));
  if (display_width_ > 0)
    size += EbmlElementSize(libwebm::kMkvDisplayWidth,
                            static_cast<uint64>(display_width_));
  if (display_height_ > 0)
    size += EbmlElementSize(libwebm::kMkvDisplayHeight,
                            static_cast<uint64>(display_height_));
  if (crop_left_ > 0)
    size += EbmlElementSize(libwebm::kMkvPixelCropLeft,
                            static_cast<uint64>(crop_left_));
  if (crop_right_ > 0)
    size += EbmlElementSize(libwebm::kMkvPixelCropRight,
                            static_cast<uint64>(crop_right_));
  if (crop_top_ > 0)
    size += EbmlElementSize(libwebm::kMkvPixelCropTop,
                            static_cast<uint64>(crop_top_));
  if (crop_bottom_ > 0)
    size += EbmlElementSize(libwebm::kMkvPixelCropBottom,
                            static_cast<uint64>(crop_bottom_));
  if (stereo_mode_ > kMono)
    size += EbmlElementSize(libwebm::kMkvStereoMode,
                            static_cast<uint64>(stereo_mode_));
  if (alpha_mode_ > kNoAlpha)
    size += EbmlElementSize(libwebm::kMkvAlphaMode,
                            static_cast<uint64>(alpha_mode_));
  if (frame_rate_ > 0.0)
    size += EbmlElementSize(libwebm::kMkvFrameRate,
                            static_cast<float>(frame_rate_));
  if (colour_space_)
    size += EbmlElementSize(libwebm::kMkvColourSpace, colour_space_);
  if (colour_)
    size += colour_->ColourSize();
  if (projection_)
    size += projection_->ProjectionSize();

  return size;
}

///////////////////////////////////////////////////////////////
//
// AudioTrack Class

AudioTrack::AudioTrack(unsigned int* seed)
    : Track(seed), bit_depth_(0), channels_(1), sample_rate_(0.0) {}

AudioTrack::~AudioTrack() {}

uint64_t AudioTrack::PayloadSize() const {
  const uint64_t parent_size = Track::PayloadSize();

  uint64_t size = EbmlElementSize(libwebm::kMkvSamplingFrequency,
                                  static_cast<float>(sample_rate_));
  size +=
      EbmlElementSize(libwebm::kMkvChannels, static_cast<uint64>(channels_));
  if (bit_depth_ > 0)
    size +=
        EbmlElementSize(libwebm::kMkvBitDepth, static_cast<uint64>(bit_depth_));
  size += EbmlMasterElementSize(libwebm::kMkvAudio, size);

  return parent_size + size;
}

bool AudioTrack::Write(IMkvWriter* writer) const {
  if (!Track::Write(writer))
    return false;

  // Calculate AudioSettings size.
  uint64_t size = EbmlElementSize(libwebm::kMkvSamplingFrequency,
                                  static_cast<float>(sample_rate_));
  size +=
      EbmlElementSize(libwebm::kMkvChannels, static_cast<uint64>(channels_));
  if (bit_depth_ > 0)
    size +=
        EbmlElementSize(libwebm::kMkvBitDepth, static_cast<uint64>(bit_depth_));

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvAudio, size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvSamplingFrequency,
                        static_cast<float>(sample_rate_)))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvChannels,
                        static_cast<uint64>(channels_)))
    return false;
  if (bit_depth_ > 0)
    if (!WriteEbmlElement(writer, libwebm::kMkvBitDepth,
                          static_cast<uint64>(bit_depth_)))
      return false;

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  return true;
}

///////////////////////////////////////////////////////////////
//
// Tracks Class

const char Tracks::kOpusCodecId[] = "A_OPUS";
const char Tracks::kVorbisCodecId[] = "A_VORBIS";
const char Tracks::kAv1CodecId[] = "V_AV1";
const char Tracks::kVp8CodecId[] = "V_VP8";
const char Tracks::kVp9CodecId[] = "V_VP9";
const char Tracks::kWebVttCaptionsId[] = "D_WEBVTT/CAPTIONS";
const char Tracks::kWebVttDescriptionsId[] = "D_WEBVTT/DESCRIPTIONS";
const char Tracks::kWebVttMetadataId[] = "D_WEBVTT/METADATA";
const char Tracks::kWebVttSubtitlesId[] = "D_WEBVTT/SUBTITLES";

Tracks::Tracks()
    : track_entries_(NULL), track_entries_size_(0), wrote_tracks_(false) {}

Tracks::~Tracks() {
  if (track_entries_) {
    for (uint32_t i = 0; i < track_entries_size_; ++i) {
      Track* const track = track_entries_[i];
      delete track;
    }
    delete[] track_entries_;
  }
}

bool Tracks::AddTrack(Track* track, int32_t number) {
  if (number < 0 || wrote_tracks_)
    return false;

  // This muxer only supports track numbers in the range [1, 126], in
  // order to be able (to use Matroska integer representation) to
  // serialize the block header (of which the track number is a part)
  // for a frame using exactly 4 bytes.

  if (number > 0x7E)
    return false;

  uint32_t track_num = number;

  if (track_num > 0) {
    // Check to make sure a track does not already have |track_num|.
    for (uint32_t i = 0; i < track_entries_size_; ++i) {
      if (track_entries_[i]->number() == track_num)
        return false;
    }
  }

  const uint32_t count = track_entries_size_ + 1;

  Track** const track_entries = new (std::nothrow) Track*[count];  // NOLINT
  if (!track_entries)
    return false;

  for (uint32_t i = 0; i < track_entries_size_; ++i) {
    track_entries[i] = track_entries_[i];
  }

  delete[] track_entries_;

  // Find the lowest availible track number > 0.
  if (track_num == 0) {
    track_num = count;

    // Check to make sure a track does not already have |track_num|.
    bool exit = false;
    do {
      exit = true;
      for (uint32_t i = 0; i < track_entries_size_; ++i) {
        if (track_entries[i]->number() == track_num) {
          track_num++;
          exit = false;
          break;
        }
      }
    } while (!exit);
  }
  track->set_number(track_num);

  track_entries_ = track_entries;
  track_entries_[track_entries_size_] = track;
  track_entries_size_ = count;
  return true;
}

const Track* Tracks::GetTrackByIndex(uint32_t index) const {
  if (track_entries_ == NULL)
    return NULL;

  if (index >= track_entries_size_)
    return NULL;

  return track_entries_[index];
}

Track* Tracks::GetTrackByNumber(uint64_t track_number) const {
  const int32_t count = track_entries_size();
  for (int32_t i = 0; i < count; ++i) {
    if (track_entries_[i]->number() == track_number)
      return track_entries_[i];
  }

  return NULL;
}

bool Tracks::TrackIsAudio(uint64_t track_number) const {
  const Track* const track = GetTrackByNumber(track_number);

  if (track->type() == kAudio)
    return true;

  return false;
}

bool Tracks::TrackIsVideo(uint64_t track_number) const {
  const Track* const track = GetTrackByNumber(track_number);

  if (track->type() == kVideo)
    return true;

  return false;
}

bool Tracks::Write(IMkvWriter* writer) const {
  uint64_t size = 0;
  const int32_t count = track_entries_size();
  for (int32_t i = 0; i < count; ++i) {
    const Track* const track = GetTrackByIndex(i);

    if (!track)
      return false;

    size += track->Size();
  }

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvTracks, size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  for (int32_t i = 0; i < count; ++i) {
    const Track* const track = GetTrackByIndex(i);
    if (!track->Write(writer))
      return false;
  }

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  wrote_tracks_ = true;
  return true;
}

///////////////////////////////////////////////////////////////
//
// Chapter Class

bool Chapter::set_id(const char* id) { return StrCpy(id, &id_); }

void Chapter::set_time(const Segment& segment, uint64_t start_ns,
                       uint64_t end_ns) {
  const SegmentInfo* const info = segment.GetSegmentInfo();
  const uint64_t timecode_scale = info->timecode_scale();
  start_timecode_ = start_ns / timecode_scale;
  end_timecode_ = end_ns / timecode_scale;
}

bool Chapter::add_string(const char* title, const char* language,
                         const char* country) {
  if (!ExpandDisplaysArray())
    return false;

  Display& d = displays_[displays_count_++];
  d.Init();

  if (!d.set_title(title))
    return false;

  if (!d.set_language(language))
    return false;

  if (!d.set_country(country))
    return false;

  return true;
}

Chapter::Chapter() {
  // This ctor only constructs the object.  Proper initialization is
  // done in Init() (called in Chapters::AddChapter()).  The only
  // reason we bother implementing this ctor is because we had to
  // declare it as private (along with the dtor), in order to prevent
  // clients from creating Chapter instances (a privelege we grant
  // only to the Chapters class).  Doing no initialization here also
  // means that creating arrays of chapter objects is more efficient,
  // because we only initialize each new chapter object as it becomes
  // active on the array.
}

Chapter::~Chapter() {}

void Chapter::Init(unsigned int* seed) {
  id_ = NULL;
  start_timecode_ = 0;
  end_timecode_ = 0;
  displays_ = NULL;
  displays_size_ = 0;
  displays_count_ = 0;
  uid_ = MakeUID(seed);
}

void Chapter::ShallowCopy(Chapter* dst) const {
  dst->id_ = id_;
  dst->start_timecode_ = start_timecode_;
  dst->end_timecode_ = end_timecode_;
  dst->uid_ = uid_;
  dst->displays_ = displays_;
  dst->displays_size_ = displays_size_;
  dst->displays_count_ = displays_count_;
}

void Chapter::Clear() {
  StrCpy(NULL, &id_);

  while (displays_count_ > 0) {
    Display& d = displays_[--displays_count_];
    d.Clear();
  }

  delete[] displays_;
  displays_ = NULL;

  displays_size_ = 0;
}

bool Chapter::ExpandDisplaysArray() {
  if (displays_size_ > displays_count_)
    return true;  // nothing to do yet

  const int size = (displays_size_ == 0) ? 1 : 2 * displays_size_;

  Display* const displays = new (std::nothrow) Display[size];  // NOLINT
  if (displays == NULL)
    return false;

  for (int idx = 0; idx < displays_count_; ++idx) {
    displays[idx] = displays_[idx];  // shallow copy
  }

  delete[] displays_;

  displays_ = displays;
  displays_size_ = size;

  return true;
}

uint64_t Chapter::WriteAtom(IMkvWriter* writer) const {
  uint64_t payload_size =
      EbmlElementSize(libwebm::kMkvChapterStringUID, id_) +
      EbmlElementSize(libwebm::kMkvChapterUID, static_cast<uint64>(uid_)) +
      EbmlElementSize(libwebm::kMkvChapterTimeStart,
                      static_cast<uint64>(start_timecode_)) +
      EbmlElementSize(libwebm::kMkvChapterTimeEnd,
                      static_cast<uint64>(end_timecode_));

  for (int idx = 0; idx < displays_count_; ++idx) {
    const Display& d = displays_[idx];
    payload_size += d.WriteDisplay(NULL);
  }

  const uint64_t atom_size =
      EbmlMasterElementSize(libwebm::kMkvChapterAtom, payload_size) +
      payload_size;

  if (writer == NULL)
    return atom_size;

  const int64_t start = writer->Position();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvChapterAtom, payload_size))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvChapterStringUID, id_))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvChapterUID,
                        static_cast<uint64>(uid_)))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvChapterTimeStart,
                        static_cast<uint64>(start_timecode_)))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvChapterTimeEnd,
                        static_cast<uint64>(end_timecode_)))
    return 0;

  for (int idx = 0; idx < displays_count_; ++idx) {
    const Display& d = displays_[idx];

    if (!d.WriteDisplay(writer))
      return 0;
  }

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != atom_size)
    return 0;

  return atom_size;
}

void Chapter::Display::Init() {
  title_ = NULL;
  language_ = NULL;
  country_ = NULL;
}

void Chapter::Display::Clear() {
  StrCpy(NULL, &title_);
  StrCpy(NULL, &language_);
  StrCpy(NULL, &country_);
}

bool Chapter::Display::set_title(const char* title) {
  return StrCpy(title, &title_);
}

bool Chapter::Display::set_language(const char* language) {
  return StrCpy(language, &language_);
}

bool Chapter::Display::set_country(const char* country) {
  return StrCpy(country, &country_);
}

uint64_t Chapter::Display::WriteDisplay(IMkvWriter* writer) const {
  uint64_t payload_size = EbmlElementSize(libwebm::kMkvChapString, title_);

  if (language_)
    payload_size += EbmlElementSize(libwebm::kMkvChapLanguage, language_);

  if (country_)
    payload_size += EbmlElementSize(libwebm::kMkvChapCountry, country_);

  const uint64_t display_size =
      EbmlMasterElementSize(libwebm::kMkvChapterDisplay, payload_size) +
      payload_size;

  if (writer == NULL)
    return display_size;

  const int64_t start = writer->Position();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvChapterDisplay,
                              payload_size))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvChapString, title_))
    return 0;

  if (language_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvChapLanguage, language_))
      return 0;
  }

  if (country_) {
    if (!WriteEbmlElement(writer, libwebm::kMkvChapCountry, country_))
      return 0;
  }

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != display_size)
    return 0;

  return display_size;
}

///////////////////////////////////////////////////////////////
//
// Chapters Class

Chapters::Chapters() : chapters_size_(0), chapters_count_(0), chapters_(NULL) {}

Chapters::~Chapters() {
  while (chapters_count_ > 0) {
    Chapter& chapter = chapters_[--chapters_count_];
    chapter.Clear();
  }

  delete[] chapters_;
  chapters_ = NULL;
}

int Chapters::Count() const { return chapters_count_; }

Chapter* Chapters::AddChapter(unsigned int* seed) {
  if (!ExpandChaptersArray())
    return NULL;

  Chapter& chapter = chapters_[chapters_count_++];
  chapter.Init(seed);

  return &chapter;
}

bool Chapters::Write(IMkvWriter* writer) const {
  if (writer == NULL)
    return false;

  const uint64_t payload_size = WriteEdition(NULL);  // return size only

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvChapters, payload_size))
    return false;

  const int64_t start = writer->Position();

  if (WriteEdition(writer) == 0)  // error
    return false;

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != payload_size)
    return false;

  return true;
}

bool Chapters::ExpandChaptersArray() {
  if (chapters_size_ > chapters_count_)
    return true;  // nothing to do yet

  const int size = (chapters_size_ == 0) ? 1 : 2 * chapters_size_;

  Chapter* const chapters = new (std::nothrow) Chapter[size];  // NOLINT
  if (chapters == NULL)
    return false;

  for (int idx = 0; idx < chapters_count_; ++idx) {
    const Chapter& src = chapters_[idx];
    Chapter* const dst = chapters + idx;
    src.ShallowCopy(dst);
  }

  delete[] chapters_;

  chapters_ = chapters;
  chapters_size_ = size;

  return true;
}

uint64_t Chapters::WriteEdition(IMkvWriter* writer) const {
  uint64_t payload_size = 0;

  for (int idx = 0; idx < chapters_count_; ++idx) {
    const Chapter& chapter = chapters_[idx];
    payload_size += chapter.WriteAtom(NULL);
  }

  const uint64_t edition_size =
      EbmlMasterElementSize(libwebm::kMkvEditionEntry, payload_size) +
      payload_size;

  if (writer == NULL)  // return size only
    return edition_size;

  const int64_t start = writer->Position();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvEditionEntry, payload_size))
    return 0;  // error

  for (int idx = 0; idx < chapters_count_; ++idx) {
    const Chapter& chapter = chapters_[idx];

    const uint64_t chapter_size = chapter.WriteAtom(writer);
    if (chapter_size == 0)  // error
      return 0;
  }

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != edition_size)
    return 0;

  return edition_size;
}

// Tag Class

bool Tag::add_simple_tag(const char* tag_name, const char* tag_string) {
  if (!ExpandSimpleTagsArray())
    return false;

  SimpleTag& st = simple_tags_[simple_tags_count_++];
  st.Init();

  if (!st.set_tag_name(tag_name))
    return false;

  if (!st.set_tag_string(tag_string))
    return false;

  return true;
}

Tag::Tag() {
  simple_tags_ = NULL;
  simple_tags_size_ = 0;
  simple_tags_count_ = 0;
}

Tag::~Tag() {}

void Tag::ShallowCopy(Tag* dst) const {
  dst->simple_tags_ = simple_tags_;
  dst->simple_tags_size_ = simple_tags_size_;
  dst->simple_tags_count_ = simple_tags_count_;
}

void Tag::Clear() {
  while (simple_tags_count_ > 0) {
    SimpleTag& st = simple_tags_[--simple_tags_count_];
    st.Clear();
  }

  delete[] simple_tags_;
  simple_tags_ = NULL;

  simple_tags_size_ = 0;
}

bool Tag::ExpandSimpleTagsArray() {
  if (simple_tags_size_ > simple_tags_count_)
    return true;  // nothing to do yet

  const int size = (simple_tags_size_ == 0) ? 1 : 2 * simple_tags_size_;

  SimpleTag* const simple_tags = new (std::nothrow) SimpleTag[size];  // NOLINT
  if (simple_tags == NULL)
    return false;

  for (int idx = 0; idx < simple_tags_count_; ++idx) {
    simple_tags[idx] = simple_tags_[idx];  // shallow copy
  }

  delete[] simple_tags_;

  simple_tags_ = simple_tags;
  simple_tags_size_ = size;

  return true;
}

uint64_t Tag::Write(IMkvWriter* writer) const {
  uint64_t payload_size = 0;

  for (int idx = 0; idx < simple_tags_count_; ++idx) {
    const SimpleTag& st = simple_tags_[idx];
    payload_size += st.Write(NULL);
  }

  const uint64_t tag_size =
      EbmlMasterElementSize(libwebm::kMkvTag, payload_size) + payload_size;

  if (writer == NULL)
    return tag_size;

  const int64_t start = writer->Position();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvTag, payload_size))
    return 0;

  for (int idx = 0; idx < simple_tags_count_; ++idx) {
    const SimpleTag& st = simple_tags_[idx];

    if (!st.Write(writer))
      return 0;
  }

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != tag_size)
    return 0;

  return tag_size;
}

// Tag::SimpleTag

void Tag::SimpleTag::Init() {
  tag_name_ = NULL;
  tag_string_ = NULL;
}

void Tag::SimpleTag::Clear() {
  StrCpy(NULL, &tag_name_);
  StrCpy(NULL, &tag_string_);
}

bool Tag::SimpleTag::set_tag_name(const char* tag_name) {
  return StrCpy(tag_name, &tag_name_);
}

bool Tag::SimpleTag::set_tag_string(const char* tag_string) {
  return StrCpy(tag_string, &tag_string_);
}

uint64_t Tag::SimpleTag::Write(IMkvWriter* writer) const {
  uint64_t payload_size = EbmlElementSize(libwebm::kMkvTagName, tag_name_);

  payload_size += EbmlElementSize(libwebm::kMkvTagString, tag_string_);

  const uint64_t simple_tag_size =
      EbmlMasterElementSize(libwebm::kMkvSimpleTag, payload_size) +
      payload_size;

  if (writer == NULL)
    return simple_tag_size;

  const int64_t start = writer->Position();

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvSimpleTag, payload_size))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvTagName, tag_name_))
    return 0;

  if (!WriteEbmlElement(writer, libwebm::kMkvTagString, tag_string_))
    return 0;

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != simple_tag_size)
    return 0;

  return simple_tag_size;
}

// Tags Class

Tags::Tags() : tags_size_(0), tags_count_(0), tags_(NULL) {}

Tags::~Tags() {
  while (tags_count_ > 0) {
    Tag& tag = tags_[--tags_count_];
    tag.Clear();
  }

  delete[] tags_;
  tags_ = NULL;
}

int Tags::Count() const { return tags_count_; }

Tag* Tags::AddTag() {
  if (!ExpandTagsArray())
    return NULL;

  Tag& tag = tags_[tags_count_++];

  return &tag;
}

bool Tags::Write(IMkvWriter* writer) const {
  if (writer == NULL)
    return false;

  uint64_t payload_size = 0;

  for (int idx = 0; idx < tags_count_; ++idx) {
    const Tag& tag = tags_[idx];
    payload_size += tag.Write(NULL);
  }

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvTags, payload_size))
    return false;

  const int64_t start = writer->Position();

  for (int idx = 0; idx < tags_count_; ++idx) {
    const Tag& tag = tags_[idx];

    const uint64_t tag_size = tag.Write(writer);
    if (tag_size == 0)  // error
      return 0;
  }

  const int64_t stop = writer->Position();

  if (stop >= start && uint64_t(stop - start) != payload_size)
    return false;

  return true;
}

bool Tags::ExpandTagsArray() {
  if (tags_size_ > tags_count_)
    return true;  // nothing to do yet

  const int size = (tags_size_ == 0) ? 1 : 2 * tags_size_;

  Tag* const tags = new (std::nothrow) Tag[size];  // NOLINT
  if (tags == NULL)
    return false;

  for (int idx = 0; idx < tags_count_; ++idx) {
    const Tag& src = tags_[idx];
    Tag* const dst = tags + idx;
    src.ShallowCopy(dst);
  }

  delete[] tags_;

  tags_ = tags;
  tags_size_ = size;

  return true;
}

///////////////////////////////////////////////////////////////
//
// Cluster class

Cluster::Cluster(uint64_t timecode, int64_t cues_pos, uint64_t timecode_scale,
                 bool write_last_frame_with_duration, bool fixed_size_timecode)
    : blocks_added_(0),
      finalized_(false),
      fixed_size_timecode_(fixed_size_timecode),
      header_written_(false),
      payload_size_(0),
      position_for_cues_(cues_pos),
      size_position_(-1),
      timecode_(timecode),
      timecode_scale_(timecode_scale),
      write_last_frame_with_duration_(write_last_frame_with_duration),
      writer_(NULL) {}

Cluster::~Cluster() {
  // Delete any stored frames that are left behind. This will happen if the
  // Cluster was not Finalized for whatever reason.
  while (!stored_frames_.empty()) {
    while (!stored_frames_.begin()->second.empty()) {
      delete stored_frames_.begin()->second.front();
      stored_frames_.begin()->second.pop_front();
    }
    stored_frames_.erase(stored_frames_.begin()->first);
  }
}

bool Cluster::Init(IMkvWriter* ptr_writer) {
  if (!ptr_writer) {
    return false;
  }
  writer_ = ptr_writer;
  return true;
}

bool Cluster::AddFrame(const Frame* const frame) {
  return QueueOrWriteFrame(frame);
}

bool Cluster::AddFrame(const uint8_t* data, uint64_t length,
                       uint64_t track_number, uint64_t abs_timecode,
                       bool is_key) {
  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_track_number(track_number);
  frame.set_timestamp(abs_timecode);
  frame.set_is_key(is_key);
  return QueueOrWriteFrame(&frame);
}

bool Cluster::AddFrameWithAdditional(const uint8_t* data, uint64_t length,
                                     const uint8_t* additional,
                                     uint64_t additional_length,
                                     uint64_t add_id, uint64_t track_number,
                                     uint64_t abs_timecode, bool is_key) {
  if (!additional || additional_length == 0) {
    return false;
  }
  Frame frame;
  if (!frame.Init(data, length) ||
      !frame.AddAdditionalData(additional, additional_length, add_id)) {
    return false;
  }
  frame.set_track_number(track_number);
  frame.set_timestamp(abs_timecode);
  frame.set_is_key(is_key);
  return QueueOrWriteFrame(&frame);
}

bool Cluster::AddFrameWithDiscardPadding(const uint8_t* data, uint64_t length,
                                         int64_t discard_padding,
                                         uint64_t track_number,
                                         uint64_t abs_timecode, bool is_key) {
  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_discard_padding(discard_padding);
  frame.set_track_number(track_number);
  frame.set_timestamp(abs_timecode);
  frame.set_is_key(is_key);
  return QueueOrWriteFrame(&frame);
}

bool Cluster::AddMetadata(const uint8_t* data, uint64_t length,
                          uint64_t track_number, uint64_t abs_timecode,
                          uint64_t duration_timecode) {
  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_track_number(track_number);
  frame.set_timestamp(abs_timecode);
  frame.set_duration(duration_timecode);
  frame.set_is_key(true);  // All metadata blocks are keyframes.
  return QueueOrWriteFrame(&frame);
}

void Cluster::AddPayloadSize(uint64_t size) { payload_size_ += size; }

bool Cluster::Finalize() {
  return !write_last_frame_with_duration_ && Finalize(false, 0);
}

bool Cluster::Finalize(bool set_last_frame_duration, uint64_t duration) {
  if (!writer_ || finalized_)
    return false;

  if (write_last_frame_with_duration_) {
    // Write out held back Frames. This essentially performs a k-way merge
    // across all tracks in the increasing order of timestamps.
    while (!stored_frames_.empty()) {
      Frame* frame = stored_frames_.begin()->second.front();

      // Get the next frame to write (frame with least timestamp across all
      // tracks).
      for (FrameMapIterator frames_iterator = ++stored_frames_.begin();
           frames_iterator != stored_frames_.end(); ++frames_iterator) {
        if (frames_iterator->second.front()->timestamp() < frame->timestamp()) {
          frame = frames_iterator->second.front();
        }
      }

      // Set the duration if it's the last frame for the track.
      if (set_last_frame_duration &&
          stored_frames_[frame->track_number()].size() == 1 &&
          !frame->duration_set()) {
        frame->set_duration(duration - frame->timestamp());
        if (!frame->is_key() && !frame->reference_block_timestamp_set()) {
          frame->set_reference_block_timestamp(
              last_block_timestamp_[frame->track_number()]);
        }
      }

      // Write the frame and remove it from |stored_frames_|.
      const bool wrote_frame = DoWriteFrame(frame);
      stored_frames_[frame->track_number()].pop_front();
      if (stored_frames_[frame->track_number()].empty()) {
        stored_frames_.erase(frame->track_number());
      }
      delete frame;
      if (!wrote_frame)
        return false;
    }
  }

  if (size_position_ == -1)
    return false;

  if (writer_->Seekable()) {
    const int64_t pos = writer_->Position();

    if (writer_->Position(size_position_))
      return false;

    if (WriteUIntSize(writer_, payload_size(), 8))
      return false;

    if (writer_->Position(pos))
      return false;
  }

  finalized_ = true;

  return true;
}

uint64_t Cluster::Size() const {
  const uint64_t element_size =
      EbmlMasterElementSize(libwebm::kMkvCluster, 0xFFFFFFFFFFFFFFFFULL) +
      payload_size_;
  return element_size;
}

bool Cluster::PreWriteBlock() {
  if (finalized_)
    return false;

  if (!header_written_) {
    if (!WriteClusterHeader())
      return false;
  }

  return true;
}

void Cluster::PostWriteBlock(uint64_t element_size) {
  AddPayloadSize(element_size);
  ++blocks_added_;
}

int64_t Cluster::GetRelativeTimecode(int64_t abs_timecode) const {
  const int64_t cluster_timecode = this->Cluster::timecode();
  const int64_t rel_timecode =
      static_cast<int64_t>(abs_timecode) - cluster_timecode;

  if (rel_timecode < 0 || rel_timecode > kMaxBlockTimecode)
    return -1;

  return rel_timecode;
}

bool Cluster::DoWriteFrame(const Frame* const frame) {
  if (!frame || !frame->IsValid())
    return false;

  if (!PreWriteBlock())
    return false;

  const uint64_t element_size = WriteFrame(writer_, frame, this);
  if (element_size == 0)
    return false;

  PostWriteBlock(element_size);
  last_block_timestamp_[frame->track_number()] = frame->timestamp();
  return true;
}

bool Cluster::QueueOrWriteFrame(const Frame* const frame) {
  if (!frame || !frame->IsValid())
    return false;

  // If |write_last_frame_with_duration_| is not set, then write the frame right
  // away.
  if (!write_last_frame_with_duration_) {
    return DoWriteFrame(frame);
  }

  // Queue the current frame.
  uint64_t track_number = frame->track_number();
  Frame* const frame_to_store = new Frame();
  frame_to_store->CopyFrom(*frame);
  stored_frames_[track_number].push_back(frame_to_store);

  // Iterate through all queued frames in the current track except the last one
  // and write it if it is okay to do so (i.e.) no other track has an held back
  // frame with timestamp <= the timestamp of the frame in question.
  std::vector<std::list<Frame*>::iterator> frames_to_erase;
  for (std::list<Frame*>::iterator
           current_track_iterator = stored_frames_[track_number].begin(),
           end = --stored_frames_[track_number].end();
       current_track_iterator != end; ++current_track_iterator) {
    const Frame* const frame_to_write = *current_track_iterator;
    bool okay_to_write = true;
    for (FrameMapIterator track_iterator = stored_frames_.begin();
         track_iterator != stored_frames_.end(); ++track_iterator) {
      if (track_iterator->first == track_number) {
        continue;
      }
      if (track_iterator->second.front()->timestamp() <
          frame_to_write->timestamp()) {
        okay_to_write = false;
        break;
      }
    }
    if (okay_to_write) {
      const bool wrote_frame = DoWriteFrame(frame_to_write);
      delete frame_to_write;
      if (!wrote_frame)
        return false;
      frames_to_erase.push_back(current_track_iterator);
    } else {
      break;
    }
  }
  for (std::vector<std::list<Frame*>::iterator>::iterator iterator =
           frames_to_erase.begin();
       iterator != frames_to_erase.end(); ++iterator) {
    stored_frames_[track_number].erase(*iterator);
  }
  return true;
}

bool Cluster::WriteClusterHeader() {
  if (finalized_)
    return false;

  if (WriteID(writer_, libwebm::kMkvCluster))
    return false;

  // Save for later.
  size_position_ = writer_->Position();

  // Write "unknown" (EBML coded -1) as cluster size value. We need to write 8
  // bytes because we do not know how big our cluster will be.
  if (SerializeInt(writer_, kEbmlUnknownValue, 8))
    return false;

  if (!WriteEbmlElement(writer_, libwebm::kMkvTimecode, timecode(),
                        fixed_size_timecode_ ? 8 : 0)) {
    return false;
  }
  AddPayloadSize(EbmlElementSize(libwebm::kMkvTimecode, timecode(),
                                 fixed_size_timecode_ ? 8 : 0));
  header_written_ = true;

  return true;
}

///////////////////////////////////////////////////////////////
//
// SeekHead Class

SeekHead::SeekHead() : start_pos_(0ULL) {
  for (int32_t i = 0; i < kSeekEntryCount; ++i) {
    seek_entry_id_[i] = 0;
    seek_entry_pos_[i] = 0;
  }
}

SeekHead::~SeekHead() {}

bool SeekHead::Finalize(IMkvWriter* writer) const {
  if (writer->Seekable()) {
    if (start_pos_ == -1)
      return false;

    uint64_t payload_size = 0;
    uint64_t entry_size[kSeekEntryCount];

    for (int32_t i = 0; i < kSeekEntryCount; ++i) {
      if (seek_entry_id_[i] != 0) {
        entry_size[i] = EbmlElementSize(libwebm::kMkvSeekID,
                                        static_cast<uint64>(seek_entry_id_[i]));
        entry_size[i] += EbmlElementSize(
            libwebm::kMkvSeekPosition, static_cast<uint64>(seek_entry_pos_[i]));

        payload_size +=
            EbmlMasterElementSize(libwebm::kMkvSeek, entry_size[i]) +
            entry_size[i];
      }
    }

    // No SeekHead elements
    if (payload_size == 0)
      return true;

    const int64_t pos = writer->Position();
    if (writer->Position(start_pos_))
      return false;

    if (!WriteEbmlMasterElement(writer, libwebm::kMkvSeekHead, payload_size))
      return false;

    for (int32_t i = 0; i < kSeekEntryCount; ++i) {
      if (seek_entry_id_[i] != 0) {
        if (!WriteEbmlMasterElement(writer, libwebm::kMkvSeek, entry_size[i]))
          return false;

        if (!WriteEbmlElement(writer, libwebm::kMkvSeekID,
                              static_cast<uint64>(seek_entry_id_[i])))
          return false;

        if (!WriteEbmlElement(writer, libwebm::kMkvSeekPosition,
                              static_cast<uint64>(seek_entry_pos_[i])))
          return false;
      }
    }

    const uint64_t total_entry_size = kSeekEntryCount * MaxEntrySize();
    const uint64_t total_size =
        EbmlMasterElementSize(libwebm::kMkvSeekHead, total_entry_size) +
        total_entry_size;
    const int64_t size_left = total_size - (writer->Position() - start_pos_);

    const uint64_t bytes_written = WriteVoidElement(writer, size_left);
    if (!bytes_written)
      return false;

    if (writer->Position(pos))
      return false;
  }

  return true;
}

bool SeekHead::Write(IMkvWriter* writer) {
  const uint64_t entry_size = kSeekEntryCount * MaxEntrySize();
  const uint64_t size =
      EbmlMasterElementSize(libwebm::kMkvSeekHead, entry_size);

  start_pos_ = writer->Position();

  const uint64_t bytes_written = WriteVoidElement(writer, size + entry_size);
  if (!bytes_written)
    return false;

  return true;
}

bool SeekHead::AddSeekEntry(uint32_t id, uint64_t pos) {
  for (int32_t i = 0; i < kSeekEntryCount; ++i) {
    if (seek_entry_id_[i] == 0) {
      seek_entry_id_[i] = id;
      seek_entry_pos_[i] = pos;
      return true;
    }
  }
  return false;
}

uint32_t SeekHead::GetId(int index) const {
  if (index < 0 || index >= kSeekEntryCount)
    return UINT32_MAX;
  return seek_entry_id_[index];
}

uint64_t SeekHead::GetPosition(int index) const {
  if (index < 0 || index >= kSeekEntryCount)
    return UINT64_MAX;
  return seek_entry_pos_[index];
}

bool SeekHead::SetSeekEntry(int index, uint32_t id, uint64_t position) {
  if (index < 0 || index >= kSeekEntryCount)
    return false;
  seek_entry_id_[index] = id;
  seek_entry_pos_[index] = position;
  return true;
}

uint64_t SeekHead::MaxEntrySize() const {
  const uint64_t max_entry_payload_size =
      EbmlElementSize(libwebm::kMkvSeekID,
                      static_cast<uint64>(UINT64_C(0xffffffff))) +
      EbmlElementSize(libwebm::kMkvSeekPosition,
                      static_cast<uint64>(UINT64_C(0xffffffffffffffff)));
  const uint64_t max_entry_size =
      EbmlMasterElementSize(libwebm::kMkvSeek, max_entry_payload_size) +
      max_entry_payload_size;

  return max_entry_size;
}

///////////////////////////////////////////////////////////////
//
// SegmentInfo Class

SegmentInfo::SegmentInfo()
    : duration_(-1.0),
      muxing_app_(NULL),
      timecode_scale_(1000000ULL),
      writing_app_(NULL),
      date_utc_(INT64_MIN),
      duration_pos_(-1) {}

SegmentInfo::~SegmentInfo() {
  delete[] muxing_app_;
  delete[] writing_app_;
}

bool SegmentInfo::Init() {
  int32_t major;
  int32_t minor;
  int32_t build;
  int32_t revision;
  GetVersion(&major, &minor, &build, &revision);
  char temp[256];
#ifdef _MSC_VER
  sprintf_s(temp, sizeof(temp) / sizeof(temp[0]), "libwebm-%d.%d.%d.%d", major,
            minor, build, revision);
#else
  snprintf(temp, sizeof(temp) / sizeof(temp[0]), "libwebm-%d.%d.%d.%d", major,
           minor, build, revision);
#endif

  const size_t app_len = strlen(temp) + 1;

  delete[] muxing_app_;

  muxing_app_ = new (std::nothrow) char[app_len];  // NOLINT
  if (!muxing_app_)
    return false;

  memcpy(muxing_app_, temp, app_len - 1);
  muxing_app_[app_len - 1] = '\0';

  set_writing_app(temp);
  if (!writing_app_)
    return false;
  return true;
}

bool SegmentInfo::Finalize(IMkvWriter* writer) const {
  if (!writer)
    return false;

  if (duration_ > 0.0) {
    if (writer->Seekable()) {
      if (duration_pos_ == -1)
        return false;

      const int64_t pos = writer->Position();

      if (writer->Position(duration_pos_))
        return false;

      if (!WriteEbmlElement(writer, libwebm::kMkvDuration,
                            static_cast<float>(duration_)))
        return false;

      if (writer->Position(pos))
        return false;
    }
  }

  return true;
}

bool SegmentInfo::Write(IMkvWriter* writer) {
  if (!writer || !muxing_app_ || !writing_app_)
    return false;

  uint64_t size = EbmlElementSize(libwebm::kMkvTimecodeScale,
                                  static_cast<uint64>(timecode_scale_));
  if (duration_ > 0.0)
    size +=
        EbmlElementSize(libwebm::kMkvDuration, static_cast<float>(duration_));
  if (date_utc_ != INT64_MIN)
    size += EbmlDateElementSize(libwebm::kMkvDateUTC);
  size += EbmlElementSize(libwebm::kMkvMuxingApp, muxing_app_);
  size += EbmlElementSize(libwebm::kMkvWritingApp, writing_app_);

  if (!WriteEbmlMasterElement(writer, libwebm::kMkvInfo, size))
    return false;

  const int64_t payload_position = writer->Position();
  if (payload_position < 0)
    return false;

  if (!WriteEbmlElement(writer, libwebm::kMkvTimecodeScale,
                        static_cast<uint64>(timecode_scale_)))
    return false;

  if (duration_ > 0.0) {
    // Save for later
    duration_pos_ = writer->Position();

    if (!WriteEbmlElement(writer, libwebm::kMkvDuration,
                          static_cast<float>(duration_)))
      return false;
  }

  if (date_utc_ != INT64_MIN)
    WriteEbmlDateElement(writer, libwebm::kMkvDateUTC, date_utc_);

  if (!WriteEbmlElement(writer, libwebm::kMkvMuxingApp, muxing_app_))
    return false;
  if (!WriteEbmlElement(writer, libwebm::kMkvWritingApp, writing_app_))
    return false;

  const int64_t stop_position = writer->Position();
  if (stop_position < 0 ||
      stop_position - payload_position != static_cast<int64_t>(size))
    return false;

  return true;
}

void SegmentInfo::set_muxing_app(const char* app) {
  if (app) {
    const size_t length = strlen(app) + 1;
    char* temp_str = new (std::nothrow) char[length];  // NOLINT
    if (!temp_str)
      return;

    memcpy(temp_str, app, length - 1);
    temp_str[length - 1] = '\0';

    delete[] muxing_app_;
    muxing_app_ = temp_str;
  }
}

void SegmentInfo::set_writing_app(const char* app) {
  if (app) {
    const size_t length = strlen(app) + 1;
    char* temp_str = new (std::nothrow) char[length];  // NOLINT
    if (!temp_str)
      return;

    memcpy(temp_str, app, length - 1);
    temp_str[length - 1] = '\0';

    delete[] writing_app_;
    writing_app_ = temp_str;
  }
}

///////////////////////////////////////////////////////////////
//
// Segment Class

Segment::Segment()
    : chunk_count_(0),
      chunk_name_(NULL),
      chunk_writer_cluster_(NULL),
      chunk_writer_cues_(NULL),
      chunk_writer_header_(NULL),
      chunking_(false),
      chunking_base_name_(NULL),
      cluster_list_(NULL),
      cluster_list_capacity_(0),
      cluster_list_size_(0),
      cues_position_(kAfterClusters),
      cues_track_(0),
      force_new_cluster_(false),
      frames_(NULL),
      frames_capacity_(0),
      frames_size_(0),
      has_video_(false),
      header_written_(false),
      last_block_duration_(0),
      last_timestamp_(0),
      max_cluster_duration_(kDefaultMaxClusterDuration),
      max_cluster_size_(0),
      mode_(kFile),
      new_cuepoint_(false),
      output_cues_(true),
      accurate_cluster_duration_(false),
      fixed_size_cluster_timecode_(false),
      estimate_file_duration_(false),
      ebml_header_size_(0),
      payload_pos_(0),
      size_position_(0),
      doc_type_version_(kDefaultDocTypeVersion),
      doc_type_version_written_(0),
      duration_(0.0),
      writer_cluster_(NULL),
      writer_cues_(NULL),
      writer_header_(NULL) {
  const time_t curr_time = time(NULL);
  seed_ = static_cast<unsigned int>(curr_time);
#ifdef _WIN32
  srand(seed_);
#endif
}

Segment::~Segment() {
  if (cluster_list_) {
    for (int32_t i = 0; i < cluster_list_size_; ++i) {
      Cluster* const cluster = cluster_list_[i];
      delete cluster;
    }
    delete[] cluster_list_;
  }

  if (frames_) {
    for (int32_t i = 0; i < frames_size_; ++i) {
      Frame* const frame = frames_[i];
      delete frame;
    }
    delete[] frames_;
  }

  delete[] chunk_name_;
  delete[] chunking_base_name_;

  if (chunk_writer_cluster_) {
    chunk_writer_cluster_->Close();
    delete chunk_writer_cluster_;
  }
  if (chunk_writer_cues_) {
    chunk_writer_cues_->Close();
    delete chunk_writer_cues_;
  }
  if (chunk_writer_header_) {
    chunk_writer_header_->Close();
    delete chunk_writer_header_;
  }
}

void Segment::MoveCuesBeforeClustersHelper(uint64_t diff, int32_t index,
                                           uint64_t* cues_size) {
  CuePoint* const cue_point = cues_.GetCueByIndex(index);
  if (cue_point == NULL)
    return;
  const uint64_t old_cue_point_size = cue_point->Size();
  const uint64_t cluster_pos = cue_point->cluster_pos() + diff;
  cue_point->set_cluster_pos(cluster_pos);  // update the new cluster position
  // New size of the cue is computed as follows
  //    Let a = current sum of size of all CuePoints
  //    Let b = Increase in Cue Point's size due to this iteration
  //    Let c = Increase in size of Cues Element's length due to this iteration
  //            (This is computed as CodedSize(a + b) - CodedSize(a))
  //    Let d = b + c. Now d is the |diff| passed to the next recursive call.
  //    Let e = a + b. Now e is the |cues_size| passed to the next recursive
  //                   call.
  const uint64_t cue_point_size_diff = cue_point->Size() - old_cue_point_size;
  const uint64_t cue_size_diff =
      GetCodedUIntSize(*cues_size + cue_point_size_diff) -
      GetCodedUIntSize(*cues_size);
  *cues_size += cue_point_size_diff;
  diff = cue_size_diff + cue_point_size_diff;
  if (diff > 0) {
    for (int32_t i = 0; i < cues_.cue_entries_size(); ++i) {
      MoveCuesBeforeClustersHelper(diff, i, cues_size);
    }
  }
}

void Segment::MoveCuesBeforeClusters() {
  const uint64_t current_cue_size = cues_.Size();
  uint64_t cue_size = 0;
  for (int32_t i = 0; i < cues_.cue_entries_size(); ++i)
    cue_size += cues_.GetCueByIndex(i)->Size();
  for (int32_t i = 0; i < cues_.cue_entries_size(); ++i)
    MoveCuesBeforeClustersHelper(current_cue_size, i, &cue_size);

  // Adjust the Seek Entry to reflect the change in position
  // of Cluster and Cues
  int32_t cluster_index = 0;
  int32_t cues_index = 0;
  for (int32_t i = 0; i < SeekHead::kSeekEntryCount; ++i) {
    if (seek_head_.GetId(i) == libwebm::kMkvCluster)
      cluster_index = i;
    if (seek_head_.GetId(i) == libwebm::kMkvCues)
      cues_index = i;
  }
  seek_head_.SetSeekEntry(cues_index, libwebm::kMkvCues,
                          seek_head_.GetPosition(cluster_index));
  seek_head_.SetSeekEntry(cluster_index, libwebm::kMkvCluster,
                          cues_.Size() + seek_head_.GetPosition(cues_index));
}

bool Segment::Init(IMkvWriter* ptr_writer) {
  if (!ptr_writer) {
    return false;
  }
  writer_cluster_ = ptr_writer;
  writer_cues_ = ptr_writer;
  writer_header_ = ptr_writer;
  memset(&track_frames_written_, 0,
         sizeof(track_frames_written_[0]) * kMaxTrackNumber);
  memset(&last_track_timestamp_, 0,
         sizeof(last_track_timestamp_[0]) * kMaxTrackNumber);
  return segment_info_.Init();
}

bool Segment::CopyAndMoveCuesBeforeClusters(mkvparser::IMkvReader* reader,
                                            IMkvWriter* writer) {
  if (!writer->Seekable() || chunking_)
    return false;
  const int64_t cluster_offset =
      cluster_list_[0]->size_position() - GetUIntSize(libwebm::kMkvCluster);

  // Copy the headers.
  if (!ChunkedCopy(reader, writer, 0, cluster_offset))
    return false;

  // Recompute cue positions and seek entries.
  MoveCuesBeforeClusters();

  // Write cues and seek entries.
  // TODO(vigneshv): As of now, it's safe to call seek_head_.Finalize() for the
  // second time with a different writer object. But the name Finalize() doesn't
  // indicate something we want to call more than once. So consider renaming it
  // to write() or some such.
  if (!cues_.Write(writer) || !seek_head_.Finalize(writer))
    return false;

  // Copy the Clusters.
  if (!ChunkedCopy(reader, writer, cluster_offset,
                   cluster_end_offset_ - cluster_offset))
    return false;

  // Update the Segment size in case the Cues size has changed.
  const int64_t pos = writer->Position();
  const int64_t segment_size = writer->Position() - payload_pos_;
  if (writer->Position(size_position_) ||
      WriteUIntSize(writer, segment_size, 8) || writer->Position(pos))
    return false;
  return true;
}

bool Segment::Finalize() {
  if (WriteFramesAll() < 0)
    return false;

  // In kLive mode, call Cluster::Finalize only if |accurate_cluster_duration_|
  // is set. In all other modes, always call Cluster::Finalize.
  if ((mode_ == kLive ? accurate_cluster_duration_ : true) &&
      cluster_list_size_ > 0) {
    // Update last cluster's size
    Cluster* const old_cluster = cluster_list_[cluster_list_size_ - 1];

    // For the last frame of the last Cluster, we don't write it as a BlockGroup
    // with Duration unless the frame itself has duration set explicitly.
    if (!old_cluster || !old_cluster->Finalize(false, 0))
      return false;
  }

  if (mode_ == kFile) {
    if (chunking_ && chunk_writer_cluster_) {
      chunk_writer_cluster_->Close();
      chunk_count_++;
    }

    double duration =
        (static_cast<double>(last_timestamp_) + last_block_duration_) /
        segment_info_.timecode_scale();
    if (duration_ > 0.0) {
      duration = duration_;
    } else {
      if (last_block_duration_ == 0 && estimate_file_duration_) {
        const int num_tracks = static_cast<int>(tracks_.track_entries_size());
        for (int i = 0; i < num_tracks; ++i) {
          if (track_frames_written_[i] < 2)
            continue;

          // Estimate the duration for the last block of a Track.
          const double nano_per_frame =
              static_cast<double>(last_track_timestamp_[i]) /
              (track_frames_written_[i] - 1);
          const double track_duration =
              (last_track_timestamp_[i] + nano_per_frame) /
              segment_info_.timecode_scale();
          if (track_duration > duration)
            duration = track_duration;
        }
      }
    }
    segment_info_.set_duration(duration);
    if (!segment_info_.Finalize(writer_header_))
      return false;

    if (output_cues_)
      if (!seek_head_.AddSeekEntry(libwebm::kMkvCues, MaxOffset()))
        return false;

    if (chunking_) {
      if (!chunk_writer_cues_)
        return false;

      char* name = NULL;
      if (!UpdateChunkName("cues", &name))
        return false;

      const bool cues_open = chunk_writer_cues_->Open(name);
      delete[] name;
      if (!cues_open)
        return false;
    }

    cluster_end_offset_ = writer_cluster_->Position();

    // Write the seek headers and cues
    if (output_cues_)
      if (!cues_.Write(writer_cues_))
        return false;

    if (!seek_head_.Finalize(writer_header_))
      return false;

    if (writer_header_->Seekable()) {
      if (size_position_ == -1)
        return false;

      const int64_t segment_size = MaxOffset();
      if (segment_size < 1)
        return false;

      const int64_t pos = writer_header_->Position();
      UpdateDocTypeVersion();
      if (doc_type_version_ != doc_type_version_written_) {
        if (writer_header_->Position(0))
          return false;

        const char* const doc_type =
            DocTypeIsWebm() ? kDocTypeWebm : kDocTypeMatroska;
        if (!WriteEbmlHeader(writer_header_, doc_type_version_, doc_type))
          return false;
        if (writer_header_->Position() != ebml_header_size_)
          return false;

        doc_type_version_written_ = doc_type_version_;
      }

      if (writer_header_->Position(size_position_))
        return false;

      if (WriteUIntSize(writer_header_, segment_size, 8))
        return false;

      if (writer_header_->Position(pos))
        return false;
    }

    if (chunking_) {
      // Do not close any writers until the segment size has been written,
      // otherwise the size may be off.
      if (!chunk_writer_cues_ || !chunk_writer_header_)
        return false;

      chunk_writer_cues_->Close();
      chunk_writer_header_->Close();
    }
  }

  return true;
}

Track* Segment::AddTrack(int32_t number) {
  Track* const track = new (std::nothrow) Track(&seed_);  // NOLINT

  if (!track)
    return NULL;

  if (!tracks_.AddTrack(track, number)) {
    delete track;
    return NULL;
  }

  return track;
}

Chapter* Segment::AddChapter() { return chapters_.AddChapter(&seed_); }

Tag* Segment::AddTag() { return tags_.AddTag(); }

uint64_t Segment::AddVideoTrack(int32_t width, int32_t height, int32_t number) {
  VideoTrack* const track = new (std::nothrow) VideoTrack(&seed_);  // NOLINT
  if (!track)
    return 0;

  track->set_type(Tracks::kVideo);
  track->set_codec_id(Tracks::kVp8CodecId);
  track->set_width(width);
  track->set_height(height);

  if (!tracks_.AddTrack(track, number)) {
    delete track;
    return 0;
  }
  has_video_ = true;

  return track->number();
}

bool Segment::AddCuePoint(uint64_t timestamp, uint64_t track) {
  if (cluster_list_size_ < 1)
    return false;

  const Cluster* const cluster = cluster_list_[cluster_list_size_ - 1];
  if (!cluster)
    return false;

  CuePoint* const cue = new (std::nothrow) CuePoint();  // NOLINT
  if (!cue)
    return false;

  cue->set_time(timestamp / segment_info_.timecode_scale());
  cue->set_block_number(cluster->blocks_added());
  cue->set_cluster_pos(cluster->position_for_cues());
  cue->set_track(track);
  if (!cues_.AddCue(cue)) {
    delete cue;
    return false;
  }

  new_cuepoint_ = false;
  return true;
}

uint64_t Segment::AddAudioTrack(int32_t sample_rate, int32_t channels,
                                int32_t number) {
  AudioTrack* const track = new (std::nothrow) AudioTrack(&seed_);  // NOLINT
  if (!track)
    return 0;

  track->set_type(Tracks::kAudio);
  track->set_codec_id(Tracks::kVorbisCodecId);
  track->set_sample_rate(sample_rate);
  track->set_channels(channels);

  if (!tracks_.AddTrack(track, number)) {
    delete track;
    return 0;
  }

  return track->number();
}

bool Segment::AddFrame(const uint8_t* data, uint64_t length,
                       uint64_t track_number, uint64_t timestamp, bool is_key) {
  if (!data)
    return false;

  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_track_number(track_number);
  frame.set_timestamp(timestamp);
  frame.set_is_key(is_key);
  return AddGenericFrame(&frame);
}

bool Segment::AddFrameWithAdditional(const uint8_t* data, uint64_t length,
                                     const uint8_t* additional,
                                     uint64_t additional_length,
                                     uint64_t add_id, uint64_t track_number,
                                     uint64_t timestamp, bool is_key) {
  if (!data || !additional)
    return false;

  Frame frame;
  if (!frame.Init(data, length) ||
      !frame.AddAdditionalData(additional, additional_length, add_id)) {
    return false;
  }
  frame.set_track_number(track_number);
  frame.set_timestamp(timestamp);
  frame.set_is_key(is_key);
  return AddGenericFrame(&frame);
}

bool Segment::AddFrameWithDiscardPadding(const uint8_t* data, uint64_t length,
                                         int64_t discard_padding,
                                         uint64_t track_number,
                                         uint64_t timestamp, bool is_key) {
  if (!data)
    return false;

  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_discard_padding(discard_padding);
  frame.set_track_number(track_number);
  frame.set_timestamp(timestamp);
  frame.set_is_key(is_key);
  return AddGenericFrame(&frame);
}

bool Segment::AddMetadata(const uint8_t* data, uint64_t length,
                          uint64_t track_number, uint64_t timestamp_ns,
                          uint64_t duration_ns) {
  if (!data)
    return false;

  Frame frame;
  if (!frame.Init(data, length))
    return false;
  frame.set_track_number(track_number);
  frame.set_timestamp(timestamp_ns);
  frame.set_duration(duration_ns);
  frame.set_is_key(true);  // All metadata blocks are keyframes.
  return AddGenericFrame(&frame);
}

bool Segment::AddGenericFrame(const Frame* frame) {
  if (!frame)
    return false;

  if (!CheckHeaderInfo())
    return false;

  // Check for non-monotonically increasing timestamps.
  if (frame->timestamp() < last_timestamp_)
    return false;

  // Check if the track number is valid.
  if (!tracks_.GetTrackByNumber(frame->track_number()))
    return false;

  if (frame->discard_padding() != 0)
    doc_type_version_ = 4;

  if (cluster_list_size_ > 0) {
    const uint64_t timecode_scale = segment_info_.timecode_scale();
    const uint64_t frame_timecode = frame->timestamp() / timecode_scale;

    const Cluster* const last_cluster = cluster_list_[cluster_list_size_ - 1];
    const uint64_t last_cluster_timecode = last_cluster->timecode();

    const uint64_t rel_timecode = frame_timecode - last_cluster_timecode;
    if (rel_timecode > kMaxBlockTimecode) {
      force_new_cluster_ = true;
    }
  }

  // If the segment has a video track hold onto audio frames to make sure the
  // audio that is associated with the start time of a video key-frame is
  // muxed into the same cluster.
  if (has_video_ && tracks_.TrackIsAudio(frame->track_number()) &&
      !force_new_cluster_) {
    Frame* const new_frame = new (std::nothrow) Frame();
    if (!new_frame || !new_frame->CopyFrom(*frame)) {
      delete new_frame;
      return false;
    }
    if (!QueueFrame(new_frame)) {
      delete new_frame;
      return false;
    }
    track_frames_written_[frame->track_number() - 1]++;
    return true;
  }

  if (!DoNewClusterProcessing(frame->track_number(), frame->timestamp(),
                              frame->is_key())) {
    return false;
  }

  if (cluster_list_size_ < 1)
    return false;

  Cluster* const cluster = cluster_list_[cluster_list_size_ - 1];
  if (!cluster)
    return false;

  // If the Frame is not a SimpleBlock, then set the reference_block_timestamp
  // if it is not set already.
  bool frame_created = false;
  if (!frame->CanBeSimpleBlock() && !frame->is_key() &&
      !frame->reference_block_timestamp_set()) {
    Frame* const new_frame = new (std::nothrow) Frame();
    if (!new_frame || !new_frame->CopyFrom(*frame)) {
      delete new_frame;
      return false;
    }
    new_frame->set_reference_block_timestamp(
        last_track_timestamp_[frame->track_number() - 1]);
    frame = new_frame;
    frame_created = true;
  }

  if (!cluster->AddFrame(frame))
    return false;

  if (new_cuepoint_ && cues_track_ == frame->track_number()) {
    if (!AddCuePoint(frame->timestamp(), cues_track_))
      return false;
  }

  last_timestamp_ = frame->timestamp();
  last_track_timestamp_[frame->track_number() - 1] = frame->timestamp();
  last_block_duration_ = frame->duration();
  track_frames_written_[frame->track_number() - 1]++;

  if (frame_created)
    delete frame;
  return true;
}

void Segment::OutputCues(bool output_cues) { output_cues_ = output_cues; }

void Segment::AccurateClusterDuration(bool accurate_cluster_duration) {
  accurate_cluster_duration_ = accurate_cluster_duration;
}

void Segment::UseFixedSizeClusterTimecode(bool fixed_size_cluster_timecode) {
  fixed_size_cluster_timecode_ = fixed_size_cluster_timecode;
}

bool Segment::SetChunking(bool chunking, const char* filename) {
  if (chunk_count_ > 0)
    return false;

  if (chunking) {
    if (!filename)
      return false;

    // Check if we are being set to what is already set.
    if (chunking_ && !strcmp(filename, chunking_base_name_))
      return true;

    const size_t filename_length = strlen(filename);
    char* const temp = new (std::nothrow) char[filename_length + 1];  // NOLINT
    if (!temp)
      return false;

    memcpy(temp, filename, filename_length);
    temp[filename_length] = '\0';

    delete[] chunking_base_name_;
    chunking_base_name_ = temp;
    // From this point, strlen(chunking_base_name_) == filename_length

    if (!UpdateChunkName("chk", &chunk_name_))
      return false;

    if (!chunk_writer_cluster_) {
      chunk_writer_cluster_ = new (std::nothrow) MkvWriter();  // NOLINT
      if (!chunk_writer_cluster_)
        return false;
    }

    if (!chunk_writer_cues_) {
      chunk_writer_cues_ = new (std::nothrow) MkvWriter();  // NOLINT
      if (!chunk_writer_cues_)
        return false;
    }

    if (!chunk_writer_header_) {
      chunk_writer_header_ = new (std::nothrow) MkvWriter();  // NOLINT
      if (!chunk_writer_header_)
        return false;
    }

    if (!chunk_writer_cluster_->Open(chunk_name_))
      return false;

    const size_t hdr_length = strlen(".hdr");
    const size_t header_length = filename_length + hdr_length + 1;
    char* const header = new (std::nothrow) char[header_length];  // NOLINT
    if (!header)
      return false;

    memcpy(header, chunking_base_name_, filename_length);
    memcpy(&header[filename_length], ".hdr", hdr_length);
    header[filename_length + hdr_length] = '\0';

    if (!chunk_writer_header_->Open(header)) {
      delete[] header;
      return false;
    }

    writer_cluster_ = chunk_writer_cluster_;
    writer_cues_ = chunk_writer_cues_;
    writer_header_ = chunk_writer_header_;

    delete[] header;
  }

  chunking_ = chunking;

  return true;
}

bool Segment::CuesTrack(uint64_t track_number) {
  const Track* const track = GetTrackByNumber(track_number);
  if (!track)
    return false;

  cues_track_ = track_number;
  return true;
}

void Segment::ForceNewClusterOnNextFrame() { force_new_cluster_ = true; }

Track* Segment::GetTrackByNumber(uint64_t track_number) const {
  return tracks_.GetTrackByNumber(track_number);
}

bool Segment::WriteSegmentHeader() {
  UpdateDocTypeVersion();

  const char* const doc_type =
      DocTypeIsWebm() ? kDocTypeWebm : kDocTypeMatroska;
  if (!WriteEbmlHeader(writer_header_, doc_type_version_, doc_type))
    return false;
  doc_type_version_written_ = doc_type_version_;
  ebml_header_size_ = static_cast<int32_t>(writer_header_->Position());

  // Write "unknown" (-1) as segment size value. If mode is kFile, Segment
  // will write over duration when the file is finalized.
  if (WriteID(writer_header_, libwebm::kMkvSegment))
    return false;

  // Save for later.
  size_position_ = writer_header_->Position();

  // Write "unknown" (EBML coded -1) as segment size value. We need to write 8
  // bytes because if we are going to overwrite the segment size later we do
  // not know how big our segment will be.
  if (SerializeInt(writer_header_, kEbmlUnknownValue, 8))
    return false;

  payload_pos_ = writer_header_->Position();

  if (mode_ == kFile && writer_header_->Seekable()) {
    // Set the duration > 0.0 so SegmentInfo will write out the duration. When
    // the muxer is done writing we will set the correct duration and have
    // SegmentInfo upadte it.
    segment_info_.set_duration(1.0);

    if (!seek_head_.Write(writer_header_))
      return false;
  }

  if (!seek_head_.AddSeekEntry(libwebm::kMkvInfo, MaxOffset()))
    return false;
  if (!segment_info_.Write(writer_header_))
    return false;

  if (!seek_head_.AddSeekEntry(libwebm::kMkvTracks, MaxOffset()))
    return false;
  if (!tracks_.Write(writer_header_))
    return false;

  if (chapters_.Count() > 0) {
    if (!seek_head_.AddSeekEntry(libwebm::kMkvChapters, MaxOffset()))
      return false;
    if (!chapters_.Write(writer_header_))
      return false;
  }

  if (tags_.Count() > 0) {
    if (!seek_head_.AddSeekEntry(libwebm::kMkvTags, MaxOffset()))
      return false;
    if (!tags_.Write(writer_header_))
      return false;
  }

  if (chunking_ && (mode_ == kLive || !writer_header_->Seekable())) {
    if (!chunk_writer_header_)
      return false;

    chunk_writer_header_->Close();
  }

  header_written_ = true;

  return true;
}

// Here we are testing whether to create a new cluster, given a frame
// having time frame_timestamp_ns.
//
int Segment::TestFrame(uint64_t track_number, uint64_t frame_timestamp_ns,
                       bool is_key) const {
  if (force_new_cluster_)
    return 1;

  // If no clusters have been created yet, then create a new cluster
  // and write this frame immediately, in the new cluster.  This path
  // should only be followed once, the first time we attempt to write
  // a frame.

  if (cluster_list_size_ <= 0)
    return 1;

  // There exists at least one cluster. We must compare the frame to
  // the last cluster, in order to determine whether the frame is
  // written to the existing cluster, or that a new cluster should be
  // created.

  const uint64_t timecode_scale = segment_info_.timecode_scale();
  const uint64_t frame_timecode = frame_timestamp_ns / timecode_scale;

  const Cluster* const last_cluster = cluster_list_[cluster_list_size_ - 1];
  const uint64_t last_cluster_timecode = last_cluster->timecode();

  // For completeness we test for the case when the frame's timecode
  // is less than the cluster's timecode.  Although in principle that
  // is allowed, this muxer doesn't actually write clusters like that,
  // so this indicates a bug somewhere in our algorithm.

  if (frame_timecode < last_cluster_timecode)  // should never happen
    return -1;

  // If the frame has a timestamp significantly larger than the last
  // cluster (in Matroska, cluster-relative timestamps are serialized
  // using a 16-bit signed integer), then we cannot write this frame
  // to that cluster, and so we must create a new cluster.

  const int64_t delta_timecode = frame_timecode - last_cluster_timecode;

  if (delta_timecode > kMaxBlockTimecode)
    return 2;

  // We decide to create a new cluster when we have a video keyframe.
  // This will flush queued (audio) frames, and write the keyframe
  // immediately, in the newly-created cluster.

  if (is_key && tracks_.TrackIsVideo(track_number))
    return 1;

  // Create a new cluster if we have accumulated too many frames
  // already, where "too many" is defined as "the total time of frames
  // in the cluster exceeds a threshold".

  const uint64_t delta_ns = delta_timecode * timecode_scale;

  if (max_cluster_duration_ > 0 && delta_ns >= max_cluster_duration_)
    return 1;

  // This is similar to the case above, with the difference that a new
  // cluster is created when the size of the current cluster exceeds a
  // threshold.

  const uint64_t cluster_size = last_cluster->payload_size();

  if (max_cluster_size_ > 0 && cluster_size >= max_cluster_size_)
    return 1;

  // There's no need to create a new cluster, so emit this frame now.

  return 0;
}

bool Segment::MakeNewCluster(uint64_t frame_timestamp_ns) {
  const int32_t new_size = cluster_list_size_ + 1;

  if (new_size > cluster_list_capacity_) {
    // Add more clusters.
    const int32_t new_capacity =
        (cluster_list_capacity_ <= 0) ? 1 : cluster_list_capacity_ * 2;
    Cluster** const clusters =
        new (std::nothrow) Cluster*[new_capacity];  // NOLINT
    if (!clusters)
      return false;

    for (int32_t i = 0; i < cluster_list_size_; ++i) {
      clusters[i] = cluster_list_[i];
    }

    delete[] cluster_list_;

    cluster_list_ = clusters;
    cluster_list_capacity_ = new_capacity;
  }

  if (!WriteFramesLessThan(frame_timestamp_ns))
    return false;

  if (cluster_list_size_ > 0) {
    // Update old cluster's size
    Cluster* const old_cluster = cluster_list_[cluster_list_size_ - 1];

    if (!old_cluster || !old_cluster->Finalize(true, frame_timestamp_ns))
      return false;
  }

  if (output_cues_)
    new_cuepoint_ = true;

  if (chunking_ && cluster_list_size_ > 0) {
    chunk_writer_cluster_->Close();
    chunk_count_++;

    if (!UpdateChunkName("chk", &chunk_name_))
      return false;
    if (!chunk_writer_cluster_->Open(chunk_name_))
      return false;
  }

  const uint64_t timecode_scale = segment_info_.timecode_scale();
  const uint64_t frame_timecode = frame_timestamp_ns / timecode_scale;

  uint64_t cluster_timecode = frame_timecode;

  if (frames_size_ > 0) {
    const Frame* const f = frames_[0];  // earliest queued frame
    const uint64_t ns = f->timestamp();
    const uint64_t tc = ns / timecode_scale;

    if (tc < cluster_timecode)
      cluster_timecode = tc;
  }

  Cluster*& cluster = cluster_list_[cluster_list_size_];
  const int64_t offset = MaxOffset();
  cluster = new (std::nothrow)
      Cluster(cluster_timecode, offset, segment_info_.timecode_scale(),
              accurate_cluster_duration_, fixed_size_cluster_timecode_);
  if (!cluster)
    return false;

  if (!cluster->Init(writer_cluster_))
    return false;

  cluster_list_size_ = new_size;
  return true;
}

bool Segment::DoNewClusterProcessing(uint64_t track_number,
                                     uint64_t frame_timestamp_ns, bool is_key) {
  for (;;) {
    // Based on the characteristics of the current frame and current
    // cluster, decide whether to create a new cluster.
    const int result = TestFrame(track_number, frame_timestamp_ns, is_key);
    if (result < 0)  // error
      return false;

    // Always set force_new_cluster_ to false after TestFrame.
    force_new_cluster_ = false;

    // A non-zero result means create a new cluster.
    if (result > 0 && !MakeNewCluster(frame_timestamp_ns))
      return false;

    // Write queued (audio) frames.
    const int frame_count = WriteFramesAll();
    if (frame_count < 0)  // error
      return false;

    // Write the current frame to the current cluster (if TestFrame
    // returns 0) or to a newly created cluster (TestFrame returns 1).
    if (result <= 1)
      return true;

    // TestFrame returned 2, which means there was a large time
    // difference between the cluster and the frame itself.  Do the
    // test again, comparing the frame to the new cluster.
  }
}

bool Segment::CheckHeaderInfo() {
  if (!header_written_) {
    if (!WriteSegmentHeader())
      return false;

    if (!seek_head_.AddSeekEntry(libwebm::kMkvCluster, MaxOffset()))
      return false;

    if (output_cues_ && cues_track_ == 0) {
      // Check for a video track
      for (uint32_t i = 0; i < tracks_.track_entries_size(); ++i) {
        const Track* const track = tracks_.GetTrackByIndex(i);
        if (!track)
          return false;

        if (tracks_.TrackIsVideo(track->number())) {
          cues_track_ = track->number();
          break;
        }
      }

      // Set first track found
      if (cues_track_ == 0) {
        const Track* const track = tracks_.GetTrackByIndex(0);
        if (!track)
          return false;

        cues_track_ = track->number();
      }
    }
  }
  return true;
}

void Segment::UpdateDocTypeVersion() {
  for (uint32_t index = 0; index < tracks_.track_entries_size(); ++index) {
    const Track* track = tracks_.GetTrackByIndex(index);
    if (track == NULL)
      break;
    if ((track->codec_delay() || track->seek_pre_roll()) &&
        doc_type_version_ < 4) {
      doc_type_version_ = 4;
      break;
    }
  }
}

bool Segment::UpdateChunkName(const char* ext, char** name) const {
  if (!name || !ext)
    return false;

  char ext_chk[64];
#ifdef _MSC_VER
  sprintf_s(ext_chk, sizeof(ext_chk), "_%06d.%s", chunk_count_, ext);
#else
  snprintf(ext_chk, sizeof(ext_chk), "_%06d.%s", chunk_count_, ext);
#endif

  const size_t chunking_base_name_length = strlen(chunking_base_name_);
  const size_t ext_chk_length = strlen(ext_chk);
  const size_t length = chunking_base_name_length + ext_chk_length + 1;
  char* const str = new (std::nothrow) char[length];  // NOLINT
  if (!str)
    return false;

  memcpy(str, chunking_base_name_, chunking_base_name_length);
  memcpy(&str[chunking_base_name_length], ext_chk, ext_chk_length);
  str[chunking_base_name_length + ext_chk_length] = '\0';

  delete[] * name;
  *name = str;

  return true;
}

int64_t Segment::MaxOffset() {
  if (!writer_header_)
    return -1;

  int64_t offset = writer_header_->Position() - payload_pos_;

  if (chunking_) {
    for (int32_t i = 0; i < cluster_list_size_; ++i) {
      Cluster* const cluster = cluster_list_[i];
      offset += cluster->Size();
    }

    if (writer_cues_)
      offset += writer_cues_->Position();
  }

  return offset;
}

bool Segment::QueueFrame(Frame* frame) {
  const int32_t new_size = frames_size_ + 1;

  if (new_size > frames_capacity_) {
    // Add more frames.
    const int32_t new_capacity = (!frames_capacity_) ? 2 : frames_capacity_ * 2;

    if (new_capacity < 1)
      return false;

    Frame** const frames = new (std::nothrow) Frame*[new_capacity];  // NOLINT
    if (!frames)
      return false;

    for (int32_t i = 0; i < frames_size_; ++i) {
      frames[i] = frames_[i];
    }

    delete[] frames_;
    frames_ = frames;
    frames_capacity_ = new_capacity;
  }

  frames_[frames_size_++] = frame;

  return true;
}

int Segment::WriteFramesAll() {
  if (frames_ == NULL)
    return 0;

  if (cluster_list_size_ < 1)
    return -1;

  Cluster* const cluster = cluster_list_[cluster_list_size_ - 1];

  if (!cluster)
    return -1;

  for (int32_t i = 0; i < frames_size_; ++i) {
    Frame*& frame = frames_[i];
    // TODO(jzern/vigneshv): using Segment::AddGenericFrame here would limit the
    // places where |doc_type_version_| needs to be updated.
    if (frame->discard_padding() != 0)
      doc_type_version_ = 4;
    if (!cluster->AddFrame(frame)) {
      delete frame;
      continue;
    }

    if (new_cuepoint_ && cues_track_ == frame->track_number()) {
      if (!AddCuePoint(frame->timestamp(), cues_track_)) {
        delete frame;
        continue;
      }
    }

    if (frame->timestamp() > last_timestamp_) {
      last_timestamp_ = frame->timestamp();
      last_track_timestamp_[frame->track_number() - 1] = frame->timestamp();
    }

    delete frame;
    frame = NULL;
  }

  const int result = frames_size_;
  frames_size_ = 0;

  return result;
}

bool Segment::WriteFramesLessThan(uint64_t timestamp) {
  // Check |cluster_list_size_| to see if this is the first cluster. If it is
  // the first cluster the audio frames that are less than the first video
  // timesatmp will be written in a later step.
  if (frames_size_ > 0 && cluster_list_size_ > 0) {
    if (!frames_)
      return false;

    Cluster* const cluster = cluster_list_[cluster_list_size_ - 1];
    if (!cluster)
      return false;

    int32_t shift_left = 0;

    // TODO(fgalligan): Change this to use the durations of frames instead of
    // the next frame's start time if the duration is accurate.
    for (int32_t i = 1; i < frames_size_; ++i) {
      const Frame* const frame_curr = frames_[i];

      if (frame_curr->timestamp() > timestamp)
        break;

      const Frame* const frame_prev = frames_[i - 1];
      if (frame_prev->discard_padding() != 0)
        doc_type_version_ = 4;
      if (!cluster->AddFrame(frame_prev)) {
        delete frame_prev;
        continue;
      }

      if (new_cuepoint_ && cues_track_ == frame_prev->track_number()) {
        if (!AddCuePoint(frame_prev->timestamp(), cues_track_)) {
          delete frame_prev;
          continue;
        }
      }

      ++shift_left;
      if (frame_prev->timestamp() > last_timestamp_) {
        last_timestamp_ = frame_prev->timestamp();
        last_track_timestamp_[frame_prev->track_number() - 1] =
            frame_prev->timestamp();
      }

      delete frame_prev;
    }

    if (shift_left > 0) {
      if (shift_left >= frames_size_)
        return false;

      const int32_t new_frames_size = frames_size_ - shift_left;
      for (int32_t i = 0; i < new_frames_size; ++i) {
        frames_[i] = frames_[i + shift_left];
      }

      frames_size_ = new_frames_size;
    }
  }

  return true;
}

bool Segment::DocTypeIsWebm() const {
  const int kNumCodecIds = 9;

  // TODO(vigneshv): Tweak .clang-format.
  const char* kWebmCodecIds[kNumCodecIds] = {
      Tracks::kOpusCodecId,          Tracks::kVorbisCodecId,
      Tracks::kAv1CodecId,           Tracks::kVp8CodecId,
      Tracks::kVp9CodecId,           Tracks::kWebVttCaptionsId,
      Tracks::kWebVttDescriptionsId, Tracks::kWebVttMetadataId,
      Tracks::kWebVttSubtitlesId};

  const int num_tracks = static_cast<int>(tracks_.track_entries_size());
  for (int track_index = 0; track_index < num_tracks; ++track_index) {
    const Track* const track = tracks_.GetTrackByIndex(track_index);
    const std::string codec_id = track->codec_id();

    bool id_is_webm = false;
    for (int id_index = 0; id_index < kNumCodecIds; ++id_index) {
      if (codec_id == kWebmCodecIds[id_index]) {
        id_is_webm = true;
        break;
      }
    }

    if (!id_is_webm)
      return false;
  }

  return true;
}

}  // namespace mkvmuxer
