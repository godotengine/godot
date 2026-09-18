// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/decode.h>
#include <jxl/types.h>
#include <jxl/version.h>

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/padded_bytes.h"

// JPEGXL_ENABLE_BOXES, JPEGXL_ENABLE_TRANSCODE_JPEG
#include "lib/jxl/common.h"

#if JPEGXL_ENABLE_BOXES || JPEGXL_ENABLE_TRANSCODE_JPEG
#include "lib/jxl/box_content_decoder.h"
#endif
#include "lib/jxl/dec_frame.h"
#if JPEGXL_ENABLE_TRANSCODE_JPEG
#include "lib/jxl/decode_to_jpeg.h"
#endif
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/memory_manager_internal.h"

namespace {

// Checks if a + b > size, taking possible integer overflow into account.
bool OutOfBounds(size_t a, size_t b, size_t size) {
  size_t pos = a + b;
  if (pos > size) return true;
  if (pos < a) return true;  // overflow happened
  return false;
}

JXL_INLINE size_t InitialBasicInfoSizeHint() {
  // Amount of bytes before the start of the codestream in the container format,
  // assuming that the codestream is the first box after the signature and
  // filetype boxes. 12 bytes signature box + 20 bytes filetype box + 16 bytes
  // codestream box length + name + optional XLBox length.
  const size_t container_header_size = 48;

  // Worst-case amount of bytes for basic info of the JPEG XL codestream header,
  // that is all information up to and including extra_channel_bits. Up to
  // around 2 bytes signature + 8 bytes SizeHeader + 31 bytes ColorEncoding + 4
  // bytes rest of ImageMetadata + 5 bytes part of ImageMetadata2.
  // TODO(lode): recompute and update this value when alpha_bits is moved to
  // extra channels info.
  const size_t max_codestream_basic_info_size = 50;

  return container_header_size + max_codestream_basic_info_size;
}

// Debug-printing failure macro similar to JXL_FAILURE, but for the status code
// JXL_DEC_ERROR
#if (JXL_CRASH_ON_ERROR)
#define JXL_API_ERROR(format, ...)                                           \
  (::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__), \
   ::jxl::Abort(), JXL_DEC_ERROR)
#else  // JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                             \
  (((JXL_IS_DEBUG_BUILD) &&                                                    \
    ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__)), \
   JXL_DEC_ERROR)
#endif  // JXL_CRASH_ON_ERROR

// Error caused by bad input (invalid file) rather than incorrect API usage.
// For now there is no way to distinguish these two types of errors yet.
#define JXL_INPUT_ERROR(format, ...) JXL_API_ERROR(format, ##__VA_ARGS__)

JxlDecoderStatus ConvertStatus(JxlDecoderStatus status) { return status; }

JxlDecoderStatus ConvertStatus(jxl::Status status) {
  return status ? JXL_DEC_SUCCESS : JXL_DEC_ERROR;
}

#define JXL_API_RETURN_IF_ERROR(expr)               \
  {                                                 \
    JxlDecoderStatus status_ = ConvertStatus(expr); \
    if (status_ != JXL_DEC_SUCCESS) return status_; \
  }

JxlSignature ReadSignature(const uint8_t* buf, size_t len, size_t* pos) {
  if (*pos >= len) return JXL_SIG_NOT_ENOUGH_BYTES;

  buf += *pos;
  len -= *pos;

  // JPEG XL codestream: 0xff 0x0a
  if (len >= 1 && buf[0] == 0xff) {
    if (len < 2) {
      return JXL_SIG_NOT_ENOUGH_BYTES;
    } else if (buf[1] == jxl::kCodestreamMarker) {
      *pos += 2;
      return JXL_SIG_CODESTREAM;
    } else {
      return JXL_SIG_INVALID;
    }
  }

  // JPEG XL container
  if (len >= 1 && buf[0] == 0) {
    if (len < 12) {
      return JXL_SIG_NOT_ENOUGH_BYTES;
    } else if (buf[1] == 0 && buf[2] == 0 && buf[3] == 0xC && buf[4] == 'J' &&
               buf[5] == 'X' && buf[6] == 'L' && buf[7] == ' ' &&
               buf[8] == 0xD && buf[9] == 0xA && buf[10] == 0x87 &&
               buf[11] == 0xA) {
      *pos += 12;
      return JXL_SIG_CONTAINER;
    } else {
      return JXL_SIG_INVALID;
    }
  }

  return JXL_SIG_INVALID;
}

}  // namespace

uint32_t JxlDecoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

JxlSignature JxlSignatureCheck(const uint8_t* buf, size_t len) {
  size_t pos = 0;
  return ReadSignature(buf, len, &pos);
}

namespace {

size_t BitsPerChannel(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 8;
    case JXL_TYPE_UINT16:
      return 16;
    case JXL_TYPE_FLOAT:
      return 32;
    case JXL_TYPE_FLOAT16:
      return 16;
    default:
      return 0;  // signals unhandled JxlDataType
  }
}

template <typename T>
uint32_t GetBitDepth(JxlBitDepth bit_depth, const T& metadata,
                     JxlPixelFormat format) {
  if (bit_depth.type == JXL_BIT_DEPTH_FROM_PIXEL_FORMAT) {
    return BitsPerChannel(format.data_type);
  } else if (bit_depth.type == JXL_BIT_DEPTH_FROM_CODESTREAM) {
    return metadata.bit_depth.bits_per_sample;
  } else if (bit_depth.type == JXL_BIT_DEPTH_CUSTOM) {
    return bit_depth.bits_per_sample;
  }
  return 0;
}

enum class DecoderStage : uint32_t {
  kInited,              // Decoder created, no JxlDecoderProcessInput called yet
  kStarted,             // Running JxlDecoderProcessInput calls
  kCodestreamFinished,  // Codestream done, but other boxes could still occur.
                        // This stage can also occur before having seen the
                        // entire codestream if the user didn't subscribe to any
                        // codestream events at all, e.g. only to box events,
                        // or, the user only subscribed to basic info, and only
                        // the header of the codestream was parsed.
  kError,               // Error occurred, decoder object no longer usable
};

enum class FrameStage : uint32_t {
  kHeader,  // Must parse frame header.
  kTOC,     // Must parse TOC
  kFull,    // Must parse full pixels
};

enum class BoxStage : uint32_t {
  kHeader,      // Parsing box header of the next box, or start of non-container
                // stream
  kFtyp,        // The ftyp box
  kSkip,        // Box whose contents are skipped
  kCodestream,  // Handling codestream box contents, or non-container stream
  kPartialCodestream,  // Handling the extra header of partial codestream box
  kJpegRecon,          // Handling jpeg reconstruction box
};

enum class JpegReconStage : uint32_t {
  kNone,             // Not outputting
  kSettingMetadata,  // Ready to output, must set metadata to the jpeg_data
  kOutputting,       // Currently outputting the JPEG bytes
};

// For each internal frame, which storage locations it references, and which
// storage locations it is stored in, using the bit mask as defined in
// FrameDecoder::References and FrameDecoder::SaveAs.
typedef struct FrameRef {
  int reference;
  int saved_as;
} FrameRef;

/*
Given list of frame references to storage slots, and storage slots in which this
frame is saved, computes which frames are required to decode the frame at the
given index and any frames after it. The frames on which this depends are
returned as a vector of their indices, in no particular order. The given index
must be smaller than saved_as.size(), and references.size() must equal
saved_as.size(). Any frames beyond saved_as and references are considered
unknown future frames and must be treated as if something depends on them.
*/
std::vector<size_t> GetFrameDependencies(size_t index,
                                         const std::vector<FrameRef>& refs) {
  JXL_DASSERT(index < refs.size());

  std::vector<size_t> result;

  constexpr size_t kNumStorage = 8;

  // value which indicates nothing is stored in this storage slot
  const size_t invalid = refs.size();
  // for each of the 8 storage slots, a vector that translates frame index to
  // frame stored in this storage slot at this point, that is, the last
  // frame that was stored in this slot before or at this index.
  std::array<std::vector<size_t>, kNumStorage> storage;
  for (size_t s = 0; s < kNumStorage; ++s) {
    storage[s].resize(refs.size());
    int mask = 1 << s;
    size_t id = invalid;
    for (size_t i = 0; i < refs.size(); ++i) {
      if (refs[i].saved_as & mask) {
        id = i;
      }
      storage[s][i] = id;
    }
  }

  std::vector<char> seen(index + 1, 0);
  std::vector<size_t> stack;
  stack.push_back(index);
  seen[index] = 1;

  // For frames after index, assume they can depend on any of the 8 storage
  // slots, so push the frame for each stored reference to the stack and result.
  // All frames after index are treated as having unknown references and with
  // the possibility that there are more frames after the last known.
  // TODO(lode): take values of saved_as and references after index, and a
  // input flag indicating if they are all frames of the image, to further
  // optimize this.
  for (size_t s = 0; s < kNumStorage; ++s) {
    size_t frame_ref = storage[s][index];
    if (frame_ref == invalid) continue;
    if (seen[frame_ref]) continue;
    stack.push_back(frame_ref);
    seen[frame_ref] = 1;
    result.push_back(frame_ref);
  }

  while (!stack.empty()) {
    size_t frame_index = stack.back();
    stack.pop_back();
    if (frame_index == 0) continue;  // first frame cannot have references
    for (size_t s = 0; s < kNumStorage; ++s) {
      int mask = 1 << s;
      if (!(refs[frame_index].reference & mask)) continue;
      size_t frame_ref = storage[s][frame_index - 1];
      if (frame_ref == invalid) continue;
      if (seen[frame_ref]) continue;
      stack.push_back(frame_ref);
      seen[frame_ref] = 1;
      result.push_back(frame_ref);
    }
  }

  return result;
}

// Parameters for user-requested extra channel output.
struct ExtraChannelOutput {
  JxlPixelFormat format;
  void* buffer;
  size_t buffer_size;
};

}  // namespace

namespace jxl {

typedef struct JxlDecoderFrameIndexBoxEntryStruct {
  // OFFi: offset of start byte of this frame compared to start
  // byte of previous frame from this index in the JPEG XL codestream. For the
  // first frame, this is the offset from the first byte of the JPEG XL
  // codestream.
  uint64_t OFFi;
  // Ti: duration in ticks between the start of this frame and
  // the start of the next frame in the index. If this is the last frame in the
  // index, this is the duration in ticks between the start of this frame and
  // the end of the stream. A tick lasts TNUM / TDEN seconds.
  uint32_t Ti;
  // Fi: amount of frames the next frame in the index occurs
  // after this frame. If this is the last frame in the index, this is the
  // amount of frames after this frame in the remainder of the stream. Only
  // frames that are presented by the decoder are counted for this purpose, this
  // excludes frames that are not intended for display but for compositing with
  // other frames, such as frames that aren't the last frame with a duration of
  // 0 ticks.
  uint32_t Fi;
} JxlDecoderFrameIndexBoxEntry;

typedef struct JxlDecoderFrameIndexBoxStruct {
  int64_t NF() const { return entries.size(); }
  int32_t TNUM = 1;
  int32_t TDEN = 1000;

  std::vector<JxlDecoderFrameIndexBoxEntry> entries;

  // That way we can ensure that every index box will have the first frame.
  // If the API user decides to mark it as an indexed frame, we call
  // the AddFrame again, this time with requested.
  void AddFrame(uint64_t OFFi, uint32_t Ti, uint32_t Fi) {
    JxlDecoderFrameIndexBoxEntry e;
    e.OFFi = OFFi;
    e.Ti = Ti;
    e.Fi = Fi;
    entries.push_back(e);
  }
} JxlDecoderFrameIndexBox;

}  // namespace jxl

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct JxlDecoderStruct {
  JxlDecoderStruct() = default;

  JxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;

  DecoderStage stage;

  // Status of progression, internal.
  bool got_signature;
  // Indicates we know that we've seen the last codestream box: either this
  // was a jxlc box, or a jxlp box that has its index indicated as last by
  // having its most significant bit set, or no boxes are used at all. This
  // does not indicate the full codestream has already been seen, only the
  // last box of it has been initiated.
  bool last_codestream_seen;
  bool got_codestream_signature;
  bool got_basic_info;
  bool got_transform_data;  // To skip everything before ICC.
  bool got_all_headers;     // Codestream metadata headers.
  bool post_headers;        // Already decoding pixels.
  std::unique_ptr<jxl::ICCReader> icc_reader;
  jxl::JxlDecoderFrameIndexBox frame_index_box;
  // This means either we actually got the preview image, or determined we
  // cannot get it or there is none.
  bool got_preview_image;
  bool preview_frame;

  // Position of next_in in the original file including box format if present
  // (as opposed to position in the codestream)
  size_t file_pos;

  size_t box_contents_begin;
  size_t box_contents_end;
  size_t box_contents_size;
  size_t box_size;
  size_t header_size;
  // Either a final box that runs until EOF, or the case of no container format
  // at all.
  bool box_contents_unbounded;

  JxlBoxType box_type;
  JxlBoxType box_decoded_type;  // Underlying type for brob boxes
  // Set to true right after a JXL_DEC_BOX event only.
  bool box_event;
  bool decompress_boxes;

  bool box_out_buffer_set;
  // Whether the out buffer is set for the current box, if the user did not yet
  // release the buffer while the next box is encountered, this will be set to
  // false. If this is false, no JXL_DEC_NEED_MORE_INPUT is emitted
  // (irrespective of the value of box_out_buffer_set), because not setting
  // output indicates the user does not wish the data of this box.
  bool box_out_buffer_set_current_box;
  uint8_t* box_out_buffer;
  size_t box_out_buffer_size;
  // which byte of the full box content the start of the out buffer points to
  size_t box_out_buffer_begin;
  // which byte of box_out_buffer to write to next
  size_t box_out_buffer_pos;

  // Settings
  bool keep_orientation;
  bool unpremul_alpha;
  bool render_spotcolors;
  bool coalescing;
  float desired_intensity_target;

  // Bitfield, for which informative events (JXL_DEC_BASIC_INFO, etc...) the
  // decoder returns a status. By default, do not return for any of the events,
  // only return when the decoder cannot continue because it needs more input or
  // output data.
  int events_wanted;
  int orig_events_wanted;

  // Fields for reading the basic info from the header.
  size_t basic_info_size_hint;
  bool have_container;
  size_t box_count;

  // The level of progressive detail in frame decoding.
  JxlProgressiveDetail prog_detail = kDC;
  // The progressive detail of the current frame.
  JxlProgressiveDetail frame_prog_detail;
  // The intended downsampling ratio for the current progression step.
  size_t downsampling_target;

  // Set to true if either an image out buffer or an image out callback was set.
  bool image_out_buffer_set;

  // Owned by the caller, buffer for preview or full resolution image.
  void* image_out_buffer;
  JxlImageOutInitCallback image_out_init_callback;
  JxlImageOutRunCallback image_out_run_callback;
  JxlImageOutDestroyCallback image_out_destroy_callback;
  void* image_out_init_opaque;
  struct SimpleImageOutCallback {
    JxlImageOutCallback callback;
    void* opaque;
  };
  SimpleImageOutCallback simple_image_out_callback;

  size_t image_out_size;

  JxlPixelFormat image_out_format;
  JxlBitDepth image_out_bit_depth;

  // For extra channels. Empty if no extra channels are requested, and they are
  // reset each frame
  std::vector<ExtraChannelOutput> extra_channel_output;

  jxl::CodecMetadata metadata;
  // Same as metadata.m, except for the color_encoding, which is set to the
  // output encoding.
  jxl::ImageMetadata image_metadata;
  std::unique_ptr<jxl::ImageBundle> ib;

  std::unique_ptr<jxl::PassesDecoderState> passes_state;
  std::unique_ptr<jxl::FrameDecoder> frame_dec;
  size_t next_section;
  std::vector<char> section_processed;

  // headers and TOC for the current frame. When got_toc is true, this is
  // always the frame header of the last frame of the current still series,
  // that is, the displayed frame.
  std::unique_ptr<jxl::FrameHeader> frame_header;

  size_t remaining_frame_size;
  FrameStage frame_stage;
  bool dc_frame_progression_done;
  // The currently processed frame is the last of the current composite still,
  // and so must be returned as pixels
  bool is_last_of_still;
  // The currently processed frame is the last of the codestream
  bool is_last_total;
  // How many frames to skip.
  size_t skip_frames;
  // Skipping the current frame. May be false if skip_frames was just set to
  // a positive value while already processing a current frame, then
  // skipping_frame will be enabled only for the next frame.
  bool skipping_frame;

  // Amount of internal frames and external frames started. External frames are
  // user-visible frames, internal frames includes all external frames and
  // also invisible frames such as patches, blending-only and dc_level frames.
  size_t internal_frames;
  size_t external_frames;

  std::vector<FrameRef> frame_refs;

  // Translates external frame index to internal frame index. The external
  // index is the index of user-visible frames. The internal index can be larger
  // since non-visible frames (such as frames with patches, ...) are included.
  std::vector<size_t> frame_external_to_internal;

  // Whether the frame with internal index is required to decode the frame
  // being skipped to or any frames after that. If no skipping is active,
  // this vector is ignored. If the current internal frame index is beyond this
  // vector, it must be treated as a required frame.
  std::vector<char> frame_required;

  // Codestream input data is copied here temporarily when the decoder needs
  // more input bytes to process the next part of the stream. We copy the input
  // data in order to be able to release it all through the API it when
  // returning JXL_DEC_NEED_MORE_INPUT.
  std::vector<uint8_t> codestream_copy;
  // Number of bytes at the end of codestream_copy that were not yet consumed
  // by calling AdvanceInput().
  size_t codestream_unconsumed;
  // Position in the codestream_copy vector that the decoder already finished
  // processing. It can be greater than the current size of codestream_copy in
  // case where the decoder skips some parts of the frame that were not yet
  // provided.
  size_t codestream_pos;
  // Number of bits after codestream_pos that were already processed.
  size_t codestream_bits_ahead;

  BoxStage box_stage;

#if JPEGXL_ENABLE_BOXES
  jxl::JxlBoxContentDecoder box_content_decoder;
#endif
#if JPEGXL_ENABLE_TRANSCODE_JPEG
  jxl::JxlToJpegDecoder jpeg_decoder;
  // Decodes Exif or XMP metadata for JPEG reconstruction
  jxl::JxlBoxContentDecoder metadata_decoder;
  std::vector<uint8_t> exif_metadata;
  std::vector<uint8_t> xmp_metadata;
  // must store JPEG reconstruction metadata from the current box
  // 0 = not stored, 1 = currently storing, 2 = finished
  int store_exif;
  int store_xmp;
  size_t recon_out_buffer_pos;
  size_t recon_exif_size;  // Expected exif size as read from the jbrd box
  size_t recon_xmp_size;   // Expected exif size as read from the jbrd box
  JpegReconStage recon_output_jpeg;

  bool JbrdNeedMoreBoxes() const {
    // jbrd box wants exif but exif box not yet seen
    if (store_exif < 2 && recon_exif_size > 0) return true;
    // jbrd box wants xmp but xmp box not yet seen
    if (store_xmp < 2 && recon_xmp_size > 0) return true;
    return false;
  }
#endif

  const uint8_t* next_in;
  size_t avail_in;
  bool input_closed;

  void AdvanceInput(size_t size) {
    JXL_DASSERT(avail_in >= size);
    next_in += size;
    avail_in -= size;
    file_pos += size;
  }

  size_t AvailableCodestream() const {
    size_t avail_codestream = avail_in;
    if (!box_contents_unbounded) {
      avail_codestream =
          std::min<size_t>(avail_codestream, box_contents_end - file_pos);
    }
    return avail_codestream;
  }

  void AdvanceCodestream(size_t size) {
    size_t avail_codestream = AvailableCodestream();
    if (codestream_copy.empty()) {
      if (size <= avail_codestream) {
        AdvanceInput(size);
      } else {
        codestream_pos = size - avail_codestream;
        AdvanceInput(avail_codestream);
      }
    } else {
      codestream_pos += size;
      if (codestream_pos + codestream_unconsumed >= codestream_copy.size()) {
        size_t advance = std::min(
            codestream_unconsumed,
            codestream_unconsumed + codestream_pos - codestream_copy.size());
        AdvanceInput(advance);
        codestream_pos -= std::min(codestream_pos, codestream_copy.size());
        codestream_unconsumed = 0;
        codestream_copy.clear();
      }
    }
  }

  JxlDecoderStatus RequestMoreInput() {
    if (codestream_copy.empty()) {
      size_t avail_codestream = AvailableCodestream();
      codestream_copy.insert(codestream_copy.end(), next_in,
                             next_in + avail_codestream);
      AdvanceInput(avail_codestream);
    } else {
      AdvanceInput(codestream_unconsumed);
      codestream_unconsumed = 0;
    }
    return JXL_DEC_NEED_MORE_INPUT;
  }

  JxlDecoderStatus GetCodestreamInput(jxl::Span<const uint8_t>* span) {
    if (codestream_copy.empty() && codestream_pos > 0) {
      size_t avail_codestream = AvailableCodestream();
      size_t skip = std::min<size_t>(codestream_pos, avail_codestream);
      AdvanceInput(skip);
      codestream_pos -= skip;
      if (codestream_pos > 0) {
        return RequestMoreInput();
      }
    }
    if (codestream_pos > codestream_copy.size()) {
      return JXL_API_ERROR("Internal: codestream_pos > codestream_copy.size()");
    }
    if (codestream_unconsumed > codestream_copy.size()) {
      return JXL_API_ERROR(
          "Internal: codestream_unconsumed > codestream_copy.size()");
    }
    size_t avail_codestream = AvailableCodestream();
    if (codestream_copy.empty()) {
      if (avail_codestream == 0) {
        return RequestMoreInput();
      }
      *span = jxl::Bytes(next_in, avail_codestream);
      return JXL_DEC_SUCCESS;
    } else {
      codestream_copy.insert(codestream_copy.end(),
                             next_in + codestream_unconsumed,
                             next_in + avail_codestream);
      codestream_unconsumed = avail_codestream;
      *span = jxl::Bytes(codestream_copy.data() + codestream_pos,
                         codestream_copy.size() - codestream_pos);
      return JXL_DEC_SUCCESS;
    }
  }

  // Whether the decoder can use more codestream input for a purpose it needs.
  // This returns false if the user didn't subscribe to any events that
  // require the codestream (e.g. only subscribed to metadata boxes), or all
  // parts of the codestream that are subscribed to (e.g. only basic info) have
  // already occurred.
  bool CanUseMoreCodestreamInput() const {
    // The decoder can set this to finished early if all relevant events were
    // processed, so this check works.
    return stage != DecoderStage::kCodestreamFinished;
  }
};

namespace {

bool CheckSizeLimit(JxlDecoder* dec, size_t xsize, size_t ysize) {
  if (xsize == 0 || ysize == 0) return true;
  size_t padded_xsize = jxl::DivCeil(xsize, 32) * 32;
  if (padded_xsize < xsize) return false;  // overflow
  size_t num_pixels = padded_xsize * ysize;
  if (num_pixels / padded_xsize != ysize) return false;  // overflow
  return true;
}

}  // namespace

// Resets the state that must be reset for both Rewind and Reset
void JxlDecoderRewindDecodingState(JxlDecoder* dec) {
  dec->stage = DecoderStage::kInited;
  dec->got_signature = false;
  dec->last_codestream_seen = false;
  dec->got_codestream_signature = false;
  dec->got_basic_info = false;
  dec->got_transform_data = false;
  dec->got_all_headers = false;
  dec->post_headers = false;
  if (dec->icc_reader) dec->icc_reader->Reset();
  dec->got_preview_image = false;
  dec->preview_frame = false;
  dec->file_pos = 0;
  dec->box_contents_begin = 0;
  dec->box_contents_end = 0;
  dec->box_contents_size = 0;
  dec->box_size = 0;
  dec->header_size = 0;
  dec->box_contents_unbounded = false;
  memset(dec->box_type, 0, sizeof(dec->box_type));
  memset(dec->box_decoded_type, 0, sizeof(dec->box_decoded_type));
  dec->box_event = false;
  dec->box_stage = BoxStage::kHeader;
  dec->box_out_buffer_set = false;
  dec->box_out_buffer_set_current_box = false;
  dec->box_out_buffer = nullptr;
  dec->box_out_buffer_size = 0;
  dec->box_out_buffer_begin = 0;
  dec->box_out_buffer_pos = 0;

#if JPEGXL_ENABLE_TRANSCODE_JPEG
  dec->exif_metadata.clear();
  dec->xmp_metadata.clear();
  dec->store_exif = 0;
  dec->store_xmp = 0;
  dec->recon_out_buffer_pos = 0;
  dec->recon_exif_size = 0;
  dec->recon_xmp_size = 0;
  dec->recon_output_jpeg = JpegReconStage::kNone;
#endif

  dec->events_wanted = dec->orig_events_wanted;
  dec->basic_info_size_hint = InitialBasicInfoSizeHint();
  dec->have_container = false;
  dec->box_count = 0;
  dec->downsampling_target = 8;
  dec->image_out_buffer_set = false;
  dec->image_out_buffer = nullptr;
  dec->image_out_init_callback = nullptr;
  dec->image_out_run_callback = nullptr;
  dec->image_out_destroy_callback = nullptr;
  dec->image_out_init_opaque = nullptr;
  dec->image_out_size = 0;
  dec->image_out_bit_depth.type = JXL_BIT_DEPTH_FROM_PIXEL_FORMAT;
  dec->extra_channel_output.clear();
  dec->next_in = nullptr;
  dec->avail_in = 0;
  dec->input_closed = false;

  dec->passes_state.reset();
  dec->frame_dec.reset();
  dec->next_section = 0;
  dec->section_processed.clear();

  dec->ib.reset();
  dec->metadata = jxl::CodecMetadata();
  dec->image_metadata = dec->metadata.m;
  dec->frame_header = jxl::make_unique<jxl::FrameHeader>(&dec->metadata);

  dec->codestream_copy.clear();
  dec->codestream_unconsumed = 0;
  dec->codestream_pos = 0;
  dec->codestream_bits_ahead = 0;

  dec->frame_stage = FrameStage::kHeader;
  dec->remaining_frame_size = 0;
  dec->is_last_of_still = false;
  dec->is_last_total = false;
  dec->skip_frames = 0;
  dec->skipping_frame = false;
  dec->internal_frames = 0;
  dec->external_frames = 0;
}

void JxlDecoderReset(JxlDecoder* dec) {
  JxlDecoderRewindDecodingState(dec);

  dec->thread_pool.reset();
  dec->keep_orientation = false;
  dec->unpremul_alpha = false;
  dec->render_spotcolors = true;
  dec->coalescing = true;
  dec->desired_intensity_target = 0;
  dec->orig_events_wanted = 0;
  dec->events_wanted = 0;
  dec->frame_refs.clear();
  dec->frame_external_to_internal.clear();
  dec->frame_required.clear();
  dec->decompress_boxes = false;
}

JxlDecoder* JxlDecoderCreate(const JxlMemoryManager* memory_manager) {
  JxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager))
    return nullptr;

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JxlDecoder));
  if (!alloc) return nullptr;
  // Placement new constructor on allocated memory
  JxlDecoder* dec = new (alloc) JxlDecoder();
  dec->memory_manager = local_memory_manager;

  JxlDecoderReset(dec);

  return dec;
}

void JxlDecoderDestroy(JxlDecoder* dec) {
  if (dec) {
    JxlMemoryManager local_memory_manager = dec->memory_manager;
    // Call destructor directly since custom free function is used.
    dec->~JxlDecoder();
    jxl::MemoryManagerFree(&local_memory_manager, dec);
  }
}

void JxlDecoderRewind(JxlDecoder* dec) { JxlDecoderRewindDecodingState(dec); }

void JxlDecoderSkipFrames(JxlDecoder* dec, size_t amount) {
  // Increment amount, rather than set it: making the amount smaller is
  // impossible because the decoder may already have skipped frames required to
  // decode earlier frames, and making the amount larger compared to an existing
  // amount is impossible because if JxlDecoderSkipFrames is called in the
  // middle of already skipping frames, the user cannot know how many frames
  // have already been skipped internally so far so an absolute value cannot
  // be defined.
  dec->skip_frames += amount;

  dec->frame_required.clear();
  size_t next_frame = dec->external_frames + dec->skip_frames;

  // A frame that has been seen before a rewind
  if (next_frame < dec->frame_external_to_internal.size()) {
    size_t internal_index = dec->frame_external_to_internal[next_frame];
    if (internal_index < dec->frame_refs.size()) {
      std::vector<size_t> deps =
          GetFrameDependencies(internal_index, dec->frame_refs);

      dec->frame_required.resize(internal_index + 1, 0);
      for (size_t idx : deps) {
        if (idx < dec->frame_required.size()) {
          dec->frame_required[idx] = 1;
        } else {
          JXL_DEBUG_ABORT("Unreachable");
        }
      }
    }
  }
}

JxlDecoderStatus JxlDecoderSkipCurrentFrame(JxlDecoder* dec) {
  if (dec->frame_stage != FrameStage::kFull) {
    return JXL_API_ERROR("JxlDecoderSkipCurrentFrame called at the wrong time");
  }
  JXL_DASSERT(dec->frame_dec);
  dec->frame_stage = FrameStage::kHeader;
  dec->AdvanceCodestream(dec->remaining_frame_size);
  if (dec->is_last_of_still) {
    dec->image_out_buffer_set = false;
  }
  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus
JxlDecoderSetParallelRunner(JxlDecoder* dec, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR(
        "JxlDecoderSetParallelRunner must be called before starting");
  }
  dec->thread_pool = jxl::make_unique<jxl::ThreadPool>(parallel_runner,
                                                       parallel_runner_opaque);
  return JXL_DEC_SUCCESS;
}

size_t JxlDecoderSizeHintBasicInfo(const JxlDecoder* dec) {
  if (dec->got_basic_info) return 0;
  return dec->basic_info_size_hint;
}

JxlDecoderStatus JxlDecoderSubscribeEvents(JxlDecoder* dec, int events_wanted) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_DEC_ERROR;  // Cannot subscribe to events after having started.
  }
  if (events_wanted & 63) {
    return JXL_DEC_ERROR;  // Can only subscribe to informative events.
  }
  dec->events_wanted = events_wanted;
  dec->orig_events_wanted = events_wanted;
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetKeepOrientation(JxlDecoder* dec,
                                              JXL_BOOL skip_reorientation) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must set keep_orientation option before starting");
  }
  dec->keep_orientation = FROM_JXL_BOOL(skip_reorientation);
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetUnpremultiplyAlpha(JxlDecoder* dec,
                                                 JXL_BOOL unpremul_alpha) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must set unpremul_alpha option before starting");
  }
  dec->unpremul_alpha = FROM_JXL_BOOL(unpremul_alpha);
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetRenderSpotcolors(JxlDecoder* dec,
                                               JXL_BOOL render_spotcolors) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must set render_spotcolors option before starting");
  }
  dec->render_spotcolors = FROM_JXL_BOOL(render_spotcolors);
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetCoalescing(JxlDecoder* dec, JXL_BOOL coalescing) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must set coalescing option before starting");
  }
  dec->coalescing = FROM_JXL_BOOL(coalescing);
  return JXL_DEC_SUCCESS;
}

namespace {
// helper function to get the dimensions of the current image buffer
void GetCurrentDimensions(const JxlDecoder* dec, size_t& xsize, size_t& ysize) {
  if (dec->frame_header->nonserialized_is_preview) {
    xsize = dec->metadata.oriented_preview_xsize(dec->keep_orientation);
    ysize = dec->metadata.oriented_preview_ysize(dec->keep_orientation);
    return;
  }
  xsize = dec->metadata.oriented_xsize(dec->keep_orientation);
  ysize = dec->metadata.oriented_ysize(dec->keep_orientation);
  if (!dec->coalescing) {
    const auto frame_dim = dec->frame_header->ToFrameDimensions();
    xsize = frame_dim.xsize_upsampled;
    ysize = frame_dim.ysize_upsampled;
    if (!dec->keep_orientation &&
        static_cast<int>(dec->metadata.m.GetOrientation()) > 4) {
      std::swap(xsize, ysize);
    }
  }
}
}  // namespace

namespace jxl {
namespace {

// Returns JXL_DEC_SUCCESS if the full bundle was successfully read, status
// indicating either error or need more input otherwise.
template <class T>
JxlDecoderStatus ReadBundle(JxlDecoder* dec, Span<const uint8_t> data,
                            BitReader* reader, T* JXL_RESTRICT t) {
  // Use a copy of the bit reader because CanRead advances bits.
  BitReader reader2(data);
  reader2.SkipBits(reader->TotalBitsConsumed());
  bool can_read = Bundle::CanRead(&reader2, t);
  JXL_API_RETURN_IF_ERROR(reader2.Close());

  if (!can_read) {
    return dec->RequestMoreInput();
  }
  if (!Bundle::Read(reader, t)) {
    return JXL_DEC_ERROR;
  }
  return JXL_DEC_SUCCESS;
}

std::unique_ptr<BitReader, std::function<void(BitReader*)>> GetBitReader(
    Span<const uint8_t> span) {
  BitReader* reader = new BitReader(span);
  return std::unique_ptr<BitReader, std::function<void(BitReader*)>>(
      reader, [](BitReader* reader) {
        // We can't allow Close to abort the program if the reader is out of
        // bounds, or all return paths in the code, even those that already
        // return failure, would have to manually call AllReadsWithinBounds().
        // Invalid JXL codestream should not cause program to quit.
        (void)reader->AllReadsWithinBounds();
        (void)reader->Close();
        delete reader;
      });
}

JxlDecoderStatus JxlDecoderReadBasicInfo(JxlDecoder* dec) {
  if (!dec->got_codestream_signature) {
    // Check and skip the codestream signature
    Span<const uint8_t> span;
    JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
    if (span.size() < 2) {
      return dec->RequestMoreInput();
    }
    if (span.data()[0] != 0xff || span.data()[1] != jxl::kCodestreamMarker) {
      return JXL_INPUT_ERROR("invalid signature");
    }
    dec->got_codestream_signature = true;
    dec->AdvanceCodestream(2);
  }

  Span<const uint8_t> span;
  JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
  auto reader = GetBitReader(span);
  JXL_API_RETURN_IF_ERROR(
      ReadBundle(dec, span, reader.get(), &dec->metadata.size));
  JXL_API_RETURN_IF_ERROR(
      ReadBundle(dec, span, reader.get(), &dec->metadata.m));
  size_t total_bits = reader->TotalBitsConsumed();
  dec->AdvanceCodestream(total_bits / jxl::kBitsPerByte);
  dec->codestream_bits_ahead = total_bits % jxl::kBitsPerByte;
  dec->got_basic_info = true;
  dec->basic_info_size_hint = 0;
  dec->image_metadata = dec->metadata.m;
  JXL_DEBUG_V(2, "Decoded BasicInfo: %s", dec->metadata.DebugString().c_str());

  if (!CheckSizeLimit(dec, dec->metadata.size.xsize(),
                      dec->metadata.size.ysize())) {
    return JXL_INPUT_ERROR("image is too large");
  }

  return JXL_DEC_SUCCESS;
}

// Reads all codestream headers (but not frame headers)
JxlDecoderStatus JxlDecoderReadAllHeaders(JxlDecoder* dec) {
  if (!dec->got_transform_data) {
    Span<const uint8_t> span;
    JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
    auto reader = GetBitReader(span);
    reader->SkipBits(dec->codestream_bits_ahead);
    dec->metadata.transform_data.nonserialized_xyb_encoded =
        dec->metadata.m.xyb_encoded;
    JXL_API_RETURN_IF_ERROR(
        ReadBundle(dec, span, reader.get(), &dec->metadata.transform_data));
    size_t total_bits = reader->TotalBitsConsumed();
    dec->AdvanceCodestream(total_bits / jxl::kBitsPerByte);
    dec->codestream_bits_ahead = total_bits % jxl::kBitsPerByte;
    dec->got_transform_data = true;
  }

  Span<const uint8_t> span;
  JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
  auto reader = GetBitReader(span);
  reader->SkipBits(dec->codestream_bits_ahead);

  if (dec->metadata.m.color_encoding.WantICC()) {
    jxl::Status status = dec->icc_reader->Init(reader.get());
    // Always check AllReadsWithinBounds, not all the C++ decoder implementation
    // handles reader out of bounds correctly  yet (e.g. context map). Not
    // checking AllReadsWithinBounds can cause reader->Close() to trigger an
    // assert, but we don't want library to quit program for invalid codestream.
    if (!reader->AllReadsWithinBounds() ||
        status.code() == StatusCode::kNotEnoughBytes) {
      return dec->RequestMoreInput();
    }
    if (!status) {
      // Other non-successful status is an error
      return JXL_DEC_ERROR;
    }
    PaddedBytes decoded_icc{&dec->memory_manager};
    status = dec->icc_reader->Process(reader.get(), &decoded_icc);
    if (status.code() == StatusCode::kNotEnoughBytes) {
      return dec->RequestMoreInput();
    }
    if (!status) {
      // Other non-successful status is an error
      return JXL_DEC_ERROR;
    }
    if (decoded_icc.empty()) {
      return JXL_DEC_ERROR;
    }
    IccBytes icc;
    Bytes(decoded_icc).AppendTo(icc);
    dec->metadata.m.color_encoding.SetICCRaw(std::move(icc));
  }

  dec->got_all_headers = true;
  JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  dec->AdvanceCodestream(reader->TotalBitsConsumed() / jxl::kBitsPerByte);
  dec->codestream_bits_ahead = 0;

  if (!dec->passes_state) {
    dec->passes_state =
        jxl::make_unique<jxl::PassesDecoderState>(&dec->memory_manager);
  }

  JXL_API_RETURN_IF_ERROR(
      dec->passes_state->output_encoding_info.SetFromMetadata(dec->metadata));
  if (dec->desired_intensity_target > 0) {
    dec->passes_state->output_encoding_info.desired_intensity_target =
        dec->desired_intensity_target;
  }
  dec->image_metadata = dec->metadata.m;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderProcessSections(JxlDecoder* dec) {
  Span<const uint8_t> span;
  JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
  const auto& toc = dec->frame_dec->Toc();
  size_t pos = 0;
  std::vector<jxl::FrameDecoder::SectionInfo> section_info;
  std::vector<jxl::FrameDecoder::SectionStatus> section_status;
  for (size_t i = dec->next_section; i < toc.size(); ++i) {
    if (dec->section_processed[i]) {
      pos += toc[i].size;
      continue;
    }
    size_t id = toc[i].id;
    size_t size = toc[i].size;
    if (OutOfBounds(pos, size, span.size())) {
      break;
    }
    auto* br = new jxl::BitReader(jxl::Bytes(span.data() + pos, size));
    section_info.emplace_back(jxl::FrameDecoder::SectionInfo{br, id, i});
    section_status.emplace_back();
    pos += size;
  }
  jxl::Status status = dec->frame_dec->ProcessSections(
      section_info.data(), section_info.size(), section_status.data());
  bool out_of_bounds = false;
  bool has_error = false;
  for (const auto& info : section_info) {
    if (!info.br->AllReadsWithinBounds()) {
      // Mark out of bounds section, but keep closing and deleting the next
      // ones as well.
      out_of_bounds = true;
    }
    if (!info.br->Close()) has_error = true;
    delete info.br;
  }
  if (has_error) {
    return JXL_INPUT_ERROR("internal: bit-reader failed to close");
  }
  if (out_of_bounds) {
    // If any bit reader indicates out of bounds, it's an error, not just
    // needing more input, since we ensure only bit readers containing
    // a complete section are provided to the FrameDecoder.
    return JXL_INPUT_ERROR("frame out of bounds");
  }
  if (!status) {
    return JXL_INPUT_ERROR("frame processing failed");
  }
  for (size_t i = 0; i < section_status.size(); ++i) {
    auto status = section_status[i];
    if (status == jxl::FrameDecoder::kDone) {
      dec->section_processed[section_info[i].index] = 1;
    } else if (status != jxl::FrameDecoder::kSkipped) {
      return JXL_INPUT_ERROR("unexpected section status");
    }
  }
  size_t completed_prefix_bytes = 0;
  while (dec->next_section < dec->section_processed.size() &&
         dec->section_processed[dec->next_section] == 1) {
    completed_prefix_bytes += toc[dec->next_section].size;
    ++dec->next_section;
  }
  dec->remaining_frame_size -= completed_prefix_bytes;
  dec->AdvanceCodestream(completed_prefix_bytes);
  return JXL_DEC_SUCCESS;
}

// TODO(eustas): no CodecInOut -> no image size reinforcement -> possible OOM.
JxlDecoderStatus JxlDecoderProcessCodestream(JxlDecoder* dec) {
  // If no parallel runner is set, use the default
  // TODO(lode): move this initialization to an appropriate location once the
  // runner is used to decode pixels.
  if (!dec->thread_pool) {
    dec->thread_pool = jxl::make_unique<jxl::ThreadPool>(nullptr, nullptr);
  }

  // No matter what events are wanted, the basic info is always required.
  if (!dec->got_basic_info) {
    JxlDecoderStatus status = JxlDecoderReadBasicInfo(dec);
    if (status != JXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JXL_DEC_BASIC_INFO) {
    dec->events_wanted &= ~JXL_DEC_BASIC_INFO;
    return JXL_DEC_BASIC_INFO;
  }

  if (!dec->events_wanted) {
    dec->stage = DecoderStage::kCodestreamFinished;
    return JXL_DEC_SUCCESS;
  }

  if (!dec->icc_reader) {
    dec->icc_reader = jxl::make_unique<ICCReader>(&dec->memory_manager);
  }

  if (!dec->got_all_headers) {
    JxlDecoderStatus status = JxlDecoderReadAllHeaders(dec);
    if (status != JXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JXL_DEC_COLOR_ENCODING) {
    dec->events_wanted &= ~JXL_DEC_COLOR_ENCODING;
    return JXL_DEC_COLOR_ENCODING;
  }

  if (!dec->events_wanted) {
    dec->stage = DecoderStage::kCodestreamFinished;
    return JXL_DEC_SUCCESS;
  }

  dec->post_headers = true;

  if (!dec->got_preview_image && dec->metadata.m.have_preview) {
    dec->preview_frame = true;
  }

  // Handle frames
  for (;;) {
    bool parse_frames =
        (dec->events_wanted &
         (JXL_DEC_PREVIEW_IMAGE | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
    if (!parse_frames) {
      break;
    }
    if (dec->frame_stage == FrameStage::kHeader && dec->is_last_total) {
      break;
    }
    if (dec->frame_stage == FrameStage::kHeader) {
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      if (dec->recon_output_jpeg == JpegReconStage::kSettingMetadata ||
          dec->recon_output_jpeg == JpegReconStage::kOutputting) {
        // The image bundle contains the JPEG reconstruction frame, but the
        // decoder is still waiting to decode an EXIF or XMP box. It's not
        // implemented to decode additional frames during this, and a JPEG
        // reconstruction image should have only one frame.
        return JXL_API_ERROR(
            "cannot decode a next frame after JPEG reconstruction frame");
      }
#endif
      if (!dec->ib) {
        dec->ib = jxl::make_unique<jxl::ImageBundle>(&dec->memory_manager,
                                                     &dec->image_metadata);
      }
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      // If JPEG reconstruction is wanted and possible, set the jpeg_data of
      // the ImageBundle.
      if (!dec->jpeg_decoder.SetImageBundleJpegData(dec->ib.get()))
        return JXL_DEC_ERROR;
#endif
      dec->frame_dec = jxl::make_unique<FrameDecoder>(
          dec->passes_state.get(), dec->metadata, dec->thread_pool.get(),
          /*use_slow_rendering_pipeline=*/false);
      dec->frame_header = jxl::make_unique<FrameHeader>(&dec->metadata);
      Span<const uint8_t> span;
      JXL_API_RETURN_IF_ERROR(dec->GetCodestreamInput(&span));
      auto reader = GetBitReader(span);
      jxl::Status status = dec->frame_dec->InitFrame(
          reader.get(), dec->ib.get(), dec->preview_frame);
      if (!reader->AllReadsWithinBounds() ||
          status.code() == StatusCode::kNotEnoughBytes) {
        return dec->RequestMoreInput();
      } else if (!status) {
        return JXL_INPUT_ERROR("invalid frame header");
      }
      dec->AdvanceCodestream(reader->TotalBitsConsumed() / kBitsPerByte);
      *dec->frame_header = dec->frame_dec->GetFrameHeader();
      jxl::FrameDimensions frame_dim = dec->frame_header->ToFrameDimensions();
      if (!CheckSizeLimit(dec, frame_dim.xsize_upsampled_padded,
                          frame_dim.ysize_upsampled_padded)) {
        return JXL_INPUT_ERROR("frame is too large");
      }
      int output_type =
          dec->preview_frame ? JXL_DEC_PREVIEW_IMAGE : JXL_DEC_FULL_IMAGE;
      bool output_needed = ((dec->events_wanted & output_type) != 0);
      if (output_needed) {
        JXL_API_RETURN_IF_ERROR(dec->frame_dec->InitFrameOutput());
      }
      dec->remaining_frame_size = dec->frame_dec->SumSectionSizes();

      dec->frame_stage = FrameStage::kTOC;
      if (dec->preview_frame) {
        if (!(dec->events_wanted & JXL_DEC_PREVIEW_IMAGE)) {
          dec->frame_stage = FrameStage::kHeader;
          dec->AdvanceCodestream(dec->remaining_frame_size);
          dec->got_preview_image = true;
          dec->preview_frame = false;
        }
        continue;
      }

      int saved_as = FrameDecoder::SavedAs(*dec->frame_header);
      // is last in entire codestream
      dec->is_last_total = dec->frame_header->is_last;
      // is last of current still
      dec->is_last_of_still =
          dec->is_last_total || dec->frame_header->animation_frame.duration > 0;
      // is kRegularFrame and coalescing is disabled
      dec->is_last_of_still |=
          (!dec->coalescing &&
           dec->frame_header->frame_type == FrameType::kRegularFrame);
      const size_t internal_frame_index = dec->internal_frames;
      const size_t external_frame_index = dec->external_frames;
      if (dec->is_last_of_still) dec->external_frames++;
      dec->internal_frames++;

      if (dec->skip_frames > 0) {
        dec->skipping_frame = true;
        if (dec->is_last_of_still) {
          dec->skip_frames--;
        }
      } else {
        dec->skipping_frame = false;
      }

      if (external_frame_index >= dec->frame_external_to_internal.size()) {
        dec->frame_external_to_internal.push_back(internal_frame_index);
        if (dec->frame_external_to_internal.size() !=
            external_frame_index + 1) {
          return JXL_API_ERROR("internal");
        }
      }

      if (internal_frame_index >= dec->frame_refs.size()) {
        // add the value 0xff (which means all references) to new slots: we only
        // know the references of the frame at FinalizeFrame, and fill in the
        // correct values there. As long as this information is not known, the
        // worst case where the frame depends on all storage slots is assumed.
        dec->frame_refs.emplace_back(FrameRef{0xFF, saved_as});
        if (dec->frame_refs.size() != internal_frame_index + 1) {
          return JXL_API_ERROR("internal");
        }
      }

      if (dec->skipping_frame) {
        // Whether this frame could be referenced by any future frame: either
        // because it's a frame saved for blending or patches, or because it's
        // a DC frame.
        bool referenceable =
            dec->frame_header->CanBeReferenced() ||
            dec->frame_header->frame_type == FrameType::kDCFrame;
        if (internal_frame_index < dec->frame_required.size() &&
            !dec->frame_required[internal_frame_index]) {
          referenceable = false;
        }
        if (!referenceable) {
          // Skip all decoding for this frame, since the user is skipping this
          // frame and no future frames can reference it.
          dec->frame_stage = FrameStage::kHeader;
          dec->AdvanceCodestream(dec->remaining_frame_size);
          continue;
        }
      }

      if ((dec->events_wanted & JXL_DEC_FRAME) && dec->is_last_of_still) {
        // Only return this for the last of a series of stills: patches frames
        // etc... before this one do not contain the correct information such
        // as animation timing, ...
        if (!dec->skipping_frame) {
          return JXL_DEC_FRAME;
        }
      }
    }

    if (dec->frame_stage == FrameStage::kTOC) {
      dec->frame_dec->SetRenderSpotcolors(dec->render_spotcolors);
      dec->frame_dec->SetCoalescing(dec->coalescing);

      if (!dec->preview_frame &&
          (dec->events_wanted & JXL_DEC_FRAME_PROGRESSION)) {
        dec->frame_prog_detail =
            dec->frame_dec->SetPauseAtProgressive(dec->prog_detail);
      } else {
        dec->frame_prog_detail = JxlProgressiveDetail::kFrames;
      }
      dec->dc_frame_progression_done = false;

      dec->next_section = 0;
      dec->section_processed.clear();
      dec->section_processed.resize(dec->frame_dec->Toc().size(), 0);

      // If we don't need pixels, we can skip actually decoding the frames.
      if (dec->preview_frame || (dec->events_wanted & JXL_DEC_FULL_IMAGE)) {
        dec->frame_stage = FrameStage::kFull;
      } else if (!dec->is_last_total) {
        dec->frame_stage = FrameStage::kHeader;
        dec->AdvanceCodestream(dec->remaining_frame_size);
        continue;
      } else {
        break;
      }
    }

    if (dec->frame_stage == FrameStage::kFull) {
      if (!dec->image_out_buffer_set) {
        if (dec->preview_frame) {
          return JXL_DEC_NEED_PREVIEW_OUT_BUFFER;
        }
        if (
#if JPEGXL_ENABLE_TRANSCODE_JPEG
            (!dec->jpeg_decoder.IsOutputSet() ||
             dec->ib->jpeg_data == nullptr) &&
#endif
            dec->is_last_of_still && !dec->skipping_frame) {
          // TODO(lode): remove the dec->is_last_of_still condition if the
          // frame decoder needs the image buffer as working space for decoding
          // non-visible or blending frames too
          return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
        }
      }

      if (dec->image_out_buffer_set) {
        size_t xsize;
        size_t ysize;
        GetCurrentDimensions(dec, xsize, ysize);
        size_t bits_per_sample = GetBitDepth(
            dec->image_out_bit_depth, dec->metadata.m, dec->image_out_format);
        dec->frame_dec->SetImageOutput(
            PixelCallback{
                dec->image_out_init_callback, dec->image_out_run_callback,
                dec->image_out_destroy_callback, dec->image_out_init_opaque},
            reinterpret_cast<uint8_t*>(dec->image_out_buffer),
            dec->image_out_size, xsize, ysize, dec->image_out_format,
            bits_per_sample, dec->unpremul_alpha, !dec->keep_orientation);
        for (size_t i = 0; i < dec->extra_channel_output.size(); ++i) {
          const auto& extra = dec->extra_channel_output[i];
          size_t ec_bits_per_sample =
              GetBitDepth(dec->image_out_bit_depth,
                          dec->metadata.m.extra_channel_info[i], extra.format);
          dec->frame_dec->AddExtraChannelOutput(extra.buffer, extra.buffer_size,
                                                xsize, extra.format,
                                                ec_bits_per_sample);
        }
      }

      size_t next_num_passes_to_pause = dec->frame_dec->NextNumPassesToPause();

      JXL_API_RETURN_IF_ERROR(JxlDecoderProcessSections(dec));

      bool all_sections_done = dec->frame_dec->HasDecodedAll();
      bool got_dc_only = !all_sections_done && dec->frame_dec->HasDecodedDC();

      if (dec->frame_prog_detail >= JxlProgressiveDetail::kDC &&
          !dec->dc_frame_progression_done && got_dc_only) {
        dec->dc_frame_progression_done = true;
        dec->downsampling_target = 8;
        return JXL_DEC_FRAME_PROGRESSION;
      }

      bool new_progression_step_done =
          dec->frame_dec->NumCompletePasses() >= next_num_passes_to_pause;

      if (!all_sections_done &&
          dec->frame_prog_detail >= JxlProgressiveDetail::kLastPasses &&
          new_progression_step_done) {
        dec->downsampling_target =
            dec->frame_header->passes.GetDownsamplingTargetForCompletedPasses(
                dec->frame_dec->NumCompletePasses());
        return JXL_DEC_FRAME_PROGRESSION;
      }

      if (!all_sections_done) {
        // Not all sections have been processed yet
        return dec->RequestMoreInput();
      }

      if (!dec->preview_frame) {
        size_t internal_index = dec->internal_frames - 1;
        if (dec->frame_refs.size() <= internal_index) {
          return JXL_API_ERROR("internal");
        }
        // Always fill this in, even if it was already written, it could be that
        // this frame was skipped before and set to 255, while only now we know
        // the true value.
        dec->frame_refs[internal_index].reference =
            dec->frame_dec->References();
      }

      if (!dec->frame_dec->FinalizeFrame()) {
        return JXL_INPUT_ERROR("decoding frame failed");
      }
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      // If jpeg output was requested, we merely return the JXL_DEC_FULL_IMAGE
      // status without outputting pixels.
      if (dec->jpeg_decoder.IsOutputSet() && dec->ib->jpeg_data != nullptr) {
        dec->frame_stage = FrameStage::kHeader;
        dec->recon_output_jpeg = JpegReconStage::kSettingMetadata;
        return JXL_DEC_FULL_IMAGE;
      }
#endif
      if (dec->preview_frame || dec->is_last_of_still) {
        dec->image_out_buffer_set = false;
        dec->extra_channel_output.clear();
      }
    }

    dec->frame_stage = FrameStage::kHeader;

    // The pixels have been output or are not needed, do not keep them in
    // memory here.
    dec->ib.reset();
    if (dec->preview_frame) {
      dec->got_preview_image = true;
      dec->preview_frame = false;
      dec->events_wanted &= ~JXL_DEC_PREVIEW_IMAGE;
      return JXL_DEC_PREVIEW_IMAGE;
    } else if (dec->is_last_of_still &&
               (dec->events_wanted & JXL_DEC_FULL_IMAGE) &&
               !dec->skipping_frame) {
      return JXL_DEC_FULL_IMAGE;
    }
  }

  dec->stage = DecoderStage::kCodestreamFinished;
  // Return success, this means there is nothing more to do.
  return JXL_DEC_SUCCESS;
}

}  // namespace
}  // namespace jxl

JxlDecoderStatus JxlDecoderSetInput(JxlDecoder* dec, const uint8_t* data,
                                    size_t size) {
  if (dec->next_in) {
    return JXL_API_ERROR("already set input, use JxlDecoderReleaseInput first");
  }
  if (dec->input_closed) {
    return JXL_API_ERROR("input already closed");
  }

  dec->next_in = data;
  dec->avail_in = size;
  return JXL_DEC_SUCCESS;
}

size_t JxlDecoderReleaseInput(JxlDecoder* dec) {
  size_t result = dec->avail_in;
  dec->next_in = nullptr;
  dec->avail_in = 0;
  return result;
}

void JxlDecoderCloseInput(JxlDecoder* dec) { dec->input_closed = true; }

JxlDecoderStatus JxlDecoderSetJPEGBuffer(JxlDecoder* dec, uint8_t* data,
                                         size_t size) {
#if JPEGXL_ENABLE_TRANSCODE_JPEG
  // JPEG reconstruction buffer can only set and updated before or during the
  // first frame, the reconstruction box refers to the first frame and in
  // theory multi-frame images should not be used with a jbrd box.
  if (dec->internal_frames > 1) {
    return JXL_API_ERROR("JPEG reconstruction only works for the first frame");
  }
  if (dec->jpeg_decoder.IsOutputSet()) {
    return JXL_API_ERROR("Already set JPEG buffer");
  }
  return dec->jpeg_decoder.SetOutputBuffer(data, size);
#else
  return JXL_API_ERROR("JPEG reconstruction is not supported.");
#endif
}

size_t JxlDecoderReleaseJPEGBuffer(JxlDecoder* dec) {
#if JPEGXL_ENABLE_TRANSCODE_JPEG
  return dec->jpeg_decoder.ReleaseOutputBuffer();
#else
  return JXL_API_ERROR("JPEG reconstruction is not supported.");
#endif
}

// Parses the header of the box, outputting the 4-character type and the box
// size, including header size, as stored in the box header.
// @param in current input bytes.
// @param size available input size.
// @param pos position in the input, must begin at the header of the box.
// @param file_pos position of pos since the start of the JXL file, rather than
// the current input, used for integer overflow checking.
// @param type the output box type.
// @param box_size output the total box size, including header, in bytes, or 0
// if it's a final unbounded box.
// @param header_size output size of the box header.
// @return JXL_DEC_SUCCESS if the box header was fully parsed. In that case the
// parsing position must be incremented by header_size bytes.
// JXL_DEC_NEED_MORE_INPUT if not enough input bytes available, in that case
// header_size indicates a lower bound for the known size the header has to be
// at least. JXL_DEC_ERROR if the box header is invalid.
static JxlDecoderStatus ParseBoxHeader(const uint8_t* in, size_t size,
                                       size_t pos, size_t file_pos,
                                       JxlBoxType type, uint64_t* box_size,
                                       uint64_t* header_size) {
  if (OutOfBounds(pos, 8, size)) {
    *header_size = 8;
    return JXL_DEC_NEED_MORE_INPUT;
  }
  size_t box_start = pos;
  // Box size, including this header itself.
  *box_size = LoadBE32(in + pos);
  pos += 4;
  memcpy(type, in + pos, 4);
  pos += 4;
  if (*box_size == 1) {
    *header_size = 16;
    if (OutOfBounds(pos, 8, size)) return JXL_DEC_NEED_MORE_INPUT;
    *box_size = LoadBE64(in + pos);
    pos += 8;
  }
  *header_size = pos - box_start;
  if (*box_size > 0 && *box_size < *header_size) {
    return JXL_INPUT_ERROR("invalid box size");
  }
  if (file_pos + *box_size < file_pos) {
    return JXL_INPUT_ERROR("Box size overflow");
  }
  return JXL_DEC_SUCCESS;
}

// This includes handling the codestream if it is not a box-based jxl file.
static JxlDecoderStatus HandleBoxes(JxlDecoder* dec) {
  // Box handling loop
  for (;;) {
    if (dec->box_stage != BoxStage::kHeader) {
      dec->AdvanceInput(dec->header_size);
      dec->header_size = 0;
#if JPEGXL_ENABLE_BOXES
      if ((dec->events_wanted & JXL_DEC_BOX) &&
          dec->box_out_buffer_set_current_box) {
        uint8_t* next_out = dec->box_out_buffer + dec->box_out_buffer_pos;
        size_t avail_out = dec->box_out_buffer_size - dec->box_out_buffer_pos;

        JxlDecoderStatus box_result = dec->box_content_decoder.Process(
            dec->next_in, dec->avail_in,
            dec->file_pos - dec->box_contents_begin, &next_out, &avail_out);
        size_t produced =
            next_out - (dec->box_out_buffer + dec->box_out_buffer_pos);
        dec->box_out_buffer_pos += produced;

        if (box_result == JXL_DEC_BOX_COMPLETE &&
            !(dec->events_wanted & JXL_DEC_BOX_COMPLETE)) {
          box_result = JXL_DEC_SUCCESS;
        }

        // Don't return JXL_DEC_NEED_MORE_INPUT: the box stages below, instead,
        // handle the input progression, and the above only outputs the part of
        // the box seen so far.
        if (box_result != JXL_DEC_SUCCESS &&
            box_result != JXL_DEC_NEED_MORE_INPUT) {
          return box_result;
        }
      }
#endif
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      if (dec->store_exif == 1 || dec->store_xmp == 1) {
        std::vector<uint8_t>& metadata =
            (dec->store_exif == 1) ? dec->exif_metadata : dec->xmp_metadata;
        for (;;) {
          if (metadata.empty()) metadata.resize(64);
          uint8_t* orig_next_out = metadata.data() + dec->recon_out_buffer_pos;
          uint8_t* next_out = orig_next_out;
          size_t avail_out = metadata.size() - dec->recon_out_buffer_pos;
          JxlDecoderStatus box_result = dec->metadata_decoder.Process(
              dec->next_in, dec->avail_in,
              dec->file_pos - dec->box_contents_begin, &next_out, &avail_out);
          size_t produced = next_out - orig_next_out;
          dec->recon_out_buffer_pos += produced;
          if (box_result == JXL_DEC_BOX_NEED_MORE_OUTPUT) {
            metadata.resize(metadata.size() * 2);
          } else if (box_result == JXL_DEC_NEED_MORE_INPUT) {
            break;  // box stage handling below will handle this instead
          } else if (box_result == JXL_DEC_BOX_COMPLETE) {
            size_t needed_size = (dec->store_exif == 1) ? dec->recon_exif_size
                                                        : dec->recon_xmp_size;
            if (dec->box_contents_unbounded &&
                dec->recon_out_buffer_pos < needed_size) {
              // Unbounded box, but we know the expected size due to the jbrd
              // box's data. Treat this as the JXL_DEC_NEED_MORE_INPUT case.
              break;
            } else {
              metadata.resize(dec->recon_out_buffer_pos);
              if (dec->store_exif == 1) dec->store_exif = 2;
              if (dec->store_xmp == 1) dec->store_xmp = 2;
              break;
            }
          } else {
            // error
            return box_result;
          }
        }
      }
#endif
    }
#if JPEGXL_ENABLE_TRANSCODE_JPEG
    if (dec->recon_output_jpeg == JpegReconStage::kSettingMetadata &&
        !dec->JbrdNeedMoreBoxes()) {
      jxl::jpeg::JPEGData* jpeg_data = dec->ib->jpeg_data.get();
      if (dec->recon_exif_size) {
        JxlDecoderStatus status = jxl::JxlToJpegDecoder::SetExif(
            dec->exif_metadata.data(), dec->exif_metadata.size(), jpeg_data);
        if (status != JXL_DEC_SUCCESS) return status;
      }
      if (dec->recon_xmp_size) {
        JxlDecoderStatus status = jxl::JxlToJpegDecoder::SetXmp(
            dec->xmp_metadata.data(), dec->xmp_metadata.size(), jpeg_data);
        if (status != JXL_DEC_SUCCESS) return status;
      }
      dec->recon_output_jpeg = JpegReconStage::kOutputting;
    }

    if (dec->recon_output_jpeg == JpegReconStage::kOutputting &&
        !dec->JbrdNeedMoreBoxes()) {
      JxlDecoderStatus status =
          dec->jpeg_decoder.WriteOutput(*dec->ib->jpeg_data);
      if (status != JXL_DEC_SUCCESS) return status;
      dec->recon_output_jpeg = JpegReconStage::kNone;
      dec->ib.reset();
      if (dec->events_wanted & JXL_DEC_FULL_IMAGE) {
        // Return the full image event here now, this may be delayed if this
        // could only be done after decoding an exif or xmp box after the
        // codestream.
        return JXL_DEC_FULL_IMAGE;
      }
    }
#endif

    if (dec->box_stage == BoxStage::kHeader) {
      if (!dec->have_container) {
        if (dec->stage == DecoderStage::kCodestreamFinished)
          return JXL_DEC_SUCCESS;
        dec->box_stage = BoxStage::kCodestream;
        dec->box_contents_unbounded = true;
        continue;
      }
      if (dec->avail_in == 0) {
        if (dec->stage != DecoderStage::kCodestreamFinished) {
          // Not yet seen (all) codestream boxes.
          return JXL_DEC_NEED_MORE_INPUT;
        }
#if JPEGXL_ENABLE_TRANSCODE_JPEG
        if (dec->JbrdNeedMoreBoxes()) {
          return JXL_DEC_NEED_MORE_INPUT;
        }
#endif
        if (dec->input_closed) {
          return JXL_DEC_SUCCESS;
        }
        if (!(dec->events_wanted & JXL_DEC_BOX)) {
          // All codestream and jbrd metadata boxes finished, and no individual
          // boxes requested by user, so no need to request any more input.
          // This returns success for backwards compatibility, when
          // JxlDecoderCloseInput and JXL_DEC_BOX did not exist, as well
          // as for efficiency.
          return JXL_DEC_SUCCESS;
        }
        // Even though we are exactly at a box end, there still may be more
        // boxes. The user may call JxlDecoderCloseInput to indicate the input
        // is finished and get success instead.
        return JXL_DEC_NEED_MORE_INPUT;
      }

      bool boxed_codestream_done =
          ((dec->events_wanted & JXL_DEC_BOX) &&
           dec->stage == DecoderStage::kCodestreamFinished &&
#if JPEGXL_ENABLE_TRANSCODE_JPEG
           !dec->JbrdNeedMoreBoxes() &&
#endif
           dec->last_codestream_seen);
      if (boxed_codestream_done && dec->avail_in >= 2 &&
          dec->next_in[0] == 0xff &&
          dec->next_in[1] == jxl::kCodestreamMarker) {
        // We detected the start of the next naked codestream, so we can return
        // success here.
        return JXL_DEC_SUCCESS;
      }

      uint64_t box_size;
      uint64_t header_size;
      JxlDecoderStatus status =
          ParseBoxHeader(dec->next_in, dec->avail_in, 0, dec->file_pos,
                         dec->box_type, &box_size, &header_size);
      if (status != JXL_DEC_SUCCESS) {
        if (status == JXL_DEC_NEED_MORE_INPUT) {
          dec->basic_info_size_hint =
              InitialBasicInfoSizeHint() + header_size - dec->file_pos;
        }
        return status;
      }
      if (memcmp(dec->box_type, "brob", 4) == 0) {
        if (dec->avail_in < header_size + 4) {
          return JXL_DEC_NEED_MORE_INPUT;
        }
        memcpy(dec->box_decoded_type, dec->next_in + header_size,
               sizeof(dec->box_decoded_type));
      } else {
        memcpy(dec->box_decoded_type, dec->box_type,
               sizeof(dec->box_decoded_type));
      }

      // Box order validity checks
      // The signature box at box_count == 1 is not checked here since that's
      // already done at the beginning.
      dec->box_count++;
      if (boxed_codestream_done && memcmp(dec->box_type, "JXL ", 4) == 0) {
        // We detected the start of the next boxed stream, so we can return
        // success here.
        return JXL_DEC_SUCCESS;
      }
      if (dec->box_count == 2 && memcmp(dec->box_type, "ftyp", 4) != 0) {
        return JXL_INPUT_ERROR("the second box must be the ftyp box");
      }
      if (memcmp(dec->box_type, "ftyp", 4) == 0 && dec->box_count != 2) {
        return JXL_INPUT_ERROR("the ftyp box must come second");
      }

      dec->box_contents_unbounded = (box_size == 0);
      dec->box_contents_begin = dec->file_pos + header_size;
      dec->box_contents_end =
          dec->box_contents_unbounded ? 0 : (dec->file_pos + box_size);
      dec->box_contents_size =
          dec->box_contents_unbounded ? 0 : (box_size - header_size);
      dec->box_size = box_size;
      dec->header_size = header_size;
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      if (dec->orig_events_wanted & JXL_DEC_JPEG_RECONSTRUCTION) {
        // Initiate storing of Exif or XMP data for JPEG reconstruction
        if (dec->store_exif == 0 &&
            memcmp(dec->box_decoded_type, "Exif", 4) == 0) {
          dec->store_exif = 1;
          dec->recon_out_buffer_pos = 0;
        }
        if (dec->store_xmp == 0 &&
            memcmp(dec->box_decoded_type, "xml ", 4) == 0) {
          dec->store_xmp = 1;
          dec->recon_out_buffer_pos = 0;
        }
      }
#endif
#if JPEGXL_ENABLE_BOXES
      if (dec->events_wanted & JXL_DEC_BOX) {
        bool decompress =
            dec->decompress_boxes && memcmp(dec->box_type, "brob", 4) == 0;
        dec->box_content_decoder.StartBox(
            decompress, dec->box_contents_unbounded, dec->box_contents_size);
      }
#endif
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      if (dec->store_exif == 1 || dec->store_xmp == 1) {
        bool brob = memcmp(dec->box_type, "brob", 4) == 0;
        dec->metadata_decoder.StartBox(brob, dec->box_contents_unbounded,
                                       dec->box_contents_size);
      }
#endif
      if (memcmp(dec->box_type, "ftyp", 4) == 0) {
        dec->box_stage = BoxStage::kFtyp;
      } else if (memcmp(dec->box_type, "jxlc", 4) == 0) {
        if (dec->last_codestream_seen) {
          return JXL_INPUT_ERROR("there can only be one jxlc box");
        }
        dec->last_codestream_seen = true;
        dec->box_stage = BoxStage::kCodestream;
      } else if (memcmp(dec->box_type, "jxlp", 4) == 0) {
        dec->box_stage = BoxStage::kPartialCodestream;
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      } else if ((dec->orig_events_wanted & JXL_DEC_JPEG_RECONSTRUCTION) &&
                 memcmp(dec->box_type, "jbrd", 4) == 0) {
        if (!(dec->events_wanted & JXL_DEC_JPEG_RECONSTRUCTION)) {
          return JXL_INPUT_ERROR(
              "multiple JPEG reconstruction boxes not supported");
        }
        dec->box_stage = BoxStage::kJpegRecon;
#endif
      } else {
        dec->box_stage = BoxStage::kSkip;
      }

      if (dec->events_wanted & JXL_DEC_BOX) {
        dec->box_event = true;
        dec->box_out_buffer_set_current_box = false;
        return JXL_DEC_BOX;
      }
    } else if (dec->box_stage == BoxStage::kFtyp) {
      if (dec->box_contents_size < 12) {
        return JXL_INPUT_ERROR("file type box too small");
      }
      if (dec->avail_in < 4) return JXL_DEC_NEED_MORE_INPUT;
      if (memcmp(dec->next_in, "jxl ", 4) != 0) {
        return JXL_INPUT_ERROR("file type box major brand must be \"jxl \"");
      }
      dec->AdvanceInput(4);
      dec->box_stage = BoxStage::kSkip;
    } else if (dec->box_stage == BoxStage::kPartialCodestream) {
      if (dec->last_codestream_seen) {
        return JXL_INPUT_ERROR("cannot have jxlp box after last jxlp box");
      }
      // TODO(lode): error if box is unbounded but last bit not set
      if (dec->avail_in < 4) return JXL_DEC_NEED_MORE_INPUT;
      if (!dec->box_contents_unbounded && dec->box_contents_size < 4) {
        return JXL_INPUT_ERROR("jxlp box too small to contain index");
      }
      size_t jxlp_index = LoadBE32(dec->next_in);
      // The high bit of jxlp_index indicates whether this is the last
      // jxlp box.
      if (jxlp_index & 0x80000000) {
        dec->last_codestream_seen = true;
      }
      dec->AdvanceInput(4);
      dec->box_stage = BoxStage::kCodestream;
    } else if (dec->box_stage == BoxStage::kCodestream) {
      JxlDecoderStatus status = jxl::JxlDecoderProcessCodestream(dec);
#if JPEGXL_ENABLE_TRANSCODE_JPEG
      if (status == JXL_DEC_FULL_IMAGE) {
        if (dec->recon_output_jpeg != JpegReconStage::kNone) {
          continue;
        }
      }
#endif
      if (status == JXL_DEC_NEED_MORE_INPUT) {
        if (dec->file_pos == dec->box_contents_end &&
            !dec->box_contents_unbounded) {
          dec->box_stage = BoxStage::kHeader;
          continue;
        }
      }

      if (status == JXL_DEC_SUCCESS) {
#if JPEGXL_ENABLE_TRANSCODE_JPEG
        if (dec->JbrdNeedMoreBoxes()) {
          dec->box_stage = BoxStage::kSkip;
          continue;
        }
#endif
        if (dec->box_contents_unbounded) {
          // Last box reached and codestream done, nothing more to do.
          break;
        }
        if (dec->events_wanted & JXL_DEC_BOX) {
          // Codestream done, but there may be more other boxes.
          dec->box_stage = BoxStage::kSkip;
          continue;
        }
      }
      return status;
#if JPEGXL_ENABLE_TRANSCODE_JPEG
    } else if (dec->box_stage == BoxStage::kJpegRecon) {
      if (!dec->jpeg_decoder.IsParsingBox()) {
        // This is a new JPEG reconstruction metadata box.
        dec->jpeg_decoder.StartBox(dec->box_contents_unbounded,
                                   dec->box_contents_size);
      }
      const uint8_t* next_in = dec->next_in;
      size_t avail_in = dec->avail_in;
      JxlDecoderStatus recon_result =
          dec->jpeg_decoder.Process(&next_in, &avail_in);
      size_t consumed = next_in - dec->next_in;
      dec->AdvanceInput(consumed);
      if (recon_result == JXL_DEC_JPEG_RECONSTRUCTION) {
        jxl::jpeg::JPEGData* jpeg_data = dec->jpeg_decoder.GetJpegData();
        size_t num_exif = jxl::JxlToJpegDecoder::NumExifMarkers(*jpeg_data);
        size_t num_xmp = jxl::JxlToJpegDecoder::NumXmpMarkers(*jpeg_data);
        if (num_exif) {
          if (num_exif > 1) {
            return JXL_INPUT_ERROR(
                "multiple exif markers for JPEG reconstruction not supported");
          }
          if (JXL_DEC_SUCCESS != jxl::JxlToJpegDecoder::ExifBoxContentSize(
                                     *jpeg_data, &dec->recon_exif_size)) {
            return JXL_INPUT_ERROR("invalid jbrd exif size");
          }
        }
        if (num_xmp) {
          if (num_xmp > 1) {
            return JXL_INPUT_ERROR(
                "multiple XMP markers for JPEG reconstruction not supported");
          }
          if (JXL_DEC_SUCCESS != jxl::JxlToJpegDecoder::XmlBoxContentSize(
                                     *jpeg_data, &dec->recon_xmp_size)) {
            return JXL_INPUT_ERROR("invalid jbrd XMP size");
          }
        }

        dec->box_stage = BoxStage::kHeader;
        // If successful JPEG reconstruction, return the success if the user
        // cares about it, otherwise continue.
        if (dec->events_wanted & JXL_DEC_JPEG_RECONSTRUCTION) {
          dec->events_wanted &= ~JXL_DEC_JPEG_RECONSTRUCTION;
          return JXL_DEC_JPEG_RECONSTRUCTION;
        }
      } else {
        // If anything else, return the result.
        return recon_result;
      }
#endif
    } else if (dec->box_stage == BoxStage::kSkip) {
      if (dec->box_contents_unbounded) {
        if (dec->input_closed) {
          return JXL_DEC_SUCCESS;
        }
        if (!(dec->box_out_buffer_set)) {
          // An unbounded box is always the last box. Not requesting box data,
          // so return success even if JxlDecoderCloseInput was not called for
          // backwards compatibility as well as efficiency since this box is
          // being skipped.
          return JXL_DEC_SUCCESS;
        }
        // Arbitrarily more bytes may follow, only JxlDecoderCloseInput can
        // mark the end.
        dec->AdvanceInput(dec->avail_in);
        return JXL_DEC_NEED_MORE_INPUT;
      }
      // Amount of remaining bytes in the box that is being skipped.
      size_t remaining = dec->box_contents_end - dec->file_pos;
      if (dec->avail_in < remaining) {
        // Indicate how many more bytes needed starting from next_in.
        dec->basic_info_size_hint =
            InitialBasicInfoSizeHint() + dec->box_contents_end - dec->file_pos;
        // Don't have the full box yet, skip all we have so far
        dec->AdvanceInput(dec->avail_in);
        return JXL_DEC_NEED_MORE_INPUT;
      } else {
        // Full box available, skip all its remaining bytes
        dec->AdvanceInput(remaining);
        dec->box_stage = BoxStage::kHeader;
      }
    } else {
      JXL_DEBUG_ABORT("Unreachable");
    }
  }
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderProcessInput(JxlDecoder* dec) {
  if (dec->stage == DecoderStage::kInited) {
    dec->stage = DecoderStage::kStarted;
  }
  if (dec->stage == DecoderStage::kError) {
    return JXL_API_ERROR(
        "Cannot keep using decoder after it encountered an error, use "
        "JxlDecoderReset to reset it");
  }

  if (!dec->got_signature) {
    JxlSignature sig = JxlSignatureCheck(dec->next_in, dec->avail_in);
    if (sig == JXL_SIG_INVALID) return JXL_INPUT_ERROR("invalid signature");
    if (sig == JXL_SIG_NOT_ENOUGH_BYTES) {
      if (dec->input_closed) {
        return JXL_INPUT_ERROR("file too small for signature");
      }
      return JXL_DEC_NEED_MORE_INPUT;
    }

    dec->got_signature = true;

    if (sig == JXL_SIG_CONTAINER) {
      dec->have_container = true;
    } else {
      dec->last_codestream_seen = true;
    }
  }

  JxlDecoderStatus status = HandleBoxes(dec);

  if (status == JXL_DEC_NEED_MORE_INPUT && dec->input_closed) {
    return JXL_INPUT_ERROR("premature end of input");
  }

  // Even if the box handling returns success, certain types of
  // data may be missing.
  if (status == JXL_DEC_SUCCESS) {
    if (dec->CanUseMoreCodestreamInput()) {
      return JXL_INPUT_ERROR("codestream never finished");
    }
#if JPEGXL_ENABLE_TRANSCODE_JPEG
    if (dec->JbrdNeedMoreBoxes()) {
      return JXL_INPUT_ERROR("missing metadata boxes for jpeg reconstruction");
    }
#endif
  }

  return status;
}

// To ensure ABI forward-compatibility, this struct has a constant size.
static_assert(sizeof(JxlBasicInfo) == 204,
              "JxlBasicInfo struct size should remain constant");

JxlDecoderStatus JxlDecoderGetBasicInfo(const JxlDecoder* dec,
                                        JxlBasicInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  if (info) {
    memset(info, 0, sizeof(*info));

    const jxl::ImageMetadata& meta = dec->metadata.m;

    info->have_container = TO_JXL_BOOL(dec->have_container);
    info->xsize = dec->metadata.size.xsize();
    info->ysize = dec->metadata.size.ysize();
    info->uses_original_profile = TO_JXL_BOOL(!meta.xyb_encoded);

    info->bits_per_sample = meta.bit_depth.bits_per_sample;
    info->exponent_bits_per_sample = meta.bit_depth.exponent_bits_per_sample;

    info->have_preview = TO_JXL_BOOL(meta.have_preview);
    info->have_animation = TO_JXL_BOOL(meta.have_animation);
    info->orientation = static_cast<JxlOrientation>(meta.orientation);

    if (!dec->keep_orientation) {
      if (info->orientation >= JXL_ORIENT_TRANSPOSE) {
        std::swap(info->xsize, info->ysize);
      }
      info->orientation = JXL_ORIENT_IDENTITY;
    }

    info->intensity_target = meta.IntensityTarget();
    if (dec->desired_intensity_target > 0) {
      info->intensity_target = dec->desired_intensity_target;
    }
    info->min_nits = meta.tone_mapping.min_nits;
    info->relative_to_max_display =
        TO_JXL_BOOL(meta.tone_mapping.relative_to_max_display);
    info->linear_below = meta.tone_mapping.linear_below;

    const jxl::ExtraChannelInfo* alpha = meta.Find(jxl::ExtraChannel::kAlpha);
    if (alpha != nullptr) {
      info->alpha_bits = alpha->bit_depth.bits_per_sample;
      info->alpha_exponent_bits = alpha->bit_depth.exponent_bits_per_sample;
      info->alpha_premultiplied = TO_JXL_BOOL(alpha->alpha_associated);
    } else {
      info->alpha_bits = 0;
      info->alpha_exponent_bits = 0;
      info->alpha_premultiplied = 0;
    }

    info->num_color_channels =
        meta.color_encoding.GetColorSpace() == jxl::ColorSpace::kGray ? 1 : 3;

    info->num_extra_channels = meta.num_extra_channels;

    if (info->have_preview) {
      info->preview.xsize = dec->metadata.m.preview_size.xsize();
      info->preview.ysize = dec->metadata.m.preview_size.ysize();
    }

    if (info->have_animation) {
      info->animation.tps_numerator = dec->metadata.m.animation.tps_numerator;
      info->animation.tps_denominator =
          dec->metadata.m.animation.tps_denominator;
      info->animation.num_loops = dec->metadata.m.animation.num_loops;
      info->animation.have_timecodes =
          TO_JXL_BOOL(dec->metadata.m.animation.have_timecodes);
    }

    if (meta.have_intrinsic_size) {
      info->intrinsic_xsize = dec->metadata.m.intrinsic_size.xsize();
      info->intrinsic_ysize = dec->metadata.m.intrinsic_size.ysize();
    } else {
      info->intrinsic_xsize = info->xsize;
      info->intrinsic_ysize = info->ysize;
    }
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelInfo(const JxlDecoder* dec,
                                               size_t index,
                                               JxlExtraChannelInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->metadata.m.extra_channel_info;

  if (index >= channels.size()) return JXL_DEC_ERROR;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  info->type = static_cast<JxlExtraChannelType>(channel.type);
  info->bits_per_sample = channel.bit_depth.bits_per_sample;
  info->exponent_bits_per_sample =
      channel.bit_depth.floating_point_sample
          ? channel.bit_depth.exponent_bits_per_sample
          : 0;
  info->dim_shift = channel.dim_shift;
  info->name_length = channel.name.size();
  info->alpha_premultiplied = TO_JXL_BOOL(channel.alpha_associated);
  info->spot_color[0] = channel.spot_color[0];
  info->spot_color[1] = channel.spot_color[1];
  info->spot_color[2] = channel.spot_color[2];
  info->spot_color[3] = channel.spot_color[3];
  info->cfa_channel = channel.cfa_channel;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelName(const JxlDecoder* dec,
                                               size_t index, char* name,
                                               size_t size) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->metadata.m.extra_channel_info;

  if (index >= channels.size()) return JXL_DEC_ERROR;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  // Also need null-termination character
  if (channel.name.size() + 1 > size) return JXL_DEC_ERROR;

  memcpy(name, channel.name.c_str(), channel.name.size() + 1);

  return JXL_DEC_SUCCESS;
}

namespace {

// Gets the jxl::ColorEncoding for the desired target, and checks errors.
// Returns the object regardless of whether the actual color space is in ICC,
// but ensures that if the color encoding is not the encoding from the
// codestream header metadata, it cannot require ICC profile.
JxlDecoderStatus GetColorEncodingForTarget(
    const JxlDecoder* dec, JxlColorProfileTarget target,
    const jxl::ColorEncoding** encoding) {
  if (!dec->got_all_headers) return JXL_DEC_NEED_MORE_INPUT;
  *encoding = nullptr;
  if (target == JXL_COLOR_PROFILE_TARGET_DATA && dec->metadata.m.xyb_encoded) {
    *encoding = &dec->passes_state->output_encoding_info.color_encoding;
  } else {
    *encoding = &dec->metadata.m.color_encoding;
  }
  return JXL_DEC_SUCCESS;
}
}  // namespace

JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
    const JxlDecoder* dec, JxlColorProfileTarget target,
    JxlColorEncoding* color_encoding) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JxlDecoderStatus status =
      GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status) return status;

  if (jxl_color_encoding->WantICC())
    return JXL_DEC_ERROR;  // Indicate no encoded profile available.

  if (color_encoding) {
    *color_encoding = jxl_color_encoding->ToExternal();
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetICCProfileSize(const JxlDecoder* dec,
                                             JxlColorProfileTarget target,
                                             size_t* size) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JxlDecoderStatus status =
      GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status != JXL_DEC_SUCCESS) return status;

  if (jxl_color_encoding->WantICC()) {
    jxl::ColorSpace color_space =
        dec->metadata.m.color_encoding.GetColorSpace();
    if (color_space == jxl::ColorSpace::kUnknown ||
        color_space == jxl::ColorSpace::kXYB) {
      // This indicates there's no ICC profile available
      // TODO(lode): for the XYB case, do we want to craft an ICC profile that
      // represents XYB as an RGB profile? It may be possible, but not with
      // only 1D transfer functions.
      return JXL_DEC_ERROR;
    }
  }

  if (size) {
    *size = jxl_color_encoding->ICC().size();
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetColorAsICCProfile(const JxlDecoder* dec,
                                                JxlColorProfileTarget target,
                                                uint8_t* icc_profile,
                                                size_t size) {
  size_t wanted_size;
  // This also checks the NEED_MORE_INPUT and the unknown/xyb cases
  JxlDecoderStatus status =
      JxlDecoderGetICCProfileSize(dec, target, &wanted_size);
  if (status != JXL_DEC_SUCCESS) return status;
  if (size < wanted_size) return JXL_API_ERROR("ICC profile output too small");

  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  status = GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status != JXL_DEC_SUCCESS) return status;

  memcpy(icc_profile, jxl_color_encoding->ICC().data(),
         jxl_color_encoding->ICC().size());

  return JXL_DEC_SUCCESS;
}

namespace {

// Returns the amount of bits needed for getting memory buffer size, and does
// all error checking required for size checking and format validity.
JxlDecoderStatus PrepareSizeCheck(const JxlDecoder* dec,
                                  const JxlPixelFormat* format, size_t* bits) {
  if (!dec->got_basic_info) {
    // Don't know image dimensions yet, cannot check for valid size.
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (!dec->coalescing &&
      (!dec->frame_header || dec->frame_stage == FrameStage::kHeader)) {
    return JXL_API_ERROR("Don't know frame dimensions yet");
  }
  if (format->num_channels > 4) {
    return JXL_API_ERROR("More than 4 channels not supported");
  }

  *bits = BitsPerChannel(format->data_type);

  if (*bits == 0) {
    return JXL_API_ERROR("Invalid/unsupported data type");
  }

  return JXL_DEC_SUCCESS;
}

}  // namespace

size_t JxlDecoderGetIntendedDownsamplingRatio(JxlDecoder* dec) {
  return dec->downsampling_target;
}

JxlDecoderStatus JxlDecoderFlushImage(JxlDecoder* dec) {
  if (!dec->image_out_buffer_set) return JXL_DEC_ERROR;
  if (dec->frame_stage != FrameStage::kFull) {
    return JXL_DEC_ERROR;
  }
  JXL_DASSERT(dec->frame_dec);
  if (!dec->frame_dec->HasDecodedDC()) {
    // FrameDecoder::Flush currently requires DC to have been decoded already
    // to work correctly.
    return JXL_DEC_ERROR;
  }

  if (!dec->frame_dec->Flush()) {
    return JXL_DEC_ERROR;
  }

  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderSetCms(JxlDecoder* dec,
                                             const JxlCmsInterface cms) {
  if (!dec->passes_state) {
    dec->passes_state =
        jxl::make_unique<jxl::PassesDecoderState>(&dec->memory_manager);
  }
  dec->passes_state->output_encoding_info.color_management_system = cms;
  dec->passes_state->output_encoding_info.cms_set = true;
  return JXL_DEC_SUCCESS;
}

static JxlDecoderStatus GetMinSize(const JxlDecoder* dec,
                                   const JxlPixelFormat* format,
                                   size_t num_channels, size_t* min_size,
                                   bool preview) {
  size_t bits;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JXL_DEC_SUCCESS) return status;
  size_t xsize;
  size_t ysize;
  if (preview) {
    xsize = dec->metadata.oriented_preview_xsize(dec->keep_orientation);
    ysize = dec->metadata.oriented_preview_ysize(dec->keep_orientation);
  } else {
    GetCurrentDimensions(dec, xsize, ysize);
  }
  if (num_channels == 0) num_channels = format->num_channels;
  size_t row_size =
      jxl::DivCeil(xsize * num_channels * bits, jxl::kBitsPerByte);
  size_t last_row_size = row_size;
  if (format->align > 1) {
    row_size = jxl::DivCeil(row_size, format->align) * format->align;
  }
  *min_size = row_size * (ysize - 1) + last_row_size;
  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderPreviewOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  if (format->num_channels < 3 &&
      !dec->image_metadata.color_encoding.IsGray()) {
    return JXL_API_ERROR("Number of channels is too low for color output");
  }
  return GetMinSize(dec, format, 0, size, true);
}

JXL_EXPORT JxlDecoderStatus JxlDecoderSetPreviewOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size) {
  if (!dec->got_basic_info || !dec->metadata.m.have_preview ||
      !(dec->orig_events_wanted & JXL_DEC_PREVIEW_IMAGE)) {
    return JXL_API_ERROR("No preview out buffer needed at this time");
  }
  if (format->num_channels < 3 &&
      !dec->image_metadata.color_encoding.IsGray()) {
    return JXL_API_ERROR("Number of channels is too low for color output");
  }

  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status =
      JxlDecoderPreviewOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->image_out_buffer_set = true;
  dec->image_out_buffer = buffer;
  dec->image_out_size = size;
  dec->image_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderImageOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  if (format->num_channels < 3 &&
      !dec->image_metadata.color_encoding.IsGray()) {
    return JXL_API_ERROR("Number of channels is too low for color output");
  }

  return GetMinSize(dec, format, 0, size, false);
}

JxlDecoderStatus JxlDecoderSetImageOutBuffer(JxlDecoder* dec,
                                             const JxlPixelFormat* format,
                                             void* buffer, size_t size) {
  if (!dec->got_basic_info || !(dec->orig_events_wanted & JXL_DEC_FULL_IMAGE)) {
    return JXL_API_ERROR("No image out buffer needed at this time");
  }
  if (dec->image_out_buffer_set && !!dec->image_out_run_callback) {
    return JXL_API_ERROR(
        "Cannot change from image out callback to image out buffer");
  }
  if (format->num_channels < 3 &&
      !dec->image_metadata.color_encoding.IsGray()) {
    return JXL_API_ERROR("Number of channels is too low for color output");
  }
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status =
      JxlDecoderImageOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->image_out_buffer_set = true;
  dec->image_out_buffer = buffer;
  dec->image_out_size = size;
  dec->image_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderExtraChannelBufferSize(const JxlDecoder* dec,
                                                  const JxlPixelFormat* format,
                                                  size_t* size,
                                                  uint32_t index) {
  if (!dec->got_basic_info || !(dec->orig_events_wanted & JXL_DEC_FULL_IMAGE)) {
    return JXL_API_ERROR("No extra channel buffer needed at this time");
  }

  if (index >= dec->metadata.m.num_extra_channels) {
    return JXL_API_ERROR("Invalid extra channel index");
  }

  return GetMinSize(dec, format, 1, size, false);
}

JxlDecoderStatus JxlDecoderSetExtraChannelBuffer(JxlDecoder* dec,
                                                 const JxlPixelFormat* format,
                                                 void* buffer, size_t size,
                                                 uint32_t index) {
  size_t min_size;
  // This also checks whether the format and index are valid and supported and
  // basic info is available.
  JxlDecoderStatus status =
      JxlDecoderExtraChannelBufferSize(dec, format, &min_size, index);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  if (dec->extra_channel_output.size() <= index) {
    dec->extra_channel_output.resize(dec->metadata.m.num_extra_channels,
                                     {{}, nullptr, 0});
  }
  // Guaranteed correct thanks to check in JxlDecoderExtraChannelBufferSize.
  JXL_DASSERT(dec->extra_channel_output.size() > index);

  dec->extra_channel_output[index].format = *format;
  dec->extra_channel_output[index].format.num_channels = 1;
  dec->extra_channel_output[index].buffer = buffer;
  dec->extra_channel_output[index].buffer_size = size;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetImageOutCallback(JxlDecoder* dec,
                                               const JxlPixelFormat* format,
                                               JxlImageOutCallback callback,
                                               void* opaque) {
  dec->simple_image_out_callback.callback = callback;
  dec->simple_image_out_callback.opaque = opaque;
  const auto init_callback =
      +[](void* init_opaque, size_t num_threads, size_t num_pixels_per_thread) {
        // No initialization to do, just reuse init_opaque as run_opaque.
        return init_opaque;
      };
  const auto run_callback =
      +[](void* run_opaque, size_t thread_id, size_t x, size_t y,
          size_t num_pixels, const void* pixels) {
        const auto* const simple_callback =
            static_cast<const JxlDecoder::SimpleImageOutCallback*>(run_opaque);
        simple_callback->callback(simple_callback->opaque, x, y, num_pixels,
                                  pixels);
      };
  const auto destroy_callback = +[](void* run_opaque) {};
  return JxlDecoderSetMultithreadedImageOutCallback(
      dec, format, init_callback, run_callback,
      /*destroy_callback=*/destroy_callback, &dec->simple_image_out_callback);
}

JxlDecoderStatus JxlDecoderSetMultithreadedImageOutCallback(
    JxlDecoder* dec, const JxlPixelFormat* format,
    JxlImageOutInitCallback init_callback, JxlImageOutRunCallback run_callback,
    JxlImageOutDestroyCallback destroy_callback, void* init_opaque) {
  if (dec->image_out_buffer_set && !!dec->image_out_buffer) {
    return JXL_API_ERROR(
        "Cannot change from image out buffer to image out callback");
  }

  if (init_callback == nullptr || run_callback == nullptr ||
      destroy_callback == nullptr) {
    return JXL_API_ERROR("All callbacks are required");
  }

  // Perform error checking for invalid format.
  size_t bits_sink;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits_sink);
  if (status != JXL_DEC_SUCCESS) return status;

  dec->image_out_buffer_set = true;
  dec->image_out_init_callback = init_callback;
  dec->image_out_run_callback = run_callback;
  dec->image_out_destroy_callback = destroy_callback;
  dec->image_out_init_opaque = init_opaque;
  dec->image_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetFrameHeader(const JxlDecoder* dec,
                                          JxlFrameHeader* header) {
  if (!dec->frame_header || dec->frame_stage == FrameStage::kHeader) {
    return JXL_API_ERROR("no frame header available");
  }
  const auto& metadata = dec->metadata.m;
  memset(header, 0, sizeof(*header));
  if (metadata.have_animation) {
    header->duration = dec->frame_header->animation_frame.duration;
    if (metadata.animation.have_timecodes) {
      header->timecode = dec->frame_header->animation_frame.timecode;
    }
  }
  header->name_length = dec->frame_header->name.size();
  header->is_last = TO_JXL_BOOL(dec->frame_header->is_last);
  size_t xsize;
  size_t ysize;
  GetCurrentDimensions(dec, xsize, ysize);
  header->layer_info.xsize = xsize;
  header->layer_info.ysize = ysize;
  if (!dec->coalescing && dec->frame_header->custom_size_or_origin) {
    header->layer_info.crop_x0 = dec->frame_header->frame_origin.x0;
    header->layer_info.crop_y0 = dec->frame_header->frame_origin.y0;
    header->layer_info.have_crop = JXL_TRUE;
  } else {
    header->layer_info.crop_x0 = 0;
    header->layer_info.crop_y0 = 0;
    header->layer_info.have_crop = JXL_FALSE;
  }
  if (!dec->keep_orientation && !dec->coalescing) {
    // orient the crop offset
    size_t W = dec->metadata.oriented_xsize(false);
    size_t H = dec->metadata.oriented_ysize(false);
    if (metadata.orientation > 4) {
      std::swap(header->layer_info.crop_x0, header->layer_info.crop_y0);
    }
    size_t o = (metadata.orientation - 1) & 3;
    if (o > 0 && o < 3) {
      header->layer_info.crop_x0 = W - xsize - header->layer_info.crop_x0;
    }
    if (o > 1) {
      header->layer_info.crop_y0 = H - ysize - header->layer_info.crop_y0;
    }
  }
  if (dec->coalescing) {
    header->layer_info.blend_info.blendmode = JXL_BLEND_REPLACE;
    header->layer_info.blend_info.source = 0;
    header->layer_info.blend_info.alpha = 0;
    header->layer_info.blend_info.clamp = JXL_FALSE;
    header->layer_info.save_as_reference = 0;
  } else {
    header->layer_info.blend_info.blendmode =
        static_cast<JxlBlendMode>(dec->frame_header->blending_info.mode);
    header->layer_info.blend_info.source =
        dec->frame_header->blending_info.source;
    header->layer_info.blend_info.alpha =
        dec->frame_header->blending_info.alpha_channel;
    header->layer_info.blend_info.clamp =
        TO_JXL_BOOL(dec->frame_header->blending_info.clamp);
    header->layer_info.save_as_reference = dec->frame_header->save_as_reference;
  }
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelBlendInfo(const JxlDecoder* dec,
                                                    size_t index,
                                                    JxlBlendInfo* blend_info) {
  if (!dec->frame_header || dec->frame_stage == FrameStage::kHeader) {
    return JXL_API_ERROR("no frame header available");
  }
  const auto& metadata = dec->metadata.m;
  if (index >= metadata.num_extra_channels) {
    return JXL_API_ERROR("Invalid extra channel index");
  }
  blend_info->blendmode = static_cast<JxlBlendMode>(
      dec->frame_header->extra_channel_blending_info[index].mode);
  blend_info->source =
      dec->frame_header->extra_channel_blending_info[index].source;
  blend_info->alpha =
      dec->frame_header->extra_channel_blending_info[index].alpha_channel;
  blend_info->clamp =
      TO_JXL_BOOL(dec->frame_header->extra_channel_blending_info[index].clamp);
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetFrameName(const JxlDecoder* dec, char* name,
                                        size_t size) {
  if (!dec->frame_header || dec->frame_stage == FrameStage::kHeader) {
    return JXL_API_ERROR("no frame header available");
  }
  if (size < dec->frame_header->name.size() + 1) {
    return JXL_API_ERROR("too small frame name output buffer");
  }
  memcpy(name, dec->frame_header->name.c_str(),
         dec->frame_header->name.size() + 1);

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetPreferredColorProfile(
    JxlDecoder* dec, const JxlColorEncoding* color_encoding) {
  return JxlDecoderSetOutputColorProfile(dec, color_encoding,
                                         /*icc_data=*/nullptr, /*icc_size=*/0);
}

JxlDecoderStatus JxlDecoderSetOutputColorProfile(
    JxlDecoder* dec, const JxlColorEncoding* color_encoding,
    const uint8_t* icc_data, size_t icc_size) {
  if ((color_encoding != nullptr) && (icc_data != nullptr)) {
    return JXL_API_ERROR("cannot set both color_encoding and icc_data");
  }
  if ((color_encoding == nullptr) && (icc_data == nullptr)) {
    return JXL_API_ERROR("one of color_encoding and icc_data must be set");
  }
  if (!dec->got_all_headers) {
    return JXL_API_ERROR("color info not yet available");
  }
  if (dec->post_headers) {
    return JXL_API_ERROR("too late to set the color encoding");
  }
  if ((!dec->passes_state->output_encoding_info.cms_set) &&
      (icc_data != nullptr)) {
    return JXL_API_ERROR(
        "must set color management system via JxlDecoderSetCms");
  }
  auto& output_encoding = dec->passes_state->output_encoding_info;
  if (color_encoding) {
    if (dec->image_metadata.color_encoding.IsGray() &&
        color_encoding->color_space != JXL_COLOR_SPACE_GRAY &&
        dec->image_out_buffer_set && dec->image_out_format.num_channels < 3) {
      return JXL_API_ERROR("Number of channels is too low for color output");
    }
    if (color_encoding->color_space == JXL_COLOR_SPACE_UNKNOWN) {
      return JXL_API_ERROR("Unknown output colorspace");
    }
    jxl::ColorEncoding c_out;
    JXL_API_RETURN_IF_ERROR(c_out.FromExternal(*color_encoding));
    JXL_API_RETURN_IF_ERROR(!c_out.ICC().empty());
    if (!c_out.SameColorEncoding(output_encoding.color_encoding)) {
      JXL_API_RETURN_IF_ERROR(output_encoding.MaybeSetColorEncoding(c_out));
      dec->image_metadata.color_encoding = output_encoding.color_encoding;
    }
    return JXL_DEC_SUCCESS;
  }
  // icc_data != nullptr
  // TODO(firsching): implement setting output color profile from icc_data.
  jxl::ColorEncoding c_dst;
  std::vector<uint8_t> padded_icc;
  padded_icc.assign(icc_data, icc_data + icc_size);
  if (!c_dst.SetICC(std::move(padded_icc),
                    &output_encoding.color_management_system)) {
    return JXL_API_ERROR(
        "setting output color profile from icc_data not yet implemented.");
  }
  JXL_API_RETURN_IF_ERROR(
      static_cast<int>(output_encoding.MaybeSetColorEncoding(c_dst)));

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetDesiredIntensityTarget(
    JxlDecoder* dec, float desired_intensity_target) {
  if (desired_intensity_target < 0) {
    return JXL_API_ERROR("negative intensity target requested");
  }
  dec->desired_intensity_target = desired_intensity_target;
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetBoxBuffer(JxlDecoder* dec, uint8_t* data,
                                        size_t size) {
  if (dec->box_out_buffer_set) {
    return JXL_API_ERROR("must release box buffer before setting it again");
  }
  if (!dec->box_event) {
    return JXL_API_ERROR("can only set box buffer after box event");
  }

  dec->box_out_buffer_set = true;
  dec->box_out_buffer_set_current_box = true;
  dec->box_out_buffer = data;
  dec->box_out_buffer_size = size;
  dec->box_out_buffer_pos = 0;
  return JXL_DEC_SUCCESS;
}

size_t JxlDecoderReleaseBoxBuffer(JxlDecoder* dec) {
  if (!dec->box_out_buffer_set) {
    return 0;
  }
  size_t result = dec->box_out_buffer_size - dec->box_out_buffer_pos;
  dec->box_out_buffer_set = false;
  dec->box_out_buffer = nullptr;
  dec->box_out_buffer_size = 0;
  if (!dec->box_out_buffer_set_current_box) {
    dec->box_out_buffer_begin = 0;
  } else {
    dec->box_out_buffer_begin += dec->box_out_buffer_pos;
  }
  dec->box_out_buffer_set_current_box = false;
  return result;
}

JxlDecoderStatus JxlDecoderSetDecompressBoxes(JxlDecoder* dec,
                                              JXL_BOOL decompress) {
  // TODO(lode): return error if libbrotli is not compiled in the jxl decoding
  // library
  dec->decompress_boxes = FROM_JXL_BOOL(decompress);
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetBoxType(JxlDecoder* dec, JxlBoxType type,
                                      JXL_BOOL decompressed) {
  if (!dec->box_event) {
    return JXL_API_ERROR("can only get box info after JXL_DEC_BOX event");
  }
  if (decompressed) {
    memcpy(type, dec->box_decoded_type, sizeof(dec->box_decoded_type));
  } else {
    memcpy(type, dec->box_type, sizeof(dec->box_type));
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetBoxSizeRaw(const JxlDecoder* dec,
                                         uint64_t* size) {
  if (!dec->box_event) {
    return JXL_API_ERROR("can only get box info after JXL_DEC_BOX event");
  }
  if (size) {
    *size = dec->box_size;
  }
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetBoxSizeContents(const JxlDecoder* dec,
                                              uint64_t* size) {
  if (!dec->box_event) {
    return JXL_API_ERROR("can only get box info after JXL_DEC_BOX event");
  }
  if (size) {
    *size = dec->box_contents_size;
  }
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetProgressiveDetail(JxlDecoder* dec,
                                                JxlProgressiveDetail detail) {
  if (detail != kDC && detail != kLastPasses && detail != kPasses) {
    return JXL_API_ERROR(
        "Values other than kDC (%d), kLastPasses (%d) and kPasses (%d), "
        "like %d are not implemented.",
        kDC, kLastPasses, kPasses, detail);
  }
  dec->prog_detail = detail;
  return JXL_DEC_SUCCESS;
}

namespace {

template <typename T>
JxlDecoderStatus VerifyOutputBitDepth(JxlBitDepth bit_depth, const T& metadata,
                                      JxlPixelFormat format) {
  uint32_t bits_per_sample = GetBitDepth(bit_depth, metadata, format);
  if (bits_per_sample == 0) return JXL_API_ERROR("Invalid output bit depth");
  if (format.data_type == JXL_TYPE_UINT8 && bits_per_sample > 8) {
    return JXL_API_ERROR("Invalid bit depth %u for uint8 output",
                         bits_per_sample);
  } else if (format.data_type == JXL_TYPE_UINT16 && bits_per_sample > 16) {
    return JXL_API_ERROR("Invalid bit depth %u for uint16 output",
                         bits_per_sample);
  }
  return JXL_DEC_SUCCESS;
}

}  // namespace

JxlDecoderStatus JxlDecoderSetImageOutBitDepth(JxlDecoder* dec,
                                               const JxlBitDepth* bit_depth) {
  if (!dec->image_out_buffer_set) {
    return JXL_API_ERROR("No image out buffer was set.");
  }
  JXL_API_RETURN_IF_ERROR(
      VerifyOutputBitDepth(*bit_depth, dec->metadata.m, dec->image_out_format));
  dec->image_out_bit_depth = *bit_depth;
  return JXL_DEC_SUCCESS;
}
