// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Demux API.
// Enables extraction of image and extended format data from WebP files.

// Code Example: Demuxing WebP data to extract all the frames, ICC profile
// and EXIF/XMP metadata.
/*
  WebPDemuxer* demux = WebPDemux(&webp_data);

  uint32_t width = WebPDemuxGetI(demux, WEBP_FF_CANVAS_WIDTH);
  uint32_t height = WebPDemuxGetI(demux, WEBP_FF_CANVAS_HEIGHT);
  // ... (Get information about the features present in the WebP file).
  uint32_t flags = WebPDemuxGetI(demux, WEBP_FF_FORMAT_FLAGS);

  // ... (Iterate over all frames).
  WebPIterator iter;
  if (WebPDemuxGetFrame(demux, 1, &iter)) {
    do {
      // ... (Consume 'iter'; e.g. Decode 'iter.fragment' with WebPDecode(),
      // ... and get other frame properties like width, height, offsets etc.
      // ... see 'struct WebPIterator' below for more info).
    } while (WebPDemuxNextFrame(&iter));
    WebPDemuxReleaseIterator(&iter);
  }

  // ... (Extract metadata).
  WebPChunkIterator chunk_iter;
  if (flags & ICCP_FLAG) WebPDemuxGetChunk(demux, "ICCP", 1, &chunk_iter);
  // ... (Consume the ICC profile in 'chunk_iter.chunk').
  WebPDemuxReleaseChunkIterator(&chunk_iter);
  if (flags & EXIF_FLAG) WebPDemuxGetChunk(demux, "EXIF", 1, &chunk_iter);
  // ... (Consume the EXIF metadata in 'chunk_iter.chunk').
  WebPDemuxReleaseChunkIterator(&chunk_iter);
  if (flags & XMP_FLAG) WebPDemuxGetChunk(demux, "XMP ", 1, &chunk_iter);
  // ... (Consume the XMP metadata in 'chunk_iter.chunk').
  WebPDemuxReleaseChunkIterator(&chunk_iter);
  WebPDemuxDelete(demux);
*/

#ifndef WEBP_WEBP_DEMUX_H_
#define WEBP_WEBP_DEMUX_H_

#include "./mux_types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define WEBP_DEMUX_ABI_VERSION 0x0101    // MAJOR(8b) + MINOR(8b)

// Note: forward declaring enumerations is not allowed in (strict) C and C++,
// the types are left here for reference.
// typedef enum WebPDemuxState WebPDemuxState;
// typedef enum WebPFormatFeature WebPFormatFeature;
typedef struct WebPDemuxer WebPDemuxer;
typedef struct WebPIterator WebPIterator;
typedef struct WebPChunkIterator WebPChunkIterator;

//------------------------------------------------------------------------------

// Returns the version number of the demux library, packed in hexadecimal using
// 8bits for each of major/minor/revision. E.g: v2.5.7 is 0x020507.
WEBP_EXTERN(int) WebPGetDemuxVersion(void);

//------------------------------------------------------------------------------
// Life of a Demux object

typedef enum WebPDemuxState {
  WEBP_DEMUX_PARSE_ERROR    = -1,  // An error occurred while parsing.
  WEBP_DEMUX_PARSING_HEADER =  0,  // Not enough data to parse full header.
  WEBP_DEMUX_PARSED_HEADER  =  1,  // Header parsing complete,
                                   // data may be available.
  WEBP_DEMUX_DONE           =  2   // Entire file has been parsed.
} WebPDemuxState;

// Internal, version-checked, entry point
WEBP_EXTERN(WebPDemuxer*) WebPDemuxInternal(
    const WebPData*, int, WebPDemuxState*, int);

// Parses the full WebP file given by 'data'.
// Returns a WebPDemuxer object on successful parse, NULL otherwise.
static WEBP_INLINE WebPDemuxer* WebPDemux(const WebPData* data) {
  return WebPDemuxInternal(data, 0, NULL, WEBP_DEMUX_ABI_VERSION);
}

// Parses the possibly incomplete WebP file given by 'data'.
// If 'state' is non-NULL it will be set to indicate the status of the demuxer.
// Returns NULL in case of error or if there isn't enough data to start parsing;
// and a WebPDemuxer object on successful parse.
// Note that WebPDemuxer keeps internal pointers to 'data' memory segment.
// If this data is volatile, the demuxer object should be deleted (by calling
// WebPDemuxDelete()) and WebPDemuxPartial() called again on the new data.
// This is usually an inexpensive operation.
static WEBP_INLINE WebPDemuxer* WebPDemuxPartial(
    const WebPData* data, WebPDemuxState* state) {
  return WebPDemuxInternal(data, 1, state, WEBP_DEMUX_ABI_VERSION);
}

// Frees memory associated with 'dmux'.
WEBP_EXTERN(void) WebPDemuxDelete(WebPDemuxer* dmux);

//------------------------------------------------------------------------------
// Data/information extraction.

typedef enum WebPFormatFeature {
  WEBP_FF_FORMAT_FLAGS,  // Extended format flags present in the 'VP8X' chunk.
  WEBP_FF_CANVAS_WIDTH,
  WEBP_FF_CANVAS_HEIGHT,
  WEBP_FF_LOOP_COUNT,
  WEBP_FF_BACKGROUND_COLOR,
  WEBP_FF_FRAME_COUNT    // Number of frames present in the demux object.
                         // In case of a partial demux, this is the number of
                         // frames seen so far, with the last frame possibly
                         // being partial.
} WebPFormatFeature;

// Get the 'feature' value from the 'dmux'.
// NOTE: values are only valid if WebPDemux() was used or WebPDemuxPartial()
// returned a state > WEBP_DEMUX_PARSING_HEADER.
WEBP_EXTERN(uint32_t) WebPDemuxGetI(
    const WebPDemuxer* dmux, WebPFormatFeature feature);

//------------------------------------------------------------------------------
// Frame iteration.

struct WebPIterator {
  int frame_num;
  int num_frames;          // equivalent to WEBP_FF_FRAME_COUNT.
  int fragment_num;
  int num_fragments;
  int x_offset, y_offset;  // offset relative to the canvas.
  int width, height;       // dimensions of this frame or fragment.
  int duration;            // display duration in milliseconds.
  WebPMuxAnimDispose dispose_method;  // dispose method for the frame.
  int complete;   // true if 'fragment' contains a full frame. partial images
                  // may still be decoded with the WebP incremental decoder.
  WebPData fragment;  // The frame or fragment given by 'frame_num' and
                      // 'fragment_num'.
  int has_alpha;      // True if the frame or fragment contains transparency.
  WebPMuxAnimBlend blend_method;  // Blend operation for the frame.

  uint32_t pad[2];         // padding for later use.
  void* private_;          // for internal use only.
};

// Retrieves frame 'frame_number' from 'dmux'.
// 'iter->fragment' points to the first fragment on return from this function.
// Individual fragments may be extracted using WebPDemuxSelectFragment().
// Setting 'frame_number' equal to 0 will return the last frame of the image.
// Returns false if 'dmux' is NULL or frame 'frame_number' is not present.
// Call WebPDemuxReleaseIterator() when use of the iterator is complete.
// NOTE: 'dmux' must persist for the lifetime of 'iter'.
WEBP_EXTERN(int) WebPDemuxGetFrame(
    const WebPDemuxer* dmux, int frame_number, WebPIterator* iter);

// Sets 'iter->fragment' to point to the next ('iter->frame_num' + 1) or
// previous ('iter->frame_num' - 1) frame. These functions do not loop.
// Returns true on success, false otherwise.
WEBP_EXTERN(int) WebPDemuxNextFrame(WebPIterator* iter);
WEBP_EXTERN(int) WebPDemuxPrevFrame(WebPIterator* iter);

// Sets 'iter->fragment' to reflect fragment number 'fragment_num'.
// Returns true if fragment 'fragment_num' is present, false otherwise.
WEBP_EXTERN(int) WebPDemuxSelectFragment(WebPIterator* iter, int fragment_num);

// Releases any memory associated with 'iter'.
// Must be called before any subsequent calls to WebPDemuxGetChunk() on the same
// iter. Also, must be called before destroying the associated WebPDemuxer with
// WebPDemuxDelete().
WEBP_EXTERN(void) WebPDemuxReleaseIterator(WebPIterator* iter);

//------------------------------------------------------------------------------
// Chunk iteration.

struct WebPChunkIterator {
  // The current and total number of chunks with the fourcc given to
  // WebPDemuxGetChunk().
  int chunk_num;
  int num_chunks;
  WebPData chunk;    // The payload of the chunk.

  uint32_t pad[6];   // padding for later use
  void* private_;
};

// Retrieves the 'chunk_number' instance of the chunk with id 'fourcc' from
// 'dmux'.
// 'fourcc' is a character array containing the fourcc of the chunk to return,
// e.g., "ICCP", "XMP ", "EXIF", etc.
// Setting 'chunk_number' equal to 0 will return the last chunk in a set.
// Returns true if the chunk is found, false otherwise. Image related chunk
// payloads are accessed through WebPDemuxGetFrame() and related functions.
// Call WebPDemuxReleaseChunkIterator() when use of the iterator is complete.
// NOTE: 'dmux' must persist for the lifetime of the iterator.
WEBP_EXTERN(int) WebPDemuxGetChunk(const WebPDemuxer* dmux,
                                   const char fourcc[4], int chunk_number,
                                   WebPChunkIterator* iter);

// Sets 'iter->chunk' to point to the next ('iter->chunk_num' + 1) or previous
// ('iter->chunk_num' - 1) chunk. These functions do not loop.
// Returns true on success, false otherwise.
WEBP_EXTERN(int) WebPDemuxNextChunk(WebPChunkIterator* iter);
WEBP_EXTERN(int) WebPDemuxPrevChunk(WebPChunkIterator* iter);

// Releases any memory associated with 'iter'.
// Must be called before destroying the associated WebPDemuxer with
// WebPDemuxDelete().
WEBP_EXTERN(void) WebPDemuxReleaseChunkIterator(WebPChunkIterator* iter);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  /* WEBP_WEBP_DEMUX_H_ */
