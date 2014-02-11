// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
//  RIFF container manipulation for WEBP images.
//
// Authors: Urvang (urvang@google.com)
//          Vikas (vikasa@google.com)

// This API allows manipulation of WebP container images containing features
// like Color profile, XMP metadata, Animation and Tiling.
//
// Code Example#1: Creating a MUX with image data, color profile and XMP
// metadata.
//
//   int copy_data = 0;
//   WebPMux* mux = WebPMuxNew();
//   // ... (Prepare image data).
//   WebPMuxSetImage(mux, &image, copy_data);
//   // ... (Prepare ICCP color profile data).
//   WebPMuxSetColorProfile(mux, &icc_profile, copy_data);
//   // ... (Prepare XMP metadata).
//   WebPMuxSetMetadata(mux, &xmp, copy_data);
//   // Get data from mux in WebP RIFF format.
//   WebPMuxAssemble(mux, &output_data);
//   WebPMuxDelete(mux);
//   // ... (Consume output_data; e.g. write output_data.bytes_ to file).
//   WebPDataClear(&output_data);
//
// Code Example#2: Get image and color profile data from a WebP file.
//
//   int copy_data = 0;
//   // ... (Read data from file).
//   WebPMux* mux = WebPMuxCreate(&data, copy_data);
//   WebPMuxGetImage(mux, &image);
//   // ... (Consume image; e.g. call WebPDecode() to decode the data).
//   WebPMuxGetColorProfile(mux, &icc_profile);
//   // ... (Consume icc_data).
//   WebPMuxDelete(mux);
//   free(data);

#ifndef WEBP_WEBP_MUX_H_
#define WEBP_WEBP_MUX_H_

#include "./types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define WEBP_MUX_ABI_VERSION 0x0100        // MAJOR(8b) + MINOR(8b)

// Error codes
typedef enum {
  WEBP_MUX_OK                 =  1,
  WEBP_MUX_NOT_FOUND          =  0,
  WEBP_MUX_INVALID_ARGUMENT   = -1,
  WEBP_MUX_BAD_DATA           = -2,
  WEBP_MUX_MEMORY_ERROR       = -3,
  WEBP_MUX_NOT_ENOUGH_DATA    = -4
} WebPMuxError;

// Flag values for different features used in VP8X chunk.
typedef enum {
  TILE_FLAG       = 0x00000001,
  ANIMATION_FLAG  = 0x00000002,
  ICCP_FLAG       = 0x00000004,
  META_FLAG       = 0x00000008,
  ALPHA_FLAG      = 0x00000010
} WebPFeatureFlags;

// IDs for different types of chunks.
typedef enum {
  WEBP_CHUNK_VP8X,     // VP8X
  WEBP_CHUNK_ICCP,     // ICCP
  WEBP_CHUNK_LOOP,     // LOOP
  WEBP_CHUNK_FRAME,    // FRM
  WEBP_CHUNK_TILE,     // TILE
  WEBP_CHUNK_ALPHA,    // ALPH
  WEBP_CHUNK_IMAGE,    // VP8/VP8L
  WEBP_CHUNK_META,     // META
  WEBP_CHUNK_UNKNOWN,  // Other chunks.
  WEBP_CHUNK_NIL
} WebPChunkId;

typedef struct WebPMux WebPMux;   // main opaque object.

// Data type used to describe 'raw' data, e.g., chunk data
// (ICC profile, metadata) and WebP compressed image data.
typedef struct {
  const uint8_t* bytes_;
  size_t size_;
} WebPData;

//------------------------------------------------------------------------------
// Manipulation of a WebPData object.

// Initializes the contents of the 'webp_data' object with default values.
WEBP_EXTERN(void) WebPDataInit(WebPData* webp_data);

// Clears the contents of the 'webp_data' object by calling free(). Does not
// deallocate the object itself.
WEBP_EXTERN(void) WebPDataClear(WebPData* webp_data);

// Allocates necessary storage for 'dst' and copies the contents of 'src'.
// Returns true on success.
WEBP_EXTERN(int) WebPDataCopy(const WebPData* src, WebPData* dst);

//------------------------------------------------------------------------------
// Life of a Mux object

// Internal, version-checked, entry point
WEBP_EXTERN(WebPMux*) WebPNewInternal(int);

// Creates an empty mux object.
// Returns:
//   A pointer to the newly created empty mux object.
static WEBP_INLINE WebPMux* WebPMuxNew(void) {
  return WebPNewInternal(WEBP_MUX_ABI_VERSION);
}

// Deletes the mux object.
// Parameters:
//   mux - (in/out) object to be deleted
WEBP_EXTERN(void) WebPMuxDelete(WebPMux* mux);

//------------------------------------------------------------------------------
// Mux creation.

// Internal, version-checked, entry point
WEBP_EXTERN(WebPMux*) WebPMuxCreateInternal(const WebPData*, int, int);

// Creates a mux object from raw data given in WebP RIFF format.
// Parameters:
//   bitstream - (in) the bitstream data in WebP RIFF format
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   A pointer to the mux object created from given data - on success.
//   NULL - In case of invalid data or memory error.
static WEBP_INLINE WebPMux* WebPMuxCreate(const WebPData* bitstream,
                                          int copy_data) {
  return WebPMuxCreateInternal(bitstream, copy_data, WEBP_MUX_ABI_VERSION);
}

//------------------------------------------------------------------------------
// Single Image.

// Sets the image in the mux object. Any existing images (including frame/tile)
// will be removed.
// Parameters:
//   mux - (in/out) object in which the image is to be set
//   bitstream - (in) can either be a raw VP8/VP8L bitstream or a single-image
//               WebP file (non-animated and non-tiled)
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL or bitstream is NULL.
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxSetImage(WebPMux* mux,
                                          const WebPData* bitstream,
                                          int copy_data);

// Gets image data from the mux object.
// The content of 'bitstream' is allocated using malloc(), and NOT
// owned by the 'mux' object. It MUST be deallocated by the caller by calling
// WebPDataClear().
// Parameters:
//   mux - (in) object from which the image is to be fetched
//   bitstream - (out) the image data
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux or bitstream is NULL
//                               OR mux contains animation/tiling.
//   WEBP_MUX_NOT_FOUND - if image is not present in mux object.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetImage(const WebPMux* mux,
                                          WebPData* bitstream);

// Deletes the image in the mux object.
// Parameters:
//   mux - (in/out) object from which the image is to be deleted
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//                               OR if mux contains animation/tiling.
//   WEBP_MUX_NOT_FOUND - if image is not present in mux object.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxDeleteImage(WebPMux* mux);

//------------------------------------------------------------------------------
// XMP Metadata.

// Sets the XMP metadata in the mux object. Any existing metadata chunk(s) will
// be removed.
// Parameters:
//   mux - (in/out) object to which the XMP metadata is to be added
//   metadata - (in) the XMP metadata data to be added
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux or metadata is NULL.
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxSetMetadata(WebPMux* mux,
                                             const WebPData* metadata,
                                             int copy_data);

// Gets a reference to the XMP metadata in the mux object.
// The caller should NOT free the returned data.
// Parameters:
//   mux - (in) object from which the XMP metadata is to be fetched
//   metadata - (out) XMP metadata
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux or metadata is NULL.
//   WEBP_MUX_NOT_FOUND - if metadata is not present in mux object.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetMetadata(const WebPMux* mux,
                                             WebPData* metadata);

// Deletes the XMP metadata in the mux object.
// Parameters:
//   mux - (in/out) object from which XMP metadata is to be deleted
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//   WEBP_MUX_NOT_FOUND - If mux does not contain metadata.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxDeleteMetadata(WebPMux* mux);

//------------------------------------------------------------------------------
// ICC Color Profile.

// Sets the color profile in the mux object. Any existing color profile chunk(s)
// will be removed.
// Parameters:
//   mux - (in/out) object to which the color profile is to be added
//   color_profile - (in) the color profile data to be added
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux or color_profile is NULL
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error
//   WEBP_MUX_OK - on success
WEBP_EXTERN(WebPMuxError) WebPMuxSetColorProfile(WebPMux* mux,
                                                 const WebPData* color_profile,
                                                 int copy_data);

// Gets a reference to the color profile in the mux object.
// The caller should NOT free the returned data.
// Parameters:
//   mux - (in) object from which the color profile data is to be fetched
//   color_profile - (out) color profile data
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux or color_profile is NULL.
//   WEBP_MUX_NOT_FOUND - if color profile is not present in mux object.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetColorProfile(const WebPMux* mux,
                                                 WebPData* color_profile);

// Deletes the color profile in the mux object.
// Parameters:
//   mux - (in/out) object from which color profile is to be deleted
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//   WEBP_MUX_NOT_FOUND - If mux does not contain color profile.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxDeleteColorProfile(WebPMux* mux);

//------------------------------------------------------------------------------
// Animation.

// Adds an animation frame at the end of the mux object.
// Note: as WebP only supports even offsets, any odd offset will be snapped to
// an even location using: offset &= ~1
// Parameters:
//   mux - (in/out) object to which an animation frame is to be added
//   bitstream - (in) the image data corresponding to the frame. It can either
//               be a raw VP8/VP8L bitstream or a single-image WebP file
//               (non-animated and non-tiled)
//   x_offset - (in) x-offset of the frame to be added
//   y_offset - (in) y-offset of the frame to be added
//   duration - (in) duration of the frame to be added (in milliseconds)
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL or bitstream is NULL
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxPushFrame(
    WebPMux* mux, const WebPData* bitstream,
    int x_offset, int y_offset, int duration, int copy_data);

// TODO(urvang): Create a struct as follows to reduce argument list size:
// typedef struct {
//  WebPData bitstream;
//  int x_offset, y_offset;
//  int duration;
// } FrameInfo;

// Gets the nth animation frame from the mux object.
// The content of 'bitstream' is allocated using malloc(), and NOT
// owned by the 'mux' object. It MUST be deallocated by the caller by calling
// WebPDataClear().
// nth=0 has a special meaning - last position.
// Parameters:
//   mux - (in) object from which the info is to be fetched
//   nth - (in) index of the frame in the mux object
//   bitstream - (out) the image data
//   x_offset - (out) x-offset of the returned frame
//   y_offset - (out) y-offset of the returned frame
//   duration - (out) duration of the returned frame (in milliseconds)
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux, bitstream, x_offset,
//                               y_offset, or duration is NULL
//   WEBP_MUX_NOT_FOUND - if there are less than nth frames in the mux object.
//   WEBP_MUX_BAD_DATA - if nth frame chunk in mux is invalid.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetFrame(
    const WebPMux* mux, uint32_t nth, WebPData* bitstream,
    int* x_offset, int* y_offset, int* duration);

// Deletes an animation frame from the mux object.
// nth=0 has a special meaning - last position.
// Parameters:
//   mux - (in/out) object from which a frame is to be deleted
//   nth - (in) The position from which the frame is to be deleted
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//   WEBP_MUX_NOT_FOUND - If there are less than nth frames in the mux object
//                        before deletion.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxDeleteFrame(WebPMux* mux, uint32_t nth);

// Sets the animation loop count in the mux object. Any existing loop count
// value(s) will be removed.
// Parameters:
//   mux - (in/out) object in which loop chunk is to be set/added
//   loop_count - (in) animation loop count value.
//                Note that loop_count of zero denotes infinite loop.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxSetLoopCount(WebPMux* mux, int loop_count);

// Gets the animation loop count from the mux object.
// Parameters:
//   mux - (in) object from which the loop count is to be fetched
//   loop_count - (out) the loop_count value present in the LOOP chunk
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either of mux or loop_count is NULL
//   WEBP_MUX_NOT_FOUND - if loop chunk is not present in mux object.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetLoopCount(const WebPMux* mux,
                                              int* loop_count);

//------------------------------------------------------------------------------
// Tiling.

// Adds a tile at the end of the mux object.
// Note: as WebP only supports even offsets, any odd offset will be snapped to
// an even location using: offset &= ~1
// Parameters:
//   mux - (in/out) object to which a tile is to be added.
//   bitstream - (in) the image data corresponding to the frame. It can either
//               be a raw VP8/VP8L bitstream or a single-image WebP file
//               (non-animated and non-tiled)
//   x_offset - (in) x-offset of the tile to be added
//   y_offset - (in) y-offset of the tile to be added
//   copy_data - (in) value 1 indicates given data WILL copied to the mux, and
//               value 0 indicates data will NOT be copied.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL or bitstream is NULL
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxPushTile(
    WebPMux* mux, const WebPData* bitstream,
    int x_offset, int y_offset, int copy_data);

// Gets the nth tile from the mux object.
// The content of 'bitstream' is allocated using malloc(), and NOT
// owned by the 'mux' object. It MUST be deallocated by the caller by calling
// WebPDataClear().
// nth=0 has a special meaning - last position.
// Parameters:
//   mux - (in) object from which the info is to be fetched
//   nth - (in) index of the tile in the mux object
//   bitstream - (out) the image data
//   x_offset - (out) x-offset of the returned tile
//   y_offset - (out) y-offset of the returned tile
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux, bitstream, x_offset or
//                               y_offset is NULL
//   WEBP_MUX_NOT_FOUND - if there are less than nth tiles in the mux object.
//   WEBP_MUX_BAD_DATA - if nth tile chunk in mux is invalid.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetTile(
    const WebPMux* mux, uint32_t nth, WebPData* bitstream,
    int* x_offset, int* y_offset);

// Deletes a tile from the mux object.
// nth=0 has a special meaning - last position
// Parameters:
//   mux - (in/out) object from which a tile is to be deleted
//   nth - (in) The position from which the tile is to be deleted
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux is NULL
//   WEBP_MUX_NOT_FOUND - If there are less than nth tiles in the mux object
//                        before deletion.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxDeleteTile(WebPMux* mux, uint32_t nth);

//------------------------------------------------------------------------------
// Misc Utilities.

// Gets the feature flags from the mux object.
// Parameters:
//   mux - (in) object from which the features are to be fetched
//   flags - (out) the flags specifying which features are present in the
//           mux object. This will be an OR of various flag values.
//           Enum 'WebPFeatureFlags' can be used to test individual flag values.
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if mux or flags is NULL
//   WEBP_MUX_NOT_FOUND - if VP8X chunk is not present in mux object.
//   WEBP_MUX_BAD_DATA - if VP8X chunk in mux is invalid.
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxGetFeatures(const WebPMux* mux,
                                             uint32_t* flags);

// Gets number of chunks having tag value tag in the mux object.
// Parameters:
//   mux - (in) object from which the info is to be fetched
//   id - (in) chunk id specifying the type of chunk
//   num_elements - (out) number of chunks with the given chunk id
// Returns:
//   WEBP_MUX_INVALID_ARGUMENT - if either mux, or num_elements is NULL
//   WEBP_MUX_OK - on success.
WEBP_EXTERN(WebPMuxError) WebPMuxNumChunks(const WebPMux* mux,
                                           WebPChunkId id, int* num_elements);

// Assembles all chunks in WebP RIFF format and returns in 'assembled_data'.
// This function also validates the mux object.
// Note: The content of 'assembled_data' will be ignored and overwritten.
// Also, the content of 'assembled_data' is allocated using malloc(), and NOT
// owned by the 'mux' object. It MUST be deallocated by the caller by calling
// WebPDataClear().
// Parameters:
//   mux - (in/out) object whose chunks are to be assembled
//   assembled_data - (out) assembled WebP data
// Returns:
//   WEBP_MUX_BAD_DATA - if mux object is invalid.
//   WEBP_MUX_INVALID_ARGUMENT - if either mux, output_data or output_size is
//                               NULL.
//   WEBP_MUX_MEMORY_ERROR - on memory allocation error.
//   WEBP_MUX_OK - on success
WEBP_EXTERN(WebPMuxError) WebPMuxAssemble(WebPMux* mux,
                                          WebPData* assembled_data);

//------------------------------------------------------------------------------
// Demux API.
// Enables extraction of image and extended format data from WebP files.

#define WEBP_DEMUX_ABI_VERSION 0x0100    // MAJOR(8b) + MINOR(8b)

typedef struct WebPDemuxer WebPDemuxer;

typedef enum {
  WEBP_DEMUX_PARSING_HEADER,  // Not enough data to parse full header.
  WEBP_DEMUX_PARSED_HEADER,   // Header parsing complete, data may be available.
  WEBP_DEMUX_DONE             // Entire file has been parsed.
} WebPDemuxState;

//------------------------------------------------------------------------------
// Life of a Demux object

// Internal, version-checked, entry point
WEBP_EXTERN(WebPDemuxer*) WebPDemuxInternal(
    const WebPData*, int, WebPDemuxState*, int);

// Parses the WebP file given by 'data'.
// A complete WebP file must be present in 'data' for the function to succeed.
// Returns a WebPDemuxer object on successful parse, NULL otherwise.
static WEBP_INLINE WebPDemuxer* WebPDemux(const WebPData* data) {
  return WebPDemuxInternal(data, 0, NULL, WEBP_DEMUX_ABI_VERSION);
}

// Parses the WebP file given by 'data'.
// If 'state' is non-NULL it will be set to indicate the status of the demuxer.
// Returns a WebPDemuxer object on successful parse, NULL otherwise.
static WEBP_INLINE WebPDemuxer* WebPDemuxPartial(
    const WebPData* data, WebPDemuxState* state) {
  return WebPDemuxInternal(data, 1, state, WEBP_DEMUX_ABI_VERSION);
}

// Frees memory associated with 'dmux'.
WEBP_EXTERN(void) WebPDemuxDelete(WebPDemuxer* dmux);

//------------------------------------------------------------------------------
// Data/information extraction.

typedef enum {
  WEBP_FF_FORMAT_FLAGS,  // Extended format flags present in the 'VP8X' chunk.
  WEBP_FF_CANVAS_WIDTH,
  WEBP_FF_CANVAS_HEIGHT,
  WEBP_FF_LOOP_COUNT
} WebPFormatFeature;

// Get the 'feature' value from the 'dmux'.
// NOTE: values are only valid if WebPDemux() was used or WebPDemuxPartial()
// returned a state > WEBP_DEMUX_PARSING_HEADER.
WEBP_EXTERN(uint32_t) WebPDemuxGetI(
    const WebPDemuxer* dmux, WebPFormatFeature feature);

//------------------------------------------------------------------------------
// Frame iteration.

typedef struct {
  int frame_num_;
  int num_frames_;
  int tile_num_;
  int num_tiles_;
  int x_offset_, y_offset_;  // offset relative to the canvas.
  int width_, height_;       // dimensions of this frame or tile.
  int duration_;   // display duration in milliseconds.
  int complete_;   // true if 'tile_' contains a full frame. partial images may
                   // still be decoded with the WebP incremental decoder.
  WebPData tile_;  // The frame or tile given by 'frame_num_' and 'tile_num_'.

  uint32_t pad[4];           // padding for later use
  void* private_;
} WebPIterator;

// Retrieves frame 'frame_number' from 'dmux'.
// 'iter->tile_' points to the first tile on return from this function.
// Individual tiles may be extracted using WebPDemuxSetTile().
// Setting 'frame_number' equal to 0 will return the last frame of the image.
// Returns false if 'dmux' is NULL or frame 'frame_number' is not present.
// Call WebPDemuxReleaseIterator() when use of the iterator is complete.
// NOTE: 'dmux' must persist for the lifetime of 'iter'.
WEBP_EXTERN(int) WebPDemuxGetFrame(
    const WebPDemuxer* dmux, int frame_number, WebPIterator* iter);

// Sets 'iter->tile_' to point to the next ('iter->frame_num_' + 1) or previous
// ('iter->frame_num_' - 1) frame. These functions do not loop.
// Returns true on success, false otherwise.
WEBP_EXTERN(int) WebPDemuxNextFrame(WebPIterator* iter);
WEBP_EXTERN(int) WebPDemuxPrevFrame(WebPIterator* iter);

// Sets 'iter->tile_' to reflect tile number 'tile_number'.
// Returns true if tile 'tile_number' is present, false otherwise.
WEBP_EXTERN(int) WebPDemuxSelectTile(WebPIterator* iter, int tile_number);

// Releases any memory associated with 'iter'.
// Must be called before destroying the associated WebPDemuxer with
// WebPDemuxDelete().
WEBP_EXTERN(void) WebPDemuxReleaseIterator(WebPIterator* iter);

//------------------------------------------------------------------------------
// Chunk iteration.

typedef struct {
  // The current and total number of chunks with the fourcc given to
  // WebPDemuxGetChunk().
  int chunk_num_;
  int num_chunks_;
  WebPData chunk_;    // The payload of the chunk.

  uint32_t pad[6];    // padding for later use
  void* private_;
} WebPChunkIterator;

// Retrieves the 'chunk_number' instance of the chunk with id 'fourcc' from
// 'dmux'.
// 'fourcc' is a character array containing the fourcc of the chunk to return,
// e.g., "ICCP", "META", "EXIF", etc.
// Setting 'chunk_number' equal to 0 will return the last chunk in a set.
// Returns true if the chunk is found, false otherwise. Image related chunk
// payloads are accessed through WebPDemuxGetFrame() and related functions.
// Call WebPDemuxReleaseChunkIterator() when use of the iterator is complete.
// NOTE: 'dmux' must persist for the lifetime of the iterator.
WEBP_EXTERN(int) WebPDemuxGetChunk(const WebPDemuxer* dmux,
                                   const char fourcc[4], int chunk_number,
                                   WebPChunkIterator* iter);

// Sets 'iter->chunk_' to point to the next ('iter->chunk_num_' + 1) or previous
// ('iter->chunk_num_' - 1) chunk. These functions do not loop.
// Returns true on success, false otherwise.
WEBP_EXTERN(int) WebPDemuxNextChunk(WebPChunkIterator* iter);
WEBP_EXTERN(int) WebPDemuxPrevChunk(WebPChunkIterator* iter);

// Releases any memory associated with 'iter'.
// Must be called before destroying the associated WebPDemuxer with
// WebPDemuxDelete().
WEBP_EXTERN(void) WebPDemuxReleaseChunkIterator(WebPChunkIterator* iter);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_WEBP_MUX_H_ */
