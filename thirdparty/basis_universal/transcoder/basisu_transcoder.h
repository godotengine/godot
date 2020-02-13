// basisu_transcoder.h
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
// Important: If compiling with gcc, be sure strict aliasing is disabled: -fno-strict-aliasing
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

// Set BASISU_DEVEL_MESSAGES to 1 to enable debug printf()'s whenever an error occurs, for easier debugging during development.
//#define BASISU_DEVEL_MESSAGES 1

#include "basisu_transcoder_internal.h"
#include "basisu_global_selector_palette.h"
#include "basisu_file_headers.h"

namespace basist
{
	// High-level composite texture formats supported by the transcoder.
	// Each of these texture formats directly correspond to OpenGL/D3D/Vulkan etc. texture formats.
	// Notes:
	// - If you specify a texture format that supports alpha, but the .basis file doesn't have alpha, the transcoder will automatically output a 
	// fully opaque (255) alpha channel.
	// - The PVRTC1 texture formats only support power of 2 dimension .basis files, but this may be relaxed in a future version.
	// - The PVRTC1 transcoders are real-time encoders, so don't expect the highest quality. We may add a slower encoder with improved quality.
	// - These enums must be kept in sync with Javascript code that calls the transcoder.
	enum class transcoder_texture_format
	{
		// Compressed formats

		// ETC1-2
		cTFETC1_RGB = 0,							// Opaque only, returns RGB or alpha data if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
		cTFETC2_RGBA = 1,							// Opaque+alpha, ETC2_EAC_A8 block followed by a ETC1 block, alpha channel will be opaque for opaque .basis files

		// BC1-5, BC7 (desktop, some mobile devices)
		cTFBC1_RGB = 2,							// Opaque only, no punchthrough alpha support yet, transcodes alpha slice if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
		cTFBC3_RGBA = 3, 							// Opaque+alpha, BC4 followed by a BC1 block, alpha channel will be opaque for opaque .basis files
		cTFBC4_R = 4,								// Red only, alpha slice is transcoded to output if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified
		cTFBC5_RG = 5,								// XY: Two BC4 blocks, X=R and Y=Alpha, .basis file should have alpha data (if not Y will be all 255's)
		cTFBC7_M6_RGB = 6,						// Opaque only, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified. Highest quality of all the non-ETC1 formats.
		cTFBC7_M5_RGBA = 7,						// Opaque+alpha, alpha channel will be opaque for opaque .basis files

		// PVRTC1 4bpp (mobile, PowerVR devices)
		cTFPVRTC1_4_RGB = 8,						// Opaque only, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified, nearly lowest quality of any texture format.
		cTFPVRTC1_4_RGBA = 9,					// Opaque+alpha, most useful for simple opacity maps. If .basis file doens't have alpha cTFPVRTC1_4_RGB will be used instead. Lowest quality of any supported texture format.

		// ASTC (mobile, Intel devices, hopefully all desktop GPU's one day)
		cTFASTC_4x4_RGBA = 10,					// Opaque+alpha, ASTC 4x4, alpha channel will be opaque for opaque .basis files. Transcoder uses RGB/RGBA/L/LA modes, void extent, and up to two ([0,47] and [0,255]) endpoint precisions.

		// ATC (mobile, Adreno devices, this is a niche format)
		cTFATC_RGB = 11,							// Opaque, RGB or alpha if cDecodeFlagsTranscodeAlphaDataToOpaqueFormats flag is specified. ATI ATC (GL_ATC_RGB_AMD)
		cTFATC_RGBA = 12,							// Opaque+alpha, alpha channel will be opaque for opaque .basis files. ATI ATC (GL_ATC_RGBA_INTERPOLATED_ALPHA_AMD) 

		// FXT1 (desktop, Intel devices, this is a super obscure format)
		cTFFXT1_RGB = 17,							// Opaque only, uses exclusively CC_MIXED blocks. Notable for having a 8x4 block size. GL_3DFX_texture_compression_FXT1 is supported on Intel integrated GPU's (such as HD 630).
														// Punch-through alpha is relatively easy to support, but full alpha is harder. This format is only here for completeness so opaque-only is fine for now.
														// See the BASISU_USE_ORIGINAL_3DFX_FXT1_ENCODING macro in basisu_transcoder_internal.h.

		cTFPVRTC2_4_RGB = 18,					// Opaque-only, almost BC1 quality, much faster to transcode and supports arbitrary texture dimensions (unlike PVRTC1 RGB).
		cTFPVRTC2_4_RGBA = 19,					// Opaque+alpha, slower to encode than cTFPVRTC2_4_RGB. Premultiplied alpha is highly recommended, otherwise the color channel can leak into the alpha channel on transparent blocks.

		cTFETC2_EAC_R11 = 20,					// R only (ETC2 EAC R11 unsigned)
		cTFETC2_EAC_RG11 = 21,					// RG only (ETC2 EAC RG11 unsigned), R=opaque.r, G=alpha - for tangent space normal maps
		
		// Uncompressed (raw pixel) formats
		cTFRGBA32 = 13,							// 32bpp RGBA image stored in raster (not block) order in memory, R is first byte, A is last byte.
		cTFRGB565 = 14,							// 166pp RGB image stored in raster (not block) order in memory, R at bit position 11
		cTFBGR565 = 15,							// 16bpp RGB image stored in raster (not block) order in memory, R at bit position 0
		cTFRGBA4444 = 16,							// 16bpp RGBA image stored in raster (not block) order in memory, R at bit position 12, A at bit position 0

		cTFTotalTextureFormats = 22,

		// Old enums for compatibility with code compiled against previous versions
		cTFETC1 = cTFETC1_RGB,
		cTFETC2 = cTFETC2_RGBA,
		cTFBC1 = cTFBC1_RGB,
		cTFBC3 = cTFBC3_RGBA,
		cTFBC4 = cTFBC4_R,
		cTFBC5 = cTFBC5_RG,
		cTFBC7_M6_OPAQUE_ONLY = cTFBC7_M6_RGB,
		cTFBC7_M5 = cTFBC7_M5_RGBA,
		cTFASTC_4x4 = cTFASTC_4x4_RGBA,
		cTFATC_RGBA_INTERPOLATED_ALPHA = cTFATC_RGBA,
	};

	uint32_t basis_get_bytes_per_block(transcoder_texture_format fmt);
	const char* basis_get_format_name(transcoder_texture_format fmt);
	bool basis_transcoder_format_has_alpha(transcoder_texture_format fmt);
	basisu::texture_format basis_get_basisu_texture_format(transcoder_texture_format fmt);
	const char* basis_get_texture_type_name(basis_texture_type tex_type);
	
	bool basis_transcoder_format_is_uncompressed(transcoder_texture_format tex_type);
	uint32_t basis_get_uncompressed_bytes_per_pixel(transcoder_texture_format fmt);
	
	uint32_t basis_get_block_width(transcoder_texture_format tex_type);
	uint32_t basis_get_block_height(transcoder_texture_format tex_type);

	// Returns true if the specified format was enabled at compile time.
	bool basis_is_format_supported(transcoder_texture_format tex_type);
		
	class basisu_transcoder;

	// This struct holds all state used during transcoding. For video, it needs to persist between image transcodes (it holds the previous frame).
	// For threading you can use one state per thread.
	struct basisu_transcoder_state
	{
		struct block_preds
		{
			uint16_t m_endpoint_index;
			uint8_t m_pred_bits;
		};

		std::vector<block_preds> m_block_endpoint_preds[2];
		
		enum { cMaxPrevFrameLevels = 16 };
		std::vector<uint32_t> m_prev_frame_indices[2][cMaxPrevFrameLevels]; // [alpha_flag][level_index] 
	};
	
	// Low-level helper class that does the actual transcoding.
	class basisu_lowlevel_transcoder
	{
		friend class basisu_transcoder;
	
	public:
		basisu_lowlevel_transcoder(const basist::etc1_global_selector_codebook *pGlobal_sel_codebook);

		bool decode_palettes(
			uint32_t num_endpoints, const uint8_t *pEndpoints_data, uint32_t endpoints_data_size,
			uint32_t num_selectors, const uint8_t *pSelectors_data, uint32_t selectors_data_size);

		bool decode_tables(const uint8_t *pTable_data, uint32_t table_data_size);

		bool transcode_slice(void *pDst_blocks, uint32_t num_blocks_x, uint32_t num_blocks_y, const uint8_t *pImage_data, uint32_t image_data_size, block_format fmt, 
			uint32_t output_block_or_pixel_stride_in_bytes, bool bc1_allow_threecolor_blocks, const basis_file_header &header, const basis_slice_desc& slice_desc, uint32_t output_row_pitch_in_blocks_or_pixels = 0,
			basisu_transcoder_state *pState = nullptr, bool astc_transcode_alpha = false, void* pAlpha_blocks = nullptr, uint32_t output_rows_in_pixels = 0);

	private:
		typedef std::vector<endpoint> endpoint_vec;
		endpoint_vec m_endpoints;

		typedef std::vector<selector> selector_vec;
		selector_vec m_selectors;

		const etc1_global_selector_codebook *m_pGlobal_sel_codebook;

		huffman_decoding_table m_endpoint_pred_model, m_delta_endpoint_model, m_selector_model, m_selector_history_buf_rle_model;

		uint32_t m_selector_history_buf_size;
		
		basisu_transcoder_state m_def_state;
	};

	struct basisu_slice_info
	{
		uint32_t m_orig_width;
		uint32_t m_orig_height;

		uint32_t m_width;
		uint32_t m_height;

		uint32_t m_num_blocks_x;
		uint32_t m_num_blocks_y;
		uint32_t m_total_blocks;

		uint32_t m_compressed_size;

		uint32_t m_slice_index;	// the slice index in the .basis file
		uint32_t m_image_index;	// the source image index originally provided to the encoder
		uint32_t m_level_index;	// the mipmap level within this image
		
		uint32_t m_unpacked_slice_crc16;
		
		bool m_alpha_flag;		// true if the slice has alpha data
		bool m_iframe_flag;		// true if the slice is an I-Frame
	};

	typedef std::vector<basisu_slice_info> basisu_slice_info_vec;

	struct basisu_image_info
	{
		uint32_t m_image_index;
		uint32_t m_total_levels;	

		uint32_t m_orig_width;
		uint32_t m_orig_height;

		uint32_t m_width;
		uint32_t m_height;

		uint32_t m_num_blocks_x;
		uint32_t m_num_blocks_y;
		uint32_t m_total_blocks;

		uint32_t m_first_slice_index;	
								
		bool m_alpha_flag;		// true if the image has alpha data
		bool m_iframe_flag;		// true if the image is an I-Frame
	};

	struct basisu_image_level_info
	{
		uint32_t m_image_index;
		uint32_t m_level_index;

		uint32_t m_orig_width;
		uint32_t m_orig_height;

		uint32_t m_width;
		uint32_t m_height;

		uint32_t m_num_blocks_x;
		uint32_t m_num_blocks_y;
		uint32_t m_total_blocks;

		uint32_t m_first_slice_index;	
								
		bool m_alpha_flag;		// true if the image has alpha data
		bool m_iframe_flag;		// true if the image is an I-Frame
	};

	struct basisu_file_info
	{
		uint32_t m_version;
		uint32_t m_total_header_size;

		uint32_t m_total_selectors;
		uint32_t m_selector_codebook_size;

		uint32_t m_total_endpoints;
		uint32_t m_endpoint_codebook_size;

		uint32_t m_tables_size;
		uint32_t m_slices_size;	

		basis_texture_type m_tex_type;
		uint32_t m_us_per_frame;

		// Low-level slice information (1 slice per image for color-only basis files, 2 for alpha basis files)
		basisu_slice_info_vec m_slice_info;

		uint32_t m_total_images;	 // total # of images
		std::vector<uint32_t> m_image_mipmap_levels; // the # of mipmap levels for each image

		uint32_t m_userdata0;
		uint32_t m_userdata1;
		
		bool m_etc1s;					// always true for basis universal
		bool m_y_flipped;				// true if the image was Y flipped
		bool m_has_alpha_slices;	// true if the texture has alpha slices (even slices RGB, odd slices alpha)
	};

	// High-level transcoder class which accepts .basis file data and allows the caller to query information about the file and transcode image levels to various texture formats.
	// If you're just starting out this is the class you care about.
	class basisu_transcoder
	{
		basisu_transcoder(basisu_transcoder&);
		basisu_transcoder& operator= (const basisu_transcoder&);

	public:
		basisu_transcoder(const etc1_global_selector_codebook *pGlobal_sel_codebook);

		// Validates the .basis file. This computes a crc16 over the entire file, so it's slow.
		bool validate_file_checksums(const void *pData, uint32_t data_size, bool full_validation) const;

		// Quick header validation - no crc16 checks.
		bool validate_header(const void *pData, uint32_t data_size) const;

		basis_texture_type get_texture_type(const void *pData, uint32_t data_size) const;
		bool get_userdata(const void *pData, uint32_t data_size, uint32_t &userdata0, uint32_t &userdata1) const;
		
		// Returns the total number of images in the basis file (always 1 or more).
		// Note that the number of mipmap levels for each image may differ, and that images may have different resolutions.
		uint32_t get_total_images(const void *pData, uint32_t data_size) const;

		// Returns the number of mipmap levels in an image.
		uint32_t get_total_image_levels(const void *pData, uint32_t data_size, uint32_t image_index) const;
		
		// Returns basic information about an image. Note that orig_width/orig_height may not be a multiple of 4.
		bool get_image_level_desc(const void *pData, uint32_t data_size, uint32_t image_index, uint32_t level_index, uint32_t &orig_width, uint32_t &orig_height, uint32_t &total_blocks) const;

		// Returns information about the specified image.
		bool get_image_info(const void *pData, uint32_t data_size, basisu_image_info &image_info, uint32_t image_index) const;

		// Returns information about the specified image's mipmap level.
		bool get_image_level_info(const void *pData, uint32_t data_size, basisu_image_level_info &level_info, uint32_t image_index, uint32_t level_index) const;
				
		// Get a description of the basis file and low-level information about each slice.
		bool get_file_info(const void *pData, uint32_t data_size, basisu_file_info &file_info) const;
				
		// start_transcoding() must be called before calling transcode_slice() or transcode_image_level().
		// This decompresses the selector/endpoint codebooks, so ideally you would only call this once per .basis file (not each image/mipmap level).
		bool start_transcoding(const void *pData, uint32_t data_size) const;
		
		// Returns true if start_transcoding() has been called.
		bool get_ready_to_transcode() const { return m_lowlevel_decoder.m_endpoints.size() > 0; }

		enum 
		{
			// PVRTC1: decode non-pow2 ETC1S texture level to the next larger power of 2 (not implemented yet, but we're going to support it). Ignored if the slice's dimensions are already a power of 2.
			cDecodeFlagsPVRTCDecodeToNextPow2 = 2,	
			
			// When decoding to an opaque texture format, if the basis file has alpha, decode the alpha slice instead of the color slice to the output texture format.
			// This is primarily to allow decoding of textures with alpha to multiple ETC1 textures (one for color, another for alpha).
			cDecodeFlagsTranscodeAlphaDataToOpaqueFormats = 4,

			// Forbid usage of BC1 3 color blocks (we don't support BC1 punchthrough alpha yet).
			// This flag is used internally when decoding to BC3.
			cDecodeFlagsBC1ForbidThreeColorBlocks = 8,

			// The output buffer contains alpha endpoint/selector indices. 
			// Used internally when decoding formats like ASTC that require both color and alpha data to be available when transcoding to the output format.
			cDecodeFlagsOutputHasAlphaIndices = 16
		};
								
		// transcode_image_level() decodes a single mipmap level from the .basis file to any of the supported output texture formats.
		// It'll first find the slice(s) to transcode, then call transcode_slice() one or two times to decode both the color and alpha texture data (or RG texture data from two slices for BC5).
		// If the .basis file doesn't have alpha slices, the output alpha blocks will be set to fully opaque (all 255's).
		// Currently, to decode to PVRTC1 the basis texture's dimensions in pixels must be a power of 2, due to PVRTC1 format requirements. 
		// output_blocks_buf_size_in_blocks_or_pixels should be at least the image level's total_blocks (num_blocks_x * num_blocks_y), or the total number of output pixels if fmt==cTFRGBA32.
		// output_row_pitch_in_blocks_or_pixels: Number of blocks or pixels per row. If 0, the transcoder uses the slice's num_blocks_x or orig_width (NOT num_blocks_x * 4). Ignored for PVRTC1 (due to texture swizzling).
		// output_rows_in_pixels: Ignored unless fmt is cRGBA32. The total number of output rows in the output buffer. If 0, the transcoder assumes the slice's orig_height (NOT num_blocks_y * 4).
		// Notes: 
		// - basisu_transcoder_init() must have been called first to initialize the transcoder lookup tables before calling this function.
		// - This method assumes the output texture buffer is readable. In some cases to handle alpha, the transcoder will write temporary data to the output texture in
		// a first pass, which will be read in a second pass.
		bool transcode_image_level(
			const void *pData, uint32_t data_size, 
			uint32_t image_index, uint32_t level_index, 
			void *pOutput_blocks, uint32_t output_blocks_buf_size_in_blocks_or_pixels,
			transcoder_texture_format fmt,
			uint32_t decode_flags = 0, uint32_t output_row_pitch_in_blocks_or_pixels = 0, basisu_transcoder_state *pState = nullptr, uint32_t output_rows_in_pixels = 0) const;

		// Finds the basis slice corresponding to the specified image/level/alpha params, or -1 if the slice can't be found.
		int find_slice(const void *pData, uint32_t data_size, uint32_t image_index, uint32_t level_index, bool alpha_data) const;

		// transcode_slice() decodes a single slice from the .basis file. It's a low-level API - most likely you want to use transcode_image_level().
		// This is a low-level API, and will be needed to be called multiple times to decode some texture formats (like BC3, BC5, or ETC2).
		// output_blocks_buf_size_in_blocks_or_pixels is just used for verification to make sure the output buffer is large enough.
		// output_blocks_buf_size_in_blocks_or_pixels should be at least the image level's total_blocks (num_blocks_x * num_blocks_y), or the total number of output pixels if fmt==cTFRGBA32.
		// output_block_stride_in_bytes: Number of bytes between each output block.
		// output_row_pitch_in_blocks_or_pixels: Number of blocks or pixels per row. If 0, the transcoder uses the slice's num_blocks_x or orig_width (NOT num_blocks_x * 4). Ignored for PVRTC1 (due to texture swizzling).
		// output_rows_in_pixels: Ignored unless fmt is cRGBA32. The total number of output rows in the output buffer. If 0, the transcoder assumes the slice's orig_height (NOT num_blocks_y * 4).
		// Notes:
		// - basisu_transcoder_init() must have been called first to initialize the transcoder lookup tables before calling this function.
		bool transcode_slice(const void *pData, uint32_t data_size, uint32_t slice_index, 
			void *pOutput_blocks, uint32_t output_blocks_buf_size_in_blocks_or_pixels,
			block_format fmt, uint32_t output_block_stride_in_bytes, uint32_t decode_flags = 0, uint32_t output_row_pitch_in_blocks_or_pixels = 0, basisu_transcoder_state * pState = nullptr, void* pAlpha_blocks = nullptr, uint32_t output_rows_in_pixels = 0) const;

	private:
		mutable basisu_lowlevel_transcoder m_lowlevel_decoder;

		int find_first_slice_index(const void* pData, uint32_t data_size, uint32_t image_index, uint32_t level_index) const;
		
		bool validate_header_quick(const void* pData, uint32_t data_size) const;
	};

	// basisu_transcoder_init() must be called before a .basis file can be transcoded.
	void basisu_transcoder_init();

	enum debug_flags_t
	{
		cDebugFlagVisCRs = 1,
		cDebugFlagVisBC1Sels = 2,
		cDebugFlagVisBC1Endpoints = 4
	};
	uint32_t get_debug_flags();
	void set_debug_flags(uint32_t f);

} // namespace basisu
