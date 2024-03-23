// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2020-2024 Arm Limited
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
// ----------------------------------------------------------------------------

/**
 * @brief The core astcenc codec library interface.
 *
 * This interface is the entry point to the core astcenc codec. It aims to be easy to use for
 * non-experts, but also to allow experts to have fine control over the compressor heuristics if
 * needed. The core codec only handles compression and decompression, transferring all inputs and
 * outputs via memory buffers. To catch obvious input/output buffer sizing issues, which can cause
 * security and stability problems, all transfer buffers are explicitly sized.
 *
 * While the aim is that we keep this interface mostly stable, it should be viewed as a mutable
 * interface tied to a specific source version. We are not trying to maintain backwards
 * compatibility across codec versions.
 *
 * The API state management is based around an explicit context object, which is the context for all
 * allocated memory resources needed to compress and decompress a single image. A context can be
 * used to sequentially compress multiple images using the same configuration, allowing setup
 * overheads to be amortized over multiple images, which is particularly important when images are
 * small.
 *
 * Multi-threading can be used two ways.
 *
 *     * An application wishing to process multiple images in parallel can allocate multiple
 *       contexts and assign each context to a thread.
 *     * An application wishing to process a single image in using multiple threads can configure
 *       contexts for multi-threaded use, and invoke astcenc_compress/decompress() once per thread
 *       for faster processing. The caller is responsible for creating the worker threads, and
 *       synchronizing between images.
 *
 * Extended instruction set support
 * ================================
 *
 * This library supports use of extended instruction sets, such as SSE4.1 and AVX2. These are
 * enabled at compile time when building the library. There is no runtime checking in the core
 * library that the instruction sets used are actually available. Checking compatibility is the
 * responsibility of the calling code.
 *
 * Threading
 * =========
 *
 * In pseudo-code, the usage for manual user threading looks like this:
 *
 *     // Configure the compressor run
 *     astcenc_config my_config;
 *     astcenc_config_init(..., &my_config);
 *
 *     // Power users can tweak <my_config> settings here ...
 *
 *     // Allocate working state given config and thread_count
 *     astcenc_context* my_context;
 *     astcenc_context_alloc(&my_config, thread_count, &my_context);
 *
 *     // Compress each image using these config settings
 *     foreach image:
 *         // For each thread in the thread pool
 *         for i in range(0, thread_count):
 *             astcenc_compress_image(my_context, &my_input, my_output, i);
 *
 *         astcenc_compress_reset(my_context);
 *
 *     // Clean up
 *     astcenc_context_free(my_context);
 *
 * Images
 * ======
 *
 * The codec supports compressing single images, which can be either 2D images or volumetric 3D
 * images. Calling code is responsible for any handling of aggregate types, such as mipmap chains,
 * texture arrays, or sliced 3D textures.
 *
 * Images are passed in as an astcenc_image structure. Inputs can be either 8-bit unorm, 16-bit
 * half-float, or 32-bit float, as indicated by the data_type field.
 *
 * Images can be any dimension; there is no requirement to be a multiple of the ASTC block size.
 *
 * Data is always passed in as 4 color components, and accessed as an array of 2D image slices. Data
 * within an image slice is always tightly packed without padding. Addressing looks like this:
 *
 *     data[z_coord][y_coord * x_dim * 4 + x_coord * 4    ]   // Red
 *     data[z_coord][y_coord * x_dim * 4 + x_coord * 4 + 1]   // Green
 *     data[z_coord][y_coord * x_dim * 4 + x_coord * 4 + 2]   // Blue
 *     data[z_coord][y_coord * x_dim * 4 + x_coord * 4 + 3]   // Alpha
 *
 * Common compressor usage
 * =======================
 *
 * One of the most important things for coding image quality is to align the input data component
 * count with the ASTC color endpoint mode. This avoids wasting bits encoding components you don't
 * actually need in the endpoint colors.
 *
 *         | Input data   | Encoding swizzle | Sampling swizzle |
 *         | ------------ | ---------------- | ---------------- |
 *         | 1 component  | RRR1             | .[rgb]           |
 *         | 2 components | RRRG             | .[rgb]a          |
 *         | 3 components | RGB1             | .rgb             |
 *         | 4 components | RGBA             | .rgba            |
 *
 * The 1 and 2 component modes recommend sampling from "g" to recover the luminance value as this
 * provide best compatibility with other texture formats where the green component may be stored at
 * higher precision than the others, such as RGB565. For ASTC any of the RGB components can be used;
 * the luminance endpoint component will be returned for all three.
 *
 * When using the normal map compression mode ASTC will store normals as a two component X+Y map.
 * Input images must contain unit-length normalized and should be passed in using a two component
 * swizzle. The astcenc command line tool defaults to an RRRG swizzle, but some developers prefer
 * to use GGGR for compatability with BC5n which will work just as well. The Z component can be
 * recovered programmatically in shader code, using knowledge that the vector is unit length and
 * that Z must be positive for a tangent-space normal map.
 *
 * Decompress-only usage
 * =====================
 *
 * For some use cases it is useful to have a cut-down context and/or library which supports
 * decompression but not compression.
 *
 * A context can be made decompress-only using the ASTCENC_FLG_DECOMPRESS_ONLY flag when the context
 * is allocated. These contexts have lower dynamic memory footprint than a full context.
 *
 * The entire library can be made decompress-only by building the files with the define
 * ASTCENC_DECOMPRESS_ONLY set. In this build the context will be smaller, and the library will
 * exclude the functionality which is only needed for compression. This reduces the binary size by
 * ~180KB. For these builds contexts must be created with the ASTCENC_FLG_DECOMPRESS_ONLY flag.
 *
 * Note that context structures returned by a library built as decompress-only are incompatible with
 * a library built with compression included, and visa versa, as they have different sizes and
 * memory layout.
 *
 * Self-decompress-only usage
 * ==========================
 *
 * ASTC is a complex format with a large search space. The parts of this search space that are
 * searched is determined by heuristics that are, in part, tied to the quality level used when
 * creating the context.
 *
 * A normal context is capable of decompressing any ASTC texture, including those generated by other
 * compressors with unknown heuristics. This is the most flexible implementation, but forces the
 * data tables used by the codec to include entries that are not needed during compression. This
 * can slow down context creation by a significant amount, especially for the faster compression
 * modes where few data table entries are actually used. To optimize this use case the context can
 * be created with the ASTCENC_FLG_SELF_DECOMPRESS_ONLY flag. This tells the compressor that it will
 * only be asked to decompress images that it compressed itself, allowing the data tables to
 * exclude entries that are not needed by the current compression configuration. This reduces the
 * size of the context data tables in memory and improves context creation performance. Note that,
 * as of the 3.6 release, this flag no longer affects compression performance.
 *
 * Using this flag while attempting to decompress an valid image which was created by another
 * compressor, or even another astcenc compressor version or configuration, may result in blocks
 * returning as solid magenta or NaN value error blocks.
 */

#ifndef ASTCENC_INCLUDED
#define ASTCENC_INCLUDED

#include <cstddef>
#include <cstdint>

#if defined(ASTCENC_DYNAMIC_LIBRARY)
	#if defined(_MSC_VER)
		#define ASTCENC_PUBLIC extern "C" __declspec(dllexport)
	#else
		#define ASTCENC_PUBLIC extern "C" __attribute__ ((visibility ("default")))
	#endif
#else
	#define ASTCENC_PUBLIC
#endif

/* ============================================================================
    Data declarations
============================================================================ */

/**
 * @brief An opaque structure; see astcenc_internal.h for definition.
 */
struct astcenc_context;

/**
 * @brief A codec API error code.
 */
enum astcenc_error {
	/** @brief The call was successful. */
	ASTCENC_SUCCESS = 0,
	/** @brief The call failed due to low memory, or undersized I/O buffers. */
	ASTCENC_ERR_OUT_OF_MEM,
	/** @brief The call failed due to the build using fast math. */
	ASTCENC_ERR_BAD_CPU_FLOAT,
	/** @brief The call failed due to an out-of-spec parameter. */
	ASTCENC_ERR_BAD_PARAM,
	/** @brief The call failed due to an out-of-spec block size. */
	ASTCENC_ERR_BAD_BLOCK_SIZE,
	/** @brief The call failed due to an out-of-spec color profile. */
	ASTCENC_ERR_BAD_PROFILE,
	/** @brief The call failed due to an out-of-spec quality value. */
	ASTCENC_ERR_BAD_QUALITY,
	/** @brief The call failed due to an out-of-spec component swizzle. */
	ASTCENC_ERR_BAD_SWIZZLE,
	/** @brief The call failed due to an out-of-spec flag set. */
	ASTCENC_ERR_BAD_FLAGS,
	/** @brief The call failed due to the context not supporting the operation. */
	ASTCENC_ERR_BAD_CONTEXT,
	/** @brief The call failed due to unimplemented functionality. */
	ASTCENC_ERR_NOT_IMPLEMENTED,
	/** @brief The call failed due to an out-of-spec decode mode flag set. */
	ASTCENC_ERR_BAD_DECODE_MODE,
#if defined(ASTCENC_DIAGNOSTICS)
	/** @brief The call failed due to an issue with diagnostic tracing. */
	ASTCENC_ERR_DTRACE_FAILURE,
#endif
};

/**
 * @brief A codec color profile.
 */
enum astcenc_profile {
	/** @brief The LDR sRGB color profile. */
	ASTCENC_PRF_LDR_SRGB = 0,
	/** @brief The LDR linear color profile. */
	ASTCENC_PRF_LDR,
	/** @brief The HDR RGB with LDR alpha color profile. */
	ASTCENC_PRF_HDR_RGB_LDR_A,
	/** @brief The HDR RGBA color profile. */
	ASTCENC_PRF_HDR
};

/** @brief The fastest, lowest quality, search preset. */
static const float ASTCENC_PRE_FASTEST = 0.0f;

/** @brief The fast search preset. */
static const float ASTCENC_PRE_FAST = 10.0f;

/** @brief The medium quality search preset. */
static const float ASTCENC_PRE_MEDIUM = 60.0f;

/** @brief The thorough quality search preset. */
static const float ASTCENC_PRE_THOROUGH = 98.0f;

/** @brief The thorough quality search preset. */
static const float ASTCENC_PRE_VERYTHOROUGH = 99.0f;

/** @brief The exhaustive, highest quality, search preset. */
static const float ASTCENC_PRE_EXHAUSTIVE = 100.0f;

/**
 * @brief A codec component swizzle selector.
 */
enum astcenc_swz
{
	/** @brief Select the red component. */
	ASTCENC_SWZ_R = 0,
	/** @brief Select the green component. */
	ASTCENC_SWZ_G = 1,
	/** @brief Select the blue component. */
	ASTCENC_SWZ_B = 2,
	/** @brief Select the alpha component. */
	ASTCENC_SWZ_A = 3,
	/** @brief Use a constant zero component. */
	ASTCENC_SWZ_0 = 4,
	/** @brief Use a constant one component. */
	ASTCENC_SWZ_1 = 5,
	/** @brief Use a reconstructed normal vector Z component. */
	ASTCENC_SWZ_Z = 6
};

/**
 * @brief A texel component swizzle.
 */
struct astcenc_swizzle
{
	/** @brief The red component selector. */
	astcenc_swz r;
	/** @brief The green component selector. */
	astcenc_swz g;
	/** @brief The blue component selector. */
	astcenc_swz b;
	/** @brief The alpha component selector. */
	astcenc_swz a;
};

/**
 * @brief A texel component data format.
 */
enum astcenc_type
{
	/** @brief Unorm 8-bit data per component. */
	ASTCENC_TYPE_U8 = 0,
	/** @brief 16-bit float per component. */
	ASTCENC_TYPE_F16 = 1,
	/** @brief 32-bit float per component. */
	ASTCENC_TYPE_F32 = 2
};

/**
 * @brief Function pointer type for compression progress reporting callback.
 */
extern "C" typedef void (*astcenc_progress_callback)(float);

/**
 * @brief Enable normal map compression.
 *
 * Input data will be treated a two component normal map, storing X and Y, and the codec will
 * optimize for angular error rather than simple linear PSNR. In this mode the input swizzle should
 * be e.g. rrrg (the default ordering for ASTC normals on the command line) or gggr (the ordering
 * used by BC5n).
 */
static const unsigned int ASTCENC_FLG_MAP_NORMAL          = 1 << 0;

/**
 * @brief Enable compression heuristics that assume use of decode_unorm8 decode mode.
 *
 * The decode_unorm8 decode mode rounds differently to the decode_fp16 decode mode, so enabling this
 * flag during compression will allow the compressor to use the correct rounding when selecting
 * encodings. This will improve the compressed image quality if your application is using the
 * decode_unorm8 decode mode, but will reduce image quality if using decode_fp16.
 *
 * Note that LDR_SRGB images will always use decode_unorm8 for the RGB channels, irrespective of
 * this setting.
 */
static const unsigned int ASTCENC_FLG_USE_DECODE_UNORM8        = 1 << 1;

/**
 * @brief Enable alpha weighting.
 *
 * The input alpha value is used for transparency, so errors in the RGB components are weighted by
 * the transparency level. This allows the codec to more accurately encode the alpha value in areas
 * where the color value is less significant.
 */
static const unsigned int ASTCENC_FLG_USE_ALPHA_WEIGHT     = 1 << 2;

/**
 * @brief Enable perceptual error metrics.
 *
 * This mode enables perceptual compression mode, which will optimize for perceptual error rather
 * than best PSNR. Only some input modes support perceptual error metrics.
 */
static const unsigned int ASTCENC_FLG_USE_PERCEPTUAL       = 1 << 3;

/**
 * @brief Create a decompression-only context.
 *
 * This mode disables support for compression. This enables context allocation to skip some
 * transient buffer allocation, resulting in lower memory usage.
 */
static const unsigned int ASTCENC_FLG_DECOMPRESS_ONLY      = 1 << 4;

/**
 * @brief Create a self-decompression context.
 *
 * This mode configures the compressor so that it is only guaranteed to be able to decompress images
 * that were actually created using the current context. This is the common case for compression use
 * cases, and setting this flag enables additional optimizations, but does mean that the context
 * cannot reliably decompress arbitrary ASTC images.
 */
static const unsigned int ASTCENC_FLG_SELF_DECOMPRESS_ONLY = 1 << 5;

/**
 * @brief Enable RGBM map compression.
 *
 * Input data will be treated as HDR data that has been stored in an LDR RGBM-encoded wrapper
 * format. Data must be preprocessed by the user to be in LDR RGBM format before calling the
 * compression function, this flag is only used to control the use of RGBM-specific heuristics and
 * error metrics.
 *
 * IMPORTANT: The ASTC format is prone to bad failure modes with unconstrained RGBM data; very small
 * M values can round to zero due to quantization and result in black or white pixels. It is highly
 * recommended that the minimum value of M used in the encoding is kept above a lower threshold (try
 * 16 or 32). Applying this threshold reduces the number of very dark colors that can be
 * represented, but is still higher precision than 8-bit LDR.
 *
 * When this flag is set the value of @c rgbm_m_scale in the context must be set to the RGBM scale
 * factor used during reconstruction. This defaults to 5 when in RGBM mode.
 *
 * It is recommended that the value of @c cw_a_weight is set to twice the value of the multiplier
 * scale, ensuring that the M value is accurately encoded. This defaults to 10 when in RGBM mode,
 * matching the default scale factor.
 */
static const unsigned int ASTCENC_FLG_MAP_RGBM             = 1 << 6;

/**
 * @brief The bit mask of all valid flags.
 */
static const unsigned int ASTCENC_ALL_FLAGS =
                              ASTCENC_FLG_MAP_NORMAL |
                              ASTCENC_FLG_MAP_RGBM |
                              ASTCENC_FLG_USE_ALPHA_WEIGHT |
                              ASTCENC_FLG_USE_PERCEPTUAL |
                              ASTCENC_FLG_USE_DECODE_UNORM8 |
                              ASTCENC_FLG_DECOMPRESS_ONLY |
                              ASTCENC_FLG_SELF_DECOMPRESS_ONLY;

/**
 * @brief The config structure.
 *
 * This structure will initially be populated by a call to astcenc_config_init, but power users may
 * modify it before calling astcenc_context_alloc. See astcenccli_toplevel_help.cpp for full user
 * documentation of the power-user settings.
 *
 * Note for any settings which are associated with a specific color component, the value in the
 * config applies to the component that exists after any compression data swizzle is applied.
 */
struct astcenc_config
{
	/** @brief The color profile. */
	astcenc_profile profile;

	/** @brief The set of set flags. */
	unsigned int flags;

	/** @brief The ASTC block size X dimension. */
	unsigned int block_x;

	/** @brief The ASTC block size Y dimension. */
	unsigned int block_y;

	/** @brief The ASTC block size Z dimension. */
	unsigned int block_z;

	/** @brief The red component weight scale for error weighting (-cw). */
	float cw_r_weight;

	/** @brief The green component weight scale for error weighting (-cw). */
	float cw_g_weight;

	/** @brief The blue component weight scale for error weighting (-cw). */
	float cw_b_weight;

	/** @brief The alpha component weight scale for error weighting (-cw). */
	float cw_a_weight;

	/**
	 * @brief The radius for any alpha-weight scaling (-a).
	 *
	 * It is recommended that this is set to 1 when using FLG_USE_ALPHA_WEIGHT on a texture that
	 * will be sampled using linear texture filtering to minimize color bleed out of transparent
	 * texels that are adjacent to non-transparent texels.
	 */
	unsigned int a_scale_radius;

	/** @brief The RGBM scale factor for the shared multiplier (-rgbm). */
	float rgbm_m_scale;

	/**
	 * @brief The maximum number of partitions searched (-partitioncountlimit).
	 *
	 * Valid values are between 1 and 4.
	 */
	unsigned int tune_partition_count_limit;

	/**
	 * @brief The maximum number of partitions searched (-2partitionindexlimit).
	 *
	 * Valid values are between 1 and 1024.
	 */
	unsigned int tune_2partition_index_limit;

	/**
	 * @brief The maximum number of partitions searched (-3partitionindexlimit).
	 *
	 * Valid values are between 1 and 1024.
	 */
	unsigned int tune_3partition_index_limit;

	/**
	 * @brief The maximum number of partitions searched (-4partitionindexlimit).
	 *
	 * Valid values are between 1 and 1024.
	 */
	unsigned int tune_4partition_index_limit;

	/**
	 * @brief The maximum centile for block modes searched (-blockmodelimit).
	 *
	 * Valid values are between 1 and 100.
	 */
	unsigned int tune_block_mode_limit;

	/**
	 * @brief The maximum iterative refinements applied (-refinementlimit).
	 *
	 * Valid values are between 1 and N; there is no technical upper limit
	 * but little benefit is expected after N=4.
	 */
	unsigned int tune_refinement_limit;

	/**
	 * @brief The number of trial candidates per mode search (-candidatelimit).
	 *
	 * Valid values are between 1 and TUNE_MAX_TRIAL_CANDIDATES.
	 */
	unsigned int tune_candidate_limit;

	/**
	 * @brief The number of trial partitionings per search (-2partitioncandidatelimit).
	 *
	 * Valid values are between 1 and TUNE_MAX_PARTITIONING_CANDIDATES.
	 */
	unsigned int tune_2partitioning_candidate_limit;

	/**
	 * @brief The number of trial partitionings per search (-3partitioncandidatelimit).
	 *
	 * Valid values are between 1 and TUNE_MAX_PARTITIONING_CANDIDATES.
	 */
	unsigned int tune_3partitioning_candidate_limit;

	/**
	 * @brief The number of trial partitionings per search (-4partitioncandidatelimit).
	 *
	 * Valid values are between 1 and TUNE_MAX_PARTITIONING_CANDIDATES.
	 */
	unsigned int tune_4partitioning_candidate_limit;

	/**
	 * @brief The dB threshold for stopping block search (-dblimit).
	 *
	 * This option is ineffective for HDR textures.
	 */
	float tune_db_limit;

	/**
	 * @brief The amount of MSE overshoot needed to early-out trials.
	 *
	 * The first early-out is for 1 partition, 1 plane trials, where we try a minimal encode using
	 * the high probability block modes. This can short-cut compression for simple blocks.
	 *
	 * The second early-out is for refinement trials, where we can exit refinement once quality is
	 * reached.
	 */
	float tune_mse_overshoot;

	/**
	 * @brief The threshold for skipping 3.1/4.1 trials (-2partitionlimitfactor).
	 *
	 * This option is further scaled for normal maps, so it skips less often.
	 */
	float tune_2partition_early_out_limit_factor;

	/**
	 * @brief The threshold for skipping 4.1 trials (-3partitionlimitfactor).
	 *
	 * This option is further scaled for normal maps, so it skips less often.
	 */
	float tune_3partition_early_out_limit_factor;

	/**
	 * @brief The threshold for skipping two weight planes (-2planelimitcorrelation).
	 *
	 * This option is ineffective for normal maps.
	 */
	float tune_2plane_early_out_limit_correlation;

	/**
	 * @brief The config enable for the mode0 fast-path search.
	 *
	 * If this is set to TUNE_MIN_TEXELS_MODE0 or higher then the early-out fast mode0
	 * search is enabled. This option is ineffective for 3D block sizes.
	 */
	float tune_search_mode0_enable;

	/**
	 * @brief The progress callback, can be @c nullptr.
	 *
	 * If this is specified the codec will peridocially report progress for
	 * compression as a percentage between 0 and 100. The callback is called from one
	 * of the compressor threads, so doing significant work in the callback will
	 * reduce compression performance.
	 */
	astcenc_progress_callback progress_callback;

#if defined(ASTCENC_DIAGNOSTICS)
	/**
	 * @brief The path to save the diagnostic trace data to.
	 *
	 * This option is not part of the public API, and requires special builds
	 * of the library.
	 */
	const char* trace_file_path;
#endif
};

/**
 * @brief An uncompressed 2D or 3D image.
 *
 * 3D image are passed in as an array of 2D slices. Each slice has identical
 * size and color format.
 */
struct astcenc_image
{
	/** @brief The X dimension of the image, in texels. */
	unsigned int dim_x;

	/** @brief The Y dimension of the image, in texels. */
	unsigned int dim_y;

	/** @brief The Z dimension of the image, in texels. */
	unsigned int dim_z;

	/** @brief The data type per component. */
	astcenc_type data_type;

	/** @brief The array of 2D slices, of length @c dim_z. */
	void** data;
};

/**
 * @brief A block encoding metadata query result.
 *
 * If the block is an error block or a constant color block or an error block all fields other than
 * the profile, block dimensions, and error/constant indicator will be zero.
 */
struct astcenc_block_info
{
	/** @brief The block encoding color profile. */
	astcenc_profile profile;

	/** @brief The number of texels in the X dimension. */
	unsigned int block_x;

	/** @brief The number of texels in the Y dimension. */
	unsigned int block_y;

	/** @brief The number of texel in the Z dimension. */
	unsigned int block_z;

	/** @brief The number of texels in the block. */
	unsigned int texel_count;

	/** @brief True if this block is an error block. */
	bool is_error_block;

	/** @brief True if this block is a constant color block. */
	bool is_constant_block;

	/** @brief True if this block is an HDR block. */
	bool is_hdr_block;

	/** @brief True if this block uses two weight planes. */
	bool is_dual_plane_block;

	/** @brief The number of partitions if not constant color. */
	unsigned int partition_count;

	/** @brief The partition index if 2 - 4 partitions used. */
	unsigned int partition_index;

	/** @brief The component index of the second plane if dual plane. */
	unsigned int dual_plane_component;

	/** @brief The color endpoint encoding mode for each partition. */
	unsigned int color_endpoint_modes[4];

	/** @brief The number of color endpoint quantization levels. */
	unsigned int color_level_count;

	/** @brief The number of weight quantization levels. */
	unsigned int weight_level_count;

	/** @brief The number of weights in the X dimension. */
	unsigned int weight_x;

	/** @brief The number of weights in the Y dimension. */
	unsigned int weight_y;

	/** @brief The number of weights in the Z dimension. */
	unsigned int weight_z;

	/** @brief The unpacked color endpoints for each partition. */
	float color_endpoints[4][2][4];

	/** @brief The per-texel interpolation weights for the block. */
	float weight_values_plane1[216];

	/** @brief The per-texel interpolation weights for the block. */
	float weight_values_plane2[216];

	/** @brief The per-texel partition assignments for the block. */
	uint8_t partition_assignment[216];
};

/**
 * Populate a codec config based on default settings.
 *
 * Power users can edit the returned config struct to fine tune before allocating the context.
 *
 * @param      profile   Color profile.
 * @param      block_x   ASTC block size X dimension.
 * @param      block_y   ASTC block size Y dimension.
 * @param      block_z   ASTC block size Z dimension.
 * @param      quality   Search quality preset / effort level. Either an
 *                       @c ASTCENC_PRE_* value, or a effort level between 0
 *                       and 100. Performance is not linear between 0 and 100.

 * @param      flags     A valid set of @c ASTCENC_FLG_* flag bits.
 * @param[out] config    Output config struct to populate.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if the inputs are invalid
 * either individually, or in combination.
 */
ASTCENC_PUBLIC astcenc_error astcenc_config_init(
	astcenc_profile profile,
	unsigned int block_x,
	unsigned int block_y,
	unsigned int block_z,
	float quality,
	unsigned int flags,
	astcenc_config* config);

/**
 * @brief Allocate a new codec context based on a config.
 *
 * This function allocates all of the memory resources and threads needed by the codec. This can be
 * slow, so it is recommended that contexts are reused to serially compress or decompress multiple
 * images to amortize setup cost.
 *
 * Contexts can be allocated to support only decompression using the @c ASTCENC_FLG_DECOMPRESS_ONLY
 * flag when creating the configuration. The compression functions will fail if invoked. For a
 * decompress-only library build the @c ASTCENC_FLG_DECOMPRESS_ONLY flag must be set when creating
 * any context.
 *
 * @param[in]  config         Codec config.
 * @param      thread_count   Thread count to configure for.
 * @param[out] context        Location to store an opaque context pointer.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if context creation failed.
 */
ASTCENC_PUBLIC astcenc_error astcenc_context_alloc(
	const astcenc_config* config,
	unsigned int thread_count,
	astcenc_context** context);

/**
 * @brief Compress an image.
 *
 * A single context can only compress or decompress a single image at a time.
 *
 * For a context configured for multi-threading, any set of the N threads can call this function.
 * Work will be dynamically scheduled across the threads available. Each thread must have a unique
 * @c thread_index.
 *
 * @param         context        Codec context.
 * @param[in,out] image          An input image, in 2D slices.
 * @param         swizzle        Compression data swizzle, applied before compression.
 * @param[out]    data_out       Pointer to output data array.
 * @param         data_len       Length of the output data array.
 * @param         thread_index   Thread index [0..N-1] of calling thread.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if compression failed.
 */
ASTCENC_PUBLIC astcenc_error astcenc_compress_image(
	astcenc_context* context,
	astcenc_image* image,
	const astcenc_swizzle* swizzle,
	uint8_t* data_out,
	size_t data_len,
	unsigned int thread_index);

/**
 * @brief Reset the codec state for a new compression.
 *
 * The caller is responsible for synchronizing threads in the worker thread pool. This function must
 * only be called when all threads have exited the @c astcenc_compress_image() function for image N,
 * but before any thread enters it for image N + 1.
 *
 * Calling this is not required (but won't hurt), if the context is created for single threaded use.
 *
 * @param context   Codec context.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if reset failed.
 */
ASTCENC_PUBLIC astcenc_error astcenc_compress_reset(
	astcenc_context* context);

/**
 * @brief Decompress an image.
 *
 * @param         context        Codec context.
 * @param[in]     data           Pointer to compressed data.
 * @param         data_len       Length of the compressed data, in bytes.
 * @param[in,out] image_out      Output image.
 * @param         swizzle        Decompression data swizzle, applied after decompression.
 * @param         thread_index   Thread index [0..N-1] of calling thread.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if decompression failed.
 */
ASTCENC_PUBLIC astcenc_error astcenc_decompress_image(
	astcenc_context* context,
	const uint8_t* data,
	size_t data_len,
	astcenc_image* image_out,
	const astcenc_swizzle* swizzle,
	unsigned int thread_index);

/**
 * @brief Reset the codec state for a new decompression.
 *
 * The caller is responsible for synchronizing threads in the worker thread pool. This function must
 * only be called when all threads have exited the @c astcenc_decompress_image() function for image
 * N, but before any thread enters it for image N + 1.
 *
 * Calling this is not required (but won't hurt), if the context is created for single threaded use.
 *
 * @param context   Codec context.
 *
 * @return @c ASTCENC_SUCCESS on success, or an error if reset failed.
 */
ASTCENC_PUBLIC astcenc_error astcenc_decompress_reset(
	astcenc_context* context);

/**
 * Free the compressor context.
 *
 * @param context   The codec context.
 */
ASTCENC_PUBLIC void astcenc_context_free(
	astcenc_context* context);

/**
 * @brief Provide a high level summary of a block's encoding.
 *
 * This feature is primarily useful for codec developers but may be useful for developers building
 * advanced content packaging pipelines.
 *
 * @param context   Codec context.
 * @param data      One block of compressed ASTC data.
 * @param info      The output info structure to populate.
 *
 * @return @c ASTCENC_SUCCESS if the block was decoded, or an error otherwise. Note that this
 *         function will return success even if the block itself was an error block encoding, as the
 *         decode was correctly handled.
 */
ASTCENC_PUBLIC astcenc_error astcenc_get_block_info(
	astcenc_context* context,
	const uint8_t data[16],
	astcenc_block_info* info);

/**
 * @brief Get a printable string for specific status code.
 *
 * @param status   The status value.
 *
 * @return A human readable nul-terminated string.
 */
ASTCENC_PUBLIC const char* astcenc_get_error_string(
	astcenc_error status);

#endif
