// File: bc7enc.h - Richard Geldreich, Jr. - MIT license or public domain (see end of bc7enc.c)
// If you use this software in a product, attribution / credits is requested but not required.
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BC7ENC_BLOCK_SIZE (16)
#define BC7ENC_MAX_PARTITIONS (64)
#define BC7ENC_MAX_UBER_LEVEL (4)

struct color_rgba { uint8_t m_c[4]; };

struct bc7enc_compress_block_params
{
	uint32_t m_mode_mask;

	// m_max_partitions may range from 0 (disables mode 1) to BC7ENC_MAX_PARTITIONS. The higher this value, the slower the compressor, but the higher the quality.
	uint32_t m_max_partitions;
	
	// Relative RGBA or YCbCrA weights.
	uint32_t m_weights[4];
	
	// m_uber_level may range from 0 to BC7ENC_MAX_UBER_LEVEL. The higher this value, the slower the compressor, but the higher the quality.
	uint32_t m_uber_level;

	// If m_perceptual is true, colorspace error is computed in YCbCr space, otherwise RGB.
	bool m_perceptual;

	// Set m_try_least_squares to false for slightly faster/lower quality compression.
	bool m_try_least_squares;
	
	// When m_mode17_partition_estimation_filterbank, the mode1 partition estimator skips lesser used partition patterns unless they are strongly predicted to be potentially useful.
	// There's a slight loss in quality with this enabled (around .08 dB RGB PSNR or .05 dB Y PSNR), but up to a 11% gain in speed depending on the other settings.
	bool m_mode17_partition_estimation_filterbank;

	bool m_force_alpha;

	bool m_force_selectors;
	uint8_t m_selectors[16];

	bool m_quant_mode6_endpoints;
	bool m_bias_mode1_pbits;

	float m_pbit1_weight;

	float m_mode1_error_weight;
	float m_mode5_error_weight;
	float m_mode6_error_weight;
	float m_mode7_error_weight;

	float m_low_frequency_partition_weight;

	void clear()
	{
		memset(this, 0, sizeof(*this));
	}

	void print()
	{
		printf("Mode mask: 0x%X\n", m_mode_mask);
		printf("Max partitions: %u\n", m_max_partitions);
		printf("Weights: %u %u %u %u\n", m_weights[0], m_weights[1], m_weights[2], m_weights[3]);
		printf("Uber level: %u\n", m_uber_level);
		printf("Perceptual: %u\n", m_perceptual);
		printf("Try least squares: %u\n", m_try_least_squares);
		printf("Mode 1/7 partition estimation filterbank: %u\n", m_mode17_partition_estimation_filterbank);
		printf("Force alpha: %u\n", m_force_alpha);
		printf("Quant mode 6 endpoints: %u\n", m_quant_mode6_endpoints);
		printf("Bias mode 1 p-bits: %u\n", m_bias_mode1_pbits);
		printf("p-bit 1 weight: %f\n", m_pbit1_weight);
		printf("Mode error weights: %f %f %f %f\n", m_mode1_error_weight, m_mode5_error_weight, m_mode6_error_weight, m_mode7_error_weight);
		printf("Low frequency partition weight: %f\n", m_low_frequency_partition_weight);
	}
};

inline void bc7enc_compress_block_params_init_linear_weights(bc7enc_compress_block_params *p)
{
	p->m_perceptual = false;
	p->m_weights[0] = 1;
	p->m_weights[1] = 1;
	p->m_weights[2] = 1;
	p->m_weights[3] = 1;
}

inline void bc7enc_compress_block_params_init_perceptual_weights(bc7enc_compress_block_params *p)
{
	p->m_perceptual = true;
	p->m_weights[0] = 128;
	p->m_weights[1] = 64;
	p->m_weights[2] = 16;
	p->m_weights[3] = 32;
}

inline void bc7enc_compress_block_params_init(bc7enc_compress_block_params *p)
{
	p->m_mode_mask = UINT32_MAX;
	p->m_max_partitions = BC7ENC_MAX_PARTITIONS;
	p->m_try_least_squares = true;
	p->m_mode17_partition_estimation_filterbank = true;
	p->m_uber_level = 0;
	p->m_force_selectors = false;
	p->m_force_alpha = false;
	p->m_quant_mode6_endpoints = false;
	p->m_bias_mode1_pbits = false;
	p->m_pbit1_weight = 1.0f;
	p->m_mode1_error_weight = 1.0f;
	p->m_mode5_error_weight = 1.0f;
	p->m_mode6_error_weight = 1.0f;
	p->m_mode7_error_weight = 1.0f;
	p->m_low_frequency_partition_weight = 1.0f;
	bc7enc_compress_block_params_init_perceptual_weights(p);
}

// bc7enc_compress_block_init() MUST be called before calling bc7enc_compress_block() (or you'll get artifacts).
void bc7enc_compress_block_init();

// Packs a single block of 16x16 RGBA pixels (R first in memory) to 128-bit BC7 block pBlock, using either mode 1 and/or 6.
// Alpha blocks will always use mode 6, and by default opaque blocks will use either modes 1 or 6.
// Returns true if the block had any pixels with alpha < 255, otherwise it return false. (This is not an error code - a block is always encoded.)
bool bc7enc_compress_block(void *pBlock, const void *pPixelsRGBA, const bc7enc_compress_block_params *pComp_params);


