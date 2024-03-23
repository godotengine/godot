// SPDX-License-Identifier: Apache-2.0
// ----------------------------------------------------------------------------
// Copyright 2011-2023 Arm Limited
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
 * @brief Functions for converting between symbolic and physical encodings.
 */

#include "astcenc_internal.h"

#include <cassert>

/**
 * @brief Reverse bits in a byte.
 *
 * @param p   The value to reverse.
  *
 * @return The reversed result.
 */
static inline int bitrev8(int p)
{
	p = ((p & 0x0F) << 4) | ((p >> 4) & 0x0F);
	p = ((p & 0x33) << 2) | ((p >> 2) & 0x33);
	p = ((p & 0x55) << 1) | ((p >> 1) & 0x55);
	return p;
}


/**
 * @brief Read up to 8 bits at an arbitrary bit offset.
 *
 * The stored value is at most 8 bits, but can be stored at an offset of between 0 and 7 bits so may
 * span two separate bytes in memory.
 *
 * @param         bitcount    The number of bits to read.
 * @param         bitoffset   The bit offset to read from, between 0 and 7.
 * @param[in,out] ptr         The data pointer to read from.
 *
 * @return The read value.
 */
static inline int read_bits(
	int bitcount,
	int bitoffset,
	const uint8_t* ptr
) {
	int mask = (1 << bitcount) - 1;
	ptr += bitoffset >> 3;
	bitoffset &= 7;
	int value = ptr[0] | (ptr[1] << 8);
	value >>= bitoffset;
	value &= mask;
	return value;
}

#if !defined(ASTCENC_DECOMPRESS_ONLY)

/**
 * @brief Write up to 8 bits at an arbitrary bit offset.
 *
 * The stored value is at most 8 bits, but can be stored at an offset of between 0 and 7 bits so
 * may span two separate bytes in memory.
 *
 * @param         value       The value to write.
 * @param         bitcount    The number of bits to write, starting from LSB.
 * @param         bitoffset   The bit offset to store at, between 0 and 7.
 * @param[in,out] ptr         The data pointer to write to.
 */
static inline void write_bits(
	int value,
	int bitcount,
	int bitoffset,
	uint8_t* ptr
) {
	int mask = (1 << bitcount) - 1;
	value &= mask;
	ptr += bitoffset >> 3;
	bitoffset &= 7;
	value <<= bitoffset;
	mask <<= bitoffset;
	mask = ~mask;

	ptr[0] &= mask;
	ptr[0] |= value;
	ptr[1] &= mask >> 8;
	ptr[1] |= value >> 8;
}

/* See header for documentation. */
void symbolic_to_physical(
	const block_size_descriptor& bsd,
	const symbolic_compressed_block& scb,
	uint8_t pcb[16]
) {
	assert(scb.block_type != SYM_BTYPE_ERROR);

	// Constant color block using UNORM16 colors
	if (scb.block_type == SYM_BTYPE_CONST_U16)
	{
		// There is currently no attempt to coalesce larger void-extents
		static const uint8_t cbytes[8] { 0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
		for (unsigned int i = 0; i < 8; i++)
		{
			pcb[i] = cbytes[i];
		}

		for (unsigned int i = 0; i < BLOCK_MAX_COMPONENTS; i++)
		{
			pcb[2 * i + 8] = scb.constant_color[i] & 0xFF;
			pcb[2 * i + 9] = (scb.constant_color[i] >> 8) & 0xFF;
		}

		return;
	}

	// Constant color block using FP16 colors
	if (scb.block_type == SYM_BTYPE_CONST_F16)
	{
		// There is currently no attempt to coalesce larger void-extents
		static const uint8_t cbytes[8]  { 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF };
		for (unsigned int i = 0; i < 8; i++)
		{
			pcb[i] = cbytes[i];
		}

		for (unsigned int i = 0; i < BLOCK_MAX_COMPONENTS; i++)
		{
			pcb[2 * i + 8] = scb.constant_color[i] & 0xFF;
			pcb[2 * i + 9] = (scb.constant_color[i] >> 8) & 0xFF;
		}

		return;
	}

	unsigned int partition_count = scb.partition_count;

	// Compress the weights.
	// They are encoded as an ordinary integer-sequence, then bit-reversed
	uint8_t weightbuf[16] { 0 };

	const auto& bm = bsd.get_block_mode(scb.block_mode);
	const auto& di = bsd.get_decimation_info(bm.decimation_mode);
	int weight_count = di.weight_count;
	quant_method weight_quant_method = bm.get_weight_quant_mode();
	float weight_quant_levels = static_cast<float>(get_quant_level(weight_quant_method));
	int is_dual_plane = bm.is_dual_plane;

	const auto& qat = quant_and_xfer_tables[weight_quant_method];

	int real_weight_count = is_dual_plane ? 2 * weight_count : weight_count;

	int bits_for_weights = get_ise_sequence_bitcount(real_weight_count, weight_quant_method);

	uint8_t weights[64];
	if (is_dual_plane)
	{
		for (int i = 0; i < weight_count; i++)
		{
			float uqw = static_cast<float>(scb.weights[i]);
			float qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weights[2 * i] = qat.scramble_map[qwi];

			uqw = static_cast<float>(scb.weights[i + WEIGHTS_PLANE2_OFFSET]);
			qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			qwi = static_cast<int>(qw + 0.5f);
			weights[2 * i + 1] = qat.scramble_map[qwi];
		}
	}
	else
	{
		for (int i = 0; i < weight_count; i++)
		{
			float uqw = static_cast<float>(scb.weights[i]);
			float qw = (uqw / 64.0f) * (weight_quant_levels - 1.0f);
			int qwi = static_cast<int>(qw + 0.5f);
			weights[i] = qat.scramble_map[qwi];
		}
	}

	encode_ise(weight_quant_method, real_weight_count, weights, weightbuf, 0);

	for (int i = 0; i < 16; i++)
	{
		pcb[i] = static_cast<uint8_t>(bitrev8(weightbuf[15 - i]));
	}

	write_bits(scb.block_mode, 11, 0, pcb);
	write_bits(partition_count - 1, 2, 11, pcb);

	int below_weights_pos = 128 - bits_for_weights;

	// Encode partition index and color endpoint types for blocks with 2+ partitions
	if (partition_count > 1)
	{
		write_bits(scb.partition_index, 6, 13, pcb);
		write_bits(scb.partition_index >> 6, PARTITION_INDEX_BITS - 6, 19, pcb);

		if (scb.color_formats_matched)
		{
			write_bits(scb.color_formats[0] << 2, 6, 13 + PARTITION_INDEX_BITS, pcb);
		}
		else
		{
			// Check endpoint types for each partition to determine the lowest class present
			int low_class = 4;

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int class_of_format = scb.color_formats[i] >> 2;
				low_class = astc::min(class_of_format, low_class);
			}

			if (low_class == 3)
			{
				low_class = 2;
			}

			int encoded_type = low_class + 1;
			int bitpos = 2;

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int classbit_of_format = (scb.color_formats[i] >> 2) - low_class;
				encoded_type |= classbit_of_format << bitpos;
				bitpos++;
			}

			for (unsigned int i = 0; i < partition_count; i++)
			{
				int lowbits_of_format = scb.color_formats[i] & 3;
				encoded_type |= lowbits_of_format << bitpos;
				bitpos += 2;
			}

			int encoded_type_lowpart = encoded_type & 0x3F;
			int encoded_type_highpart = encoded_type >> 6;
			int encoded_type_highpart_size = (3 * partition_count) - 4;
			int encoded_type_highpart_pos = 128 - bits_for_weights - encoded_type_highpart_size;
			write_bits(encoded_type_lowpart, 6, 13 + PARTITION_INDEX_BITS, pcb);
			write_bits(encoded_type_highpart, encoded_type_highpart_size, encoded_type_highpart_pos, pcb);
			below_weights_pos -= encoded_type_highpart_size;
		}
	}
	else
	{
		write_bits(scb.color_formats[0], 4, 13, pcb);
	}

	// In dual-plane mode, encode the color component of the second plane of weights
	if (is_dual_plane)
	{
		write_bits(scb.plane2_component, 2, below_weights_pos - 2, pcb);
	}

	// Encode the color components
	uint8_t values_to_encode[32];
	int valuecount_to_encode = 0;

	const uint8_t* pack_table = color_uquant_to_scrambled_pquant_tables[scb.quant_mode - QUANT_6];
	for (unsigned int i = 0; i < scb.partition_count; i++)
	{
		int vals = 2 * (scb.color_formats[i] >> 2) + 2;
		assert(vals <= 8);
		for (int j = 0; j < vals; j++)
		{
			values_to_encode[j + valuecount_to_encode] = pack_table[scb.color_values[i][j]];
		}
		valuecount_to_encode += vals;
	}

	encode_ise(scb.get_color_quant_mode(), valuecount_to_encode, values_to_encode, pcb,
	           scb.partition_count == 1 ? 17 : 19 + PARTITION_INDEX_BITS);
}

#endif

/* See header for documentation. */
void physical_to_symbolic(
	const block_size_descriptor& bsd,
	const uint8_t pcb[16],
	symbolic_compressed_block& scb
) {
	uint8_t bswapped[16];

	scb.block_type = SYM_BTYPE_NONCONST;

	// Extract header fields
	int block_mode = read_bits(11, 0, pcb);
	if ((block_mode & 0x1FF) == 0x1FC)
	{
		// Constant color block

		// Check what format the data has
		if (block_mode & 0x200)
		{
			scb.block_type = SYM_BTYPE_CONST_F16;
		}
		else
		{
			scb.block_type = SYM_BTYPE_CONST_U16;
		}

		scb.partition_count = 0;
		for (int i = 0; i < 4; i++)
		{
			scb.constant_color[i] = pcb[2 * i + 8] | (pcb[2 * i + 9] << 8);
		}

		// Additionally, check that the void-extent
		if (bsd.zdim == 1)
		{
			// 2D void-extent
			int rsvbits = read_bits(2, 10, pcb);
			if (rsvbits != 3)
			{
				scb.block_type = SYM_BTYPE_ERROR;
				return;
			}

			int vx_low_s = read_bits(8, 12, pcb) | (read_bits(5, 12 + 8, pcb) << 8);
			int vx_high_s = read_bits(8, 25, pcb) | (read_bits(5, 25 + 8, pcb) << 8);
			int vx_low_t = read_bits(8, 38, pcb) | (read_bits(5, 38 + 8, pcb) << 8);
			int vx_high_t = read_bits(8, 51, pcb) | (read_bits(5, 51 + 8, pcb) << 8);

			int all_ones = vx_low_s == 0x1FFF && vx_high_s == 0x1FFF && vx_low_t == 0x1FFF && vx_high_t == 0x1FFF;

			if ((vx_low_s >= vx_high_s || vx_low_t >= vx_high_t) && !all_ones)
			{
				scb.block_type = SYM_BTYPE_ERROR;
				return;
			}
		}
		else
		{
			// 3D void-extent
			int vx_low_s = read_bits(9, 10, pcb);
			int vx_high_s = read_bits(9, 19, pcb);
			int vx_low_t = read_bits(9, 28, pcb);
			int vx_high_t = read_bits(9, 37, pcb);
			int vx_low_p = read_bits(9, 46, pcb);
			int vx_high_p = read_bits(9, 55, pcb);

			int all_ones = vx_low_s == 0x1FF && vx_high_s == 0x1FF && vx_low_t == 0x1FF && vx_high_t == 0x1FF && vx_low_p == 0x1FF && vx_high_p == 0x1FF;

			if ((vx_low_s >= vx_high_s || vx_low_t >= vx_high_t || vx_low_p >= vx_high_p) && !all_ones)
			{
				scb.block_type = SYM_BTYPE_ERROR;
				return;
			}
		}

		return;
	}

	unsigned int packed_index = bsd.block_mode_packed_index[block_mode];
	if (packed_index == BLOCK_BAD_BLOCK_MODE)
	{
		scb.block_type = SYM_BTYPE_ERROR;
		return;
	}

	const auto& bm = bsd.get_block_mode(block_mode);
	const auto& di = bsd.get_decimation_info(bm.decimation_mode);

	int weight_count = di.weight_count;
	promise(weight_count > 0);

	quant_method weight_quant_method = static_cast<quant_method>(bm.quant_mode);
	int is_dual_plane = bm.is_dual_plane;

	int real_weight_count = is_dual_plane ? 2 * weight_count : weight_count;

	int partition_count = read_bits(2, 11, pcb) + 1;
	promise(partition_count > 0);

	scb.block_mode = static_cast<uint16_t>(block_mode);
	scb.partition_count = static_cast<uint8_t>(partition_count);

	for (int i = 0; i < 16; i++)
	{
		bswapped[i] = static_cast<uint8_t>(bitrev8(pcb[15 - i]));
	}

	int bits_for_weights = get_ise_sequence_bitcount(real_weight_count, weight_quant_method);

	int below_weights_pos = 128 - bits_for_weights;

	uint8_t indices[64];
	const auto& qat = quant_and_xfer_tables[weight_quant_method];

	decode_ise(weight_quant_method, real_weight_count, bswapped, indices, 0);

	if (is_dual_plane)
	{
		for (int i = 0; i < weight_count; i++)
		{
			scb.weights[i] = qat.unscramble_and_unquant_map[indices[2 * i]];
			scb.weights[i + WEIGHTS_PLANE2_OFFSET] = qat.unscramble_and_unquant_map[indices[2 * i + 1]];
		}
	}
	else
	{
		for (int i = 0; i < weight_count; i++)
		{
			scb.weights[i] = qat.unscramble_and_unquant_map[indices[i]];
		}
	}

	if (is_dual_plane && partition_count == 4)
	{
		scb.block_type = SYM_BTYPE_ERROR;
		return;
	}

	scb.color_formats_matched = 0;

	// Determine the format of each endpoint pair
	int color_formats[BLOCK_MAX_PARTITIONS];
	int encoded_type_highpart_size = 0;
	if (partition_count == 1)
	{
		color_formats[0] = read_bits(4, 13, pcb);
		scb.partition_index = 0;
	}
	else
	{
		encoded_type_highpart_size = (3 * partition_count) - 4;
		below_weights_pos -= encoded_type_highpart_size;
		int encoded_type = read_bits(6, 13 + PARTITION_INDEX_BITS, pcb) |
		                  (read_bits(encoded_type_highpart_size, below_weights_pos, pcb) << 6);
		int baseclass = encoded_type & 0x3;
		if (baseclass == 0)
		{
			for (int i = 0; i < partition_count; i++)
			{
				color_formats[i] = (encoded_type >> 2) & 0xF;
			}

			below_weights_pos += encoded_type_highpart_size;
			scb.color_formats_matched = 1;
			encoded_type_highpart_size = 0;
		}
		else
		{
			int bitpos = 2;
			baseclass--;

			for (int i = 0; i < partition_count; i++)
			{
				color_formats[i] = (((encoded_type >> bitpos) & 1) + baseclass) << 2;
				bitpos++;
			}

			for (int i = 0; i < partition_count; i++)
			{
				color_formats[i] |= (encoded_type >> bitpos) & 3;
				bitpos += 2;
			}
		}
		scb.partition_index = static_cast<uint16_t>(read_bits(6, 13, pcb) |
		                                            (read_bits(PARTITION_INDEX_BITS - 6, 19, pcb) << 6));
	}

	for (int i = 0; i < partition_count; i++)
	{
		scb.color_formats[i] = static_cast<uint8_t>(color_formats[i]);
	}

	// Determine number of color endpoint integers
	int color_integer_count = 0;
	for (int i = 0; i < partition_count; i++)
	{
		int endpoint_class = color_formats[i] >> 2;
		color_integer_count += (endpoint_class + 1) * 2;
	}

	if (color_integer_count > 18)
	{
		scb.block_type = SYM_BTYPE_ERROR;
		return;
	}

	// Determine the color endpoint format to use
	static const int color_bits_arr[5] { -1, 115 - 4, 113 - 4 - PARTITION_INDEX_BITS, 113 - 4 - PARTITION_INDEX_BITS, 113 - 4 - PARTITION_INDEX_BITS };
	int color_bits = color_bits_arr[partition_count] - bits_for_weights - encoded_type_highpart_size;
	if (is_dual_plane)
	{
		color_bits -= 2;
	}

	if (color_bits < 0)
	{
		color_bits = 0;
	}

	int color_quant_level = quant_mode_table[color_integer_count >> 1][color_bits];
	if (color_quant_level < QUANT_6)
	{
		scb.block_type = SYM_BTYPE_ERROR;
		return;
	}

	// Unpack the integer color values and assign to endpoints
	scb.quant_mode = static_cast<quant_method>(color_quant_level);

	uint8_t values_to_decode[32];
	decode_ise(static_cast<quant_method>(color_quant_level), color_integer_count, pcb,
	           values_to_decode, (partition_count == 1 ? 17 : 19 + PARTITION_INDEX_BITS));

	int valuecount_to_decode = 0;
	const uint8_t* unpack_table = color_scrambled_pquant_to_uquant_tables[scb.quant_mode - QUANT_6];
	for (int i = 0; i < partition_count; i++)
	{
		int vals = 2 * (color_formats[i] >> 2) + 2;
		for (int j = 0; j < vals; j++)
		{
			scb.color_values[i][j] = unpack_table[values_to_decode[j + valuecount_to_decode]];
		}
		valuecount_to_decode += vals;
	}

	// Fetch component for second-plane in the case of dual plane of weights.
	scb.plane2_component = -1;
	if (is_dual_plane)
	{
		scb.plane2_component = static_cast<int8_t>(read_bits(2, below_weights_pos - 2, pcb));
	}
}
