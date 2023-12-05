/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Utilities for querying info from a data format descriptor.
 * @author Mark Callow
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <KHR/khr_df.h>
#include "dfd.h"

/**
 * @~English
 * @brief Get the number and size of the image components from a DFD.
 *
 * This simplified function is for use only with the DFDs for unpacked
 * formats which means all components have the same size.
 *
 * @param DFD Pointer to a Data Format Descriptor to interpret,
              described as 32-bit words in native endianness.
              Note that this is the whole descriptor, not just
              the basic descriptor block.
 * @param numComponents pointer to a 32-bit word in which the number of
                        components will be written.
 * @param componentByteLength pointer to a 32-bit word in which the size of
                              a component in bytes will be written.
 */
void
getDFDComponentInfoUnpacked(const uint32_t* DFD, uint32_t* numComponents,
                            uint32_t* componentByteLength)
{
    const uint32_t *BDFDB = DFD+1;
    uint32_t numSamples = KHR_DFDSAMPLECOUNT(BDFDB);
    uint32_t sampleCounter;
    uint32_t currentChannel = ~0U; /* Don't start matched. */

    /* This is specifically for unpacked formats which means the size of */
    /* each component is the same. */
    *numComponents = 0;
    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        uint32_t sampleByteLength = (KHR_DFDSVAL(BDFDB, sampleCounter, BITLENGTH) + 1) >> 3U;
        uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleCounter, CHANNELID);

        if (sampleChannel == currentChannel) {
            /* Continuation of the same channel. */
            /* Accumulate the byte length. */
            *componentByteLength += sampleByteLength;
        } else {
            /* Everything is new. Hopefully. */
            currentChannel = sampleChannel;
            (*numComponents)++;
            *componentByteLength = sampleByteLength;
        }
    }
}

/**
 * @~English
 * @brief Return the number of "components" in the data.
 *
 * Calculates the number of uniques samples in the DFD by combining
 * multiple samples for the same channel. For uncompressed colorModels
 * this is the same as the number of components in the image data. For
 * block-compressed color models this is the number of samples in
 * the color model, typically 1 and in a few cases 2.
 *
 * @param DFD Pointer to a Data Format Descriptor for which,
 *            described as 32-bit words in native endianness.
 *            Note that this is the whole descriptor, not just
 *            the basic descriptor block.
 */
uint32_t getDFDNumComponents(const uint32_t* DFD)
{
    const uint32_t *BDFDB = DFD+1;
    uint32_t currentChannel = ~0U; /* Don't start matched. */
    uint32_t numComponents = 0;
    uint32_t numSamples = KHR_DFDSAMPLECOUNT(BDFDB);
    uint32_t sampleCounter;

    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleCounter, CHANNELID);
        if (sampleChannel != currentChannel) {
            numComponents++;
            currentChannel = sampleChannel;
        }
    }
    return numComponents;
}

/**
 * @~English
 * @brief Recreate the value of bytesPlane0 from sample info.
 *
 * This can be use to recreate the value of bytesPlane0 for data that
 * has been variable-rate compressed so has bytesPlane0 = 0.  For DFDs
 * that are valid for KTX files. Little-endian data only and no multi-plane
 * formats.
 *
 * @param DFD Pointer to a Data Format Descriptor for which,
 *            described as 32-bit words in native endianness.
 *            Note that this is the whole descriptor, not just
 *            the basic descriptor block.
 * @param bytesPlane0  pointer to a 32-bit word in which the recreated
 *                    value of bytesPlane0 will be written.
 */
void
recreateBytesPlane0FromSampleInfo(const uint32_t* DFD, uint32_t* bytesPlane0)
{
    const uint32_t *BDFDB = DFD+1;
    uint32_t numSamples = KHR_DFDSAMPLECOUNT(BDFDB);
    uint32_t sampleCounter;

    uint32_t bitsPlane0 = 0;
    uint32_t* bitOffsets = malloc(sizeof(uint32_t) * numSamples);
    memset(bitOffsets, -1, sizeof(uint32_t) * numSamples);
    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        uint32_t sampleBitOffset = KHR_DFDSVAL(BDFDB, sampleCounter, BITOFFSET);
        /* The sample bitLength field stores the bit length - 1. */
        uint32_t sampleBitLength = KHR_DFDSVAL(BDFDB, sampleCounter, BITLENGTH) + 1;
        uint32_t i;
        for (i = 0; i < numSamples; i++) {
            if (sampleBitOffset == bitOffsets[i]) {
                // This sample is being repeated as in e.g. RGB9E5.
                break;
            }
        }
        if (i == numSamples) {
            // Previously unseen bitOffset. Bump size.
            bitsPlane0 += sampleBitLength;
            bitOffsets[sampleCounter] = sampleBitOffset;
        }
    }
    free(bitOffsets);
    *bytesPlane0 = bitsPlane0 >> 3U;
}

