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
    uint32_t sampleNumber;
    uint32_t currentChannel = ~0U; /* Don't start matched. */

    /* This is specifically for unpacked formats which means the size of */
    /* each component is the same. */
    *numComponents = 0;
    for (sampleNumber = 0; sampleNumber < numSamples; ++sampleNumber) {
        uint32_t sampleByteLength = (KHR_DFDSVAL(BDFDB, sampleNumber, BITLENGTH) + 1) >> 3U;
        uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleNumber, CHANNELID);

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
    uint32_t sampleNumber;

    for (sampleNumber = 0; sampleNumber < numSamples; ++sampleNumber) {
        uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleNumber, CHANNELID);
        if (sampleChannel != currentChannel) {
            numComponents++;
            currentChannel = sampleChannel;
        }
    }
    return numComponents;
}

/**
 * @~English
 * @brief Reconstruct the value of bytesPlane0 from sample info.
 *
 * @deprecated Use reconstructDFDBytesPlanesFromSamples. This does not handle
 *             the possible second plane of the ETC1S model.
 *
 * Reconstruct the value for data that has been variable-rate compressed
 * and and whose bytesPlane0 value has been set to 0.  For DFDs that
 * are valid for KTX files. Little-endian data only and no multi-plane models
 * except ETC1S.
 *
 * @param DFD Pointer to the Data Format Descriptor for which to provide
 *            the value described as 32-bit words in native endianness.
 *            Note that this is the whole descriptor, not just
 *            the basic descriptor block.
 * @return The number of bytes a pixel occupies in bytesPlane0.
 */
uint32_t
reconstructDFDBytesPlane0FromSamples(const uint32_t* DFD)
{
    const uint32_t *BDFDB = DFD+1;
    uint32_t numSamples = KHR_DFDSAMPLECOUNT(BDFDB);
    uint32_t sampleNumber;

    uint32_t bitsPlane0 = 0;
    int32_t largestOffset = 0;
    uint32_t sampleNumberWithLargestOffset = 0;

    // Special case these depth{,-stencil} formats. The unused bits are
    // in the MSBs so have no visibility in the DFD therefore the max offset
    // algorithm below returns a value that is too small.
    if (KHR_DFDSVAL(BDFDB, 0, CHANNELID) == KHR_DF_CHANNEL_COMMON_DEPTH) {
        if (numSamples == 1) {
            if (KHR_DFDSVAL(BDFDB, 0, BITLENGTH) + 1 == 24) {
                // X8_D24_UNORM_PACK32,
                return 4;
            }
        } else if (numSamples == 2) {
            if (KHR_DFDSVAL(BDFDB, 0, BITLENGTH) + 1 == 16) {
                // D16_UNORM_S8_UINT
                return 4;
            }
            if (KHR_DFDSVAL(BDFDB, 0, BITLENGTH) + 1 == 32
                && KHR_DFDSVAL(BDFDB, 1, CHANNELID) == KHR_DF_CHANNEL_COMMON_STENCIL) {
                // D32_SFLOAT_S8_UINT
                return 8;
            }
        }
    }
    if (KHR_DFDVAL(BDFDB, MODEL) == KHR_DF_MODEL_ETC1S) {
        // Size of the first plane.
        return 8;
    }
    for (sampleNumber = 0; sampleNumber < numSamples; ++sampleNumber) {
        int32_t sampleBitOffset = KHR_DFDSVAL(BDFDB, sampleNumber, BITOFFSET);
        if (sampleBitOffset > largestOffset) {
            largestOffset = sampleBitOffset;
            sampleNumberWithLargestOffset = sampleNumber;
        }
    }

    /* The sample bitLength field stores the bit length - 1. */
    uint32_t sampleBitLength = KHR_DFDSVAL(BDFDB, sampleNumberWithLargestOffset, BITLENGTH) + 1;
    bitsPlane0 = largestOffset + sampleBitLength;
    return bitsPlane0 >> 3U;
}

/**
 * @~English
 * @brief Reconstruct the values of bytesPlane[01] from sample info.
 *
 * Reconstruct the values for data that has been variable-rate compressed
 * and whose bytesPlane[01] values have been set to 0 and update the
 * fields of the target DFD. For DFDs that are valid for KTX files.
 * Little-endian data only and no multi-plane models except ETC1S hence
 * only looking at bytesPlane0 abd bytesPlane1.
 *
 * @param DFD Pointer to a Data Format Descriptor for which,
 *            described as 32-bit words in native endianness.
 *            Note that this is the whole descriptor, not just
 *            the basic descriptor block.
 */

void
reconstructDFDBytesPlanesFromSamples(uint32_t* DFD)
{
    uint32_t *BDFDB = DFD+1;

    KHR_DFDSETVAL(BDFDB, BYTESPLANE0, reconstructDFDBytesPlane0FromSamples(DFD));
    if (KHR_DFDVAL(BDFDB, MODEL) == KHR_DF_MODEL_ETC1S) {
        if (KHR_DFDSAMPLECOUNT(BDFDB) == 2)
            KHR_DFDSETVAL(BDFDB, BYTESPLANE1, 8);
    }
}

/**
 * @~English
 * @brief Reconstruct the value of bytesPlane0 from sample info.
 *
 * @see reconstructDFDBytesPlane0FromSamples for details.
 * @deprecated For backward comparibility only. Use
 *             reconstructDFDBytesPlanesFromSamples.
 *
 * @param DFD Pointer to the Data Format Descriptor for which to provide
 *            the value described as 32-bit words in native endianness.
 *            Note that this is the whole descriptor, not just
 *            the basic descriptor block.
 * @param bytesPlane0  pointer to a 32-bit word in which the recreated
 *                    value of bytesPlane0 will be written.
 */
void
recreateBytesPlane0FromSampleInfo(const uint32_t* DFD, uint32_t* bytesPlane0)
{
    *bytesPlane0 = reconstructDFDBytesPlane0FromSamples(DFD);
}
