/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Utility for interpreting a data format descriptor.
 * @author Andrew Garrard
 */

#include <stdint.h>
#include <stdio.h>
#include <KHR/khr_df.h>
#include "dfd.h"

/**
 * @~English
 * @brief Interpret a Data Format Descriptor for a simple format.
 *
 * @param DFD Pointer to a Data Format Descriptor to interpret,
              described as 32-bit words in native endianness.
              Note that this is the whole descriptor, not just
              the basic descriptor block.
 * @param R Information about the decoded red channel, if any.
 * @param G Information about the decoded green channel, if any.
 * @param B Information about the decoded blue channel, if any.
 * @param A Information about the decoded alpha channel, if any.
 * @param wordBytes Byte size of the channels (unpacked) or total size (packed).
 *
 * @return An enumerant describing the decoded value,
 *         or an error code in case of failure.
 **/
enum InterpretDFDResult interpretDFD(const uint32_t *DFD,
                                     InterpretedDFDChannel *R,
                                     InterpretedDFDChannel *G,
                                     InterpretedDFDChannel *B,
                                     InterpretedDFDChannel *A,
                                     uint32_t *wordBytes)
{
    /* We specifically handle "simple" cases that can be translated */
    /* to things a GPU can access. For simplicity, we also ignore */
    /* the compressed formats, which are generally a single sample */
    /* (and I believe are all defined to be little-endian in their */
    /* in-memory layout, even if some documentation confuses this). */
    /* We also just worry about layout and ignore sRGB, since that's */
    /* trivial to extract anyway. */

    /* DFD points to the whole descriptor, not the basic descriptor block. */
    /* Make everything else relative to the basic descriptor block. */
    const uint32_t *BDFDB = DFD+1;

    uint32_t numSamples = KHR_DFDSAMPLECOUNT(BDFDB);

    uint32_t sampleCounter;
    int determinedEndianness = 0;
    int determinedNormalizedness = 0;
    int determinedSignedness = 0;
    int determinedFloatness = 0;
    enum InterpretDFDResult result = 0; /* Build this up incrementally. */

    /* Clear these so following code doesn't get confused. */
    R->offset = R->size = 0;
    G->offset = G->size = 0;
    B->offset = B->size = 0;
    A->offset = A->size = 0;

    /* First rule out the multiple planes case (trivially) */
    /* - that is, we check that only bytesPlane0 is non-zero. */
    /* This means we don't handle YUV even if the API could. */
    /* (We rely on KHR_DF_WORD_BYTESPLANE0..3 being the same and */
    /* KHR_DF_WORD_BYTESPLANE4..7 being the same as a short cut.) */
    if ((BDFDB[KHR_DF_WORD_BYTESPLANE0] & ~KHR_DF_MASK_BYTESPLANE0)
        || BDFDB[KHR_DF_WORD_BYTESPLANE4]) return i_UNSUPPORTED_MULTIPLE_PLANES;

    /* Only support the RGB color model. */
    /* We could expand this to allow "UNSPECIFIED" as well. */
    if (KHR_DFDVAL(BDFDB, MODEL) != KHR_DF_MODEL_RGBSDA) return i_UNSUPPORTED_CHANNEL_TYPES;

    /* We only pay attention to sRGB. */
    if (KHR_DFDVAL(BDFDB, TRANSFER) == KHR_DF_TRANSFER_SRGB) result |= i_SRGB_FORMAT_BIT;

    /* We only support samples at coordinate 0,0,0,0. */
    /* (We could confirm this from texel_block_dimensions in 1.2, but */
    /* the interpretation might change in later versions.) */
    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        if (KHR_DFDSVAL(BDFDB, sampleCounter, SAMPLEPOSITION_ALL))
            return i_UNSUPPORTED_MULTIPLE_SAMPLE_LOCATIONS;
    }

    /* Set flags and check for consistency. */
    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        /* Note: We're ignoring 9995, which is weird and worth special-casing */
        /* rather than trying to generalise to all float formats. */
        if (!determinedFloatness) {
            if (KHR_DFDSVAL(BDFDB, sampleCounter, QUALIFIERS)
                 & KHR_DF_SAMPLE_DATATYPE_FLOAT) {
                result |= i_FLOAT_FORMAT_BIT;
                determinedFloatness = 1;
            }
        } else {
            /* Check whether we disagree with our predetermined floatness. */
            /* Note that this could justifiably happen with (say) D24S8. */
            if (KHR_DFDSVAL(BDFDB, sampleCounter, QUALIFIERS)
                 & KHR_DF_SAMPLE_DATATYPE_FLOAT) {
                if (!(result & i_FLOAT_FORMAT_BIT)) return i_UNSUPPORTED_MIXED_CHANNELS;
            } else {
                if ((result & i_FLOAT_FORMAT_BIT)) return i_UNSUPPORTED_MIXED_CHANNELS;
            }
        }
        if (!determinedSignedness) {
            if (KHR_DFDSVAL(BDFDB, sampleCounter, QUALIFIERS)
                 & KHR_DF_SAMPLE_DATATYPE_SIGNED) {
                result |= i_SIGNED_FORMAT_BIT;
                determinedSignedness = 1;
            }
        } else {
            /* Check whether we disagree with our predetermined signedness. */
            if (KHR_DFDSVAL(BDFDB, sampleCounter, QUALIFIERS)
                 & KHR_DF_SAMPLE_DATATYPE_SIGNED) {
                if (!(result & i_SIGNED_FORMAT_BIT)) return i_UNSUPPORTED_MIXED_CHANNELS;
            } else {
                if ((result & i_SIGNED_FORMAT_BIT)) return i_UNSUPPORTED_MIXED_CHANNELS;
            }
        }
        /* We define "unnormalized" as "sample_upper = 1". */
        /* We don't check whether any non-1 normalization value is correct */
        /* (i.e. set to the maximum bit value, and check min value) on */
        /* the assumption that we're looking at a format which *came* from */
        /* an API we can support. */
        if (!determinedNormalizedness) {
            /* The ambiguity here is if the bottom bit is a single-bit value, */
            /* as in RGBA 5:5:5:1, so we defer the decision if the channel only has one bit. */
            if (KHR_DFDSVAL(BDFDB, sampleCounter, BITLENGTH) > 0) {
                if ((result & i_FLOAT_FORMAT_BIT)) {
                    if (*(float *)(void *)&BDFDB[KHR_DF_WORD_SAMPLESTART +
                                                 KHR_DF_WORD_SAMPLEWORDS * sampleCounter +
                                                 KHR_DF_SAMPLEWORD_SAMPLEUPPER] != 1.0f) {
                        result |= i_NORMALIZED_FORMAT_BIT;
                    }
                } else {
                    if (KHR_DFDSVAL(BDFDB, sampleCounter, SAMPLEUPPER) != 1U) {
                        result |= i_NORMALIZED_FORMAT_BIT;
                    }
                }
                determinedNormalizedness = 1;
            }
        }
        /* Note: We don't check for inconsistent normalization, because */
        /* channels composed of multiple samples will have 0 in the */
        /* lower/upper range. */
        /* This heuristic should handle 64-bit integers, too. */
    }

    /* If this is a packed format, we work out our offsets differently. */
    /* We assume a packed format has channels that aren't byte-aligned. */
    /* If we have a format in which every channel is byte-aligned *and* packed, */
    /* we have the RGBA/ABGR ambiguity; we *probably* don't want the packed */
    /* version in this case, and if hardware has to pack it and swizzle, */
    /* that's up to the hardware to special-case. */
    for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
        if (KHR_DFDSVAL(BDFDB, sampleCounter, BITOFFSET) & 0x7U) {
            result |= i_PACKED_FORMAT_BIT;
            /* Once we're packed, we're packed, no need to keep checking. */
            break;
        }
    }

    /* Remember: the canonical ordering of samples is to start with */
    /* the lowest bit of the channel/location which touches bit 0 of */
    /* the data, when the latter is concatenated in little-endian order, */
    /* and then progress until all the bits of that channel/location */
    /* have been processed. Multiple channels sharing the same source */
    /* bits are processed in channel ID order. (I should clarify this */
    /* for partially-shared data, but it doesn't really matter so long */
    /* as everything is consecutive, except to make things canonical.) */
    /* Note: For standard formats we could determine big/little-endianness */
    /* simply from whether the first sample starts in bit 0; technically */
    /* it's possible to have a format with unaligned channels wherein the */
    /* first channel starts at bit 0 and is one byte, yet other channels */
    /* take more bytes or aren't aligned (e.g. D24S8), but this should be */
    /* irrelevant for the formats that we support. */
    if ((result & i_PACKED_FORMAT_BIT)) {
        /* A packed format. */
        uint32_t currentChannel = ~0U; /* Don't start matched. */
        uint32_t currentBitOffset = 0;
        uint32_t currentByteOffset = 0;
        uint32_t currentBitLength = 0;
        *wordBytes = (BDFDB[KHR_DF_WORD_BYTESPLANE0] & 0xFFU);
        for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
            uint32_t sampleBitOffset = KHR_DFDSVAL(BDFDB, sampleCounter, BITOFFSET);
            uint32_t sampleByteOffset = sampleBitOffset >> 3U;
            /* The sample bitLength field stores the bit length - 1. */
            uint32_t sampleBitLength = KHR_DFDSVAL(BDFDB, sampleCounter, BITLENGTH) + 1;
            uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleCounter, CHANNELID);
            InterpretedDFDChannel *sampleChannelPtr;
            switch (sampleChannel) {
            case KHR_DF_CHANNEL_RGBSDA_RED:
                sampleChannelPtr = R;
                break;
            case KHR_DF_CHANNEL_RGBSDA_GREEN:
                sampleChannelPtr = G;
                break;
            case KHR_DF_CHANNEL_RGBSDA_BLUE:
                sampleChannelPtr = B;
                break;
            case KHR_DF_CHANNEL_RGBSDA_ALPHA:
                sampleChannelPtr = A;
                break;
            default:
                return i_UNSUPPORTED_CHANNEL_TYPES;
            }
            if (sampleChannel == currentChannel) {
                /* Continuation of the same channel. */
                /* Since a big (>32-bit) channel isn't "packed", */
                /* this should only happen in big-endian, or if */
                /* we have a wacky format that we won't support. */
                if (sampleByteOffset == currentByteOffset - 1U && /* One byte earlier */
                    ((currentBitOffset + currentBitLength) & 7U) == 0 && /* Already at the end of a byte */
                    (sampleBitOffset & 7U) == 0) { /* Start at the beginning of the byte */
                    /* All is good, continue big-endian. */
                    /* N.B. We shouldn't be here if we decided we were little-endian, */
                    /* so we don't bother to check that disagreement. */
                    result |= i_BIG_ENDIAN_FORMAT_BIT;
                    determinedEndianness = 1;
                } else {
                    /* Oh dear. */
                    /* We could be little-endian, but not with any standard format. */
                    /* More likely we've got something weird that we can't support. */
                    return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                }
                /* Remember where we are. */
                currentBitOffset = sampleBitOffset;
                currentByteOffset = sampleByteOffset;
                currentBitLength = sampleBitLength;
                /* Accumulate the bit length. */
                sampleChannelPtr->size += sampleBitLength;
            } else {
                /* Everything is new. Hopefully. */
                currentChannel = sampleChannel;
                currentBitOffset = sampleBitOffset;
                currentByteOffset = sampleByteOffset;
                currentBitLength = sampleBitLength;
                if (sampleChannelPtr->size) {
                    /* Uh-oh, we've seen this channel before. */
                    return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                }
                /* For now, record the bit offset in little-endian terms, */
                /* because we may not know to reverse it yet. */
                sampleChannelPtr->offset = sampleBitOffset;
                sampleChannelPtr->size = sampleBitLength;
            }
        }
        if ((result & i_BIG_ENDIAN_FORMAT_BIT)) {
            /* Our bit offsets to bit 0 of each channel are in little-endian terms. */
            /* We need to do a byte swap to work out where they should be. */
            /* We assume, for sanity, that byte sizes are a power of two for this. */
            uint32_t offsetMask = (*wordBytes - 1U) << 3U;
            R->offset ^= offsetMask;
            G->offset ^= offsetMask;
            B->offset ^= offsetMask;
            A->offset ^= offsetMask;
        }
    } else {
        /* Not a packed format. */
        /* Everything is byte-aligned. */
        /* Question is whether there multiple samples per channel. */
        uint32_t currentChannel = ~0U; /* Don't start matched. */
        uint32_t currentByteOffset = 0;
        uint32_t currentByteLength = 0;
        for (sampleCounter = 0; sampleCounter < numSamples; ++sampleCounter) {
            uint32_t sampleByteOffset = KHR_DFDSVAL(BDFDB, sampleCounter, BITOFFSET) >> 3U;
            uint32_t sampleByteLength = (KHR_DFDSVAL(BDFDB, sampleCounter, BITLENGTH) + 1) >> 3U;
            uint32_t sampleChannel = KHR_DFDSVAL(BDFDB, sampleCounter, CHANNELID);
            InterpretedDFDChannel *sampleChannelPtr;
            switch (sampleChannel) {
            case KHR_DF_CHANNEL_RGBSDA_RED:
                sampleChannelPtr = R;
                break;
            case KHR_DF_CHANNEL_RGBSDA_GREEN:
                sampleChannelPtr = G;
                break;
            case KHR_DF_CHANNEL_RGBSDA_BLUE:
                sampleChannelPtr = B;
                break;
            case KHR_DF_CHANNEL_RGBSDA_ALPHA:
                sampleChannelPtr = A;
                break;
            default:
                return i_UNSUPPORTED_CHANNEL_TYPES;
            }
            if (sampleChannel == currentChannel) {
                /* Continuation of the same channel. */
                /* Either big-endian, or little-endian with a very large channel. */
                if (sampleByteOffset == currentByteOffset - 1) { /* One byte earlier */
                    if (determinedEndianness && !(result & i_BIG_ENDIAN_FORMAT_BIT)) {
                        return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                    }
                    /* All is good, continue big-endian. */
                    result |= i_BIG_ENDIAN_FORMAT_BIT;
                    determinedEndianness = 1;
                    /* Update the start */
                    sampleChannelPtr->offset = sampleByteOffset;
                } else if (sampleByteOffset == currentByteOffset + currentByteLength) {
                    if (determinedEndianness && (result & i_BIG_ENDIAN_FORMAT_BIT)) {
                        return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                    }
                    /* All is good, continue little-endian. */
                    determinedEndianness = 1;
                } else {
                    /* Oh dear. */
                    /* We could be little-endian, but not with any standard format. */
                    /* More likely we've got something weird that we can't support. */
                    return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                }
                /* Remember where we are. */
                currentByteOffset = sampleByteOffset;
                currentByteLength = sampleByteLength;
                /* Accumulate the byte length. */
                sampleChannelPtr->size += sampleByteLength;
                /* Assume these are all the same. */
                *wordBytes = sampleChannelPtr->size;
            } else {
                /* Everything is new. Hopefully. */
                currentChannel = sampleChannel;
                currentByteOffset = sampleByteOffset;
                currentByteLength = sampleByteLength;
                if (sampleChannelPtr->size) {
                    /* Uh-oh, we've seen this channel before. */
                    return i_UNSUPPORTED_NONTRIVIAL_ENDIANNESS;
                }
                /* For now, record the byte offset in little-endian terms, */
                /* because we may not know to reverse it yet. */
                sampleChannelPtr->offset = sampleByteOffset;
                sampleChannelPtr->size = sampleByteLength;
                /* Assume these are all the same. */
                *wordBytes = sampleByteLength;
            }
        }
    }
    return result;
}
