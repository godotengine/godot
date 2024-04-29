/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Utilities for printing data format descriptors.
 */

/*
 * Author: Andrew Garrard
 */

#include <stdio.h>
#include <KHR/khr_df.h>
#include "dfd.h"

/**
 * @~English
 * @brief Print a human-readable interpretation of a data format descriptor.
 *
 * @param DFD Pointer to a data format descriptor.
 **/
void printDFD(uint32_t *DFD)
{
    uint32_t *BDB = DFD+1;
    int samples = (KHR_DFDVAL(BDB, DESCRIPTORBLOCKSIZE) - 4 * KHR_DF_WORD_SAMPLESTART) / (4 * KHR_DF_WORD_SAMPLEWORDS);
    int sample;
    int model = KHR_DFDVAL(BDB, MODEL);
    printf("DFD total bytes: %d\n", DFD[0]);
    printf("BDB descriptor type 0x%04x vendor id = 0x%05x\n",
           KHR_DFDVAL(BDB, DESCRIPTORTYPE),
           KHR_DFDVAL(BDB, VENDORID));
    printf("Descriptor block size %d (%d samples) versionNumber = 0x%04x\n",
           KHR_DFDVAL(BDB, DESCRIPTORBLOCKSIZE),
           samples,
           KHR_DFDVAL(BDB, VERSIONNUMBER));
    printf("Flags 0x%02x Xfer %02d Primaries %02d Model %03d\n",
           KHR_DFDVAL(BDB, FLAGS),
           KHR_DFDVAL(BDB, TRANSFER),
           KHR_DFDVAL(BDB, PRIMARIES),
           model);
    printf("Dimensions: %d,%d,%d,%d\n",
           KHR_DFDVAL(BDB, TEXELBLOCKDIMENSION0) + 1,
           KHR_DFDVAL(BDB, TEXELBLOCKDIMENSION1) + 1,
           KHR_DFDVAL(BDB, TEXELBLOCKDIMENSION2) + 1,
           KHR_DFDVAL(BDB, TEXELBLOCKDIMENSION3) + 1);
    printf("Plane bytes: %d,%d,%d,%d,%d,%d,%d,%d\n",
           KHR_DFDVAL(BDB, BYTESPLANE0),
           KHR_DFDVAL(BDB, BYTESPLANE1),
           KHR_DFDVAL(BDB, BYTESPLANE2),
           KHR_DFDVAL(BDB, BYTESPLANE3),
           KHR_DFDVAL(BDB, BYTESPLANE4),
           KHR_DFDVAL(BDB, BYTESPLANE5),
           KHR_DFDVAL(BDB, BYTESPLANE6),
           KHR_DFDVAL(BDB, BYTESPLANE7));
    for (sample = 0; sample < samples; ++sample) {
        int channelId = KHR_DFDSVAL(BDB, sample, CHANNELID);
        printf("    Sample %d\n", sample);
        printf("Qualifiers %x", KHR_DFDSVAL(BDB, sample, QUALIFIERS) >> 4);
        printf(" Channel 0x%x", channelId);
        if (model == KHR_DF_MODEL_UASTC) {
            printf(" (%s)",
                   channelId == KHR_DF_CHANNEL_UASTC_RRRG ? "RRRG"
                 : channelId == KHR_DF_CHANNEL_UASTC_RGBA ? "RGBA"
                 : channelId == KHR_DF_CHANNEL_UASTC_RRR ? "RRR"
                 : channelId == KHR_DF_CHANNEL_UASTC_RGB ? "RGB"
                 : channelId == KHR_DF_CHANNEL_UASTC_RG ? "RG"
                 : "unknown");
        } else if (model == KHR_DF_MODEL_ETC1S) {
            printf(" (%s)",
                   channelId == KHR_DF_CHANNEL_ETC1S_AAA ? "AAA"
                 : channelId == KHR_DF_CHANNEL_ETC1S_GGG ? "GGG"
                 : channelId == KHR_DF_CHANNEL_ETC1S_RRR ? "RRR"
                 : channelId == KHR_DF_CHANNEL_ETC1S_RGB ? "RGB"
                 : "unknown");
        } else {
            printf(" (%c)",
               "RGB3456789abcdeA"[channelId]);
        }
        printf(" Length %d bits Offset %d\n",
               KHR_DFDSVAL(BDB, sample, BITLENGTH) + 1,
               KHR_DFDSVAL(BDB, sample, BITOFFSET));
        printf("Position: %d,%d,%d,%d\n",
               KHR_DFDSVAL(BDB, sample, SAMPLEPOSITION0),
               KHR_DFDSVAL(BDB, sample, SAMPLEPOSITION1),
               KHR_DFDSVAL(BDB, sample, SAMPLEPOSITION2),
               KHR_DFDSVAL(BDB, sample, SAMPLEPOSITION3));
        printf("Lower 0x%08x\nUpper 0x%08x\n",
               KHR_DFDSVAL(BDB, sample, SAMPLELOWER),
               KHR_DFDSVAL(BDB, sample, SAMPLEUPPER));
    }
}
