/* Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file
 * @~English
 * @brief Helper functions for colourspaces.
 */

#include <KHR/khr_df.h>
#include "dfd.h"

typedef struct s_PrimaryMapping {
    khr_df_primaries_e  dfPrimaryEnum;
    Primaries           primaries;
} sPrimaryMapping;

sPrimaryMapping primaryMap[] = {
    { KHR_DF_PRIMARIES_BT709,       { 0.640f,0.330f,   0.300f,0.600f, 0.150f,0.060f,   0.3127f,0.3290f}},
    { KHR_DF_PRIMARIES_BT601_EBU,   { 0.640f,0.330f,   0.290f,0.600f, 0.150f,0.060f,   0.3127f,0.3290f}},
    { KHR_DF_PRIMARIES_BT601_SMPTE, { 0.630f,0.340f,   0.310f,0.595f, 0.155f,0.070f,   0.3127f,0.3290f}},
    { KHR_DF_PRIMARIES_BT2020,      { 0.708f,0.292f,   0.170f,0.797f, 0.131f,0.046f,   0.3127f,0.3290f}},
    { KHR_DF_PRIMARIES_CIEXYZ,      { 1.0f,0.0f,       0.0f,1.0f,     0.0f,0.0f,       0.0f,1.0f}},
    { KHR_DF_PRIMARIES_ACES,        { 0.7347f,0.2653f, 0.0f,1.0f,     0.0001f,-0.077f, 0.32168f,0.33767f}},
    { KHR_DF_PRIMARIES_ACESCC,      { 0.713f,0.293f,   0.165f,0.830f, 0.128f,0.044f,   0.32168f,0.33767f}},
    { KHR_DF_PRIMARIES_NTSC1953,    { 0.67f,0.33f,     0.21f,0.71f,   0.14f,0.08f,     0.310f,0.316f}},
    { KHR_DF_PRIMARIES_PAL525,      { 0.630f,0.340f,   0.310f,0.595f, 0.155f,0.070f,   0.3101f,0.3162f}},
    { KHR_DF_PRIMARIES_DISPLAYP3,   { 0.6800f,0.3200f, 0.2650f,0.69f, 0.1500f,0.0600f, 0.3127f,0.3290f}},
    { KHR_DF_PRIMARIES_ADOBERGB,    { 0.6400f,0.3300f, 0.2100f,0.71f, 0.1500f,0.0600f, 0.3127f,0.3290f}}};

/**
 * @brief Map a set of primaries to a KDFS primaries enum.
 *
 * @param[in] p           pointer to a Primaries struct filled in with the primary values.
 * @param[in] latitude tolerance to use while matching. A suitable value might be 0.002
 *                 but it depends on the application.
 */
khr_df_primaries_e findMapping(Primaries *p, float latitude) {
    unsigned int i;
    for (i = 0; i < sizeof(primaryMap)/sizeof(sPrimaryMapping); ++i) {
        if (primaryMap[i].primaries.Rx - p->Rx <= latitude && p->Rx - primaryMap[i].primaries.Rx <= latitude &&
            primaryMap[i].primaries.Gx - p->Gx <= latitude && p->Gx - primaryMap[i].primaries.Gx <= latitude &&
            primaryMap[i].primaries.Bx - p->Bx <= latitude && p->Bx - primaryMap[i].primaries.Bx <= latitude &&
            primaryMap[i].primaries.Wx - p->Wx <= latitude && p->Wx - primaryMap[i].primaries.Wx <= latitude) {
            return primaryMap[i].dfPrimaryEnum;
        }
    }
    /* No match */
    return KHR_DF_PRIMARIES_UNSPECIFIED;
}
