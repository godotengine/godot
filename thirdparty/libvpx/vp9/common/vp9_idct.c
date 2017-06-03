/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <math.h>

#include "./vp9_rtcd.h"
#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"
#include "vpx_dsp/inv_txfm.h"
#include "vpx_ports/mem.h"

void vp9_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         int tx_type) {
  const transform_2d IHT_4[] = {
    { idct4_c, idct4_c  },  // DCT_DCT  = 0
    { iadst4_c, idct4_c  },   // ADST_DCT = 1
    { idct4_c, iadst4_c },    // DCT_ADST = 2
    { iadst4_c, iadst4_c }      // ADST_ADST = 3
  };

  int i, j;
  tran_low_t out[4 * 4];
  tran_low_t *outptr = out;
  tran_low_t temp_in[4], temp_out[4];

  // inverse transform row vectors
  for (i = 0; i < 4; ++i) {
    IHT_4[tx_type].rows(input, outptr);
    input  += 4;
    outptr += 4;
  }

  // inverse transform column vectors
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j)
      temp_in[j] = out[j * 4 + i];
    IHT_4[tx_type].cols(temp_in, temp_out);
    for (j = 0; j < 4; ++j) {
      dest[j * stride + i] = clip_pixel_add(dest[j * stride + i],
                                            ROUND_POWER_OF_TWO(temp_out[j], 4));
    }
  }
}

static const transform_2d IHT_8[] = {
  { idct8_c,  idct8_c  },  // DCT_DCT  = 0
  { iadst8_c, idct8_c  },  // ADST_DCT = 1
  { idct8_c,  iadst8_c },  // DCT_ADST = 2
  { iadst8_c, iadst8_c }   // ADST_ADST = 3
};

void vp9_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                         int tx_type) {
  int i, j;
  tran_low_t out[8 * 8];
  tran_low_t *outptr = out;
  tran_low_t temp_in[8], temp_out[8];
  const transform_2d ht = IHT_8[tx_type];

  // inverse transform row vectors
  for (i = 0; i < 8; ++i) {
    ht.rows(input, outptr);
    input += 8;
    outptr += 8;
  }

  // inverse transform column vectors
  for (i = 0; i < 8; ++i) {
    for (j = 0; j < 8; ++j)
      temp_in[j] = out[j * 8 + i];
    ht.cols(temp_in, temp_out);
    for (j = 0; j < 8; ++j) {
      dest[j * stride + i] = clip_pixel_add(dest[j * stride + i],
                                            ROUND_POWER_OF_TWO(temp_out[j], 5));
    }
  }
}

static const transform_2d IHT_16[] = {
  { idct16_c,  idct16_c  },  // DCT_DCT  = 0
  { iadst16_c, idct16_c  },  // ADST_DCT = 1
  { idct16_c,  iadst16_c },  // DCT_ADST = 2
  { iadst16_c, iadst16_c }   // ADST_ADST = 3
};

void vp9_iht16x16_256_add_c(const tran_low_t *input, uint8_t *dest, int stride,
                            int tx_type) {
  int i, j;
  tran_low_t out[16 * 16];
  tran_low_t *outptr = out;
  tran_low_t temp_in[16], temp_out[16];
  const transform_2d ht = IHT_16[tx_type];

  // Rows
  for (i = 0; i < 16; ++i) {
    ht.rows(input, outptr);
    input += 16;
    outptr += 16;
  }

  // Columns
  for (i = 0; i < 16; ++i) {
    for (j = 0; j < 16; ++j)
      temp_in[j] = out[j * 16 + i];
    ht.cols(temp_in, temp_out);
    for (j = 0; j < 16; ++j) {
      dest[j * stride + i] = clip_pixel_add(dest[j * stride + i],
                                            ROUND_POWER_OF_TWO(temp_out[j], 6));
    }
  }
}

// idct
void vp9_idct4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob) {
  if (eob > 1)
    vpx_idct4x4_16_add(input, dest, stride);
  else
    vpx_idct4x4_1_add(input, dest, stride);
}


void vp9_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob) {
  if (eob > 1)
    vpx_iwht4x4_16_add(input, dest, stride);
  else
    vpx_iwht4x4_1_add(input, dest, stride);
}

void vp9_idct8x8_add(const tran_low_t *input, uint8_t *dest, int stride,
                     int eob) {
  // If dc is 1, then input[0] is the reconstructed value, do not need
  // dequantization. Also, when dc is 1, dc is counted in eobs, namely eobs >=1.

  // The calculation can be simplified if there are not many non-zero dct
  // coefficients. Use eobs to decide what to do.
  // TODO(yunqingwang): "eobs = 1" case is also handled in vp9_short_idct8x8_c.
  // Combine that with code here.
  if (eob == 1)
    // DC only DCT coefficient
    vpx_idct8x8_1_add(input, dest, stride);
  else if (eob <= 12)
    vpx_idct8x8_12_add(input, dest, stride);
  else
    vpx_idct8x8_64_add(input, dest, stride);
}

void vp9_idct16x16_add(const tran_low_t *input, uint8_t *dest, int stride,
                       int eob) {
  /* The calculation can be simplified if there are not many non-zero dct
   * coefficients. Use eobs to separate different cases. */
  if (eob == 1)
    /* DC only DCT coefficient. */
    vpx_idct16x16_1_add(input, dest, stride);
  else if (eob <= 10)
    vpx_idct16x16_10_add(input, dest, stride);
  else
    vpx_idct16x16_256_add(input, dest, stride);
}

void vp9_idct32x32_add(const tran_low_t *input, uint8_t *dest, int stride,
                       int eob) {
  if (eob == 1)
    vpx_idct32x32_1_add(input, dest, stride);
  else if (eob <= 34)
    // non-zero coeff only in upper-left 8x8
    vpx_idct32x32_34_add(input, dest, stride);
  else if (eob <= 135)
    // non-zero coeff only in upper-left 16x16
    vpx_idct32x32_135_add(input, dest, stride);
  else
    vpx_idct32x32_1024_add(input, dest, stride);
}

// iht
void vp9_iht4x4_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                    int stride, int eob) {
  if (tx_type == DCT_DCT)
    vp9_idct4x4_add(input, dest, stride, eob);
  else
    vp9_iht4x4_16_add(input, dest, stride, tx_type);
}

void vp9_iht8x8_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                    int stride, int eob) {
  if (tx_type == DCT_DCT) {
    vp9_idct8x8_add(input, dest, stride, eob);
  } else {
    vp9_iht8x8_64_add(input, dest, stride, tx_type);
  }
}

void vp9_iht16x16_add(TX_TYPE tx_type, const tran_low_t *input, uint8_t *dest,
                      int stride, int eob) {
  if (tx_type == DCT_DCT) {
    vp9_idct16x16_add(input, dest, stride, eob);
  } else {
    vp9_iht16x16_256_add(input, dest, stride, tx_type);
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_highbd_iht4x4_16_add_c(const tran_low_t *input, uint8_t *dest8,
                                int stride, int tx_type, int bd) {
  const highbd_transform_2d IHT_4[] = {
    { vpx_highbd_idct4_c, vpx_highbd_idct4_c  },    // DCT_DCT  = 0
    { vpx_highbd_iadst4_c, vpx_highbd_idct4_c },    // ADST_DCT = 1
    { vpx_highbd_idct4_c, vpx_highbd_iadst4_c },    // DCT_ADST = 2
    { vpx_highbd_iadst4_c, vpx_highbd_iadst4_c }    // ADST_ADST = 3
  };
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);

  int i, j;
  tran_low_t out[4 * 4];
  tran_low_t *outptr = out;
  tran_low_t temp_in[4], temp_out[4];

  // Inverse transform row vectors.
  for (i = 0; i < 4; ++i) {
    IHT_4[tx_type].rows(input, outptr, bd);
    input  += 4;
    outptr += 4;
  }

  // Inverse transform column vectors.
  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j)
      temp_in[j] = out[j * 4 + i];
    IHT_4[tx_type].cols(temp_in, temp_out, bd);
    for (j = 0; j < 4; ++j) {
      dest[j * stride + i] = highbd_clip_pixel_add(
          dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 4), bd);
    }
  }
}

static const highbd_transform_2d HIGH_IHT_8[] = {
  { vpx_highbd_idct8_c,  vpx_highbd_idct8_c  },  // DCT_DCT  = 0
  { vpx_highbd_iadst8_c, vpx_highbd_idct8_c  },  // ADST_DCT = 1
  { vpx_highbd_idct8_c,  vpx_highbd_iadst8_c },  // DCT_ADST = 2
  { vpx_highbd_iadst8_c, vpx_highbd_iadst8_c }   // ADST_ADST = 3
};

void vp9_highbd_iht8x8_64_add_c(const tran_low_t *input, uint8_t *dest8,
                                int stride, int tx_type, int bd) {
  int i, j;
  tran_low_t out[8 * 8];
  tran_low_t *outptr = out;
  tran_low_t temp_in[8], temp_out[8];
  const highbd_transform_2d ht = HIGH_IHT_8[tx_type];
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);

  // Inverse transform row vectors.
  for (i = 0; i < 8; ++i) {
    ht.rows(input, outptr, bd);
    input += 8;
    outptr += 8;
  }

  // Inverse transform column vectors.
  for (i = 0; i < 8; ++i) {
    for (j = 0; j < 8; ++j)
      temp_in[j] = out[j * 8 + i];
    ht.cols(temp_in, temp_out, bd);
    for (j = 0; j < 8; ++j) {
      dest[j * stride + i] = highbd_clip_pixel_add(
          dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 5), bd);
    }
  }
}

static const highbd_transform_2d HIGH_IHT_16[] = {
  { vpx_highbd_idct16_c,  vpx_highbd_idct16_c  },  // DCT_DCT  = 0
  { vpx_highbd_iadst16_c, vpx_highbd_idct16_c  },  // ADST_DCT = 1
  { vpx_highbd_idct16_c,  vpx_highbd_iadst16_c },  // DCT_ADST = 2
  { vpx_highbd_iadst16_c, vpx_highbd_iadst16_c }   // ADST_ADST = 3
};

void vp9_highbd_iht16x16_256_add_c(const tran_low_t *input, uint8_t *dest8,
                                   int stride, int tx_type, int bd) {
  int i, j;
  tran_low_t out[16 * 16];
  tran_low_t *outptr = out;
  tran_low_t temp_in[16], temp_out[16];
  const highbd_transform_2d ht = HIGH_IHT_16[tx_type];
  uint16_t *dest = CONVERT_TO_SHORTPTR(dest8);

  // Rows
  for (i = 0; i < 16; ++i) {
    ht.rows(input, outptr, bd);
    input += 16;
    outptr += 16;
  }

  // Columns
  for (i = 0; i < 16; ++i) {
    for (j = 0; j < 16; ++j)
      temp_in[j] = out[j * 16 + i];
    ht.cols(temp_in, temp_out, bd);
    for (j = 0; j < 16; ++j) {
      dest[j * stride + i] = highbd_clip_pixel_add(
          dest[j * stride + i], ROUND_POWER_OF_TWO(temp_out[j], 6), bd);
    }
  }
}

// idct
void vp9_highbd_idct4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd) {
  if (eob > 1)
    vpx_highbd_idct4x4_16_add(input, dest, stride, bd);
  else
    vpx_highbd_idct4x4_1_add(input, dest, stride, bd);
}


void vp9_highbd_iwht4x4_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd) {
  if (eob > 1)
    vpx_highbd_iwht4x4_16_add(input, dest, stride, bd);
  else
    vpx_highbd_iwht4x4_1_add(input, dest, stride, bd);
}

void vp9_highbd_idct8x8_add(const tran_low_t *input, uint8_t *dest, int stride,
                            int eob, int bd) {
  // If dc is 1, then input[0] is the reconstructed value, do not need
  // dequantization. Also, when dc is 1, dc is counted in eobs, namely eobs >=1.

  // The calculation can be simplified if there are not many non-zero dct
  // coefficients. Use eobs to decide what to do.
  // TODO(yunqingwang): "eobs = 1" case is also handled in vp9_short_idct8x8_c.
  // Combine that with code here.
  // DC only DCT coefficient
  if (eob == 1) {
    vpx_highbd_idct8x8_1_add(input, dest, stride, bd);
  } else if (eob <= 10) {
    vpx_highbd_idct8x8_10_add(input, dest, stride, bd);
  } else {
    vpx_highbd_idct8x8_64_add(input, dest, stride, bd);
  }
}

void vp9_highbd_idct16x16_add(const tran_low_t *input, uint8_t *dest,
                              int stride, int eob, int bd) {
  // The calculation can be simplified if there are not many non-zero dct
  // coefficients. Use eobs to separate different cases.
  // DC only DCT coefficient.
  if (eob == 1) {
    vpx_highbd_idct16x16_1_add(input, dest, stride, bd);
  } else if (eob <= 10) {
    vpx_highbd_idct16x16_10_add(input, dest, stride, bd);
  } else {
    vpx_highbd_idct16x16_256_add(input, dest, stride, bd);
  }
}

void vp9_highbd_idct32x32_add(const tran_low_t *input, uint8_t *dest,
                              int stride, int eob, int bd) {
  // Non-zero coeff only in upper-left 8x8
  if (eob == 1) {
    vpx_highbd_idct32x32_1_add(input, dest, stride, bd);
  } else if (eob <= 34) {
    vpx_highbd_idct32x32_34_add(input, dest, stride, bd);
  } else {
    vpx_highbd_idct32x32_1024_add(input, dest, stride, bd);
  }
}

// iht
void vp9_highbd_iht4x4_add(TX_TYPE tx_type, const tran_low_t *input,
                           uint8_t *dest, int stride, int eob, int bd) {
  if (tx_type == DCT_DCT)
    vp9_highbd_idct4x4_add(input, dest, stride, eob, bd);
  else
    vp9_highbd_iht4x4_16_add(input, dest, stride, tx_type, bd);
}

void vp9_highbd_iht8x8_add(TX_TYPE tx_type, const tran_low_t *input,
                           uint8_t *dest, int stride, int eob, int bd) {
  if (tx_type == DCT_DCT) {
    vp9_highbd_idct8x8_add(input, dest, stride, eob, bd);
  } else {
    vp9_highbd_iht8x8_64_add(input, dest, stride, tx_type, bd);
  }
}

void vp9_highbd_iht16x16_add(TX_TYPE tx_type, const tran_low_t *input,
                           uint8_t *dest, int stride, int eob, int bd) {
  if (tx_type == DCT_DCT) {
    vp9_highbd_idct16x16_add(input, dest, stride, eob, bd);
  } else {
    vp9_highbd_iht16x16_256_add(input, dest, stride, tx_type, bd);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH
