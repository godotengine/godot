#ifndef TREES_H_
#define TREES_H_

/* Constants */

#define DIST_CODE_LEN  512
/* see definition of array dist_code in trees.c */

#define MAX_BL_BITS 7
/* Bit length codes must not exceed MAX_BL_BITS bits */

#define REP_3_6      16
/* repeat previous bit length 3-6 times (2 bits of repeat count) */

#define REPZ_3_10    17
/* repeat a zero length 3-10 times  (3 bits of repeat count) */

#define REPZ_11_138  18
/* repeat a zero length 11-138 times  (7 bits of repeat count) */

static const int extra_lbits[LENGTH_CODES] /* extra bits for each length code */
    = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};

static const int extra_dbits[D_CODES] /* extra bits for each distance code */
    = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

static const int extra_blbits[BL_CODES] /* extra bits for each bit length code */
    = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7};

static const unsigned char bl_order[BL_CODES]
    = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
 /* The lengths of the bit length codes are sent in order of decreasing
  * probability, to avoid transmitting the lengths for unused bit length codes.
  */


/* Function definitions */
void gen_codes        (ct_data *tree, int max_code, uint16_t *bl_count);

#endif
