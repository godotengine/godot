/*
 * Copyright (C) 2011 Red Hat Inc.
 *
 * block compression parts are:
 * Copyright (C) 2004  Roland Scheidegger   All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Author:
 *    Dave Airlie
 */

/* included by texcompress_rgtc to define byte/ubyte compressors */

void TAG(fetch_texel_rgtc)(unsigned srcRowStride, const TYPE *pixdata,
	                   unsigned i, unsigned j, TYPE *value, unsigned comps)
{
   TYPE decode;
   const TYPE *blksrc = (pixdata + ((srcRowStride + 3) / 4 * (j / 4) + (i / 4)) * 8 * comps);
   const TYPE alpha0 = blksrc[0];
   const TYPE alpha1 = blksrc[1];
   const char bit_pos = ((j&3) * 4 + (i&3)) * 3;
   const unsigned char acodelow = blksrc[2 + bit_pos / 8];
   const unsigned char acodehigh = (3 + bit_pos / 8) < 8 ? blksrc[3 + bit_pos / 8] : 0;
   const unsigned char code = (acodelow >> (bit_pos & 0x7) |
      (acodehigh  << (8 - (bit_pos & 0x7)))) & 0x7;

   if (code == 0)
      decode = alpha0;
   else if (code == 1)
      decode = alpha1;
   else if (alpha0 > alpha1)
      decode = ((alpha0 * (8 - code) + (alpha1 * (code - 1))) / 7);
   else if (code < 6)
      decode = ((alpha0 * (6 - code) + (alpha1 * (code - 1))) / 5);
   else if (code == 6)
      decode = T_MIN;
   else
      decode = T_MAX;

   *value = decode;
}

static void TAG(write_rgtc_encoded_channel)(TYPE *blkaddr,
                                            TYPE alphabase1,
                                            TYPE alphabase2,
                                            TYPE alphaenc[16])
{
   *blkaddr++ = alphabase1;
   *blkaddr++ = alphabase2;
   *blkaddr++ = alphaenc[0] | (alphaenc[1] << 3) | ((alphaenc[2] & 3) << 6);
   *blkaddr++ = (alphaenc[2] >> 2) | (alphaenc[3] << 1) | (alphaenc[4] << 4) | ((alphaenc[5] & 1) << 7);
   *blkaddr++ = (alphaenc[5] >> 1) | (alphaenc[6] << 2) | (alphaenc[7] << 5);
   *blkaddr++ = alphaenc[8] | (alphaenc[9] << 3) | ((alphaenc[10] & 3) << 6);
   *blkaddr++ = (alphaenc[10] >> 2) | (alphaenc[11] << 1) | (alphaenc[12] << 4) | ((alphaenc[13] & 1) << 7);
   *blkaddr++ = (alphaenc[13] >> 1) | (alphaenc[14] << 2) | (alphaenc[15] << 5);
}

void TAG(encode_rgtc_ubyte)(TYPE *blkaddr, TYPE srccolors[4][4],
                            int numxpixels, int numypixels)
{
   TYPE alphabase[2], alphause[2];
   short alphatest[2] = { 0 };
   unsigned int alphablockerror1, alphablockerror2, alphablockerror3;
   TYPE i, j, aindex, acutValues[7];
   TYPE alphaenc1[16], alphaenc2[16], alphaenc3[16];
   int alphaabsmin = 0, alphaabsmax = 0;
   short alphadist;

   /* find lowest and highest alpha value in block, alphabase[0] lowest, alphabase[1] highest */
   alphabase[0] = T_MAX; alphabase[1] = T_MIN;
   for (j = 0; j < numypixels; j++) {
      for (i = 0; i < numxpixels; i++) {
	 if (srccolors[j][i] == T_MIN)
            alphaabsmin = 1;
         else if (srccolors[j][i] == T_MAX)
            alphaabsmax = 1;
         else {
            if (srccolors[j][i] > alphabase[1])
               alphabase[1] = srccolors[j][i];
            if (srccolors[j][i] < alphabase[0])
               alphabase[0] = srccolors[j][i];
         }
      }
   }


   if (((alphabase[0] > alphabase[1]) && !(alphaabsmin && alphaabsmax))
       || (alphabase[0] == alphabase[1] && !alphaabsmin && !alphaabsmax)) { /* one color, either max or min */
      /* shortcut here since it is a very common case (and also avoids later problems) */
      /* could also thest for alpha0 == alpha1 (and not min/max), but probably not common, so don't bother */

      *blkaddr++ = srccolors[0][0];
      blkaddr++;
      *blkaddr++ = 0;
      *blkaddr++ = 0;
      *blkaddr++ = 0;
      *blkaddr++ = 0;
      *blkaddr++ = 0;
      *blkaddr++ = 0;
#if RGTC_DEBUG
      fprintf(stderr, "enc0 used\n");
#endif
      return;
   }

   /* find best encoding for alpha0 > alpha1 */
   /* it's possible this encoding is better even if both alphaabsmin and alphaabsmax are true */
   alphablockerror1 = 0x0;
   alphablockerror2 = 0xffffffff;
   alphablockerror3 = 0xffffffff;
   if (alphaabsmin) alphause[0] = T_MIN;
   else alphause[0] = alphabase[0];
   if (alphaabsmax) alphause[1] = T_MAX;
   else alphause[1] = alphabase[1];
   /* calculate the 7 cut values, just the middle between 2 of the computed alpha values */
   for (aindex = 0; aindex < 7; aindex++) {
      /* don't forget here is always rounded down */
      acutValues[aindex] = (alphause[0] * (2*aindex + 1) + alphause[1] * (14 - (2*aindex + 1))) / 14;
   }

   for (j = 0; j < numypixels; j++) {
      for (i = 0; i < numxpixels; i++) {
         /* maybe it's overkill to have the most complicated calculation just for the error
            calculation which we only need to figure out if encoding1 or encoding2 is better... */
         if (srccolors[j][i] > acutValues[0]) {
            alphaenc1[4*j + i] = 0;
            alphadist = srccolors[j][i] - alphause[1];
         }
         else if (srccolors[j][i] > acutValues[1]) {
            alphaenc1[4*j + i] = 2;
            alphadist = srccolors[j][i] - (alphause[1] * 6 + alphause[0] * 1) / 7;
         }
         else if (srccolors[j][i] > acutValues[2]) {
            alphaenc1[4*j + i] = 3;
            alphadist = srccolors[j][i] - (alphause[1] * 5 + alphause[0] * 2) / 7;
         }
         else if (srccolors[j][i] > acutValues[3]) {
            alphaenc1[4*j + i] = 4;
            alphadist = srccolors[j][i] - (alphause[1] * 4 + alphause[0] * 3) / 7;
         }
         else if (srccolors[j][i] > acutValues[4]) {
            alphaenc1[4*j + i] = 5;
            alphadist = srccolors[j][i] - (alphause[1] * 3 + alphause[0] * 4) / 7;
         }
         else if (srccolors[j][i] > acutValues[5]) {
            alphaenc1[4*j + i] = 6;
            alphadist = srccolors[j][i] - (alphause[1] * 2 + alphause[0] * 5) / 7;
         }
         else if (srccolors[j][i] > acutValues[6]) {
            alphaenc1[4*j + i] = 7;
            alphadist = srccolors[j][i] - (alphause[1] * 1 + alphause[0] * 6) / 7;
         }
         else {
            alphaenc1[4*j + i] = 1;
            alphadist = srccolors[j][i] - alphause[0];
         }
         alphablockerror1 += alphadist * alphadist;
      }
   }

#if RGTC_DEBUG
   for (i = 0; i < 16; i++) {
      fprintf(stderr, "%d ", alphaenc1[i]);
   }
   fprintf(stderr, "cutVals ");
   for (i = 0; i < 7; i++) {
      fprintf(stderr, "%d ", acutValues[i]);
   }
   fprintf(stderr, "srcVals ");
   for (j = 0; j < numypixels; j++) {
      for (i = 0; i < numxpixels; i++) {
	 fprintf(stderr, "%d ", srccolors[j][i]);
      }
   }
   fprintf(stderr, "\n");
#endif

   /* it's not very likely this encoding is better if both alphaabsmin and alphaabsmax
      are false but try it anyway */
   if (alphablockerror1 >= 32) {

      /* don't bother if encoding is already very good, this condition should also imply
      we have valid alphabase colors which we absolutely need (alphabase[0] <= alphabase[1]) */
      alphablockerror2 = 0;
      for (aindex = 0; aindex < 5; aindex++) {
         /* don't forget here is always rounded down */
         acutValues[aindex] = (alphabase[0] * (10 - (2*aindex + 1)) + alphabase[1] * (2*aindex + 1)) / 10;
      }
      for (j = 0; j < numypixels; j++) {
         for (i = 0; i < numxpixels; i++) {
             /* maybe it's overkill to have the most complicated calculation just for the error
               calculation which we only need to figure out if encoding1 or encoding2 is better... */
            if (srccolors[j][i] == T_MIN) {
               alphaenc2[4*j + i] = 6;
               alphadist = 0;
            }
            else if (srccolors[j][i] == T_MAX) {
               alphaenc2[4*j + i] = 7;
               alphadist = 0;
            }
            else if (srccolors[j][i] <= acutValues[0]) {
               alphaenc2[4*j + i] = 0;
               alphadist = srccolors[j][i] - alphabase[0];
            }
            else if (srccolors[j][i] <= acutValues[1]) {
               alphaenc2[4*j + i] = 2;
               alphadist = srccolors[j][i] - (alphabase[0] * 4 + alphabase[1] * 1) / 5;
            }
            else if (srccolors[j][i] <= acutValues[2]) {
               alphaenc2[4*j + i] = 3;
               alphadist = srccolors[j][i] - (alphabase[0] * 3 + alphabase[1] * 2) / 5;
            }
            else if (srccolors[j][i] <= acutValues[3]) {
               alphaenc2[4*j + i] = 4;
               alphadist = srccolors[j][i] - (alphabase[0] * 2 + alphabase[1] * 3) / 5;
            }
            else if (srccolors[j][i] <= acutValues[4]) {
               alphaenc2[4*j + i] = 5;
               alphadist = srccolors[j][i] - (alphabase[0] * 1 + alphabase[1] * 4) / 5;
            }
            else {
               alphaenc2[4*j + i] = 1;
               alphadist = srccolors[j][i] - alphabase[1];
            }
            alphablockerror2 += alphadist * alphadist;
         }
      }


      /* skip this if the error is already very small
         this encoding is MUCH better on average than #2 though, but expensive! */
      if ((alphablockerror2 > 96) && (alphablockerror1 > 96)) {
         short blockerrlin1 = 0;
         short blockerrlin2 = 0;
         TYPE nralphainrangelow = 0;
         TYPE nralphainrangehigh = 0;
         alphatest[0] = T_MAX;
         alphatest[1] = T_MIN;
         /* if we have large range it's likely there are values close to 0/255, try to map them to 0/255 */
         for (j = 0; j < numypixels; j++) {
            for (i = 0; i < numxpixels; i++) {
               if ((srccolors[j][i] > alphatest[1]) && (srccolors[j][i] < (T_MAX -(alphabase[1] - alphabase[0]) / 28)))
                  alphatest[1] = srccolors[j][i];
               if ((srccolors[j][i] < alphatest[0]) && (srccolors[j][i] > (alphabase[1] - alphabase[0]) / 28))
                  alphatest[0] = srccolors[j][i];
            }
         }
          /* shouldn't happen too often, don't really care about those degenerated cases */
          if (alphatest[1] <= alphatest[0]) {
             alphatest[0] = T_MIN+1;
             alphatest[1] = T_MAX-1;
         }
         for (aindex = 0; aindex < 5; aindex++) {
         /* don't forget here is always rounded down */
            acutValues[aindex] = (alphatest[0] * (10 - (2*aindex + 1)) + alphatest[1] * (2*aindex + 1)) / 10;
         }

         /* find the "average" difference between the alpha values and the next encoded value.
            This is then used to calculate new base values.
            Should there be some weighting, i.e. those values closer to alphatest[x] have more weight,
            since they will see more improvement, and also because the values in the middle are somewhat
            likely to get no improvement at all (because the base values might move in different directions)?
            OTOH it would mean the values in the middle are even less likely to get an improvement
         */
         for (j = 0; j < numypixels; j++) {
            for (i = 0; i < numxpixels; i++) {
               if (srccolors[j][i] <= alphatest[0] / 2) {
               }
               else if (srccolors[j][i] > ((T_MAX + alphatest[1]) / 2)) {
               }
               else if (srccolors[j][i] <= acutValues[0]) {
                  blockerrlin1 += (srccolors[j][i] - alphatest[0]);
                  nralphainrangelow += 1;
               }
               else if (srccolors[j][i] <= acutValues[1]) {
                  blockerrlin1 += (srccolors[j][i] - (alphatest[0] * 4 + alphatest[1] * 1) / 5);
                  blockerrlin2 += (srccolors[j][i] - (alphatest[0] * 4 + alphatest[1] * 1) / 5);
                  nralphainrangelow += 1;
                  nralphainrangehigh += 1;
               }
               else if (srccolors[j][i] <= acutValues[2]) {
                  blockerrlin1 += (srccolors[j][i] - (alphatest[0] * 3 + alphatest[1] * 2) / 5);
                  blockerrlin2 += (srccolors[j][i] - (alphatest[0] * 3 + alphatest[1] * 2) / 5);
                  nralphainrangelow += 1;
                  nralphainrangehigh += 1;
               }
               else if (srccolors[j][i] <= acutValues[3]) {
                  blockerrlin1 += (srccolors[j][i] - (alphatest[0] * 2 + alphatest[1] * 3) / 5);
                  blockerrlin2 += (srccolors[j][i] - (alphatest[0] * 2 + alphatest[1] * 3) / 5);
                  nralphainrangelow += 1;
                  nralphainrangehigh += 1;
               }
               else if (srccolors[j][i] <= acutValues[4]) {
                  blockerrlin1 += (srccolors[j][i] - (alphatest[0] * 1 + alphatest[1] * 4) / 5);
                  blockerrlin2 += (srccolors[j][i] - (alphatest[0] * 1 + alphatest[1] * 4) / 5);
                  nralphainrangelow += 1;
                  nralphainrangehigh += 1;
                  }
               else {
                  blockerrlin2 += (srccolors[j][i] - alphatest[1]);
                  nralphainrangehigh += 1;
               }
            }
         }
         /* shouldn't happen often, needed to avoid div by zero */
         if (nralphainrangelow == 0) nralphainrangelow = 1;
         if (nralphainrangehigh == 0) nralphainrangehigh = 1;
         alphatest[0] = alphatest[0] + (blockerrlin1 / nralphainrangelow);
#if RGTC_DEBUG
         fprintf(stderr, "block err lin low %d, nr %d\n", blockerrlin1, nralphainrangelow);
         fprintf(stderr, "block err lin high %d, nr %d\n", blockerrlin2, nralphainrangehigh);
#endif
         /* again shouldn't really happen often... */
         if (alphatest[0] < T_MIN) {
            alphatest[0] = T_MIN;
         }
         alphatest[1] = alphatest[1] + (blockerrlin2 / nralphainrangehigh);
         if (alphatest[1] > T_MAX) {
            alphatest[1] = T_MAX;
         }

         alphablockerror3 = 0;
         for (aindex = 0; aindex < 5; aindex++) {
         /* don't forget here is always rounded down */
            acutValues[aindex] = (alphatest[0] * (10 - (2*aindex + 1)) + alphatest[1] * (2*aindex + 1)) / 10;
         }
         for (j = 0; j < numypixels; j++) {
            for (i = 0; i < numxpixels; i++) {
                /* maybe it's overkill to have the most complicated calculation just for the error
                  calculation which we only need to figure out if encoding1 or encoding2 is better... */
               if (srccolors[j][i] <= alphatest[0] / 2) {
                  alphaenc3[4*j + i] = 6;
                  alphadist = srccolors[j][i];
               }
               else if (srccolors[j][i] > ((T_MAX + alphatest[1]) / 2)) {
                  alphaenc3[4*j + i] = 7;
                  alphadist = T_MAX - srccolors[j][i];
               }
               else if (srccolors[j][i] <= acutValues[0]) {
                  alphaenc3[4*j + i] = 0;
                  alphadist = srccolors[j][i] - alphatest[0];
               }
               else if (srccolors[j][i] <= acutValues[1]) {
                 alphaenc3[4*j + i] = 2;
                 alphadist = srccolors[j][i] - (alphatest[0] * 4 + alphatest[1] * 1) / 5;
               }
               else if (srccolors[j][i] <= acutValues[2]) {
                  alphaenc3[4*j + i] = 3;
                  alphadist = srccolors[j][i] - (alphatest[0] * 3 + alphatest[1] * 2) / 5;
               }
               else if (srccolors[j][i] <= acutValues[3]) {
                  alphaenc3[4*j + i] = 4;
                  alphadist = srccolors[j][i] - (alphatest[0] * 2 + alphatest[1] * 3) / 5;
               }
               else if (srccolors[j][i] <= acutValues[4]) {
                  alphaenc3[4*j + i] = 5;
                  alphadist = srccolors[j][i] - (alphatest[0] * 1 + alphatest[1] * 4) / 5;
               }
               else {
                  alphaenc3[4*j + i] = 1;
                  alphadist = srccolors[j][i] - alphatest[1];
               }
               alphablockerror3 += alphadist * alphadist;
            }
         }
      }
   }

  /* write the alpha values and encoding back. */
   if ((alphablockerror1 <= alphablockerror2) && (alphablockerror1 <= alphablockerror3)) {
#if RGTC_DEBUG
      if (alphablockerror1 > 96) fprintf(stderr, "enc1 used, error %d\n", alphablockerror1);
      fprintf(stderr,"w1: min %d max %d au0 %d au1 %d\n",
	      T_MIN, T_MAX,
	      alphause[1], alphause[0]);
#endif

      TAG(write_rgtc_encoded_channel)( blkaddr, alphause[1], alphause[0], alphaenc1 );
   }
   else if (alphablockerror2 <= alphablockerror3) {
#if RGTC_DEBUG
      if (alphablockerror2 > 96) fprintf(stderr, "enc2 used, error %d\n", alphablockerror2);
      fprintf(stderr,"w2: min %d max %d au0 %d au1 %d\n",
	      T_MIN, T_MAX,
	      alphabase[0], alphabase[1]);
#endif

      TAG(write_rgtc_encoded_channel)( blkaddr, alphabase[0], alphabase[1], alphaenc2 );
   }
   else {
#if RGTC_DEBUG
      fprintf(stderr, "enc3 used, error %d\n", alphablockerror3);
      fprintf(stderr,"w3: min %d max %d au0 %d au1 %d\n",
	      T_MIN, T_MAX,
	      alphatest[0], alphatest[1]);
#endif

      TAG(write_rgtc_encoded_channel)( blkaddr, (TYPE)alphatest[0], (TYPE)alphatest[1], alphaenc3 );
   }
}
