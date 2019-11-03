/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include <stdio.h>
#include "blockd.h"


void vp8_print_modes_and_motion_vectors(MODE_INFO *mi, int rows, int cols, int frame)
{

    int mb_row;
    int mb_col;
    int mb_index = 0;
    FILE *mvs = fopen("mvs.stt", "a");

    /* print out the macroblock Y modes */
    mb_index = 0;
    fprintf(mvs, "Mb Modes for Frame %d\n", frame);

    for (mb_row = 0; mb_row < rows; mb_row++)
    {
        for (mb_col = 0; mb_col < cols; mb_col++)
        {

            fprintf(mvs, "%2d ", mi[mb_index].mbmi.mode);

            mb_index++;
        }

        fprintf(mvs, "\n");
        mb_index++;
    }

    fprintf(mvs, "\n");

    mb_index = 0;
    fprintf(mvs, "Mb mv ref for Frame %d\n", frame);

    for (mb_row = 0; mb_row < rows; mb_row++)
    {
        for (mb_col = 0; mb_col < cols; mb_col++)
        {

            fprintf(mvs, "%2d ", mi[mb_index].mbmi.ref_frame);

            mb_index++;
        }

        fprintf(mvs, "\n");
        mb_index++;
    }

    fprintf(mvs, "\n");

    /* print out the macroblock UV modes */
    mb_index = 0;
    fprintf(mvs, "UV Modes for Frame %d\n", frame);

    for (mb_row = 0; mb_row < rows; mb_row++)
    {
        for (mb_col = 0; mb_col < cols; mb_col++)
        {

            fprintf(mvs, "%2d ", mi[mb_index].mbmi.uv_mode);

            mb_index++;
        }

        mb_index++;
        fprintf(mvs, "\n");
    }

    fprintf(mvs, "\n");

    /* print out the block modes */
    fprintf(mvs, "Mbs for Frame %d\n", frame);
    {
        int b_row;

        for (b_row = 0; b_row < 4 * rows; b_row++)
        {
            int b_col;
            int bindex;

            for (b_col = 0; b_col < 4 * cols; b_col++)
            {
                mb_index = (b_row >> 2) * (cols + 1) + (b_col >> 2);
                bindex = (b_row & 3) * 4 + (b_col & 3);

                if (mi[mb_index].mbmi.mode == B_PRED)
                    fprintf(mvs, "%2d ", mi[mb_index].bmi[bindex].as_mode);
                else
                    fprintf(mvs, "xx ");

            }

            fprintf(mvs, "\n");
        }
    }
    fprintf(mvs, "\n");

    /* print out the macroblock mvs */
    mb_index = 0;
    fprintf(mvs, "MVs for Frame %d\n", frame);

    for (mb_row = 0; mb_row < rows; mb_row++)
    {
        for (mb_col = 0; mb_col < cols; mb_col++)
        {
            fprintf(mvs, "%5d:%-5d", mi[mb_index].mbmi.mv.as_mv.row / 2, mi[mb_index].mbmi.mv.as_mv.col / 2);

            mb_index++;
        }

        mb_index++;
        fprintf(mvs, "\n");
    }

    fprintf(mvs, "\n");


    /* print out the block modes */
    fprintf(mvs, "MVs for Frame %d\n", frame);
    {
        int b_row;

        for (b_row = 0; b_row < 4 * rows; b_row++)
        {
            int b_col;
            int bindex;

            for (b_col = 0; b_col < 4 * cols; b_col++)
            {
                mb_index = (b_row >> 2) * (cols + 1) + (b_col >> 2);
                bindex = (b_row & 3) * 4 + (b_col & 3);
                fprintf(mvs, "%3d:%-3d ", mi[mb_index].bmi[bindex].mv.as_mv.row, mi[mb_index].bmi[bindex].mv.as_mv.col);

            }

            fprintf(mvs, "\n");
        }
    }
    fprintf(mvs, "\n");


    fclose(mvs);
}
