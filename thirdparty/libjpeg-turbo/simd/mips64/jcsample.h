/*
 * jcsample.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1991-1996, Thomas G. Lane.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 */

LOCAL(void)
expand_right_edge(JSAMPARRAY image_data, int num_rows, JDIMENSION input_cols,
                  JDIMENSION output_cols)
{
  register JSAMPROW ptr;
  register JSAMPLE pixval;
  register int count;
  int row;
  int numcols = (int)(output_cols - input_cols);

  if (numcols > 0) {
    for (row = 0; row < num_rows; row++) {
      ptr = image_data[row] + input_cols;
      pixval = ptr[-1];
      for (count = numcols; count > 0; count--)
        *ptr++ = pixval;
    }
  }
}
