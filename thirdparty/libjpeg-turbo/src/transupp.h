/*
 * transupp.h
 *
 * This file was part of the Independent JPEG Group's software:
 * Copyright (C) 1997-2019, Thomas G. Lane, Guido Vollbeding.
 * libjpeg-turbo Modifications:
 * Copyright (C) 2017, 2021, D. R. Commander.
 * For conditions of distribution and use, see the accompanying README.ijg
 * file.
 *
 * This file contains declarations for image transformation routines and
 * other utility code used by the jpegtran sample application.  These are
 * NOT part of the core JPEG library.  But we keep these routines separate
 * from jpegtran.c to ease the task of maintaining jpegtran-like programs
 * that have other user interfaces.
 *
 * NOTE: all the routines declared here have very specific requirements
 * about when they are to be executed during the reading and writing of the
 * source and destination files.  See the comments in transupp.c, or see
 * jpegtran.c for an example of correct usage.
 */

/* If you happen not to want the image transform support, disable it here */
#ifndef TRANSFORMS_SUPPORTED
#define TRANSFORMS_SUPPORTED  1         /* 0 disables transform code */
#endif

/*
 * Although rotating and flipping data expressed as DCT coefficients is not
 * hard, there is an asymmetry in the JPEG format specification for images
 * whose dimensions aren't multiples of the iMCU size.  The right and bottom
 * image edges are padded out to the next iMCU boundary with junk data; but
 * no padding is possible at the top and left edges.  If we were to flip
 * the whole image including the pad data, then pad garbage would become
 * visible at the top and/or left, and real pixels would disappear into the
 * pad margins --- perhaps permanently, since encoders & decoders may not
 * bother to preserve DCT blocks that appear to be completely outside the
 * nominal image area.  So, we have to exclude any partial iMCUs from the
 * basic transformation.
 *
 * Transpose is the only transformation that can handle partial iMCUs at the
 * right and bottom edges completely cleanly.  flip_h can flip partial iMCUs
 * at the bottom, but leaves any partial iMCUs at the right edge untouched.
 * Similarly flip_v leaves any partial iMCUs at the bottom edge untouched.
 * The other transforms are defined as combinations of these basic transforms
 * and process edge blocks in a way that preserves the equivalence.
 *
 * The "trim" option causes untransformable partial iMCUs to be dropped;
 * this is not strictly lossless, but it usually gives the best-looking
 * result for odd-size images.  Note that when this option is active,
 * the expected mathematical equivalences between the transforms may not hold.
 * (For example, -rot 270 -trim trims only the bottom edge, but -rot 90 -trim
 * followed by -rot 180 -trim trims both edges.)
 *
 * We also offer a lossless-crop option, which discards data outside a given
 * image region but losslessly preserves what is inside.  Like the rotate and
 * flip transforms, lossless crop is restricted by the JPEG format: the upper
 * left corner of the selected region must fall on an iMCU boundary.  If this
 * does not hold for the given crop parameters, we silently move the upper left
 * corner up and/or left to make it so, simultaneously increasing the region
 * dimensions to keep the lower right crop corner unchanged.  (Thus, the
 * output image covers at least the requested region, but may cover more.)
 * The adjustment of the region dimensions may be optionally disabled.
 *
 * A complementary lossless wipe option is provided to discard (gray out) data
 * inside a given image region while losslessly preserving what is outside.
 * A lossless drop option is also provided, which allows another JPEG image to
 * be inserted ("dropped") into the source image data at a given position,
 * replacing the existing image data at that position.  Both the source image
 * and the drop image must have the same subsampling level.  It is best if they
 * also have the same quantization (quality.)  Otherwise, the quantization of
 * the output image will be adapted to accommodate the higher of the source
 * image quality and the drop image quality.  The trim option can be used with
 * the drop option to requantize the drop image to match the source image.
 *
 * We also provide a lossless-resize option, which is kind of a lossless-crop
 * operation in the DCT coefficient block domain - it discards higher-order
 * coefficients and losslessly preserves lower-order coefficients of a
 * sub-block.
 *
 * Rotate/flip transform, resize, and crop can be requested together in a
 * single invocation.  The crop is applied last --- that is, the crop region
 * is specified in terms of the destination image after transform/resize.
 *
 * We also offer a "force to grayscale" option, which simply discards the
 * chrominance channels of a YCbCr image.  This is lossless in the sense that
 * the luminance channel is preserved exactly.  It's not the same kind of
 * thing as the rotate/flip transformations, but it's convenient to handle it
 * as part of this package, mainly because the transformation routines have to
 * be aware of the option to know how many components to work on.
 */


/*
 * Codes for supported types of image transformations.
 */

typedef enum {
  JXFORM_NONE,            /* no transformation */
  JXFORM_FLIP_H,          /* horizontal flip */
  JXFORM_FLIP_V,          /* vertical flip */
  JXFORM_TRANSPOSE,       /* transpose across UL-to-LR axis */
  JXFORM_TRANSVERSE,      /* transpose across UR-to-LL axis */
  JXFORM_ROT_90,          /* 90-degree clockwise rotation */
  JXFORM_ROT_180,         /* 180-degree rotation */
  JXFORM_ROT_270,         /* 270-degree clockwise (or 90 ccw) */
  JXFORM_WIPE,            /* wipe */
  JXFORM_DROP             /* drop */
} JXFORM_CODE;

/*
 * Codes for crop parameters, which can individually be unspecified,
 * positive or negative for xoffset or yoffset,
 * positive or force or reflect for width or height.
 */

typedef enum {
  JCROP_UNSET,
  JCROP_POS,
  JCROP_NEG,
  JCROP_FORCE,
  JCROP_REFLECT
} JCROP_CODE;

/*
 * Transform parameters struct.
 * NB: application must not change any elements of this struct after
 * calling jtransform_request_workspace.
 */

typedef struct {
  /* Options: set by caller */
  JXFORM_CODE transform;        /* image transform operator */
  boolean perfect;              /* if TRUE, fail if partial MCUs are requested */
  boolean trim;                 /* if TRUE, trim partial MCUs as needed */
  boolean force_grayscale;      /* if TRUE, convert color image to grayscale */
  boolean crop;                 /* if TRUE, crop or wipe source image, or drop */
  boolean slow_hflip;  /* For best performance, the JXFORM_FLIP_H transform
                          normally modifies the source coefficients in place.
                          Setting this to TRUE will instead use a slower,
                          double-buffered algorithm, which leaves the source
                          coefficients in tact (necessary if other transformed
                          images must be generated from the same set of
                          coefficients. */

  /* Crop parameters: application need not set these unless crop is TRUE.
   * These can be filled in by jtransform_parse_crop_spec().
   */
  JDIMENSION crop_width;        /* Width of selected region */
  JCROP_CODE crop_width_set;    /* (force-disables adjustment) */
  JDIMENSION crop_height;       /* Height of selected region */
  JCROP_CODE crop_height_set;   /* (force-disables adjustment) */
  JDIMENSION crop_xoffset;      /* X offset of selected region */
  JCROP_CODE crop_xoffset_set;  /* (negative measures from right edge) */
  JDIMENSION crop_yoffset;      /* Y offset of selected region */
  JCROP_CODE crop_yoffset_set;  /* (negative measures from bottom edge) */

  /* Drop parameters: set by caller for drop request */
  j_decompress_ptr drop_ptr;
  jvirt_barray_ptr *drop_coef_arrays;

  /* Internal workspace: caller should not touch these */
  int num_components;           /* # of components in workspace */
  jvirt_barray_ptr *workspace_coef_arrays; /* workspace for transformations */
  JDIMENSION output_width;      /* cropped destination dimensions */
  JDIMENSION output_height;
  JDIMENSION x_crop_offset;     /* destination crop offsets measured in iMCUs */
  JDIMENSION y_crop_offset;
  JDIMENSION drop_width;        /* drop/wipe dimensions measured in iMCUs */
  JDIMENSION drop_height;
  int iMCU_sample_width;        /* destination iMCU size */
  int iMCU_sample_height;
} jpeg_transform_info;


#if TRANSFORMS_SUPPORTED

/* Parse a crop specification (written in X11 geometry style) */
EXTERN(boolean) jtransform_parse_crop_spec(jpeg_transform_info *info,
                                           const char *spec);
/* Request any required workspace */
EXTERN(boolean) jtransform_request_workspace(j_decompress_ptr srcinfo,
                                             jpeg_transform_info *info);
/* Adjust output image parameters */
EXTERN(jvirt_barray_ptr *) jtransform_adjust_parameters
  (j_decompress_ptr srcinfo, j_compress_ptr dstinfo,
   jvirt_barray_ptr *src_coef_arrays, jpeg_transform_info *info);
/* Execute the actual transformation, if any */
EXTERN(void) jtransform_execute_transform(j_decompress_ptr srcinfo,
                                          j_compress_ptr dstinfo,
                                          jvirt_barray_ptr *src_coef_arrays,
                                          jpeg_transform_info *info);
/* Determine whether lossless transformation is perfectly
 * possible for a specified image and transformation.
 */
EXTERN(boolean) jtransform_perfect_transform(JDIMENSION image_width,
                                             JDIMENSION image_height,
                                             int MCU_width, int MCU_height,
                                             JXFORM_CODE transform);

/* jtransform_execute_transform used to be called
 * jtransform_execute_transformation, but some compilers complain about
 * routine names that long.  This macro is here to avoid breaking any
 * old source code that uses the original name...
 */
#define jtransform_execute_transformation       jtransform_execute_transform

#endif /* TRANSFORMS_SUPPORTED */


/*
 * Support for copying optional markers from source to destination file.
 */

typedef enum {
  JCOPYOPT_NONE,           /* copy no optional markers */
  JCOPYOPT_COMMENTS,       /* copy only comment (COM) markers */
  JCOPYOPT_ALL,            /* copy all optional markers */
  JCOPYOPT_ALL_EXCEPT_ICC, /* copy all optional markers except APP2 */
  JCOPYOPT_ICC             /* copy only ICC profile (APP2) markers */
} JCOPY_OPTION;

#define JCOPYOPT_DEFAULT  JCOPYOPT_COMMENTS     /* recommended default */

/* Setup decompression object to save desired markers in memory */
EXTERN(void) jcopy_markers_setup(j_decompress_ptr srcinfo,
                                 JCOPY_OPTION option);
/* Copy markers saved in the given source object to the destination object */
EXTERN(void) jcopy_markers_execute(j_decompress_ptr srcinfo,
                                   j_compress_ptr dstinfo,
                                   JCOPY_OPTION option);
