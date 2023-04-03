#ifndef __GL_CONFIG_H__
#define __GL_CONFIG_H__

#include "util/glheader.h"

/**
 * Framebuffer configuration (aka visual / pixelformat)
 * Note: some of these fields should be boolean, but it appears that
 * code in drivers/dri/common/util.c requires int-sized fields.
 */
struct gl_config
{
   GLboolean floatMode;
   GLuint doubleBufferMode;
   GLuint stereoMode;

   GLint redBits, greenBits, blueBits, alphaBits;	/* bits per comp */
   GLuint redMask, greenMask, blueMask, alphaMask;
   GLint redShift, greenShift, blueShift, alphaShift;
   GLint rgbBits;		/* total bits for rgb */

   GLint accumRedBits, accumGreenBits, accumBlueBits, accumAlphaBits;
   GLint depthBits;
   GLint stencilBits;

   /* ARB_multisample / SGIS_multisample */
   GLuint samples;

   /* OML_swap_method */
   GLint swapMethod;

   /* EXT_framebuffer_sRGB */
   GLint sRGBCapable;
};


#endif
