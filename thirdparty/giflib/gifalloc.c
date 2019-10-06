/*****************************************************************************

 GIF construction tools

SPDX-License-Identifier: MIT

****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gif_lib.h"
#include "gif_lib_private.h"

#define MAX(x, y)    (((x) > (y)) ? (x) : (y))

/******************************************************************************
 Miscellaneous utility functions                          
******************************************************************************/

/* return smallest bitfield size n will fit in */
int
GifBitSize(int n)
{
    register int i;

    for (i = 1; i <= 8; i++)
        if ((1 << i) >= n)
            break;
    return (i);
}

/******************************************************************************
  Color map object functions                              
******************************************************************************/

/*
 * Allocate a color map of given size; initialize with contents of
 * ColorMap if that pointer is non-NULL.
 */
ColorMapObject *
GifMakeMapObject(int ColorCount, const GifColorType *ColorMap)
{
    ColorMapObject *Object;

    /*** FIXME: Our ColorCount has to be a power of two.  Is it necessary to
     * make the user know that or should we automatically round up instead? */
    if (ColorCount != (1 << GifBitSize(ColorCount))) {
        return ((ColorMapObject *) NULL);
    }
    
    Object = (ColorMapObject *)malloc(sizeof(ColorMapObject));
    if (Object == (ColorMapObject *) NULL) {
        return ((ColorMapObject *) NULL);
    }

    Object->Colors = (GifColorType *)calloc(ColorCount, sizeof(GifColorType));
    if (Object->Colors == (GifColorType *) NULL) {
	free(Object);
        return ((ColorMapObject *) NULL);
    }

    Object->ColorCount = ColorCount;
    Object->BitsPerPixel = GifBitSize(ColorCount);
    Object->SortFlag = false;

    if (ColorMap != NULL) {
        memcpy((char *)Object->Colors,
               (char *)ColorMap, ColorCount * sizeof(GifColorType));
    }

    return (Object);
}

/*******************************************************************************
Free a color map object
*******************************************************************************/
void
GifFreeMapObject(ColorMapObject *Object)
{
    if (Object != NULL) {
        (void)free(Object->Colors);
        (void)free(Object);
    }
}

#ifdef DEBUG
void
DumpColorMap(ColorMapObject *Object,
             FILE * fp)
{
    if (Object != NULL) {
        int i, j, Len = Object->ColorCount;

        for (i = 0; i < Len; i += 4) {
            for (j = 0; j < 4 && j < Len; j++) {
                (void)fprintf(fp, "%3d: %02x %02x %02x   ", i + j,
			      Object->Colors[i + j].Red,
			      Object->Colors[i + j].Green,
			      Object->Colors[i + j].Blue);
            }
            (void)fprintf(fp, "\n");
        }
    }
}
#endif /* DEBUG */

/*******************************************************************************
 Compute the union of two given color maps and return it.  If result can't 
 fit into 256 colors, NULL is returned, the allocated union otherwise.
 ColorIn1 is copied as is to ColorUnion, while colors from ColorIn2 are
 copied iff they didn't exist before.  ColorTransIn2 maps the old
 ColorIn2 into the ColorUnion color map table./
*******************************************************************************/
ColorMapObject *
GifUnionColorMap(const ColorMapObject *ColorIn1,
              const ColorMapObject *ColorIn2,
              GifPixelType ColorTransIn2[])
{
    int i, j, CrntSlot, RoundUpTo, NewGifBitSize;
    ColorMapObject *ColorUnion;

    /*
     * We don't worry about duplicates within either color map; if
     * the caller wants to resolve those, he can perform unions
     * with an empty color map.
     */

    /* Allocate table which will hold the result for sure. */
    ColorUnion = GifMakeMapObject(MAX(ColorIn1->ColorCount,
                               ColorIn2->ColorCount) * 2, NULL);

    if (ColorUnion == NULL)
        return (NULL);

    /* 
     * Copy ColorIn1 to ColorUnion.
     */
    for (i = 0; i < ColorIn1->ColorCount; i++)
        ColorUnion->Colors[i] = ColorIn1->Colors[i];
    CrntSlot = ColorIn1->ColorCount;

    /* 
     * Potentially obnoxious hack:
     *
     * Back CrntSlot down past all contiguous {0, 0, 0} slots at the end
     * of table 1.  This is very useful if your display is limited to
     * 16 colors.
     */
    while (ColorIn1->Colors[CrntSlot - 1].Red == 0
           && ColorIn1->Colors[CrntSlot - 1].Green == 0
           && ColorIn1->Colors[CrntSlot - 1].Blue == 0)
        CrntSlot--;

    /* Copy ColorIn2 to ColorUnion (use old colors if they exist): */
    for (i = 0; i < ColorIn2->ColorCount && CrntSlot <= 256; i++) {
        /* Let's see if this color already exists: */
        for (j = 0; j < ColorIn1->ColorCount; j++)
            if (memcmp (&ColorIn1->Colors[j], &ColorIn2->Colors[i], 
                        sizeof(GifColorType)) == 0)
                break;

        if (j < ColorIn1->ColorCount)
            ColorTransIn2[i] = j;    /* color exists in Color1 */
        else {
            /* Color is new - copy it to a new slot: */
            ColorUnion->Colors[CrntSlot] = ColorIn2->Colors[i];
            ColorTransIn2[i] = CrntSlot++;
        }
    }

    if (CrntSlot > 256) {
        GifFreeMapObject(ColorUnion);
        return ((ColorMapObject *) NULL);
    }

    NewGifBitSize = GifBitSize(CrntSlot);
    RoundUpTo = (1 << NewGifBitSize);

    if (RoundUpTo != ColorUnion->ColorCount) {
        register GifColorType *Map = ColorUnion->Colors;

        /* 
         * Zero out slots up to next power of 2.
         * We know these slots exist because of the way ColorUnion's
         * start dimension was computed.
         */
        for (j = CrntSlot; j < RoundUpTo; j++)
            Map[j].Red = Map[j].Green = Map[j].Blue = 0;

        /* perhaps we can shrink the map? */
        if (RoundUpTo < ColorUnion->ColorCount) {
            GifColorType *new_map = (GifColorType *)reallocarray(Map,
                                 RoundUpTo, sizeof(GifColorType));
            if( new_map == NULL ) {
                GifFreeMapObject(ColorUnion);
                return ((ColorMapObject *) NULL);
            }
            ColorUnion->Colors = new_map;
        }
    }

    ColorUnion->ColorCount = RoundUpTo;
    ColorUnion->BitsPerPixel = NewGifBitSize;

    return (ColorUnion);
}

/*******************************************************************************
 Apply a given color translation to the raster bits of an image
*******************************************************************************/
void
GifApplyTranslation(SavedImage *Image, GifPixelType Translation[])
{
    register int i;
    register int RasterSize = Image->ImageDesc.Height * Image->ImageDesc.Width;

    for (i = 0; i < RasterSize; i++)
        Image->RasterBits[i] = Translation[Image->RasterBits[i]];
}

/******************************************************************************
 Extension record functions                              
******************************************************************************/
int
GifAddExtensionBlock(int *ExtensionBlockCount,
		     ExtensionBlock **ExtensionBlocks,
		     int Function,
		     unsigned int Len,
		     unsigned char ExtData[])
{
    ExtensionBlock *ep;

    if (*ExtensionBlocks == NULL)
        *ExtensionBlocks=(ExtensionBlock *)malloc(sizeof(ExtensionBlock));
    else {
        ExtensionBlock* ep_new = (ExtensionBlock *)reallocarray
				 (*ExtensionBlocks, (*ExtensionBlockCount + 1),
                                      sizeof(ExtensionBlock));
        if( ep_new == NULL )
            return (GIF_ERROR);
        *ExtensionBlocks = ep_new;
    }

    if (*ExtensionBlocks == NULL)
        return (GIF_ERROR);

    ep = &(*ExtensionBlocks)[(*ExtensionBlockCount)++];

    ep->Function = Function;
    ep->ByteCount=Len;
    ep->Bytes = (GifByteType *)malloc(ep->ByteCount);
    if (ep->Bytes == NULL)
        return (GIF_ERROR);

    if (ExtData != NULL) {
        memcpy(ep->Bytes, ExtData, Len);
    }

    return (GIF_OK);
}

void
GifFreeExtensions(int *ExtensionBlockCount,
		  ExtensionBlock **ExtensionBlocks)
{
    ExtensionBlock *ep;

    if (*ExtensionBlocks == NULL)
        return;

    for (ep = *ExtensionBlocks;
	 ep < (*ExtensionBlocks + *ExtensionBlockCount); 
	 ep++)
        (void)free((char *)ep->Bytes);
    (void)free((char *)*ExtensionBlocks);
    *ExtensionBlocks = NULL;
    *ExtensionBlockCount = 0;
}

/******************************************************************************
 Image block allocation functions                          
******************************************************************************/

/* Private Function:
 * Frees the last image in the GifFile->SavedImages array
 */
void
FreeLastSavedImage(GifFileType *GifFile)
{
    SavedImage *sp;
    
    if ((GifFile == NULL) || (GifFile->SavedImages == NULL))
        return;

    /* Remove one SavedImage from the GifFile */
    GifFile->ImageCount--;
    sp = &GifFile->SavedImages[GifFile->ImageCount];

    /* Deallocate its Colormap */
    if (sp->ImageDesc.ColorMap != NULL) {
        GifFreeMapObject(sp->ImageDesc.ColorMap);
        sp->ImageDesc.ColorMap = NULL;
    }

    /* Deallocate the image data */
    if (sp->RasterBits != NULL)
        free((char *)sp->RasterBits);

    /* Deallocate any extensions */
    GifFreeExtensions(&sp->ExtensionBlockCount, &sp->ExtensionBlocks);

    /*** FIXME: We could realloc the GifFile->SavedImages structure but is
     * there a point to it? Saves some memory but we'd have to do it every
     * time.  If this is used in GifFreeSavedImages then it would be inefficient
     * (The whole array is going to be deallocated.)  If we just use it when
     * we want to free the last Image it's convenient to do it here.
     */
}

/*
 * Append an image block to the SavedImages array  
 */
SavedImage *
GifMakeSavedImage(GifFileType *GifFile, const SavedImage *CopyFrom)
{
    if (GifFile->SavedImages == NULL)
        GifFile->SavedImages = (SavedImage *)malloc(sizeof(SavedImage));
    else {
        SavedImage* newSavedImages = (SavedImage *)reallocarray(GifFile->SavedImages,
                               (GifFile->ImageCount + 1), sizeof(SavedImage));
        if( newSavedImages == NULL)
            return ((SavedImage *)NULL);
        GifFile->SavedImages = newSavedImages;
    }
    if (GifFile->SavedImages == NULL)
        return ((SavedImage *)NULL);
    else {
        SavedImage *sp = &GifFile->SavedImages[GifFile->ImageCount++];

        if (CopyFrom != NULL) {
            memcpy((char *)sp, CopyFrom, sizeof(SavedImage));

            /* 
             * Make our own allocated copies of the heap fields in the
             * copied record.  This guards against potential aliasing
             * problems.
             */

            /* first, the local color map */
            if (CopyFrom->ImageDesc.ColorMap != NULL) {
                sp->ImageDesc.ColorMap = GifMakeMapObject(
                                         CopyFrom->ImageDesc.ColorMap->ColorCount,
                                         CopyFrom->ImageDesc.ColorMap->Colors);
                if (sp->ImageDesc.ColorMap == NULL) {
                    FreeLastSavedImage(GifFile);
                    return (SavedImage *)(NULL);
                }
            }

            /* next, the raster */
            sp->RasterBits = (unsigned char *)reallocarray(NULL,
                                                  (CopyFrom->ImageDesc.Height *
                                                  CopyFrom->ImageDesc.Width),
						  sizeof(GifPixelType));
            if (sp->RasterBits == NULL) {
                FreeLastSavedImage(GifFile);
                return (SavedImage *)(NULL);
            }
            memcpy(sp->RasterBits, CopyFrom->RasterBits,
                   sizeof(GifPixelType) * CopyFrom->ImageDesc.Height *
                   CopyFrom->ImageDesc.Width);

            /* finally, the extension blocks */
            if (CopyFrom->ExtensionBlocks != NULL) {
                sp->ExtensionBlocks = (ExtensionBlock *)reallocarray(NULL,
                                      CopyFrom->ExtensionBlockCount,
				      sizeof(ExtensionBlock));
                if (sp->ExtensionBlocks == NULL) {
                    FreeLastSavedImage(GifFile);
                    return (SavedImage *)(NULL);
                }
                memcpy(sp->ExtensionBlocks, CopyFrom->ExtensionBlocks,
                       sizeof(ExtensionBlock) * CopyFrom->ExtensionBlockCount);
            }
        }
        else {
            memset((char *)sp, '\0', sizeof(SavedImage));
        }

        return (sp);
    }
}

void
GifFreeSavedImages(GifFileType *GifFile)
{
    SavedImage *sp;

    if ((GifFile == NULL) || (GifFile->SavedImages == NULL)) {
        return;
    }
    for (sp = GifFile->SavedImages;
         sp < GifFile->SavedImages + GifFile->ImageCount; sp++) {
        if (sp->ImageDesc.ColorMap != NULL) {
            GifFreeMapObject(sp->ImageDesc.ColorMap);
            sp->ImageDesc.ColorMap = NULL;
        }

        if (sp->RasterBits != NULL)
            free((char *)sp->RasterBits);
	
	GifFreeExtensions(&sp->ExtensionBlockCount, &sp->ExtensionBlocks);
    }
    free((char *)GifFile->SavedImages);
    GifFile->SavedImages = NULL;
}

/* end */
