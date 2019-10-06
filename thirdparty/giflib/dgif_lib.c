/******************************************************************************

dgif_lib.c - GIF decoding

The functions here and in egif_lib.c are partitioned carefully so that
if you only require one of read and write capability, only one of these
two modules will be linked.  Preserve this property!

SPDX-License-Identifier: MIT

*****************************************************************************/

#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif /* _WIN32 */

#include "gif_lib.h"
#include "gif_lib_private.h"

/* compose unsigned little endian value */
#define UNSIGNED_LITTLE_ENDIAN(lo, hi)	((lo) | ((hi) << 8))

/* avoid extra function call in case we use fread (TVT) */
static int InternalRead(GifFileType *gif, GifByteType *buf, int len) {
    //fprintf(stderr, "### Read: %d\n", len);
    return 
	(((GifFilePrivateType*)gif->Private)->Read ?
	 ((GifFilePrivateType*)gif->Private)->Read(gif,buf,len) : 
	 fread(buf,1,len,((GifFilePrivateType*)gif->Private)->File));
}

static int DGifGetWord(GifFileType *GifFile, GifWord *Word);
static int DGifSetupDecompress(GifFileType *GifFile);
static int DGifDecompressLine(GifFileType *GifFile, GifPixelType *Line,
                              int LineLen);
static int DGifGetPrefixChar(GifPrefixType *Prefix, int Code, int ClearCode);
static int DGifDecompressInput(GifFileType *GifFile, int *Code);
static int DGifBufferedInput(GifFileType *GifFile, GifByteType *Buf,
                             GifByteType *NextByte);

/******************************************************************************
 Open a new GIF file for read, given by its name.
 Returns dynamically allocated GifFileType pointer which serves as the GIF
 info record.
******************************************************************************/
GifFileType *
DGifOpenFileName(const char *FileName, int *Error)
{
    int FileHandle;
    GifFileType *GifFile;

    if ((FileHandle = open(FileName, O_RDONLY)) == -1) {
	if (Error != NULL)
	    *Error = D_GIF_ERR_OPEN_FAILED;
        return NULL;
    }

    GifFile = DGifOpenFileHandle(FileHandle, Error);
    return GifFile;
}

/******************************************************************************
 Update a new GIF file, given its file handle.
 Returns dynamically allocated GifFileType pointer which serves as the GIF
 info record.
******************************************************************************/
GifFileType *
DGifOpenFileHandle(int FileHandle, int *Error)
{
    char Buf[GIF_STAMP_LEN + 1];
    GifFileType *GifFile;
    GifFilePrivateType *Private;
    FILE *f;

    GifFile = (GifFileType *)malloc(sizeof(GifFileType));
    if (GifFile == NULL) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_ENOUGH_MEM;
        (void)close(FileHandle);
        return NULL;
    }

    /*@i1@*/memset(GifFile, '\0', sizeof(GifFileType));

    /* Belt and suspenders, in case the null pointer isn't zero */
    GifFile->SavedImages = NULL;
    GifFile->SColorMap = NULL;

    Private = (GifFilePrivateType *)calloc(1, sizeof(GifFilePrivateType));
    if (Private == NULL) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_ENOUGH_MEM;
        (void)close(FileHandle);
        free((char *)GifFile);
        return NULL;
    }

    /*@i1@*/memset(Private, '\0', sizeof(GifFilePrivateType));

#ifdef _WIN32
    _setmode(FileHandle, O_BINARY);    /* Make sure it is in binary mode. */
#endif /* _WIN32 */

    f = fdopen(FileHandle, "rb");    /* Make it into a stream: */

    /*@-mustfreeonly@*/
    GifFile->Private = (void *)Private;
    Private->FileHandle = FileHandle;
    Private->File = f;
    Private->FileState = FILE_STATE_READ;
    Private->Read = NULL;        /* don't use alternate input method (TVT) */
    GifFile->UserData = NULL;    /* TVT */
    /*@=mustfreeonly@*/

    /* Let's see if this is a GIF file: */
    /* coverity[check_return] */
    if (InternalRead(GifFile, (unsigned char *)Buf, GIF_STAMP_LEN) != GIF_STAMP_LEN) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_READ_FAILED;
        (void)fclose(f);
        free((char *)Private);
        free((char *)GifFile);
        return NULL;
    }

    /* Check for GIF prefix at start of file */
    Buf[GIF_STAMP_LEN] = 0;
    if (strncmp(GIF_STAMP, Buf, GIF_VERSION_POS) != 0) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_GIF_FILE;
        (void)fclose(f);
        free((char *)Private);
        free((char *)GifFile);
        return NULL;
    }

    if (DGifGetScreenDesc(GifFile) == GIF_ERROR) {
        (void)fclose(f);
        free((char *)Private);
        free((char *)GifFile);
        return NULL;
    }

    GifFile->Error = 0;

    /* What version of GIF? */
    Private->gif89 = (Buf[GIF_VERSION_POS] == '9');

    return GifFile;
}

/******************************************************************************
 GifFileType constructor with user supplied input function (TVT)
******************************************************************************/
GifFileType *
DGifOpen(void *userData, InputFunc readFunc, int *Error)
{
    char Buf[GIF_STAMP_LEN + 1];
    GifFileType *GifFile;
    GifFilePrivateType *Private;

    GifFile = (GifFileType *)malloc(sizeof(GifFileType));
    if (GifFile == NULL) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_ENOUGH_MEM;
        return NULL;
    }

    memset(GifFile, '\0', sizeof(GifFileType));

    /* Belt and suspenders, in case the null pointer isn't zero */
    GifFile->SavedImages = NULL;
    GifFile->SColorMap = NULL;

    Private = (GifFilePrivateType *)calloc(1, sizeof(GifFilePrivateType));
    if (!Private) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_ENOUGH_MEM;
        free((char *)GifFile);
        return NULL;
    }
    /*@i1@*/memset(Private, '\0', sizeof(GifFilePrivateType));

    GifFile->Private = (void *)Private;
    Private->FileHandle = 0;
    Private->File = NULL;
    Private->FileState = FILE_STATE_READ;

    Private->Read = readFunc;    /* TVT */
    GifFile->UserData = userData;    /* TVT */

    /* Lets see if this is a GIF file: */
    /* coverity[check_return] */
    if (InternalRead(GifFile, (unsigned char *)Buf, GIF_STAMP_LEN) != GIF_STAMP_LEN) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_READ_FAILED;
        free((char *)Private);
        free((char *)GifFile);
        return NULL;
    }

    /* Check for GIF prefix at start of file */
    Buf[GIF_STAMP_LEN] = '\0';
    if (strncmp(GIF_STAMP, Buf, GIF_VERSION_POS) != 0) {
        if (Error != NULL)
	    *Error = D_GIF_ERR_NOT_GIF_FILE;
        free((char *)Private);
        free((char *)GifFile);
        return NULL;
    }

    if (DGifGetScreenDesc(GifFile) == GIF_ERROR) {
        free((char *)Private);
        free((char *)GifFile);
        if (Error != NULL)
	    *Error = D_GIF_ERR_NO_SCRN_DSCR;
        return NULL;
    }

    GifFile->Error = 0;

    /* What version of GIF? */
    Private->gif89 = (Buf[GIF_VERSION_POS] == '9');

    return GifFile;
}

/******************************************************************************
 This routine should be called before any other DGif calls. Note that
 this routine is called automatically from DGif file open routines.
******************************************************************************/
int
DGifGetScreenDesc(GifFileType *GifFile)
{
    int BitsPerPixel;
    bool SortFlag;
    GifByteType Buf[3];
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    /* Put the screen descriptor into the file: */
    if (DGifGetWord(GifFile, &GifFile->SWidth) == GIF_ERROR ||
        DGifGetWord(GifFile, &GifFile->SHeight) == GIF_ERROR)
        return GIF_ERROR;

    if (InternalRead(GifFile, Buf, 3) != 3) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
	GifFreeMapObject(GifFile->SColorMap);
	GifFile->SColorMap = NULL;
        return GIF_ERROR;
    }
    GifFile->SColorResolution = (((Buf[0] & 0x70) + 1) >> 4) + 1;
    SortFlag = (Buf[0] & 0x08) != 0;
    BitsPerPixel = (Buf[0] & 0x07) + 1;
    GifFile->SBackGroundColor = Buf[1];
    GifFile->AspectByte = Buf[2]; 
    if (Buf[0] & 0x80) {    /* Do we have global color map? */
	int i;

        GifFile->SColorMap = GifMakeMapObject(1 << BitsPerPixel, NULL);
        if (GifFile->SColorMap == NULL) {
            GifFile->Error = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }

        /* Get the global color map: */
	GifFile->SColorMap->SortFlag = SortFlag;
        for (i = 0; i < GifFile->SColorMap->ColorCount; i++) {
	    /* coverity[check_return] */
            if (InternalRead(GifFile, Buf, 3) != 3) {
                GifFreeMapObject(GifFile->SColorMap);
                GifFile->SColorMap = NULL;
                GifFile->Error = D_GIF_ERR_READ_FAILED;
                return GIF_ERROR;
            }
            GifFile->SColorMap->Colors[i].Red = Buf[0];
            GifFile->SColorMap->Colors[i].Green = Buf[1];
            GifFile->SColorMap->Colors[i].Blue = Buf[2];
        }
    } else {
        GifFile->SColorMap = NULL;
    }

    /*
     * No check here for whether the background color is in range for the
     * screen color map.  Possibly there should be.
     */
    
    return GIF_OK;
}

const char *
DGifGetGifVersion(GifFileType *GifFile)
{
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (Private->gif89)
	return GIF89_STAMP;
    else
	return GIF87_STAMP;
}

/******************************************************************************
 This routine should be called before any attempt to read an image.
******************************************************************************/
int
DGifGetRecordType(GifFileType *GifFile, GifRecordType* Type)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    /* coverity[check_return] */
    if (InternalRead(GifFile, &Buf, 1) != 1) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        return GIF_ERROR;
    }

    //fprintf(stderr, "### DGifGetRecordType: %02x\n", Buf);
    switch (Buf) {
      case DESCRIPTOR_INTRODUCER:
          *Type = IMAGE_DESC_RECORD_TYPE;
          break;
      case EXTENSION_INTRODUCER:
          *Type = EXTENSION_RECORD_TYPE;
          break;
      case TERMINATOR_INTRODUCER:
          *Type = TERMINATE_RECORD_TYPE;
          break;
      default:
          *Type = UNDEFINED_RECORD_TYPE;
          GifFile->Error = D_GIF_ERR_WRONG_RECORD;
          return GIF_ERROR;
    }

    return GIF_OK;
}

int
DGifGetImageHeader(GifFileType *GifFile)
{
    unsigned int BitsPerPixel;
    GifByteType Buf[3];
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    if (DGifGetWord(GifFile, &GifFile->Image.Left) == GIF_ERROR ||
        DGifGetWord(GifFile, &GifFile->Image.Top) == GIF_ERROR ||
        DGifGetWord(GifFile, &GifFile->Image.Width) == GIF_ERROR ||
        DGifGetWord(GifFile, &GifFile->Image.Height) == GIF_ERROR)
        return GIF_ERROR;
    if (InternalRead(GifFile, Buf, 1) != 1) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        GifFreeMapObject(GifFile->Image.ColorMap);
        GifFile->Image.ColorMap = NULL;
        return GIF_ERROR;
    }
    BitsPerPixel = (Buf[0] & 0x07) + 1;
    GifFile->Image.Interlace = (Buf[0] & 0x40) ? true : false;

    /* Setup the colormap */
    if (GifFile->Image.ColorMap) {
        GifFreeMapObject(GifFile->Image.ColorMap);
        GifFile->Image.ColorMap = NULL;
    }
    /* Does this image have local color map? */
    if (Buf[0] & 0x80) {
        unsigned int i;

        GifFile->Image.ColorMap = GifMakeMapObject(1 << BitsPerPixel, NULL);
        if (GifFile->Image.ColorMap == NULL) {
            GifFile->Error = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }

        /* Get the image local color map: */
        for (i = 0; i < GifFile->Image.ColorMap->ColorCount; i++) {
            /* coverity[check_return] */
            if (InternalRead(GifFile, Buf, 3) != 3) {
                GifFreeMapObject(GifFile->Image.ColorMap);
                GifFile->Error = D_GIF_ERR_READ_FAILED;
                GifFile->Image.ColorMap = NULL;
                return GIF_ERROR;
            }
            GifFile->Image.ColorMap->Colors[i].Red = Buf[0];
            GifFile->Image.ColorMap->Colors[i].Green = Buf[1];
            GifFile->Image.ColorMap->Colors[i].Blue = Buf[2];
        }
    }

    Private->PixelCount = (long)GifFile->Image.Width *
       (long)GifFile->Image.Height;

    /* Reset decompress algorithm parameters. */
    return DGifSetupDecompress(GifFile);
}

/******************************************************************************
 This routine should be called before any attempt to read an image.
 Note it is assumed the Image desc. header has been read.
******************************************************************************/
int
DGifGetImageDesc(GifFileType *GifFile)
{
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;
    SavedImage *sp;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    if (DGifGetImageHeader(GifFile) == GIF_ERROR) {
        return GIF_ERROR;
    }

    if (GifFile->SavedImages) {
        SavedImage* new_saved_images =
            (SavedImage *)reallocarray(GifFile->SavedImages,
                            (GifFile->ImageCount + 1), sizeof(SavedImage));
        if (new_saved_images == NULL) {
            GifFile->Error = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }
        GifFile->SavedImages = new_saved_images;
    } else {
        if ((GifFile->SavedImages =
             (SavedImage *) malloc(sizeof(SavedImage))) == NULL) {
            GifFile->Error = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }
    }

    sp = &GifFile->SavedImages[GifFile->ImageCount];
    memcpy(&sp->ImageDesc, &GifFile->Image, sizeof(GifImageDesc));
    if (GifFile->Image.ColorMap != NULL) {
        sp->ImageDesc.ColorMap = GifMakeMapObject(
                                 GifFile->Image.ColorMap->ColorCount,
                                 GifFile->Image.ColorMap->Colors);
        if (sp->ImageDesc.ColorMap == NULL) {
            GifFile->Error = D_GIF_ERR_NOT_ENOUGH_MEM;
            return GIF_ERROR;
        }
    }
    sp->RasterBits = (unsigned char *)NULL;
    sp->ExtensionBlockCount = 0;
    sp->ExtensionBlocks = (ExtensionBlock *) NULL;

    GifFile->ImageCount++;

    return GIF_OK;
}

/******************************************************************************
 Get one full scanned line (Line) of length LineLen from GIF file.
******************************************************************************/
int
DGifGetLine(GifFileType *GifFile, GifPixelType *Line, int LineLen)
{
    GifByteType *Dummy;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    if (!LineLen)
        LineLen = GifFile->Image.Width;

    if ((Private->PixelCount -= LineLen) > 0xffff0000UL) {
        GifFile->Error = D_GIF_ERR_DATA_TOO_BIG;
        return GIF_ERROR;
    }

    if (DGifDecompressLine(GifFile, Line, LineLen) == GIF_OK) {
        if (Private->PixelCount == 0) {
            /* We probably won't be called any more, so let's clean up
             * everything before we return: need to flush out all the
             * rest of image until an empty block (size 0)
             * detected. We use GetCodeNext.
	     */
            do
                if (DGifGetCodeNext(GifFile, &Dummy) == GIF_ERROR)
                    return GIF_ERROR;
            while (Dummy != NULL) ;
        }
        return GIF_OK;
    } else
        return GIF_ERROR;
}

/******************************************************************************
 Put one pixel (Pixel) into GIF file.
******************************************************************************/
int
DGifGetPixel(GifFileType *GifFile, GifPixelType Pixel)
{
    GifByteType *Dummy;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }
    if (--Private->PixelCount > 0xffff0000UL)
    {
        GifFile->Error = D_GIF_ERR_DATA_TOO_BIG;
        return GIF_ERROR;
    }

    if (DGifDecompressLine(GifFile, &Pixel, 1) == GIF_OK) {
        if (Private->PixelCount == 0) {
            /* We probably won't be called any more, so let's clean up
             * everything before we return: need to flush out all the
             * rest of image until an empty block (size 0)
             * detected. We use GetCodeNext.
	     */
            do
                if (DGifGetCodeNext(GifFile, &Dummy) == GIF_ERROR)
                    return GIF_ERROR;
            while (Dummy != NULL) ;
        }
        return GIF_OK;
    } else
        return GIF_ERROR;
}

/******************************************************************************
 Get an extension block (see GIF manual) from GIF file. This routine only
 returns the first data block, and DGifGetExtensionNext should be called
 after this one until NULL extension is returned.
 The Extension should NOT be freed by the user (not dynamically allocated).
 Note it is assumed the Extension description header has been read.
******************************************************************************/
int
DGifGetExtension(GifFileType *GifFile, int *ExtCode, GifByteType **Extension)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    //fprintf(stderr, "### -> DGifGetExtension:\n");
    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    /* coverity[check_return] */
    if (InternalRead(GifFile, &Buf, 1) != 1) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        return GIF_ERROR;
    }
    *ExtCode = Buf;
    //fprintf(stderr, "### <- DGifGetExtension: %02x, about to call next\n", Buf);

    return DGifGetExtensionNext(GifFile, Extension);
}

/******************************************************************************
 Get a following extension block (see GIF manual) from GIF file. This
 routine should be called until NULL Extension is returned.
 The Extension should NOT be freed by the user (not dynamically allocated).
******************************************************************************/
int
DGifGetExtensionNext(GifFileType *GifFile, GifByteType ** Extension)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    //fprintf(stderr, "### -> DGifGetExtensionNext\n");
    if (InternalRead(GifFile, &Buf, 1) != 1) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        return GIF_ERROR;
    }
    //fprintf(stderr, "### DGifGetExtensionNext sees %d\n", Buf);

    if (Buf > 0) {
        *Extension = Private->Buf;    /* Use private unused buffer. */
        (*Extension)[0] = Buf;  /* Pascal strings notation (pos. 0 is len.). */
	/* coverity[tainted_data,check_return] */
        if (InternalRead(GifFile, &((*Extension)[1]), Buf) != Buf) {
            GifFile->Error = D_GIF_ERR_READ_FAILED;
            return GIF_ERROR;
        }
    } else
        *Extension = NULL;
    //fprintf(stderr, "### <- DGifGetExtensionNext: %p\n", Extension);

    return GIF_OK;
}

/******************************************************************************
 Extract a Graphics Control Block from raw extension data
******************************************************************************/

int DGifExtensionToGCB(const size_t GifExtensionLength,
		       const GifByteType *GifExtension,
		       GraphicsControlBlock *GCB)
{
    if (GifExtensionLength != 4) {
	return GIF_ERROR;
    }

    GCB->DisposalMode = (GifExtension[0] >> 2) & 0x07;
    GCB->UserInputFlag = (GifExtension[0] & 0x02) != 0;
    GCB->DelayTime = UNSIGNED_LITTLE_ENDIAN(GifExtension[1], GifExtension[2]);
    if (GifExtension[0] & 0x01)
	GCB->TransparentColor = (int)GifExtension[3];
    else
	GCB->TransparentColor = NO_TRANSPARENT_COLOR;

    return GIF_OK;
}

/******************************************************************************
 Extract the Graphics Control Block for a saved image, if it exists.
******************************************************************************/

int DGifSavedExtensionToGCB(GifFileType *GifFile,
			    int ImageIndex, GraphicsControlBlock *GCB)
{
    int i;

    if (ImageIndex < 0 || ImageIndex > GifFile->ImageCount - 1)
	return GIF_ERROR;

    GCB->DisposalMode = DISPOSAL_UNSPECIFIED;
    GCB->UserInputFlag = false;
    GCB->DelayTime = 0;
    GCB->TransparentColor = NO_TRANSPARENT_COLOR;

    for (i = 0; i < GifFile->SavedImages[ImageIndex].ExtensionBlockCount; i++) {
	ExtensionBlock *ep = &GifFile->SavedImages[ImageIndex].ExtensionBlocks[i];
	if (ep->Function == GRAPHICS_EXT_FUNC_CODE)
	    return DGifExtensionToGCB(ep->ByteCount, ep->Bytes, GCB);
    }

    return GIF_ERROR;
}

/******************************************************************************
 This routine should be called last, to close the GIF file.
******************************************************************************/
int
DGifCloseFile(GifFileType *GifFile, int *ErrorCode)
{
    GifFilePrivateType *Private;

    if (GifFile == NULL || GifFile->Private == NULL)
        return GIF_ERROR;

    if (GifFile->Image.ColorMap) {
        GifFreeMapObject(GifFile->Image.ColorMap);
        GifFile->Image.ColorMap = NULL;
    }

    if (GifFile->SColorMap) {
        GifFreeMapObject(GifFile->SColorMap);
        GifFile->SColorMap = NULL;
    }

    if (GifFile->SavedImages) {
        GifFreeSavedImages(GifFile);
        GifFile->SavedImages = NULL;
    }

    GifFreeExtensions(&GifFile->ExtensionBlockCount, &GifFile->ExtensionBlocks);

    Private = (GifFilePrivateType *) GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
	if (ErrorCode != NULL)
	    *ErrorCode = D_GIF_ERR_NOT_READABLE;
	free((char *)GifFile->Private);
	free(GifFile);
        return GIF_ERROR;
    }

    if (Private->File && (fclose(Private->File) != 0)) {
	if (ErrorCode != NULL)
	    *ErrorCode = D_GIF_ERR_CLOSE_FAILED;
	free((char *)GifFile->Private);
	free(GifFile);
        return GIF_ERROR;
    }

    free((char *)GifFile->Private);
    free(GifFile);
    if (ErrorCode != NULL)
	*ErrorCode = D_GIF_SUCCEEDED;
    return GIF_OK;
}

/******************************************************************************
 Get 2 bytes (word) from the given file:
******************************************************************************/
static int
DGifGetWord(GifFileType *GifFile, GifWord *Word)
{
    unsigned char c[2];

    /* coverity[check_return] */
    if (InternalRead(GifFile, c, 2) != 2) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        return GIF_ERROR;
    }

    *Word = (GifWord)UNSIGNED_LITTLE_ENDIAN(c[0], c[1]);
    return GIF_OK;
}

/******************************************************************************
 Get the image code in compressed form.  This routine can be called if the
 information needed to be piped out as is. Obviously this is much faster
 than decoding and encoding again. This routine should be followed by calls
 to DGifGetCodeNext, until NULL block is returned.
 The block should NOT be freed by the user (not dynamically allocated).
******************************************************************************/
int
DGifGetCode(GifFileType *GifFile, int *CodeSize, GifByteType **CodeBlock)
{
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    *CodeSize = Private->BitsPerPixel;

    return DGifGetCodeNext(GifFile, CodeBlock);
}

/******************************************************************************
 Continue to get the image code in compressed form. This routine should be
 called until NULL block is returned.
 The block should NOT be freed by the user (not dynamically allocated).
******************************************************************************/
int
DGifGetCodeNext(GifFileType *GifFile, GifByteType **CodeBlock)
{
    GifByteType Buf;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    /* coverity[tainted_data_argument] */
    /* coverity[check_return] */
    if (InternalRead(GifFile, &Buf, 1) != 1) {
        GifFile->Error = D_GIF_ERR_READ_FAILED;
        return GIF_ERROR;
    }

    /* coverity[lower_bounds] */
    if (Buf > 0) {
        *CodeBlock = Private->Buf;    /* Use private unused buffer. */
        (*CodeBlock)[0] = Buf;  /* Pascal strings notation (pos. 0 is len.). */
	/* coverity[tainted_data] */
        if (InternalRead(GifFile, &((*CodeBlock)[1]), Buf) != Buf) {
            GifFile->Error = D_GIF_ERR_READ_FAILED;
            return GIF_ERROR;
        }
    } else {
        *CodeBlock = NULL;
        Private->Buf[0] = 0;    /* Make sure the buffer is empty! */
        Private->PixelCount = 0;    /* And local info. indicate image read. */
    }

    return GIF_OK;
}

/******************************************************************************
 Setup the LZ decompression for this image:
******************************************************************************/
static int
DGifSetupDecompress(GifFileType *GifFile)
{
    int i, BitsPerPixel;
    GifByteType CodeSize;
    GifPrefixType *Prefix;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    /* coverity[check_return] */
    if (InternalRead(GifFile, &CodeSize, 1) < 1) {    /* Read Code size from file. */
	return GIF_ERROR;    /* Failed to read Code size. */
    }
    BitsPerPixel = CodeSize;

    /* this can only happen on a severely malformed GIF */
    if (BitsPerPixel > 8) {
	GifFile->Error = D_GIF_ERR_READ_FAILED;	/* somewhat bogus error code */
	return GIF_ERROR;    /* Failed to read Code size. */
    }

    Private->Buf[0] = 0;    /* Input Buffer empty. */
    Private->BitsPerPixel = BitsPerPixel;
    Private->ClearCode = (1 << BitsPerPixel);
    Private->EOFCode = Private->ClearCode + 1;
    Private->RunningCode = Private->EOFCode + 1;
    Private->RunningBits = BitsPerPixel + 1;    /* Number of bits per code. */
    Private->MaxCode1 = 1 << Private->RunningBits;    /* Max. code + 1. */
    Private->StackPtr = 0;    /* No pixels on the pixel stack. */
    Private->LastCode = NO_SUCH_CODE;
    Private->CrntShiftState = 0;    /* No information in CrntShiftDWord. */
    Private->CrntShiftDWord = 0;

    Prefix = Private->Prefix;
    for (i = 0; i <= LZ_MAX_CODE; i++)
        Prefix[i] = NO_SUCH_CODE;

    return GIF_OK;
}

/******************************************************************************
 The LZ decompression routine:
 This version decompress the given GIF file into Line of length LineLen.
 This routine can be called few times (one per scan line, for example), in
 order the complete the whole image.
******************************************************************************/
static int
DGifDecompressLine(GifFileType *GifFile, GifPixelType *Line, int LineLen)
{
    int i = 0;
    int j, CrntCode, EOFCode, ClearCode, CrntPrefix, LastCode, StackPtr;
    GifByteType *Stack, *Suffix;
    GifPrefixType *Prefix;
    GifFilePrivateType *Private = (GifFilePrivateType *) GifFile->Private;

    StackPtr = Private->StackPtr;
    Prefix = Private->Prefix;
    Suffix = Private->Suffix;
    Stack = Private->Stack;
    EOFCode = Private->EOFCode;
    ClearCode = Private->ClearCode;
    LastCode = Private->LastCode;

    if (StackPtr > LZ_MAX_CODE) {
        return GIF_ERROR;
    }

    if (StackPtr != 0) {
        /* Let pop the stack off before continueing to read the GIF file: */
        while (StackPtr != 0 && i < LineLen)
            Line[i++] = Stack[--StackPtr];
    }

    while (i < LineLen) {    /* Decode LineLen items. */
        if (DGifDecompressInput(GifFile, &CrntCode) == GIF_ERROR)
            return GIF_ERROR;

        if (CrntCode == EOFCode) {
            /* Note however that usually we will not be here as we will stop
             * decoding as soon as we got all the pixel, or EOF code will
             * not be read at all, and DGifGetLine/Pixel clean everything.  */
	    GifFile->Error = D_GIF_ERR_EOF_TOO_SOON;
	    return GIF_ERROR;
        } else if (CrntCode == ClearCode) {
            /* We need to start over again: */
            for (j = 0; j <= LZ_MAX_CODE; j++)
                Prefix[j] = NO_SUCH_CODE;
            Private->RunningCode = Private->EOFCode + 1;
            Private->RunningBits = Private->BitsPerPixel + 1;
            Private->MaxCode1 = 1 << Private->RunningBits;
            LastCode = Private->LastCode = NO_SUCH_CODE;
        } else {
            /* Its regular code - if in pixel range simply add it to output
             * stream, otherwise trace to codes linked list until the prefix
             * is in pixel range: */
            if (CrntCode < ClearCode) {
                /* This is simple - its pixel scalar, so add it to output: */
                Line[i++] = CrntCode;
            } else {
                /* Its a code to needed to be traced: trace the linked list
                 * until the prefix is a pixel, while pushing the suffix
                 * pixels on our stack. If we done, pop the stack in reverse
                 * (thats what stack is good for!) order to output.  */
                if (Prefix[CrntCode] == NO_SUCH_CODE) {
                    CrntPrefix = LastCode;

                    /* Only allowed if CrntCode is exactly the running code:
                     * In that case CrntCode = XXXCode, CrntCode or the
                     * prefix code is last code and the suffix char is
                     * exactly the prefix of last code! */
                    if (CrntCode == Private->RunningCode - 2) {
                        Suffix[Private->RunningCode - 2] =
                           Stack[StackPtr++] = DGifGetPrefixChar(Prefix,
                                                                 LastCode,
                                                                 ClearCode);
                    } else {
                        Suffix[Private->RunningCode - 2] =
                           Stack[StackPtr++] = DGifGetPrefixChar(Prefix,
                                                                 CrntCode,
                                                                 ClearCode);
                    }
                } else
                    CrntPrefix = CrntCode;

                /* Now (if image is O.K.) we should not get a NO_SUCH_CODE
                 * during the trace. As we might loop forever, in case of
                 * defective image, we use StackPtr as loop counter and stop
                 * before overflowing Stack[]. */
                while (StackPtr < LZ_MAX_CODE &&
                       CrntPrefix > ClearCode && CrntPrefix <= LZ_MAX_CODE) {
                    Stack[StackPtr++] = Suffix[CrntPrefix];
                    CrntPrefix = Prefix[CrntPrefix];
                }
                if (StackPtr >= LZ_MAX_CODE || CrntPrefix > LZ_MAX_CODE) {
                    GifFile->Error = D_GIF_ERR_IMAGE_DEFECT;
                    return GIF_ERROR;
                }
                /* Push the last character on stack: */
                Stack[StackPtr++] = CrntPrefix;

                /* Now lets pop all the stack into output: */
                while (StackPtr != 0 && i < LineLen)
                    Line[i++] = Stack[--StackPtr];
            }
            if (LastCode != NO_SUCH_CODE && Private->RunningCode - 2 < (LZ_MAX_CODE+1) && Prefix[Private->RunningCode - 2] == NO_SUCH_CODE) {
                Prefix[Private->RunningCode - 2] = LastCode;

                if (CrntCode == Private->RunningCode - 2) {
                    /* Only allowed if CrntCode is exactly the running code:
                     * In that case CrntCode = XXXCode, CrntCode or the
                     * prefix code is last code and the suffix char is
                     * exactly the prefix of last code! */
                    Suffix[Private->RunningCode - 2] =
                       DGifGetPrefixChar(Prefix, LastCode, ClearCode);
                } else {
                    Suffix[Private->RunningCode - 2] =
                       DGifGetPrefixChar(Prefix, CrntCode, ClearCode);
                }
            }
            LastCode = CrntCode;
        }
    }

    Private->LastCode = LastCode;
    Private->StackPtr = StackPtr;

    return GIF_OK;
}

/******************************************************************************
 Routine to trace the Prefixes linked list until we get a prefix which is
 not code, but a pixel value (less than ClearCode). Returns that pixel value.
 If image is defective, we might loop here forever, so we limit the loops to
 the maximum possible if image O.k. - LZ_MAX_CODE times.
******************************************************************************/
static int
DGifGetPrefixChar(GifPrefixType *Prefix, int Code, int ClearCode)
{
    int i = 0;

    while (Code > ClearCode && i++ <= LZ_MAX_CODE) {
        if (Code > LZ_MAX_CODE) {
            return NO_SUCH_CODE;
        }
        Code = Prefix[Code];
    }
    return Code;
}

/******************************************************************************
 Interface for accessing the LZ codes directly. Set Code to the real code
 (12bits), or to -1 if EOF code is returned.
******************************************************************************/
int
DGifGetLZCodes(GifFileType *GifFile, int *Code)
{
    GifByteType *CodeBlock;
    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    if (!IS_READABLE(Private)) {
        /* This file was NOT open for reading: */
        GifFile->Error = D_GIF_ERR_NOT_READABLE;
        return GIF_ERROR;
    }

    if (DGifDecompressInput(GifFile, Code) == GIF_ERROR)
        return GIF_ERROR;

    if (*Code == Private->EOFCode) {
        /* Skip rest of codes (hopefully only NULL terminating block): */
        do {
            if (DGifGetCodeNext(GifFile, &CodeBlock) == GIF_ERROR)
                return GIF_ERROR;
        } while (CodeBlock != NULL) ;

        *Code = -1;
    } else if (*Code == Private->ClearCode) {
        /* We need to start over again: */
        Private->RunningCode = Private->EOFCode + 1;
        Private->RunningBits = Private->BitsPerPixel + 1;
        Private->MaxCode1 = 1 << Private->RunningBits;
    }

    return GIF_OK;
}

/******************************************************************************
 The LZ decompression input routine:
 This routine is responsable for the decompression of the bit stream from
 8 bits (bytes) packets, into the real codes.
 Returns GIF_OK if read successfully.
******************************************************************************/
static int
DGifDecompressInput(GifFileType *GifFile, int *Code)
{
    static const unsigned short CodeMasks[] = {
	0x0000, 0x0001, 0x0003, 0x0007,
	0x000f, 0x001f, 0x003f, 0x007f,
	0x00ff, 0x01ff, 0x03ff, 0x07ff,
	0x0fff
    };

    GifFilePrivateType *Private = (GifFilePrivateType *)GifFile->Private;

    GifByteType NextByte;

    /* The image can't contain more than LZ_BITS per code. */
    if (Private->RunningBits > LZ_BITS) {
        GifFile->Error = D_GIF_ERR_IMAGE_DEFECT;
        return GIF_ERROR;
    }
    
    while (Private->CrntShiftState < Private->RunningBits) {
        /* Needs to get more bytes from input stream for next code: */
        if (DGifBufferedInput(GifFile, Private->Buf, &NextByte) == GIF_ERROR) {
            return GIF_ERROR;
        }
        Private->CrntShiftDWord |=
	    ((unsigned long)NextByte) << Private->CrntShiftState;
        Private->CrntShiftState += 8;
    }
    *Code = Private->CrntShiftDWord & CodeMasks[Private->RunningBits];

    Private->CrntShiftDWord >>= Private->RunningBits;
    Private->CrntShiftState -= Private->RunningBits;

    /* If code cannot fit into RunningBits bits, must raise its size. Note
     * however that codes above 4095 are used for special signaling.
     * If we're using LZ_BITS bits already and we're at the max code, just
     * keep using the table as it is, don't increment Private->RunningCode.
     */
    if (Private->RunningCode < LZ_MAX_CODE + 2 &&
	++Private->RunningCode > Private->MaxCode1 &&
	Private->RunningBits < LZ_BITS) {
        Private->MaxCode1 <<= 1;
        Private->RunningBits++;
    }
    return GIF_OK;
}

/******************************************************************************
 This routines read one GIF data block at a time and buffers it internally
 so that the decompression routine could access it.
 The routine returns the next byte from its internal buffer (or read next
 block in if buffer empty) and returns GIF_OK if succesful.
******************************************************************************/
static int
DGifBufferedInput(GifFileType *GifFile, GifByteType *Buf, GifByteType *NextByte)
{
    if (Buf[0] == 0) {
        /* Needs to read the next buffer - this one is empty: */
	/* coverity[check_return] */
        if (InternalRead(GifFile, Buf, 1) != 1) {
            GifFile->Error = D_GIF_ERR_READ_FAILED;
            return GIF_ERROR;
        }
        /* There shouldn't be any empty data blocks here as the LZW spec
         * says the LZW termination code should come first.  Therefore we
         * shouldn't be inside this routine at that point.
         */
        if (Buf[0] == 0) {
            GifFile->Error = D_GIF_ERR_IMAGE_DEFECT;
            return GIF_ERROR;
        }
        if (InternalRead(GifFile, &Buf[1], Buf[0]) != Buf[0]) {
            GifFile->Error = D_GIF_ERR_READ_FAILED;
            return GIF_ERROR;
        }
        *NextByte = Buf[1];
        Buf[1] = 2;    /* We use now the second place as last char read! */
        Buf[0]--;
    } else {
        *NextByte = Buf[Buf[1]++];
        Buf[0]--;
    }

    return GIF_OK;
}

/******************************************************************************
 This routine reads an entire GIF into core, hanging all its state info off
 the GifFileType pointer.  Call DGifOpenFileName() or DGifOpenFileHandle()
 first to initialize I/O.  Its inverse is EGifSpew().
*******************************************************************************/
int
DGifSlurp(GifFileType *GifFile)
{
    size_t ImageSize;
    GifRecordType RecordType;
    SavedImage *sp;
    GifByteType *ExtData;
    int ExtFunction;

    GifFile->ExtensionBlocks = NULL;
    GifFile->ExtensionBlockCount = 0;

    do {
        if (DGifGetRecordType(GifFile, &RecordType) == GIF_ERROR)
            return (GIF_ERROR);

        switch (RecordType) {
          case IMAGE_DESC_RECORD_TYPE:
              if (DGifGetImageDesc(GifFile) == GIF_ERROR)
                  return (GIF_ERROR);

              sp = &GifFile->SavedImages[GifFile->ImageCount - 1];
              /* Allocate memory for the image */
              if (sp->ImageDesc.Width <= 0 || sp->ImageDesc.Height <= 0 ||
                      sp->ImageDesc.Width > (INT_MAX / sp->ImageDesc.Height)) {
                  return GIF_ERROR;
              }
              ImageSize = sp->ImageDesc.Width * sp->ImageDesc.Height;

              if (ImageSize > (SIZE_MAX / sizeof(GifPixelType))) {
                  return GIF_ERROR;
              }
              sp->RasterBits = (unsigned char *)reallocarray(NULL, ImageSize,
                      sizeof(GifPixelType));

              if (sp->RasterBits == NULL) {
                  return GIF_ERROR;
              }

	      if (sp->ImageDesc.Interlace) {
		  int i, j;
		   /* 
		    * The way an interlaced image should be read - 
		    * offsets and jumps...
		    */
		  int InterlacedOffset[] = { 0, 4, 2, 1 };
		  int InterlacedJumps[] = { 8, 8, 4, 2 };
		  /* Need to perform 4 passes on the image */
		  for (i = 0; i < 4; i++)
		      for (j = InterlacedOffset[i]; 
			   j < sp->ImageDesc.Height;
			   j += InterlacedJumps[i]) {
			  if (DGifGetLine(GifFile, 
					  sp->RasterBits+j*sp->ImageDesc.Width, 
					  sp->ImageDesc.Width) == GIF_ERROR)
			      return GIF_ERROR;
		      }
	      }
	      else {
		  if (DGifGetLine(GifFile,sp->RasterBits,ImageSize)==GIF_ERROR)
		      return (GIF_ERROR);
	      }

              if (GifFile->ExtensionBlocks) {
                  sp->ExtensionBlocks = GifFile->ExtensionBlocks;
                  sp->ExtensionBlockCount = GifFile->ExtensionBlockCount;

                  GifFile->ExtensionBlocks = NULL;
                  GifFile->ExtensionBlockCount = 0;
              }
              break;

          case EXTENSION_RECORD_TYPE:
              if (DGifGetExtension(GifFile,&ExtFunction,&ExtData) == GIF_ERROR)
                  return (GIF_ERROR);
	      /* Create an extension block with our data */
              if (ExtData != NULL) {
		  if (GifAddExtensionBlock(&GifFile->ExtensionBlockCount,
					   &GifFile->ExtensionBlocks, 
					   ExtFunction, ExtData[0], &ExtData[1])
		      == GIF_ERROR)
		      return (GIF_ERROR);
	      }
              for (;;) {
                  if (DGifGetExtensionNext(GifFile, &ExtData) == GIF_ERROR)
                      return (GIF_ERROR);
		  if (ExtData == NULL)
		      break;
                  /* Continue the extension block */
		  if (ExtData != NULL)
		      if (GifAddExtensionBlock(&GifFile->ExtensionBlockCount,
					       &GifFile->ExtensionBlocks,
					       CONTINUE_EXT_FUNC_CODE, 
					       ExtData[0], &ExtData[1]) == GIF_ERROR)
                      return (GIF_ERROR);
              }
              break;

          case TERMINATE_RECORD_TYPE:
              break;

          default:    /* Should be trapped by DGifGetRecordType */
              break;
        }
    } while (RecordType != TERMINATE_RECORD_TYPE);

    /* Sanity check for corrupted file */
    if (GifFile->ImageCount == 0) {
	GifFile->Error = D_GIF_ERR_NO_IMAG_DSCR;
	return(GIF_ERROR);
    }

    return (GIF_OK);
}

/* end */
