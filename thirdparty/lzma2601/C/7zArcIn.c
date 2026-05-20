/* 7zArcIn.c -- 7z Input functions
: Igor Pavlov : Public domain */

#include "Precomp.h"

#include <string.h>

#include "7z.h"
#include "7zBuf.h"
#include "7zCrc.h"
#include "CpuArch.h"

#define MY_ALLOC(T, p, size, alloc) \
  { if ((p = (T *)ISzAlloc_Alloc(alloc, (size) * sizeof(T))) == NULL) return SZ_ERROR_MEM; }

#define MY_ALLOC_ZE(T, p, size, alloc) \
  { if ((size) == 0) p = NULL; else MY_ALLOC(T, p, size, alloc) }

#define MY_ALLOC_AND_CPY(to, size, from, alloc) \
  { MY_ALLOC(Byte, to, size, alloc); memcpy(to, from, size); }

#define MY_ALLOC_ZE_AND_CPY(to, size, from, alloc) \
  { if ((size) == 0) to = NULL; else { MY_ALLOC_AND_CPY(to, size, from, alloc) } }

#define k7zMajorVersion 0

enum EIdEnum
{
  k7zIdEnd,
  k7zIdHeader,
  k7zIdArchiveProperties,
  k7zIdAdditionalStreamsInfo,
  k7zIdMainStreamsInfo,
  k7zIdFilesInfo,
  k7zIdPackInfo,
  k7zIdUnpackInfo,
  k7zIdSubStreamsInfo,
  k7zIdSize,
  k7zIdCRC,
  k7zIdFolder,
  k7zIdCodersUnpackSize,
  k7zIdNumUnpackStream,
  k7zIdEmptyStream,
  k7zIdEmptyFile,
  k7zIdAnti,
  k7zIdName,
  k7zIdCTime,
  k7zIdATime,
  k7zIdMTime,
  k7zIdWinAttrib,
  k7zIdComment,
  k7zIdEncodedHeader,
  k7zIdStartPos,
  k7zIdDummy
  // k7zNtSecure,
  // k7zParent,
  // k7zIsReal
};

const Byte k7zSignature[k7zSignatureSize] = {'7', 'z', 0xBC, 0xAF, 0x27, 0x1C};

#define SzBitUi32s_INIT(p) { (p)->Defs = NULL; (p)->Vals = NULL; }

static SRes SzBitUi32s_Alloc(CSzBitUi32s *p, size_t num, ISzAllocPtr alloc)
{
  if (num == 0)
  {
    p->Defs = NULL;
    p->Vals = NULL;
  }
  else
  {
    MY_ALLOC(Byte, p->Defs, (num + 7) >> 3, alloc)
    MY_ALLOC(UInt32, p->Vals, num, alloc)
  }
  return SZ_OK;
}

static void SzBitUi32s_Free(CSzBitUi32s *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->Defs); p->Defs = NULL;
  ISzAlloc_Free(alloc, p->Vals); p->Vals = NULL;
}

#define SzBitUi64s_INIT(p) { (p)->Defs = NULL; (p)->Vals = NULL; }

static void SzBitUi64s_Free(CSzBitUi64s *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->Defs); p->Defs = NULL;
  ISzAlloc_Free(alloc, p->Vals); p->Vals = NULL;
}


static void SzAr_Init(CSzAr *p)
{
  p->NumPackStreams = 0;
  p->NumFolders = 0;
  
  p->PackPositions = NULL;
  SzBitUi32s_INIT(&p->FolderCRCs)

  p->FoCodersOffsets = NULL;
  p->FoStartPackStreamIndex = NULL;
  p->FoToCoderUnpackSizes = NULL;
  p->FoToMainUnpackSizeIndex = NULL;
  p->CoderUnpackSizes = NULL;

  p->CodersData = NULL;

  p->RangeLimit = 0;
}

static void SzAr_Free(CSzAr *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->PackPositions);
  SzBitUi32s_Free(&p->FolderCRCs, alloc);
 
  ISzAlloc_Free(alloc, p->FoCodersOffsets);
  ISzAlloc_Free(alloc, p->FoStartPackStreamIndex);
  ISzAlloc_Free(alloc, p->FoToCoderUnpackSizes);
  ISzAlloc_Free(alloc, p->FoToMainUnpackSizeIndex);
  ISzAlloc_Free(alloc, p->CoderUnpackSizes);
  
  ISzAlloc_Free(alloc, p->CodersData);

  SzAr_Init(p);
}


void SzArEx_Init(CSzArEx *p)
{
  SzAr_Init(&p->db);
  
  p->NumFiles = 0;
  p->dataPos = 0;
  
  p->UnpackPositions = NULL;
  p->IsDirs = NULL;
  
  p->FolderToFile = NULL;
  p->FileToFolder = NULL;
  
  p->FileNameOffsets = NULL;
  p->FileNames = NULL;
  
  SzBitUi32s_INIT(&p->CRCs)
  SzBitUi32s_INIT(&p->Attribs)
  // SzBitUi32s_INIT(&p->Parents)
  SzBitUi64s_INIT(&p->MTime)
  SzBitUi64s_INIT(&p->CTime)
}

void SzArEx_Free(CSzArEx *p, ISzAllocPtr alloc)
{
  ISzAlloc_Free(alloc, p->UnpackPositions);
  ISzAlloc_Free(alloc, p->IsDirs);

  ISzAlloc_Free(alloc, p->FolderToFile);
  ISzAlloc_Free(alloc, p->FileToFolder);

  ISzAlloc_Free(alloc, p->FileNameOffsets);
  ISzAlloc_Free(alloc, p->FileNames);

  SzBitUi32s_Free(&p->CRCs, alloc);
  SzBitUi32s_Free(&p->Attribs, alloc);
  // SzBitUi32s_Free(&p->Parents, alloc);
  SzBitUi64s_Free(&p->MTime, alloc);
  SzBitUi64s_Free(&p->CTime, alloc);
  
  SzAr_Free(&p->db, alloc);
  SzArEx_Init(p);
}


static int TestSignatureCandidate(const Byte *testBytes)
{
  unsigned i;
  for (i = 0; i < k7zSignatureSize; i++)
    if (testBytes[i] != k7zSignature[i])
      return 0;
  return 1;
}

#define SzData_CLEAR(p) { (p)->Data = NULL; (p)->Size = 0; }

#define SZ_READ_BYTE_SD_NOCHECK(_sd_, dest) \
    (_sd_)->Size--; dest = *(_sd_)->Data++;

#define SZ_READ_BYTE_SD(_sd_, dest) \
    if ((_sd_)->Size == 0) return SZ_ERROR_ARCHIVE; \
    SZ_READ_BYTE_SD_NOCHECK(_sd_, dest)

#define SZ_READ_BYTE(dest) SZ_READ_BYTE_SD(sd, dest)

#define SZ_READ_BYTE_2(dest) \
    if (sd.Size == 0) return SZ_ERROR_ARCHIVE; \
    sd.Size--; dest = *sd.Data++;

#define SKIP_DATA(sd, size) { sd->Size -= (size_t)(size); sd->Data += (size_t)(size); }
#define SKIP_DATA2(sd, size) { sd.Size -= (size_t)(size); sd.Data += (size_t)(size); }

#define SZ_READ_32(dest) if (sd.Size < 4) return SZ_ERROR_ARCHIVE; \
   dest = GetUi32(sd.Data); SKIP_DATA2(sd, 4);

static Z7_NO_INLINE SRes ReadNumber(CSzData *sd, UInt64 *value)
{
  Byte firstByte, mask;
  unsigned i;
  UInt32 v;

  SZ_READ_BYTE(firstByte)
  if ((firstByte & 0x80) == 0)
  {
    *value = firstByte;
    return SZ_OK;
  }
  SZ_READ_BYTE(v)
  if ((firstByte & 0x40) == 0)
  {
    *value = (((UInt32)firstByte & 0x3F) << 8) | v;
    return SZ_OK;
  }
  SZ_READ_BYTE(mask)
  *value = v | ((UInt32)mask << 8);
  mask = 0x20;
  for (i = 2; i < 8; i++)
  {
    Byte b;
    if ((firstByte & mask) == 0)
    {
      const UInt64 highPart = (unsigned)firstByte & (unsigned)(mask - 1);
      *value |= (highPart << (8 * i));
      return SZ_OK;
    }
    SZ_READ_BYTE(b)
    *value |= ((UInt64)b << (8 * i));
    mask >>= 1;
  }
  return SZ_OK;
}


static Z7_NO_INLINE SRes SzReadNumber32(CSzData *sd, UInt32 *value)
{
  Byte firstByte;
  UInt64 value64;
  if (sd->Size == 0)
    return SZ_ERROR_ARCHIVE;
  firstByte = *sd->Data;
  if ((firstByte & 0x80) == 0)
  {
    *value = firstByte;
    sd->Data++;
    sd->Size--;
    return SZ_OK;
  }
  RINOK(ReadNumber(sd, &value64))
  if (value64 >= (UInt32)0x80000000 - 1)
    return SZ_ERROR_UNSUPPORTED;
  if (value64 >= ((UInt64)(1) << ((sizeof(size_t) - 1) * 8 + 4)))
    return SZ_ERROR_UNSUPPORTED;
  *value = (UInt32)value64;
  return SZ_OK;
}

#define ReadID(sd, value) ReadNumber(sd, value)

static SRes SkipData(CSzData *sd)
{
  UInt64 size;
  RINOK(ReadNumber(sd, &size))
  if (size > sd->Size)
    return SZ_ERROR_ARCHIVE;
  SKIP_DATA(sd, size)
  return SZ_OK;
}

static SRes WaitId(CSzData *sd, UInt32 id)
{
  for (;;)
  {
    UInt64 type;
    RINOK(ReadID(sd, &type))
    if (type == id)
      return SZ_OK;
    if (type == k7zIdEnd)
      return SZ_ERROR_ARCHIVE;
    RINOK(SkipData(sd))
  }
}

static SRes RememberBitVector(CSzData *sd, size_t numItems, const Byte **v)
{
  const size_t numBytes = (numItems + 7) >> 3;
  if (numBytes > sd->Size)
    return SZ_ERROR_ARCHIVE;
  *v = sd->Data;
  SKIP_DATA(sd, numBytes)
  return SZ_OK;
}

static UInt32 CountDefinedBits(const Byte *bits, UInt32 numItems)
{
  unsigned b = 0;
  unsigned m = 0;
  UInt32 sum = 0;
  for (; numItems != 0; numItems--)
  {
    if (m == 0)
    {
      b = *bits++;
      m = 8;
    }
    m--;
    sum += (UInt32)((b >> m) & 1);
  }
  return sum;
}

static Z7_NO_INLINE SRes ReadBitVector(CSzData *sd, size_t numItems, Byte **v, ISzAllocPtr alloc)
{
  Byte allAreDefined;
  Byte *v2;
  const size_t numBytes = (numItems + 7) >> 3;
  *v = NULL;
  SZ_READ_BYTE(allAreDefined)
  if (numBytes == 0)
    return SZ_OK;
  if (allAreDefined == 0)
  {
    if (numBytes > sd->Size)
      return SZ_ERROR_ARCHIVE;
    MY_ALLOC_AND_CPY(*v, numBytes, sd->Data, alloc)
    SKIP_DATA(sd, numBytes)
    return SZ_OK;
  }
  MY_ALLOC(Byte, *v, numBytes, alloc)
  v2 = *v;
  memset(v2, 0xFF, (size_t)numBytes);
  {
    const unsigned numBits = (unsigned)numItems & 7;
    if (numBits != 0)
      v2[(size_t)numBytes - 1] = (Byte)((((UInt32)1 << numBits) - 1) << (8 - numBits));
  }
  return SZ_OK;
}

static Z7_NO_INLINE SRes ReadUi32s(CSzData *sd2, size_t numItems, CSzBitUi32s *crcs, ISzAllocPtr alloc)
{
  size_t i;
  CSzData sd;
  UInt32 *vals;
  const Byte *defs;
  MY_ALLOC_ZE(UInt32, crcs->Vals, numItems, alloc)
  sd = *sd2;
  defs = crcs->Defs;
  vals = crcs->Vals;
  for (i = 0; i < numItems; i++)
    if (SzBitArray_Check(defs, i))
    {
      SZ_READ_32(vals[i])
    }
    else
      vals[i] = 0;
  *sd2 = sd;
  return SZ_OK;
}

static SRes ReadBitUi32s(CSzData *sd, size_t numItems, CSzBitUi32s *crcs, ISzAllocPtr alloc)
{
  SzBitUi32s_Free(crcs, alloc);
  RINOK(ReadBitVector(sd, numItems, &crcs->Defs, alloc))
  return ReadUi32s(sd, numItems, crcs, alloc);
}

static SRes SkipBitUi32s(CSzData *sd, UInt32 numItems)
{
  Byte allAreDefined;
  UInt32 numDefined = numItems;
  SZ_READ_BYTE(allAreDefined)
  if (!allAreDefined)
  {
    const size_t numBytes = (numItems + 7) >> 3;
    if (numBytes > sd->Size)
      return SZ_ERROR_ARCHIVE;
    numDefined = CountDefinedBits(sd->Data, numItems);
    SKIP_DATA(sd, numBytes)
  }
  if (numDefined > (sd->Size >> 2))
    return SZ_ERROR_ARCHIVE;
  SKIP_DATA(sd, (size_t)numDefined * 4)
  return SZ_OK;
}

static SRes ReadPackInfo(CSzAr *p, CSzData *sd, ISzAllocPtr alloc)
{
  RINOK(SzReadNumber32(sd, &p->NumPackStreams))

  RINOK(WaitId(sd, k7zIdSize))
  MY_ALLOC(UInt64, p->PackPositions, (size_t)p->NumPackStreams + 1, alloc)
  {
    UInt64 sum = 0;
    UInt32 i;
    const UInt32 numPackStreams = p->NumPackStreams;
    for (i = 0; i < numPackStreams; i++)
    {
      UInt64 packSize;
      p->PackPositions[i] = sum;
      RINOK(ReadNumber(sd, &packSize))
      sum += packSize;
      if (sum < packSize)
        return SZ_ERROR_ARCHIVE;
    }
    p->PackPositions[i] = sum;
  }

  for (;;)
  {
    UInt64 type;
    RINOK(ReadID(sd, &type))
    if (type == k7zIdEnd)
      return SZ_OK;
    if (type == k7zIdCRC)
    {
      /* CRC of packed streams is unused now */
      RINOK(SkipBitUi32s(sd, p->NumPackStreams))
      continue;
    }
    RINOK(SkipData(sd))
  }
}

/*
static SRes SzReadSwitch(CSzData *sd)
{
  Byte external;
  RINOK(SzReadByte(sd, &external));
  return (external == 0) ? SZ_OK: SZ_ERROR_UNSUPPORTED;
}
*/

#define k_NumCodersStreams_in_Folder_MAX (SZ_NUM_BONDS_IN_FOLDER_MAX + SZ_NUM_PACK_STREAMS_IN_FOLDER_MAX)

SRes SzGetNextFolderItem(CSzFolder *f, CSzData *sd)
{
  UInt32 numCoders, i;
  UInt32 numInStreams = 0;
  const Byte *dataStart = sd->Data;

  f->NumCoders = 0;
  f->NumBonds = 0;
  f->NumPackStreams = 0;
  f->UnpackStream = 0;
  
  RINOK(SzReadNumber32(sd, &numCoders))
  if (numCoders == 0 || numCoders > SZ_NUM_CODERS_IN_FOLDER_MAX)
    return SZ_ERROR_UNSUPPORTED;
  
  for (i = 0; i < numCoders; i++)
  {
    Byte mainByte;
    CSzCoderInfo *coder = f->Coders + i;
    unsigned idSize, j;
    UInt64 id;
    
    SZ_READ_BYTE(mainByte)
    if ((mainByte & 0xC0) != 0)
      return SZ_ERROR_UNSUPPORTED;
    
    idSize = (unsigned)(mainByte & 0xF);
    if (idSize > sizeof(id))
      return SZ_ERROR_UNSUPPORTED;
    if (idSize > sd->Size)
      return SZ_ERROR_ARCHIVE;
    id = 0;
    for (j = 0; j < idSize; j++)
    {
      id = ((id << 8) | *sd->Data);
      sd->Data++;
      sd->Size--;
    }
    if (id > (UInt32)0xFFFFFFFF)
      return SZ_ERROR_UNSUPPORTED;
    coder->MethodID = (UInt32)id;
    
    coder->NumStreams = 1;
    coder->PropsOffset = 0;
    coder->PropsSize = 0;
    
    if ((mainByte & 0x10) != 0)
    {
      UInt32 numStreams;
      
      RINOK(SzReadNumber32(sd, &numStreams))
      if (numStreams > k_NumCodersStreams_in_Folder_MAX)
        return SZ_ERROR_UNSUPPORTED;
      coder->NumStreams = (Byte)numStreams;

      RINOK(SzReadNumber32(sd, &numStreams))
      if (numStreams != 1)
        return SZ_ERROR_UNSUPPORTED;
    }

    numInStreams += coder->NumStreams;

    if (numInStreams > k_NumCodersStreams_in_Folder_MAX)
      return SZ_ERROR_UNSUPPORTED;

    if ((mainByte & 0x20) != 0)
    {
      UInt32 propsSize = 0;
      RINOK(SzReadNumber32(sd, &propsSize))
      if (propsSize > sd->Size)
        return SZ_ERROR_ARCHIVE;
      if (propsSize >= 0x80)
        return SZ_ERROR_UNSUPPORTED;
      coder->PropsOffset = (size_t)(sd->Data - dataStart);
      coder->PropsSize = (Byte)propsSize;
      sd->Data += (size_t)propsSize;
      sd->Size -= (size_t)propsSize;
    }
  }

  /*
  if (numInStreams == 1 && numCoders == 1)
  {
    f->NumPackStreams = 1;
    f->PackStreams[0] = 0;
  }
  else
  */
  {
    Byte streamUsed[k_NumCodersStreams_in_Folder_MAX];
    UInt32 numBonds, numPackStreams;
    
    numBonds = numCoders - 1;
    if (numInStreams < numBonds)
      return SZ_ERROR_ARCHIVE;
    if (numBonds > SZ_NUM_BONDS_IN_FOLDER_MAX)
      return SZ_ERROR_UNSUPPORTED;
    f->NumBonds = numBonds;
    
    numPackStreams = numInStreams - numBonds;
    if (numPackStreams > SZ_NUM_PACK_STREAMS_IN_FOLDER_MAX)
      return SZ_ERROR_UNSUPPORTED;
    f->NumPackStreams = numPackStreams;
  
    for (i = 0; i < numInStreams; i++)
      streamUsed[i] = False;
    
    if (numBonds != 0)
    {
      Byte coderUsed[SZ_NUM_CODERS_IN_FOLDER_MAX];

      for (i = 0; i < numCoders; i++)
        coderUsed[i] = False;
      
      for (i = 0; i < numBonds; i++)
      {
        CSzBond *bp = f->Bonds + i;
        
        RINOK(SzReadNumber32(sd, &bp->InIndex))
        if (bp->InIndex >= numInStreams || streamUsed[bp->InIndex])
          return SZ_ERROR_ARCHIVE;
        streamUsed[bp->InIndex] = True;
        
        RINOK(SzReadNumber32(sd, &bp->OutIndex))
        if (bp->OutIndex >= numCoders || coderUsed[bp->OutIndex])
          return SZ_ERROR_ARCHIVE;
        coderUsed[bp->OutIndex] = True;
      }
      
      for (i = 0; i < numCoders; i++)
        if (!coderUsed[i])
        {
          f->UnpackStream = i;
          break;
        }
      
      if (i == numCoders)
        return SZ_ERROR_ARCHIVE;
    }
    
    if (numPackStreams == 1)
    {
      for (i = 0; i < numInStreams; i++)
        if (!streamUsed[i])
          break;
      if (i == numInStreams)
        return SZ_ERROR_ARCHIVE;
      f->PackStreams[0] = i;
    }
    else
      for (i = 0; i < numPackStreams; i++)
      {
        UInt32 index;
        RINOK(SzReadNumber32(sd, &index))
        if (index >= numInStreams || streamUsed[index])
          return SZ_ERROR_ARCHIVE;
        streamUsed[index] = True;
        f->PackStreams[i] = index;
      }
  }

  f->NumCoders = numCoders;

  return SZ_OK;
}


static Z7_NO_INLINE SRes SkipNumbers(CSzData *sd2, UInt32 num)
{
  CSzData sd;
  sd = *sd2;
  for (; num != 0; num--)
  {
    Byte firstByte, mask;
    unsigned i;
    SZ_READ_BYTE_2(firstByte)
    if ((firstByte & 0x80) == 0)
      continue;
    if ((firstByte & 0x40) == 0)
    {
      if (sd.Size == 0)
        return SZ_ERROR_ARCHIVE;
      sd.Size--;
      sd.Data++;
      continue;
    }
    mask = 0x20;
    for (i = 2; i < 8 && (firstByte & mask) != 0; i++)
      mask >>= 1;
    if (i > sd.Size)
      return SZ_ERROR_ARCHIVE;
    SKIP_DATA2(sd, i)
  }
  *sd2 = sd;
  return SZ_OK;
}


#define k_Scan_NumCoders_MAX 64
#define k_Scan_NumCodersStreams_in_Folder_MAX 64


static SRes ReadUnpackInfo(CSzAr *p,
    CSzData *sd2,
    UInt32 numFoldersMax,
    const CBuf *tempBufs, UInt32 numTempBufs,
    ISzAllocPtr alloc)
{
  CSzData sd;
  
  UInt32 fo, numFolders, numCodersOutStreams, packStreamIndex;
  const Byte *startBufPtr;
  Byte external;
  
  RINOK(WaitId(sd2, k7zIdFolder))
  
  RINOK(SzReadNumber32(sd2, &numFolders))
  if (numFolders > numFoldersMax)
    return SZ_ERROR_UNSUPPORTED;
  p->NumFolders = numFolders;

  SZ_READ_BYTE_SD(sd2, external)
  if (external == 0)
    sd = *sd2;
  else
  {
    UInt32 index;
    RINOK(SzReadNumber32(sd2, &index))
    if (index >= numTempBufs)
      return SZ_ERROR_ARCHIVE;
    sd.Data = tempBufs[index].data;
    sd.Size = tempBufs[index].size;
  }
  
  MY_ALLOC(size_t, p->FoCodersOffsets, (size_t)numFolders + 1, alloc)
  MY_ALLOC(UInt32, p->FoStartPackStreamIndex, (size_t)numFolders + 1, alloc)
  MY_ALLOC(UInt32, p->FoToCoderUnpackSizes, (size_t)numFolders + 1, alloc)
  MY_ALLOC_ZE(Byte, p->FoToMainUnpackSizeIndex, (size_t)numFolders, alloc)
  
  startBufPtr = sd.Data;
  
  packStreamIndex = 0;
  numCodersOutStreams = 0;

  for (fo = 0; fo < numFolders; fo++)
  {
    UInt32 numCoders, ci, numInStreams = 0;
    
    p->FoCodersOffsets[fo] = (size_t)(sd.Data - startBufPtr);
    
    RINOK(SzReadNumber32(&sd, &numCoders))
    if (numCoders == 0 || numCoders > k_Scan_NumCoders_MAX)
      return SZ_ERROR_UNSUPPORTED;
    
    for (ci = 0; ci < numCoders; ci++)
    {
      Byte mainByte;
      unsigned idSize;
      UInt32 coderInStreams;
      
      SZ_READ_BYTE_2(mainByte)
      if ((mainByte & 0xC0) != 0)
        return SZ_ERROR_UNSUPPORTED;
      idSize = (mainByte & 0xF);
      if (idSize > 8)
        return SZ_ERROR_UNSUPPORTED;
      if (idSize > sd.Size)
        return SZ_ERROR_ARCHIVE;
      SKIP_DATA2(sd, idSize)
      
      coderInStreams = 1;
      
      if ((mainByte & 0x10) != 0)
      {
        UInt32 coderOutStreams;
        RINOK(SzReadNumber32(&sd, &coderInStreams))
        RINOK(SzReadNumber32(&sd, &coderOutStreams))
        if (coderInStreams > k_Scan_NumCodersStreams_in_Folder_MAX || coderOutStreams != 1)
          return SZ_ERROR_UNSUPPORTED;
      }
      
      numInStreams += coderInStreams;

      if ((mainByte & 0x20) != 0)
      {
        UInt32 propsSize;
        RINOK(SzReadNumber32(&sd, &propsSize))
        if (propsSize > sd.Size)
          return SZ_ERROR_ARCHIVE;
        SKIP_DATA2(sd, propsSize)
      }
    }
    
    {
      UInt32 indexOfMainStream = 0;
      UInt32 numPackStreams = 1;
      
      if (numCoders != 1 || numInStreams != 1)
      {
        Byte streamUsed[k_Scan_NumCodersStreams_in_Folder_MAX];
        Byte coderUsed[k_Scan_NumCoders_MAX];
    
        UInt32 i;
        const UInt32 numBonds = numCoders - 1;
        if (numInStreams < numBonds)
          return SZ_ERROR_ARCHIVE;
        
        if (numInStreams > k_Scan_NumCodersStreams_in_Folder_MAX)
          return SZ_ERROR_UNSUPPORTED;
        
        for (i = 0; i < numInStreams; i++)
          streamUsed[i] = False;
        for (i = 0; i < numCoders; i++)
          coderUsed[i] = False;
        
        for (i = 0; i < numBonds; i++)
        {
          UInt32 index;
          
          RINOK(SzReadNumber32(&sd, &index))
          if (index >= numInStreams || streamUsed[index])
            return SZ_ERROR_ARCHIVE;
          streamUsed[index] = True;
          
          RINOK(SzReadNumber32(&sd, &index))
          if (index >= numCoders || coderUsed[index])
            return SZ_ERROR_ARCHIVE;
          coderUsed[index] = True;
        }
        
        numPackStreams = numInStreams - numBonds;
        
        if (numPackStreams != 1)
          for (i = 0; i < numPackStreams; i++)
          {
            UInt32 index;
            RINOK(SzReadNumber32(&sd, &index))
            if (index >= numInStreams || streamUsed[index])
              return SZ_ERROR_ARCHIVE;
            streamUsed[index] = True;
          }
          
        for (i = 0; i < numCoders; i++)
          if (!coderUsed[i])
          {
            indexOfMainStream = i;
            break;
          }
 
        if (i == numCoders)
          return SZ_ERROR_ARCHIVE;
      }
      
      p->FoStartPackStreamIndex[fo] = packStreamIndex;
      p->FoToCoderUnpackSizes[fo] = numCodersOutStreams;
      p->FoToMainUnpackSizeIndex[fo] = (Byte)indexOfMainStream;
      numCodersOutStreams += numCoders;
      if (numCodersOutStreams < numCoders)
        return SZ_ERROR_UNSUPPORTED;
      if (numPackStreams > p->NumPackStreams - packStreamIndex)
        return SZ_ERROR_ARCHIVE;
      packStreamIndex += numPackStreams;
    }
  }

  p->FoToCoderUnpackSizes[fo] = numCodersOutStreams;
  
  {
    const size_t dataSize = (size_t)(sd.Data - startBufPtr);
    p->FoStartPackStreamIndex[fo] = packStreamIndex;
    p->FoCodersOffsets[fo] = dataSize;
    MY_ALLOC_ZE_AND_CPY(p->CodersData, dataSize, startBufPtr, alloc)
  }
  
  if (external != 0)
  {
    if (sd.Size != 0)
      return SZ_ERROR_ARCHIVE;
    sd = *sd2;
  }
  
  RINOK(WaitId(&sd, k7zIdCodersUnpackSize))
  
  MY_ALLOC_ZE(UInt64, p->CoderUnpackSizes, (size_t)numCodersOutStreams, alloc)
  {
    UInt32 i;
    for (i = 0; i < numCodersOutStreams; i++)
    {
      RINOK(ReadNumber(&sd, p->CoderUnpackSizes + i))
    }
  }

  for (;;)
  {
    UInt64 type;
    RINOK(ReadID(&sd, &type))
    if (type == k7zIdEnd)
    {
      *sd2 = sd;
      return SZ_OK;
    }
    if (type == k7zIdCRC)
    {
      RINOK(ReadBitUi32s(&sd, numFolders, &p->FolderCRCs, alloc))
      continue;
    }
    RINOK(SkipData(&sd))
  }
}


UInt64 SzAr_GetFolderUnpackSize(const CSzAr *p, UInt32 folderIndex)
{
  return p->CoderUnpackSizes[p->FoToCoderUnpackSizes[folderIndex] + p->FoToMainUnpackSizeIndex[folderIndex]];
}


typedef struct
{
  UInt32 NumTotalSubStreams;
  UInt32 NumSubDigests;
  CSzData sdNumSubStreams;
  CSzData sdSizes;
  CSzData sdCRCs;
} CSubStreamInfo;


static SRes ReadSubStreamsInfo(CSzAr *p, CSzData *sd, CSubStreamInfo *ssi)
{
  UInt64 type = 0;
  UInt32 numSubDigests = 0;
  const UInt32 numFolders = p->NumFolders;
  UInt32 numUnpackStreams = numFolders;
  UInt32 numUnpackSizesInData = 0;

  for (;;)
  {
    RINOK(ReadID(sd, &type))
    if (type == k7zIdNumUnpackStream)
    {
      UInt32 i;
      ssi->sdNumSubStreams.Data = sd->Data;
      numUnpackStreams = 0;
      numSubDigests = 0;
      for (i = 0; i < numFolders; i++)
      {
        UInt32 numStreams;
        RINOK(SzReadNumber32(sd, &numStreams))
        if (numUnpackStreams > numUnpackStreams + numStreams)
          return SZ_ERROR_UNSUPPORTED;
        numUnpackStreams += numStreams;
        if (numStreams != 0)
          numUnpackSizesInData += (numStreams - 1);
        if (numStreams != 1 || !SzBitWithVals_Check(&p->FolderCRCs, i))
          numSubDigests += numStreams;
      }
      ssi->sdNumSubStreams.Size = (size_t)(sd->Data - ssi->sdNumSubStreams.Data);
      continue;
    }
    if (type == k7zIdCRC || type == k7zIdSize || type == k7zIdEnd)
      break;
    RINOK(SkipData(sd))
  }

  if (!ssi->sdNumSubStreams.Data)
  {
    numSubDigests = numFolders;
    if (p->FolderCRCs.Defs)
      numSubDigests = numFolders - CountDefinedBits(p->FolderCRCs.Defs, numFolders);
  }
  
  ssi->NumTotalSubStreams = numUnpackStreams;
  ssi->NumSubDigests = numSubDigests;

  if (type == k7zIdSize)
  {
    ssi->sdSizes.Data = sd->Data;
    RINOK(SkipNumbers(sd, numUnpackSizesInData))
    ssi->sdSizes.Size = (size_t)(sd->Data - ssi->sdSizes.Data);
    RINOK(ReadID(sd, &type))
  }

  for (;;)
  {
    if (type == k7zIdEnd)
      return SZ_OK;
    if (type == k7zIdCRC)
    {
      ssi->sdCRCs.Data = sd->Data;
      RINOK(SkipBitUi32s(sd, numSubDigests))
      ssi->sdCRCs.Size = (size_t)(sd->Data - ssi->sdCRCs.Data);
    }
    else
    {
      RINOK(SkipData(sd))
    }
    RINOK(ReadID(sd, &type))
  }
}

static SRes SzReadStreamsInfo(CSzAr *p,
    CSzData *sd,
    UInt32 numFoldersMax, const CBuf *tempBufs, UInt32 numTempBufs,
    UInt64 *dataOffset,
    CSubStreamInfo *ssi,
    ISzAllocPtr alloc)
{
  UInt64 type;

  SzData_CLEAR(&ssi->sdSizes)
  SzData_CLEAR(&ssi->sdCRCs)
  SzData_CLEAR(&ssi->sdNumSubStreams)

  *dataOffset = 0;
  RINOK(ReadID(sd, &type))
  if (type == k7zIdPackInfo)
  {
    RINOK(ReadNumber(sd, dataOffset))
    if (*dataOffset > p->RangeLimit)
      return SZ_ERROR_ARCHIVE;
    RINOK(ReadPackInfo(p, sd, alloc))
    if (p->PackPositions[p->NumPackStreams] > p->RangeLimit - *dataOffset)
      return SZ_ERROR_ARCHIVE;
    RINOK(ReadID(sd, &type))
  }
  if (type == k7zIdUnpackInfo)
  {
    RINOK(ReadUnpackInfo(p, sd, numFoldersMax, tempBufs, numTempBufs, alloc))
    RINOK(ReadID(sd, &type))
  }
  if (type == k7zIdSubStreamsInfo)
  {
    RINOK(ReadSubStreamsInfo(p, sd, ssi))
    RINOK(ReadID(sd, &type))
  }
  else
  {
    ssi->NumTotalSubStreams = p->NumFolders;
    // ssi->NumSubDigests = 0;
  }

  return (type == k7zIdEnd ? SZ_OK : SZ_ERROR_UNSUPPORTED);
}

static SRes SzReadAndDecodePackedStreams(
    ILookInStreamPtr inStream,
    CSzData *sd,
    CBuf *tempBufs,
    UInt32 numFoldersMax,
    UInt64 baseOffset,
    CSzAr *p,
    ISzAllocPtr allocTemp)
{
  UInt64 dataStartPos;
  UInt32 fo;
  CSubStreamInfo ssi;

  RINOK(SzReadStreamsInfo(p, sd, numFoldersMax, NULL, 0, &dataStartPos, &ssi, allocTemp))
  
  dataStartPos += baseOffset;
  if (p->NumFolders == 0)
    return SZ_ERROR_ARCHIVE;
 
  for (fo = 0; fo < p->NumFolders; fo++)
    Buf_Init(tempBufs + fo);
  
  for (fo = 0; fo < p->NumFolders; fo++)
  {
    CBuf *tempBuf = tempBufs + fo;
    const UInt64 unpackSize = SzAr_GetFolderUnpackSize(p, fo);
    if ((size_t)unpackSize != unpackSize)
      return SZ_ERROR_MEM;
    if (!Buf_Create(tempBuf, (size_t)unpackSize, allocTemp))
      return SZ_ERROR_MEM;
  }
  
  for (fo = 0; fo < p->NumFolders; fo++)
  {
    const CBuf *tempBuf = tempBufs + fo;
    RINOK(LookInStream_SeekTo(inStream, dataStartPos))
    RINOK(SzAr_DecodeFolder(p, fo, inStream, dataStartPos, tempBuf->data, tempBuf->size, allocTemp))
  }
  
  return SZ_OK;
}

// (size & 1) == 0
// (data) is aligned for 2-bytes
static SRes SzReadFileNames(const Byte *data, size_t size, UInt32 numFiles, size_t *offsets)
{
  const Byte *p, *lim;
  *offsets++ = 0;
  if (numFiles == 0)
    return (size == 0) ? SZ_OK : SZ_ERROR_ARCHIVE;
  if (size < 2)
    return SZ_ERROR_ARCHIVE;
  lim = data + size;
  if (*(const UInt16 *)(const void *)(lim - 2))
    return SZ_ERROR_ARCHIVE;
  p = data;
  do
  {
    if (p >= lim)
      return SZ_ERROR_ARCHIVE;
    for (; *(const UInt16 *)(const void *)p; p += 2);
    p += 2;
    *offsets++ = (size_t)(p - data) >> 1;
  }
  while (--numFiles);
  return (p == lim) ? SZ_OK : SZ_ERROR_ARCHIVE;
}

static Z7_NO_INLINE SRes ReadTime(CSzBitUi64s *p, size_t num,
    CSzData *sd2,
    const CBuf *tempBufs, UInt32 numTempBufs,
    ISzAllocPtr alloc)
{
  CSzData sd;
  size_t i;
  CNtfsFileTime *vals;
  Byte *defs;
  Byte external;
  
  RINOK(ReadBitVector(sd2, num, &p->Defs, alloc))
  
  SZ_READ_BYTE_SD(sd2, external)
  if (external == 0)
    sd = *sd2;
  else
  {
    UInt32 index;
    RINOK(SzReadNumber32(sd2, &index))
    if (index >= numTempBufs)
      return SZ_ERROR_ARCHIVE;
    sd.Data = tempBufs[index].data;
    sd.Size = tempBufs[index].size;
  }
  
  MY_ALLOC_ZE(CNtfsFileTime, p->Vals, num, alloc)
  vals = p->Vals;
  defs = p->Defs;
  for (i = 0; i < num; i++)
    if (SzBitArray_Check(defs, i))
    {
      if (sd.Size < 8)
        return SZ_ERROR_ARCHIVE;
      vals[i].Low = GetUi32(sd.Data);
      vals[i].High = GetUi32(sd.Data + 4);
      SKIP_DATA2(sd, 8)
    }
    else
      vals[i].High = vals[i].Low = 0;
  
  if (external == 0)
    *sd2 = sd;
  
  return SZ_OK;
}


#define NUM_ADDITIONAL_STREAMS_MAX 8


static SRes SzReadHeader2(
    CSzArEx *p,   /* allocMain */
    CSzData *sd,
    ILookInStreamPtr inStream,
    CBuf *tempBufs, UInt32 *numTempBufs,
    ISzAllocPtr allocMain,
    ISzAllocPtr allocTemp
    )
{
  CSubStreamInfo ssi;

{
  UInt64 type;
  
  SzData_CLEAR(&ssi.sdSizes)
  SzData_CLEAR(&ssi.sdCRCs)
  SzData_CLEAR(&ssi.sdNumSubStreams)

  ssi.NumSubDigests = 0;
  ssi.NumTotalSubStreams = 0;

  RINOK(ReadID(sd, &type))

  if (type == k7zIdArchiveProperties)
  {
    for (;;)
    {
      UInt64 type2;
      RINOK(ReadID(sd, &type2))
      if (type2 == k7zIdEnd)
        break;
      RINOK(SkipData(sd))
    }
    RINOK(ReadID(sd, &type))
  }

  if (type == k7zIdAdditionalStreamsInfo)
  {
    CSzAr tempAr;
    SRes res;
    
    SzAr_Init(&tempAr);
    tempAr.RangeLimit = p->db.RangeLimit;

    res = SzReadAndDecodePackedStreams(inStream, sd, tempBufs, NUM_ADDITIONAL_STREAMS_MAX,
        p->startPosAfterHeader, &tempAr, allocTemp);
    *numTempBufs = tempAr.NumFolders;
    SzAr_Free(&tempAr, allocTemp);
    
    if (res != SZ_OK)
      return res;
    RINOK(ReadID(sd, &type))
  }

  if (type == k7zIdMainStreamsInfo)
  {
    RINOK(SzReadStreamsInfo(&p->db, sd, (UInt32)1 << 30, tempBufs, *numTempBufs,
        &p->dataPos, &ssi, allocMain))
    p->dataPos += p->startPosAfterHeader;
    RINOK(ReadID(sd, &type))
  }

  if (type == k7zIdEnd)
  {
    return SZ_OK;
  }

  if (type != k7zIdFilesInfo)
    return SZ_ERROR_ARCHIVE;
}

{
  UInt32 numFiles = 0;
  UInt32 numEmptyStreams = 0;
  const Byte *emptyStreams = NULL;
  const Byte *emptyFiles = NULL;
  
  RINOK(SzReadNumber32(sd, &numFiles))
  p->NumFiles = numFiles;

  for (;;)
  {
    UInt64 type;
    UInt64 size;
    RINOK(ReadID(sd, &type))
    if (type == k7zIdEnd)
      break;
    RINOK(ReadNumber(sd, &size))
    if (size > sd->Size)
      return SZ_ERROR_ARCHIVE;
    
    if (type >= ((UInt32)1 << 8))
    {
      SKIP_DATA(sd, size)
    }
    else switch ((unsigned)type)
    {
      case k7zIdName:
      {
        size_t namesSize;
        const Byte *namesData;
        Byte external;

        SZ_READ_BYTE(external)
        if (external == 0)
        {
          namesSize = (size_t)size - 1;
          namesData = sd->Data;
          SKIP_DATA(sd, namesSize)
        }
        else
        {
          UInt32 index;
          RINOK(SzReadNumber32(sd, &index))
          if (index >= *numTempBufs)
            return SZ_ERROR_ARCHIVE;
          namesData = (tempBufs)[index].data;
          namesSize = (tempBufs)[index].size;
        }

        if (namesSize & 1)
          return SZ_ERROR_ARCHIVE;
        MY_ALLOC(size_t, p->FileNameOffsets, numFiles + 1, allocMain)
        MY_ALLOC_ZE_AND_CPY(p->FileNames, namesSize, namesData, allocMain)
        RINOK(SzReadFileNames(p->FileNames, namesSize, numFiles, p->FileNameOffsets))
        break;
      }
      case k7zIdEmptyStream:
      {
        RINOK(RememberBitVector(sd, numFiles, &emptyStreams))
        numEmptyStreams = CountDefinedBits(emptyStreams, numFiles);
        emptyFiles = NULL;
        break;
      }
      case k7zIdEmptyFile:
      {
        RINOK(RememberBitVector(sd, numEmptyStreams, &emptyFiles))
        break;
      }
      case k7zIdWinAttrib:
      {
        Byte external;
        CSzData sdSwitch;
        CSzData *sdPtr;
        SzBitUi32s_Free(&p->Attribs, allocMain);
        RINOK(ReadBitVector(sd, numFiles, &p->Attribs.Defs, allocMain))

        SZ_READ_BYTE(external)
        if (external == 0)
          sdPtr = sd;
        else
        {
          UInt32 index;
          RINOK(SzReadNumber32(sd, &index))
          if (index >= *numTempBufs)
            return SZ_ERROR_ARCHIVE;
          sdSwitch.Data = (tempBufs)[index].data;
          sdSwitch.Size = (tempBufs)[index].size;
          sdPtr = &sdSwitch;
        }
        RINOK(ReadUi32s(sdPtr, numFiles, &p->Attribs, allocMain))
        break;
      }
      /*
      case k7zParent:
      {
        SzBitUi32s_Free(&p->Parents, allocMain);
        RINOK(ReadBitVector(sd, numFiles, &p->Parents.Defs, allocMain));
        RINOK(SzReadSwitch(sd));
        RINOK(ReadUi32s(sd, numFiles, &p->Parents, allocMain));
        break;
      }
      */
      case k7zIdMTime: RINOK(ReadTime(&p->MTime, numFiles, sd, tempBufs, *numTempBufs, allocMain)) break;
      case k7zIdCTime: RINOK(ReadTime(&p->CTime, numFiles, sd, tempBufs, *numTempBufs, allocMain)) break;
      default:
      {
        SKIP_DATA(sd, size)
      }
    }
  }

  if (numFiles - numEmptyStreams != ssi.NumTotalSubStreams)
    return SZ_ERROR_ARCHIVE;

  for (;;)
  {
    UInt64 type;
    RINOK(ReadID(sd, &type))
    if (type == k7zIdEnd)
      break;
    RINOK(SkipData(sd))
  }

  {
    UInt32 i;
    UInt32 emptyFileIndex = 0;
    UInt32 folderIndex = 0;
    UInt32 remSubStreams = 0;
    UInt32 numSubStreams = 0;
    UInt64 unpackPos = 0;
    const Byte *digestsDefs = NULL;
    const Byte *digestsVals = NULL;
    UInt32 digestIndex = 0;
    Byte isDirMask = 0;
    Byte crcMask = 0;
    Byte mask = 0x80;
    
    MY_ALLOC(UInt32, p->FolderToFile, p->db.NumFolders + 1, allocMain)
    MY_ALLOC_ZE(UInt32, p->FileToFolder, p->NumFiles, allocMain)
    MY_ALLOC(UInt64, p->UnpackPositions, p->NumFiles + 1, allocMain)
    MY_ALLOC_ZE(Byte, p->IsDirs, (p->NumFiles + 7) >> 3, allocMain)

    RINOK(SzBitUi32s_Alloc(&p->CRCs, p->NumFiles, allocMain))

    if (ssi.sdCRCs.Size != 0)
    {
      Byte allDigestsDefined = 0;
      SZ_READ_BYTE_SD_NOCHECK(&ssi.sdCRCs, allDigestsDefined)
      if (allDigestsDefined)
        digestsVals = ssi.sdCRCs.Data;
      else
      {
        const size_t numBytes = (ssi.NumSubDigests + 7) >> 3;
        digestsDefs = ssi.sdCRCs.Data;
        digestsVals = digestsDefs + numBytes;
      }
    }

    for (i = 0; i < numFiles; i++, mask >>= 1)
    {
      if (mask == 0)
      {
        const UInt32 byteIndex = (i - 1) >> 3;
        p->IsDirs[byteIndex] = isDirMask;
        p->CRCs.Defs[byteIndex] = crcMask;
        isDirMask = 0;
        crcMask = 0;
        mask = 0x80;
      }

      p->UnpackPositions[i] = unpackPos;
      p->CRCs.Vals[i] = 0;
      
      if (emptyStreams && SzBitArray_Check(emptyStreams, i))
      {
        if (emptyFiles)
        {
          if (!SzBitArray_Check(emptyFiles, emptyFileIndex))
            isDirMask |= mask;
          emptyFileIndex++;
        }
        else
          isDirMask |= mask;
        if (remSubStreams == 0)
        {
          p->FileToFolder[i] = (UInt32)-1;
          continue;
        }
      }
      
      if (remSubStreams == 0)
      {
        for (;;)
        {
          if (folderIndex >= p->db.NumFolders)
            return SZ_ERROR_ARCHIVE;
          p->FolderToFile[folderIndex] = i;
          numSubStreams = 1;
          if (ssi.sdNumSubStreams.Data)
          {
            RINOK(SzReadNumber32(&ssi.sdNumSubStreams, &numSubStreams))
          }
          remSubStreams = numSubStreams;
          if (numSubStreams != 0)
            break;
          {
            const UInt64 folderUnpackSize = SzAr_GetFolderUnpackSize(&p->db, folderIndex);
            unpackPos += folderUnpackSize;
            if (unpackPos < folderUnpackSize)
              return SZ_ERROR_ARCHIVE;
          }
          folderIndex++;
        }
      }
      
      p->FileToFolder[i] = folderIndex;
      
      if (emptyStreams && SzBitArray_Check(emptyStreams, i))
        continue;
      
      if (--remSubStreams == 0)
      {
        const UInt64 folderUnpackSize = SzAr_GetFolderUnpackSize(&p->db, folderIndex);
        const UInt64 startFolderUnpackPos = p->UnpackPositions[p->FolderToFile[folderIndex]];
        if (folderUnpackSize < unpackPos - startFolderUnpackPos)
          return SZ_ERROR_ARCHIVE;
        unpackPos = startFolderUnpackPos + folderUnpackSize;
        if (unpackPos < folderUnpackSize)
          return SZ_ERROR_ARCHIVE;

        if (numSubStreams == 1 && SzBitWithVals_Check(&p->db.FolderCRCs, folderIndex))
        {
          p->CRCs.Vals[i] = p->db.FolderCRCs.Vals[folderIndex];
          crcMask |= mask;
        }
        folderIndex++;
      }
      else
      {
        UInt64 v;
        RINOK(ReadNumber(&ssi.sdSizes, &v))
        unpackPos += v;
        if (unpackPos < v)
          return SZ_ERROR_ARCHIVE;
      }
      if ((crcMask & mask) == 0 && digestsVals)
      {
        if (!digestsDefs || SzBitArray_Check(digestsDefs, digestIndex))
        {
          p->CRCs.Vals[i] = GetUi32(digestsVals);
          digestsVals += 4;
          crcMask |= mask;
        }
        digestIndex++;
      }
    }

    if (mask != 0x80)
    {
      const UInt32 byteIndex = (i - 1) >> 3;
      p->IsDirs[byteIndex] = isDirMask;
      p->CRCs.Defs[byteIndex] = crcMask;
    }
    
    p->UnpackPositions[i] = unpackPos;

    if (remSubStreams != 0)
      return SZ_ERROR_ARCHIVE;

    for (;;)
    {
      p->FolderToFile[folderIndex] = i;
      if (folderIndex >= p->db.NumFolders)
        break;
      if (!ssi.sdNumSubStreams.Data)
        return SZ_ERROR_ARCHIVE;
      RINOK(SzReadNumber32(&ssi.sdNumSubStreams, &numSubStreams))
      if (numSubStreams != 0)
        return SZ_ERROR_ARCHIVE;
      /*
      {
        UInt64 folderUnpackSize = SzAr_GetFolderUnpackSize(&p->db, folderIndex);
        unpackPos += folderUnpackSize;
        if (unpackPos < folderUnpackSize)
          return SZ_ERROR_ARCHIVE;
      }
      */
      folderIndex++;
    }

    if (ssi.sdNumSubStreams.Data && ssi.sdNumSubStreams.Size != 0)
      return SZ_ERROR_ARCHIVE;
  }
}
  return SZ_OK;
}


static SRes SzReadHeader(
    CSzArEx *p,
    CSzData *sd,
    ILookInStreamPtr inStream,
    ISzAllocPtr allocMain,
    ISzAllocPtr allocTemp)
{
  UInt32 i;
  UInt32 numTempBufs = 0;
  SRes res;
  CBuf tempBufs[NUM_ADDITIONAL_STREAMS_MAX];

  for (i = 0; i < NUM_ADDITIONAL_STREAMS_MAX; i++)
    Buf_Init(tempBufs + i);
  
  res = SzReadHeader2(p, sd, inStream,
      tempBufs, &numTempBufs,
      allocMain, allocTemp);
  
  for (i = 0; i < NUM_ADDITIONAL_STREAMS_MAX; i++)
    Buf_Free(tempBufs + i, allocTemp);

  RINOK(res)

  if (sd->Size != 0)
    return SZ_ERROR_FAIL;

  return res;
}

static SRes SzArEx_Open2(
    CSzArEx *p,
    ILookInStreamPtr inStream,
    ISzAllocPtr allocMain,
    ISzAllocPtr allocTemp)
{
  Byte header[k7zStartHeaderSize];
  Int64 startArcPos;
  UInt64 nextHeaderOffset, nextHeaderSize;
  size_t nextHeaderSizeT;
  UInt32 nextHeaderCRC;
  CBuf buf;
  SRes res;

  startArcPos = 0;
  RINOK(ILookInStream_Seek(inStream, &startArcPos, SZ_SEEK_CUR))

  RINOK(LookInStream_Read2(inStream, header, k7zStartHeaderSize, SZ_ERROR_NO_ARCHIVE))

  if (!TestSignatureCandidate(header))
    return SZ_ERROR_NO_ARCHIVE;
  if (header[6] != k7zMajorVersion)
    return SZ_ERROR_UNSUPPORTED;

  nextHeaderOffset = GetUi64(header + 12);
  nextHeaderSize = GetUi64(header + 20);
  nextHeaderCRC = GetUi32(header + 28);

  p->startPosAfterHeader = (UInt64)startArcPos + k7zStartHeaderSize;
  
  if (CrcCalc(header + 12, 20) != GetUi32(header + 8))
    return SZ_ERROR_CRC;

  p->db.RangeLimit = nextHeaderOffset;

  nextHeaderSizeT = (size_t)nextHeaderSize;
  if (nextHeaderSizeT != nextHeaderSize)
    return SZ_ERROR_MEM;
  if (nextHeaderSizeT == 0)
    return SZ_OK;
  if (nextHeaderOffset > nextHeaderOffset + nextHeaderSize ||
      nextHeaderOffset > nextHeaderOffset + nextHeaderSize + k7zStartHeaderSize)
    return SZ_ERROR_NO_ARCHIVE;

  {
    Int64 pos = 0;
    RINOK(ILookInStream_Seek(inStream, &pos, SZ_SEEK_END))
    if ((UInt64)pos < (UInt64)startArcPos + nextHeaderOffset ||
        (UInt64)pos < (UInt64)startArcPos + k7zStartHeaderSize + nextHeaderOffset ||
        (UInt64)pos < (UInt64)startArcPos + k7zStartHeaderSize + nextHeaderOffset + nextHeaderSize)
      return SZ_ERROR_INPUT_EOF;
  }

  RINOK(LookInStream_SeekTo(inStream, (UInt64)startArcPos + k7zStartHeaderSize + nextHeaderOffset))

  if (!Buf_Create(&buf, nextHeaderSizeT, allocTemp))
    return SZ_ERROR_MEM;

  res = LookInStream_Read(inStream, buf.data, nextHeaderSizeT);
  
  if (res == SZ_OK)
  {
    res = SZ_ERROR_ARCHIVE;
    if (CrcCalc(buf.data, nextHeaderSizeT) == nextHeaderCRC)
    {
      CSzData sd;
      UInt64 type;
      sd.Data = buf.data;
      sd.Size = buf.size;
      
      res = ReadID(&sd, &type);
      
      if (res == SZ_OK && type == k7zIdEncodedHeader)
      {
        CSzAr tempAr;
        CBuf tempBuf;
        Buf_Init(&tempBuf);
        
        SzAr_Init(&tempAr);
        tempAr.RangeLimit = p->db.RangeLimit;

        res = SzReadAndDecodePackedStreams(inStream, &sd, &tempBuf, 1, p->startPosAfterHeader, &tempAr, allocTemp);
        SzAr_Free(&tempAr, allocTemp);
       
        if (res != SZ_OK)
        {
          Buf_Free(&tempBuf, allocTemp);
        }
        else
        {
          Buf_Free(&buf, allocTemp);
          buf.data = tempBuf.data;
          buf.size = tempBuf.size;
          sd.Data = buf.data;
          sd.Size = buf.size;
          res = ReadID(&sd, &type);
        }
      }
  
      if (res == SZ_OK)
      {
        if (type == k7zIdHeader)
        {
          /*
          CSzData sd2;
          unsigned ttt;
          for (ttt = 0; ttt < 40000; ttt++)
          {
            SzArEx_Free(p, allocMain);
            sd2 = sd;
            res = SzReadHeader(p, &sd2, inStream, allocMain, allocTemp);
            if (res != SZ_OK)
              break;
          }
          */
          res = SzReadHeader(p, &sd, inStream, allocMain, allocTemp);
        }
        else
          res = SZ_ERROR_UNSUPPORTED;
      }
    }
  }
 
  Buf_Free(&buf, allocTemp);
  return res;
}


SRes SzArEx_Open(CSzArEx *p, ILookInStreamPtr inStream,
    ISzAllocPtr allocMain, ISzAllocPtr allocTemp)
{
  const SRes res = SzArEx_Open2(p, inStream, allocMain, allocTemp);
  if (res != SZ_OK)
    SzArEx_Free(p, allocMain);
  return res;
}


SRes SzArEx_Extract(
    const CSzArEx *p,
    ILookInStreamPtr inStream,
    UInt32 fileIndex,
    UInt32 *blockIndex,
    Byte **tempBuf,
    size_t *outBufferSize,
    size_t *offset,
    size_t *outSizeProcessed,
    ISzAllocPtr allocMain,
    ISzAllocPtr allocTemp)
{
  const UInt32 folderIndex = p->FileToFolder[fileIndex];
  SRes res = SZ_OK;
  
  *offset = 0;
  *outSizeProcessed = 0;
  
  if (folderIndex == (UInt32)-1)
  {
    ISzAlloc_Free(allocMain, *tempBuf);
    *blockIndex = folderIndex;
    *tempBuf = NULL;
    *outBufferSize = 0;
    return SZ_OK;
  }

  if (*tempBuf == NULL || *blockIndex != folderIndex)
  {
    const UInt64 unpackSizeSpec = SzAr_GetFolderUnpackSize(&p->db, folderIndex);
    /*
    UInt64 unpackSizeSpec =
        p->UnpackPositions[p->FolderToFile[(size_t)folderIndex + 1]] -
        p->UnpackPositions[p->FolderToFile[folderIndex]];
    */
    const size_t unpackSize = (size_t)unpackSizeSpec;

    if (unpackSize != unpackSizeSpec)
      return SZ_ERROR_MEM;
    *blockIndex = folderIndex;
    ISzAlloc_Free(allocMain, *tempBuf);
    *tempBuf = NULL;
    
    if (res == SZ_OK)
    {
      *outBufferSize = unpackSize;
      if (unpackSize != 0)
      {
        *tempBuf = (Byte *)ISzAlloc_Alloc(allocMain, unpackSize);
        if (*tempBuf == NULL)
          res = SZ_ERROR_MEM;
      }
  
      if (res == SZ_OK)
      {
        res = SzAr_DecodeFolder(&p->db, folderIndex,
            inStream, p->dataPos, *tempBuf, unpackSize, allocTemp);
      }
    }
  }

  if (res == SZ_OK)
  {
    const UInt64 unpackPos = p->UnpackPositions[fileIndex];
    *offset = (size_t)(unpackPos - p->UnpackPositions[p->FolderToFile[folderIndex]]);
    *outSizeProcessed = (size_t)(p->UnpackPositions[(size_t)fileIndex + 1] - unpackPos);
    if (*offset + *outSizeProcessed > *outBufferSize)
      return SZ_ERROR_FAIL;
    if (SzBitWithVals_Check(&p->CRCs, fileIndex))
      if (CrcCalc(*tempBuf + *offset, *outSizeProcessed) != p->CRCs.Vals[fileIndex])
        res = SZ_ERROR_CRC;
  }

  return res;
}


size_t SzArEx_GetFileNameUtf16(const CSzArEx *p, size_t fileIndex, UInt16 *dest)
{
  const size_t offs = p->FileNameOffsets[fileIndex];
  const size_t len = p->FileNameOffsets[fileIndex + 1] - offs;
  if (dest != 0)
  {
    size_t i;
    const Byte *src = p->FileNames + offs * 2;
    for (i = 0; i < len; i++)
      dest[i] = GetUi16(src + i * 2);
  }
  return len;
}

/*
size_t SzArEx_GetFullNameLen(const CSzArEx *p, size_t fileIndex)
{
  size_t len;
  if (!p->FileNameOffsets)
    return 1;
  len = 0;
  for (;;)
  {
    UInt32 parent = (UInt32)(Int32)-1;
    len += p->FileNameOffsets[fileIndex + 1] - p->FileNameOffsets[fileIndex];
    if SzBitWithVals_Check(&p->Parents, fileIndex)
      parent = p->Parents.Vals[fileIndex];
    if (parent == (UInt32)(Int32)-1)
      return len;
    fileIndex = parent;
  }
}

UInt16 *SzArEx_GetFullNameUtf16_Back(const CSzArEx *p, size_t fileIndex, UInt16 *dest)
{
  BoolInt needSlash;
  if (!p->FileNameOffsets)
  {
    *(--dest) = 0;
    return dest;
  }
  needSlash = False;
  for (;;)
  {
    UInt32 parent = (UInt32)(Int32)-1;
    size_t curLen = p->FileNameOffsets[fileIndex + 1] - p->FileNameOffsets[fileIndex];
    SzArEx_GetFileNameUtf16(p, fileIndex, dest - curLen);
    if (needSlash)
      *(dest - 1) = '/';
    needSlash = True;
    dest -= curLen;

    if SzBitWithVals_Check(&p->Parents, fileIndex)
      parent = p->Parents.Vals[fileIndex];
    if (parent == (UInt32)(Int32)-1)
      return dest;
    fileIndex = parent;
  }
}
*/
