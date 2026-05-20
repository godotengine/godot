// MultiOutStream.cpp

#include "StdAfx.h"

// #define DEBUG_VOLUMES

#ifdef DEBUG_VOLUMES
#include <stdio.h>
  #define PRF(x) x;
#else
  #define PRF(x)
#endif

#include "../../Common/ComTry.h"

#include "../../Windows/FileDir.h"
#include "../../Windows/FileFind.h"
#include "../../Windows/System.h"

#include "MultiOutStream.h"

using namespace NWindows;
using namespace NFile;
using namespace NDir;

static const unsigned k_NumVols_MAX = k_VectorSizeMax - 1;
      // 2; // for debug
 
/*
#define UPDATE_HRES(hres, x) \
  { const HRESULT res2 = (x); if (hres == SZ_OK) hres = res2; }
*/

HRESULT CMultiOutStream::Destruct()
{
  COM_TRY_BEGIN
  HRESULT hres = S_OK;
  HRESULT hres3 = S_OK;
  
  while (!Streams.IsEmpty())
  {
    try
    {
      HRESULT hres2;
      if (NeedDelete)
      {
        /* we could call OptReOpen_and_SetSize() to test that we try to delete correct file,
           but we cannot guarantee that (RealSize) will be correct after Write() or another failures.
           And we still want to delete files even for such cases.
           So we don't check for OptReOpen_and_SetSize() here: */
        // if (OptReOpen_and_SetSize(Streams.Size() - 1, 0) == S_OK)
        hres2 = CloseStream_and_DeleteFile(Streams.Size() - 1);
      }
      else
      {
        hres2 = CloseStream(Streams.Size() - 1);
      }
      if (hres == S_OK)
        hres = hres2;
    }
    catch(...)
    {
      hres3 = E_OUTOFMEMORY;
    }

    {
      /* Stream was released in CloseStream_*() above already, and it was removed from linked list
         it's some unexpected case, if Stream is still attached here.
         So the following code is optional: */
      CVolStream &s = Streams.Back();
      if (s.Stream)
      {
        if (hres3 == S_OK)
          hres3 = E_FAIL;
        s.Stream.Detach();
        /* it will be not failure, even if we call RemoveFromLinkedList()
           twice for same CVolStream in this Destruct() function */
        RemoveFromLinkedList(Streams.Size() - 1);
      }
    }
    Streams.DeleteBack();
    // Delete_LastStream_Records();
  }

  if (hres == S_OK)
    hres = hres3;
  if (hres == S_OK && NumListItems != 0)
    hres = E_FAIL;
  return hres;
  COM_TRY_END
}


CMultiOutStream::~CMultiOutStream()
{
  // we try to avoid exception in destructors
  Destruct();
}


void CMultiOutStream::Init(const CRecordVector<UInt64> &sizes)
{
  Streams.Clear();
  InitLinkedList();
  Sizes = sizes;
  NeedDelete = true;
  MTime_Defined = false;
  FinalVol_WasReopen = false;
  NumOpenFiles_AllowedMax = NSystem::Get_File_OPEN_MAX_Reduced_for_3_tasks();

  _streamIndex = 0;
  _offsetPos = 0;
  _absPos = 0;
  _length = 0;
  _absLimit = (UInt64)(Int64)-1;

  _restrict_Begin = 0;
  _restrict_End = (UInt64)(Int64)-1;
  _restrict_Global = 0;

  UInt64 sum = 0;
  unsigned i = 0;
  for (i = 0; i < Sizes.Size(); i++)
  {
    if (i >= k_NumVols_MAX)
    {
      _absLimit = sum;
      break;
    }
    const UInt64 size = Sizes[i];
    const UInt64 next = sum + size;
    if (next < sum)
      break;
    sum = next;
  }

  // if (Sizes.IsEmpty()) throw "no volume sizes";
  const UInt64 size = Sizes.Back();
  if (size == 0)
    throw "zero size last volume";
  
  if (i == Sizes.Size())
    if ((_absLimit - sum) / size >= (k_NumVols_MAX - i))
      _absLimit = sum + (k_NumVols_MAX - i) * size;
}


/* IsRestricted():
   we must call only if volume is full (s.RealSize==VolSize) or finished.
   the function doesn't use VolSize and it uses s.RealSize instead.
   it returns true  : if stream is restricted, and we can't close that stream
   it returns false : if there is no restriction, and we can close that stream
 Note: (RealSize == 0) (empty volume) on restriction bounds are supposed as non-restricted
*/
bool CMultiOutStream::IsRestricted(const CVolStream &s) const
{
  if (s.Start < _restrict_Global)
    return true;
  if (_restrict_Begin == _restrict_End)
    return false;
  if (_restrict_Begin <= s.Start)
    return _restrict_End > s.Start;
  return _restrict_Begin < s.Start + s.RealSize;
}

/*
// this function check also _length and volSize
bool CMultiOutStream::IsRestricted_for_Close(unsigned index) const
{
  const CVolStream &s = Streams[index];
  if (_length <= s.Start) // we don't close streams after the end, because we still can write them later
    return true;
  // (_length > s.Start)
  const UInt64 volSize = GetVolSize_for_Stream(index);
  if (volSize == 0)
    return IsRestricted_Empty(s);
  if (_length - s.Start < volSize)
    return true;
  return IsRestricted(s);
}
*/

FString CMultiOutStream::GetFilePath(unsigned index)
{
  FString name;
  name.Add_UInt32((UInt32)(index + 1));
  while (name.Len() < 3)
    name.InsertAtFront(FTEXT('0'));
  name.Insert(0, Prefix);
  return name;
}


// we close stream, but we still keep item in Streams[] vector
HRESULT CMultiOutStream::CloseStream(unsigned index)
{
  CVolStream &s = Streams[index];
  if (s.Stream)
  {
    RINOK(s.StreamSpec->Close())
    // the following two commands must be called together:
    s.Stream.Release();
    RemoveFromLinkedList(index);
  }
  return S_OK;
}
  

// we close stream and delete file, but we still keep item in Streams[] vector
HRESULT CMultiOutStream::CloseStream_and_DeleteFile(unsigned index)
{
  PRF(printf("\n====== %u, CloseStream_AndDelete \n", index))
  RINOK(CloseStream(index))
  FString path = GetFilePath(index);
  path += Streams[index].Postfix;
  // we can checki that file exist
  // if (NFind::DoesFileExist_Raw(path))
  if (!DeleteFileAlways(path))
    return GetLastError_noZero_HRESULT();
  return S_OK;
}


HRESULT CMultiOutStream::CloseStream_and_FinalRename(unsigned index)
{
  PRF(printf("\n====== %u, CloseStream_and_FinalRename \n", index))
  CVolStream &s = Streams[index];
  // HRESULT res = S_OK;
  bool mtime_WasSet = false;
  if (MTime_Defined && s.Stream)
  {
    if (s.StreamSpec->SetMTime(&MTime))
      mtime_WasSet = true;
    // else res = GetLastError_noZero_HRESULT();
  }

  RINOK(CloseStream(index))
  if (s.Postfix.IsEmpty()) // if Postfix is empty, the path is already final
    return S_OK;
  const FString path = GetFilePath(index);
  FString tempPath = path;
  tempPath += s.Postfix;

  if (MTime_Defined && !mtime_WasSet)
  {
    if (!SetDirTime(tempPath, NULL, NULL, &MTime))
    {
      // res = GetLastError_noZero_HRESULT();
    }
  }
  if (!MyMoveFile(tempPath, path))
    return GetLastError_noZero_HRESULT();
  /* we clear CVolStream::Postfix. So we will not use Temp path
     anymore for this stream, and we will work only with final path */
  s.Postfix.Empty();
  // we can ignore set_mtime error or we can return it
  return S_OK;
  // return res;
}


HRESULT CMultiOutStream::PrepareToOpenNew()
{
  PRF(printf("PrepareToOpenNew NumListItems =%u,  NumOpenFiles_AllowedMax = %u \n", NumListItems, NumOpenFiles_AllowedMax))
  
  if (NumListItems < NumOpenFiles_AllowedMax)
    return S_OK;
  /* when we create zip archive: in most cases we need only starting
     data of restricted region for rewriting zip's local header.
     So here we close latest created volume (from Head), and we try to
     keep oldest volumes that will be used for header rewriting later. */
  const int index = Head;
  if (index == -1)
    return E_FAIL;
  PRF(printf("\n== %u, PrepareToOpenNew::CloseStream, NumListItems =%u \n", index, NumListItems))
  /* we don't expect non-restricted stream here in normal cases (if _restrict_Global was not changed).
     if there was non-restricted stream, it should be closed before */
  // if (!IsRestricted_for_Close(index)) return CloseStream_and_FinalRename(index);
  return CloseStream((unsigned)index);
}


HRESULT CMultiOutStream::CreateNewStream(UInt64 newSize)
{
  PRF(printf("\n== %u, CreateNewStream, size =%u \n", Streams.Size(), (unsigned)newSize))

  if (Streams.Size() >= k_NumVols_MAX)
    return E_INVALIDARG; // E_OUTOFMEMORY

  RINOK(PrepareToOpenNew())
  CVolStream s;
  s.StreamSpec = new COutFileStream;
  s.Stream = s.StreamSpec;
  const FString path = GetFilePath(Streams.Size());

  if (NFind::DoesFileExist_Raw(path))
    return HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS);
  if (!CreateTempFile2(path, false, s.Postfix, &s.StreamSpec->File))
    return GetLastError_noZero_HRESULT();
  
  s.Start = GetGlobalOffset_for_NewStream();
  s.Pos = 0;
  s.RealSize = 0;

  const unsigned index = Streams.Add(s);
  InsertToLinkedList(index);
  
  if (newSize != 0)
    return s.SetSize2(newSize);
  return S_OK;
}


HRESULT CMultiOutStream::CreateStreams_If_Required(unsigned streamIndex)
{
  // UInt64 lastStreamSize = 0;
  for (;;)
  {
    const unsigned numStreamsBefore = Streams.Size();
    if (streamIndex < numStreamsBefore)
      return S_OK;
    UInt64 newSize;
    if (streamIndex == numStreamsBefore)
    {
      // it's final volume that will be used for real writing.
      /* SetSize(_offsetPos) is not required,
      because the file Size will be set later by calling Seek() with Write() */
      newSize = 0; // lastStreamSize;
    }
    else
    {
      // it's intermediate volume. So we need full volume size
      newSize = GetVolSize_for_Stream(numStreamsBefore);
    }
    
    RINOK(CreateNewStream(newSize))

    // optional check
    if (numStreamsBefore + 1 != Streams.Size()) return E_FAIL;
    
    if (streamIndex != numStreamsBefore)
    {
      // it's intermediate volume. So we can close it, if it's non-restricted
      bool isRestricted;
      {
        const CVolStream &s = Streams[numStreamsBefore];
        if (newSize == 0)
          isRestricted = IsRestricted_Empty(s);
        else
          isRestricted = IsRestricted(s);
      }
      if (!isRestricted)
      {
        RINOK(CloseStream_and_FinalRename(numStreamsBefore))
      }
    }
  }
}


HRESULT CMultiOutStream::ReOpenStream(unsigned streamIndex)
{
  PRF(printf("\n====== %u, ReOpenStream \n", streamIndex))
  RINOK(PrepareToOpenNew())
  CVolStream &s = Streams[streamIndex];

  FString path = GetFilePath(streamIndex);
  path += s.Postfix;

  s.StreamSpec = new COutFileStream;
  s.Stream = s.StreamSpec;
  s.Pos = 0;

  HRESULT hres;
  if (s.StreamSpec->Open_EXISTING(path))
  {
    if (s.Postfix.IsEmpty())
    {
      /* it's unexpected case that we open finished volume.
         It can mean that the code for restriction is incorrect */
      FinalVol_WasReopen = true;
    }
    UInt64 realSize = 0;
    hres = s.StreamSpec->GetSize(&realSize);
    if (hres == S_OK)
    {
      if (realSize == s.RealSize)
      {
        PRF(printf("\n ReOpenStream OK realSize = %u\n", (unsigned)realSize))
        InsertToLinkedList(streamIndex);
        return S_OK;
      }
      // file size was changed between Close() and ReOpen()
      // we must release Stream to be consistent with linked list
      hres = E_FAIL;
    }
  }
  else
    hres = GetLastError_noZero_HRESULT();
  s.Stream.Release();
  s.StreamSpec = NULL;
  return hres;
}


/* Sets size of stream, if new size is not equal to old size (RealSize).
   If stream was closed and size change is required, it reopens the stream. */

HRESULT CMultiOutStream::OptReOpen_and_SetSize(unsigned index, UInt64 size)
{
  CVolStream &s = Streams[index];
  if (size == s.RealSize)
    return S_OK;
  if (!s.Stream)
  {
    RINOK(ReOpenStream(index))
  }
  PRF(printf("\n== %u, OptReOpen_and_SetSize, size =%u RealSize = %u\n", index, (unsigned)size, (unsigned)s.RealSize))
  // comment it to debug tail after data
  return s.SetSize2(size);
}


/*
call Normalize_finalMode(false), if _length was changed.
  for all streams starting after _length:
    - it sets zero size
    - it still keeps file open
  Note: after _length reducing with CMultiOutStream::SetSize() we can
    have very big number of empty streams at the end of Streams[] list.
    And Normalize_finalMode() will runs all these empty streams of Streams[] vector.
    So it can be ineffective, if we call Normalize_finalMode() many
    times after big reducing of (_length).

call Normalize_finalMode(true) to set final presentations of all streams
  for all streams starting after _length:
    - it sets zero size
    - it removes file
    - it removes CVolStream object from Streams[] vector

Note: we don't remove zero sized first volume, if (_length == 0)
*/

HRESULT CMultiOutStream::Normalize_finalMode(bool finalMode)
{
  PRF(printf("\n== Normalize_finalMode: _length =%d \n", (unsigned)_length))
  
  unsigned i = Streams.Size();

  UInt64 offset = 0;
  
  /* At first we normalize (reduce or increase) the sizes of all existing
     streams in Streams[] that can be affected by changed _length.
     And we remove tailing zero-size streams, if (finalMode == true) */
  while (i != 0)
  {
    offset = Streams[--i].Start; // it's last item in Streams[]
    // we don't want to remove first volume
    if (offset < _length || i == 0)
    {
      const UInt64 volSize = GetVolSize_for_Stream(i);
      UInt64 size = _length - offset; // (size != 0) here
      if (size > volSize)
        size = volSize;
      RINOK(OptReOpen_and_SetSize(i, size))
      if (_length - offset <= volSize)
        return S_OK;
      // _length - offset > volSize
      offset += volSize;
      // _length > offset
      break;
      // UPDATE_HRES(res, OptReOpen_and_SetSize(i, size));
    }

    /* we Set Size of stream to zero even for (finalMode==true), although
       that stream will be deleted in next commands */
    // UPDATE_HRES(res, OptReOpen_and_SetSize(i, 0));
    RINOK(OptReOpen_and_SetSize(i, 0))
    if (finalMode)
    {
      RINOK(CloseStream_and_DeleteFile(i))
      /* CVolStream::Stream was released above already, and it was
         removed from linked list. So we don't need to update linked list
         structure, when we delete last item in Streams[] */
      Streams.DeleteBack();
      // Delete_LastStream_Records();
    }
  }

  /* now we create new zero-filled streams to cover all data up to _length */

  if (_length == 0)
    return S_OK;

  // (offset) is start offset of next stream after existing Streams[]

  for (;;)
  {
    // _length > offset
    const UInt64 volSize = GetVolSize_for_Stream(Streams.Size());
    UInt64 size = _length - offset; // (size != 0) here
    if (size > volSize)
      size = volSize;
    RINOK(CreateNewStream(size))
    if (_length - offset <= volSize)
      return S_OK;
    // _length - offset > volSize)
    offset += volSize;
    // _length > offset
  }
}


HRESULT CMultiOutStream::FinalFlush_and_CloseFiles(unsigned &numTotalVolumesRes)
{
  // at first we remove unused zero-sized streams after _length
  HRESULT res = Normalize_finalMode(true);
  numTotalVolumesRes = Streams.Size();
  FOR_VECTOR (i, Streams)
  {
    const HRESULT res2 = CloseStream_and_FinalRename(i);
    if (res == S_OK)
      res = res2;
  }
  if (NumListItems != 0 && res == S_OK)
    res = E_FAIL;
  return res;
}


bool CMultiOutStream::SetMTime_Final(const CFiTime &mTime)
{
  // we will set mtime only if new value differs from previous
  if (!FinalVol_WasReopen && MTime_Defined && Compare_FiTime(&MTime, &mTime) == 0)
    return true;
  bool res = true;
  FOR_VECTOR (i, Streams)
  {
    CVolStream &s = Streams[i];
    if (s.Stream)
    {
      if (!s.StreamSpec->SetMTime(&mTime))
        res = false;
    }
    else
    {
      if (!SetDirTime(GetFilePath(i), NULL, NULL, &mTime))
        res = false;
    }
  }
  return res;
}


Z7_COM7F_IMF(CMultiOutStream::SetSize(UInt64 newSize))
{
  COM_TRY_BEGIN
  if ((Int64)newSize < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  if (newSize > _absLimit)
  {
    /* big seek value was sent to SetSize() or to Seek()+Write().
       It can mean one of two situations:
         1) some incorrect code called it with big seek value.
         2) volume size was small, and we have too big number of volumes
    */
    /* in Windows SetEndOfFile() can return:
       ERROR_NEGATIVE_SEEK:     for >= (1 << 63)
       ERROR_INVALID_PARAMETER: for >  (16 TiB - 64 KiB)
       ERROR_DISK_FULL:         for <= (16 TiB - 64 KiB)
    */
    // return E_FAIL;
    // return E_OUTOFMEMORY;
    return E_INVALIDARG;
  }

  if (newSize > _length)
  {
    // we don't expect such case. So we just define global restriction */
    _restrict_Global = newSize;
  }
  else if (newSize < _restrict_Global)
    _restrict_Global = newSize;

  PRF(printf("\n== CMultiOutStream::SetSize, size =%u \n", (unsigned)newSize))

  _length = newSize;
  return Normalize_finalMode(false);

  COM_TRY_END
}


Z7_COM7F_IMF(CMultiOutStream::Write(const void *data, UInt32 size, UInt32 *processedSize))
{
  COM_TRY_BEGIN
  if (processedSize)
    *processedSize = 0;
  if (size == 0)
    return S_OK;

  PRF(printf("\n -- CMultiOutStream::Write() : _absPos = %6u, size =%6u \n",
      (unsigned)_absPos, (unsigned)size))

  if (_absPos > _length)
  {
    // it create data only up to _absPos.
    // but we still can need additional new streams, if _absPos at range of volume
    RINOK(SetSize(_absPos))
  }
  
  while (size != 0)
  {
    UInt64 volSize;
    {
      if (_streamIndex < Sizes.Size() - 1)
      {
        volSize = Sizes[_streamIndex];
        if (_offsetPos >= volSize)
        {
          _offsetPos -= volSize;
          _streamIndex++;
          continue;
        }
      }
      else
      {
        volSize = Sizes[Sizes.Size() - 1];
        if (_offsetPos >= volSize)
        {
          const UInt64 v = _offsetPos / volSize;
          if (v >= ((UInt32)(Int32)-1) - _streamIndex)
            return E_INVALIDARG;
            // throw 202208;
          _streamIndex += (unsigned)v;
          _offsetPos -= (unsigned)v * volSize;
        }
        if (_streamIndex >= k_NumVols_MAX)
          return E_INVALIDARG;
      }
    }

    // (_offsetPos < volSize) here

    /* we can need to create one or more streams here,
       vol_size for some streams is allowed to be 0.
       Also we close some new created streams, if they are non-restricted */
    // file Size will be set later by calling Seek() with Write()

    /* the case (_absPos > _length) was processed above with SetSize(_absPos),
       so here it's expected. that we can create optional zero-size streams and then _streamIndex */
    RINOK(CreateStreams_If_Required(_streamIndex))

    CVolStream &s = Streams[_streamIndex];

    PRF(printf("\n%d, == Write : Pos = %u, RealSize = %u size =%u \n",
        _streamIndex, (unsigned)s.Pos, (unsigned)s.RealSize, size))

    if (!s.Stream)
    {
      RINOK(ReOpenStream(_streamIndex))
    }
    if (_offsetPos != s.Pos)
    {
      RINOK(s.Stream->Seek((Int64)_offsetPos, STREAM_SEEK_SET, NULL))
      s.Pos = _offsetPos;
    }

    UInt32 curSize = size;
    {
      const UInt64 rem = volSize - _offsetPos;
      if (curSize > rem)
        curSize = (UInt32)rem;
    }
    // curSize != 0
    UInt32 realProcessed = 0;
    
    HRESULT hres = s.Stream->Write(data, curSize, &realProcessed);
    
    data = (const void *)((const Byte *)data + realProcessed);
    size -= realProcessed;
    s.Pos += realProcessed;
    _offsetPos += realProcessed;
    _absPos += realProcessed;
    if (_length < _absPos)
      _length = _absPos;
    if (s.RealSize < _offsetPos)
      s.RealSize = _offsetPos;
    if (processedSize)
      *processedSize += realProcessed;
    
    if (s.Pos == volSize)
    {
      bool isRestricted;
      if (volSize == 0)
        isRestricted = IsRestricted_Empty(s);
      else
        isRestricted = IsRestricted(s);
      if (!isRestricted)
      {
        const HRESULT res2 = CloseStream_and_FinalRename(_streamIndex);
        if (hres == S_OK)
          hres = res2;
      }
      _streamIndex++;
      _offsetPos = 0;
    }
    
    RINOK(hres)
    if (realProcessed == 0 && curSize != 0)
      return E_FAIL;
    // break;
  }
  return S_OK;
  COM_TRY_END
}


Z7_COM7F_IMF(CMultiOutStream::Seek(Int64 offset, UInt32 seekOrigin, UInt64 *newPosition))
{
  PRF(printf("\n-- CMultiOutStream::Seek seekOrigin=%u Seek =%u\n", seekOrigin, (unsigned)offset))

  switch (seekOrigin)
  {
    case STREAM_SEEK_SET: break;
    case STREAM_SEEK_CUR: offset += _absPos; break;
    case STREAM_SEEK_END: offset += _length; break;
    default: return STG_E_INVALIDFUNCTION;
  }
  if (offset < 0)
    return HRESULT_WIN32_ERROR_NEGATIVE_SEEK;
  if ((UInt64)offset != _absPos)
  {
    _absPos = (UInt64)offset;
    _offsetPos = (UInt64)offset;
    _streamIndex = 0;
  }
  if (newPosition)
    *newPosition = (UInt64)offset;
  return S_OK;
}


// result value will be saturated to (UInt32)(Int32)-1

unsigned CMultiOutStream::GetStreamIndex_for_Offset(UInt64 offset, UInt64 &relOffset) const
{
  const unsigned last = Sizes.Size() - 1;
  for (unsigned i = 0; i < last; i++)
  {
    const UInt64 size = Sizes[i];
    if (offset < size)
    {
      relOffset = offset;
      return i;
    }
    offset -= size;
  }
  const UInt64 size = Sizes[last];
  const UInt64 v = offset / size;
  if (v >= ((UInt32)(Int32)-1) - last)
    return (unsigned)(int)-1; // saturation
  relOffset = offset - (unsigned)v * size;
  return last + (unsigned)(v);
}


Z7_COM7F_IMF(CMultiOutStream::SetRestriction(UInt64 begin, UInt64 end))
{
  COM_TRY_BEGIN

  // begin = end = 0; // for debug

  PRF(printf("\n==================== CMultiOutStream::SetRestriction %u, %u\n", (unsigned)begin, (unsigned)end))
  if (begin > end)
  {
    // these value are FAILED values.
    return E_FAIL;
    // return E_INVALIDARG;
    /*
    // or we can ignore error with 3 ways: no change, non-restricted, saturation:
    end = begin;             // non-restricted
    end = (UInt64)(Int64)-1; // saturation:
    return S_OK;
    */
  }
  UInt64 b = _restrict_Begin;
  UInt64 e = _restrict_End;
  _restrict_Begin = begin;
  _restrict_End = end;

  if (b == e)    // if there were no restriction before
    return S_OK; // no work to derestrict now.

  /* [b, e) is previous restricted region. So all volumes that
     intersect that [b, e) region are candidats for derestriction */

  if (begin != end) // if there is new non-empty restricted region
  {
    /* Now we will try to reduce or change (b) and (e) bounds
       to reduce main loop that checks volumes for derestriction.
       We still use one big derestriction region in main loop, although
       in some cases we could have two smaller derestriction regions.
       Also usually restriction region cannot move back from previous start position,
       so (b <= begin) is expected here for normal cases */
    if (b == begin) // if same low bounds
      b = end;      // we need to derestrict only after the end of new restricted region
    if (e == end)   // if same high bounds
      e = begin;    // we need to derestrict only before the begin of new restricted region
  }

  if (b > e) //  || b == (UInt64)(Int64)-1
    return S_OK;
  
  /* Here we close finished volumes that are not restricted anymore.
     We close (low number) volumes at first. */

  UInt64 offset;
  unsigned index = GetStreamIndex_for_Offset(b, offset);
  
  for (; index < Streams.Size(); index++)
  {
    {
      const CVolStream &s = Streams[index];
      if (_length <= s.Start)
        break; // we don't close streams after _length
      // (_length > s.Start)
      const UInt64 volSize = GetVolSize_for_Stream(index);
      if (volSize == 0)
      {
        if (e < s.Start)
          break;
        // we don't close empty stream, if next byte [s.Start, s.Start] is restricted
        if (IsRestricted_Empty(s))
          continue;
      }
      else
      {
        if (e <= s.Start)
          break;
        // we don't close non full streams
        if (_length - s.Start < volSize)
          break;
        // (volSize == s.RealSize) is expected here. So no need to check it
        // if (volSize != s.RealSize) break;
        if (IsRestricted(s))
          continue;
      }
    }
    RINOK(CloseStream_and_FinalRename(index))
  }

  return S_OK;
  COM_TRY_END
}
