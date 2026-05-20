// MultiOutStream.h

#ifndef ZIP7_INC_MULTI_OUT_STREAM_H
#define ZIP7_INC_MULTI_OUT_STREAM_H

#include "FileStreams.h"

Z7_CLASS_IMP_COM_2(
  CMultiOutStream
  , IOutStream
  , IStreamSetRestriction
)
  Z7_IFACE_COM7_IMP(ISequentialOutStream)

  Z7_CLASS_NO_COPY(CMultiOutStream)

  struct CVolStream
  {
    COutFileStream *StreamSpec;
    CMyComPtr<IOutStream> Stream;
    UInt64 Start;   // start pos of current Stream in global stream
    UInt64 Pos;     // pos in current Stream
    UInt64 RealSize;
    int Next;       // next older
    int Prev;       // prev newer
    AString Postfix;

    HRESULT SetSize2(UInt64 size)
    {
      const HRESULT res = Stream->SetSize(size);
      if (res == SZ_OK)
        RealSize = size;
      return res;
    }
  };

  unsigned _streamIndex; // (_streamIndex >= Stream.Size()) is allowed in some internal code
  UInt64 _offsetPos;     // offset relative to Streams[_streamIndex] volume. (_offsetPos >= volSize is allowed)
  UInt64 _absPos;
  UInt64 _length;        // virtual Length
  UInt64 _absLimit;
  
  CObjectVector<CVolStream> Streams;
  CRecordVector<UInt64> Sizes;
  
  UInt64 _restrict_Begin;
  UInt64 _restrict_End;
  UInt64 _restrict_Global;

  unsigned NumOpenFiles_AllowedMax;

  // ----- Double Linked List -----

  unsigned NumListItems;
  int Head; // newest
  int Tail; // oldest

  void InitLinkedList()
  {
    Head = -1;
    Tail = -1;
    NumListItems = 0;
  }

  void InsertToLinkedList(unsigned index)
  {
    {
      CVolStream &node = Streams[index];
      node.Next = Head;
      node.Prev = -1;
    }
    if (Head != -1)
      Streams[(unsigned)Head].Prev = (int)index;
    else
    {
      // if (Tail != -1) throw 1;
      Tail = (int)index;
    }
    Head = (int)index;
    NumListItems++;
  }

  void RemoveFromLinkedList(unsigned index)
  {
    CVolStream &s = Streams[index];
    if (s.Next != -1) Streams[(unsigned)s.Next].Prev = s.Prev; else Tail = s.Prev;
    if (s.Prev != -1) Streams[(unsigned)s.Prev].Next = s.Next; else Head = s.Next;
    s.Next = -1; // optional
    s.Prev = -1; // optional
    NumListItems--;
  }

  /*
  void Delete_LastStream_Records()
  {
    if (Streams.Back().Stream)
      RemoveFromLinkedList(Streams.Size() - 1);
    Streams.DeleteBack();
  }
  */

  UInt64 GetVolSize_for_Stream(unsigned i) const
  {
    const unsigned last = Sizes.Size() - 1;
    return Sizes[i < last ? i : last];
  }
  UInt64 GetGlobalOffset_for_NewStream() const
  {
    return Streams.Size() == 0 ? 0:
        Streams.Back().Start +
        GetVolSize_for_Stream(Streams.Size() - 1);
  }
  unsigned GetStreamIndex_for_Offset(UInt64 offset, UInt64 &relOffset) const;
  bool IsRestricted(const CVolStream &s) const;
  bool IsRestricted_Empty(const CVolStream &s) const
  {
    // (s) must be stream that has (VolSize == 0).
    // we treat empty stream as restricted, if next byte is restricted.
    if (s.Start < _restrict_Global)
      return true;
    return
         (_restrict_Begin != _restrict_End)
      && (_restrict_Begin <= s.Start)
      && (_restrict_Begin == s.Start || _restrict_End > s.Start);
  }
  // bool IsRestricted_for_Close(unsigned index) const;
  FString GetFilePath(unsigned index);
  
  HRESULT CloseStream(unsigned index);
  HRESULT CloseStream_and_DeleteFile(unsigned index);
  HRESULT CloseStream_and_FinalRename(unsigned index);
  
  HRESULT PrepareToOpenNew();
  HRESULT CreateNewStream(UInt64 newSize);
  HRESULT CreateStreams_If_Required(unsigned streamIndex);
  HRESULT ReOpenStream(unsigned streamIndex);
  HRESULT OptReOpen_and_SetSize(unsigned index, UInt64 size);

  HRESULT Normalize_finalMode(bool finalMode);
public:
  FString Prefix;
  CFiTime MTime;
  bool MTime_Defined;
  bool FinalVol_WasReopen;
  bool NeedDelete;

  CMultiOutStream() {}
  ~CMultiOutStream();
  void Init(const CRecordVector<UInt64> &sizes);
  bool SetMTime_Final(const CFiTime &mTime);
  UInt64 GetSize() const { return _length; }
  /* it makes final flushing, closes open files and renames to final name if required
     but it still keeps Streams array of all closed files.
     So we still can delete all files later, if required */
  HRESULT FinalFlush_and_CloseFiles(unsigned &numTotalVolumesRes);
  // Destruct object without exceptions
  HRESULT Destruct();
};

#endif
