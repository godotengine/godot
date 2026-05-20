// 7zFolderInStream.h

#ifndef ZIP7_INC_7Z_FOLDER_IN_STREAM_H
#define ZIP7_INC_7Z_FOLDER_IN_STREAM_H

#include "../../../../C/7zCrc.h"

#include "../../../Common/MyCom.h"
#include "../../../Common/MyVector.h"
// #include "../Common/InStreamWithCRC.h"

#include "../../ICoder.h"
#include "../IArchive.h"

namespace NArchive {
namespace N7z {

Z7_CLASS_IMP_COM_2(
  CFolderInStream
  , ISequentialInStream
  , ICompressGetSubStreamSize
)
  /*
  Z7_COM7F_IMP(GetNextStream(UInt64 *streamIndex))
  Z7_IFACE_COM7_IMP(ICompressInSubStreams)
  */

  CMyComPtr<ISequentialInStream> _stream;
  UInt64 _totalSize_for_Coder;
  UInt64 _pos;
  UInt32 _crc;
  bool _size_Defined;
  bool _times_Defined;
  UInt64 _size;
  FILETIME _mTime;
  FILETIME _cTime;
  FILETIME _aTime;
  UInt32 _attrib;

  unsigned _numFiles;
  const UInt32 *_indexes;

  CMyComPtr<IArchiveUpdateCallback> _updateCallback;

  void ClearFileInfo();
  HRESULT OpenStream();
  HRESULT AddFileInfo(bool isProcessed);
  // HRESULT CloseCrcStream();
public:
  bool Need_MTime;
  bool Need_CTime;
  bool Need_ATime;
  bool Need_Attrib;
  // bool Need_Crc;
  // bool Need_FolderCrc;
  // unsigned AlignLog;
  
  CRecordVector<bool> Processed;
  CRecordVector<UInt64> Sizes;
  CRecordVector<UInt32> CRCs;
  CRecordVector<UInt32> Attribs;
  CRecordVector<bool> TimesDefined;
  CRecordVector<UInt64> MTimes;
  CRecordVector<UInt64> CTimes;
  CRecordVector<UInt64> ATimes;
  // UInt32 FolderCrc;

  // UInt32 GetFolderCrc() const { return CRC_GET_DIGEST(FolderCrc); }
  // CSequentialInStreamWithCRC *_crcStream_Spec;
  // CMyComPtr<ISequentialInStream> _crcStream;
  // CMyComPtr<IArchiveUpdateCallbackArcProp> _reportArcProp;

  void Init(IArchiveUpdateCallback *updateCallback, const UInt32 *indexes, unsigned numFiles);

  bool WasFinished() const { return Processed.Size() == _numFiles; }

  UInt64 Get_TotalSize_for_Coder() const { return _totalSize_for_Coder; }
  /*
  UInt64 GetFullSize() const
  {
    UInt64 size = 0;
    FOR_VECTOR (i, Sizes)
      size += Sizes[i];
    return size;
  }
  */

  CFolderInStream():
      Need_MTime(false),
      Need_CTime(false),
      Need_ATime(false),
      Need_Attrib(false)
      // , Need_Crc(true)
      // , Need_FolderCrc(false)
      // , AlignLog(0)
      {}
};

}}

#endif
