// FileStreams.h

#ifndef ZIP7_INC_FILE_STREAMS_H
#define ZIP7_INC_FILE_STREAMS_H

#ifdef _WIN32
#define Z7_FILE_STREAMS_USE_WIN_FILE
#endif

#include "../../Common/MyCom.h"
#include "../../Common/MyString.h"

#include "../../Windows/FileIO.h"

#include "../IStream.h"

#include "UniqBlocks.h"


class CInFileStream;

Z7_PURE_INTERFACES_BEGIN
DECLARE_INTERFACE(IInFileStream_Callback)
{
  virtual HRESULT InFileStream_On_Error(UINT_PTR val, DWORD error) = 0;
  virtual void InFileStream_On_Destroy(CInFileStream *stream, UINT_PTR val) = 0;
};
Z7_PURE_INTERFACES_END


/*
Z7_CLASS_IMP_COM_5(
  CInFileStream
  , IInStream
  , IStreamGetSize
  , IStreamGetProps
  , IStreamGetProps2
  , IStreamGetProp
)
*/
Z7_class_final(CInFileStream) :
  public IInStream,
  public IStreamGetSize,
  public IStreamGetProps,
  public IStreamGetProps2,
  public IStreamGetProp,
  public CMyUnknownImp
{
  Z7_COM_UNKNOWN_IMP_6(
      IInStream,
      ISequentialInStream,
      IStreamGetSize,
      IStreamGetProps,
      IStreamGetProps2,
      IStreamGetProp)

  Z7_IFACE_COM7_IMP(ISequentialInStream)
  Z7_IFACE_COM7_IMP(IInStream)
public:
  Z7_IFACE_COM7_IMP(IStreamGetSize)
private:
  Z7_IFACE_COM7_IMP(IStreamGetProps)
public:
  Z7_IFACE_COM7_IMP(IStreamGetProps2)
  Z7_IFACE_COM7_IMP(IStreamGetProp)

private:
  NWindows::NFile::NIO::CInFile File;
public:

  #ifdef Z7_FILE_STREAMS_USE_WIN_FILE
  
  #ifdef Z7_DEVICE_FILE
  UInt64 VirtPos;
  UInt64 PhyPos;
  UInt64 BufStartPos;
  Byte *Buf;
  UInt32 BufSize;
  #endif

  #endif

 #ifdef _WIN32
  BY_HANDLE_FILE_INFORMATION _info;
 #else
  struct stat _info;
  uid_t _uid; // uid_t can be unsigned or signed int
  gid_t _gid;
  UString OwnerName;
  UString OwnerGroup;
  bool StoreOwnerId;
  bool StoreOwnerName;
 #endif

  bool _info_WasLoaded;
  bool SupportHardLinks;
  IInFileStream_Callback *Callback;
  UINT_PTR CallbackRef;

  CInFileStream();
  ~CInFileStream();
    
  void Set_PreserveATime(bool v)
  {
    File.PreserveATime = v;
  }

  bool GetLength(UInt64 &length) const throw()
  {
    return File.GetLength(length);
  }

#if 0
  bool OpenStdIn();
#endif
  
  bool Open(CFSTR fileName)
  {
    _info_WasLoaded = false;
    return File.Open(fileName);
  }
  
  bool OpenShared(CFSTR fileName, bool shareForWrite)
  {
    _info_WasLoaded = false;
    return File.OpenShared(fileName, shareForWrite);
  }
};

// bool CreateStdInStream(CMyComPtr<ISequentialInStream> &str);

Z7_CLASS_IMP_NOQIB_1(
  CStdInFileStream
  , ISequentialInStream
)
};


Z7_CLASS_IMP_COM_1(
  COutFileStream
  , IOutStream
)
  Z7_IFACE_COM7_IMP(ISequentialOutStream)
public:

  NWindows::NFile::NIO::COutFile File;

  bool Create_NEW(CFSTR fileName)
  {
    ProcessedSize = 0;
    return File.Create_NEW(fileName);
  }

  bool Create_ALWAYS(CFSTR fileName)
  {
    ProcessedSize = 0;
    return File.Create_ALWAYS(fileName);
  }

  bool Open_EXISTING(CFSTR fileName)
  {
    ProcessedSize = 0;
    return File.Open_EXISTING(fileName);
  }

  bool Create_ALWAYS_or_Open_ALWAYS(CFSTR fileName, bool createAlways)
  {
    ProcessedSize = 0;
    return File.Create_ALWAYS_or_Open_ALWAYS(fileName, createAlways);
  }

  HRESULT Close();
  
  UInt64 ProcessedSize;

  bool SetTime(const CFiTime *cTime, const CFiTime *aTime, const CFiTime *mTime)
  {
    return File.SetTime(cTime, aTime, mTime);
  }
  bool SetMTime(const CFiTime *mTime) {  return File.SetMTime(mTime); }

  bool SeekToBegin_bool()
  {
    #ifdef Z7_FILE_STREAMS_USE_WIN_FILE
    return File.SeekToBegin();
    #else
    return File.seekToBegin() == 0;
    #endif
  }

  HRESULT GetSize(UInt64 *size);
};


Z7_CLASS_IMP_NOQIB_1(
  CStdOutFileStream
  , ISequentialOutStream
)
  UInt64 _size;
public:
  UInt64 GetSize() const { return _size; }
  CStdOutFileStream(): _size(0) {}
};

#endif
