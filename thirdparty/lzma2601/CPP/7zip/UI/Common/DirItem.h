// DirItem.h

#ifndef ZIP7_INC_DIR_ITEM_H
#define ZIP7_INC_DIR_ITEM_H

#ifdef _WIN32
#include "../../../Common/MyLinux.h"
#endif

#include "../../../Common/MyString.h"

#include "../../../Windows/FileFind.h"
#include "../../../Windows/PropVariant.h"
#include "../../../Windows/TimeUtils.h"

#include "../../Common/UniqBlocks.h"

#include "../../Archive/IArchive.h"

struct CDirItemsStat
{
  UInt64 NumDirs;
  UInt64 NumFiles;
  UInt64 NumAltStreams;
  UInt64 FilesSize;
  UInt64 AltStreamsSize;
  
  UInt64 NumErrors;
  
  // UInt64 Get_NumItems() const { return NumDirs + NumFiles + NumAltStreams; }
  UInt64 Get_NumDataItems() const { return NumFiles + NumAltStreams; }
  UInt64 GetTotalBytes() const { return FilesSize + AltStreamsSize; }

  bool IsEmpty() const { return
           0 == NumDirs
        && 0 == NumFiles
        && 0 == NumAltStreams
        && 0 == FilesSize
        && 0 == AltStreamsSize
        && 0 == NumErrors; }
  
  CDirItemsStat():
      NumDirs(0),
      NumFiles(0),
      NumAltStreams(0),
      FilesSize(0),
      AltStreamsSize(0),
      NumErrors(0)
    {}
};


struct CDirItemsStat2: public CDirItemsStat
{
  UInt64 Anti_NumDirs;
  UInt64 Anti_NumFiles;
  UInt64 Anti_NumAltStreams;
  
  // UInt64 Get_NumItems() const { return Anti_NumDirs + Anti_NumFiles + Anti_NumAltStreams + CDirItemsStat::Get_NumItems(); }
  UInt64 Get_NumDataItems2() const { return Anti_NumFiles + Anti_NumAltStreams + CDirItemsStat::Get_NumDataItems(); }

  bool IsEmpty() const { return CDirItemsStat::IsEmpty()
        && 0 == Anti_NumDirs
        && 0 == Anti_NumFiles
        && 0 == Anti_NumAltStreams; }
  
  CDirItemsStat2():
      Anti_NumDirs(0),
      Anti_NumFiles(0),
      Anti_NumAltStreams(0)
    {}
};


Z7_PURE_INTERFACES_BEGIN

#define Z7_IFACEN_IDirItemsCallback(x) \
  virtual HRESULT ScanError(const FString &path, DWORD systemError) x \
  virtual HRESULT ScanProgress(const CDirItemsStat &st, const FString &path, bool isDir) x \

Z7_IFACE_DECL_PURE(IDirItemsCallback)

Z7_PURE_INTERFACES_END


struct CArcTime
{
  FILETIME FT;
  UInt16 Prec;
  Byte Ns100;
  bool Def;

  CArcTime()
  {
    Clear();
  }

  void Clear()
  {
    FT.dwHighDateTime = FT.dwLowDateTime = 0;
    Prec = 0;
    Ns100 = 0;
    Def = false;
  }

  bool IsZero() const
  {
    return FT.dwLowDateTime == 0 && FT.dwHighDateTime == 0 && Ns100 == 0;
  }

  int CompareWith(const CArcTime &a) const
  {
    const int res = CompareFileTime(&FT, &a.FT);
    if (res != 0)
      return res;
    if (Ns100 < a.Ns100) return -1;
    if (Ns100 > a.Ns100) return 1;
    return 0;
  }

  UInt64 Get_FILETIME_as_UInt64() const
  {
    return (((UInt64)FT.dwHighDateTime) << 32) + FT.dwLowDateTime;
  }

  UInt32 Get_DosTime() const
  {
    FILETIME ft2 = FT;
    if ((Prec == k_PropVar_TimePrec_Base + 8 ||
         Prec == k_PropVar_TimePrec_Base + 9)
        && Ns100 != 0)
    {
      UInt64 u64 = Get_FILETIME_as_UInt64();
      // we round up even small (ns < 100ns) as FileTimeToDosTime()
      if (u64 % 20000000 == 0)
      {
        u64++;
        ft2.dwHighDateTime = (DWORD)(u64 >> 32);
        ft2.dwHighDateTime = (DWORD)u64;
      }
    }
    // FileTimeToDosTime() is expected to round up in Windows
    UInt32 dosTime;
    // we use simplified code with utctime->dos.
    // do we need local time instead here?
    NWindows::NTime::FileTime_To_DosTime(ft2, dosTime);
    return dosTime;
  }

  int GetNumDigits() const
  {
    if (Prec == k_PropVar_TimePrec_Unix ||
        Prec == k_PropVar_TimePrec_DOS)
      return 0;
    if (Prec == k_PropVar_TimePrec_HighPrec)
      return 9;
    if (Prec == k_PropVar_TimePrec_0)
      return 7;
    int digits = (int)Prec - (int)k_PropVar_TimePrec_Base;
    if (digits < 0)
      digits = 0;
    return digits;
  }

  void Write_To_FiTime(CFiTime &dest) const
  {
   #ifdef _WIN32
    dest = FT;
   #else
    if (FILETIME_To_timespec(FT, dest))
    if ((Prec == k_PropVar_TimePrec_Base + 8 ||
         Prec == k_PropVar_TimePrec_Base + 9)
        && Ns100 != 0)
    {
      dest.tv_nsec += Ns100;
    }
   #endif
  }

  // (Def) is not set
  void Set_From_FILETIME(const FILETIME &ft)
  {
    FT = ft;
    // Prec = k_PropVar_TimePrec_CompatNTFS;
    Prec = k_PropVar_TimePrec_Base + 7;
    Ns100 = 0;
  }

  // (Def) is not set
  // it set full form precision: k_PropVar_TimePrec_Base + numDigits
  void Set_From_FiTime(const CFiTime &ts)
  {
   #ifdef _WIN32
    FT = ts;
    Prec = k_PropVar_TimePrec_Base + 7;
    // Prec = k_PropVar_TimePrec_Base; // for debug
    // Prec = 0; // for debug
    Ns100 = 0;
   #else
    unsigned ns100;
    FiTime_To_FILETIME_ns100(ts, FT, ns100);
    Ns100 = (Byte)ns100;
    Prec = k_PropVar_TimePrec_Base + 9;
   #endif
  }

  void Set_From_Prop(const PROPVARIANT &prop)
  {
    FT = prop.filetime;
    unsigned prec = 0;
    unsigned ns100 = 0;
    const unsigned prec_Temp = prop.wReserved1;
    if (prec_Temp != 0
        && prec_Temp <= k_PropVar_TimePrec_1ns
        && prop.wReserved3 == 0)
    {
      const unsigned ns100_Temp = prop.wReserved2;
      if (ns100_Temp < 100)
      {
        ns100 = ns100_Temp;
        prec = prec_Temp;
      }
    }
    Prec = (UInt16)prec;
    Ns100 = (Byte)ns100;
    Def = true;
  }
};


struct CDirItem: public NWindows::NFile::NFind::CFileInfoBase
{
  UString Name;
  
 #ifndef UNDER_CE
  CByteBuffer ReparseData;

 #ifdef _WIN32
  // UString ShortName;
  CByteBuffer ReparseData2; // fixed (reduced) absolute links for WIM format
  bool AreReparseData() const { return ReparseData.Size() != 0 || ReparseData2.Size() != 0; }
 #else
  bool AreReparseData() const { return ReparseData.Size() != 0; }
 #endif // _WIN32

 #endif // !UNDER_CE
  
  void Copy_From_FileInfoBase(const NWindows::NFile::NFind::CFileInfoBase &fi)
  {
    (NWindows::NFile::NFind::CFileInfoBase &)*this = fi;
  }

  int PhyParent;
  int LogParent;
  int SecureIndex;

 #ifdef _WIN32
 #else
  int OwnerNameIndex;
  int OwnerGroupIndex;
 #endif

  // bool Attrib_IsDefined;

  CDirItem():
      PhyParent(-1)
    , LogParent(-1)
    , SecureIndex(-1)
   #ifdef _WIN32
   #else
    , OwnerNameIndex(-1)
    , OwnerGroupIndex(-1)
   #endif
    // , Attrib_IsDefined(true)
  {
  }


  CDirItem(const NWindows::NFile::NFind::CFileInfo &fi,
      int phyParent, int logParent, int secureIndex):
    CFileInfoBase(fi)
    , Name(fs2us(fi.Name))
   #if defined(_WIN32) && !defined(UNDER_CE)
    // , ShortName(fs2us(fi.ShortName))
   #endif
    , PhyParent(phyParent)
    , LogParent(logParent)
    , SecureIndex(secureIndex)
   #ifdef _WIN32
   #else
    , OwnerNameIndex(-1)
    , OwnerGroupIndex(-1)
   #endif
    {}
};



class CDirItems
{
  UStringVector Prefixes;
  CIntVector PhyParents;
  CIntVector LogParents;

  UString GetPrefixesPath(const CIntVector &parents, int index, const UString &name) const;

  HRESULT EnumerateDir(int phyParent, int logParent, const FString &phyPrefix);

public:
  CObjectVector<CDirItem> Items;

  bool SymLinks;
  bool ScanAltStreams;
  bool ExcludeDirItems;
  bool ExcludeFileItems;
  bool ShareForWrite;

  /* it must be called after anotrher checks */
  bool CanIncludeItem(bool isDir) const
  {
    return isDir ? !ExcludeDirItems : !ExcludeFileItems;
  }
 

  CDirItemsStat Stat;

  #if !defined(UNDER_CE)
  HRESULT SetLinkInfo(CDirItem &dirItem, const NWindows::NFile::NFind::CFileInfo &fi,
      const FString &phyPrefix);
  #endif

 #if defined(_WIN32) && !defined(UNDER_CE)

  CUniqBlocks SecureBlocks;
  CByteBuffer TempSecureBuf;
  bool _saclEnabled;
  bool ReadSecure;
  
  HRESULT AddSecurityItem(const FString &path, int &secureIndex);
  HRESULT FillFixedReparse();

 #endif

 #ifndef _WIN32
  
  C_UInt32_UString_Map OwnerNameMap;
  C_UInt32_UString_Map OwnerGroupMap;
  bool StoreOwnerName;
  
  HRESULT FillDeviceSizes();

 #endif

  IDirItemsCallback *Callback;

  CDirItems();

  void AddDirFileInfo(int phyParent, int logParent, int secureIndex,
      const NWindows::NFile::NFind::CFileInfo &fi);

  HRESULT AddError(const FString &path, DWORD errorCode);
  HRESULT AddError(const FString &path);

  HRESULT ScanProgress(const FString &path);

  // unsigned GetNumFolders() const { return Prefixes.Size(); }
  FString GetPhyPath(unsigned index) const;
  UString GetLogPath(unsigned index) const;

  unsigned AddPrefix(int phyParent, int logParent, const UString &prefix);
  void DeleteLastPrefix();

  // HRESULT EnumerateOneDir(const FString &phyPrefix, CObjectVector<NWindows::NFile::NFind::CDirEntry> &files);
  HRESULT EnumerateOneDir(const FString &phyPrefix, CObjectVector<NWindows::NFile::NFind::CFileInfo> &files);
  
  HRESULT EnumerateItems2(
    const FString &phyPrefix,
    const UString &logPrefix,
    const FStringVector &filePaths,
    FStringVector *requestedPaths);

  void ReserveDown();
};




struct CArcItem
{
  UInt64 Size;
  UString Name;
  CArcTime MTime;  // it can be mtime of archive file, if MTime is not defined for item in archive
  bool IsDir;
  bool IsAltStream;
  bool Size_Defined;
  bool Censored;
  UInt32 IndexInServer;
  
  CArcItem():
      IsDir(false),
      IsAltStream(false),
      Size_Defined(false),
      Censored(false)
    {}
};

#endif
