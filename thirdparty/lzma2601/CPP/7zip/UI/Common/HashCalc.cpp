// HashCalc.cpp

#include "StdAfx.h"

#include "../../../../C/Alloc.h"
#include "../../../../C/CpuArch.h"

#include "../../../Common/DynLimBuf.h"
#include "../../../Common/IntToString.h"
#include "../../../Common/StringToInt.h"

#include "../../Common/FileStreams.h"
#include "../../Common/ProgressUtils.h"
#include "../../Common/StreamObjects.h"
#include "../../Common/StreamUtils.h"

#include "../../Archive/Common/ItemNameUtils.h"
#include "../../Archive/IArchive.h"

#include "EnumDirItems.h"
#include "HashCalc.h"

using namespace NWindows;

#ifdef Z7_EXTERNAL_CODECS
extern const CExternalCodecs *g_ExternalCodecs_Ptr;
#endif

class CHashMidBuf
{
  void *_data;
public:
  CHashMidBuf(): _data(NULL) {}
  operator void *() { return _data; }
  bool Alloc(size_t size)
  {
    if (_data)
      return false;
    _data = ::MidAlloc(size);
    return _data != NULL;
  }
  ~CHashMidBuf() { ::MidFree(_data); }
};

static const char * const k_DefaultHashMethod = "CRC32";

HRESULT CHashBundle::SetMethods(DECL_EXTERNAL_CODECS_LOC_VARS const UStringVector &hashMethods)
{
  UStringVector names = hashMethods;
  if (names.IsEmpty())
    names.Add(UString(k_DefaultHashMethod));

  CRecordVector<CMethodId> ids;
  CObjectVector<COneMethodInfo> methods;
  
  unsigned i;
  for (i = 0; i < names.Size(); i++)
  {
    COneMethodInfo m;
    RINOK(m.ParseMethodFromString(names[i]))

    if (m.MethodName.IsEmpty())
      m.MethodName = k_DefaultHashMethod;
    
    if (m.MethodName.IsEqualTo("*"))
    {
      CRecordVector<CMethodId> tempMethods;
      GetHashMethods(EXTERNAL_CODECS_LOC_VARS tempMethods);
      methods.Clear();
      ids.Clear();
      FOR_VECTOR (t, tempMethods)
      {
        unsigned index = ids.AddToUniqueSorted(tempMethods[t]);
        if (ids.Size() != methods.Size())
          methods.Insert(index, m);
      }
      break;
    }
    else
    {
      // m.MethodName.RemoveChar(L'-');
      CMethodId id;
      if (!FindHashMethod(EXTERNAL_CODECS_LOC_VARS m.MethodName, id))
        return E_NOTIMPL;
      unsigned index = ids.AddToUniqueSorted(id);
      if (ids.Size() != methods.Size())
        methods.Insert(index, m);
    }
  }

  for (i = 0; i < ids.Size(); i++)
  {
    CMyComPtr<IHasher> hasher;
    AString name;
    RINOK(CreateHasher(EXTERNAL_CODECS_LOC_VARS ids[i], name, hasher))
    if (!hasher)
      throw "Can't create hasher";
    const COneMethodInfo &m = methods[i];
    {
      CMyComPtr<ICompressSetCoderProperties> scp;
      hasher.QueryInterface(IID_ICompressSetCoderProperties, &scp);
      if (scp)
        RINOK(m.SetCoderProps(scp, NULL))
    }
    const UInt32 digestSize = hasher->GetDigestSize();
    if (digestSize > k_HashCalc_DigestSize_Max)
      return E_NOTIMPL;
    CHasherState &h = Hashers.AddNew();
    h.DigestSize = digestSize;
    h.Hasher = hasher;
    h.Name = name;
    for (unsigned k = 0; k < k_HashCalc_NumGroups; k++)
      h.InitDigestGroup(k);
  }

  return S_OK;
}

void CHashBundle::InitForNewFile()
{
  CurSize = 0;
  FOR_VECTOR (i, Hashers)
  {
    CHasherState &h = Hashers[i];
    h.Hasher->Init();
    h.InitDigestGroup(k_HashCalc_Index_Current);
  }
}

void CHashBundle::Update(const void *data, UInt32 size)
{
  CurSize += size;
  FOR_VECTOR (i, Hashers)
    Hashers[i].Hasher->Update(data, size);
}

void CHashBundle::SetSize(UInt64 size)
{
  CurSize = size;
}

static void AddDigests(Byte *dest, const Byte *src, UInt32 size)
{
  unsigned next = 0;
  /*
  // we could use big-endian addition for sha-1 and sha-256
  // but another hashers are little-endian
  if (size > 8)
  {
    for (unsigned i = size; i != 0;)
    {
      i--;
      next += (unsigned)dest[i] + (unsigned)src[i];
      dest[i] = (Byte)next;
      next >>= 8;
    }
  }
  else
  */
  {
    for (unsigned i = 0; i < size; i++)
    {
      next += (unsigned)dest[i] + (unsigned)src[i];
      dest[i] = (Byte)next;
      next >>= 8;
    }
  }
  
  // we use little-endian to store extra bytes
  dest += k_HashCalc_DigestSize_Max;
  for (unsigned i = 0; i < k_HashCalc_ExtraSize; i++)
  {
    next += (unsigned)dest[i];
    dest[i] = (Byte)next;
    next >>= 8;
  }
}

void CHasherState::AddDigest(unsigned groupIndex, const Byte *data)
{
  NumSums[groupIndex]++;
  AddDigests(Digests[groupIndex], data, DigestSize);
}

void CHashBundle::Final(bool isDir, bool isAltStream, const UString &path)
{
  if (isDir)
    NumDirs++;
  else if (isAltStream)
  {
    NumAltStreams++;
    AltStreamsSize += CurSize;
  }
  else
  {
    NumFiles++;
    FilesSize += CurSize;
  }

  Byte pre[16];
  memset(pre, 0, sizeof(pre));
  if (isDir)
    pre[0] = 1;
  
  FOR_VECTOR (i, Hashers)
  {
    CHasherState &h = Hashers[i];
    if (!isDir)
    {
      h.Hasher->Final(h.Digests[0]); // k_HashCalc_Index_Current
      if (!isAltStream)
        h.AddDigest(k_HashCalc_Index_DataSum, h.Digests[0]);
    }

    h.Hasher->Init();
    h.Hasher->Update(pre, sizeof(pre));
    h.Hasher->Update(h.Digests[0], h.DigestSize);
    
    for (unsigned k = 0; k < path.Len(); k++)
    {
      wchar_t c = path[k];
      
      // 21.04: we want same hash for linux and windows paths
      #if CHAR_PATH_SEPARATOR != '/'
      if (c == CHAR_PATH_SEPARATOR)
        c = '/';
      // if (c == (wchar_t)('\\' + 0xf000)) c = '\\'; // to debug WSL
      // if (c > 0xf000 && c < 0xf080) c -= 0xf000; // to debug WSL
      #endif

      Byte temp[2] = { (Byte)(c & 0xFF), (Byte)((c >> 8) & 0xFF) };
      h.Hasher->Update(temp, 2);
    }
  
    Byte tempDigest[k_HashCalc_DigestSize_Max];
    h.Hasher->Final(tempDigest);
    if (!isAltStream)
      h.AddDigest(k_HashCalc_Index_NamesSum, tempDigest);
    h.AddDigest(k_HashCalc_Index_StreamsSum, tempDigest);
  }
}


static void CSum_Name_OriginalToEscape(const AString &src, AString &dest)
{
  dest.Empty();
  for (unsigned i = 0; i < src.Len();)
  {
    char c = src[i++];
    if (c == '\n')
    {
      dest.Add_Char('\\');
      c = 'n';
    }
    else if (c == '\\')
      dest.Add_Char('\\');
    dest.Add_Char(c);
  }
}


static bool CSum_Name_EscapeToOriginal(const char *s, AString &dest)
{
  bool isOK = true;
  dest.Empty();
  for (;;)
  {
    char c = *s++;
    if (c == 0)
      break;
    if (c == '\\')
    {
      const char c1 = *s;
      if (c1 == 'n')
      {
        c = '\n';
        s++;
      }
      else if (c1 == '\\')
      {
        c = c1;
        s++;
      }
      else
      {
        // original md5sum returns NULL for such bad strings
        isOK = false;
      }
    }
    dest.Add_Char(c);
  }
  return isOK;
}



static void SetSpacesAndNul(char *s, unsigned num)
{
  for (unsigned i = 0; i < num; i++)
    s[i] = ' ';
  s[num] = 0;
}

static const unsigned kHashColumnWidth_Min = 4 * 2;

static unsigned GetColumnWidth(unsigned digestSize)
{
  const unsigned width = digestSize * 2;
  return width < kHashColumnWidth_Min ? kHashColumnWidth_Min: width;
}


static void AddHashResultLine(
    AString &_s,
    // bool showHash,
    // UInt64 fileSize, bool showSize,
    const CObjectVector<CHasherState> &hashers
    // unsigned digestIndex, = k_HashCalc_Index_Current
    )
{
  FOR_VECTOR (i, hashers)
  {
    const CHasherState &h = hashers[i];
    char s[k_HashCalc_DigestSize_Max * 2 + 64];
    s[0] = 0;
    // if (showHash)
      HashHexToString(s, h.Digests[k_HashCalc_Index_Current], h.DigestSize);
    const unsigned pos = (unsigned)strlen(s);
    const int numSpaces = (int)GetColumnWidth(h.DigestSize) - (int)pos;
    if (numSpaces > 0)
      SetSpacesAndNul(s + pos, (unsigned)numSpaces);
    if (i != 0)
      _s.Add_Space();
    _s += s;
  }
  
  /*
  if (showSize)
  {
    _s.Add_Space();
    static const unsigned kSizeField_Len = 13; // same as in HashCon.cpp
    char s[kSizeField_Len + 32];
    char *p = s;
    SetSpacesAndNul(s, kSizeField_Len);
    p = s + kSizeField_Len;
    ConvertUInt64ToString(fileSize, p);
    int numSpaces = (int)kSizeField_Len - (int)strlen(p);
    if (numSpaces > 0)
      p -= (unsigned)numSpaces;
    _s += p;
  }
  */
}


static void Add_LF(CDynLimBuf &hashFileString, const CHashOptionsLocal &options)
{
  hashFileString += (char)(options.HashMode_Zero.Val ? 0 : '\n');
}




static void WriteLine(CDynLimBuf &hashFileString,
    const CHashOptionsLocal &options,
    const UString &path2,
    bool isDir,
    const AString &methodName,
    const AString &hashesString)
{
  if (options.HashMode_OnlyHash.Val)
  {
    hashFileString += hashesString;
    Add_LF(hashFileString, options);
    return;
  }
     
  UString path = path2;
      
  bool isBin = false;
  const bool zeroMode = options.HashMode_Zero.Val;
  const bool tagMode = options.HashMode_Tag.Val;
  
#if CHAR_PATH_SEPARATOR != '/'
  path.Replace(WCHAR_PATH_SEPARATOR, L'/');
  // path.Replace((wchar_t)('\\' + 0xf000), L'\\'); // to debug WSL
#endif
  
  AString utf8;
  ConvertUnicodeToUTF8(path, utf8);
  
  AString esc;
  CSum_Name_OriginalToEscape(utf8, esc);
  
  if (!zeroMode)
  {
    if (esc != utf8)
    {
      /* Original md5sum writes escape in that case.
      We do same for compatibility with original md5sum. */
      hashFileString += '\\';
    }
  }
  
  if (isDir && !esc.IsEmpty() && esc.Back() != '/')
    esc.Add_Slash();
  
  if (tagMode)
  {
    if (!methodName.IsEmpty())
    {
      hashFileString += methodName;
      hashFileString += ' ';
    }
    hashFileString += '(';
    hashFileString += esc;
    hashFileString += ')';
    hashFileString += " = ";
  }
  
  hashFileString += hashesString;
  
  if (!tagMode)
  {
    hashFileString += ' ';
    hashFileString += (char)(isBin ? '*' : ' ');
    hashFileString += esc;
  }

  Add_LF(hashFileString, options);
}


static void Convert_TagName_to_MethodName(AString &method)
{
  // we need to convert at least SHA512/256 to SHA512-256, and SHA512/224 to SHA512-224
  // but we convert any '/' to '-'.
  method.Replace('/', '-');
}

static void Convert_MethodName_to_TagName(AString &method)
{
  if (method.IsPrefixedBy_Ascii_NoCase("SHA512-2"))
    method.ReplaceOneCharAtPos(6, '/');
}


static void WriteLine(CDynLimBuf &hashFileString,
    const CHashOptionsLocal &options,
    const UString &path,
    bool isDir,
    const CHashBundle &hb)
{
  AString methodName;
  if (!hb.Hashers.IsEmpty())
  {
    methodName = hb.Hashers[0].Name;
    Convert_MethodName_to_TagName(methodName);
  }
  AString hashesString;
  AddHashResultLine(hashesString, hb.Hashers);
  WriteLine(hashFileString, options, path, isDir, methodName, hashesString);
}


HRESULT HashCalc(
    DECL_EXTERNAL_CODECS_LOC_VARS
    const NWildcard::CCensor &censor,
    const CHashOptions &options,
    AString &errorInfo,
    IHashCallbackUI *callback)
{
  CDirItems dirItems;
  dirItems.Callback = callback;

  if (options.StdInMode)
  {
    CDirItem di;
    if (!di.SetAs_StdInFile())
      return GetLastError_noZero_HRESULT();
    dirItems.Items.Add(di);
  }
  else
  {
    RINOK(callback->StartScanning())

    dirItems.SymLinks = options.SymLinks.Val;
    dirItems.ScanAltStreams = options.AltStreamsMode;
    dirItems.ExcludeDirItems = censor.ExcludeDirItems;
    dirItems.ExcludeFileItems = censor.ExcludeFileItems;

    dirItems.ShareForWrite = options.OpenShareForWrite;

    HRESULT res = EnumerateItems(censor,
        options.PathMode,
        UString(),
        dirItems);
    
    if (res != S_OK)
    {
      if (res != E_ABORT)
        errorInfo = "Scanning error";
      return res;
    }
    RINOK(callback->FinishScanning(dirItems.Stat))
  }

  unsigned i;
  CHashBundle hb;
  RINOK(hb.SetMethods(EXTERNAL_CODECS_LOC_VARS options.Methods))
  // hb.Init();

  hb.NumErrors = dirItems.Stat.NumErrors;

  UInt64 totalSize = 0;
  if (options.StdInMode)
  {
    RINOK(callback->SetNumFiles(1))
  }
  else
  {
    totalSize = dirItems.Stat.GetTotalBytes();
    RINOK(callback->SetTotal(totalSize))
  }

  const UInt32 kBufSize = 1 << 15;
  CHashMidBuf buf;
  if (!buf.Alloc(kBufSize))
    return E_OUTOFMEMORY;

  UInt64 completeValue = 0;

  RINOK(callback->BeforeFirstFile(hb))

  /*
  CDynLimBuf hashFileString((size_t)1 << 31);
  const bool needGenerate = !options.HashFilePath.IsEmpty();
  */

  for (i = 0; i < dirItems.Items.Size(); i++)
  {
    CMyComPtr<ISequentialInStream> inStream;
    UString path;
    bool isDir = false;
    bool isAltStream = false;
    
    if (options.StdInMode)
    {
#if 1
      inStream = new CStdInFileStream;
#else
      if (!CreateStdInStream(inStream))
      {
        const DWORD lastError = ::GetLastError();
        const HRESULT res = callback->OpenFileError(FString("stdin"), lastError);
        hb.NumErrors++;
        if (res != S_FALSE && res != S_OK)
          return res;
        continue;
      }
#endif
    }
    else
    {
      path = dirItems.GetLogPath(i);
      const CDirItem &di = dirItems.Items[i];
     #ifdef _WIN32
      isAltStream = di.IsAltStream;
     #endif

      #ifndef UNDER_CE
      // if (di.AreReparseData())
      if (di.ReparseData.Size() != 0)
      {
        CBufInStream *inStreamSpec = new CBufInStream();
        inStream = inStreamSpec;
        inStreamSpec->Init(di.ReparseData, di.ReparseData.Size());
      }
      else
      #endif
      {
        CInFileStream *inStreamSpec = new CInFileStream;
        inStreamSpec->Set_PreserveATime(options.PreserveATime);
        inStream = inStreamSpec;
        isDir = di.IsDir();
        if (!isDir)
        {
          const FString phyPath = dirItems.GetPhyPath(i);
          if (!inStreamSpec->OpenShared(phyPath, options.OpenShareForWrite))
          {
            const HRESULT res = callback->OpenFileError(phyPath, ::GetLastError());
            hb.NumErrors++;
            if (res != S_FALSE)
              return res;
            continue;
          }
          if (!options.StdInMode)
          {
            UInt64 curSize = 0;
            if (inStreamSpec->GetSize(&curSize) == S_OK)
            {
              if (curSize > di.Size)
              {
                totalSize += curSize - di.Size;
                RINOK(callback->SetTotal(totalSize))
                // printf("\ntotal = %d MiB\n", (unsigned)(totalSize >> 20));
              }
            }
          }
          // inStreamSpec->ReloadProps();
        }
      }
    }
    
    RINOK(callback->GetStream(path, isDir))
    UInt64 fileSize = 0;

    hb.InitForNewFile();
    
    if (!isDir)
    {
      for (UInt32 step = 0;; step++)
      {
        if ((step & 0xFF) == 0)
        {
          // printf("\ncompl = %d\n", (unsigned)(completeValue >> 20));
          RINOK(callback->SetCompleted(&completeValue))
        }
        UInt32 size;
        RINOK(inStream->Read(buf, kBufSize, &size))
        if (size == 0)
          break;
        hb.Update(buf, size);
        fileSize += size;
        completeValue += size;
      }
    }
    
    hb.Final(isDir, isAltStream, path);
    
    /*
    if (needGenerate
        && (options.HashMode_Dirs.Val || !isDir))
    {
      WriteLine(hashFileString,
          options,
          path, // change it
          isDir,
          hb);
        
      if (hashFileString.IsError())
        return E_OUTOFMEMORY;
    }
    */

    RINOK(callback->SetOperationResult(fileSize, hb, !isDir))
    RINOK(callback->SetCompleted(&completeValue))
  }
  
  /*
  if (needGenerate)
  {
    NFile::NIO::COutFile file;
    if (!file.Create(us2fs(options.HashFilePath), true)) // createAlways
      return GetLastError_noZero_HRESULT();
    if (!file.WriteFull(hashFileString, hashFileString.Len()))
      return GetLastError_noZero_HRESULT();
  }
  */

  return callback->AfterLastFile(hb);
}


void HashHexToString(char *dest, const Byte *data, size_t size)
{
  if (!data)
  {
    for (size_t i = 0; i < size; i++)
    {
      dest[0] = ' ';
      dest[1] = ' ';
      dest += 2;
    }
    *dest = 0;
    return;
  }
  
  if (size > 8)
    ConvertDataToHex_Lower(dest, data, size);
  else if (size == 0)
  {
    *dest = 0;
    return;
  }
  else
  {
    const char *dest_start = dest;
    dest += size * 2;
    *dest = 0;
    do
    {
      const size_t b = *data++;
      dest -= 2;
      dest[0] = GET_HEX_CHAR_UPPER(b >> 4);
      dest[1] = GET_HEX_CHAR_UPPER(b & 15);
    }
    while (dest != dest_start);
  }
}

void CHasherState::WriteToString(unsigned digestIndex, char *s) const
{
  HashHexToString(s, Digests[digestIndex], DigestSize);

  if (digestIndex != 0 && NumSums[digestIndex] != 1)
  {
    unsigned numExtraBytes = GetNumExtraBytes_for_Group(digestIndex);
    if (numExtraBytes > 4)
      numExtraBytes = 8;
    else // if (numExtraBytes >= 0)
      numExtraBytes = 4;
    // if (numExtraBytes != 0)
    {
      s += strlen(s);
      *s++ = '-';
      // *s = 0;
      HashHexToString(s, GetExtraData_for_Group(digestIndex), numExtraBytes);
    }
  }
}



// ---------- Hash Handler ----------

namespace NHash {

#define IsWhite(c) ((c) == ' ' || (c) == '\t')

bool CHashPair::IsDir() const
{
  if (Name.IsEmpty() || Name.Back() != '/')
    return false;
  // here we expect that Dir items contain only zeros or no Hash
  for (size_t i = 0; i < Hash.Size(); i++)
    if (Hash.ConstData()[i] != 0)
      return false;
  return true;
}


bool CHashPair::ParseCksum(const char *s)
{
  const char *end;
  
  const UInt32 crc = ConvertStringToUInt32(s, &end);
  if (*end != ' ')
    return false;
  end++;
  
  const UInt64 size = ConvertStringToUInt64(end, &end);
  if (*end != ' ')
    return false;
  end++;
  
  Name = end;
  
  Hash.Alloc(4);
  SetBe32a(Hash, crc)

  Size_from_Arc = size;
  Size_from_Arc_Defined = true;

  return true;
}



static const char *SkipWhite(const char *s)
{
  while (IsWhite(*s))
    s++;
  return s;
}

static const char * const k_CsumMethodNames[] =
{
    "sha256"
  , "sha224"
  , "sha512-224"
  , "sha512-256"
  , "sha384"
  , "sha512"
  , "sha3-224"
  , "sha3-256"
  , "sha3-384"
  , "sha3-512"
//  , "shake128"
//  , "shake256"
  , "sha1"
  , "sha2"
  , "sha3"
  , "sha"
  , "md5"
  , "blake2s"
  , "blake2b"
  , "blake2sp"
  , "xxh64"
  , "crc32"
  , "crc64"
  , "cksum"
};


// returns true, if (method) is known hash method or hash method group name.
static bool GetMethod_from_FileName(const UString &name, AString &method)
{
  method.Empty();
  AString s;
  ConvertUnicodeToUTF8(name, s);
  const int dotPos = s.ReverseFind_Dot();
  if (dotPos >= 0)
  {
    method = s.Ptr(dotPos + 1);
    if (method.IsEqualTo_Ascii_NoCase("txt") ||
        method.IsEqualTo_Ascii_NoCase("asc"))
    {
      method.Empty();
      const int dotPos2 = s.Find('.');
      if (dotPos2 >= 0)
        s.DeleteFrom(dotPos2);
    }
  }
  if (method.IsEmpty())
  {
    // we support file names with "sum" and "sums" postfixes: "sha256sum", "sha256sums"
    unsigned size;
    if (s.Len() > 4 && StringsAreEqualNoCase_Ascii(s.RightPtr(4), "sums"))
      size = 4;
    else if (s.Len() > 3 && StringsAreEqualNoCase_Ascii(s.RightPtr(3), "sum"))
      size = 3;
    else
      return false;
    method = s;
    method.DeleteFrom(s.Len() - size);
  }

  unsigned i;
  for (i = 0; i < Z7_ARRAY_SIZE(k_CsumMethodNames); i++)
  {
    const char *m = k_CsumMethodNames[i];
    if (method.IsEqualTo_Ascii_NoCase(m))
    {
      // method = m; // we can get lowcase
      return true;
    }
  }

/*
  for (i = 0; i < Z7_ARRAY_SIZE(k_CsumMethodNames); i++)
  {
    const char *m = k_CsumMethodNames[i];
    if (method.IsPrefixedBy_Ascii_NoCase(m))
    {
      method = m; // we get lowcase
      return true;
    }
  }
*/
  return false;
}


bool CHashPair::Parse(const char *s)
{
  // here we keep compatibility with original md5sum / shasum
  bool escape = false;

  s = SkipWhite(s);

  if (*s == '\\')
  {
    s++;
    escape = true;
  }
  Escape = escape;
  
  // const char *kMethod = GetMethod_from_FileName(s);
  // if (kMethod)
  if ((size_t)(FindNonHexChar(s) - s) < 4)
  {
    // BSD-style checksum line
    {
      const char *s2 = s;
      for (; *s2 != 0; s2++)
      {
        const char c = *s2;
        if (c == 0)
          return false;
        if (c == ' ' || c == '(')
          break;
      }
      Method.SetFrom(s, (unsigned)(s2 - s));
      s = s2;
    }
    IsBSD = true;
    if (*s == ' ')
      s++;
    if (*s != '(')
      return false;
    s++;
    {
      const char *s2 = s;
      for (; *s2 != 0; s2++)
      {}
      for (;;)
      {
        s2--;
        if (s2 < s)
          return false;
        if (*s2 == ')')
          break;
      }
      Name.SetFrom(s, (unsigned)(s2 - s));
      s = s2 + 1;
    }

    s = SkipWhite(s);
    if (*s != '=')
      return false;
    s++;
    s = SkipWhite(s);
  }

  {
    const size_t numChars = (size_t)(FindNonHexChar(s) - s) & ~(size_t)1;
    Hash.Alloc(numChars / 2);
    if ((size_t)(ParseHexString(s, Hash) - Hash) != numChars / 2)
      throw 101;
    HashString.SetFrom(s, (unsigned)numChars);
    s += numChars;
  }
  
  if (IsBSD)
  {
    if (*s != 0)
      return false;
    if (escape)
    {
      const AString temp (Name);
      return CSum_Name_EscapeToOriginal(temp, Name);
    }
    return true;
  }

  if (*s == 0)
    return true;

  if (*s != ' ')
    return false;
  s++;
  const char c = *s;
  if (c != ' '
      && c != '*'
      && c != 'U' // shasum Universal
      && c != '^' // shasum 0/1
     )
    return false;
  Mode = c;
  s++;
  if (escape)
    return CSum_Name_EscapeToOriginal(s, Name);
  Name = s;
  return true;
}


static bool GetLine(CByteBuffer &buf, bool zeroMode, bool cr_lf_Mode, size_t &posCur, AString &s)
{
  s.Empty();
  size_t pos = posCur;
  const Byte *p = buf;
  unsigned numDigits = 0;
  for (; pos < buf.Size(); pos++)
  {
    const Byte b = p[pos];
    if (b == 0)
    {
      numDigits = 1;
      break;
    }
    if (zeroMode)
      continue;
    if (b == 0x0a)
    {
      numDigits = 1;
      break;
    }
    if (!cr_lf_Mode)
      continue;
    if (b == 0x0d)
    {
      if (pos + 1 >= buf.Size())
      {
        numDigits = 1;
        break;
        // return false;
      }
      if (p[pos + 1] == 0x0a)
      {
        numDigits = 2;
        break;
      }
    }
  }
  s.SetFrom((const char *)(p + posCur), (unsigned)(pos - posCur));
  posCur = pos + numDigits;
  return true;
}


static bool Is_CR_LF_Data(const Byte *buf, size_t size)
{
  bool isCrLf = false;
  for (size_t i = 0; i < size;)
  {
    const Byte b = buf[i];
    if (b == 0x0a)
      return false;
    if (b == 0x0d)
    {
      if (i == size - 1)
        return false;
      if (buf[i + 1] != 0x0a)
        return false;
      isCrLf = true;
      i += 2;
    }
    else
      i++;
  }
  return isCrLf;
}


static const Byte kArcProps[] =
{
  // kpidComment,
  kpidCharacts
};

static const Byte kProps[] =
{
  kpidPath,
  kpidSize,
  kpidPackSize,
  kpidMethod
};

static const Byte kRawProps[] =
{
  kpidChecksum
};


Z7_COM7F_IMF(CHandler::GetParent(UInt32 /* index */ , UInt32 *parent, UInt32 *parentType))
{
  *parentType = NParentType::kDir;
  *parent = (UInt32)(Int32)-1;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetNumRawProps(UInt32 *numProps))
{
  *numProps = Z7_ARRAY_SIZE(kRawProps);
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetRawPropInfo(UInt32 index, BSTR *name, PROPID *propID))
{
  *propID = kRawProps[index];
  *name = NULL;
  return S_OK;
}

Z7_COM7F_IMF(CHandler::GetRawProp(UInt32 index, PROPID propID, const void **data, UInt32 *dataSize, UInt32 *propType))
{
  *data = NULL;
  *dataSize = 0;
  *propType = 0;

  if (propID == kpidChecksum)
  {
    const CHashPair &hp = HashPairs[index];
    if (hp.Hash.Size() != 0)
    {
      *data = hp.Hash;
      *dataSize = (UInt32)hp.Hash.Size();
      *propType = NPropDataType::kRaw;
    }
    return S_OK;
  }

  return S_OK;
}

IMP_IInArchive_Props
IMP_IInArchive_ArcProps

Z7_COM7F_IMF(CHandler::GetNumberOfItems(UInt32 *numItems))
{
  *numItems = HashPairs.Size();
  return S_OK;
}

static void Add_OptSpace_String(UString &dest, const char *src)
{
  dest.Add_Space_if_NotEmpty();
  dest += src;
}

Z7_COM7F_IMF(CHandler::GetArchiveProperty(PROPID propID, PROPVARIANT *value))
{
  NCOM::CPropVariant prop;
  switch (propID)
  {
    case kpidPhySize: if (_phySize != 0) prop = _phySize; break;
    /*
    case kpidErrorFlags:
    {
      UInt32 v = 0;
      if (!_isArc) v |= kpv_ErrorFlags_IsNotArc;
      // if (_sres == k_Base64_RES_NeedMoreInput) v |= kpv_ErrorFlags_UnexpectedEnd;
      if (v != 0)
        prop = v;
      break;
    }
    */
    case kpidCharacts:
    {
      UString s;
      if (_hashSize_Defined)
      {
        s.Add_Space_if_NotEmpty();
        s.Add_UInt32(_hashSize * 8);
        s += "-bit";
      }
      if (_is_PgpMethod)
      {
        Add_OptSpace_String(s, "PGP");
        if (!_pgpMethod.IsEmpty())
        {
          s.Add_Colon();
          s += _pgpMethod;
        }
      }
      if (_is_ZeroMode)
        Add_OptSpace_String(s, "ZERO");
      if (_are_there_Tags)
        Add_OptSpace_String(s, "TAG");
      if (_are_there_Dirs)
        Add_OptSpace_String(s, "DIRS");
      if (!_method_from_FileName.IsEmpty())
      {
        Add_OptSpace_String(s, "filename_method:");
        s += _method_from_FileName;
        if (!_is_KnownMethod_in_FileName)
          s += ":UNKNOWN";
      }
      if (!_methods.IsEmpty())
      {
        Add_OptSpace_String(s, "cmd_method:");
        s += _methods[0];
      }
      prop = s;
      break;
    }

    case kpidReadOnly:
    {
      if (_isArc)
        if (!CanUpdate())
          prop = true;
      break;
    }
    default: break;
  }
  prop.Detach(value);
  return S_OK;
}


Z7_COM7F_IMF(CHandler::GetProperty(UInt32 index, PROPID propID, PROPVARIANT *value))
{
  // COM_TRY_BEGIN
  NCOM::CPropVariant prop;
  const CHashPair &hp = HashPairs[index];
  switch (propID)
  {
    case kpidIsDir:
    {
      prop = hp.IsDir();
      break;
    }
    case kpidPath:
    {
      UString path;
      hp.Get_UString_Path(path);

      bool useBackslashReplacement = true;
      if (_supportWindowsBackslash && !hp.Escape && path.Find(L"\\\\") < 0)
      {
#if WCHAR_PATH_SEPARATOR == L'/'
        path.Replace(L'\\', L'/');
#else
        useBackslashReplacement = false;
#endif
      }
      NArchive::NItemName::ReplaceToOsSlashes_Remove_TailSlash(
          path, useBackslashReplacement);
      prop = path;
      break;
    }
    case kpidSize:
    {
      // client needs processed size of last file
      if (hp.Size_from_Disk_Defined)
        prop = (UInt64)hp.Size_from_Disk;
      else if (hp.Size_from_Arc_Defined)
        prop = (UInt64)hp.Size_from_Arc;
      break;
    }
    case kpidPackSize:
    {
      prop = (UInt64)hp.Hash.Size();
      break;
    }
    case kpidMethod:
    {
      if (!hp.Method.IsEmpty())
        prop = hp.Method;
      break;
    }
    default: break;
  }
  prop.Detach(value);
  return S_OK;
  // COM_TRY_END
}


static HRESULT ReadStream_to_Buf(IInStream *stream, CByteBuffer &buf, IArchiveOpenCallback *openCallback)
{
  buf.Free();
  UInt64 len;
  RINOK(InStream_AtBegin_GetSize(stream, len))
  if (len == 0 || len >= ((UInt64)1 << 31))
    return S_FALSE;
  buf.Alloc((size_t)len);
  UInt64 pos = 0;
  // return ReadStream_FALSE(stream, buf, (size_t)len);
  for (;;)
  {
    const UInt32 kBlockSize = ((UInt32)1 << 24);
    const UInt32 curSize = (len < kBlockSize) ? (UInt32)len : kBlockSize;
    UInt32 processedSizeLoc;
    RINOK(stream->Read((Byte *)buf + pos, curSize, &processedSizeLoc))
    if (processedSizeLoc == 0)
      return E_FAIL;
    len -= processedSizeLoc;
    pos += processedSizeLoc;
    if (len == 0)
      return S_OK;
    if (openCallback)
    {
      const UInt64 files = 0;
      RINOK(openCallback->SetCompleted(&files, &pos))
    }
  }
}


static bool isThere_Zero_Byte(const Byte *data, size_t size)
{
  for (size_t i = 0; i < size; i++)
    if (data[i] == 0)
      return true;
  return false;
}


Z7_COM7F_IMF(CHandler::Open(IInStream *stream, const UInt64 *, IArchiveOpenCallback *openCallback))
{
  COM_TRY_BEGIN
  {
    Close();

    CByteBuffer buf;
    RINOK(ReadStream_to_Buf(stream, buf, openCallback))

    CObjectVector<CHashPair> &pairs = HashPairs;

    const bool zeroMode = isThere_Zero_Byte(buf, buf.Size());
    _is_ZeroMode = zeroMode;
    bool cr_lf_Mode = false;
    if (!zeroMode)
      cr_lf_Mode = Is_CR_LF_Data(buf, buf.Size());

    if (openCallback)
    {
      Z7_DECL_CMyComPtr_QI_FROM(
          IArchiveOpenVolumeCallback,
          openVolumeCallback, openCallback)
      if (openVolumeCallback)
      {
        NCOM::CPropVariant prop;
        RINOK(openVolumeCallback->GetProperty(kpidName, &prop))
        if (prop.vt == VT_BSTR)
          _is_KnownMethod_in_FileName = GetMethod_from_FileName(prop.bstrVal, _method_from_FileName);
      }
    }

    if (!_methods.IsEmpty())
    {
      ConvertUnicodeToUTF8(_methods[0], _method_for_Extraction);
    }
    if (_method_for_Extraction.IsEmpty())
    {
      // if (_is_KnownMethod_in_FileName)
      _method_for_Extraction = _method_from_FileName;
    }

    const bool cksumMode = _method_for_Extraction.IsEqualTo_Ascii_NoCase("cksum");
    _is_CksumMode = cksumMode;

    size_t pos = 0;
    AString s;
    bool minusMode = false;
    unsigned numLines = 0;
    
    while (pos < buf.Size())
    {
      if (!GetLine(buf, zeroMode, cr_lf_Mode, pos, s))
        return S_FALSE;
      numLines++;
      if (s.IsEmpty())
        continue;
      
      if (s.IsPrefixedBy_Ascii_NoCase("; "))
      {
        if (numLines != 1)
          return S_FALSE;
        // comment line of FileVerifier++
        continue;
      }
      
      if (s.IsPrefixedBy_Ascii_NoCase("-----"))
      {
        if (minusMode)
          break; // end of pgp mode
        minusMode = true;
        if (s.IsPrefixedBy_Ascii_NoCase("-----BEGIN PGP SIGNED MESSAGE"))
        {
          if (_is_PgpMethod)
            return S_FALSE;
          if (!GetLine(buf, zeroMode, cr_lf_Mode, pos, s))
            return S_FALSE;
          const char *kStart = "Hash: ";
          if (!s.IsPrefixedBy_Ascii_NoCase(kStart))
            return S_FALSE;
          _pgpMethod = s.Ptr((unsigned)strlen(kStart));
          _is_PgpMethod = true;
        }
        continue;
      }
      
      CHashPair pair;
      pair.FullLine = s;
      if (cksumMode)
      {
        if (!pair.ParseCksum(s))
          return S_FALSE;
      }
      else if (!pair.Parse(s))
        return S_FALSE;
      pairs.Add(pair);
    }

    {
      unsigned hashSize = 0;
      bool hashSize_Dismatch = false;
      for (unsigned i = 0; i < HashPairs.Size(); i++)
      {
        const CHashPair &hp = HashPairs[i];
        if (i == 0)
          hashSize = (unsigned)hp.Hash.Size();
        else
          if (hashSize != hp.Hash.Size())
            hashSize_Dismatch = true;

        if (hp.IsBSD)
          _are_there_Tags = true;
        if (!_are_there_Dirs && hp.IsDir())
          _are_there_Dirs = true;
      }
      if (!hashSize_Dismatch && hashSize != 0)
      {
        _hashSize = hashSize;
        _hashSize_Defined = true;
      }
    }

    _phySize = buf.Size();
    _isArc = true;
    return S_OK;
  }
  COM_TRY_END
}


void CHandler::ClearVars()
{
  _phySize = 0;
  _isArc = false;
  _is_CksumMode = false;
  _is_PgpMethod = false;
  _is_ZeroMode = false;
  _are_there_Tags = false;
  _are_there_Dirs = false;
  _is_KnownMethod_in_FileName = false;
  _hashSize_Defined = false;
  _hashSize = 0;
}


Z7_COM7F_IMF(CHandler::Close())
{
  ClearVars();
  _method_from_FileName.Empty();
  _method_for_Extraction.Empty();
  _pgpMethod.Empty();
  HashPairs.Clear();
  return S_OK;
}


static bool CheckDigests(const Byte *a, const Byte *b, size_t size)
{
  if (size <= 8)
  {
    /* we use reversed order for one digest, when text representation
       uses big-order for crc-32 and crc-64 */
    for (size_t i = 0; i < size; i++)
      if (a[i] != b[size - 1 - i])
        return false;
    return true;
  }
  {
    for (size_t i = 0; i < size; i++)
      if (a[i] != b[i])
        return false;
    return true;
  }
}


static void AddDefaultMethod(UStringVector &methods,
    const char *name, unsigned size)
{
  int shaVersion = -1;
  if (name)
  {
    if (StringsAreEqualNoCase_Ascii(name, "sha"))
    {
      shaVersion = 0;
      if (size == 0)
        size = 32;
    }
    else if (StringsAreEqualNoCase_Ascii(name, "sha1"))
    {
      shaVersion = 1;
      if (size == 0)
        size = 20;
    }
    else if (StringsAreEqualNoCase_Ascii(name, "sha2"))
    {
      shaVersion = 2;
      if (size == 0)
        size = 32;
    }
    else if (StringsAreEqualNoCase_Ascii(name, "sha3"))
    {
      if (size == 0 ||
               size == 32) name = "sha3-256";
      else if (size == 28) name = "sha3-224";
      else if (size == 48) name = "sha3-384";
      else if (size == 64) name = "sha3-512";
    }
    else if (StringsAreEqualNoCase_Ascii(name, "sha512"))
    {
      // we allow any sha512 derived hash inside .sha512 file:
           if (size == 48) name = "sha384";
      else if (size == 32) name = "sha512-256";
      else if (size == 28) name = "sha512-224";
    }
    if (shaVersion >= 0)
      name = NULL;
  }
  
  const char *m = NULL;
  if (name)
    m = name;
  else
  {
         if (size == 64) m = "sha512";
    else if (size == 48) m = "sha384";
    else if (size == 32) m = "sha256";
    else if (size == 28) m = "sha224";
    else if (size == 20) m = "sha1";
    else if (shaVersion < 0)
    {
           if (size == 16) m = "md5";
      else if (size ==  8) m = "crc64";
      else if (size ==  4) m = "crc32";
    }
  }

  if (!m)
    return;

#ifdef Z7_EXTERNAL_CODECS
  const CExternalCodecs *_externalCodecs = g_ExternalCodecs_Ptr;
#endif
  CMethodId id;
  if (FindHashMethod(EXTERNAL_CODECS_LOC_VARS
      AString(m), id))
    methods.Add(UString(m));
}


Z7_COM7F_IMF(CHandler::Extract(const UInt32 *indices, UInt32 numItems,
    Int32 testMode, IArchiveExtractCallback *extractCallback))
{
  COM_TRY_BEGIN

  /*
  if (testMode == 0)
    return E_NOTIMPL;
  */

  const bool allFilesMode = (numItems == (UInt32)(Int32)-1);
  if (allFilesMode)
    numItems = HashPairs.Size();
  if (numItems == 0)
    return S_OK;

  #ifdef Z7_EXTERNAL_CODECS
  const CExternalCodecs *_externalCodecs = g_ExternalCodecs_Ptr;
  #endif
  
  CHashBundle hb_Glob;
  // UStringVector methods = options.Methods;
  UStringVector methods;

/*
  if (methods.IsEmpty() && !utf_nameExtenstion.IsEmpty() && !_hashSize_Defined)
  {
    CMethodId id;
    if (FindHashMethod(EXTERNAL_CODECS_LOC_VARS utf_nameExtenstion, id))
      methods.Add(_nameExtenstion);
  }
*/
  
  if (methods.IsEmpty() && !_pgpMethod.IsEmpty())
  {
    CMethodId id;
    if (FindHashMethod(EXTERNAL_CODECS_LOC_VARS _pgpMethod, id))
      methods.Add(UString(_pgpMethod));
  }

/*
  if (methods.IsEmpty() && _pgpMethod.IsEmpty() && _hashSize_Defined)
  {
    AddDefaultMethod(methods,
        utf_nameExtenstion.IsEmpty() ? NULL : utf_nameExtenstion.Ptr(),
        _hashSize);
  }
*/

  if (!methods.IsEmpty())
  {
    RINOK(hb_Glob.SetMethods(
      EXTERNAL_CODECS_LOC_VARS
      methods))
  }

  Z7_DECL_CMyComPtr_QI_FROM(
      IArchiveUpdateCallbackFile,
      updateCallbackFile, extractCallback)
  if (!updateCallbackFile)
    return E_NOTIMPL;
  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IArchiveGetDiskProperty,
        GetDiskProperty, extractCallback)
    if (GetDiskProperty)
    {
      UInt64 totalSize = 0;
      UInt32 i;
      for (i = 0; i < numItems; i++)
      {
        const UInt32 index = allFilesMode ? i : indices[i];
        const CHashPair &hp = HashPairs[index];
        if (hp.IsDir())
          continue;
        {
          NCOM::CPropVariant prop;
          RINOK(GetDiskProperty->GetDiskProperty(index, kpidSize, &prop))
          if (prop.vt != VT_UI8)
            continue;
          totalSize += prop.uhVal.QuadPart;
        }
      }
      RINOK(extractCallback->SetTotal(totalSize))
      // RINOK(Hash_SetTotalUnpacked->Hash_SetTotalUnpacked(indices, numItems));
    }
  }

  const UInt32 kBufSize = 1 << 15;
  CHashMidBuf buf;
  if (!buf.Alloc(kBufSize))
    return E_OUTOFMEMORY;

  CMyComPtr2_Create<ICompressProgressInfo, CLocalProgress> lps;
  lps->Init(extractCallback, false);

  for (UInt32 i = 0;; i++)
  {
    RINOK(lps->SetCur())
    if (i >= numItems)
      break;
    const UInt32 index = allFilesMode ? i : indices[i];

    CHashPair &hp = HashPairs[index];
    
    UString path;
    hp.Get_UString_Path(path);

    CMyComPtr<ISequentialInStream> inStream;
    const bool isDir = hp.IsDir();
    if (!isDir)
    {
      RINOK(updateCallbackFile->GetStream2(index, &inStream, NUpdateNotifyOp::kHashRead))
      if (!inStream)
      {
        continue; // we have shown error in GetStream2()
      }
      // askMode = NArchive::NExtract::NAskMode::kSkip;
    }

    Int32 askMode = testMode ?
        NArchive::NExtract::NAskMode::kTest :
        NArchive::NExtract::NAskMode::kExtract;

    CMyComPtr<ISequentialOutStream> realOutStream;
    RINOK(extractCallback->GetStream(index, &realOutStream, askMode))

    /* PrepareOperation() can expect kExtract to set
       Attrib and security of output file */
    askMode = NArchive::NExtract::NAskMode::kReadExternal;

    RINOK(extractCallback->PrepareOperation(askMode))
    
    const bool isAltStream = false;

    UInt64 fileSize = 0;

    CHashBundle hb_Loc;
    
    CHashBundle *hb_Use = &hb_Glob;

    HRESULT res_SetMethods = S_OK;

    UStringVector methods_loc;
    
    if (!hp.Method.IsEmpty())
    {
      hb_Use = &hb_Loc;
      CMethodId id;
      AString methodName = hp.Method;
      Convert_TagName_to_MethodName(methodName);
      if (FindHashMethod(EXTERNAL_CODECS_LOC_VARS methodName, id))
      {
        methods_loc.Add(UString(methodName));
        RINOK(hb_Loc.SetMethods(
            EXTERNAL_CODECS_LOC_VARS
            methods_loc))
      }
      else
        res_SetMethods = E_NOTIMPL;
    }
    else if (methods.IsEmpty())
    {
      AddDefaultMethod(methods_loc,
          _method_for_Extraction.IsEmpty() ? NULL :
          _method_for_Extraction.Ptr(),
          (unsigned)hp.Hash.Size());
      if (!methods_loc.IsEmpty())
      {
        hb_Use = &hb_Loc;
        RINOK(hb_Loc.SetMethods(
            EXTERNAL_CODECS_LOC_VARS
            methods_loc))
      }
    }

    const bool isSupportedMode = hp.IsSupportedMode();
    hb_Use->InitForNewFile();
    
    if (inStream)
    {
      for (UInt32 step = 0;; step++)
      {
        if ((step & 0xFF) == 0)
        {
          RINOK(lps.Interface()->SetRatioInfo(NULL, &fileSize))
        }
        UInt32 size;
        RINOK(inStream->Read(buf, kBufSize, &size))
        if (size == 0)
          break;
        hb_Use->Update(buf, size);
        if (realOutStream)
        {
          RINOK(WriteStream(realOutStream, buf, size))
        }
        fileSize += size;
      }

      hp.Size_from_Disk = fileSize;
      hp.Size_from_Disk_Defined = true;
    }

    realOutStream.Release();
    inStream.Release();

    lps->InSize += hp.Hash.Size();
    lps->OutSize += fileSize;

    hb_Use->Final(isDir, isAltStream, path);

    Int32 opRes = NArchive::NExtract::NOperationResult::kUnsupportedMethod;
    if (isSupportedMode
        && res_SetMethods != E_NOTIMPL
        && !hb_Use->Hashers.IsEmpty()
        )
    {
      const CHasherState &hs = hb_Use->Hashers[0];
      if (hs.DigestSize == hp.Hash.Size())
      {
        opRes = NArchive::NExtract::NOperationResult::kCRCError;
        if (CheckDigests(hp.Hash, hs.Digests[0], hs.DigestSize))
          if (!hp.Size_from_Arc_Defined || hp.Size_from_Arc == fileSize)
            opRes = NArchive::NExtract::NOperationResult::kOK;
      }
    }

    RINOK(extractCallback->SetOperationResult(opRes))
  }

  return S_OK;
  COM_TRY_END
}


// ---------- UPDATE ----------

struct CUpdateItem
{
  int IndexInArc;
  unsigned IndexInClient;
  UInt64 Size;
  bool NewData;
  bool NewProps;
  bool IsDir;
  UString Path;

  CUpdateItem(): Size(0), IsDir(false) {}
};


static HRESULT GetPropString(IArchiveUpdateCallback *callback, UInt32 index, PROPID propId,
    UString &res,
    bool convertSlash)
{
  NCOM::CPropVariant prop;
  RINOK(callback->GetProperty(index, propId, &prop))
  if (prop.vt == VT_BSTR)
  {
    res = prop.bstrVal;
    if (convertSlash)
      NArchive::NItemName::ReplaceSlashes_OsToUnix(res);
  }
  else if (prop.vt != VT_EMPTY)
    return E_INVALIDARG;
  return S_OK;
}


Z7_COM7F_IMF(CHandler::GetFileTimeType(UInt32 *type))
{
  *type = NFileTimeType::kUnix;
  return S_OK;
}


Z7_COM7F_IMF(CHandler::UpdateItems(ISequentialOutStream *outStream, UInt32 numItems,
    IArchiveUpdateCallback *callback))
{
  COM_TRY_BEGIN

  if (_isArc && !CanUpdate())
    return E_NOTIMPL;

  /*
  Z7_DECL_CMyComPtr_QI_FROM(IArchiveUpdateCallbackArcProp,
      reportArcProp, callback)
  */

  CObjectVector<CUpdateItem> updateItems;

  UInt64 complexity = 0;

  UInt32 i;
  for (i = 0; i < numItems; i++)
  {
    CUpdateItem ui;
    Int32 newData;
    Int32 newProps;
    UInt32 indexInArc;
    
    if (!callback)
      return E_FAIL;
    
    RINOK(callback->GetUpdateItemInfo(i, &newData, &newProps, &indexInArc))

    ui.NewProps = IntToBool(newProps);
    ui.NewData = IntToBool(newData);
    ui.IndexInArc = (int)indexInArc;
    ui.IndexInClient = i;
    if (IntToBool(newProps))
    {
      {
        NCOM::CPropVariant prop;
        RINOK(callback->GetProperty(i, kpidIsDir, &prop))
        if (prop.vt == VT_EMPTY)
          ui.IsDir = false;
        else if (prop.vt != VT_BOOL)
          return E_INVALIDARG;
        else
          ui.IsDir = (prop.boolVal != VARIANT_FALSE);
      }

      RINOK(GetPropString(callback, i, kpidPath, ui.Path,
          true)) // convertSlash
      /*
      if (ui.IsDir && !ui.Name.IsEmpty() && ui.Name.Back() != '/')
        ui.Name += '/';
      */
    }

    if (IntToBool(newData))
    {
      NCOM::CPropVariant prop;
      RINOK(callback->GetProperty(i, kpidSize, &prop))
      if (prop.vt == VT_UI8)
      {
        ui.Size = prop.uhVal.QuadPart;
        complexity += ui.Size;
      }
      else if (prop.vt == VT_EMPTY)
        ui.Size = (UInt64)(Int64)-1;
      else
        return E_INVALIDARG;
    }
    
    updateItems.Add(ui);
  }

  if (complexity != 0)
  {
    RINOK(callback->SetTotal(complexity))
  }

  #ifdef Z7_EXTERNAL_CODECS
  const CExternalCodecs *_externalCodecs = g_ExternalCodecs_Ptr;
  #endif

  CHashBundle hb;
  UStringVector methods;
  if (!_methods.IsEmpty())
  {
    FOR_VECTOR(k, _methods)
    {
      methods.Add(_methods[k]);
    }
  }
  else
  {
    Z7_DECL_CMyComPtr_QI_FROM(
        IArchiveGetRootProps,
        getRootProps, callback)
    if (getRootProps)
    {
      NCOM::CPropVariant prop;
      RINOK(getRootProps->GetRootProp(kpidArcFileName, &prop))
      if (prop.vt == VT_BSTR)
      {
        AString method;
        /* const bool isKnownMethod = */ GetMethod_from_FileName(prop.bstrVal, method);
        if (!method.IsEmpty())
        {
          AddDefaultMethod(methods, method, _crcSize_WasSet ? _crcSize : 0);
          if (methods.IsEmpty())
            return E_NOTIMPL;
        }
      }
    }
  }
  if (methods.IsEmpty() && _crcSize_WasSet)
  {
    AddDefaultMethod(methods,
        NULL, // name
        _crcSize);
  }

  RINOK(hb.SetMethods(EXTERNAL_CODECS_LOC_VARS methods))

  CMyComPtr2_Create<ICompressProgressInfo, CLocalProgress> lps;
  lps->Init(callback, true);

  const UInt32 kBufSize = 1 << 15;
  CHashMidBuf buf;
  if (!buf.Alloc(kBufSize))
    return E_OUTOFMEMORY;

  CDynLimBuf hashFileString((size_t)1 << 31);

  CHashOptionsLocal options = _options;
  
  if (_isArc)
  {
    if (!options.HashMode_Zero.Def && _is_ZeroMode)
      options.HashMode_Zero.Val = true;
    if (!options.HashMode_Tag.Def && _are_there_Tags)
      options.HashMode_Tag.Val = true;
    if (!options.HashMode_Dirs.Def && _are_there_Dirs)
      options.HashMode_Dirs.Val = true;
  }
  if (options.HashMode_OnlyHash.Val && updateItems.Size() != 1)
    options.HashMode_OnlyHash.Val = false;

  complexity = 0;

  for (i = 0; i < updateItems.Size(); i++)
  {
    lps->InSize = complexity;
    RINOK(lps->SetCur())

    const CUpdateItem &ui = updateItems[i];
    
    /*
    CHashPair item;
    if (!ui.NewProps)
      item = HashPairs[(unsigned)ui.IndexInArc];
    */

    if (ui.NewData)
    {
      UInt64 currentComplexity = ui.Size;
      UInt64 fileSize = 0;

      CMyComPtr<ISequentialInStream> fileInStream;
      bool needWrite = true;
      {
        HRESULT res = callback->GetStream(ui.IndexInClient, &fileInStream);

        if (res == S_FALSE)
          needWrite = false;
        else
        {
          RINOK(res)
          
          if (fileInStream)
          {
            Z7_DECL_CMyComPtr_QI_FROM(
                IStreamGetSize,
                streamGetSize, fileInStream)
            if (streamGetSize)
            {
              UInt64 size;
              if (streamGetSize->GetSize(&size) == S_OK)
                currentComplexity = size;
            }
            /*
            Z7_DECL_CMyComPtr_QI_FROM(
                IStreamGetProps,
                getProps, fileInStream)
            if (getProps)
            {
              FILETIME mTime;
              UInt64 size2;
              if (getProps->GetProps(&size2, NULL, NULL, &mTime, NULL) == S_OK)
              {
                currentComplexity = size2;
                // item.MTime = NTime::FileTimeToUnixTime64(mTime);;
              }
            }
            */
          }
          else
          {
            currentComplexity = 0;
          }
        }
      }

      hb.InitForNewFile();
      const bool isDir = ui.IsDir;
      
      if (needWrite && fileInStream && !isDir)
      {
        for (UInt32 step = 0;; step++)
        {
          if ((step & 0xFF) == 0)
          {
            RINOK(lps.Interface()->SetRatioInfo(&fileSize, NULL))
            // RINOK(callback->SetCompleted(&completeValue));
          }
          UInt32 size;
          RINOK(fileInStream->Read(buf, kBufSize, &size))
          if (size == 0)
            break;
          hb.Update(buf, size);
          fileSize += size;
        }
        currentComplexity = fileSize;
      }

      fileInStream.Release();
      const bool isAltStream = false;
      hb.Final(isDir, isAltStream, ui.Path);

      if (options.HashMode_Dirs.Val || !isDir)
      {
        if (!hb.Hashers.IsEmpty())
          lps->OutSize += hb.Hashers[0].DigestSize;
        WriteLine(hashFileString,
            options,
            ui.Path,
            isDir,
            hb);
        if (hashFileString.IsError())
          return E_OUTOFMEMORY;
      }

      complexity += currentComplexity;

      /*
      if (reportArcProp)
      {
        PROPVARIANT prop;
        prop.vt = VT_EMPTY;
        prop.wReserved1 = 0;
          
        NCOM::PropVarEm_Set_UInt64(&prop, fileSize);
        RINOK(reportArcProp->ReportProp(NArchive::NEventIndexType::kOutArcIndex, ui.IndexInClient, kpidSize, &prop));

        for (unsigned k = 0; k < hb.Hashers.Size(); k++)
        {
          const CHasherState &hs = hb.Hashers[k];

          if (hs.DigestSize == 4 && hs.Name.IsEqualTo_Ascii_NoCase("crc32"))
          {
            NCOM::PropVarEm_Set_UInt32(&prop, GetUi32(hs.Digests[k_HashCalc_Index_Current]));
            RINOK(reportArcProp->ReportProp(NArchive::NEventIndexType::kOutArcIndex, ui.IndexInClient, kpidCRC, &prop));
          }
          else
          {
            RINOK(reportArcProp->ReportRawProp(NArchive::NEventIndexType::kOutArcIndex, ui.IndexInClient,
              kpidChecksum, hs.Digests[k_HashCalc_Index_Current],
              hs.DigestSize, NPropDataType::kRaw));
          }
          RINOK(reportArcProp->ReportFinished(NArchive::NEventIndexType::kOutArcIndex, ui.IndexInClient, NArchive::NUpdate::NOperationResult::kOK));
        }
      }
      */
      RINOK(callback->SetOperationResult(NArchive::NUpdate::NOperationResult::kOK))
    }
    else
    {
      // old data
      const CHashPair &existItem = HashPairs[(unsigned)ui.IndexInArc];
      if (ui.NewProps)
      {
        WriteLine(hashFileString,
            options,
            ui.Path,
            ui.IsDir,
            existItem.Method, existItem.HashString
            );
      }
      else
      {
        hashFileString += existItem.FullLine;
        Add_LF(hashFileString, options);
      }
    }
    if (hashFileString.IsError())
      return E_OUTOFMEMORY;
  }

  RINOK(WriteStream(outStream, hashFileString, hashFileString.Len()))

  return S_OK;
  COM_TRY_END
}



HRESULT CHandler::SetProperty(const wchar_t *nameSpec, const PROPVARIANT &value)
{
  UString name = nameSpec;
  name.MakeLower_Ascii();
  if (name.IsEmpty())
    return E_INVALIDARG;
  
  if (name.IsEqualTo("m")) // "hm" hash method
  {
    // COneMethodInfo omi;
    // RINOK(omi.ParseMethodFromPROPVARIANT(L"", value));
    // _methods.Add(omi.MethodName); // change it. use omi.PropsString
    if (value.vt != VT_BSTR)
      return E_INVALIDARG;
    UString s (value.bstrVal);
    _methods.Add(s);
    return S_OK;
  }

  if (name.IsEqualTo("flags"))
  {
    if (value.vt != VT_BSTR)
      return E_INVALIDARG;
    if (!_options.ParseString(value.bstrVal))
      return E_INVALIDARG;
    return S_OK;
  }

  if (name.IsEqualTo("backslash"))
    return PROPVARIANT_to_bool(value, _supportWindowsBackslash);

  if (name.IsPrefixedBy_Ascii_NoCase("crc"))
  {
    name.Delete(0, 3);
    _crcSize = 4;
    _crcSize_WasSet = true;
    return ParsePropToUInt32(name, value, _crcSize);
  }

  // common properties
  if (name.IsPrefixedBy_Ascii_NoCase("mt")
      || name.IsPrefixedBy_Ascii_NoCase("memuse"))
    return S_OK;
  
  return E_INVALIDARG;
}


void CHandler::InitProps()
{
  _supportWindowsBackslash = true;
  _crcSize_WasSet = false;
  _crcSize = 4;
  _methods.Clear();
  _options.Init_HashOptionsLocal();
}

Z7_COM7F_IMF(CHandler::SetProperties(const wchar_t * const *names, const PROPVARIANT *values, UInt32 numProps))
{
  COM_TRY_BEGIN

  InitProps();

  for (UInt32 i = 0; i < numProps; i++)
  {
    RINOK(SetProperty(names[i], values[i]))
  }
  return S_OK;
  COM_TRY_END
}

CHandler::CHandler()
{
  ClearVars();
  InitProps();
}

}



static IInArchive  *CreateHashHandler_In()  { return new NHash::CHandler; }
static IOutArchive *CreateHashHandler_Out() { return new NHash::CHandler; }

void Codecs_AddHashArcHandler(CCodecs *codecs)
{
  {
    CArcInfoEx item;
    
    item.Name = "Hash";
    item.CreateInArchive = CreateHashHandler_In;
    item.CreateOutArchive = CreateHashHandler_Out;
    item.IsArcFunc = NULL;
    item.Flags =
        NArcInfoFlags::kKeepName
      | NArcInfoFlags::kStartOpen
      | NArcInfoFlags::kByExtOnlyOpen
      // | NArcInfoFlags::kPureStartOpen
      | NArcInfoFlags::kHashHandler
      ;
  
    // ubuntu uses "SHA256SUMS" file
    item.AddExts(UString (
        "sha256"
        " sha512"
        " sha384"
        " sha224"
        " sha512-224"
        " sha512-256"
        " sha3-224"
        " sha3-256"
        " sha3-384"
        " sha3-512"
        // " shake128"
        // " shake256"
        " sha1"
        " sha2"
        " sha3"
        " sha"
        " md5"
        " blake2s"
        " blake2b"
        " blake2sp"
        " xxh64"
        " crc32"
        " crc64"
        " cksum"
        " asc"
        // " b2sum"
        ),
        UString());

    item.UpdateEnabled = (item.CreateOutArchive != NULL);
    item.SignatureOffset = 0;
    // item.Version = MY_VER_MIX;
    item.NewInterface = true;
    
    item.Signatures.AddNew().CopyFrom(NULL, 0);
    
    codecs->Formats.Add(item);
  }
}
