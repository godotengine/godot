// PropIDUtils.cpp

#include "StdAfx.h"

#include "../../../../C/CpuArch.h"

#include "../../../Common/IntToString.h"
#include "../../../Common/StringConvert.h"

#include "../../../Windows/FileIO.h"
#include "../../../Windows/PropVariantConv.h"

#include "../../PropID.h"

#include "PropIDUtils.h"

#ifndef Z7_SFX
#define Get16(x) GetUi16(x)
#define Get32(x) GetUi32(x)
#endif

using namespace NWindows;

static const unsigned kNumWinAtrribFlags = 30;
static const char g_WinAttribChars[kNumWinAtrribFlags + 1] = "RHS8DAdNTsLCOIEVvX.PU.M......B";

/*
FILE_ATTRIBUTE_

0 READONLY
1 HIDDEN
2 SYSTEM
3 (Volume label - obsolete)
4 DIRECTORY
5 ARCHIVE
6 DEVICE
7 NORMAL
8 TEMPORARY
9 SPARSE_FILE
10 REPARSE_POINT
11 COMPRESSED
12 OFFLINE
13 NOT_CONTENT_INDEXED (I - Win10 attrib/Explorer)
14 ENCRYPTED
15 INTEGRITY_STREAM (V - ReFS Win8/Win2012)
16 VIRTUAL (reserved)
17 NO_SCRUB_DATA (X - ReFS Win8/Win2012 attrib)
18 RECALL_ON_OPEN or EA
19 PINNED
20 UNPINNED
21 STRICTLY_SEQUENTIAL  (10.0.16267)
22 RECALL_ON_DATA_ACCESS
29 STRICTLY_SEQUENTIAL  (10.0.17134+) (SMR Blob)
*/


static const char kPosixTypes[16] = { '0', 'p', 'c', '3', 'd', '5', 'b', '7', '-', '9', 'l', 'B', 's', 'D', 'E', 'F' };
#define MY_ATTR_CHAR(a, n, c) (((a) & (1 << (n))) ? c : '-')

static void ConvertPosixAttribToString(char *s, UInt32 a) throw()
{
  s[0] = kPosixTypes[(a >> 12) & 0xF];
  for (int i = 6; i >= 0; i -= 3)
  {
    s[7 - i] = MY_ATTR_CHAR(a, i + 2, 'r');
    s[8 - i] = MY_ATTR_CHAR(a, i + 1, 'w');
    s[9 - i] = MY_ATTR_CHAR(a, i + 0, 'x');
  }
  if ((a & 0x800) != 0) s[3] = ((a & (1 << 6)) ? 's' : 'S'); // S_ISUID
  if ((a & 0x400) != 0) s[6] = ((a & (1 << 3)) ? 's' : 'S'); // S_ISGID
  if ((a & 0x200) != 0) s[9] = ((a & (1 << 0)) ? 't' : 'T'); // S_ISVTX
  s[10] = 0;
  
  a &= ~(UInt32)0xFFFF;
  if (a != 0)
  {
    s[10] = ' ';
    ConvertUInt32ToHex8Digits(a, s + 11);
  }
}


void ConvertWinAttribToString(char *s, UInt32 wa) throw()
{
  /*
  some programs store posix attributes in high 16 bits.
    p7zip - stores additional 0x8000 flag marker.
    macos - stores additional 0x4000 flag marker.
    info-zip - no additional marker.
  But this code works with Attrib from internal 7zip code.
  So we expect that 0x8000 marker is set, if there are posix attributes.
  (DT_UNKNOWN == 0) type in high bits is possible in some case for linux files.
  0x8000 flag is possible also in ReFS (Windows)?
  */

  const bool isPosix = (
      (wa & 0x8000) != 0 // FILE_ATTRIBUTE_UNIX_EXTENSION;
      // && (wa & 0xFFFF0000u) != 0
      );
 
  UInt32 posix = 0;
  if (isPosix)
  {
    posix = wa >> 16;
    if ((wa & 0xF0000000u) != 0)
      wa &= (UInt32)0x3FFF;
  }

  for (unsigned i = 0; i < kNumWinAtrribFlags; i++)
  {
    const UInt32 flag = (UInt32)1 << i;
    if (wa & flag)
    {
      const char c = g_WinAttribChars[i];
      if (c != '.')
      {
        wa &= ~flag;
        // if (i != 7) // we can disable N (NORMAL) printing
        *s++ = c;
      }
    }
  }
  
  if (wa != 0)
  {
    *s++ = ' ';
    ConvertUInt32ToHex8Digits(wa, s);
    s += strlen(s);
  }

  *s = 0;

  if (isPosix)
  {
    *s++ = ' ';
    ConvertPosixAttribToString(s, posix);
  }
}


void ConvertPropertyToShortString2(char *dest, const PROPVARIANT &prop, PROPID propID, int level) throw()
{
  *dest = 0;
  
  if (prop.vt == VT_FILETIME)
  {
    const FILETIME &ft = prop.filetime;
    unsigned ns100 = 0;
    int numDigits = kTimestampPrintLevel_NTFS;
    const unsigned prec = prop.wReserved1;
    const unsigned ns100_Temp = prop.wReserved2;
    if (prec != 0
        && prec <= k_PropVar_TimePrec_1ns
        && ns100_Temp < 100
        && prop.wReserved3 == 0)
    {
      ns100 = ns100_Temp;
      if (prec == k_PropVar_TimePrec_Unix ||
          prec == k_PropVar_TimePrec_DOS)
        numDigits = 0;
      else if (prec == k_PropVar_TimePrec_HighPrec)
        numDigits = 9;
      else
      {
        numDigits = (int)prec - (int)k_PropVar_TimePrec_Base;
        if (
            // numDigits < kTimestampPrintLevel_DAY // for debuf
            numDigits < kTimestampPrintLevel_SEC
            )

          numDigits = kTimestampPrintLevel_NTFS;
      }
    }
    if (ft.dwHighDateTime == 0 && ft.dwLowDateTime == 0 && ns100 == 0)
      return;
    if (level > numDigits)
      level = numDigits;
    ConvertUtcFileTimeToString2(ft, ns100, dest, level);
    return;
  }

  switch (propID)
  {
    case kpidCRC:
    {
      if (prop.vt != VT_UI4)
        break;
      ConvertUInt32ToHex8Digits(prop.ulVal, dest);
      return;
    }
    case kpidAttrib:
    {
      if (prop.vt != VT_UI4)
        break;
      const UInt32 a = prop.ulVal;

      /*
      if ((a & 0x8000) && (a & 0x7FFF) == 0)
        ConvertPosixAttribToString(dest, a >> 16);
      else
      */
      ConvertWinAttribToString(dest, a);
      return;
    }
    case kpidPosixAttrib:
    {
      if (prop.vt != VT_UI4)
        break;
      ConvertPosixAttribToString(dest, prop.ulVal);
      return;
    }
    case kpidINode:
    {
      if (prop.vt != VT_UI8)
        break;
      ConvertUInt32ToString((UInt32)(prop.uhVal.QuadPart >> 48), dest);
      dest += strlen(dest);
      *dest++ = '-';
      const UInt64 low = prop.uhVal.QuadPart & (((UInt64)1 << 48) - 1);
      ConvertUInt64ToString(low, dest);
      return;
    }
    case kpidVa:
    {
      UInt64 v = 0;
      if (prop.vt == VT_UI4)
        v = prop.ulVal;
      else if (prop.vt == VT_UI8)
        v = (UInt64)prop.uhVal.QuadPart;
      else
        break;
      dest[0] = '0';
      dest[1] = 'x';
      ConvertUInt64ToHex(v, dest + 2);
      return;
    }

    /*
    case kpidDevice:
    {
      UInt64 v = 0;
      if (prop.vt == VT_UI4)
        v = prop.ulVal;
      else if (prop.vt == VT_UI8)
        v = (UInt64)prop.uhVal.QuadPart;
      else
        break;
      ConvertUInt32ToString(MY_dev_major(v), dest);
      dest += strlen(dest);
      *dest++ = ',';
      ConvertUInt32ToString(MY_dev_minor(v), dest);
      return;
    }
    */
    default: break;
  }
  
  ConvertPropVariantToShortString(prop, dest);
}

void ConvertPropertyToString2(UString &dest, const PROPVARIANT &prop, PROPID propID, int level)
{
  if (prop.vt == VT_BSTR)
  {
    dest.SetFromBstr(prop.bstrVal);
    return;
  }
  char temp[64];
  ConvertPropertyToShortString2(temp, prop, propID, level);
  dest = temp;
}

#ifndef Z7_SFX

static inline void AddHexToString(AString &res, unsigned v)
{
  res.Add_Char((char)GET_HEX_CHAR_UPPER(v >> 4));
  res.Add_Char((char)GET_HEX_CHAR_UPPER(v & 15));
}

/*
static AString Data_To_Hex(const Byte *data, size_t size)
{
  AString s;
  for (size_t i = 0; i < size; i++)
    AddHexToString(s, data[i]);
  return s;
}
*/

static const char * const sidNames[] =
{
    "0"
  , "Dialup"
  , "Network"
  , "Batch"
  , "Interactive"
  , "Logon"  // S-1-5-5-X-Y
  , "Service"
  , "Anonymous"
  , "Proxy"
  , "EnterpriseDC"
  , "Self"
  , "AuthenticatedUsers"
  , "RestrictedCode"
  , "TerminalServer"
  , "RemoteInteractiveLogon"
  , "ThisOrganization"
  , "16"
  , "IUserIIS"
  , "LocalSystem"
  , "LocalService"
  , "NetworkService"
  , "Domains"
};

struct CSecID2Name
{
  UInt32 n;
  const char *sz;
};

static int FindPairIndex(const CSecID2Name * pairs, unsigned num, UInt32 id)
{
  for (unsigned i = 0; i < num; i++)
    if (pairs[i].n == id)
      return (int)i;
  return -1;
}

static const CSecID2Name sid_32_Names[] =
{
  { 544, "Administrators" },
  { 545, "Users" },
  { 546, "Guests" },
  { 547, "PowerUsers" },
  { 548, "AccountOperators" },
  { 549, "ServerOperators" },
  { 550, "PrintOperators" },
  { 551, "BackupOperators" },
  { 552, "Replicators" },
  { 553, "Backup Operators" },
  { 554, "PreWindows2000CompatibleAccess" },
  { 555, "RemoteDesktopUsers" },
  { 556, "NetworkConfigurationOperators" },
  { 557, "IncomingForestTrustBuilders" },
  { 558, "PerformanceMonitorUsers" },
  { 559, "PerformanceLogUsers" },
  { 560, "WindowsAuthorizationAccessGroup" },
  { 561, "TerminalServerLicenseServers" },
  { 562, "DistributedCOMUsers" },
  { 569, "CryptographicOperators" },
  { 573, "EventLogReaders" },
  { 574, "CertificateServiceDCOMAccess" }
};

static const CSecID2Name sid_21_Names[] =
{
  { 500, "Administrator" },
  { 501, "Guest" },
  { 502, "KRBTGT" },
  { 512, "DomainAdmins" },
  { 513, "DomainUsers" },
  { 515, "DomainComputers" },
  { 516, "DomainControllers" },
  { 517, "CertPublishers" },
  { 518, "SchemaAdmins" },
  { 519, "EnterpriseAdmins" },
  { 520, "GroupPolicyCreatorOwners" },
  { 553, "RASandIASServers" },
  { 553, "RASandIASServers" },
  { 571, "AllowedRODCPasswordReplicationGroup" },
  { 572, "DeniedRODCPasswordReplicationGroup" }
};

struct CServicesToName
{
  UInt32 n[5];
  const char *sz;
};

static const CServicesToName services_to_name[] =
{
  { { 0x38FB89B5, 0xCBC28419, 0x6D236C5C, 0x6E770057, 0x876402C0 } , "TrustedInstaller" }
};

static void ParseSid(AString &s, const Byte *p, size_t lim /* , unsigned &sidSize */)
{
  // sidSize = 0;
  if (lim < 8)
  {
    s += "ERROR";
    return;
  }
  if (p[0] != 1) // rev
  {
    s += "UNSUPPORTED";
    return;
  }
  const unsigned num = p[1];
  const unsigned sidSize_Loc = 8 + num * 4;
  if (sidSize_Loc > lim)
  {
    s += "ERROR";
    return;
  }
  // sidSize = sidSize_Loc;
  const UInt32 authority = GetBe32(p + 4);

  if (p[2] == 0 && p[3] == 0 && authority == 5 && num >= 1)
  {
    const UInt32 v0 = Get32(p + 8);
    if (v0 < Z7_ARRAY_SIZE(sidNames))
    {
      s += sidNames[v0];
      return;
    }
    if (v0 == 32 && num == 2)
    {
      const UInt32 v1 = Get32(p + 12);
      const int index = FindPairIndex(sid_32_Names, Z7_ARRAY_SIZE(sid_32_Names), v1);
      if (index >= 0)
      {
        s += sid_32_Names[(unsigned)index].sz;
        return;
      }
    }
    if (v0 == 21 && num == 5)
    {
      UInt32 v4 = Get32(p + 8 + 4 * 4);
      const int index = FindPairIndex(sid_21_Names, Z7_ARRAY_SIZE(sid_21_Names), v4);
      if (index >= 0)
      {
        s += sid_21_Names[(unsigned)index].sz;
        return;
      }
    }
    if (v0 == 80 && num == 6)
    {
      for (unsigned i = 0; i < Z7_ARRAY_SIZE(services_to_name); i++)
      {
        const CServicesToName &sn = services_to_name[i];
        int j;
        for (j = 0; j < 5 && sn.n[j] == Get32(p + 8 + 4 + j * 4); j++);
        if (j == 5)
        {
          s += sn.sz;
          return;
        }
      }
    }
  }
  
  s += "S-1-";
  if (p[2] == 0 && p[3] == 0)
    s.Add_UInt32(authority);
  else
  {
    s += "0x";
    for (int i = 2; i < 8; i++)
      AddHexToString(s, p[i]);
  }
  for (UInt32 i = 0; i < num; i++)
  {
    s.Add_Minus();
    s.Add_UInt32(Get32(p + 8 + i * 4));
  }
}

static void ParseOwner(AString &s, const Byte *p, size_t size, UInt32 pos)
{
  if (pos > size)
  {
    s += "ERROR";
    return;
  }
  // unsigned sidSize = 0;
  ParseSid(s, p + pos, size - pos /* , sidSize */);
}

static void ParseAcl(AString &s, const Byte *p, size_t size, const char *strName, UInt32 flags, UInt32 offset)
{
  const unsigned control = Get16(p + 2);
  if ((flags & control) == 0)
    return;
  const UInt32 pos = Get32(p + offset);
  s.Add_Space();
  s += strName;
  if (pos >= size)
    return;
  p += pos;
  size -= (size_t)pos;
  if (size < 8)
    return;
  if (Get16(p) != 2) // revision
    return;
  const UInt32 num = Get32(p + 4);
  s.Add_UInt32(num);
  
  /*
  UInt32 aclSize = Get16(p + 2);
  if (num >= (1 << 16))
    return;
  if (aclSize > size)
    return;
  size = aclSize;
  size -= 8;
  p += 8;
  for (UInt32 i = 0 ; i < num; i++)
  {
    if (size <= 8)
      return;
    // Byte type = p[0];
    // Byte flags = p[1];
    // UInt32 aceSize = Get16(p + 2);
    // UInt32 mask = Get32(p + 4);
    p += 8;
    size -= 8;

    UInt32 sidSize = 0;
    s.Add_Space();
    ParseSid(s, p, size, sidSize);
    if (sidSize == 0)
      return;
    p += sidSize;
    size -= sidSize;
  }

  // the tail can contain zeros. So (size != 0) is not ERROR
  // if (size != 0) s += " ERROR";
  */
}

/*
#define MY_SE_OWNER_DEFAULTED       (0x0001)
#define MY_SE_GROUP_DEFAULTED       (0x0002)
*/
#define MY_SE_DACL_PRESENT          (0x0004)
/*
#define MY_SE_DACL_DEFAULTED        (0x0008)
*/
#define MY_SE_SACL_PRESENT          (0x0010)
/*
#define MY_SE_SACL_DEFAULTED        (0x0020)
#define MY_SE_DACL_AUTO_INHERIT_REQ (0x0100)
#define MY_SE_SACL_AUTO_INHERIT_REQ (0x0200)
#define MY_SE_DACL_AUTO_INHERITED   (0x0400)
#define MY_SE_SACL_AUTO_INHERITED   (0x0800)
#define MY_SE_DACL_PROTECTED        (0x1000)
#define MY_SE_SACL_PROTECTED        (0x2000)
#define MY_SE_RM_CONTROL_VALID      (0x4000)
#define MY_SE_SELF_RELATIVE         (0x8000)
*/

void ConvertNtSecureToString(const Byte *data, size_t size, AString &s)
{
  s.Empty();
  if (size < 20 || size > (1 << 18))
  {
    s += "ERROR";
    return;
  }
  if (Get16(data) != 1) // revision
  {
    s += "UNSUPPORTED";
    return;
  }
  ParseOwner(s, data, size, Get32(data + 4));
  s.Add_Space();
  ParseOwner(s, data, size, Get32(data + 8));
  ParseAcl(s, data, size, "s:", MY_SE_SACL_PRESENT, 12);
  ParseAcl(s, data, size, "d:", MY_SE_DACL_PRESENT, 16);
  s.Add_Space();
  s.Add_UInt32((UInt32)size);
  // s.Add_LF();
  // s += Data_To_Hex(data, size);
}

#ifdef _WIN32

static bool CheckSid(const Byte *data, size_t size, UInt32 pos) throw()
{
  if (pos >= size)
    return false;
  size -= pos;
  if (size < 8)
    return false;
  if (data[pos] != 1) // rev
    return false;
  const unsigned num = data[pos + 1];
  return (8 + num * 4 <= size);
}

static bool CheckAcl(const Byte *p, size_t size, UInt32 flags, size_t offset) throw()
{
  const unsigned control = Get16(p + 2);
  if ((flags & control) == 0)
    return true;
  const UInt32 pos = Get32(p + offset);
  if (pos >= size)
    return false;
  p += pos;
  size -= pos;
  if (size < 8)
    return false;
  const unsigned aclSize = Get16(p + 2);
  return (aclSize <= size);
}

bool CheckNtSecure(const Byte *data, size_t size) throw()
{
  if (size < 20)
    return false;
  if (Get16(data) != 1) // revision
    return true; // windows function can handle such error, so we allow it
  if (size > (1 << 18))
    return false;
  if (!CheckSid(data, size, Get32(data + 4))) return false;
  if (!CheckSid(data, size, Get32(data + 8))) return false;
  if (!CheckAcl(data, size, MY_SE_SACL_PRESENT, 12)) return false;
  if (!CheckAcl(data, size, MY_SE_DACL_PRESENT, 16)) return false;
  return true;
}

#endif



// IO_REPARSE_TAG_*

static const CSecID2Name k_ReparseTags[] =
{
  { 0xA0000003, "MOUNT_POINT" },
  { 0xC0000004, "HSM" },
  { 0x80000005, "DRIVE_EXTENDER" },
  { 0x80000006, "HSM2" },
  { 0x80000007, "SIS" },
  { 0x80000008, "WIM" },
  { 0x80000009, "CSV" },
  { 0x8000000A, "DFS" },
  { 0x8000000B, "FILTER_MANAGER" },
  { 0xA000000C, "SYMLINK" },
  { 0xA0000010, "IIS_CACHE" },
  { 0x80000012, "DFSR" },
  { 0x80000013, "DEDUP" },
  { 0xC0000014, "APPXSTRM" },
  { 0x80000014, "NFS" },
  { 0x80000015, "FILE_PLACEHOLDER" },
  { 0x80000016, "DFM" },
  { 0x80000017, "WOF" },
  { 0x80000018, "WCI" },
  { 0x8000001B, "APPEXECLINK" },
  { 0xA000001D, "LX_SYMLINK" },
  { 0x80000023, "AF_UNIX" },
  { 0x80000024, "LX_FIFO" },
  { 0x80000025, "LX_CHR" },
  { 0x80000026, "LX_BLK" }
};

bool ConvertNtReparseToString(const Byte *data, size_t size, UString &s)
{
  s.Empty();
  NFile::CReparseAttr attr;

  if (attr.Parse(data, size))
  {
    if (attr.IsSymLink_WSL())
    {
      s += "WSL: ";
      s += attr.GetPath();
    }
    else
    {
      if (!attr.IsSymLink_Win())
        s += "Junction: ";
      s += attr.GetPath();
      if (s.IsEmpty())
        s += "Link: ";
      if (!attr.IsOkNamePair())
      {
        s += " : ";
        s += attr.PrintName;
      }
    }
    if (attr.MinorError)
      s += " : MINOR_ERROR";
    return true;
    // s.Add_Space(); // for debug
  }

  if (size < 8)
    return false;
  const UInt32 tag = Get32(data);
  const UInt32 len = Get16(data + 4);
  if (len + 8 > size)
    return false;
  if (Get16(data + 6) != 0) // padding
    return false;

  /*
  #define my_IO_REPARSE_TAG_DEDUP        (0x80000013L)
  if (tag == my_IO_REPARSE_TAG_DEDUP)
  {
  }
  */

  {
    const int index = FindPairIndex(k_ReparseTags, Z7_ARRAY_SIZE(k_ReparseTags), tag);
    if (index >= 0)
      s += k_ReparseTags[(unsigned)index].sz;
    else
    {
      s += "REPARSE:";
      char hex[16];
      ConvertUInt32ToHex8Digits(tag, hex);
      s += hex;
    }
  }

  s.Add_Colon();
  s.Add_UInt32(len);

  if (len != 0)
  {
    s.Add_Space();
    
    data += 8;
    
    for (UInt32 i = 0; i < len; i++)
    {
      if (i >= 16)
      {
        s += "...";
        break;
      }
      const unsigned b = data[i];
      s.Add_Char((char)GET_HEX_CHAR_UPPER(b >> 4));
      s.Add_Char((char)GET_HEX_CHAR_UPPER(b & 15));
    }
  }

  return true;
}

#endif
