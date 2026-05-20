  // LzmaAlone.cpp

#include "StdAfx.h"

// #include <stdio.h>

#if (defined(_WIN32) || defined(OS2) || defined(MSDOS)) && !defined(UNDER_CE)
#include <fcntl.h>
#include <io.h>
#define MY_SET_BINARY_MODE(file) _setmode(_fileno(file), O_BINARY)
#else
#define MY_SET_BINARY_MODE(file)
#endif

#include "../../../../C/CpuArch.h"
#include "../../../../C/7zVersion.h"
#include "../../../../C/Alloc.h"
#include "../../../../C/Lzma86.h"

#include "../../../Common/MyWindows.h"
#include "../../../Common/MyInitGuid.h"

#include "../../../Windows/NtCheck.h"

#ifndef Z7_ST
#include "../../../Windows/System.h"
#endif

#include "../../../Common/IntToString.h"
#include "../../../Common/CommandLineParser.h"
#include "../../../Common/StringConvert.h"
#include "../../../Common/StringToInt.h"

#include "../../Common/FileStreams.h"
#include "../../Common/StreamUtils.h"

#include "../../Compress/LzmaDecoder.h"
#include "../../Compress/LzmaEncoder.h"

#include "../../UI/Console/BenchCon.h"
#include "../../UI/Console/ConsoleClose.h"

extern
bool g_LargePagesMode;
bool g_LargePagesMode = false;

using namespace NCommandLineParser;

static const unsigned kDictSizeLog = 24;

#define kCopyrightString "\nLZMA " MY_VERSION_CPU " : " MY_COPYRIGHT_DATE "\n\n"

static const char * const kHelpString =
    "Usage:  lzma <command> [inputFile] [outputFile] [<switches>...]\n"
    "\n"
    "<command>\n"
    "  e : Encode file\n"
    "  d : Decode file\n"
    "  b : Benchmark\n"
    "<switches>\n"
    "  -a{N}  : set compression mode : [0, 1] : default = 1 (max)\n"
    "  -d{N}  : set dictionary size : [12, 31] : default = 24 (16 MiB)\n"
    "  -fb{N} : set number of fast bytes : [5, 273] : default = 128\n"
    "  -mc{N} : set number of cycles for match finder\n"
    "  -lc{N} : set number of literal context bits : [0, 8] : default = 3\n"
    "  -lp{N} : set number of literal pos bits : [0, 4] : default = 0\n"
    "  -pb{N} : set number of pos bits : [0, 4] : default = 2\n"
    "  -mf{M} : set match finder: [hc4, hc5, bt2, bt3, bt4, bt5] : default = bt4\n"
    "  -mt{N} : set number of CPU threads\n"
    "  -eos   : write end of stream marker\n"
    "  -si    : read data from stdin\n"
    "  -so    : write data to stdout\n";


static const char * const kCantAllocate = "Cannot allocate memory";
static const char * const kReadError = "Read error";
static const char * const kWriteError = "Write error";


namespace NKey {
enum Enum
{
  kHelp1 = 0,
  kHelp2,
  kMethod,
  kLevel,
  kAlgo,
  kDict,
  kFb,
  kMc,
  kLc,
  kLp,
  kPb,
  kMatchFinder,
  kMultiThread,
  kEOS,
  kStdIn,
  kStdOut,
  kFilter86
};
}

#define SWFRM_3(t, mu, mi) t, mu, mi, NULL

#define SWFRM_1(t) SWFRM_3(t, false, 0)
#define SWFRM_SIMPLE SWFRM_1(NSwitchType::kSimple)
#define SWFRM_STRING SWFRM_1(NSwitchType::kString)

#define SWFRM_STRING_SINGL(mi) SWFRM_3(NSwitchType::kString, false, mi)

static const CSwitchForm kSwitchForms[] =
{
  { "?",  SWFRM_SIMPLE },
  { "H",  SWFRM_SIMPLE },
  { "MM", SWFRM_STRING_SINGL(1) },
  { "X", SWFRM_STRING_SINGL(1) },
  { "A", SWFRM_STRING_SINGL(1) },
  { "D", SWFRM_STRING_SINGL(1) },
  { "FB", SWFRM_STRING_SINGL(1) },
  { "MC", SWFRM_STRING_SINGL(1) },
  { "LC", SWFRM_STRING_SINGL(1) },
  { "LP", SWFRM_STRING_SINGL(1) },
  { "PB", SWFRM_STRING_SINGL(1) },
  { "MF", SWFRM_STRING_SINGL(1) },
  { "MT", SWFRM_STRING },
  { "EOS", SWFRM_SIMPLE },
  { "SI",  SWFRM_SIMPLE },
  { "SO",  SWFRM_SIMPLE },
  { "F86",  NSwitchType::kChar, false, 0, "+" }
};


static void Convert_UString_to_AString(const UString &s, AString &temp)
{
  int codePage = CP_OEMCP;
  /*
  int g_CodePage = -1;
  int codePage = g_CodePage;
  if (codePage == -1)
    codePage = CP_OEMCP;
  if (codePage == CP_UTF8)
    ConvertUnicodeToUTF8(s, temp);
  else
  */
    UnicodeStringToMultiByte2(temp, s, (UINT)codePage);
}

static void PrintErr(const char *s)
{
  fputs(s, stderr);
}

static void PrintErr_LF(const char *s)
{
  PrintErr(s);
  fputc('\n', stderr);
}


static void PrintError(const char *s)
{
  PrintErr("\nERROR: ");
  PrintErr_LF(s);
}

static void PrintError2(const char *s1, const UString &s2)
{
  PrintError(s1);
  AString a;
  Convert_UString_to_AString(s2, a);
  PrintErr_LF(a);
}

static void PrintError_int(const char *s, int code)
{
  PrintError(s);
  char temp[32];
  ConvertInt64ToString(code, temp);
  PrintErr("Error code = ");
  PrintErr_LF(temp);
}



static void Print(const char *s)
{
  fputs(s, stdout);
}

static void Print_UInt64(UInt64 v)
{
  char temp[32];
  ConvertUInt64ToString(v, temp);
  Print(temp);
}

static void Print_MB(UInt64 v)
{
  Print_UInt64(v);
  Print(" MiB");
}

static void Print_Size(const char *s, UInt64 v)
{
  Print(s);
  Print_UInt64(v);
  Print(" (");
  Print_MB(v >> 20);
  Print(")\n");
}

static void PrintTitle()
{
  Print(kCopyrightString);
}

static void PrintHelp()
{
  PrintTitle();
  Print(kHelpString);
}


Z7_CLASS_IMP_COM_1(
  CProgressPrint,
  ICompressProgressInfo
)
  UInt64 _size1;
  UInt64 _size2;
public:
  CProgressPrint(): _size1(0), _size2(0) {}

  void ClosePrint();
};

#define BACK_STR \
"\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b"
static const char * const kBackSpaces =
BACK_STR
"                                                                "
BACK_STR;


void CProgressPrint::ClosePrint()
{
  Print(kBackSpaces);
}

Z7_COM7F_IMF(CProgressPrint::SetRatioInfo(const UInt64 *inSize, const UInt64 *outSize))
{
  if (NConsoleClose::TestBreakSignal())
    return E_ABORT;
  if (inSize)
  {
    UInt64 v1 = *inSize >> 20;
    UInt64 v2 = _size2;
    if (outSize)
      v2 = *outSize >> 20;
    if (v1 != _size1 || v2 != _size2)
    {
      _size1 = v1;
      _size2 = v2;
      ClosePrint();
      Print_MB(_size1);
      Print(" -> ");
      Print_MB(_size2);
    }
  }
  return S_OK;
}


Z7_ATTR_NORETURN
static void IncorrectCommand()
{
  throw "Incorrect command";
}

static UInt32 GetNumber(const wchar_t *s)
{
  const wchar_t *end;
  UInt32 v = ConvertStringToUInt32(s, &end);
  if (*end != 0)
    IncorrectCommand();
  return v;
}

static void ParseUInt32(const CParser &parser, unsigned index, UInt32 &res)
{
  if (parser[index].ThereIs)
    res = GetNumber(parser[index].PostStrings[0]);
}


static int Error_HRESULT(const char *s, HRESULT res)
{
  if (res == E_ABORT)
  {
    Print("\n\nBreak signaled\n");
    return 255;
  }

  PrintError(s);

  if (res == E_OUTOFMEMORY)
  {
    PrintErr_LF(kCantAllocate);
    return 8;
  }
  if (res == E_INVALIDARG)
  {
    PrintErr_LF("Ununsupported parameter");
  }
  else
  {
    char temp[32];
    ConvertUInt32ToHex((UInt32)res, temp);
    PrintErr("Error code = 0x");
    PrintErr_LF(temp);
  }
  return 1;
}

#if defined(_UNICODE) && !defined(_WIN64) && !defined(UNDER_CE)
#define NT_CHECK_FAIL_ACTION PrintError("Unsupported Windows version"); return 1;
#endif

static void AddProp(CObjectVector<CProperty> &props2, const char *name, const wchar_t *val)
{
  CProperty &prop = props2.AddNew();
  prop.Name = name;
  prop.Value = val;
}

static int main2(int numArgs, const char *args[])
{
  NT_CHECK

  if (numArgs == 1)
  {
    PrintHelp();
    return 0;
  }

  /*
  bool unsupportedTypes = (sizeof(Byte) != 1 || sizeof(UInt32) < 4 || sizeof(UInt64) < 8);
  if (unsupportedTypes)
    throw "Unsupported base types. Edit Common/Types.h and recompile";
  */

  UStringVector commandStrings;
  for (int i = 1; i < numArgs; i++)
    commandStrings.Add(MultiByteToUnicodeString(args[i]));
  
  CParser parser;
  try
  {
    if (!parser.ParseStrings(kSwitchForms, Z7_ARRAY_SIZE(kSwitchForms), commandStrings))
    {
      PrintError2(parser.ErrorMessage, parser.ErrorLine);
      return 1;
    }
  }
  catch(...)
  {
    IncorrectCommand();
  }

  if (parser[NKey::kHelp1].ThereIs || parser[NKey::kHelp2].ThereIs)
  {
    PrintHelp();
    return 0;
  }

  const bool stdInMode = parser[NKey::kStdIn].ThereIs;
  const bool stdOutMode = parser[NKey::kStdOut].ThereIs;

  if (!stdOutMode)
    PrintTitle();

  const UStringVector &params = parser.NonSwitchStrings;

  unsigned paramIndex = 0;
  if (paramIndex >= params.Size())
    IncorrectCommand();
  const UString &command = params[paramIndex++];

  CObjectVector<CProperty> props2;
  bool dictDefined = false;
  UInt32 dict = (UInt32)(Int32)-1;
  
  if (parser[NKey::kDict].ThereIs)
  {
    UInt32 dictLog;
    const UString &s = parser[NKey::kDict].PostStrings[0];
    dictLog = GetNumber(s);
    if (dictLog >= 32)
      throw "unsupported dictionary size";
    // we only want to use dictionary sizes that are powers of 2,
    // because 7-zip only recognizes such dictionary sizes in the lzma header.#if 0
#if 0
    if (dictLog == 32)
      dict = (UInt32)3840 << 20;
    else
#endif
    dict = (UInt32)1 << dictLog;
    dictDefined = true;
    AddProp(props2, "d", s);
  }
  
  if (parser[NKey::kLevel].ThereIs)
  {
    const UString &s = parser[NKey::kLevel].PostStrings[0];
    /* UInt32 level = */ GetNumber(s);
    AddProp(props2, "x", s);
  }
  
  UString mf ("BT4");
  if (parser[NKey::kMatchFinder].ThereIs)
    mf = parser[NKey::kMatchFinder].PostStrings[0];

  UInt32 numThreads = (UInt32)(Int32)-1;

  #ifndef Z7_ST
  
  if (parser[NKey::kMultiThread].ThereIs)
  {
    const UString &s = parser[NKey::kMultiThread].PostStrings[0];
    if (s.IsEmpty())
      numThreads = NWindows::NSystem::GetNumberOfProcessors();
    else
      numThreads = GetNumber(s);
    AddProp(props2, "mt", s);
  }
  
  #endif

  
  if (parser[NKey::kMethod].ThereIs)
  {
    const UString &s = parser[NKey::kMethod].PostStrings[0];
    if (s.IsEmpty() || s[0] != '=')
      IncorrectCommand();
    AddProp(props2, "m", s.Ptr(1));
  }

  if (StringsAreEqualNoCase_Ascii(command, "b"))
  {
    UInt32 numIterations = 1;
    if (paramIndex < params.Size())
      numIterations = GetNumber(params[paramIndex++]);
    if (params.Size() != paramIndex)
      IncorrectCommand();
  
    HRESULT res = BenchCon(props2, numIterations, stdout);
    
    if (res == S_OK)
      return 0;
    return Error_HRESULT("Benchmark error", res);
  }

  {
    UInt32 needParams = 3;
    if (stdInMode) needParams--;
    if (stdOutMode) needParams--;
    if (needParams != params.Size())
      IncorrectCommand();
  }

  if (numThreads == (UInt32)(Int32)-1)
    numThreads = 1;

  bool encodeMode = false;
  
  if (StringsAreEqualNoCase_Ascii(command, "e"))
    encodeMode = true;
  else if (!StringsAreEqualNoCase_Ascii(command, "d"))
    IncorrectCommand();

  CMyComPtr<ISequentialInStream> inStream;
  CInFileStream *inStreamSpec = NULL;
  
  if (stdInMode)
  {
    inStream = new CStdInFileStream;
    MY_SET_BINARY_MODE(stdin);
  }
  else
  {
    const UString &inputName = params[paramIndex++];
    inStreamSpec = new CInFileStream;
    inStream = inStreamSpec;
    if (!inStreamSpec->Open(us2fs(inputName)))
    {
      PrintError2("Cannot open input file", inputName);
      return 1;
    }
  }

  CMyComPtr<ISequentialOutStream> outStream;
  COutFileStream *outStreamSpec = NULL;
  
  if (stdOutMode)
  {
    outStream = new CStdOutFileStream;
    MY_SET_BINARY_MODE(stdout);
  }
  else
  {
    const UString &outputName = params[paramIndex++];
    outStreamSpec = new COutFileStream;
    outStream = outStreamSpec;
    if (!outStreamSpec->Create_ALWAYS(us2fs(outputName)))
    {
      PrintError2("Cannot open output file", outputName);
      return 1;
    }
  }

  bool fileSizeDefined = false;
  UInt64 fileSize = 0;
  
  if (inStreamSpec)
  {
    if (!inStreamSpec->GetLength(fileSize))
      throw "Cannot get file length";
    fileSizeDefined = true;
    if (!stdOutMode)
      Print_Size("Input size:  ", fileSize);
  }

  if (encodeMode && !dictDefined)
  {
    dict = (UInt32)1 << kDictSizeLog;
    if (fileSizeDefined)
    {
      unsigned i;
      for (i = 16; i < kDictSizeLog; i++)
        if ((UInt32)((UInt32)1 << i) >= fileSize)
          break;
      dict = (UInt32)1 << i;
    }
  }

  if (parser[NKey::kFilter86].ThereIs)
  {
    /* -f86 switch is for x86 filtered mode: BCJ + LZMA.
       It uses modified header format.
       It's not recommended to use -f86 mode now.
       You can use xz format instead, if you want to use filters */

    if (parser[NKey::kEOS].ThereIs || stdInMode)
      throw "Cannot use stdin in this mode";

    size_t inSize = (size_t)fileSize;

    if (inSize != fileSize)
      throw "File is too big";

    Byte *inBuffer = NULL;
    
    if (inSize != 0)
    {
      inBuffer = (Byte *)MyAlloc((size_t)inSize);
      if (!inBuffer)
        throw kCantAllocate;
    }
    
    if (ReadStream_FAIL(inStream, inBuffer, inSize) != S_OK)
      throw "Cannot read";

    Byte *outBuffer = NULL;
    size_t outSize;
    
    if (encodeMode)
    {
      // we allocate 105% of original size for output buffer
      UInt64 outSize64 = fileSize / 20 * 21 + (1 << 16);

      outSize = (size_t)outSize64;
      
      if (outSize != outSize64)
        throw "File is too big";

      if (outSize != 0)
      {
        outBuffer = (Byte *)MyAlloc((size_t)outSize);
        if (!outBuffer)
          throw kCantAllocate;
      }
      
      int res = Lzma86_Encode(outBuffer, &outSize, inBuffer, inSize,
          5, dict, parser[NKey::kFilter86].PostCharIndex == 0 ? SZ_FILTER_YES : SZ_FILTER_AUTO);
  
      if (res != 0)
      {
        PrintError_int("Encode error", (int)res);
        return 1;
      }
    }
    else
    {
      UInt64 outSize64;
      
      if (Lzma86_GetUnpackSize(inBuffer, inSize, &outSize64) != 0)
        throw "data error";
      
      outSize = (size_t)outSize64;
      if (outSize != outSize64)
        throw "Unpack size is too big";
      if (outSize != 0)
      {
        outBuffer = (Byte *)MyAlloc(outSize);
        if (!outBuffer)
          throw kCantAllocate;
      }
      
      int res = Lzma86_Decode(outBuffer, &outSize, inBuffer, &inSize);
      
      if (inSize != (size_t)fileSize)
        throw "incorrect processed size";
      if (res != 0)
      {
        PrintError_int("Decode error", (int)res);
        return 1;
      }
    }
    
    if (WriteStream(outStream, outBuffer, outSize) != S_OK)
      throw kWriteError;
    
    MyFree(outBuffer);
    MyFree(inBuffer);
  }
  else
  {

  CProgressPrint *progressSpec = NULL;
  CMyComPtr<ICompressProgressInfo> progress;

  if (!stdOutMode)
  {
    progressSpec = new CProgressPrint;
    progress = progressSpec;
  }

  if (encodeMode)
  {
    NCompress::NLzma::CEncoder *encoderSpec = new NCompress::NLzma::CEncoder;
    CMyComPtr<ICompressCoder> encoder = encoderSpec;

    UInt32 pb = 2;
    UInt32 lc = 3; // = 0; for 32-bit data
    UInt32 lp = 0; // = 2; for 32-bit data
    UInt32 algo = 1;
    UInt32 fb = 128;
    UInt32 mc = 16 + fb / 2;
    bool mcDefined = false;

    bool eos = parser[NKey::kEOS].ThereIs || stdInMode;
 
    ParseUInt32(parser, NKey::kAlgo, algo);
    ParseUInt32(parser, NKey::kFb, fb);
    ParseUInt32(parser, NKey::kLc, lc);
    ParseUInt32(parser, NKey::kLp, lp);
    ParseUInt32(parser, NKey::kPb, pb);

    mcDefined = parser[NKey::kMc].ThereIs;
    if (mcDefined)
      mc = GetNumber(parser[NKey::kMc].PostStrings[0]);
    
    const PROPID propIDs[] =
    {
      NCoderPropID::kDictionarySize,
      NCoderPropID::kPosStateBits,
      NCoderPropID::kLitContextBits,
      NCoderPropID::kLitPosBits,
      NCoderPropID::kAlgorithm,
      NCoderPropID::kNumFastBytes,
      NCoderPropID::kMatchFinder,
      NCoderPropID::kEndMarker,
      NCoderPropID::kNumThreads,
      NCoderPropID::kMatchFinderCycles,
    };

    const unsigned kNumPropsMax = Z7_ARRAY_SIZE(propIDs);

    PROPVARIANT props[kNumPropsMax];
    for (int p = 0; p < 6; p++)
      props[p].vt = VT_UI4;

    props[0].ulVal = (UInt32)dict;
    props[1].ulVal = (UInt32)pb;
    props[2].ulVal = (UInt32)lc;
    props[3].ulVal = (UInt32)lp;
    props[4].ulVal = (UInt32)algo;
    props[5].ulVal = (UInt32)fb;

    props[6].vt = VT_BSTR;
    props[6].bstrVal = const_cast<BSTR>((const wchar_t *)mf);

    props[7].vt = VT_BOOL;
    props[7].boolVal = eos ? VARIANT_TRUE : VARIANT_FALSE;

    props[8].vt = VT_UI4;
    props[8].ulVal = (UInt32)numThreads;

    // it must be last in property list
    props[9].vt = VT_UI4;
    props[9].ulVal = (UInt32)mc;

    unsigned numProps = kNumPropsMax;
    if (!mcDefined)
      numProps--;

    HRESULT res = encoderSpec->SetCoderProperties(propIDs, props, numProps);
    if (res != S_OK)
      return Error_HRESULT("incorrect encoder properties", res);

    if (encoderSpec->WriteCoderProperties(outStream) != S_OK)
      throw kWriteError;

    bool fileSizeWasUsed = true;
    if (eos || stdInMode)
    {
      fileSize = (UInt64)(Int64)-1;
      fileSizeWasUsed = false;
    }

    {
      Byte temp[8];
      for (int i = 0; i < 8; i++)
        temp[i]= (Byte)(fileSize >> (8 * i));
      if (WriteStream(outStream, temp, 8) != S_OK)
        throw kWriteError;
    }
  
    res = encoder->Code(inStream, outStream, NULL, NULL, progress);
    if (progressSpec)
      progressSpec->ClosePrint();

    if (res != S_OK)
      return Error_HRESULT("Encoding error", res);

    UInt64 processedSize = encoderSpec->GetInputProcessedSize();
    
    if (fileSizeWasUsed && processedSize != fileSize)
      throw "Incorrect size of processed data";
  }
  else
  {
    NCompress::NLzma::CDecoder *decoderSpec = new NCompress::NLzma::CDecoder;
    CMyComPtr<ICompressCoder> decoder = decoderSpec;
    
    decoderSpec->FinishStream = true;
    
    const unsigned kPropertiesSize = 5;
    Byte header[kPropertiesSize + 8];

    if (ReadStream_FALSE(inStream, header, kPropertiesSize + 8) != S_OK)
      throw kReadError;
    
    if (decoderSpec->SetDecoderProperties2(header, kPropertiesSize) != S_OK)
      throw "SetDecoderProperties error";
    
    UInt64 unpackSize = 0;
    for (unsigned i = 0; i < 8; i++)
      unpackSize |= ((UInt64)header[kPropertiesSize + i]) << (8 * i);

    bool unpackSizeDefined = (unpackSize != (UInt64)(Int64)-1);

    HRESULT res = decoder->Code(inStream, outStream, NULL, unpackSizeDefined ? &unpackSize : NULL, progress);
    if (progressSpec)
      progressSpec->ClosePrint();

    if (res != S_OK)
    {
      if (res == S_FALSE)
      {
        PrintError("Decoding error");
        return 1;
      }
      return Error_HRESULT("Decoding error", res);
    }
    
    if (unpackSizeDefined && unpackSize != decoderSpec->GetOutputProcessedSize())
      throw "incorrect uncompressed size in header";
  }
  }

  if (outStreamSpec)
  {
    if (!stdOutMode)
      Print_Size("Output size: ", outStreamSpec->ProcessedSize);
    if (outStreamSpec->Close() != S_OK)
      throw "File closing error";
  }

  return 0;
}

int Z7_CDECL main(int numArgs, const char *args[])
{
  NConsoleClose::CCtrlHandlerSetter ctrlHandlerSetter;

  try { return main2(numArgs, args); }
  catch (const char *s)
  {
    PrintError(s);
    return 1;
  }
  catch(...)
  {
    PrintError("Unknown Error");
    return 1;
  }
}
