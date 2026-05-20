/* LzmaSpec.cpp -- LZMA Reference Decoder
2015-06-14 : Igor Pavlov : Public domain */

// This code implements LZMA file decoding according to LZMA specification.
// This code is not optimized for speed.

#include <stdio.h>

#ifdef _MSC_VER
  #pragma warning(disable : 4710) // function not inlined
  #pragma warning(disable : 4996) // This function or variable may be unsafe
#endif

typedef unsigned char Byte;
typedef unsigned short UInt16;

#ifdef _LZMA_UINT32_IS_ULONG
  typedef unsigned long UInt32;
#else
  typedef unsigned int UInt32;
#endif

#if defined(_MSC_VER) || defined(__BORLANDC__)
  typedef unsigned __int64 UInt64;
#else
  typedef unsigned long long int UInt64;
#endif


struct CInputStream
{
  FILE *File;
  UInt64 Processed;
  
  void Init() { Processed = 0; }

  Byte ReadByte()
  {
    int c = getc(File);
    if (c < 0)
      throw "Unexpected end of file";
    Processed++;
    return (Byte)c;
  }
};


struct COutStream
{
  FILE *File;
  UInt64 Processed;

  void Init() { Processed = 0; }

  void WriteByte(Byte b)
  {
    if (putc(b, File) == EOF)
      throw "File writing error";
    Processed++;
  }
};


class COutWindow
{
  Byte *Buf;
  UInt32 Pos;
  UInt32 Size;
  bool IsFull;

public:
  unsigned TotalPos;
  COutStream OutStream;

  COutWindow(): Buf(NULL) {}
  ~COutWindow() { delete []Buf; }
 
  void Create(UInt32 dictSize)
  {
    Buf = new Byte[dictSize];
    Pos = 0;
    Size = dictSize;
    IsFull = false;
    TotalPos = 0;
  }

  void PutByte(Byte b)
  {
    TotalPos++;
    Buf[Pos++] = b;
    if (Pos == Size)
    {
      Pos = 0;
      IsFull = true;
    }
    OutStream.WriteByte(b);
  }

  Byte GetByte(UInt32 dist) const
  {
    return Buf[dist <= Pos ? Pos - dist : Size - dist + Pos];
  }

  void CopyMatch(UInt32 dist, unsigned len)
  {
    for (; len > 0; len--)
      PutByte(GetByte(dist));
  }

  bool CheckDistance(UInt32 dist) const
  {
    return dist <= Pos || IsFull;
  }

  bool IsEmpty() const
  {
    return Pos == 0 && !IsFull;
  }
};


#define kNumBitModelTotalBits 11
#define kNumMoveBits 5

typedef UInt16 CProb;

#define PROB_INIT_VAL ((1 << kNumBitModelTotalBits) / 2)

#define INIT_PROBS(p) \
 { for (unsigned i = 0; i < sizeof(p) / sizeof(p[0]); i++) p[i] = PROB_INIT_VAL; }

class CRangeDecoder
{
  UInt32 Range;
  UInt32 Code;

  void Normalize();

public:

  CInputStream *InStream;
  bool Corrupted;

  bool Init();
  bool IsFinishedOK() const { return Code == 0; }

  UInt32 DecodeDirectBits(unsigned numBits);
  unsigned DecodeBit(CProb *prob);
};

bool CRangeDecoder::Init()
{
  Corrupted = false;
  Range = 0xFFFFFFFF;
  Code = 0;

  Byte b = InStream->ReadByte();
  
  for (int i = 0; i < 4; i++)
    Code = (Code << 8) | InStream->ReadByte();
  
  if (b != 0 || Code == Range)
    Corrupted = true;
  return b == 0;
}

#define kTopValue ((UInt32)1 << 24)

void CRangeDecoder::Normalize()
{
  if (Range < kTopValue)
  {
    Range <<= 8;
    Code = (Code << 8) | InStream->ReadByte();
  }
}

UInt32 CRangeDecoder::DecodeDirectBits(unsigned numBits)
{
  UInt32 res = 0;
  do
  {
    Range >>= 1;
    Code -= Range;
    UInt32 t = 0 - ((UInt32)Code >> 31);
    Code += Range & t;
    
    if (Code == Range)
      Corrupted = true;
    
    Normalize();
    res <<= 1;
    res += t + 1;
  }
  while (--numBits);
  return res;
}

unsigned CRangeDecoder::DecodeBit(CProb *prob)
{
  unsigned v = *prob;
  UInt32 bound = (Range >> kNumBitModelTotalBits) * v;
  unsigned symbol;
  if (Code < bound)
  {
    v += ((1 << kNumBitModelTotalBits) - v) >> kNumMoveBits;
    Range = bound;
    symbol = 0;
  }
  else
  {
    v -= v >> kNumMoveBits;
    Code -= bound;
    Range -= bound;
    symbol = 1;
  }
  *prob = (CProb)v;
  Normalize();
  return symbol;
}


unsigned BitTreeReverseDecode(CProb *probs, unsigned numBits, CRangeDecoder *rc)
{
  unsigned m = 1;
  unsigned symbol = 0;
  for (unsigned i = 0; i < numBits; i++)
  {
    unsigned bit = rc->DecodeBit(&probs[m]);
    m <<= 1;
    m += bit;
    symbol |= (bit << i);
  }
  return symbol;
}

template <unsigned NumBits>
class CBitTreeDecoder
{
  CProb Probs[(unsigned)1 << NumBits];

public:

  void Init()
  {
    INIT_PROBS(Probs);
  }

  unsigned Decode(CRangeDecoder *rc)
  {
    unsigned m = 1;
    for (unsigned i = 0; i < NumBits; i++)
      m = (m << 1) + rc->DecodeBit(&Probs[m]);
    return m - ((unsigned)1 << NumBits);
  }

  unsigned ReverseDecode(CRangeDecoder *rc)
  {
    return BitTreeReverseDecode(Probs, NumBits, rc);
  }
};

#define kNumPosBitsMax 4

#define kNumStates 12
#define kNumLenToPosStates 4
#define kNumAlignBits 4
#define kStartPosModelIndex 4
#define kEndPosModelIndex 14
#define kNumFullDistances (1 << (kEndPosModelIndex >> 1))
#define kMatchMinLen 2

class CLenDecoder
{
  CProb Choice;
  CProb Choice2;
  CBitTreeDecoder<3> LowCoder[1 << kNumPosBitsMax];
  CBitTreeDecoder<3> MidCoder[1 << kNumPosBitsMax];
  CBitTreeDecoder<8> HighCoder;

public:

  void Init()
  {
    Choice = PROB_INIT_VAL;
    Choice2 = PROB_INIT_VAL;
    HighCoder.Init();
    for (unsigned i = 0; i < (1 << kNumPosBitsMax); i++)
    {
      LowCoder[i].Init();
      MidCoder[i].Init();
    }
  }

  unsigned Decode(CRangeDecoder *rc, unsigned posState)
  {
    if (rc->DecodeBit(&Choice) == 0)
      return LowCoder[posState].Decode(rc);
    if (rc->DecodeBit(&Choice2) == 0)
      return 8 + MidCoder[posState].Decode(rc);
    return 16 + HighCoder.Decode(rc);
  }
};

unsigned UpdateState_Literal(unsigned state)
{
  if (state < 4) return 0;
  else if (state < 10) return state - 3;
  else return state - 6;
}
unsigned UpdateState_Match   (unsigned state) { return state < 7 ? 7 : 10; }
unsigned UpdateState_Rep     (unsigned state) { return state < 7 ? 8 : 11; }
unsigned UpdateState_ShortRep(unsigned state) { return state < 7 ? 9 : 11; }

#define LZMA_DIC_MIN (1 << 12)

class CLzmaDecoder
{
public:
  CRangeDecoder RangeDec;
  COutWindow OutWindow;

  bool markerIsMandatory;
  unsigned lc, pb, lp;
  UInt32 dictSize;
  UInt32 dictSizeInProperties;

  void DecodeProperties(const Byte *properties)
  {
    unsigned d = properties[0];
    if (d >= (9 * 5 * 5))
      throw "Incorrect LZMA properties";
    lc = d % 9;
    d /= 9;
    pb = d / 5;
    lp = d % 5;
    dictSizeInProperties = 0;
    for (int i = 0; i < 4; i++)
      dictSizeInProperties |= (UInt32)properties[i + 1] << (8 * i);
    dictSize = dictSizeInProperties;
    if (dictSize < LZMA_DIC_MIN)
      dictSize = LZMA_DIC_MIN;
  }

  CLzmaDecoder(): LitProbs(NULL) {}
  ~CLzmaDecoder() { delete []LitProbs; }

  void Create()
  {
    OutWindow.Create(dictSize);
    CreateLiterals();
  }

  int Decode(bool unpackSizeDefined, UInt64 unpackSize);
  
private:

  CProb *LitProbs;

  void CreateLiterals()
  {
    LitProbs = new CProb[(UInt32)0x300 << (lc + lp)];
  }
  
  void InitLiterals()
  {
    UInt32 num = (UInt32)0x300 << (lc + lp);
    for (UInt32 i = 0; i < num; i++)
      LitProbs[i] = PROB_INIT_VAL;
  }
  
  void DecodeLiteral(unsigned state, UInt32 rep0)
  {
    unsigned prevByte = 0;
    if (!OutWindow.IsEmpty())
      prevByte = OutWindow.GetByte(1);
    
    unsigned symbol = 1;
    unsigned litState = ((OutWindow.TotalPos & ((1 << lp) - 1)) << lc) + (prevByte >> (8 - lc));
    CProb *probs = &LitProbs[(UInt32)0x300 * litState];
    
    if (state >= 7)
    {
      unsigned matchByte = OutWindow.GetByte(rep0 + 1);
      do
      {
        unsigned matchBit = (matchByte >> 7) & 1;
        matchByte <<= 1;
        unsigned bit = RangeDec.DecodeBit(&probs[((1 + matchBit) << 8) + symbol]);
        symbol = (symbol << 1) | bit;
        if (matchBit != bit)
          break;
      }
      while (symbol < 0x100);
    }
    while (symbol < 0x100)
      symbol = (symbol << 1) | RangeDec.DecodeBit(&probs[symbol]);
    OutWindow.PutByte((Byte)(symbol - 0x100));
  }

  CBitTreeDecoder<6> PosSlotDecoder[kNumLenToPosStates];
  CBitTreeDecoder<kNumAlignBits> AlignDecoder;
  CProb PosDecoders[1 + kNumFullDistances - kEndPosModelIndex];
  
  void InitDist()
  {
    for (unsigned i = 0; i < kNumLenToPosStates; i++)
      PosSlotDecoder[i].Init();
    AlignDecoder.Init();
    INIT_PROBS(PosDecoders);
  }
  
  unsigned DecodeDistance(unsigned len)
  {
    unsigned lenState = len;
    if (lenState > kNumLenToPosStates - 1)
      lenState = kNumLenToPosStates - 1;
    
    unsigned posSlot = PosSlotDecoder[lenState].Decode(&RangeDec);
    if (posSlot < 4)
      return posSlot;
    
    unsigned numDirectBits = (unsigned)((posSlot >> 1) - 1);
    UInt32 dist = ((2 | (posSlot & 1)) << numDirectBits);
    if (posSlot < kEndPosModelIndex)
      dist += BitTreeReverseDecode(PosDecoders + dist - posSlot, numDirectBits, &RangeDec);
    else
    {
      dist += RangeDec.DecodeDirectBits(numDirectBits - kNumAlignBits) << kNumAlignBits;
      dist += AlignDecoder.ReverseDecode(&RangeDec);
    }
    return dist;
  }

  CProb IsMatch[kNumStates << kNumPosBitsMax];
  CProb IsRep[kNumStates];
  CProb IsRepG0[kNumStates];
  CProb IsRepG1[kNumStates];
  CProb IsRepG2[kNumStates];
  CProb IsRep0Long[kNumStates << kNumPosBitsMax];

  CLenDecoder LenDecoder;
  CLenDecoder RepLenDecoder;

  void Init()
  {
    InitLiterals();
    InitDist();

    INIT_PROBS(IsMatch);
    INIT_PROBS(IsRep);
    INIT_PROBS(IsRepG0);
    INIT_PROBS(IsRepG1);
    INIT_PROBS(IsRepG2);
    INIT_PROBS(IsRep0Long);

    LenDecoder.Init();
    RepLenDecoder.Init();
  }
};
    

#define LZMA_RES_ERROR                   0
#define LZMA_RES_FINISHED_WITH_MARKER    1
#define LZMA_RES_FINISHED_WITHOUT_MARKER 2

int CLzmaDecoder::Decode(bool unpackSizeDefined, UInt64 unpackSize)
{
  if (!RangeDec.Init())
    return LZMA_RES_ERROR;

  Init();

  UInt32 rep0 = 0, rep1 = 0, rep2 = 0, rep3 = 0;
  unsigned state = 0;
  
  for (;;)
  {
    if (unpackSizeDefined && unpackSize == 0 && !markerIsMandatory)
      if (RangeDec.IsFinishedOK())
        return LZMA_RES_FINISHED_WITHOUT_MARKER;

    unsigned posState = OutWindow.TotalPos & ((1 << pb) - 1);

    if (RangeDec.DecodeBit(&IsMatch[(state << kNumPosBitsMax) + posState]) == 0)
    {
      if (unpackSizeDefined && unpackSize == 0)
        return LZMA_RES_ERROR;
      DecodeLiteral(state, rep0);
      state = UpdateState_Literal(state);
      unpackSize--;
      continue;
    }
    
    unsigned len;
    
    if (RangeDec.DecodeBit(&IsRep[state]) != 0)
    {
      if (unpackSizeDefined && unpackSize == 0)
        return LZMA_RES_ERROR;
      if (OutWindow.IsEmpty())
        return LZMA_RES_ERROR;
      if (RangeDec.DecodeBit(&IsRepG0[state]) == 0)
      {
        if (RangeDec.DecodeBit(&IsRep0Long[(state << kNumPosBitsMax) + posState]) == 0)
        {
          state = UpdateState_ShortRep(state);
          OutWindow.PutByte(OutWindow.GetByte(rep0 + 1));
          unpackSize--;
          continue;
        }
      }
      else
      {
        UInt32 dist;
        if (RangeDec.DecodeBit(&IsRepG1[state]) == 0)
          dist = rep1;
        else
        {
          if (RangeDec.DecodeBit(&IsRepG2[state]) == 0)
            dist = rep2;
          else
          {
            dist = rep3;
            rep3 = rep2;
          }
          rep2 = rep1;
        }
        rep1 = rep0;
        rep0 = dist;
      }
      len = RepLenDecoder.Decode(&RangeDec, posState);
      state = UpdateState_Rep(state);
    }
    else
    {
      rep3 = rep2;
      rep2 = rep1;
      rep1 = rep0;
      len = LenDecoder.Decode(&RangeDec, posState);
      state = UpdateState_Match(state);
      rep0 = DecodeDistance(len);
      if (rep0 == 0xFFFFFFFF)
        return RangeDec.IsFinishedOK() ?
            LZMA_RES_FINISHED_WITH_MARKER :
            LZMA_RES_ERROR;

      if (unpackSizeDefined && unpackSize == 0)
        return LZMA_RES_ERROR;
      if (rep0 >= dictSize || !OutWindow.CheckDistance(rep0))
        return LZMA_RES_ERROR;
    }
    len += kMatchMinLen;
    bool isError = false;
    if (unpackSizeDefined && unpackSize < len)
    {
      len = (unsigned)unpackSize;
      isError = true;
    }
    OutWindow.CopyMatch(rep0 + 1, len);
    unpackSize -= len;
    if (isError)
      return LZMA_RES_ERROR;
  }
}

static void Print(const char *s)
{
  fputs(s, stdout);
}

static void PrintError(const char *s)
{
  fputs(s, stderr);
}


#define CONVERT_INT_TO_STR(charType, tempSize) \

void ConvertUInt64ToString(UInt64 val, char *s)
{
  char temp[32];
  unsigned i = 0;
  while (val >= 10)
  {
    temp[i++] = (char)('0' + (unsigned)(val % 10));
    val /= 10;
  }
  *s++ = (char)('0' + (unsigned)val);
  while (i != 0)
  {
    i--;
    *s++ = temp[i];
  }
  *s = 0;
}

void PrintUInt64(const char *title, UInt64 v)
{
  Print(title);
  Print(" : ");
  char s[32];
  ConvertUInt64ToString(v, s);
  Print(s);
  Print(" bytes \n");
}

int main2(int numArgs, const char *args[])
{
  Print("\nLZMA Reference Decoder 15.00 : Igor Pavlov : Public domain : 2015-04-16\n");
  if (numArgs == 1)
    Print("\nUse: lzmaSpec a.lzma outFile");

  if (numArgs != 3)
    throw "you must specify two parameters";

  CInputStream inStream;
  inStream.File = fopen(args[1], "rb");
  inStream.Init();
  if (inStream.File == 0)
    throw "Can't open input file";

  CLzmaDecoder lzmaDecoder;
  lzmaDecoder.OutWindow.OutStream.File = fopen(args[2], "wb+");
  lzmaDecoder.OutWindow.OutStream.Init();
  if (inStream.File == 0)
    throw "Can't open output file";

  Byte header[13];
  int i;
  for (i = 0; i < 13; i++)
    header[i] = inStream.ReadByte();

  lzmaDecoder.DecodeProperties(header);

  printf("\nlc=%d, lp=%d, pb=%d", lzmaDecoder.lc, lzmaDecoder.lp, lzmaDecoder.pb);
  printf("\nDictionary Size in properties = %u", lzmaDecoder.dictSizeInProperties);
  printf("\nDictionary Size for decoding  = %u", lzmaDecoder.dictSize);

  UInt64 unpackSize = 0;
  bool unpackSizeDefined = false;
  for (i = 0; i < 8; i++)
  {
    Byte b = header[5 + i];
    if (b != 0xFF)
      unpackSizeDefined = true;
    unpackSize |= (UInt64)b << (8 * i);
  }

  lzmaDecoder.markerIsMandatory = !unpackSizeDefined;

  Print("\n");
  if (unpackSizeDefined)
    PrintUInt64("Uncompressed Size", unpackSize);
  else
    Print("End marker is expected\n");
  lzmaDecoder.RangeDec.InStream = &inStream;

  Print("\n");

  lzmaDecoder.Create();
  
  int res = lzmaDecoder.Decode(unpackSizeDefined, unpackSize);

  PrintUInt64("Read    ", inStream.Processed);
  PrintUInt64("Written ", lzmaDecoder.OutWindow.OutStream.Processed);

  if (res == LZMA_RES_ERROR)
    throw "LZMA decoding error";
  else if (res == LZMA_RES_FINISHED_WITHOUT_MARKER)
    Print("Finished without end marker");
  else if (res == LZMA_RES_FINISHED_WITH_MARKER)
  {
    if (unpackSizeDefined)
    {
      if (lzmaDecoder.OutWindow.OutStream.Processed != unpackSize)
        throw "Finished with end marker before than specified size";
      Print("Warning: ");
    }
    Print("Finished with end marker");
  }
  else
    throw "Internal Error";

  Print("\n");
  
  if (lzmaDecoder.RangeDec.Corrupted)
  {
    Print("\nWarning: LZMA stream is corrupted\n");
  }

  return 0;
}


int
  #ifdef _MSC_VER
    __cdecl
  #endif
main(int numArgs, const char *args[])
{
  try { return main2(numArgs, args); }
  catch (const char *s)
  {
    PrintError("\nError:\n");
    PrintError(s);
    PrintError("\n");
    return 1;
  }
  catch(...)
  {
    PrintError("\nError\n");
    return 1;
  }
}
