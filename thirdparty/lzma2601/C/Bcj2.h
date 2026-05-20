/* Bcj2.h -- BCJ2 converter for x86 code (Branch CALL/JUMP variant2)
2023-03-02 : Igor Pavlov : Public domain */

#ifndef ZIP7_INC_BCJ2_H
#define ZIP7_INC_BCJ2_H

#include "7zTypes.h"

EXTERN_C_BEGIN

#define BCJ2_NUM_STREAMS 4

enum
{
  BCJ2_STREAM_MAIN,
  BCJ2_STREAM_CALL,
  BCJ2_STREAM_JUMP,
  BCJ2_STREAM_RC
};

enum
{
  BCJ2_DEC_STATE_ORIG_0 = BCJ2_NUM_STREAMS,
  BCJ2_DEC_STATE_ORIG_1,
  BCJ2_DEC_STATE_ORIG_2,
  BCJ2_DEC_STATE_ORIG_3,
  
  BCJ2_DEC_STATE_ORIG,
  BCJ2_DEC_STATE_ERROR     /* after detected data error */
};

enum
{
  BCJ2_ENC_STATE_ORIG = BCJ2_NUM_STREAMS,
  BCJ2_ENC_STATE_FINISHED  /* it's state after fully encoded stream */
};


/* #define BCJ2_IS_32BIT_STREAM(s) ((s) == BCJ2_STREAM_CALL || (s) == BCJ2_STREAM_JUMP) */
#define BCJ2_IS_32BIT_STREAM(s) ((unsigned)((unsigned)(s) - (unsigned)BCJ2_STREAM_CALL) < 2)

/*
CBcj2Dec / CBcj2Enc
bufs sizes:
  BUF_SIZE(n) = lims[n] - bufs[n]
bufs sizes for BCJ2_STREAM_CALL and BCJ2_STREAM_JUMP must be multiply of 4:
    (BUF_SIZE(BCJ2_STREAM_CALL) & 3) == 0
    (BUF_SIZE(BCJ2_STREAM_JUMP) & 3) == 0
*/

// typedef UInt32 CBcj2Prob;
typedef UInt16 CBcj2Prob;

/*
BCJ2 encoder / decoder internal requirements:
  - If last bytes of stream contain marker (e8/e8/0f8x), then
    there is also encoded symbol (0 : no conversion) in RC stream.
  - One case of overlapped instructions is supported,
    if last byte of converted instruction is (0f) and next byte is (8x):
      marker [xx xx xx 0f] 8x
    then the pair (0f 8x) is treated as marker.
*/

/* ---------- BCJ2 Decoder ---------- */

/*
CBcj2Dec:
(dest) is allowed to overlap with bufs[BCJ2_STREAM_MAIN], with the following conditions:
  bufs[BCJ2_STREAM_MAIN] >= dest &&
  bufs[BCJ2_STREAM_MAIN] - dest >=
        BUF_SIZE(BCJ2_STREAM_CALL) +
        BUF_SIZE(BCJ2_STREAM_JUMP)
  reserve = bufs[BCJ2_STREAM_MAIN] - dest -
      ( BUF_SIZE(BCJ2_STREAM_CALL) +
        BUF_SIZE(BCJ2_STREAM_JUMP) )
  and additional conditions:
  if (it's first call of Bcj2Dec_Decode() after Bcj2Dec_Init())
  {
    (reserve != 1) : if (ver <  v23.00)
  }
  else // if there are more than one calls of Bcj2Dec_Decode() after Bcj2Dec_Init())
  {
    (reserve >= 6) : if (ver <  v23.00)
    (reserve >= 4) : if (ver >= v23.00)
    We need that (reserve) because after first call of Bcj2Dec_Decode(),
    CBcj2Dec::temp can contain up to 4 bytes for writing to (dest).
  }
  (reserve == 0) is allowed, if we decode full stream via single call of Bcj2Dec_Decode().
  (reserve == 0) also is allowed in case of multi-call, if we use fixed buffers,
     and (reserve) is calculated from full (final) sizes of all streams before first call.
*/

typedef struct
{
  const Byte *bufs[BCJ2_NUM_STREAMS];
  const Byte *lims[BCJ2_NUM_STREAMS];
  Byte *dest;
  const Byte *destLim;

  unsigned state; /* BCJ2_STREAM_MAIN has more priority than BCJ2_STATE_ORIG */

  UInt32 ip;      /* property of starting base for decoding */
  UInt32 temp;    /* Byte temp[4]; */
  UInt32 range;
  UInt32 code;
  CBcj2Prob probs[2 + 256];
} CBcj2Dec;


/* Note:
   Bcj2Dec_Init() sets (CBcj2Dec::ip = 0)
   if (ip != 0) property is required, the caller must set CBcj2Dec::ip after Bcj2Dec_Init()
*/
void Bcj2Dec_Init(CBcj2Dec *p);


/* Bcj2Dec_Decode():
   returns:
     SZ_OK
     SZ_ERROR_DATA : if data in 5 starting bytes of BCJ2_STREAM_RC stream are not correct
*/
SRes Bcj2Dec_Decode(CBcj2Dec *p);

/* To check that decoding was finished you can compare
   sizes of processed streams with sizes known from another sources.
   You must do at least one mandatory check from the two following options:
      - the check for size of processed output (ORIG) stream.
      - the check for size of processed input  (MAIN) stream.
   additional optional checks:
      - the checks for processed sizes of all input streams (MAIN, CALL, JUMP, RC)
      - the checks Bcj2Dec_IsMaybeFinished*()
   also before actual decoding you can check that the
   following condition is met for stream sizes:
     ( size(ORIG) == size(MAIN) + size(CALL) + size(JUMP) )
*/

/* (state == BCJ2_STREAM_MAIN) means that decoder is ready for
      additional input data in BCJ2_STREAM_MAIN stream.
   Note that (state == BCJ2_STREAM_MAIN) is allowed for non-finished decoding.
*/
#define Bcj2Dec_IsMaybeFinished_state_MAIN(_p_) ((_p_)->state == BCJ2_STREAM_MAIN)

/* if the stream decoding was finished correctly, then range decoder
   part of CBcj2Dec also was finished, and then (CBcj2Dec::code == 0).
   Note that (CBcj2Dec::code == 0) is allowed for non-finished decoding.
*/
#define Bcj2Dec_IsMaybeFinished_code(_p_) ((_p_)->code == 0)

/* use Bcj2Dec_IsMaybeFinished() only as additional check
    after at least one mandatory check from the two following options:
      - the check for size of processed output (ORIG) stream.
      - the check for size of processed input  (MAIN) stream.
*/
#define Bcj2Dec_IsMaybeFinished(_p_) ( \
        Bcj2Dec_IsMaybeFinished_state_MAIN(_p_) && \
        Bcj2Dec_IsMaybeFinished_code(_p_))



/* ---------- BCJ2 Encoder ---------- */

typedef enum
{
  BCJ2_ENC_FINISH_MODE_CONTINUE,
  BCJ2_ENC_FINISH_MODE_END_BLOCK,
  BCJ2_ENC_FINISH_MODE_END_STREAM
} EBcj2Enc_FinishMode;

/*
  BCJ2_ENC_FINISH_MODE_CONTINUE:
     process non finished encoding.
     It notifies the encoder that additional further calls
     can provide more input data (src) than provided by current call.
     In  that case the CBcj2Enc encoder still can move (src) pointer
     up to (srcLim), but CBcj2Enc encoder can store some of the last
     processed bytes (up to 4 bytes) from src to internal CBcj2Enc::temp[] buffer.
   at return:
       (CBcj2Enc::src will point to position that includes
       processed data and data copied to (temp[]) buffer)
       That data from (temp[]) buffer will be used in further calls.
  
  BCJ2_ENC_FINISH_MODE_END_BLOCK:
     finish encoding of current block (ended at srcLim) without RC flushing.
   at return: if (CBcj2Enc::state == BCJ2_ENC_STATE_ORIG) &&
                  CBcj2Enc::src == CBcj2Enc::srcLim)
        :  it shows that block encoding was finished. And the encoder is
           ready for new (src) data or for stream finish operation.
     finished block means
     {
       CBcj2Enc has completed block encoding up to (srcLim).
       (1 + 4 bytes) or (2 + 4 bytes) CALL/JUMP cortages will
       not cross block boundary at (srcLim).
       temporary CBcj2Enc buffer for (ORIG) src data is empty.
       3 output uncompressed streams (MAIN, CALL, JUMP) were flushed.
       RC stream was not flushed. And RC stream will cross block boundary.
     }
     Note: some possible implementation of BCJ2 encoder could
     write branch marker (e8/e8/0f8x) in one call of Bcj2Enc_Encode(),
     and it could calculate symbol for RC in another call of Bcj2Enc_Encode().
     BCJ2 encoder uses ip/fileIp/fileSize/relatLimit values to calculate RC symbol.
     And these CBcj2Enc variables can have different values in different Bcj2Enc_Encode() calls.
     So caller must finish each block with BCJ2_ENC_FINISH_MODE_END_BLOCK
     to ensure that RC symbol is calculated and written in proper block.
    
  BCJ2_ENC_FINISH_MODE_END_STREAM
     finish encoding of stream (ended at srcLim) fully including RC flushing.
   at return: if (CBcj2Enc::state == BCJ2_ENC_STATE_FINISHED)
        : it shows that stream encoding was finished fully,
          and all output streams were flushed fully.
     also Bcj2Enc_IsFinished() can be called.
*/


/*
  32-bit relative offset in JUMP/CALL commands is
    - (mod 4 GiB)  for 32-bit x86 code
    - signed Int32 for 64-bit x86-64 code
  BCJ2 encoder also does internal relative to absolute address conversions.
  And there are 2 possible ways to do it:
    before v23: we used 32-bit variables and (mod 4 GiB) conversion
    since  v23: we use  64-bit variables and (signed Int32 offset) conversion.
  The absolute address condition for conversion in v23:
    ((UInt64)((Int64)ip64 - (Int64)fileIp64 + 5 + (Int32)offset) < (UInt64)fileSize64)
  note that if (fileSize64 > 2 GiB). there is difference between
  old (mod 4 GiB) way (v22) and new (signed Int32 offset) way (v23).
  And new (v23) way is more suitable to encode 64-bit x86-64 code for (fileSize64 > 2 GiB) cases.
*/

/*
// for old (v22) way for conversion:
typedef UInt32 CBcj2Enc_ip_unsigned;
typedef  Int32 CBcj2Enc_ip_signed;
#define BCJ2_ENC_FileSize_MAX ((UInt32)1 << 31)
*/
typedef UInt64 CBcj2Enc_ip_unsigned;
typedef  Int64 CBcj2Enc_ip_signed;

/* maximum size of file that can be used for conversion condition */
#define BCJ2_ENC_FileSize_MAX             ((CBcj2Enc_ip_unsigned)0 - 2)

/* default value of fileSize64_minus1 variable that means
   that absolute address limitation will not be used */
#define BCJ2_ENC_FileSizeField_UNLIMITED  ((CBcj2Enc_ip_unsigned)0 - 1)

/* calculate value that later can be set to CBcj2Enc::fileSize64_minus1 */
#define BCJ2_ENC_GET_FileSizeField_VAL_FROM_FileSize(fileSize) \
    ((CBcj2Enc_ip_unsigned)(fileSize) - 1)

/* set CBcj2Enc::fileSize64_minus1 variable from size of file */
#define Bcj2Enc_SET_FileSize(p, fileSize) \
    (p)->fileSize64_minus1 = BCJ2_ENC_GET_FileSizeField_VAL_FROM_FileSize(fileSize);


typedef struct
{
  Byte *bufs[BCJ2_NUM_STREAMS];
  const Byte *lims[BCJ2_NUM_STREAMS];
  const Byte *src;
  const Byte *srcLim;

  unsigned state;
  EBcj2Enc_FinishMode finishMode;

  Byte context;
  Byte flushRem;
  Byte isFlushState;

  Byte cache;
  UInt32 range;
  UInt64 low;
  UInt64 cacheSize;
  
  // UInt32 context;  // for marker version, it can include marker flag.

  /* (ip64) and (fileIp64) correspond to virtual source stream position
     that doesn't include data in temp[] */
  CBcj2Enc_ip_unsigned ip64;         /* current (ip) position */
  CBcj2Enc_ip_unsigned fileIp64;     /* start (ip) position of current file */
  CBcj2Enc_ip_unsigned fileSize64_minus1;   /* size of current file (for conversion limitation) */
  UInt32 relatLimit;  /* (relatLimit <= ((UInt32)1 << 31)) : 0 means disable_conversion */
  // UInt32 relatExcludeBits;

  UInt32 tempTarget;
  unsigned tempPos; /* the number of bytes that were copied to temp[] buffer
                       (tempPos <= 4) outside of Bcj2Enc_Encode() */
  // Byte temp[4]; // for marker version
  Byte temp[8];
  CBcj2Prob probs[2 + 256];
} CBcj2Enc;

void Bcj2Enc_Init(CBcj2Enc *p);


/*
Bcj2Enc_Encode(): at exit:
  p->State <  BCJ2_NUM_STREAMS    : we need more buffer space for output stream
                                    (bufs[p->State] == lims[p->State])
  p->State == BCJ2_ENC_STATE_ORIG : we need more data in input src stream
                                    (src == srcLim)
  p->State == BCJ2_ENC_STATE_FINISHED : after fully encoded stream
*/
void Bcj2Enc_Encode(CBcj2Enc *p);

/* Bcj2Enc encoder can look ahead for up 4 bytes of source stream.
   CBcj2Enc::tempPos : is the number of bytes that were copied from input stream to temp[] buffer.
   (CBcj2Enc::src) after Bcj2Enc_Encode() is starting position after
   fully processed data and after data copied to temp buffer.
   So if the caller needs to get real number of fully processed input
   bytes (without look ahead data in temp buffer),
   the caller must subtruct (CBcj2Enc::tempPos) value from processed size
   value that is calculated based on current (CBcj2Enc::src):
     cur_processed_pos = Calc_Big_Processed_Pos(enc.src)) -
        Bcj2Enc_Get_AvailInputSize_in_Temp(&enc);
*/
/* get the size of input data that was stored in temp[] buffer: */
#define Bcj2Enc_Get_AvailInputSize_in_Temp(p) ((p)->tempPos)

#define Bcj2Enc_IsFinished(p) ((p)->flushRem == 0)

/* Note : the decoder supports overlapping of marker (0f 80).
   But we can eliminate such overlapping cases by setting
   the limit for relative offset conversion as
     CBcj2Enc::relatLimit <= (0x0f << 24) == (240 MiB)
*/
/* default value for CBcj2Enc::relatLimit */
#define BCJ2_ENC_RELAT_LIMIT_DEFAULT  ((UInt32)0x0f << 24)
#define BCJ2_ENC_RELAT_LIMIT_MAX      ((UInt32)1 << 31)
// #define BCJ2_RELAT_EXCLUDE_NUM_BITS 5

EXTERN_C_END

#endif
