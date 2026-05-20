/* Delta.c -- Delta converter
2021-02-09 : Igor Pavlov : Public domain */

#include "Precomp.h"

#include "Delta.h"

void Delta_Init(Byte *state)
{
  unsigned i;
  for (i = 0; i < DELTA_STATE_SIZE; i++)
    state[i] = 0;
}


void Delta_Encode(Byte *state, unsigned delta, Byte *data, SizeT size)
{
  Byte temp[DELTA_STATE_SIZE];

  if (size == 0)
    return;

  {
    unsigned i = 0;
    do
      temp[i] = state[i];
    while (++i != delta);
  }

  if (size <= delta)
  {
    unsigned i = 0, k;
    do
    {
      Byte b = *data;
      *data++ = (Byte)(b - temp[i]);
      temp[i] = b;
    }
    while (++i != size);
    
    k = 0;
    
    do
    {
      if (i == delta)
        i = 0;
      state[k] = temp[i++];
    }
    while (++k != delta);
    
    return;
  }
    
  {
    Byte *p = data + size - delta;
    {
      unsigned i = 0;
      do
        state[i] = *p++;
      while (++i != delta);
    }
    {
      const Byte *lim = data + delta;
      ptrdiff_t dif = -(ptrdiff_t)delta;
      
      if (((ptrdiff_t)size + dif) & 1)
      {
        --p;  *p = (Byte)(*p - p[dif]);
      }

      while (p != lim)
      {
        --p;  *p = (Byte)(*p - p[dif]);
        --p;  *p = (Byte)(*p - p[dif]);
      }
      
      dif = -dif;
      
      do
      {
        --p;  *p = (Byte)(*p - temp[--dif]);
      }
      while (dif != 0);
    }
  }
}


void Delta_Decode(Byte *state, unsigned delta, Byte *data, SizeT size)
{
  unsigned i;
  const Byte *lim;

  if (size == 0)
    return;
  
  i = 0;
  lim = data + size;
  
  if (size <= delta)
  {
    do
      *data = (Byte)(*data + state[i++]);
    while (++data != lim);

    for (; delta != i; state++, delta--)
      *state = state[i];
    data -= i;
  }
  else
  {
    /*
    #define B(n) b ## n
    #define I(n) Byte B(n) = state[n];
    #define U(n) { B(n) = (Byte)((B(n)) + *data++); data[-1] = (B(n)); }
    #define F(n) if (data != lim) { U(n) }

    if (delta == 1)
    {
      I(0)
      if ((lim - data) & 1) { U(0) }
      while (data != lim) { U(0) U(0) }
      data -= 1;
    }
    else if (delta == 2)
    {
      I(0) I(1)
      lim -= 1; while (data < lim) { U(0) U(1) }
      lim += 1; F(0)
      data -= 2;
    }
    else if (delta == 3)
    {
      I(0) I(1) I(2)
      lim -= 2; while (data < lim) { U(0) U(1) U(2) }
      lim += 2; F(0) F(1)
      data -= 3;
    }
    else if (delta == 4)
    {
      I(0) I(1) I(2) I(3)
      lim -= 3; while (data < lim) { U(0) U(1) U(2) U(3) }
      lim += 3; F(0) F(1) F(2)
      data -= 4;
    }
    else
    */
    {
      do
      {
        *data = (Byte)(*data + state[i++]);
        data++;
      }
      while (i != delta);
  
      {
        ptrdiff_t dif = -(ptrdiff_t)delta;
        do
          *data = (Byte)(*data + data[dif]);
        while (++data != lim);
        data += dif;
      }
    }
  }

  do
    *state++ = *data;
  while (++data != lim);
}
