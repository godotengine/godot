/* Copyright (C) 2006 David Rowe */
/**
   @file quant_lsp_bfin.h
   @author David Rowe
   @brief Various compatibility routines for Speex (Blackfin version)
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define OVERRIDE_LSP_QUANT
#ifdef OVERRIDE_LSP_QUANT

/*
  Note http://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html
  well tell you all the magic resgister constraints used below
  for gcc in-line asm.
*/

static int lsp_quant(
  spx_word16_t      *x, 
  const signed char *cdbk, 
  int                nbVec, 
  int                nbDim
)
{
   int          j;
   spx_word32_t best_dist=1<<30;
   int          best_id=0;

   __asm__ __volatile__
     (
"	%0 = 1 (X);\n\t"                       /* %0: best_dist */    
"	%0 <<= 30;\n\t"     
"	%1 = 0 (X);\n\t"                       /* %1: best_i         */
"       P2 = %3\n\t"                           /* P2: ptr to cdbk    */
"       R5 = 0;\n\t"                           /* R5: best cb entry  */

"       R0 = %5;\n\t"                          /* set up circ addr   */
"       R0 <<= 1;\n\t"
"       L0 = R0;\n\t"                          
"       I0 = %2;\n\t"                          /* %2: &x[0]          */
"       B0 = %2;\n\t"                          

"       R2.L = W [I0++];\n\t"
"	LSETUP (1f, 2f) LC0 = %4;\n\t"
"1:	  R3 = 0;\n\t"                         /* R3: dist           */
"	  LSETUP (3f, 4f) LC1 = %5;\n\t"
"3:       R1 = B [P2++] (X);\n\t"            
"	    R1 <<= 5;\n\t"
"	    R0.L = R2.L - R1.L || R2.L = W [I0++];\n\t"
"	    R0 = R0.L*R0.L;\n\t"
"4:	    R3 = R3 + R0;\n\t"

"	  cc =R3<%0;\n\t"
"	  if cc %0=R3;\n\t"
"	  if cc %1=R5;\n\t"
"2:     R5 += 1;\n\t"
"         L0 = 0;\n\t"
   : "=&d" (best_dist), "=&d" (best_id)
   : "a" (x), "b" (cdbk), "a" (nbVec), "a" (nbDim)
   : "I0", "P2", "R0", "R1", "R2", "R3", "R5", "L0", "B0", "A0"
   );

   for (j=0;j<nbDim;j++) {
      x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));
   }
   return best_id;
}
#endif

#define OVERRIDE_LSP_WEIGHT_QUANT
#ifdef OVERRIDE_LSP_WEIGHT_QUANT

/*
  Note http://gcc.gnu.org/onlinedocs/gcc/Machine-Constraints.html
  well tell you all the magic resgister constraints used below
  for gcc in-line asm.
*/

static int lsp_weight_quant(
  spx_word16_t      *x, 
  spx_word16_t      *weight, 
  const signed char *cdbk, 
  int                nbVec, 
  int                nbDim
)
{
   int          j;
   spx_word32_t best_dist=1<<30;
   int          best_id=0;

   __asm__ __volatile__
     (
"	%0 = 1 (X);\n\t"                       /* %0: best_dist */    
"	%0 <<= 30;\n\t"     
"	%1 = 0 (X);\n\t"                       /* %1: best_i         */
"       P2 = %4\n\t"                           /* P2: ptr to cdbk    */
"       R5 = 0;\n\t"                           /* R5: best cb entry  */

"       R0 = %6;\n\t"                          /* set up circ addr   */
"       R0 <<= 1;\n\t"
"       L0 = R0;\n\t"                          
"       L1 = R0;\n\t"
"       I0 = %2;\n\t"                          /* %2: &x[0]          */
"	I1 = %3;\n\t"                          /* %3: &weight[0]     */
"       B0 = %2;\n\t"                          
"	B1 = %3;\n\t"                          

"	LSETUP (1f, 2f) LC0 = %5;\n\t"
"1:	  R3 = 0 (X);\n\t"                     /* R3: dist           */
"	  LSETUP (3f, 4f) LC1 = %6;\n\t"
"3:	    R0.L = W [I0++] || R2.L = W [I1++];\n\t"
"           R1 = B [P2++] (X);\n\t"            
"	    R1 <<= 5;\n\t"
"	    R0.L = R0.L - R1.L;\n\t"
"           R0 = R0.L*R0.L;\n\t"
"	    A1 = R2.L*R0.L (M,IS);\n\t"
"	    A1 = A1 >>> 16;\n\t"
"	    R1 = (A1 += R2.L*R0.H) (IS);\n\t"
"4:	    R3 = R3 + R1;\n\t"

"	  cc =R3<%0;\n\t"
"	  if cc %0=R3;\n\t"
"	  if cc %1=R5;\n\t"
"2:    R5 += 1;\n\t"
"         L0 = 0;\n\t"
"         L1 = 0;\n\t"
   : "=&d" (best_dist), "=&d" (best_id)
   : "a" (x), "a" (weight), "b" (cdbk), "a" (nbVec), "a" (nbDim)
   : "I0", "I1", "P2", "R0", "R1", "R2", "R3", "R5", "A1",
     "L0", "L1", "B0", "B1"
   );

   for (j=0;j<nbDim;j++) {
      x[j] = SUB16(x[j],SHL16((spx_word16_t)cdbk[best_id*nbDim+j],5));
   }
   return best_id;
}
#endif
