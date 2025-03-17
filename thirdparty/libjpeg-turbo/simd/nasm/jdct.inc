;
; jdct.inc - private declarations for forward & reverse DCT subsystems
;
; Copyright 2009 Pierre Ossman <ossman@cendio.se> for Cendio AB
; Copyright (C) 2018, D. R. Commander.
;
; Based on the x86 SIMD extension for IJG JPEG library
; Copyright (C) 1999-2006, MIYASAKA Masaru.
; For conditions of distribution and use, see copyright notice in jsimdext.inc

; Each IDCT routine is responsible for range-limiting its results and
; converting them to unsigned form (0..MAXJSAMPLE).  The raw outputs could
; be quite far out of range if the input data is corrupt, so a bulletproof
; range-limiting step is required.  We use a mask-and-table-lookup method
; to do the combined operations quickly.
;
%define RANGE_MASK  (MAXJSAMPLE * 4 + 3)  ; 2 bits wider than legal samples

%define ROW(n, b, s)  ((b) + (n) * (s))
%define COL(n, b, s)  ((b) + (n) * (s) * DCTSIZE)

%define DWBLOCK(m, n, b, s) \
  ((b) + (m) * DCTSIZE * (s) + (n) * SIZEOF_DWORD)
%define MMBLOCK(m, n, b, s) \
  ((b) + (m) * DCTSIZE * (s) + (n) * SIZEOF_MMWORD)
%define XMMBLOCK(m, n, b, s) \
  ((b) + (m) * DCTSIZE * (s) + (n) * SIZEOF_XMMWORD)
%define YMMBLOCK(m, n, b, s) \
  ((b) + (m) * DCTSIZE * (s) + (n) * SIZEOF_YMMWORD)

; --------------------------------------------------------------------------
