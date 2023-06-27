/****************************************************************************
 *
 * fthash.h
 *
 *   Hashing functions (specification).
 *
 */

/*
 * Copyright 2000 Computing Research Labs, New Mexico State University
 * Copyright 2001-2015
 *   Francesco Zappa Nardelli
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COMPUTING RESEARCH LAB OR NEW MEXICO STATE UNIVERSITY BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
 * THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

  /**************************************************************************
   *
   * This file is based on code from bdf.c,v 1.22 2000/03/16 20:08:50
   *
   * taken from Mark Leisher's xmbdfed package
   *
   */


#ifndef FTHASH_H_
#define FTHASH_H_


#include <freetype/freetype.h>


FT_BEGIN_HEADER


  typedef union  FT_Hashkey_
  {
    FT_Int       num;
    const char*  str;

  } FT_Hashkey;


  typedef struct  FT_HashnodeRec_
  {
    FT_Hashkey  key;
    size_t      data;

  } FT_HashnodeRec;

  typedef struct FT_HashnodeRec_  *FT_Hashnode;


  typedef FT_ULong
  (*FT_Hash_LookupFunc)( FT_Hashkey*  key );

  typedef FT_Bool
  (*FT_Hash_CompareFunc)( FT_Hashkey*  a,
                          FT_Hashkey*  b );


  typedef struct  FT_HashRec_
  {
    FT_UInt  limit;
    FT_UInt  size;
    FT_UInt  used;

    FT_Hash_LookupFunc   lookup;
    FT_Hash_CompareFunc  compare;

    FT_Hashnode*  table;

  } FT_HashRec;

  typedef struct FT_HashRec_  *FT_Hash;


  FT_Error
  ft_hash_str_init( FT_Hash    hash,
                    FT_Memory  memory );

  FT_Error
  ft_hash_num_init( FT_Hash    hash,
                    FT_Memory  memory );

  void
  ft_hash_str_free( FT_Hash    hash,
                    FT_Memory  memory );

#define ft_hash_num_free  ft_hash_str_free

  FT_Error
  ft_hash_str_insert( const char*  key,
                      size_t       data,
                      FT_Hash      hash,
                      FT_Memory    memory );

  FT_Error
  ft_hash_num_insert( FT_Int     num,
                      size_t     data,
                      FT_Hash    hash,
                      FT_Memory  memory );

  size_t*
  ft_hash_str_lookup( const char*  key,
                      FT_Hash      hash );

  size_t*
  ft_hash_num_lookup( FT_Int   num,
                      FT_Hash  hash );


FT_END_HEADER


#endif /* FTHASH_H_ */


/* END */
