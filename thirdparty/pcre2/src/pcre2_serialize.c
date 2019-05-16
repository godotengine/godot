/*************************************************
*      Perl-Compatible Regular Expressions       *
*************************************************/

/* PCRE is a library of functions to support regular expressions whose syntax
and semantics are as close as possible to those of the Perl 5 language.

                       Written by Philip Hazel
     Original API code Copyright (c) 1997-2012 University of Cambridge
          New API code Copyright (c) 2016-2018 University of Cambridge

-----------------------------------------------------------------------------
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of the University of Cambridge nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
-----------------------------------------------------------------------------
*/

/* This module contains functions for serializing and deserializing
a sequence of compiled codes. */


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif


#include "pcre2_internal.h"

/* Magic number to provide a small check against being handed junk. */

#define SERIALIZED_DATA_MAGIC 0x50523253u

/* Deserialization is limited to the current PCRE version and
character width. */

#define SERIALIZED_DATA_VERSION \
  ((PCRE2_MAJOR) | ((PCRE2_MINOR) << 16))

#define SERIALIZED_DATA_CONFIG \
  (sizeof(PCRE2_UCHAR) | ((sizeof(void*)) << 8) | ((sizeof(PCRE2_SIZE)) << 16))



/*************************************************
*           Serialize compiled patterns          *
*************************************************/

PCRE2_EXP_DEFN int32_t PCRE2_CALL_CONVENTION
pcre2_serialize_encode(const pcre2_code **codes, int32_t number_of_codes,
   uint8_t **serialized_bytes, PCRE2_SIZE *serialized_size,
   pcre2_general_context *gcontext)
{
uint8_t *bytes;
uint8_t *dst_bytes;
int32_t i;
PCRE2_SIZE total_size;
const pcre2_real_code *re;
const uint8_t *tables;
pcre2_serialized_data *data;

const pcre2_memctl *memctl = (gcontext != NULL) ?
  &gcontext->memctl : &PRIV(default_compile_context).memctl;

if (codes == NULL || serialized_bytes == NULL || serialized_size == NULL)
  return PCRE2_ERROR_NULL;

if (number_of_codes <= 0) return PCRE2_ERROR_BADDATA;

/* Compute total size. */
total_size = sizeof(pcre2_serialized_data) + tables_length;
tables = NULL;

for (i = 0; i < number_of_codes; i++)
  {
  if (codes[i] == NULL) return PCRE2_ERROR_NULL;
  re = (const pcre2_real_code *)(codes[i]);
  if (re->magic_number != MAGIC_NUMBER) return PCRE2_ERROR_BADMAGIC;
  if (tables == NULL)
    tables = re->tables;
  else if (tables != re->tables)
    return PCRE2_ERROR_MIXEDTABLES;
  total_size += re->blocksize;
  }

/* Initialize the byte stream. */
bytes = memctl->malloc(total_size + sizeof(pcre2_memctl), memctl->memory_data);
if (bytes == NULL) return PCRE2_ERROR_NOMEMORY;

/* The controller is stored as a hidden parameter. */
memcpy(bytes, memctl, sizeof(pcre2_memctl));
bytes += sizeof(pcre2_memctl);

data = (pcre2_serialized_data *)bytes;
data->magic = SERIALIZED_DATA_MAGIC;
data->version = SERIALIZED_DATA_VERSION;
data->config = SERIALIZED_DATA_CONFIG;
data->number_of_codes = number_of_codes;

/* Copy all compiled code data. */
dst_bytes = bytes + sizeof(pcre2_serialized_data);
memcpy(dst_bytes, tables, tables_length);
dst_bytes += tables_length;

for (i = 0; i < number_of_codes; i++)
  {
  re = (const pcre2_real_code *)(codes[i]);
  (void)memcpy(dst_bytes, (char *)re, re->blocksize);
  
  /* Certain fields in the compiled code block are re-set during 
  deserialization. In order to ensure that the serialized data stream is always 
  the same for the same pattern, set them to zero here. We can't assume the 
  copy of the pattern is correctly aligned for accessing the fields as part of 
  a structure. Note the use of sizeof(void *) in the second of these, to
  specify the size of a pointer. If sizeof(uint8_t *) is used (tables is a 
  pointer to uint8_t), gcc gives a warning because the first argument is also a 
  pointer to uint8_t. Casting the first argument to (void *) can stop this, but 
  it didn't stop Coverity giving the same complaint. */
  
  (void)memset(dst_bytes + offsetof(pcre2_real_code, memctl), 0, 
    sizeof(pcre2_memctl));
  (void)memset(dst_bytes + offsetof(pcre2_real_code, tables), 0, 
    sizeof(void *));
  (void)memset(dst_bytes + offsetof(pcre2_real_code, executable_jit), 0,
    sizeof(void *));        
 
  dst_bytes += re->blocksize;
  }

*serialized_bytes = bytes;
*serialized_size = total_size;
return number_of_codes;
}


/*************************************************
*          Deserialize compiled patterns         *
*************************************************/

PCRE2_EXP_DEFN int32_t PCRE2_CALL_CONVENTION
pcre2_serialize_decode(pcre2_code **codes, int32_t number_of_codes,
   const uint8_t *bytes, pcre2_general_context *gcontext)
{
const pcre2_serialized_data *data = (const pcre2_serialized_data *)bytes;
const pcre2_memctl *memctl = (gcontext != NULL) ?
  &gcontext->memctl : &PRIV(default_compile_context).memctl;

const uint8_t *src_bytes;
pcre2_real_code *dst_re;
uint8_t *tables;
int32_t i, j;

/* Sanity checks. */

if (data == NULL || codes == NULL) return PCRE2_ERROR_NULL;
if (number_of_codes <= 0) return PCRE2_ERROR_BADDATA;
if (data->number_of_codes <= 0) return PCRE2_ERROR_BADSERIALIZEDDATA;
if (data->magic != SERIALIZED_DATA_MAGIC) return PCRE2_ERROR_BADMAGIC;
if (data->version != SERIALIZED_DATA_VERSION) return PCRE2_ERROR_BADMODE;
if (data->config != SERIALIZED_DATA_CONFIG) return PCRE2_ERROR_BADMODE;

if (number_of_codes > data->number_of_codes)
  number_of_codes = data->number_of_codes;

src_bytes = bytes + sizeof(pcre2_serialized_data);

/* Decode tables. The reference count for the tables is stored immediately
following them. */

tables = memctl->malloc(tables_length + sizeof(PCRE2_SIZE), memctl->memory_data);
if (tables == NULL) return PCRE2_ERROR_NOMEMORY;

memcpy(tables, src_bytes, tables_length);
*(PCRE2_SIZE *)(tables + tables_length) = number_of_codes;
src_bytes += tables_length;

/* Decode the byte stream. We must not try to read the size from the compiled
code block in the stream, because it might be unaligned, which causes errors on
hardware such as Sparc-64 that doesn't like unaligned memory accesses. The type
of the blocksize field is given its own name to ensure that it is the same here
as in the block. */

for (i = 0; i < number_of_codes; i++)
  {
  CODE_BLOCKSIZE_TYPE blocksize;
  memcpy(&blocksize, src_bytes + offsetof(pcre2_real_code, blocksize),
    sizeof(CODE_BLOCKSIZE_TYPE));
  if (blocksize <= sizeof(pcre2_real_code))
    return PCRE2_ERROR_BADSERIALIZEDDATA;

  /* The allocator provided by gcontext replaces the original one. */

  dst_re = (pcre2_real_code *)PRIV(memctl_malloc)(blocksize,
    (pcre2_memctl *)gcontext);
  if (dst_re == NULL)
    {
    memctl->free(tables, memctl->memory_data);
    for (j = 0; j < i; j++)
      {
      memctl->free(codes[j], memctl->memory_data);
      codes[j] = NULL;
      }
    return PCRE2_ERROR_NOMEMORY;
    }

  /* The new allocator must be preserved. */

  memcpy(((uint8_t *)dst_re) + sizeof(pcre2_memctl),
    src_bytes + sizeof(pcre2_memctl), blocksize - sizeof(pcre2_memctl));
  if (dst_re->magic_number != MAGIC_NUMBER ||
      dst_re->name_entry_size > MAX_NAME_SIZE + IMM2_SIZE + 1 ||
      dst_re->name_count > MAX_NAME_COUNT)
    {   
    memctl->free(dst_re, memctl->memory_data); 
    return PCRE2_ERROR_BADSERIALIZEDDATA;
    } 

  /* At the moment only one table is supported. */

  dst_re->tables = tables;
  dst_re->executable_jit = NULL;
  dst_re->flags |= PCRE2_DEREF_TABLES;

  codes[i] = dst_re;
  src_bytes += blocksize;
  }

return number_of_codes;
}


/*************************************************
*    Get the number of serialized patterns       *
*************************************************/

PCRE2_EXP_DEFN int32_t PCRE2_CALL_CONVENTION
pcre2_serialize_get_number_of_codes(const uint8_t *bytes)
{
const pcre2_serialized_data *data = (const pcre2_serialized_data *)bytes;

if (data == NULL) return PCRE2_ERROR_NULL;
if (data->magic != SERIALIZED_DATA_MAGIC) return PCRE2_ERROR_BADMAGIC;
if (data->version != SERIALIZED_DATA_VERSION) return PCRE2_ERROR_BADMODE;
if (data->config != SERIALIZED_DATA_CONFIG) return PCRE2_ERROR_BADMODE;

return data->number_of_codes;
}


/*************************************************
*            Free the allocated stream           *
*************************************************/

PCRE2_EXP_DEFN void PCRE2_CALL_CONVENTION
pcre2_serialize_free(uint8_t *bytes)
{
if (bytes != NULL)
  {
  pcre2_memctl *memctl = (pcre2_memctl *)(bytes - sizeof(pcre2_memctl));
  memctl->free(memctl, memctl->memory_data);
  }
}

/* End of pcre2_serialize.c */
