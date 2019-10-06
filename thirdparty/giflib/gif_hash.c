/*****************************************************************************

gif_hash.c -- module to support the following operations:

1. InitHashTable - initialize hash table.
2. ClearHashTable - clear the hash table to an empty state.
2. InsertHashTable - insert one item into data structure.
3. ExistsHashTable - test if item exists in data structure.

This module is used to hash the GIF codes during encoding.

SPDX-License-Identifier: MIT

*****************************************************************************/

#include <stdint.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

#include "gif_lib.h"
#include "gif_hash.h"
#include "gif_lib_private.h"

/* #define  DEBUG_HIT_RATE    Debug number of misses per hash Insert/Exists. */

#ifdef	DEBUG_HIT_RATE
static long NumberOfTests = 0,
	    NumberOfMisses = 0;
#endif	/* DEBUG_HIT_RATE */

static int KeyItem(uint32_t Item);

/******************************************************************************
 Initialize HashTable - allocate the memory needed and clear it.	      *
******************************************************************************/
GifHashTableType *_InitHashTable(void)
{
    GifHashTableType *HashTable;

    if ((HashTable = (GifHashTableType *) malloc(sizeof(GifHashTableType)))
	== NULL)
	return NULL;

    _ClearHashTable(HashTable);

    return HashTable;
}

/******************************************************************************
 Routine to clear the HashTable to an empty state.			      *
 This part is a little machine depended. Use the commented part otherwise.   *
******************************************************************************/
void _ClearHashTable(GifHashTableType *HashTable)
{
    memset(HashTable -> HTable, 0xFF, HT_SIZE * sizeof(uint32_t));
}

/******************************************************************************
 Routine to insert a new Item into the HashTable. The data is assumed to be  *
 new one.								      *
******************************************************************************/
void _InsertHashTable(GifHashTableType *HashTable, uint32_t Key, int Code)
{
    int HKey = KeyItem(Key);
    uint32_t *HTable = HashTable -> HTable;

#ifdef DEBUG_HIT_RATE
	NumberOfTests++;
	NumberOfMisses++;
#endif /* DEBUG_HIT_RATE */

    while (HT_GET_KEY(HTable[HKey]) != 0xFFFFFL) {
#ifdef DEBUG_HIT_RATE
	    NumberOfMisses++;
#endif /* DEBUG_HIT_RATE */
	HKey = (HKey + 1) & HT_KEY_MASK;
    }
    HTable[HKey] = HT_PUT_KEY(Key) | HT_PUT_CODE(Code);
}

/******************************************************************************
 Routine to test if given Key exists in HashTable and if so returns its code *
 Returns the Code if key was found, -1 if not.				      *
******************************************************************************/
int _ExistsHashTable(GifHashTableType *HashTable, uint32_t Key)
{
    int HKey = KeyItem(Key);
    uint32_t *HTable = HashTable -> HTable, HTKey;

#ifdef DEBUG_HIT_RATE
	NumberOfTests++;
	NumberOfMisses++;
#endif /* DEBUG_HIT_RATE */

    while ((HTKey = HT_GET_KEY(HTable[HKey])) != 0xFFFFFL) {
#ifdef DEBUG_HIT_RATE
	    NumberOfMisses++;
#endif /* DEBUG_HIT_RATE */
	if (Key == HTKey) return HT_GET_CODE(HTable[HKey]);
	HKey = (HKey + 1) & HT_KEY_MASK;
    }

    return -1;
}

/******************************************************************************
 Routine to generate an HKey for the hashtable out of the given unique key.  *
 The given Key is assumed to be 20 bits as follows: lower 8 bits are the     *
 new postfix character, while the upper 12 bits are the prefix code.	      *
 Because the average hit ratio is only 2 (2 hash references per entry),      *
 evaluating more complex keys (such as twin prime keys) does not worth it!   *
******************************************************************************/
static int KeyItem(uint32_t Item)
{
    return ((Item >> 12) ^ Item) & HT_KEY_MASK;
}

#ifdef	DEBUG_HIT_RATE
/******************************************************************************
 Debugging routine to print the hit ratio - number of times the hash table   *
 was tested per operation. This routine was used to test the KeyItem routine *
******************************************************************************/
void HashTablePrintHitRatio(void)
{
    printf("Hash Table Hit Ratio is %ld/%ld = %ld%%.\n",
	NumberOfMisses, NumberOfTests,
	NumberOfMisses * 100 / NumberOfTests);
}
#endif	/* DEBUG_HIT_RATE */

/* end */
