/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file hashlist.c
 * @~English
 *
 * @brief Functions for creating and using a hash list of key-value
 *        pairs.
 *
 * @author Mark Callow, HI Corporation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// This is to avoid compile warnings. strlen is defined as returning
// size_t and is used by the uthash macros. This avoids having to
// make changes to uthash and a bunch of casts in this file. The
// casts would be required because the key and value lengths in KTX
// are specified as 4 byte quantities so we can't change _keyAndValue
// below to use size_t.
#define strlen(x) ((unsigned int)strlen(x))

#include "uthash.h"

#include "ktx.h"
#include "ktxint.h"


/**
 * @internal
 * @struct ktxKVListEntry
 * @brief Hash list entry structure
 */
typedef struct ktxKVListEntry {
    unsigned int keyLen;    /*!< Length of the key */
    char* key;              /*!< Pointer to key string */
    unsigned int valueLen;  /*!< Length of the value */
    void* value;            /*!< Pointer to the value */
    UT_hash_handle hh;      /*!< handle used by UT hash */
} ktxKVListEntry;


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Construct an empty hash list for storing key-value pairs.
 *
 * @param [in] pHead pointer to the location to write the list head.
 */
void
ktxHashList_Construct(ktxHashList* pHead)
{
    *pHead = NULL;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Construct a hash list by copying another.
 *
 * @param [in] pHead pointer to head of the list.
 * @param [in] orig  head of the original hash list.
 */
void
ktxHashList_ConstructCopy(ktxHashList* pHead, ktxHashList orig)
{
    ktxHashListEntry* entry = orig;
    *pHead = NULL;
    for (; entry != NULL; entry = ktxHashList_Next(entry)) {
        (void)ktxHashList_AddKVPair(pHead,
                                    entry->key, entry->valueLen, entry->value);
    }
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Destruct a hash list.
 *
 * All memory associated with the hash list's keys and values
 * is freed.
 *
 * @param [in] pHead pointer to the hash list to be destroyed.
 */
void
ktxHashList_Destruct(ktxHashList* pHead)
{
    ktxKVListEntry* kv;
    ktxKVListEntry* head = *pHead;

    for(kv = head; kv != NULL;) {
        ktxKVListEntry* tmp = (ktxKVListEntry*)kv->hh.next;
        HASH_DELETE(hh, head, kv);
        free(kv);
        kv = tmp;
    }
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Create an empty hash list for storing key-value pairs.
 *
 * @param [in,out] ppHl address of a variable in which to set a pointer to
 *                 the newly created hash list.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 * @exception KTX_OUT_OF_MEMORY if not enough memory.
 */
KTX_error_code
ktxHashList_Create(ktxHashList** ppHl)
{
    ktxHashList* hl = (ktxHashList*)malloc(sizeof (ktxKVListEntry*));
    if (hl == NULL)
        return KTX_OUT_OF_MEMORY;

    ktxHashList_Construct(hl);
    *ppHl = hl;
    return KTX_SUCCESS;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Create a copy of a hash list.
 *
 * @param [in,out] ppHl address of a variable in which to set a pointer to
 *                      the newly created hash list.
 * @param [in]     orig head of the ktxHashList to copy.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 * @exception KTX_OUT_OF_MEMORY if not enough memory.
 */
KTX_error_code
ktxHashList_CreateCopy(ktxHashList** ppHl, ktxHashList orig)
{
    ktxHashList* hl = (ktxHashList*)malloc(sizeof (ktxKVListEntry*));
    if (hl == NULL)
        return KTX_OUT_OF_MEMORY;

    ktxHashList_ConstructCopy(hl, orig);
    *ppHl = hl;
    return KTX_SUCCESS;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Destroy a hash list.
 *
 * All memory associated with the hash list's keys and values
 * is freed. The hash list is also freed.
 *
 * @param [in] pHead pointer to the hash list to be destroyed.
 */
void
ktxHashList_Destroy(ktxHashList* pHead)
{
    ktxHashList_Destruct(pHead);
    free(pHead);
}

#if !__clang__ && __GNUC__ // Grumble clang grumble
// These are in uthash.h macros. I don't want to change that file.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
#endif

/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Add a key value pair to a hash list.
 *
 * The value can be empty, i.e, its length can be 0.
 *
 * @param [in] pHead    pointer to the head of the target hash list.
 * @param [in] key      pointer to the UTF8 NUL-terminated string to be used as the key.
 * @param [in] valueLen the number of bytes of data in @p value.
 * @param [in] value    pointer to the bytes of data constituting the value.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 * @exception KTX_INVALID_VALUE if @p pHead, @p key or @p value are NULL, @p key is an
 *            empty string or @p valueLen == 0.
 */
KTX_error_code
ktxHashList_AddKVPair(ktxHashList* pHead, const char* key, unsigned int valueLen, const void* value)
{
    if (pHead && key && (valueLen == 0 || value)) {
        unsigned int keyLen = (unsigned int)strlen(key) + 1;
        ktxKVListEntry* kv;

        if (keyLen == 1)
            return KTX_INVALID_VALUE;   /* Empty string */

        /* Allocate all the memory as a block */
        kv = (ktxKVListEntry*)malloc(sizeof(ktxKVListEntry) + keyLen + valueLen);
        /* Put key first */
        kv->key = (char *)kv + sizeof(ktxKVListEntry);
        kv->keyLen = keyLen;
        memcpy(kv->key, key, keyLen);
        /* then value */
        kv->valueLen = valueLen;
        if (valueLen > 0) {
            kv->value = kv->key + keyLen;
            memcpy(kv->value, value, valueLen);
        } else {
            kv->value = 0;
        }

        HASH_ADD_KEYPTR( hh, *pHead, kv->key, kv->keyLen-1, kv);
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Delete a key value pair in a hash list.
 *
 * Is a nop if the key is not in the hash.
 *
 * @param [in] pHead    pointer to the head of the target hash list.
 * @param [in] key      pointer to the UTF8 NUL-terminated string to be used as the key.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 * @exception KTX_INVALID_VALUE if @p pHead is NULL or @p key is an empty
 *            string.
 */
KTX_error_code
ktxHashList_DeleteKVPair(ktxHashList* pHead, const char* key)
{
    if (pHead && key) {
        ktxKVListEntry* kv;

        HASH_FIND_STR( *pHead, key, kv );  /* kv: pointer to target entry. */
        if (kv != NULL)
            HASH_DEL(*pHead, kv);
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Delete an entry from a hash list.
 *
 * @param [in] pHead    pointer to the head of the target hash list.
 * @param [in] pEntry   pointer to the ktxHashListEntry to delete.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 * @exception KTX_INVALID_VALUE if @p pHead is NULL or @p key is an empty
 *            string.
 */
KTX_error_code
ktxHashList_DeleteEntry(ktxHashList* pHead, ktxHashListEntry* pEntry)
{
    if (pHead && pEntry) {
        HASH_DEL(*pHead, pEntry);
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Looks up a key in a hash list and returns the entry.
 *
 * @param [in]     pHead        pointer to the head of the target hash list.
 * @param [in]     key          pointer to a UTF8 NUL-terminated string to find.
 * @param [in,out] ppEntry      @p *ppEntry is set to the point at the
 *                              ktxHashListEntry.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p This, @p key or @p pValueLen or @p ppValue
 *                              is NULL.
 * @exception KTX_NOT_FOUND     an entry matching @p key was not found.
 */
KTX_error_code
ktxHashList_FindEntry(ktxHashList* pHead, const char* key,
                      ktxHashListEntry** ppEntry)
{
    if (pHead && key) {
        ktxKVListEntry* kv;

        HASH_FIND_STR( *pHead, key, kv );  /* kv: output pointer */

        if (kv) {
            *ppEntry = kv;
            return KTX_SUCCESS;
        } else
            return KTX_NOT_FOUND;
    } else
        return KTX_INVALID_VALUE;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Looks up a key in a hash list and returns the value.
 *
 * @param [in]     pHead        pointer to the head of the target hash list.
 * @param [in]     key          pointer to a UTF8 NUL-terminated string to find.
 * @param [in,out] pValueLen    @p *pValueLen is set to the number of bytes of
 *                              data in the returned value.
 * @param [in,out] ppValue      @p *ppValue is set to the point to the value for
 *                              @p key.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p This, @p key or @p pValueLen or @p ppValue
 *                              is NULL.
 * @exception KTX_NOT_FOUND     an entry matching @p key was not found.
 */
KTX_error_code
ktxHashList_FindValue(ktxHashList *pHead, const char* key, unsigned int* pValueLen, void** ppValue)
{
    if (pValueLen && ppValue) {
        ktxHashListEntry* pEntry;
        KTX_error_code result;

        result = ktxHashList_FindEntry(pHead, key, &pEntry);
        if (result == KTX_SUCCESS) {
            ktxHashListEntry_GetValue(pEntry, pValueLen, ppValue);
            return KTX_SUCCESS;
        } else
            return result;
    } else
        return KTX_INVALID_VALUE;
}

#if !__clang__ && __GNUC__
#pragma GCC diagnostic pop
#endif

/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Returns the next entry in a ktxHashList.
 *
 * Use for iterating through the list:
 * @code
 *    ktxHashListEntry* entry;
 *    for (entry = listHead; entry != NULL; entry = ktxHashList_Next(entry)) {
 *       ...
 *    };
 * @endcode
 *
 * Note
 *
 * @param [in]  entry   pointer to a hash list entry. Note that a ktxHashList*,
 *                      i.e. the list head, is also a pointer to an entry so
 *                      can be passed to this function.
 *
 * @return a pointer to the next entry or NULL.
 *
 */
ktxHashListEntry*
ktxHashList_Next(ktxHashListEntry* entry)
{
    if (entry) {
        return ((ktxKVListEntry*)entry)->hh.next;
    } else
        return NULL;
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Serialize a hash list to a block of data suitable for writing
 *        to a file.
 *
 * The caller is responsible for freeing the data block returned by this
 * function.
 *
 * @param [in]     pHead        pointer to the head of the target hash list.
 * @param [in,out] pKvdLen      @p *pKvdLen is set to the number of bytes of
 *                              data in the returned data block.
 * @param [in,out] ppKvd        @p *ppKvd is set to the point to the block of
 *                              memory containing the serialized data or
 *                              NULL. if the hash list is empty.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p This, @p pKvdLen or @p ppKvd is NULL.
 * @exception KTX_OUT_OF_MEMORY there was not enough memory to serialize the
 *                              data.
 */
KTX_error_code
ktxHashList_Serialize(ktxHashList* pHead,
                      unsigned int* pKvdLen, unsigned char** ppKvd)
{

    if (pHead && pKvdLen && ppKvd) {
        ktxKVListEntry* kv;
        unsigned int bytesOfKeyValueData = 0;
        unsigned int keyValueLen;
        unsigned char* sd;
        char padding[4] = {0, 0, 0, 0};

        for (kv = *pHead; kv != NULL; kv = kv->hh.next) {
            /* sizeof(sd) is to make space to write keyAndValueByteSize */
            keyValueLen = kv->keyLen + kv->valueLen + sizeof(ktx_uint32_t);
            /* Add valuePadding */
            keyValueLen = _KTX_PAD4(keyValueLen);
            bytesOfKeyValueData += keyValueLen;
        }

        if (bytesOfKeyValueData == 0) {
            *pKvdLen = 0;
            *ppKvd = NULL;
        } else {
            sd = malloc(bytesOfKeyValueData);
            if (!sd)
                return KTX_OUT_OF_MEMORY;

            *pKvdLen = bytesOfKeyValueData;
            *ppKvd = sd;

            for (kv = *pHead; kv != NULL; kv = kv->hh.next) {
                int padLen;

                keyValueLen = kv->keyLen + kv->valueLen;
                *(ktx_uint32_t*)sd = keyValueLen;
                sd += sizeof(ktx_uint32_t);
                memcpy(sd, kv->key, kv->keyLen);
                sd += kv->keyLen;
                if (kv->valueLen > 0)
                    memcpy(sd, kv->value, kv->valueLen);
                sd += kv->valueLen;
                padLen = _KTX_PAD4_LEN(keyValueLen);
                memcpy(sd, padding, padLen);
                sd += padLen;
            }
        }
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}


int sort_by_key_codepoint(ktxKVListEntry* a, ktxKVListEntry* b) {
  return strcmp(a->key, b->key);
}

/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Sort a hash list in order of the UTF8 codepoints.
 *
 * @param [in]     pHead        pointer to the head of the target hash list.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p This is NULL.
 */
KTX_error_code
ktxHashList_Sort(ktxHashList* pHead)
{
    if (pHead) {
        //ktxKVListEntry* kv = (ktxKVListEntry*)pHead;

        HASH_SORT(*pHead, sort_by_key_codepoint);
        return KTX_SUCCESS;
    } else {
        return KTX_INVALID_VALUE;
    }
}


/**
 * @memberof ktxHashList @public
 * @~English
 * @brief Construct a hash list from a block of serialized key-value
 *        data read from a file.
 * @note The bytes of the 32-bit key-value lengths within the serialized data
 *       are expected to be in native endianness.
 *
 * @param [in]      pHead       pointer to the head of the target hash list.
 * @param [in]      kvdLen      the length of the serialized key-value data.
 * @param [in]      pKvd        pointer to the serialized key-value data.
 *                              table.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_OPERATION if @p pHead does not point to an empty list.
 * @exception KTX_INVALID_VALUE if @p pKvd or @p pHt is NULL or kvdLen == 0.
 * @exception KTX_OUT_OF_MEMORY there was not enough memory to create the hash
 *                              table.
 */
KTX_error_code
ktxHashList_Deserialize(ktxHashList* pHead, unsigned int kvdLen, void* pKvd)
{
    char* src = pKvd;
    KTX_error_code result;

    if (kvdLen == 0 || pKvd == NULL || pHead == NULL)
        return KTX_INVALID_VALUE;

    if (*pHead != NULL)
        return KTX_INVALID_OPERATION;

    result = KTX_SUCCESS;
    while (result == KTX_SUCCESS && src < (char *)pKvd + kvdLen) {
        char* key;
        unsigned int keyLen, valueLen;
        void* value;
        ktx_uint32_t keyAndValueByteSize = *((ktx_uint32_t*)src);

        src += sizeof(keyAndValueByteSize);
        key = src;
        keyLen = (unsigned int)strlen(key) + 1;
        value = key + keyLen;

        valueLen = keyAndValueByteSize - keyLen;
        result = ktxHashList_AddKVPair(pHead, key, valueLen,
                                       valueLen > 0 ? value : NULL);
        if (result == KTX_SUCCESS) {
            src += _KTX_PAD4(keyAndValueByteSize);
        }
    }
    return result;
}


/**
 * @memberof ktxHashListEntry @public
 * @~English
 * @brief Return the key of a ktxHashListEntry
 *
 * @param [in]     This       The target hash list entry.
 * @param [in,out] pKeyLen    @p *pKeyLen is set to the byte length of
 *                            the returned key.
 * @param [in,out] ppKey      @p *ppKey is set to the point to the value of
 *                            @p the key.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p pKvd or @p pHt is NULL or kvdLen == 0.
 */
KTX_error_code
ktxHashListEntry_GetKey(ktxHashListEntry* This,
                        unsigned int* pKeyLen, char** ppKey)
{
    if (pKeyLen && ppKey) {
        ktxKVListEntry* kv = (ktxKVListEntry*)This;
        *pKeyLen = kv->keyLen;
        *ppKey = kv->key;
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}


/**
 * @memberof ktxHashListEntry @public
 * @~English
 * @brief Return the value from a ktxHashListEntry
 *
 * @param [in]     This         The target hash list entry.
 * @param [in,out] pValueLen    @p *pValueLen is set to the number of bytes of
 *                              data in the returned value.
 * @param [in,out] ppValue      @p *ppValue is set to point to the value of
 *                              of the target entry.
 *
 * @return KTX_SUCCESS or one of the following error codes.
 *
 * @exception KTX_INVALID_VALUE if @p pKvd or @p pHt is NULL or kvdLen == 0.
 */
KTX_error_code
ktxHashListEntry_GetValue(ktxHashListEntry* This,
                          unsigned int* pValueLen, void** ppValue)
{
    if (pValueLen && ppValue) {
        ktxKVListEntry* kv = (ktxKVListEntry*)This;
        *pValueLen = kv->valueLen;
        *ppValue = kv->valueLen > 0 ? kv->value : NULL;
        return KTX_SUCCESS;
    } else
        return KTX_INVALID_VALUE;
}
