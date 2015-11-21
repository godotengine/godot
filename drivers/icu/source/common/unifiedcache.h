/*
******************************************************************************
* Copyright (C) 2014, International Business Machines Corporation and
* others. All Rights Reserved.
******************************************************************************
*
* File UNIFIEDCACHE.H - The ICU Unified cache.
******************************************************************************
*/

#ifndef __UNIFIED_CACHE_H__
#define __UNIFIED_CACHE_H__

#include "utypeinfo.h"  // for 'typeid' to work

#include "unicode/uobject.h"
#include "unicode/locid.h"
#include "sharedobject.h"
#include "unicode/unistr.h"
#include "cstring.h"
#include "ustr_imp.h"

struct UHashtable;
struct UHashElement;

U_NAMESPACE_BEGIN

class UnifiedCache;

/**
 * A base class for all cache keys
 */
class U_COMMON_API CacheKeyBase : public UObject {
 public:
   CacheKeyBase() : creationStatus(U_ZERO_ERROR) {}

   /**
    * Copy constructor. Needed to support cloning.
    */
   CacheKeyBase(const CacheKeyBase &other) 
           : UObject(other), creationStatus(other.creationStatus) { }
   virtual ~CacheKeyBase();

   /**
    * Returns the hash code for this object.
    */
   virtual int32_t hashCode() const = 0;

   /**
    * Clones this object polymorphically. Caller owns returned value.
    */
   virtual CacheKeyBase *clone() const = 0;

   /**
    * Equality operator.
    */
   virtual UBool operator == (const CacheKeyBase &other) const = 0;

   /**
    * Create a new object for this key. Called by cache on cache miss.
    * createObject must add a reference to the object it returns. Note
    * that getting an object from the cache and returning it without calling
    * removeRef on it satisfies this requirement. It can also return NULL
    * and set status to an error.
    *
    * @param creationContext the context in which the object is being
    *                        created. May be NULL.
    * @param status          Implementations can return a failure here.
    *                        In addition, implementations may return a
    *                        non NULL object and set a warning status.
    */
   virtual const SharedObject *createObject(
           const void *creationContext, UErrorCode &status) const = 0;

   /**
    * Writes a description of this key to buffer and returns buffer. Written
    * description is NULL terminated.
    */
   virtual char *writeDescription(char *buffer, int32_t bufSize) const = 0;

   /**
    * Inequality operator.
    */
   UBool operator != (const CacheKeyBase &other) const {
       return !(*this == other);
   }
 private:
   mutable UErrorCode creationStatus;
   friend class UnifiedCache;
};



/**
 * Templated version of CacheKeyBase. 
 * A key of type LocaleCacheKey<T> maps to a value of type T.
 */
template<typename T>
class CacheKey : public CacheKeyBase {
 public:
   virtual ~CacheKey() { }
   /**
    * The template parameter, T, determines the hash code returned.
    */
   virtual int32_t hashCode() const {
       const char *s = typeid(T).name();
       return ustr_hashCharsN(s, uprv_strlen(s));
   }

   /**
    * Use the value type, T,  as the description.
    */
   virtual char *writeDescription(char *buffer, int32_t bufLen) const {
       const char *s = typeid(T).name();
       uprv_strncpy(buffer, s, bufLen);
       buffer[bufLen - 1] = 0;
       return buffer;
   }

   /**
    * Two objects are equal if they are of the same type.
    */
   virtual UBool operator == (const CacheKeyBase &other) const {
       return typeid(*this) == typeid(other);
   }
};

/**
 * Cache key based on locale.
 * A key of type LocaleCacheKey<T> maps to a value of type T.
 */
template<typename T>
class LocaleCacheKey : public CacheKey<T> {
 protected:
   Locale   fLoc;
 public:
   LocaleCacheKey(const Locale &loc) : fLoc(loc) {};
   LocaleCacheKey(const LocaleCacheKey<T> &other)
           : CacheKey<T>(other), fLoc(other.fLoc) { }
   virtual ~LocaleCacheKey() { }
   virtual int32_t hashCode() const {
       return 37 *CacheKey<T>::hashCode() + fLoc.hashCode();
   }
   virtual UBool operator == (const CacheKeyBase &other) const {
       // reflexive
       if (this == &other) {
           return TRUE;
       }
       if (!CacheKey<T>::operator == (other)) {
           return FALSE;
       }
       // We know this and other are of same class because operator== on
       // CacheKey returned true.
       const LocaleCacheKey<T> *fOther =
               static_cast<const LocaleCacheKey<T> *>(&other);
       return fLoc == fOther->fLoc;
   }
   virtual CacheKeyBase *clone() const {
       return new LocaleCacheKey<T>(*this);
   }
   virtual const T *createObject(
           const void *creationContext, UErrorCode &status) const;
   /**
    * Use the locale id as the description.
    */
   virtual char *writeDescription(char *buffer, int32_t bufLen) const {
       const char *s = fLoc.getName();
       uprv_strncpy(buffer, s, bufLen);
       buffer[bufLen - 1] = 0;
       return buffer;
   }

};

/**
 * The unified cache. A singleton type.
 */
class U_COMMON_API UnifiedCache : public UObject {
 public:
   /**
    * @internal
    */
   UnifiedCache(UErrorCode &status);

   /**
    * Returns the cache instance.
    */
   static const UnifiedCache *getInstance(UErrorCode &status);

   /**
    * Fetches a value from the cache by key. Equivalent to
    * get(key, NULL, ptr, status);
    */
   template<typename T>
   void get(
           const CacheKey<T>& key,
           const T *&ptr,
           UErrorCode &status) const {
       get(key, NULL, ptr, status);
   }

   /**
    * Fetches value from the cache by key.
    *
    * @param key             the cache key.
    * @param creationContext passed verbatim to createObject method of key
    * @param ptr             On entry, ptr must be NULL or be included if
    *                        the reference count of the object it points
    *                        to. On exit, ptr points to the fetched object
    *                        from the cache or is left unchanged on
    *                        failure. Caller must call removeRef on ptr
    *                        if set to a non NULL value.
    * @param status          Any error returned here. May be set to a
    *                        warning value even if ptr is set.
    */
   template<typename T>
   void get(
           const CacheKey<T>& key,
           const void *creationContext,
           const T *&ptr,
           UErrorCode &status) const {
       if (U_FAILURE(status)) {
           return;
       }
       UErrorCode creationStatus = U_ZERO_ERROR;
       const SharedObject *value = NULL;
       _get(key, value, creationContext, creationStatus);
       const T *tvalue = (const T *) value;
       if (U_SUCCESS(creationStatus)) {
           SharedObject::copyPtr(tvalue, ptr);
       }
       SharedObject::clearPtr(tvalue);
       // Take care not to overwrite a warning status passed in with
       // another warning or U_ZERO_ERROR.
       if (status == U_ZERO_ERROR || U_FAILURE(creationStatus)) {
           status = creationStatus;
       }
   }

#ifdef UNIFIED_CACHE_DEBUG
   /**
    * Dumps the contents of this cache to standard error. Used for testing of
    * cache only.
    */
   void dumpContents() const;
#endif

   /**
    * Convenience method to get a value of type T from cache for a
    * particular locale with creationContext == NULL.
    * @param loc    the locale
    * @param ptr    On entry, must be NULL or included in the ref count
    *               of the object to which it points.
    *               On exit, fetched value stored here or is left
    *               unchanged on failure. Caller must call removeRef on
    *               ptr if set to a non NULL value.
    * @param status Any error returned here. May be set to a
    *               warning value even if ptr is set.
    */
   template<typename T>
   static void getByLocale(
           const Locale &loc, const T *&ptr, UErrorCode &status) {
       const UnifiedCache *cache = getInstance(status);
       if (U_FAILURE(status)) {
           return;
       }
       cache->get(LocaleCacheKey<T>(loc), ptr, status);
   }

#ifdef UNIFIED_CACHE_DEBUG
   /**
    * Dumps the cache contents to stderr. For testing only.
    */
   static void dump();
#endif

   /**
    * Returns the number of keys in this cache. For testing only.
    */
   int32_t keyCount() const;

   /**
    * Removes any values from cache that are not referenced outside
    * the cache.
    */
   void flush() const;

   virtual ~UnifiedCache();
 private:
   UHashtable *fHashtable;
   UnifiedCache(const UnifiedCache &other);
   UnifiedCache &operator=(const UnifiedCache &other);
   UBool _flush(UBool all) const;
   void _get(
           const CacheKeyBase &key,
           const SharedObject *&value,
           const void *creationContext,
           UErrorCode &status) const;
   UBool _poll(
           const CacheKeyBase &key,
           const SharedObject *&value,
           UErrorCode &status) const;
   void _putNew(
           const CacheKeyBase &key,
           const SharedObject *value,
           const UErrorCode creationStatus,
           UErrorCode &status) const;
   void _putIfAbsentAndGet(
           const CacheKeyBase &key,
           const SharedObject *&value,
           UErrorCode &status) const;
#ifdef UNIFIED_CACHE_DEBUG
   void _dumpContents() const;
#endif
   static void _put(
           const UHashElement *element,
           const SharedObject *value,
           const UErrorCode status);
   static void _fetch(
           const UHashElement *element,
           const SharedObject *&value,
           UErrorCode &status);
   static UBool _inProgress(const UHashElement *element);
};

U_NAMESPACE_END

#endif
