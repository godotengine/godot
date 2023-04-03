/*
 * XML DRI client-side driver configuration
 * Copyright (C) 2003 Felix Kuehling
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * FELIX KUEHLING, OR ANY OTHER CONTRIBUTORS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */
/**
 * \file xmlconfig.h
 * \brief Driver-independent client-side part of the XML configuration
 * \author Felix Kuehling
 */

#ifndef __XMLCONFIG_H
#define __XMLCONFIG_H

#include "util/mesa-sha1.h"
#include "util/ralloc.h"
#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STRING_CONF_MAXLEN 1024

/** \brief Option data types */
typedef enum driOptionType {
   DRI_BOOL, DRI_ENUM, DRI_INT, DRI_FLOAT, DRI_STRING, DRI_SECTION
} driOptionType;

/** \brief Option value */
typedef union driOptionValue {
   unsigned char _bool; /**< \brief Boolean */
   int _int;      /**< \brief Integer or Enum */
   float _float;  /**< \brief Floating-point */
   char *_string;   /**< \brief String */
} driOptionValue;

/** \brief Single range of valid values
 *
 * For empty ranges (a single value) start == end */
typedef struct driOptionRange {
   driOptionValue start; /**< \brief Start */
   driOptionValue end;   /**< \brief End */
} driOptionRange;

/** \brief Information about an option */
typedef struct driOptionInfo {
   char *name;             /**< \brief Name */
   driOptionType type;     /**< \brief Type */
   driOptionRange range;   /**< \brief Valid range of the option (or 0:0) */
} driOptionInfo;

/** \brief Option cache
 *
 * \li One in <driver>Screen caching option info and the default values
 * \li One in each <driver>Context with the actual values for that context */
typedef struct driOptionCache {
   driOptionInfo *info;
   /**< \brief Array of option infos
    *
    * Points to the same array in the screen and all contexts */
   driOptionValue *values;
   /**< \brief Array of option values
    *
    * \li Default values in screen
    * \li Actual values in contexts
    */
   unsigned int tableSize;
   /**< \brief Size of the arrays
    *
    * In the current implementation it's not actually a size but log2(size).
    * The value is the same in the screen and all contexts. */
} driOptionCache;

typedef struct driEnumDescription {
   int value;
   const char *desc;
} driEnumDescription;

/**
 * Struct for a driver's definition of an option, its default value, and the
 * text documenting it.
 */
typedef struct driOptionDescription {
   const char *desc;

   driOptionInfo info;
   driOptionValue value;
   driEnumDescription enums[4];
} driOptionDescription;

/** Returns an XML string describing the options for the driver. */
char *
driGetOptionsXml(const driOptionDescription *configOptions, unsigned numOptions);

/** \brief Parse driconf option array from configOptions
 *
 * To be called in <driver>CreateScreen
 *
 * \param info    pointer to a driOptionCache that will store the option info
 * \param configOptions   Array of XML document describing available configuration opts
 *
 * For the option information to be available to external configuration tools
 * it must be a public symbol __driConfigOptions. It is also passed as a
 * parameter to driParseOptionInfo in order to avoid driver-independent code
 * depending on symbols in driver-specific code. */
void driParseOptionInfo(driOptionCache *info,
                        const driOptionDescription *configOptions,
                        unsigned numOptions);
/** \brief Initialize option cache from info and parse configuration files
 *
 * To be called in <driver>CreateContext. screenNum, driverName,
 * kernelDriverName, applicationName and engineName select device sections. */
void driParseConfigFiles(driOptionCache *cache, const driOptionCache *info,
                         int screenNum, const char *driverName,
                         const char *kernelDriverName,
                         const char *deviceName,
                         const char *applicationName, uint32_t applicationVersion,
                         const char *engineName, uint32_t engineVersion);
/** \brief Destroy option info
 *
 * To be called in <driver>DestroyScreen */
void driDestroyOptionInfo(driOptionCache *info);
/** \brief Destroy option cache
 *
 * To be called in <driver>DestroyContext */
void driDestroyOptionCache(driOptionCache *cache);

/** \brief Check if there exists a certain option */
unsigned char driCheckOption(const driOptionCache *cache, const char *name,
                             driOptionType type);

/** \brief Query a boolean option value */
unsigned char driQueryOptionb(const driOptionCache *cache, const char *name);
/** \brief Query an integer option value */
int driQueryOptioni(const driOptionCache *cache, const char *name);
/** \brief Query a floating-point option value */
float driQueryOptionf(const driOptionCache *cache, const char *name);
/** \brief Query a string option value */
char *driQueryOptionstr(const driOptionCache *cache, const char *name);

/* Overrides for the unit tests to control drirc parsing. */
void driInjectDataDir(const char *dir);
void driInjectExecName(const char *exec);

/**
 * Returns a hash of the options for this application.
 */
static inline void
driComputeOptionsSha1(const driOptionCache *cache, unsigned char *sha1)
{
   void *ctx = ralloc_context(NULL);
   char *dri_options = ralloc_strdup(ctx, "");

   for (int i = 0; i < 1 << cache->tableSize; i++) {
      if (cache->info[i].name == NULL)
         continue;

      bool ret = false;
      switch (cache->info[i].type) {
      case DRI_BOOL:
         ret = ralloc_asprintf_append(&dri_options, "%s:%u,",
                                      cache->info[i].name,
                                      cache->values[i]._bool);
         break;
      case DRI_INT:
      case DRI_ENUM:
         ret = ralloc_asprintf_append(&dri_options, "%s:%d,",
                                      cache->info[i].name,
                                      cache->values[i]._int);
         break;
      case DRI_FLOAT:
         ret = ralloc_asprintf_append(&dri_options, "%s:%f,",
                                      cache->info[i].name,
                                      cache->values[i]._float);
         break;
      case DRI_STRING:
         ret = ralloc_asprintf_append(&dri_options, "%s:%s,",
                                      cache->info[i].name,
                                      cache->values[i]._string);
         break;
      default:
         unreachable("unsupported dri config type!");
      }

      if (!ret) {
         break;
      }
   }

   _mesa_sha1_compute(dri_options, strlen(dri_options), sha1);
   ralloc_free(ctx);
}

#ifdef __cplusplus
} /* extern C */
#endif

#endif
