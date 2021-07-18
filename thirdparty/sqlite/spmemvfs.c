/*
* BSD 2-Clause License
*
* Copyright 2009 Stephen Liu
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma warning(disable: 4996) // Fixes "unsafe" warnings for strdup and strncpy

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "spmemvfs.h"

#include "sqlite3.h"

/* Useful macros used in several places */
#define SPMEMVFS_MIN(x,y) ((x)<(y)?(x):(y))
#define SPMEMVFS_MAX(x,y) ((x)>(y)?(x):(y))

static void spmemvfsDebug(const char *format, ...){

#if defined(SPMEMVFS_DEBUG)

	char logTemp[ 1024 ] = { 0 };

	va_list vaList;
	va_start( vaList, format );
	vsnprintf( logTemp, sizeof( logTemp ), format, vaList );
	va_end ( vaList );

	if( strchr( logTemp, '\n' ) ) {
		printf( "%s", logTemp );
	} else {
		printf( "%s\n", logTemp );
	}
#endif

}

//===========================================================================

typedef struct spmemfile_t {
	sqlite3_file base;
	char * path;
	int flags;
	spmembuffer_t * mem;
} spmemfile_t;

static int spmemfileClose( sqlite3_file * file );
static int spmemfileRead( sqlite3_file * file, void * buffer, int len, sqlite3_int64 offset );
static int spmemfileWrite( sqlite3_file * file, const void * buffer, int len, sqlite3_int64 offset );
static int spmemfileTruncate( sqlite3_file * file, sqlite3_int64 size );
static int spmemfileSync( sqlite3_file * file, int flags );
static int spmemfileFileSize( sqlite3_file * file, sqlite3_int64 * size );
static int spmemfileLock( sqlite3_file * file, int type );
static int spmemfileUnlock( sqlite3_file * file, int type );
static int spmemfileCheckReservedLock( sqlite3_file * file, int * result );
static int spmemfileFileControl( sqlite3_file * file, int op, void * arg );
static int spmemfileSectorSize( sqlite3_file * file );
static int spmemfileDeviceCharacteristics( sqlite3_file * file );

static sqlite3_io_methods g_spmemfile_io_memthods = {
	1,                                  /* iVersion */
	spmemfileClose,                     /* xClose */
	spmemfileRead,                      /* xRead */
	spmemfileWrite,                     /* xWrite */
	spmemfileTruncate,                  /* xTruncate */
	spmemfileSync,                      /* xSync */
	spmemfileFileSize,                  /* xFileSize */
	spmemfileLock,                      /* xLock */
	spmemfileUnlock,                    /* xUnlock */
	spmemfileCheckReservedLock,         /* xCheckReservedLock */
	spmemfileFileControl,               /* xFileControl */
	spmemfileSectorSize,                /* xSectorSize */
	spmemfileDeviceCharacteristics      /* xDeviceCharacteristics */
};

int spmemfileClose( sqlite3_file * file )
{
	spmemfile_t * memfile = (spmemfile_t*)file;

	spmemvfsDebug( "call %s( %p )", __func__, memfile );

	if( SQLITE_OPEN_MAIN_DB & memfile->flags ) {
		// noop
	} else {
		if( NULL != memfile->mem ) {
			if( memfile->mem->data ) free( memfile->mem->data );
			free( memfile->mem );
		}
	}

	free( memfile->path );

	return SQLITE_OK;
}

int spmemfileRead( sqlite3_file * file, void * buffer, int len, sqlite3_int64 offset )
{
	spmemfile_t * memfile = (spmemfile_t*)file;

	spmemvfsDebug( "call %s( %p, ..., %d, %lld ), len %d",
		__func__, memfile, len, offset, memfile->mem->used );

	if( ( offset + len ) > memfile->mem->used ) {
		return SQLITE_IOERR_SHORT_READ;
	}

	memcpy( buffer, memfile->mem->data + offset, len );

	return SQLITE_OK;
}

int spmemfileWrite( sqlite3_file * file, const void * buffer, int len, sqlite3_int64 offset )
{
	spmemfile_t * memfile = (spmemfile_t*)file;
	spmembuffer_t * mem = memfile->mem;

	spmemvfsDebug( "call %s( %p, ..., %d, %lld ), len %d",
		__func__, memfile, len, offset, mem->used );

	if( ( offset + len ) > mem->total ) {
		int64_t newTotal = 2 * ( offset + len + mem->total );
		char * newBuffer = (char*)realloc( mem->data, newTotal );
		if( NULL == newBuffer ) {
			return SQLITE_NOMEM;
		}

		mem->total = newTotal;
		mem->data = newBuffer;
	}

	memcpy( mem->data + offset, buffer, len );

	mem->used = SPMEMVFS_MAX( mem->used, offset + len );

	return SQLITE_OK;
}

int spmemfileTruncate( sqlite3_file * file, sqlite3_int64 size )
{
	spmemfile_t * memfile = (spmemfile_t*)file;

	spmemvfsDebug( "call %s( %p )", __func__, memfile );

	memfile->mem->used = SPMEMVFS_MIN( memfile->mem->used, size );

	return SQLITE_OK;
}

int spmemfileSync( sqlite3_file * file, int flags )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return SQLITE_OK;
}

int spmemfileFileSize( sqlite3_file * file, sqlite3_int64 * size )
{
	spmemfile_t * memfile = (spmemfile_t*)file;

	spmemvfsDebug( "call %s( %p )", __func__, memfile );

	* size = memfile->mem->used;

	return SQLITE_OK;
}

int spmemfileLock( sqlite3_file * file, int type )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return SQLITE_OK;
}

int spmemfileUnlock( sqlite3_file * file, int type )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return SQLITE_OK;
}

int spmemfileCheckReservedLock( sqlite3_file * file, int * result )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	*result = 0;

	return SQLITE_OK;
}

int spmemfileFileControl( sqlite3_file * file, int op, void * arg )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return SQLITE_OK;
}

int spmemfileSectorSize( sqlite3_file * file )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return 0;
}

int spmemfileDeviceCharacteristics( sqlite3_file * file )
{
	spmemvfsDebug( "call %s( %p )", __func__, file );

	return 0;
}

//===========================================================================

typedef struct spmemvfs_cb_t {
	void * arg;
	spmembuffer_t * ( * load ) ( void * args, const char * path );
} spmemvfs_cb_t;

typedef struct spmemvfs_t {
	sqlite3_vfs base;
	spmemvfs_cb_t cb;
	sqlite3_vfs * parent;
} spmemvfs_t;

static int spmemvfsOpen( sqlite3_vfs * vfs, const char * path, sqlite3_file * file, int flags, int * outflags );
static int spmemvfsDelete( sqlite3_vfs * vfs, const char * path, int syncDir );
static int spmemvfsAccess( sqlite3_vfs * vfs, const char * path, int flags, int * result );
static int spmemvfsFullPathname( sqlite3_vfs * vfs, const char * path, int len, char * fullpath );
static void * spmemvfsDlOpen( sqlite3_vfs * vfs, const char * path );
static void spmemvfsDlError( sqlite3_vfs * vfs, int len, char * errmsg );
static void ( * spmemvfsDlSym ( sqlite3_vfs * vfs, void * handle, const char * symbol ) ) ( void );
static void spmemvfsDlClose( sqlite3_vfs * vfs, void * handle );
static int spmemvfsRandomness( sqlite3_vfs * vfs, int len, char * buffer );
static int spmemvfsSleep( sqlite3_vfs * vfs, int microseconds );
static int spmemvfsCurrentTime( sqlite3_vfs * vfs, double * result );

static spmemvfs_t g_spmemvfs = {
	{
		1,                                           /* iVersion */
		0,                                           /* szOsFile */
		0,                                           /* mxPathname */
		0,                                           /* pNext */
		SPMEMVFS_NAME,                               /* zName */
		0,                                           /* pAppData */
		spmemvfsOpen,                                /* xOpen */
		spmemvfsDelete,                              /* xDelete */
		spmemvfsAccess,                              /* xAccess */
		spmemvfsFullPathname,                        /* xFullPathname */
		spmemvfsDlOpen,                              /* xDlOpen */
		spmemvfsDlError,                             /* xDlError */
		spmemvfsDlSym,                               /* xDlSym */
		spmemvfsDlClose,                             /* xDlClose */
		spmemvfsRandomness,                          /* xRandomness */
		spmemvfsSleep,                               /* xSleep */
		spmemvfsCurrentTime                          /* xCurrentTime */
	}, 
	{ 0 },
	0                                                /* pParent */
};

int spmemvfsOpen( sqlite3_vfs * vfs, const char * path, sqlite3_file * file, int flags, int * outflags )
{
	spmemvfs_t * memvfs = (spmemvfs_t*)vfs;
	spmemfile_t * memfile = (spmemfile_t*)file;

	spmemvfsDebug( "call %s( %p(%p), %s, %p, %x, %p )\n",
			__func__, vfs, &g_spmemvfs, path, file, flags, outflags );

	memset( memfile, 0, sizeof( spmemfile_t ) );
	memfile->base.pMethods = &g_spmemfile_io_memthods;
	memfile->flags = flags;

	memfile->path = strdup( path );

	if( SQLITE_OPEN_MAIN_DB & memfile->flags ) {
		memfile->mem = memvfs->cb.load( memvfs->cb.arg, path );
	} else {
		memfile->mem = (spmembuffer_t*)calloc( sizeof( spmembuffer_t ), 1 );
	}

	return memfile->mem ? SQLITE_OK : SQLITE_ERROR;
}

int spmemvfsDelete( sqlite3_vfs * vfs, const char * path, int syncDir )
{
	spmemvfsDebug( "call %s( %p(%p), %s, %d )\n",
			__func__, vfs, &g_spmemvfs, path, syncDir );

	return SQLITE_OK;
}

int spmemvfsAccess( sqlite3_vfs * vfs, const char * path, int flags, int * result )
{
	* result = 0;
	return SQLITE_OK;
}

int spmemvfsFullPathname( sqlite3_vfs * vfs, const char * path, int len, char * fullpath )
{
	strncpy( fullpath, path, len );
	fullpath[ len - 1 ] = '\0';

	return SQLITE_OK;
}

void * spmemvfsDlOpen( sqlite3_vfs * vfs, const char * path )
{
	return NULL;
}

void spmemvfsDlError( sqlite3_vfs * vfs, int len, char * errmsg )
{
	// noop
}

void ( * spmemvfsDlSym ( sqlite3_vfs * vfs, void * handle, const char * symbol ) ) ( void )
{
	return NULL;
}

void spmemvfsDlClose( sqlite3_vfs * vfs, void * handle )
{
	// noop
}

int spmemvfsRandomness( sqlite3_vfs * vfs, int len, char * buffer )
{
	return SQLITE_OK;
}

int spmemvfsSleep( sqlite3_vfs * vfs, int microseconds )
{
	return SQLITE_OK;
}

int spmemvfsCurrentTime( sqlite3_vfs * vfs, double * result )
{
	return SQLITE_OK;
}

//===========================================================================

int spmemvfs_init( spmemvfs_cb_t * cb )
{
	sqlite3_vfs * parent = NULL;

	if( g_spmemvfs.parent ) return SQLITE_OK;

	parent = sqlite3_vfs_find( 0 );

	g_spmemvfs.parent = parent;

	g_spmemvfs.base.mxPathname = parent->mxPathname;
	g_spmemvfs.base.szOsFile = sizeof( spmemfile_t );

	g_spmemvfs.cb = * cb;

	return sqlite3_vfs_register( (sqlite3_vfs*)&g_spmemvfs, 0 );
}

//===========================================================================

typedef struct spmembuffer_link_t {
	char * path;
	spmembuffer_t * mem;
	struct spmembuffer_link_t * next;
} spmembuffer_link_t;

spmembuffer_link_t * spmembuffer_link_remove( spmembuffer_link_t ** head, const char * path )
{
	spmembuffer_link_t * ret = NULL;

	spmembuffer_link_t ** iter = head;
	for( ; NULL != *iter; ) {
		spmembuffer_link_t * curr = *iter;

		if( 0 == strcmp( path, curr->path ) ) {
			ret = curr;
			*iter = curr->next;
			break;
		} else {
			iter = &( curr->next );
		}
	}

	return ret;
}

void spmembuffer_link_free( spmembuffer_link_t * iter )
{
	free( iter->path );
	free( iter->mem->data );
	free( iter->mem );
	free( iter );
}

//===========================================================================

typedef struct spmemvfs_env_t {
	spmembuffer_link_t * head;
	sqlite3_mutex * mutex;
} spmemvfs_env_t;

static spmemvfs_env_t * g_spmemvfs_env = NULL;

static spmembuffer_t * load_cb( void * arg, const char * path )
{
	spmembuffer_t * ret = NULL;

	spmemvfs_env_t * env = (spmemvfs_env_t*)arg;

	sqlite3_mutex_enter( env->mutex );
	{
		spmembuffer_link_t * toFind = spmembuffer_link_remove( &( env->head ), path );

		if( NULL != toFind ) {
			ret = toFind->mem;
			free( toFind->path );
			free( toFind );
		}
	}
	sqlite3_mutex_leave( env->mutex );

	return ret;
}

int spmemvfs_env_init()
{
	int ret = 0;

	if( NULL == g_spmemvfs_env ) {
		spmemvfs_cb_t cb;

		g_spmemvfs_env = (spmemvfs_env_t*)calloc( sizeof( spmemvfs_env_t ), 1 );
		g_spmemvfs_env->mutex = sqlite3_mutex_alloc( SQLITE_MUTEX_FAST );

		cb.arg = g_spmemvfs_env;
		cb.load = load_cb;

		ret = spmemvfs_init( &cb );
	}

	return ret;
}

void spmemvfs_env_fini()
{
	if( NULL != g_spmemvfs_env ) {
		spmembuffer_link_t * iter = NULL;

		sqlite3_vfs_unregister( (sqlite3_vfs*)&g_spmemvfs );
		g_spmemvfs.parent = NULL;

		sqlite3_mutex_free( g_spmemvfs_env->mutex );

		iter = g_spmemvfs_env->head;
		for( ; NULL != iter; ) {
			spmembuffer_link_t * next = iter->next;

			spmembuffer_link_free( iter );

			iter = next;
		}

		free( g_spmemvfs_env );
		g_spmemvfs_env = NULL;
	}
}

int spmemvfs_open_db( spmemvfs_db_t * db, const char * path, spmembuffer_t * mem )
{
	int ret = 0;

	spmembuffer_link_t * iter = NULL;

	memset( db, 0, sizeof( spmemvfs_db_t ) );

	iter = (spmembuffer_link_t*)calloc( sizeof( spmembuffer_link_t ), 1 );
	iter->path = strdup( path );
	iter->mem = mem;

	sqlite3_mutex_enter( g_spmemvfs_env->mutex );
	{
		iter->next = g_spmemvfs_env->head;
		g_spmemvfs_env->head = iter;
	}
	sqlite3_mutex_leave( g_spmemvfs_env->mutex );

	ret = sqlite3_open_v2( path, &(db->handle),
			SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, SPMEMVFS_NAME );

	if( 0 == ret ) {
		db->mem = mem;
	} else {
		sqlite3_mutex_enter( g_spmemvfs_env->mutex );
		{
			iter = spmembuffer_link_remove( &(g_spmemvfs_env->head), path );
			if( NULL != iter ) spmembuffer_link_free( iter );
		}
		sqlite3_mutex_leave( g_spmemvfs_env->mutex );
	}

	return ret;
}

int spmemvfs_close_db( spmemvfs_db_t * db )
{
	int ret = 0;

	if( NULL == db ) return 0;

	if( NULL != db->handle ) {
		ret = sqlite3_close( db->handle );
		db->handle = NULL;
	}

	if( NULL != db->mem ) {
		if( NULL != db->mem->data ) free( db->mem->data );
		free( db->mem );
		db->mem = NULL;
	}

	return ret;
}

