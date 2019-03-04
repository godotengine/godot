/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010-2018 Andy Green <andy@warmcat.com>
 *
 *  This library is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation:
 *  version 2.1 of the License.
 *
 *  This library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 *  MA  02110-1301  USA
 *
 * included from libwebsockets.h
 */

/*! \defgroup fops file operation wrapping
 *
 * ##File operation wrapping
 *
 * Use these helper functions if you want to access a file from the perspective
 * of a specific wsi, which is usually the case.  If you just want contextless
 * file access, use the fops callbacks directly with NULL wsi instead of these
 * helpers.
 *
 * If so, then it calls the platform handler or user overrides where present
 * (as defined in info->fops)
 *
 * The advantage from all this is user code can be portable for file operations
 * without having to deal with differences between platforms.
 */
//@{

/** struct lws_plat_file_ops - Platform-specific file operations
 *
 * These provide platform-agnostic ways to deal with filesystem access in the
 * library and in the user code.
 */

#if defined(LWS_WITH_ESP32)
/* sdk preprocessor defs? compiler issue? gets confused with member names */
#define LWS_FOP_OPEN		_open
#define LWS_FOP_CLOSE		_close
#define LWS_FOP_SEEK_CUR	_seek_cur
#define LWS_FOP_READ		_read
#define LWS_FOP_WRITE		_write
#else
#define LWS_FOP_OPEN		open
#define LWS_FOP_CLOSE		close
#define LWS_FOP_SEEK_CUR	seek_cur
#define LWS_FOP_READ		read
#define LWS_FOP_WRITE		write
#endif

#define LWS_FOP_FLAGS_MASK		   ((1 << 23) - 1)
#define LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP (1 << 24)
#define LWS_FOP_FLAG_COMPR_IS_GZIP	   (1 << 25)
#define LWS_FOP_FLAG_MOD_TIME_VALID	   (1 << 26)
#define LWS_FOP_FLAG_VIRTUAL		   (1 << 27)

struct lws_plat_file_ops;

struct lws_fop_fd {
	lws_filefd_type			fd;
	/**< real file descriptor related to the file... */
	const struct lws_plat_file_ops	*fops;
	/**< fops that apply to this fop_fd */
	void				*filesystem_priv;
	/**< ignored by lws; owned by the fops handlers */
	lws_filepos_t			pos;
	/**< generic "position in file" */
	lws_filepos_t			len;
	/**< generic "length of file" */
	lws_fop_flags_t			flags;
	/**< copy of the returned flags */
	uint32_t			mod_time;
	/**< optional "modification time of file", only valid if .open()
	 * set the LWS_FOP_FLAG_MOD_TIME_VALID flag */
};
typedef struct lws_fop_fd *lws_fop_fd_t;

struct lws_fops_index {
	const char *sig;	/* NULL or vfs signature, eg, ".zip/" */
	uint8_t len;		/* length of above string */
};

struct lws_plat_file_ops {
	lws_fop_fd_t (*LWS_FOP_OPEN)(const struct lws_plat_file_ops *fops,
				     const char *filename, const char *vpath,
				     lws_fop_flags_t *flags);
	/**< Open file (always binary access if plat supports it)
	 * vpath may be NULL, or if the fops understands it, the point at which
	 * the filename's virtual part starts.
	 * *flags & LWS_FOP_FLAGS_MASK should be set to O_RDONLY or O_RDWR.
	 * If the file may be gzip-compressed,
	 * LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP is set.  If it actually is
	 * gzip-compressed, then the open handler should OR
	 * LWS_FOP_FLAG_COMPR_IS_GZIP on to *flags before returning.
	 */
	int (*LWS_FOP_CLOSE)(lws_fop_fd_t *fop_fd);
	/**< close file AND set the pointer to NULL */
	lws_fileofs_t (*LWS_FOP_SEEK_CUR)(lws_fop_fd_t fop_fd,
					  lws_fileofs_t offset_from_cur_pos);
	/**< seek from current position */
	int (*LWS_FOP_READ)(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			    uint8_t *buf, lws_filepos_t len);
	/**< Read from file, on exit *amount is set to amount actually read */
	int (*LWS_FOP_WRITE)(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			     uint8_t *buf, lws_filepos_t len);
	/**< Write to file, on exit *amount is set to amount actually written */

	struct lws_fops_index fi[3];
	/**< vfs path signatures implying use of this fops */

	const struct lws_plat_file_ops *next;
	/**< NULL or next fops in list */

	/* Add new things just above here ---^
	 * This is part of the ABI, don't needlessly break compatibility */
};

/**
 * lws_get_fops() - get current file ops
 *
 * \param context: context
 */
LWS_VISIBLE LWS_EXTERN struct lws_plat_file_ops * LWS_WARN_UNUSED_RESULT
lws_get_fops(struct lws_context *context);
LWS_VISIBLE LWS_EXTERN void
lws_set_fops(struct lws_context *context, const struct lws_plat_file_ops *fops);
/**
 * lws_vfs_tell() - get current file position
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN lws_filepos_t LWS_WARN_UNUSED_RESULT
lws_vfs_tell(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_get_length() - get current file total length in bytes
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN lws_filepos_t LWS_WARN_UNUSED_RESULT
lws_vfs_get_length(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_get_mod_time() - get time file last modified
 *
 * \param fop_fd: fop_fd we are asking about
 */
LWS_VISIBLE LWS_EXTERN uint32_t LWS_WARN_UNUSED_RESULT
lws_vfs_get_mod_time(lws_fop_fd_t fop_fd);
/**
 * lws_vfs_file_seek_set() - seek relative to start of file
 *
 * \param fop_fd: fop_fd we are seeking in
 * \param offset: offset from start of file
 */
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
lws_vfs_file_seek_set(lws_fop_fd_t fop_fd, lws_fileofs_t offset);
/**
 * lws_vfs_file_seek_end() - seek relative to end of file
 *
 * \param fop_fd: fop_fd we are seeking in
 * \param offset: offset from start of file
 */
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
lws_vfs_file_seek_end(lws_fop_fd_t fop_fd, lws_fileofs_t offset);

extern struct lws_plat_file_ops fops_zip;

/**
 * lws_plat_file_open() - open vfs filepath
 *
 * \param fops: file ops struct that applies to this descriptor
 * \param vfs_path: filename to open
 * \param flags: pointer to open flags
 *
 * The vfs_path is scanned for known fops signatures, and the open directed
 * to any matching fops open.
 *
 * User code should use this api to perform vfs opens.
 *
 * returns semi-opaque handle
 */
LWS_VISIBLE LWS_EXTERN lws_fop_fd_t LWS_WARN_UNUSED_RESULT
lws_vfs_file_open(const struct lws_plat_file_ops *fops, const char *vfs_path,
		  lws_fop_flags_t *flags);

/**
 * lws_plat_file_close() - close file
 *
 * \param fop_fd: file handle to close
 */
static LWS_INLINE int
lws_vfs_file_close(lws_fop_fd_t *fop_fd)
{
	return (*fop_fd)->fops->LWS_FOP_CLOSE(fop_fd);
}

/**
 * lws_plat_file_seek_cur() - close file
 *
 *
 * \param fop_fd: file handle
 * \param offset: position to seek to
 */
static LWS_INLINE lws_fileofs_t
lws_vfs_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	return fop_fd->fops->LWS_FOP_SEEK_CUR(fop_fd, offset);
}
/**
 * lws_plat_file_read() - read from file
 *
 * \param fop_fd: file handle
 * \param amount: how much to read (rewritten by call)
 * \param buf: buffer to write to
 * \param len: max length
 */
static LWS_INLINE int LWS_WARN_UNUSED_RESULT
lws_vfs_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		   uint8_t *buf, lws_filepos_t len)
{
	return fop_fd->fops->LWS_FOP_READ(fop_fd, amount, buf, len);
}
/**
 * lws_plat_file_write() - write from file
 *
 * \param fop_fd: file handle
 * \param amount: how much to write (rewritten by call)
 * \param buf: buffer to read from
 * \param len: max length
 */
static LWS_INLINE int LWS_WARN_UNUSED_RESULT
lws_vfs_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	return fop_fd->fops->LWS_FOP_WRITE(fop_fd, amount, buf, len);
}

/* these are the platform file operations implementations... they can
 * be called directly and used in fops arrays
 */

LWS_VISIBLE LWS_EXTERN lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_close(lws_fop_fd_t *fop_fd);
LWS_VISIBLE LWS_EXTERN lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len);
LWS_VISIBLE LWS_EXTERN int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		     uint8_t *buf, lws_filepos_t len);

LWS_VISIBLE LWS_EXTERN int
lws_alloc_vfs_file(struct lws_context *context, const char *filename,
		   uint8_t **buf, lws_filepos_t *amount);
//@}
