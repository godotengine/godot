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
 */

#define _GNU_SOURCE
#include "core/private.h"

#include <pwd.h>
#include <grp.h>

#ifdef LWS_WITH_PLUGINS
#include <dlfcn.h>
#endif
#include <dirent.h>

int lws_plat_apply_FD_CLOEXEC(int n)
{
	if (n == -1)
		return 0;

	return fcntl(n, F_SETFD, FD_CLOEXEC);
}

int
lws_plat_write_file(const char *filename, void *buf, int len)
{
	int m, fd;

	fd = lws_open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0600);

	if (fd == -1)
		return 1;

	m = write(fd, buf, len);
	close(fd);

	return m != len;
}

int
lws_plat_read_file(const char *filename, void *buf, int len)
{
	int n, fd = lws_open(filename, O_RDONLY);
	if (fd == -1)
		return -1;

	n = read(fd, buf, len);
	close(fd);

	return n;
}

lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags)
{
	struct stat stat_buf;
	int ret = lws_open(filename, (*flags) & LWS_FOP_FLAGS_MASK, 0664);
	lws_fop_fd_t fop_fd;

	if (ret < 0)
		return NULL;

	if (fstat(ret, &stat_buf) < 0)
		goto bail;

	fop_fd = malloc(sizeof(*fop_fd));
	if (!fop_fd)
		goto bail;

	fop_fd->fops = fops;
	fop_fd->flags = *flags;
	fop_fd->fd = ret;
	fop_fd->filesystem_priv = NULL; /* we don't use it */
	fop_fd->len = stat_buf.st_size;
	fop_fd->pos = 0;

	return fop_fd;

bail:
	close(ret);
	return NULL;
}

int
_lws_plat_file_close(lws_fop_fd_t *fop_fd)
{
	int fd = (*fop_fd)->fd;

	free(*fop_fd);
	*fop_fd = NULL;

	return close(fd);
}

lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	lws_fileofs_t r;

	if (offset > 0 &&
	    offset > (lws_fileofs_t)fop_fd->len - (lws_fileofs_t)fop_fd->pos)
		offset = fop_fd->len - fop_fd->pos;

	if ((lws_fileofs_t)fop_fd->pos + offset < 0)
		offset = -fop_fd->pos;

	r = lseek(fop_fd->fd, offset, SEEK_CUR);

	if (r >= 0)
		fop_fd->pos = r;
	else
		lwsl_err("error seeking from cur %ld, offset %ld\n",
                        (long)fop_fd->pos, (long)offset);

	return r;
}

int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	long n;

	n = read((int)fop_fd->fd, buf, len);
	if (n == -1) {
		*amount = 0;
		return -1;
	}
	fop_fd->pos += n;
	lwsl_debug("%s: read %ld of req %ld, pos %ld, len %ld\n", __func__, n,
                  (long)len, (long)fop_fd->pos, (long)fop_fd->len);
	*amount = n;

	return 0;
}

int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		     uint8_t *buf, lws_filepos_t len)
{
	long n;

	n = write((int)fop_fd->fd, buf, len);
	if (n == -1) {
		*amount = 0;
		return -1;
	}

	fop_fd->pos += n;
	*amount = n;

	return 0;
}

