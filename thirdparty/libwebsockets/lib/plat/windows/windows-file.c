/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Copyright (C) 2010 - 2018 Andy Green <andy@warmcat.com>
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

#ifndef _WINSOCK_DEPRECATED_NO_WARNINGS
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#endif
#include "core/private.h"

int lws_plat_apply_FD_CLOEXEC(int n)
{
	return 0;
}

lws_fop_fd_t
_lws_plat_file_open(const struct lws_plat_file_ops *fops, const char *filename,
		    const char *vpath, lws_fop_flags_t *flags)
{
	HANDLE ret;
	WCHAR buf[MAX_PATH];
	lws_fop_fd_t fop_fd;
	FILE_STANDARD_INFO fInfo = {0};

	MultiByteToWideChar(CP_UTF8, 0, filename, -1, buf, LWS_ARRAY_SIZE(buf));

#if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0602 // Windows 8 (minimum when UWP_ENABLED, but can be used in Windows builds)
	CREATEFILE2_EXTENDED_PARAMETERS extParams = {0};
	extParams.dwFileAttributes = FILE_ATTRIBUTE_NORMAL;

	if (((*flags) & 7) == _O_RDONLY) {
		ret = CreateFile2(buf, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING, &extParams);
	} else {
		ret = CreateFile2(buf, GENERIC_WRITE, 0, CREATE_ALWAYS, &extParams);
	}
#else
	if (((*flags) & 7) == _O_RDONLY) {
		ret = CreateFileW(buf, GENERIC_READ, FILE_SHARE_READ,
			  NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	} else {
		ret = CreateFileW(buf, GENERIC_WRITE, 0, NULL,
			  CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
	}
#endif

	if (ret == LWS_INVALID_FILE)
		goto bail;

	fop_fd = malloc(sizeof(*fop_fd));
	if (!fop_fd)
		goto bail;

	fop_fd->fops = fops;
	fop_fd->fd = ret;
	fop_fd->filesystem_priv = NULL; /* we don't use it */
	fop_fd->flags = *flags;
	fop_fd->len = 0;
	if(GetFileInformationByHandleEx(ret, FileStandardInfo, &fInfo, sizeof(fInfo)))
		fop_fd->len = fInfo.EndOfFile.QuadPart;

	fop_fd->pos = 0;

	return fop_fd;

bail:
	return NULL;
}

int
_lws_plat_file_close(lws_fop_fd_t *fop_fd)
{
	HANDLE fd = (*fop_fd)->fd;

	free(*fop_fd);
	*fop_fd = NULL;

	CloseHandle((HANDLE)fd);

	return 0;
}

lws_fileofs_t
_lws_plat_file_seek_cur(lws_fop_fd_t fop_fd, lws_fileofs_t offset)
{
	LARGE_INTEGER l;

	l.QuadPart = offset;
	return SetFilePointerEx((HANDLE)fop_fd->fd, l, NULL, FILE_CURRENT);
}

int
_lws_plat_file_read(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
		    uint8_t *buf, lws_filepos_t len)
{
	DWORD _amount;

	if (!ReadFile((HANDLE)fop_fd->fd, buf, (DWORD)len, &_amount, NULL)) {
		*amount = 0;

		return 1;
	}

	fop_fd->pos += _amount;
	*amount = (unsigned long)_amount;

	return 0;
}

int
_lws_plat_file_write(lws_fop_fd_t fop_fd, lws_filepos_t *amount,
			 uint8_t* buf, lws_filepos_t len)
{
	DWORD _amount;

	if (!WriteFile((HANDLE)fop_fd->fd, buf, (DWORD)len, &_amount, NULL)) {
		*amount = 0;

		return 1;
	}

	fop_fd->pos += _amount;
	*amount = (unsigned long)_amount;

	return 0;
}


int
lws_plat_write_cert(struct lws_vhost *vhost, int is_key, int fd, void *buf,
			int len)
{
	int n;

	n = write(fd, buf, len);

	lseek(fd, 0, SEEK_SET);

	return n != len;
}

int
lws_plat_write_file(const char *filename, void *buf, int len)
{
	int m, fd;

	fd = lws_open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0600);

	if (fd == -1)
		return -1;

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

