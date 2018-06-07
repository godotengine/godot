/*
 * libwebsockets - small server side websockets and web server implementation
 *
 * Original code used in this source file:
 *
 * https://github.com/PerBothner/DomTerm.git @912add15f3d0aec
 *
 * ./lws-term/io.c
 * ./lws-term/junzip.c
 *
 * Copyright (C) 2017  Per Bothner <per@bothner.com>
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * ( copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * lws rewrite:
 *
 * Copyright (C) 2017  Andy Green <andy@warmcat.com>
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

#include "private-libwebsockets.h"

#include <zlib.h>

/*
 * This code works with zip format containers which may have files compressed
 * with gzip deflate (type 8) or store uncompressed (type 0).
 *
 * Linux zip produces such zipfiles by default, eg
 *
 *  $ zip ../myzip.zip file1 file2 file3
 */

#define ZIP_COMPRESSION_METHOD_STORE 0
#define ZIP_COMPRESSION_METHOD_DEFLATE 8

typedef struct {
	lws_filepos_t		filename_start;
	uint32_t		crc32;
	uint32_t		comp_size;
	uint32_t		uncomp_size;
	uint32_t		offset;
	uint32_t		mod_time;
	uint16_t		filename_len;
	uint16_t		extra;
	uint16_t		method;
	uint16_t		file_com_len;
} lws_fops_zip_hdr_t;

typedef struct {
	struct lws_fop_fd	fop_fd; /* MUST BE FIRST logical fop_fd into
	 	 	 	 	 * file inside zip: fops_zip fops */
	lws_fop_fd_t		zip_fop_fd; /* logical fop fd on to zip file
	 	 	 	 	     * itself: using platform fops */
	lws_fops_zip_hdr_t	hdr;
	z_stream		inflate;
	lws_filepos_t		content_start;
	lws_filepos_t		exp_uncomp_pos;
	union {
		uint8_t		trailer8[8];
		uint32_t	trailer32[2];
	} u;
	uint8_t			rbuf[128]; /* decompression chunk size */
	int			entry_count;

	unsigned int		decompress:1; /* 0 = direct from file */
	unsigned int		add_gzip_container:1;
} *lws_fops_zip_t;

struct lws_plat_file_ops fops_zip;
#define fop_fd_to_priv(FD) ((lws_fops_zip_t)(FD))

static const uint8_t hd[] = { 31, 139, 8, 0, 0, 0, 0, 0, 0, 3 };

enum {
	ZC_SIGNATURE				= 0,
	ZC_VERSION_MADE_BY 			= 4,
	ZC_VERSION_NEEDED_TO_EXTRACT 		= 6,
	ZC_GENERAL_PURPOSE_BIT_FLAG 		= 8,
	ZC_COMPRESSION_METHOD 			= 10,
	ZC_LAST_MOD_FILE_TIME 			= 12,
	ZC_LAST_MOD_FILE_DATE 			= 14,
	ZC_CRC32 				= 16,
	ZC_COMPRESSED_SIZE 			= 20,
	ZC_UNCOMPRESSED_SIZE 			= 24,
	ZC_FILE_NAME_LENGTH 			= 28,
	ZC_EXTRA_FIELD_LENGTH 			= 30,

	ZC_FILE_COMMENT_LENGTH 			= 32,
	ZC_DISK_NUMBER_START 			= 34,
	ZC_INTERNAL_FILE_ATTRIBUTES 		= 36,
	ZC_EXTERNAL_FILE_ATTRIBUTES 		= 38,
	ZC_REL_OFFSET_LOCAL_HEADER 		= 42,
	ZC_DIRECTORY_LENGTH 			= 46,

	ZE_SIGNATURE_OFFSET 			= 0,
	ZE_DESK_NUMBER 				= 4,
	ZE_CENTRAL_DIRECTORY_DISK_NUMBER 	= 6,
	ZE_NUM_ENTRIES_THIS_DISK 		= 8,
	ZE_NUM_ENTRIES 				= 10,
	ZE_CENTRAL_DIRECTORY_SIZE 		= 12,
	ZE_CENTRAL_DIR_OFFSET 			= 16,
	ZE_ZIP_COMMENT_LENGTH 			= 20,
	ZE_DIRECTORY_LENGTH 			= 22,

	ZL_REL_OFFSET_CONTENT			= 28,
	ZL_HEADER_LENGTH			= 30,

	LWS_FZ_ERR_SEEK_END_RECORD		= 1,
	LWS_FZ_ERR_READ_END_RECORD,
	LWS_FZ_ERR_END_RECORD_MAGIC,
	LWS_FZ_ERR_END_RECORD_SANITY,
	LWS_FZ_ERR_CENTRAL_SEEK,
	LWS_FZ_ERR_CENTRAL_READ,
	LWS_FZ_ERR_CENTRAL_SANITY,
	LWS_FZ_ERR_NAME_TOO_LONG,
	LWS_FZ_ERR_NAME_SEEK,
	LWS_FZ_ERR_NAME_READ,
	LWS_FZ_ERR_CONTENT_SANITY,
	LWS_FZ_ERR_CONTENT_SEEK,
	LWS_FZ_ERR_SCAN_SEEK,
	LWS_FZ_ERR_NOT_FOUND,
	LWS_FZ_ERR_ZLIB_INIT,
	LWS_FZ_ERR_READ_CONTENT,
	LWS_FZ_ERR_SEEK_COMPRESSED,
};

static uint16_t
get_u16(void *p)
{
	const uint8_t *c = (const uint8_t *)p;

	return (uint16_t)((c[0] | (c[1] << 8)));
}

static uint32_t
get_u32(void *p)
{
	const uint8_t *c = (const uint8_t *)p;

	return (uint32_t)((c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24)));
}

int
lws_fops_zip_scan(lws_fops_zip_t priv, const char *name, int len)
{
	lws_filepos_t amount;
	uint8_t buf[96];
	int i;

	if (lws_vfs_file_seek_end(priv->zip_fop_fd, -ZE_DIRECTORY_LENGTH) < 0)
		return LWS_FZ_ERR_SEEK_END_RECORD;

	if (lws_vfs_file_read(priv->zip_fop_fd, &amount, buf,
			      ZE_DIRECTORY_LENGTH))
		return LWS_FZ_ERR_READ_END_RECORD;

	if (amount != ZE_DIRECTORY_LENGTH)
		return LWS_FZ_ERR_READ_END_RECORD;

	/*
	 * We require the zip to have the last record right at the end
	 * Linux zip always does this if no zip comment.
	 */
	if (buf[0] != 'P' || buf[1] != 'K' || buf[2] != 5 || buf[3] != 6)
		return LWS_FZ_ERR_END_RECORD_MAGIC;

	i = get_u16(buf + ZE_NUM_ENTRIES);

	if (get_u16(buf + ZE_DESK_NUMBER) ||
	    get_u16(buf + ZE_CENTRAL_DIRECTORY_DISK_NUMBER) ||
	    i != get_u16(buf + ZE_NUM_ENTRIES_THIS_DISK))
		return LWS_FZ_ERR_END_RECORD_SANITY;

	/* end record is OK... look for our file in the central dir */

	if (lws_vfs_file_seek_set(priv->zip_fop_fd,
				  get_u32(buf + ZE_CENTRAL_DIR_OFFSET)) < 0)
		return LWS_FZ_ERR_CENTRAL_SEEK;

	while (i--) {
		priv->content_start = lws_vfs_tell(priv->zip_fop_fd);

		if (lws_vfs_file_read(priv->zip_fop_fd, &amount, buf,
				      ZC_DIRECTORY_LENGTH))
			return LWS_FZ_ERR_CENTRAL_READ;

		if (amount != ZC_DIRECTORY_LENGTH)
			return LWS_FZ_ERR_CENTRAL_READ;

		if (get_u32(buf + ZC_SIGNATURE) != 0x02014B50)
			return LWS_FZ_ERR_CENTRAL_SANITY;

               lwsl_debug("cstart 0x%lx\n", (unsigned long)priv->content_start);

		priv->hdr.filename_len = get_u16(buf + ZC_FILE_NAME_LENGTH);
		priv->hdr.extra = get_u16(buf + ZC_EXTRA_FIELD_LENGTH);
		priv->hdr.filename_start = lws_vfs_tell(priv->zip_fop_fd);

		priv->hdr.method = get_u16(buf + ZC_COMPRESSION_METHOD);
		priv->hdr.crc32 = get_u32(buf + ZC_CRC32);
		priv->hdr.comp_size = get_u32(buf + ZC_COMPRESSED_SIZE);
		priv->hdr.uncomp_size = get_u32(buf + ZC_UNCOMPRESSED_SIZE);
		priv->hdr.offset = get_u32(buf + ZC_REL_OFFSET_LOCAL_HEADER);
		priv->hdr.mod_time = get_u32(buf + ZC_LAST_MOD_FILE_TIME);
		priv->hdr.file_com_len = get_u16(buf + ZC_FILE_COMMENT_LENGTH);

		if (priv->hdr.filename_len != len)
			goto next;

		if (len >= sizeof(buf) - 1)
			return LWS_FZ_ERR_NAME_TOO_LONG;

		if (priv->zip_fop_fd->fops->LWS_FOP_READ(priv->zip_fop_fd,
							&amount, buf, len))
			return LWS_FZ_ERR_NAME_READ;
		if (amount != len)
			return LWS_FZ_ERR_NAME_READ;

		buf[len] = '\0';
		lwsl_debug("check %s vs %s\n", buf, name);

		if (strcmp((const char *)buf, name))
			goto next;

		/* we found a match */
		if (lws_vfs_file_seek_set(priv->zip_fop_fd, priv->hdr.offset) < 0)
			return LWS_FZ_ERR_NAME_SEEK;
		if (priv->zip_fop_fd->fops->LWS_FOP_READ(priv->zip_fop_fd,
							&amount, buf,
							ZL_HEADER_LENGTH))
			return LWS_FZ_ERR_NAME_READ;
		if (amount != ZL_HEADER_LENGTH)
			return LWS_FZ_ERR_NAME_READ;

		priv->content_start = priv->hdr.offset +
				      ZL_HEADER_LENGTH +
				      priv->hdr.filename_len +
				      get_u16(buf + ZL_REL_OFFSET_CONTENT);

		lwsl_debug("content supposed to start at 0x%lx\n",
                          (unsigned long)priv->content_start);

		if (priv->content_start > priv->zip_fop_fd->len)
			return LWS_FZ_ERR_CONTENT_SANITY;

		if (lws_vfs_file_seek_set(priv->zip_fop_fd,
					  priv->content_start) < 0)
			return LWS_FZ_ERR_CONTENT_SEEK;

		/* we are aligned at the start of the content */

		priv->exp_uncomp_pos = 0;

		return 0;

next:
		if (i && lws_vfs_file_seek_set(priv->zip_fop_fd,
					       priv->content_start +
					       ZC_DIRECTORY_LENGTH +
					       priv->hdr.filename_len +
					       priv->hdr.extra +
					       priv->hdr.file_com_len) < 0)
			return LWS_FZ_ERR_SCAN_SEEK;
	}

	return LWS_FZ_ERR_NOT_FOUND;
}

static int
lws_fops_zip_reset_inflate(lws_fops_zip_t priv)
{
	if (priv->decompress)
		inflateEnd(&priv->inflate);

	priv->inflate.zalloc = Z_NULL;
	priv->inflate.zfree = Z_NULL;
	priv->inflate.opaque = Z_NULL;
	priv->inflate.avail_in = 0;
	priv->inflate.next_in = Z_NULL;

	if (inflateInit2(&priv->inflate, -MAX_WBITS) != Z_OK) {
		lwsl_err("inflate init failed\n");
		return LWS_FZ_ERR_ZLIB_INIT;
	}

	if (lws_vfs_file_seek_set(priv->zip_fop_fd, priv->content_start) < 0)
		return LWS_FZ_ERR_CONTENT_SEEK;

	priv->exp_uncomp_pos = 0;

	return 0;
}

static lws_fop_fd_t
lws_fops_zip_open(const struct lws_plat_file_ops *fops, const char *vfs_path,
		  const char *vpath, lws_fop_flags_t *flags)
{
	lws_fop_flags_t local_flags = 0;
	lws_fops_zip_t priv;
	char rp[192];
	int m;

	/*
	 * vpath points at the / after the fops signature in vfs_path, eg
	 * with a vfs_path "/var/www/docs/manual.zip/index.html", vpath
	 * will come pointing at "/index.html"
	 */

	priv = lws_zalloc(sizeof(*priv), "fops_zip priv");
	if (!priv)
		return NULL;

	priv->fop_fd.fops = &fops_zip;

	m = sizeof(rp) - 1;
	if ((vpath - vfs_path - 1) < m)
		m = vpath - vfs_path - 1;
	strncpy(rp, vfs_path, m);
	rp[m] = '\0';

	/* open the zip file itself using the incoming fops, not fops_zip */

	priv->zip_fop_fd = fops->LWS_FOP_OPEN(fops, rp, NULL, &local_flags);
	if (!priv->zip_fop_fd) {
		lwsl_err("unable to open zip %s\n", rp);
		goto bail1;
	}

	if (*vpath == '/')
		vpath++;

	m = lws_fops_zip_scan(priv, vpath, strlen(vpath));
	if (m) {
		lwsl_err("unable to find record matching '%s' %d\n", vpath, m);
		goto bail2;
	}

	/* the directory metadata tells us modification time, so pass it on */
	priv->fop_fd.mod_time = priv->hdr.mod_time;
	*flags |= LWS_FOP_FLAG_MOD_TIME_VALID | LWS_FOP_FLAG_VIRTUAL;
	priv->fop_fd.flags = *flags;

	/* The zip fop_fd is left pointing at the start of the content.
	 *
	 * 1) Content could be uncompressed (STORE), and we can always serve
	 *    that directly
	 *
	 * 2) Content could be compressed (GZIP), and the client can handle
	 *    receiving GZIP... we can wrap it in a GZIP header and trailer
	 *    and serve the content part directly.  The flag indicating we
	 *    are providing GZIP directly is set so lws will send the right
	 *    headers.
	 *
	 * 3) Content could be compressed (GZIP) but the client can't handle
	 *    receiving GZIP... we can decompress it and serve as it is
	 *    inflated piecemeal.
	 *
	 * 4) Content may be compressed some unknown way... fail
	 *
	 */
	if (priv->hdr.method == ZIP_COMPRESSION_METHOD_STORE) {
		/*
		 * it is stored uncompressed, leave it indicated as
		 * uncompressed, and just serve it from inside the
		 * zip with no gzip container;
		 */

		lwsl_info("direct zip serving (stored)\n");

		priv->fop_fd.len = priv->hdr.uncomp_size;

		return &priv->fop_fd;
	}

	if ((*flags & LWS_FOP_FLAG_COMPR_ACCEPTABLE_GZIP) &&
	    priv->hdr.method == ZIP_COMPRESSION_METHOD_DEFLATE) {

		/*
		 * We can serve the gzipped file contents directly as gzip
		 * from inside the zip container; client says it is OK.
		 *
		 * To convert to standalone gzip, we have to add a 10-byte
		 * constant header and a variable 8-byte trailer around the
		 * content.
		 *
		 * The 8-byte trailer is prepared now and held in the priv.
		 */

		lwsl_info("direct zip serving (gzipped)\n");

		priv->fop_fd.len = sizeof(hd) + priv->hdr.comp_size +
				   sizeof(priv->u);

		if (lws_is_be()) {
			uint8_t *p = priv->u.trailer8;

			*p++ = (uint8_t)priv->hdr.crc32;
			*p++ = (uint8_t)(priv->hdr.crc32 >> 8);
			*p++ = (uint8_t)(priv->hdr.crc32 >> 16);
			*p++ = (uint8_t)(priv->hdr.crc32 >> 24);
			*p++ = (uint8_t)priv->hdr.uncomp_size;
			*p++ = (uint8_t)(priv->hdr.uncomp_size >> 8);
			*p++ = (uint8_t)(priv->hdr.uncomp_size >> 16);
			*p   = (uint8_t)(priv->hdr.uncomp_size >> 24);
		} else {
			priv->u.trailer32[0] = priv->hdr.crc32;
			priv->u.trailer32[1] = priv->hdr.uncomp_size;
		}

		*flags |= LWS_FOP_FLAG_COMPR_IS_GZIP;
		priv->fop_fd.flags = *flags;
		priv->add_gzip_container = 1;

		return &priv->fop_fd;
	}

	if (priv->hdr.method == ZIP_COMPRESSION_METHOD_DEFLATE) {

		/* we must decompress it to serve it */

		lwsl_info("decompressed zip serving\n");

		priv->fop_fd.len = priv->hdr.uncomp_size;

		if (lws_fops_zip_reset_inflate(priv)) {
			lwsl_err("inflate init failed\n");
			goto bail2;
		}

		priv->decompress = 1;

		return &priv->fop_fd;
	}

	/* we can't handle it ... */

	lwsl_err("zipped file %s compressed in unknown way (%d)\n", vfs_path,
		 priv->hdr.method);

bail2:
	lws_vfs_file_close(&priv->zip_fop_fd);
bail1:
	free(priv);

	return NULL;
}

/* ie, we are closing the fop_fd for the file inside the gzip */

static int
lws_fops_zip_close(lws_fop_fd_t *fd)
{
	lws_fops_zip_t priv = fop_fd_to_priv(*fd);

	if (priv->decompress)
		inflateEnd(&priv->inflate);

	lws_vfs_file_close(&priv->zip_fop_fd); /* close the gzip fop_fd */

	free(priv);
	*fd = NULL;

	return 0;
}

static lws_fileofs_t
lws_fops_zip_seek_cur(lws_fop_fd_t fd, lws_fileofs_t offset_from_cur_pos)
{
	fd->pos += offset_from_cur_pos;

	return fd->pos;
}

static int
lws_fops_zip_read(lws_fop_fd_t fd, lws_filepos_t *amount, uint8_t *buf,
		  lws_filepos_t len)
{
	lws_fops_zip_t priv = fop_fd_to_priv(fd);
	lws_filepos_t ramount, rlen, cur = lws_vfs_tell(fd);
	int ret;

	if (priv->decompress) {

		if (priv->exp_uncomp_pos != fd->pos) {
			/*
			 *  there has been a seek in the uncompressed fop_fd
			 * we have to restart the decompression and loop eating
			 * the decompressed data up to the seek point
			 */
			lwsl_info("seek in decompressed\n");

			lws_fops_zip_reset_inflate(priv);

			while (priv->exp_uncomp_pos != fd->pos) {
				rlen = len;
				if (rlen > fd->pos - priv->exp_uncomp_pos)
					rlen = fd->pos - priv->exp_uncomp_pos;
				if (lws_fops_zip_read(fd, amount, buf, rlen))
					return LWS_FZ_ERR_SEEK_COMPRESSED;
			}
			*amount = 0;
		}

		priv->inflate.avail_out = (unsigned int)len;
		priv->inflate.next_out = buf;

spin:
		if (!priv->inflate.avail_in) {
			rlen = sizeof(priv->rbuf);
			if (rlen > priv->hdr.comp_size -
				   (cur - priv->content_start))
				rlen = priv->hdr.comp_size -
				       (priv->hdr.comp_size -
					priv->content_start);

			if (priv->zip_fop_fd->fops->LWS_FOP_READ(
					priv->zip_fop_fd, &ramount, priv->rbuf,
					rlen))
				return LWS_FZ_ERR_READ_CONTENT;

			cur += ramount;

			priv->inflate.avail_in = (unsigned int)ramount;
			priv->inflate.next_in = priv->rbuf;
		}

		ret = inflate(&priv->inflate, Z_NO_FLUSH);
		if (ret == Z_STREAM_ERROR)
			return ret;

		switch (ret) {
		case Z_NEED_DICT:
			ret = Z_DATA_ERROR;
			/* and fall through */
		case Z_DATA_ERROR:
		case Z_MEM_ERROR:

			return ret;
		}

		if (!priv->inflate.avail_in && priv->inflate.avail_out &&
		     cur != priv->content_start + priv->hdr.comp_size)
			goto spin;

		*amount = len - priv->inflate.avail_out;

		priv->exp_uncomp_pos += *amount;
		fd->pos += *amount;

		return 0;
	}

	if (priv->add_gzip_container) {

		lwsl_info("%s: gzip + container\n", __func__);
		*amount = 0;

		/* place the canned header at the start */

		if (len && fd->pos < sizeof(hd)) {
			rlen = sizeof(hd) - fd->pos;
			if (rlen > len)
				rlen = len;
			/* provide stuff from canned header */
			memcpy(buf, hd + fd->pos, (size_t)rlen);
			fd->pos += rlen;
			buf += rlen;
			len -= rlen;
			*amount += rlen;
		}

		/* serve gzipped data direct from zipfile */

		if (len && fd->pos >= sizeof(hd) &&
		    fd->pos < priv->hdr.comp_size + sizeof(hd)) {

			rlen = priv->hdr.comp_size - (priv->zip_fop_fd->pos -
						      priv->content_start);
			if (rlen > len)
				rlen = len;

			if (rlen &&
			    priv->zip_fop_fd->pos < (priv->hdr.comp_size +
					    	     priv->content_start)) {
				if (lws_vfs_file_read(priv->zip_fop_fd,
						      &ramount, buf, rlen))
					return LWS_FZ_ERR_READ_CONTENT;
				*amount += ramount;
				fd->pos += ramount; // virtual pos
				buf += ramount;
				len -= ramount;
			}
		}

		/* place the prepared trailer at the end */

		if (len && fd->pos >= priv->hdr.comp_size + sizeof(hd) &&
		    fd->pos < priv->hdr.comp_size + sizeof(hd) +
		    	      sizeof(priv->u)) {
			cur = fd->pos - priv->hdr.comp_size - sizeof(hd);
			rlen = sizeof(priv->u) - cur;
			if (rlen > len)
				rlen = len;

			memcpy(buf, priv->u.trailer8 + cur, (size_t)rlen);

			*amount += rlen;
			fd->pos += rlen;
		}

		return 0;
	}

	lwsl_info("%s: store\n", __func__);

	if (len > priv->hdr.uncomp_size - (cur - priv->content_start))
		len = priv->hdr.comp_size - (priv->hdr.comp_size -
					     priv->content_start);

	if (priv->zip_fop_fd->fops->LWS_FOP_READ(priv->zip_fop_fd,
						 amount, buf, len))
		return LWS_FZ_ERR_READ_CONTENT;

	return 0;
}

struct lws_plat_file_ops fops_zip = {
	lws_fops_zip_open,
	lws_fops_zip_close,
	lws_fops_zip_seek_cur,
	lws_fops_zip_read,
	NULL,
	{ { ".zip/", 5 }, { ".jar/", 5 }, { ".war/", 5 } },
	NULL,
};
