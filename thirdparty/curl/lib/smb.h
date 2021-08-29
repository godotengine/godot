#ifndef HEADER_CURL_SMB_H
#define HEADER_CURL_SMB_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) 2014, Bill Nagel <wnagel@tycoint.com>, Exacq Technologies
 * Copyright (C) 2018 - 2021, Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

enum smb_conn_state {
  SMB_NOT_CONNECTED = 0,
  SMB_CONNECTING,
  SMB_NEGOTIATE,
  SMB_SETUP,
  SMB_CONNECTED
};

struct smb_conn {
  enum smb_conn_state state;
  char *user;
  char *domain;
  char *share;
  unsigned char challenge[8];
  unsigned int session_key;
  unsigned short uid;
  char *recv_buf;
  size_t upload_size;
  size_t send_size;
  size_t sent;
  size_t got;
};

/*
 * Definitions for SMB protocol data structures
 */
#ifdef BUILDING_CURL_SMB_C

#if defined(_MSC_VER) || defined(__ILEC400__)
#  define PACK
#  pragma pack(push)
#  pragma pack(1)
#elif defined(__GNUC__)
#  define PACK __attribute__((packed))
#else
#  define PACK
#endif

#define SMB_COM_CLOSE                 0x04
#define SMB_COM_READ_ANDX             0x2e
#define SMB_COM_WRITE_ANDX            0x2f
#define SMB_COM_TREE_DISCONNECT       0x71
#define SMB_COM_NEGOTIATE             0x72
#define SMB_COM_SETUP_ANDX            0x73
#define SMB_COM_TREE_CONNECT_ANDX     0x75
#define SMB_COM_NT_CREATE_ANDX        0xa2
#define SMB_COM_NO_ANDX_COMMAND       0xff

#define SMB_WC_CLOSE                  0x03
#define SMB_WC_READ_ANDX              0x0c
#define SMB_WC_WRITE_ANDX             0x0e
#define SMB_WC_SETUP_ANDX             0x0d
#define SMB_WC_TREE_CONNECT_ANDX      0x04
#define SMB_WC_NT_CREATE_ANDX         0x18

#define SMB_FLAGS_CANONICAL_PATHNAMES 0x10
#define SMB_FLAGS_CASELESS_PATHNAMES  0x08
#define SMB_FLAGS2_UNICODE_STRINGS    0x8000
#define SMB_FLAGS2_IS_LONG_NAME       0x0040
#define SMB_FLAGS2_KNOWS_LONG_NAME    0x0001

#define SMB_CAP_LARGE_FILES           0x08
#define SMB_GENERIC_WRITE             0x40000000
#define SMB_GENERIC_READ              0x80000000
#define SMB_FILE_SHARE_ALL            0x07
#define SMB_FILE_OPEN                 0x01
#define SMB_FILE_OVERWRITE_IF         0x05

#define SMB_ERR_NOACCESS              0x00050001

struct smb_header {
  unsigned char nbt_type;
  unsigned char nbt_flags;
  unsigned short nbt_length;
  unsigned char magic[4];
  unsigned char command;
  unsigned int status;
  unsigned char flags;
  unsigned short flags2;
  unsigned short pid_high;
  unsigned char signature[8];
  unsigned short pad;
  unsigned short tid;
  unsigned short pid;
  unsigned short uid;
  unsigned short mid;
} PACK;

struct smb_negotiate_response {
  struct smb_header h;
  unsigned char word_count;
  unsigned short dialect_index;
  unsigned char security_mode;
  unsigned short max_mpx_count;
  unsigned short max_number_vcs;
  unsigned int max_buffer_size;
  unsigned int max_raw_size;
  unsigned int session_key;
  unsigned int capabilities;
  unsigned int system_time_low;
  unsigned int system_time_high;
  unsigned short server_time_zone;
  unsigned char encryption_key_length;
  unsigned short byte_count;
  char bytes[1];
} PACK;

struct andx {
  unsigned char command;
  unsigned char pad;
  unsigned short offset;
} PACK;

struct smb_setup {
  unsigned char word_count;
  struct andx andx;
  unsigned short max_buffer_size;
  unsigned short max_mpx_count;
  unsigned short vc_number;
  unsigned int session_key;
  unsigned short lengths[2];
  unsigned int pad;
  unsigned int capabilities;
  unsigned short byte_count;
  char bytes[1024];
} PACK;

struct smb_tree_connect {
  unsigned char word_count;
  struct andx andx;
  unsigned short flags;
  unsigned short pw_len;
  unsigned short byte_count;
  char bytes[1024];
} PACK;

struct smb_nt_create {
  unsigned char word_count;
  struct andx andx;
  unsigned char pad;
  unsigned short name_length;
  unsigned int flags;
  unsigned int root_fid;
  unsigned int access;
  curl_off_t allocation_size;
  unsigned int ext_file_attributes;
  unsigned int share_access;
  unsigned int create_disposition;
  unsigned int create_options;
  unsigned int impersonation_level;
  unsigned char security_flags;
  unsigned short byte_count;
  char bytes[1024];
} PACK;

struct smb_nt_create_response {
  struct smb_header h;
  unsigned char word_count;
  struct andx andx;
  unsigned char op_lock_level;
  unsigned short fid;
  unsigned int create_disposition;

  curl_off_t create_time;
  curl_off_t last_access_time;
  curl_off_t last_write_time;
  curl_off_t last_change_time;
  unsigned int ext_file_attributes;
  curl_off_t allocation_size;
  curl_off_t end_of_file;
} PACK;

struct smb_read {
  unsigned char word_count;
  struct andx andx;
  unsigned short fid;
  unsigned int offset;
  unsigned short max_bytes;
  unsigned short min_bytes;
  unsigned int timeout;
  unsigned short remaining;
  unsigned int offset_high;
  unsigned short byte_count;
} PACK;

struct smb_write {
  struct smb_header h;
  unsigned char word_count;
  struct andx andx;
  unsigned short fid;
  unsigned int offset;
  unsigned int timeout;
  unsigned short write_mode;
  unsigned short remaining;
  unsigned short pad;
  unsigned short data_length;
  unsigned short data_offset;
  unsigned int offset_high;
  unsigned short byte_count;
  unsigned char pad2;
} PACK;

struct smb_close {
  unsigned char word_count;
  unsigned short fid;
  unsigned int last_mtime;
  unsigned short byte_count;
} PACK;

struct smb_tree_disconnect {
  unsigned char word_count;
  unsigned short byte_count;
} PACK;

#if defined(_MSC_VER) || defined(__ILEC400__)
#  pragma pack(pop)
#endif

#endif /* BUILDING_CURL_SMB_C */

#if !defined(CURL_DISABLE_SMB) && defined(USE_CURL_NTLM_CORE) && \
    (SIZEOF_CURL_OFF_T > 4)

extern const struct Curl_handler Curl_handler_smb;
extern const struct Curl_handler Curl_handler_smbs;

#endif /* CURL_DISABLE_SMB && USE_CURL_NTLM_CORE &&
          SIZEOF_CURL_OFF_T > 4 */

#endif /* HEADER_CURL_SMB_H */
