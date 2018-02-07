
#include <zlib.h>

#define DEFLATE_FRAME_COMPRESSION_LEVEL_SERVER 1
#define DEFLATE_FRAME_COMPRESSION_LEVEL_CLIENT Z_DEFAULT_COMPRESSION

enum arg_indexes {
	PMD_SERVER_NO_CONTEXT_TAKEOVER,
	PMD_CLIENT_NO_CONTEXT_TAKEOVER,
	PMD_SERVER_MAX_WINDOW_BITS,
	PMD_CLIENT_MAX_WINDOW_BITS,
	PMD_RX_BUF_PWR2,
	PMD_TX_BUF_PWR2,
	PMD_COMP_LEVEL,
	PMD_MEM_LEVEL,

	PMD_ARG_COUNT
};

struct lws_ext_pm_deflate_priv {
	z_stream rx;
	z_stream tx;

	unsigned char *buf_rx_inflated; /* RX inflated output buffer */
	unsigned char *buf_tx_deflated; /* TX deflated output buffer */

	size_t count_rx_between_fin;

	unsigned char args[PMD_ARG_COUNT];
	unsigned char tx_held[5];
	unsigned char rx_held;

	unsigned char tx_init:1;
	unsigned char rx_init:1;
	unsigned char compressed_out:1;
	unsigned char rx_held_valid:1;
	unsigned char tx_held_valid:1;
	unsigned char rx_append_trailer:1;
	unsigned char pending_tx_trailer:1;
};

