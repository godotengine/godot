// jpgd.h - C++ class for JPEG decompression.
// Public domain, Rich Geldreich <richgel99@gmail.com>
#ifndef JPEG_DECODER_H
#define JPEG_DECODER_H

#include <stdlib.h>
#include <stdio.h>
#include <setjmp.h>
#include <assert.h>
#include <stdint.h>

#ifdef _MSC_VER
#define JPGD_NORETURN __declspec(noreturn) 
#elif defined(__GNUC__)
#define JPGD_NORETURN __attribute__ ((noreturn))
#else
#define JPGD_NORETURN
#endif

#define JPGD_HUFF_TREE_MAX_LENGTH 512
#define JPGD_HUFF_CODE_SIZE_MAX_LENGTH 256

namespace jpgd
{
	typedef unsigned char  uint8;
	typedef   signed short int16;
	typedef unsigned short uint16;
	typedef unsigned int   uint;
	typedef   signed int   int32;

	// Loads a JPEG image from a memory buffer or a file.
	// req_comps can be 1 (grayscale), 3 (RGB), or 4 (RGBA).
	// On return, width/height will be set to the image's dimensions, and actual_comps will be set to the either 1 (grayscale) or 3 (RGB).
	// Notes: For more control over where and how the source data is read, see the decompress_jpeg_image_from_stream() function below, or call the jpeg_decoder class directly.
	// Requesting a 8 or 32bpp image is currently a little faster than 24bpp because the jpeg_decoder class itself currently always unpacks to either 8 or 32bpp.
	unsigned char* decompress_jpeg_image_from_memory(const unsigned char* pSrc_data, int src_data_size, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags = 0);
	unsigned char* decompress_jpeg_image_from_file(const char* pSrc_filename, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags = 0);

	// Success/failure error codes.
	enum jpgd_status
	{
		JPGD_SUCCESS = 0, JPGD_FAILED = -1, JPGD_DONE = 1,
		JPGD_BAD_DHT_COUNTS = -256, JPGD_BAD_DHT_INDEX, JPGD_BAD_DHT_MARKER, JPGD_BAD_DQT_MARKER, JPGD_BAD_DQT_TABLE,
		JPGD_BAD_PRECISION, JPGD_BAD_HEIGHT, JPGD_BAD_WIDTH, JPGD_TOO_MANY_COMPONENTS,
		JPGD_BAD_SOF_LENGTH, JPGD_BAD_VARIABLE_MARKER, JPGD_BAD_DRI_LENGTH, JPGD_BAD_SOS_LENGTH,
		JPGD_BAD_SOS_COMP_ID, JPGD_W_EXTRA_BYTES_BEFORE_MARKER, JPGD_NO_ARITHMITIC_SUPPORT, JPGD_UNEXPECTED_MARKER,
		JPGD_NOT_JPEG, JPGD_UNSUPPORTED_MARKER, JPGD_BAD_DQT_LENGTH, JPGD_TOO_MANY_BLOCKS,
		JPGD_UNDEFINED_QUANT_TABLE, JPGD_UNDEFINED_HUFF_TABLE, JPGD_NOT_SINGLE_SCAN, JPGD_UNSUPPORTED_COLORSPACE,
		JPGD_UNSUPPORTED_SAMP_FACTORS, JPGD_DECODE_ERROR, JPGD_BAD_RESTART_MARKER,
		JPGD_BAD_SOS_SPECTRAL, JPGD_BAD_SOS_SUCCESSIVE, JPGD_STREAM_READ, JPGD_NOTENOUGHMEM, JPGD_TOO_MANY_SCANS
	};

	// Input stream interface.
	// Derive from this class to read input data from sources other than files or memory. Set m_eof_flag to true when no more data is available.
	// The decoder is rather greedy: it will keep on calling this method until its internal input buffer is full, or until the EOF flag is set.
	// It the input stream contains data after the JPEG stream's EOI (end of image) marker it will probably be pulled into the internal buffer.
	// Call the get_total_bytes_read() method to determine the actual size of the JPEG stream after successful decoding.
	class jpeg_decoder_stream
	{
	public:
		jpeg_decoder_stream() { }
		virtual ~jpeg_decoder_stream() { }

		// The read() method is called when the internal input buffer is empty.
		// Parameters:
		// pBuf - input buffer
		// max_bytes_to_read - maximum bytes that can be written to pBuf
		// pEOF_flag - set this to true if at end of stream (no more bytes remaining)
		// Returns -1 on error, otherwise return the number of bytes actually written to the buffer (which may be 0).
		// Notes: This method will be called in a loop until you set *pEOF_flag to true or the internal buffer is full.
		virtual int read(uint8* pBuf, int max_bytes_to_read, bool* pEOF_flag) = 0;
	};

	// stdio FILE stream class.
	class jpeg_decoder_file_stream : public jpeg_decoder_stream
	{
		jpeg_decoder_file_stream(const jpeg_decoder_file_stream&);
		jpeg_decoder_file_stream& operator =(const jpeg_decoder_file_stream&);

		FILE* m_pFile;
		bool m_eof_flag, m_error_flag;

	public:
		jpeg_decoder_file_stream();
		virtual ~jpeg_decoder_file_stream();

		bool open(const char* Pfilename);
		void close();

		virtual int read(uint8* pBuf, int max_bytes_to_read, bool* pEOF_flag);
	};

	// Memory stream class.
	class jpeg_decoder_mem_stream : public jpeg_decoder_stream
	{
		const uint8* m_pSrc_data;
		uint m_ofs, m_size;

	public:
		jpeg_decoder_mem_stream() : m_pSrc_data(NULL), m_ofs(0), m_size(0) { }
		jpeg_decoder_mem_stream(const uint8* pSrc_data, uint size) : m_pSrc_data(pSrc_data), m_ofs(0), m_size(size) { }

		virtual ~jpeg_decoder_mem_stream() { }

		bool open(const uint8* pSrc_data, uint size);
		void close() { m_pSrc_data = NULL; m_ofs = 0; m_size = 0; }

		virtual int read(uint8* pBuf, int max_bytes_to_read, bool* pEOF_flag);
	};

	// Loads JPEG file from a jpeg_decoder_stream.
	unsigned char* decompress_jpeg_image_from_stream(jpeg_decoder_stream* pStream, int* width, int* height, int* actual_comps, int req_comps, uint32_t flags = 0);

	enum
	{
		JPGD_IN_BUF_SIZE = 8192, JPGD_MAX_BLOCKS_PER_MCU = 10, JPGD_MAX_HUFF_TABLES = 8, JPGD_MAX_QUANT_TABLES = 4,
		JPGD_MAX_COMPONENTS = 4, JPGD_MAX_COMPS_IN_SCAN = 4, JPGD_MAX_BLOCKS_PER_ROW = 16384, JPGD_MAX_HEIGHT = 32768, JPGD_MAX_WIDTH = 32768
	};

	typedef int16 jpgd_quant_t;
	typedef int16 jpgd_block_t;

	class jpeg_decoder
	{
	public:
		enum
		{
			cFlagLinearChromaFiltering = 1
		};

		// Call get_error_code() after constructing to determine if the stream is valid or not. You may call the get_width(), get_height(), etc.
		// methods after the constructor is called. You may then either destruct the object, or begin decoding the image by calling begin_decoding(), then decode() on each scanline.
		jpeg_decoder(jpeg_decoder_stream* pStream, uint32_t flags = cFlagLinearChromaFiltering);

		~jpeg_decoder();

		// Call this method after constructing the object to begin decompression.
		// If JPGD_SUCCESS is returned you may then call decode() on each scanline.

		int begin_decoding();

		// Returns the next scan line.
		// For grayscale images, pScan_line will point to a buffer containing 8-bit pixels (get_bytes_per_pixel() will return 1). 
		// Otherwise, it will always point to a buffer containing 32-bit RGBA pixels (A will always be 255, and get_bytes_per_pixel() will return 4).
		// Returns JPGD_SUCCESS if a scan line has been returned.
		// Returns JPGD_DONE if all scan lines have been returned.
		// Returns JPGD_FAILED if an error occurred. Call get_error_code() for a more info.
		int decode(const void** pScan_line, uint* pScan_line_len);

		inline jpgd_status get_error_code() const { return m_error_code; }

		inline int get_width() const { return m_image_x_size; }
		inline int get_height() const { return m_image_y_size; }

		inline int get_num_components() const { return m_comps_in_frame; }

		inline int get_bytes_per_pixel() const { return m_dest_bytes_per_pixel; }
		inline int get_bytes_per_scan_line() const { return m_image_x_size * get_bytes_per_pixel(); }

		// Returns the total number of bytes actually consumed by the decoder (which should equal the actual size of the JPEG file).
		inline int get_total_bytes_read() const { return m_total_bytes_read; }

	private:
		jpeg_decoder(const jpeg_decoder&);
		jpeg_decoder& operator =(const jpeg_decoder&);

		typedef void (*pDecode_block_func)(jpeg_decoder*, int, int, int);

		struct huff_tables
		{
			bool ac_table;
			uint  look_up[256];
			uint  look_up2[256];
			uint8 code_size[JPGD_HUFF_CODE_SIZE_MAX_LENGTH];
			uint  tree[JPGD_HUFF_TREE_MAX_LENGTH];
		};

		struct coeff_buf
		{
			uint8* pData;
			int block_num_x, block_num_y;
			int block_len_x, block_len_y;
			int block_size;
		};

		struct mem_block
		{
			mem_block* m_pNext;
			size_t m_used_count;
			size_t m_size;
			char m_data[1];
		};

		jmp_buf m_jmp_state;
		uint32_t m_flags;
		mem_block* m_pMem_blocks;
		int m_image_x_size;
		int m_image_y_size;
		jpeg_decoder_stream* m_pStream;

		int m_progressive_flag;

		uint8 m_huff_ac[JPGD_MAX_HUFF_TABLES];
		uint8* m_huff_num[JPGD_MAX_HUFF_TABLES];      // pointer to number of Huffman codes per bit size
		uint8* m_huff_val[JPGD_MAX_HUFF_TABLES];      // pointer to Huffman codes per bit size
		jpgd_quant_t* m_quant[JPGD_MAX_QUANT_TABLES]; // pointer to quantization tables
		int m_scan_type;                              // Gray, Yh1v1, Yh1v2, Yh2v1, Yh2v2 (CMYK111, CMYK4114 no longer supported)
		int m_comps_in_frame;                         // # of components in frame
		int m_comp_h_samp[JPGD_MAX_COMPONENTS];       // component's horizontal sampling factor
		int m_comp_v_samp[JPGD_MAX_COMPONENTS];       // component's vertical sampling factor
		int m_comp_quant[JPGD_MAX_COMPONENTS];        // component's quantization table selector
		int m_comp_ident[JPGD_MAX_COMPONENTS];        // component's ID
		int m_comp_h_blocks[JPGD_MAX_COMPONENTS];
		int m_comp_v_blocks[JPGD_MAX_COMPONENTS];
		int m_comps_in_scan;                          // # of components in scan
		int m_comp_list[JPGD_MAX_COMPS_IN_SCAN];      // components in this scan
		int m_comp_dc_tab[JPGD_MAX_COMPONENTS];       // component's DC Huffman coding table selector
		int m_comp_ac_tab[JPGD_MAX_COMPONENTS];       // component's AC Huffman coding table selector
		int m_spectral_start;                         // spectral selection start
		int m_spectral_end;                           // spectral selection end
		int m_successive_low;                         // successive approximation low
		int m_successive_high;                        // successive approximation high
		int m_max_mcu_x_size;                         // MCU's max. X size in pixels
		int m_max_mcu_y_size;                         // MCU's max. Y size in pixels
		int m_blocks_per_mcu;
		int m_max_blocks_per_row;
		int m_mcus_per_row, m_mcus_per_col;
		int m_mcu_org[JPGD_MAX_BLOCKS_PER_MCU];
		int m_total_lines_left;                       // total # lines left in image
		int m_mcu_lines_left;                         // total # lines left in this MCU
		int m_num_buffered_scanlines;
		int m_real_dest_bytes_per_scan_line;
		int m_dest_bytes_per_scan_line;               // rounded up
		int m_dest_bytes_per_pixel;                   // 4 (RGB) or 1 (Y)
		huff_tables* m_pHuff_tabs[JPGD_MAX_HUFF_TABLES];
		coeff_buf* m_dc_coeffs[JPGD_MAX_COMPONENTS];
		coeff_buf* m_ac_coeffs[JPGD_MAX_COMPONENTS];
		int m_eob_run;
		int m_block_y_mcu[JPGD_MAX_COMPONENTS];
		uint8* m_pIn_buf_ofs;
		int m_in_buf_left;
		int m_tem_flag;

		uint8 m_in_buf_pad_start[64];
		uint8 m_in_buf[JPGD_IN_BUF_SIZE + 128];
		uint8 m_in_buf_pad_end[64];

		int m_bits_left;
		uint m_bit_buf;
		int m_restart_interval;
		int m_restarts_left;
		int m_next_restart_num;
		int m_max_mcus_per_row;
		int m_max_blocks_per_mcu;

		int m_max_mcus_per_col;
		uint m_last_dc_val[JPGD_MAX_COMPONENTS];
		jpgd_block_t* m_pMCU_coefficients;
		int m_mcu_block_max_zag[JPGD_MAX_BLOCKS_PER_MCU];
		uint8* m_pSample_buf;
		uint8* m_pSample_buf_prev;
		int m_crr[256];
		int m_cbb[256];
		int m_crg[256];
		int m_cbg[256];
		uint8* m_pScan_line_0;
		uint8* m_pScan_line_1;
		jpgd_status m_error_code;
		int m_total_bytes_read;

		bool m_ready_flag;
		bool m_eof_flag;
		bool m_sample_buf_prev_valid;

		inline int check_sample_buf_ofs(int ofs) const { assert(ofs >= 0); assert(ofs < m_max_blocks_per_row * 64); return ofs; }
		void free_all_blocks();
		JPGD_NORETURN void stop_decoding(jpgd_status status);
		void* alloc(size_t n, bool zero = false);
		void word_clear(void* p, uint16 c, uint n);
		void prep_in_buffer();
		void read_dht_marker();
		void read_dqt_marker();
		void read_sof_marker();
		void skip_variable_marker();
		void read_dri_marker();
		void read_sos_marker();
		int next_marker();
		int process_markers();
		void locate_soi_marker();
		void locate_sof_marker();
		int locate_sos_marker();
		void init(jpeg_decoder_stream* pStream, uint32_t flags);
		void create_look_ups();
		void fix_in_buffer();
		void transform_mcu(int mcu_row);
		coeff_buf* coeff_buf_open(int block_num_x, int block_num_y, int block_len_x, int block_len_y);
		inline jpgd_block_t* coeff_buf_getp(coeff_buf* cb, int block_x, int block_y);
		void load_next_row();
		void decode_next_row();
		void make_huff_table(int index, huff_tables* pH);
		void check_quant_tables();
		void check_huff_tables();
		bool calc_mcu_block_order();
		int init_scan();
		void init_frame();
		void process_restart();
		void decode_scan(pDecode_block_func decode_block_func);
		void init_progressive();
		void init_sequential();
		void decode_start();
		void decode_init(jpeg_decoder_stream* pStream, uint32_t flags);
		void H2V2Convert();
		uint32_t H2V2ConvertFiltered();
		void H2V1Convert();
		void H2V1ConvertFiltered();
		void H1V2Convert();
		void H1V2ConvertFiltered();
		void H1V1Convert();
		void gray_convert();
		void find_eoi();
		inline uint get_char();
		inline uint get_char(bool* pPadding_flag);
		inline void stuff_char(uint8 q);
		inline uint8 get_octet();
		inline uint get_bits(int num_bits);
		inline uint get_bits_no_markers(int numbits);
		inline int huff_decode(huff_tables* pH);
		inline int huff_decode(huff_tables* pH, int& extrabits);

		// Clamps a value between 0-255.
		static inline uint8 clamp(int i)
		{
			if (static_cast<uint>(i) > 255)
				i = (((~i) >> 31) & 0xFF);
			return static_cast<uint8>(i);
		}
		int decode_next_mcu_row();

		static void decode_block_dc_first(jpeg_decoder* pD, int component_id, int block_x, int block_y);
		static void decode_block_dc_refine(jpeg_decoder* pD, int component_id, int block_x, int block_y);
		static void decode_block_ac_first(jpeg_decoder* pD, int component_id, int block_x, int block_y);
		static void decode_block_ac_refine(jpeg_decoder* pD, int component_id, int block_x, int block_y);
	};

} // namespace jpgd

#endif // JPEG_DECODER_H
