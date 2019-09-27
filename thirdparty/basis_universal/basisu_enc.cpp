// basisu_enc.cpp
// Copyright (C) 2019 Binomial LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "basisu_enc.h"
#include "lodepng.h"
#include "basisu_resampler.h"
#include "basisu_resampler_filters.h"
#include "basisu_etc.h"
#include "transcoder/basisu_transcoder.h"

#if defined(_WIN32)
// For QueryPerformanceCounter/QueryPerformanceFrequency
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace basisu
{
	uint64_t interval_timer::g_init_ticks, interval_timer::g_freq;
	double interval_timer::g_timer_freq;

	uint8_t g_hamming_dist[256] =
	{
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
		4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
	};

	// Encoder library initialization (just call once at startup)
	void basisu_encoder_init()
	{
		basist::basisu_transcoder_init();
	}

	void error_printf(const char *pFmt, ...)
	{
		char buf[2048];

		va_list args;
		va_start(args, pFmt);
#ifdef _WIN32		
		vsprintf_s(buf, sizeof(buf), pFmt, args);
#else
		vsnprintf(buf, sizeof(buf), pFmt, args);
#endif
		va_end(args);

		fprintf(stderr, "ERROR: %s", buf);
	}

#if defined(_WIN32)
	inline void query_counter(timer_ticks* pTicks)
	{
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(pTicks));
	}
	inline void query_counter_frequency(timer_ticks* pTicks)
	{
		QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(pTicks));
	}
#elif defined(__APPLE__)
#include <sys/time.h>
	inline void query_counter(timer_ticks* pTicks)
	{
		struct timeval cur_time;
		gettimeofday(&cur_time, NULL);
		*pTicks = static_cast<unsigned long long>(cur_time.tv_sec) * 1000000ULL + static_cast<unsigned long long>(cur_time.tv_usec);
	}
	inline void query_counter_frequency(timer_ticks* pTicks)
	{
		*pTicks = 1000000;
	}
#elif defined(__GNUC__)
#include <sys/timex.h>
	inline void query_counter(timer_ticks* pTicks)
	{
		struct timeval cur_time;
		gettimeofday(&cur_time, NULL);
		*pTicks = static_cast<unsigned long long>(cur_time.tv_sec) * 1000000ULL + static_cast<unsigned long long>(cur_time.tv_usec);
	}
	inline void query_counter_frequency(timer_ticks* pTicks)
	{
		*pTicks = 1000000;
	}
#else
#error TODO
#endif

	interval_timer::interval_timer() : m_start_time(0), m_stop_time(0), m_started(false), m_stopped(false)
	{
		if (!g_timer_freq)
			init();
	}

	void interval_timer::start()
	{
		query_counter(&m_start_time);
		m_started = true;
		m_stopped = false;
	}

	void interval_timer::stop()
	{
		assert(m_started);
		query_counter(&m_stop_time);
		m_stopped = true;
	}

	double interval_timer::get_elapsed_secs() const
	{
		assert(m_started);
		if (!m_started)
			return 0;

		timer_ticks stop_time = m_stop_time;
		if (!m_stopped)
			query_counter(&stop_time);

		timer_ticks delta = stop_time - m_start_time;
		return delta * g_timer_freq;
	}
		
	void interval_timer::init()
	{
		if (!g_timer_freq)
		{
			query_counter_frequency(&g_freq);
			g_timer_freq = 1.0f / g_freq;
			query_counter(&g_init_ticks);
		}
	}

	timer_ticks interval_timer::get_ticks()
	{
		if (!g_timer_freq)
			init();
		timer_ticks ticks;
		query_counter(&ticks);
		return ticks - g_init_ticks;
	}

	double interval_timer::ticks_to_secs(timer_ticks ticks)
	{
		if (!g_timer_freq)
			init();
		return ticks * g_timer_freq;
	}
		
	bool load_png(const char* pFilename, image& img)
	{
		std::vector<uint8_t> buffer;
		unsigned err = lodepng::load_file(buffer, std::string(pFilename));
		if (err)
			return false;

		unsigned w = 0, h = 0;
				
		if (sizeof(void *) == sizeof(uint32_t))
		{
			// Inspect the image first on 32-bit builds, to see if the image would require too much memory.
			lodepng::State state;
			err = lodepng_inspect(&w, &h, &state, &buffer[0], buffer.size());
			if ((err != 0) || (!w) || (!h))
				return false;

			const uint32_t exepected_alloc_size = w * h * sizeof(uint32_t);
			
			// If the file is too large on 32-bit builds then just bail now, to prevent causing a memory exception.
			const uint32_t MAX_ALLOC_SIZE = 250000000;
			if (exepected_alloc_size >= MAX_ALLOC_SIZE)
			{
				error_printf("Image \"%s\" is too large (%ux%u) to process in a 32-bit build!\n", pFilename, w, h);
				return false;
			}
			
			w = h = 0;
		}
				
		std::vector<uint8_t> out;
		err = lodepng::decode(out, w, h, &buffer[0], buffer.size());
		if ((err != 0) || (!w) || (!h))
			return false;

		if (out.size() != (w * h * 4))
			return false;

		img.resize(w, h);

		memcpy(img.get_ptr(), &out[0], out.size());

		return true;
	}
	
	bool save_png(const char* pFilename, const image & img, uint32_t image_save_flags, uint32_t grayscale_comp)
	{
		if (!img.get_total_pixels())
			return false;

		std::vector<uint8_t> out;
		unsigned err = 0;
				
		if (image_save_flags & cImageSaveGrayscale)
		{
			uint8_vec g_pixels(img.get_width() * img.get_height());
			uint8_t *pDst = &g_pixels[0];

			for (uint32_t y = 0; y < img.get_height(); y++)
				for (uint32_t x = 0; x < img.get_width(); x++)
					*pDst++ = img(x, y)[grayscale_comp];

			err = lodepng::encode(out, (const uint8_t*)& g_pixels[0], img.get_width(), img.get_height(), LCT_GREY, 8);
		}
		else
		{
			bool has_alpha = img.has_alpha();
			if ((!has_alpha) || ((image_save_flags & cImageSaveIgnoreAlpha) != 0))
			{
				uint8_vec rgb_pixels(img.get_width() * 3 * img.get_height());
				uint8_t *pDst = &rgb_pixels[0];

				for (uint32_t y = 0; y < img.get_height(); y++)
				{
					for (uint32_t x = 0; x < img.get_width(); x++)
					{
						const color_rgba& c = img(x, y);
						pDst[0] = c.r;
						pDst[1] = c.g;
						pDst[2] = c.b;
						pDst += 3;
					}
				}

				err = lodepng::encode(out, (const uint8_t*)& rgb_pixels[0], img.get_width(), img.get_height(), LCT_RGB, 8);
			}
			else
			{
				err = lodepng::encode(out, (const uint8_t*)img.get_ptr(), img.get_width(), img.get_height(), LCT_RGBA, 8);
			}
		}

		err = lodepng::save_file(out, std::string(pFilename));
		if (err)
			return false;

		return true;
	}
		
	bool read_file_to_vec(const char* pFilename, uint8_vec& data)
	{
		FILE* pFile = nullptr;
#ifdef _WIN32
		fopen_s(&pFile, pFilename, "rb");
#else
		pFile = fopen(pFilename, "rb");
#endif
		if (!pFile)
			return false;
				
		fseek(pFile, 0, SEEK_END);
#ifdef _WIN32
		int64_t filesize = _ftelli64(pFile);
#else
		int64_t filesize = ftello(pFile);
#endif
		if (filesize < 0)
		{
			fclose(pFile);
			return false;
		}
		fseek(pFile, 0, SEEK_SET);

		if (sizeof(size_t) == sizeof(uint32_t))
		{
			if (filesize > 0x70000000)
			{
				// File might be too big to load safely in one alloc
				fclose(pFile);
				return false;
			}
		}

		data.resize((size_t)filesize);

		if (filesize)
		{
			if (fread(&data[0], 1, (size_t)filesize, pFile) != (size_t)filesize)
			{
				fclose(pFile);
				return false;
			}
		}

		fclose(pFile);
		return true;
	}

	bool write_data_to_file(const char* pFilename, const void* pData, size_t len)
	{
		FILE* pFile = nullptr;
#ifdef _WIN32
		fopen_s(&pFile, pFilename, "wb");
#else
		pFile = fopen(pFilename, "wb");
#endif
		if (!pFile)
			return false;

		if (len)
		{
			if (fwrite(pData, 1, len, pFile) != len)
			{
				fclose(pFile);
				return false;
			}
		}

		return fclose(pFile) != EOF;
	}

	float linear_to_srgb(float l)
	{
		assert(l >= 0.0f && l <= 1.0f);
		if (l < .0031308f)
			return saturate(l * 12.92f);
		else
			return saturate(1.055f * powf(l, 1.0f/2.4f) - .055f);
	}

	float srgb_to_linear(float s)
	{
		assert(s >= 0.0f && s <= 1.0f);
		if (s < .04045f)
			return saturate(s * (1.0f/12.92f));
		else
			return saturate(powf((s + .055f) * (1.0f/1.055f), 2.4f));
	}

	bool image_resample(const image &src, image &dst, bool srgb,
		const char *pFilter, float filter_scale, 
		bool wrapping,
		uint32_t first_comp, uint32_t num_comps)
	{
		assert((first_comp + num_comps) <= 4);

		const int cMaxComps = 4;
				
		const uint32_t src_w = src.get_width(), src_h = src.get_height();
		const uint32_t dst_w = dst.get_width(), dst_h = dst.get_height();
				
		if (maximum(src_w, src_h) > BASISU_RESAMPLER_MAX_DIMENSION)
		{
			printf("Image is too large!\n");
			return false;
		}

		if (!src_w || !src_h || !dst_w || !dst_h)
			return false;
				
		if ((num_comps < 1) || (num_comps > cMaxComps))
			return false;
				
		if ((minimum(dst_w, dst_h) < 1) || (maximum(dst_w, dst_h) > BASISU_RESAMPLER_MAX_DIMENSION))
		{
			printf("Image is too large!\n");
			return false;
		}

		if ((src_w == dst_w) && (src_h == dst_h))
		{
			dst = src;
			return true;
		}

		float srgb_to_linear_table[256];
		if (srgb)
		{
			for (int i = 0; i < 256; ++i)
				srgb_to_linear_table[i] = srgb_to_linear((float)i * (1.0f/255.0f));
		}

		const int LINEAR_TO_SRGB_TABLE_SIZE = 8192;
		uint8_t linear_to_srgb_table[LINEAR_TO_SRGB_TABLE_SIZE];

		if (srgb)
		{
			for (int i = 0; i < LINEAR_TO_SRGB_TABLE_SIZE; ++i)
				linear_to_srgb_table[i] = (uint8_t)clamp<int>((int)(255.0f * linear_to_srgb((float)i * (1.0f / (LINEAR_TO_SRGB_TABLE_SIZE - 1))) + .5f), 0, 255);
		}

		std::vector<float> samples[cMaxComps];
		Resampler *resamplers[cMaxComps];
		
		resamplers[0] = new Resampler(src_w, src_h, dst_w, dst_h,
			wrapping ? Resampler::BOUNDARY_WRAP : Resampler::BOUNDARY_CLAMP, 0.0f, 1.0f,
			pFilter, nullptr, nullptr, filter_scale, filter_scale, 0, 0);
		samples[0].resize(src_w);

		for (uint32_t i = 1; i < num_comps; ++i)
		{
			resamplers[i] = new Resampler(src_w, src_h, dst_w, dst_h,
				wrapping ? Resampler::BOUNDARY_WRAP : Resampler::BOUNDARY_CLAMP, 0.0f, 1.0f,
				pFilter, resamplers[0]->get_clist_x(), resamplers[0]->get_clist_y(), filter_scale, filter_scale, 0, 0);
			samples[i].resize(src_w);
		}

		uint32_t dst_y = 0;

		for (uint32_t src_y = 0; src_y < src_h; ++src_y)
		{
			const color_rgba *pSrc = &src(0, src_y);

			// Put source lines into resampler(s)
			for (uint32_t x = 0; x < src_w; ++x)
			{
				for (uint32_t c = 0; c < num_comps; ++c)
				{
					const uint32_t comp_index = first_comp + c;
					const uint32_t v = (*pSrc)[comp_index];

					if (!srgb || (comp_index == 3))
						samples[c][x] = v * (1.0f / 255.0f);
					else
						samples[c][x] = srgb_to_linear_table[v];
				}

				pSrc++;
			}

			for (uint32_t c = 0; c < num_comps; ++c)
			{
				if (!resamplers[c]->put_line(&samples[c][0]))
				{
					for (uint32_t i = 0; i < num_comps; i++)
						delete resamplers[i];
					return false;
				}
			}

			// Now retrieve any output lines
			for (;;)
			{
				uint32_t c;
				for (c = 0; c < num_comps; ++c)
				{
					const uint32_t comp_index = first_comp + c;

					const float *pOutput_samples = resamplers[c]->get_line();
					if (!pOutput_samples)
						break;

					const bool linear_flag = !srgb || (comp_index == 3);
					
					color_rgba *pDst = &dst(0, dst_y);

					for (uint32_t x = 0; x < dst_w; x++)
					{
						// TODO: Add dithering
						if (linear_flag)
						{
							int j = (int)(255.0f * pOutput_samples[x] + .5f);
							(*pDst)[comp_index] = (uint8_t)clamp<int>(j, 0, 255);
						}
						else
						{
							int j = (int)((LINEAR_TO_SRGB_TABLE_SIZE - 1) * pOutput_samples[x] + .5f);
							(*pDst)[comp_index] = linear_to_srgb_table[clamp<int>(j, 0, LINEAR_TO_SRGB_TABLE_SIZE - 1)];
						}

						pDst++;
					}
				}
				if (c < num_comps)
					break;

				++dst_y;
			}
		}

		for (uint32_t i = 0; i < num_comps; ++i)
			delete resamplers[i];

		return true;
	}

	void canonical_huffman_calculate_minimum_redundancy(sym_freq *A, int num_syms)
	{
		// See the paper "In-Place Calculation of Minimum Redundancy Codes" by Moffat and Katajainen
		if (!num_syms)
			return;

		if (1 == num_syms)
		{
			A[0].m_key = 1;
			return;
		}
		
		A[0].m_key += A[1].m_key;
		
		int s = 2, r = 0, next;
		for (next = 1; next < (num_syms - 1); ++next)
		{
			if ((s >= num_syms) || (A[r].m_key < A[s].m_key))
			{
				A[next].m_key = A[r].m_key;
				A[r].m_key = static_cast<uint16_t>(next);
				++r;
			}
			else
			{
				A[next].m_key = A[s].m_key;
				++s;
			}

			if ((s >= num_syms) || ((r < next) && A[r].m_key < A[s].m_key))
			{
				A[next].m_key = static_cast<uint16_t>(A[next].m_key + A[r].m_key);
				A[r].m_key = static_cast<uint16_t>(next);
				++r;
			}
			else
			{
				A[next].m_key = static_cast<uint16_t>(A[next].m_key + A[s].m_key);
				++s;
			}
		}
		A[num_syms - 2].m_key = 0;

		for (next = num_syms - 3; next >= 0; --next)
		{
			A[next].m_key = 1 + A[A[next].m_key].m_key;
		}

		int num_avail = 1, num_used = 0, depth = 0;
		r = num_syms - 2;
		next = num_syms - 1;
		while (num_avail > 0)
		{
			for ( ; (r >= 0) && ((int)A[r].m_key == depth); ++num_used, --r )
				;

			for ( ; num_avail > num_used; --next, --num_avail)
				A[next].m_key = static_cast<uint16_t>(depth);

			num_avail = 2 * num_used;
			num_used = 0;
			++depth;
		}
	}

	void canonical_huffman_enforce_max_code_size(int *pNum_codes, int code_list_len, int max_code_size)
	{
		int i;
		uint32_t total = 0;
		if (code_list_len <= 1)
			return;

		for (i = max_code_size + 1; i <= cHuffmanMaxSupportedInternalCodeSize; i++)
			pNum_codes[max_code_size] += pNum_codes[i];

		for (i = max_code_size; i > 0; i--)
			total += (((uint32_t)pNum_codes[i]) << (max_code_size - i));

		while (total != (1UL << max_code_size))
		{
			pNum_codes[max_code_size]--;
			for (i = max_code_size - 1; i > 0; i--)
			{
				if (pNum_codes[i])
				{
					pNum_codes[i]--;
					pNum_codes[i + 1] += 2;
					break;
				}
			}

			total--;
		}
	}

	sym_freq *canonical_huffman_radix_sort_syms(uint32_t num_syms, sym_freq *pSyms0, sym_freq *pSyms1)
	{
		uint32_t total_passes = 2, pass_shift, pass, i, hist[256 * 2];
		sym_freq *pCur_syms = pSyms0, *pNew_syms = pSyms1;

		clear_obj(hist);

		for (i = 0; i < num_syms; i++)
		{
			uint32_t freq = pSyms0[i].m_key;
			hist[freq & 0xFF]++;
			hist[256 + ((freq >> 8) & 0xFF)]++;
		}

		while ((total_passes > 1) && (num_syms == hist[(total_passes - 1) * 256]))
			total_passes--;

		for (pass_shift = 0, pass = 0; pass < total_passes; pass++, pass_shift += 8)
		{
			const uint32_t *pHist = &hist[pass << 8];
			uint32_t offsets[256], cur_ofs = 0;
			for (i = 0; i < 256; i++)
			{
				offsets[i] = cur_ofs;
				cur_ofs += pHist[i];
			}

			for (i = 0; i < num_syms; i++)
				pNew_syms[offsets[(pCur_syms[i].m_key >> pass_shift) & 0xFF]++] = pCur_syms[i];

			sym_freq *t = pCur_syms;
			pCur_syms = pNew_syms;
			pNew_syms = t;
		}

		return pCur_syms;
	}

	bool huffman_encoding_table::init(uint32_t num_syms, const uint16_t *pFreq, uint32_t max_code_size)
	{
		if (max_code_size > cHuffmanMaxSupportedCodeSize)
			return false;
		if ((!num_syms) || (num_syms > cHuffmanMaxSyms))
			return false;

		uint32_t total_used_syms = 0;
		for (uint32_t i = 0; i < num_syms; i++)
			if (pFreq[i])
				total_used_syms++;

		if (!total_used_syms)
			return false;

		std::vector<sym_freq> sym_freq0(total_used_syms), sym_freq1(total_used_syms);
		for (uint32_t i = 0, j = 0; i < num_syms; i++)
		{
			if (pFreq[i])
			{
				sym_freq0[j].m_key = pFreq[i];
				sym_freq0[j++].m_sym_index = static_cast<uint16_t>(i);
			}
		}

		sym_freq *pSym_freq = canonical_huffman_radix_sort_syms(total_used_syms, &sym_freq0[0], &sym_freq1[0]);

		canonical_huffman_calculate_minimum_redundancy(pSym_freq, total_used_syms);

		int num_codes[cHuffmanMaxSupportedInternalCodeSize + 1];
		clear_obj(num_codes);

		for (uint32_t i = 0; i < total_used_syms; i++)
		{
			if (pSym_freq[i].m_key > cHuffmanMaxSupportedInternalCodeSize)
				return false;

			num_codes[pSym_freq[i].m_key]++;
		}

		canonical_huffman_enforce_max_code_size(num_codes, total_used_syms, max_code_size);

		m_code_sizes.resize(0);
		m_code_sizes.resize(num_syms);

		m_codes.resize(0);
		m_codes.resize(num_syms);

		for (uint32_t i = 1, j = total_used_syms; i <= max_code_size; i++)
			for (uint32_t l = num_codes[i]; l > 0; l--)
				m_code_sizes[pSym_freq[--j].m_sym_index] = static_cast<uint8_t>(i);

		uint32_t next_code[cHuffmanMaxSupportedInternalCodeSize + 1];

		next_code[1] = 0;
		for (uint32_t j = 0, i = 2; i <= max_code_size; i++)
			next_code[i] = j = ((j + num_codes[i - 1]) << 1);

		for (uint32_t i = 0; i < num_syms; i++)
		{
			uint32_t rev_code = 0, code, code_size;
			if ((code_size = m_code_sizes[i]) == 0)
				continue;
			if (code_size > cHuffmanMaxSupportedInternalCodeSize)
				return false;
			code = next_code[code_size]++;
			for (uint32_t l = code_size; l > 0; l--, code >>= 1)
				rev_code = (rev_code << 1) | (code & 1);
			m_codes[i] = static_cast<uint16_t>(rev_code);
		}

		return true;
	}

	bool huffman_encoding_table::init(uint32_t num_syms, const uint32_t *pSym_freq, uint32_t max_code_size)
	{
		if ((!num_syms) || (num_syms > cHuffmanMaxSyms))
			return false;

		uint16_vec sym_freq(num_syms);

		uint32_t max_freq = 0;
		for (uint32_t i = 0; i < num_syms; i++)
			max_freq = maximum(max_freq, pSym_freq[i]);

		if (max_freq < UINT16_MAX)
		{
			for (uint32_t i = 0; i < num_syms; i++)
				sym_freq[i] = static_cast<uint16_t>(pSym_freq[i]);
		}
		else
		{
			for (uint32_t i = 0; i < num_syms; i++)
				if (pSym_freq[i])
					sym_freq[i] = static_cast<uint16_t>(maximum<uint32_t>((pSym_freq[i] * 65534U + (max_freq >> 1)) / max_freq, 1));
		}

		return init(num_syms, &sym_freq[0], max_code_size);
	}

	void bitwise_coder::end_nonzero_run(uint16_vec &syms, uint32_t &run_size, uint32_t len)
	{
		if (run_size)
		{
			if (run_size < cHuffmanSmallRepeatSizeMin)
			{
				while (run_size--)
					syms.push_back(static_cast<uint16_t>(len));
			}
			else if (run_size <= cHuffmanSmallRepeatSizeMax)
			{
				syms.push_back(static_cast<uint16_t>(cHuffmanSmallRepeatCode | ((run_size - cHuffmanSmallRepeatSizeMin) << 6)));
			}
			else
			{
				assert((run_size >= cHuffmanBigRepeatSizeMin) && (run_size <= cHuffmanBigRepeatSizeMax));
				syms.push_back(static_cast<uint16_t>(cHuffmanBigRepeatCode | ((run_size - cHuffmanBigRepeatSizeMin) << 6)));
			}
		}

		run_size = 0;
	}

	void bitwise_coder::end_zero_run(uint16_vec &syms, uint32_t &run_size)
	{
		if (run_size)
		{
			if (run_size < cHuffmanSmallZeroRunSizeMin)
			{
				while (run_size--)
					syms.push_back(0);
			}
			else if (run_size <= cHuffmanSmallZeroRunSizeMax)
			{
				syms.push_back(static_cast<uint16_t>(cHuffmanSmallZeroRunCode | ((run_size - cHuffmanSmallZeroRunSizeMin) << 6)));
			}
			else
			{
				assert((run_size >= cHuffmanBigZeroRunSizeMin) && (run_size <= cHuffmanBigZeroRunSizeMax));
				syms.push_back(static_cast<uint16_t>(cHuffmanBigZeroRunCode | ((run_size - cHuffmanBigZeroRunSizeMin) << 6)));
			}
		}

		run_size = 0;
	}

	uint32_t bitwise_coder::emit_huffman_table(const huffman_encoding_table &tab)
	{
		const uint64_t start_bits = m_total_bits;

		const uint8_vec &code_sizes = tab.get_code_sizes();

		uint32_t total_used = tab.get_total_used_codes();
		put_bits(total_used, cHuffmanMaxSymsLog2);
			
		if (!total_used)
			return 0;

		uint16_vec syms;
		syms.reserve(total_used + 16);

		uint32_t prev_code_len = UINT_MAX, zero_run_size = 0, nonzero_run_size = 0;

		for (uint32_t i = 0; i <= total_used; ++i)
		{
			const uint32_t code_len = (i == total_used) ? 0xFF : code_sizes[i];
			assert((code_len == 0xFF) || (code_len <= 16));

			if (code_len)
			{
				end_zero_run(syms, zero_run_size);

				if (code_len != prev_code_len)
				{
					end_nonzero_run(syms, nonzero_run_size, prev_code_len);
					if (code_len != 0xFF)
						syms.push_back(static_cast<uint16_t>(code_len));
				}
				else if (++nonzero_run_size == cHuffmanBigRepeatSizeMax)
					end_nonzero_run(syms, nonzero_run_size, prev_code_len);
			}
			else
			{
				end_nonzero_run(syms, nonzero_run_size, prev_code_len);

				if (++zero_run_size == cHuffmanBigZeroRunSizeMax)
					end_zero_run(syms, zero_run_size);
			}

			prev_code_len = code_len;
		}

		histogram h(cHuffmanTotalCodelengthCodes);
		for (uint32_t i = 0; i < syms.size(); i++)
			h.inc(syms[i] & 63);

		huffman_encoding_table ct;
		if (!ct.init(h, 7))
			return 0;

		assert(cHuffmanTotalSortedCodelengthCodes == cHuffmanTotalCodelengthCodes);

		uint32_t total_codelength_codes;
		for (total_codelength_codes = cHuffmanTotalSortedCodelengthCodes; total_codelength_codes > 0; total_codelength_codes--)
			if (ct.get_code_sizes()[g_huffman_sorted_codelength_codes[total_codelength_codes - 1]])
				break;

		assert(total_codelength_codes);

		put_bits(total_codelength_codes, 5);
		for (uint32_t i = 0; i < total_codelength_codes; i++)
			put_bits(ct.get_code_sizes()[g_huffman_sorted_codelength_codes[i]], 3);

		for (uint32_t i = 0; i < syms.size(); ++i)
		{
			const uint32_t l = syms[i] & 63, e = syms[i] >> 6;

			put_code(l, ct);
				
			if (l == cHuffmanSmallZeroRunCode)
				put_bits(e, cHuffmanSmallZeroRunExtraBits);
			else if (l == cHuffmanBigZeroRunCode)
				put_bits(e, cHuffmanBigZeroRunExtraBits);
			else if (l == cHuffmanSmallRepeatCode)
				put_bits(e, cHuffmanSmallRepeatExtraBits);
			else if (l == cHuffmanBigRepeatCode)
				put_bits(e, cHuffmanBigRepeatExtraBits);
		}

		return (uint32_t)(m_total_bits - start_bits);
	}

	bool huffman_test(int rand_seed)
	{
		histogram h(19);

		// Feed in a fibonacci sequence to force large codesizes
		h[0] += 1; h[1] += 1; h[2] += 2; h[3] += 3;
		h[4] += 5; h[5] += 8; h[6] += 13; h[7] += 21;
		h[8] += 34; h[9] += 55; h[10] += 89; h[11] += 144;
		h[12] += 233; h[13] += 377; h[14] += 610; h[15] += 987;
		h[16] += 1597; h[17] += 2584; h[18] += 4181;

		huffman_encoding_table etab;
		etab.init(h, 16);
		
		{
			bitwise_coder c;
			c.init(1024);

			c.emit_huffman_table(etab);
			for (int i = 0; i < 19; i++)
				c.put_code(i, etab);

			c.flush();

			basist::bitwise_decoder d;
			d.init(&c.get_bytes()[0], static_cast<uint32_t>(c.get_bytes().size()));

			basist::huffman_decoding_table dtab;
			bool success = d.read_huffman_table(dtab);
			if (!success)
			{
				assert(0);
				printf("Failure 5\n");
				return false;
			}

			for (uint32_t i = 0; i < 19; i++)
			{
				uint32_t s = d.decode_huffman(dtab);
				if (s != i)
				{
					assert(0);
					printf("Failure 5\n");
					return false;
				}
			}
		}

		basisu::rand r;
		r.seed(rand_seed);

		for (int iter = 0; iter < 500000; iter++)
		{
			printf("%u\n", iter);

			uint32_t max_sym = r.irand(0, 8193);
			uint32_t num_codes = r.irand(1, 10000);
			uint_vec syms(num_codes);

			for (uint32_t i = 0; i < num_codes; i++)
			{
				if (r.bit())
					syms[i] = r.irand(0, max_sym);
				else
				{
					int s = (int)(r.gaussian((float)max_sym / 2, (float)maximum<int>(1, max_sym / 2)) + .5f);
					s = basisu::clamp<int>(s, 0, max_sym);

					syms[i] = s;
				}

			}

			histogram h1(max_sym + 1);
			for (uint32_t i = 0; i < num_codes; i++)
				h1[syms[i]]++;

			huffman_encoding_table etab2;
			if (!etab2.init(h1, 16))
			{
				assert(0);
				printf("Failed 0\n");
				return false;
			}

			bitwise_coder c;
			c.init(1024);

			c.emit_huffman_table(etab2);

			for (uint32_t i = 0; i < num_codes; i++)
				c.put_code(syms[i], etab2);

			c.flush();

			basist::bitwise_decoder d;
			d.init(&c.get_bytes()[0], (uint32_t)c.get_bytes().size());

			basist::huffman_decoding_table dtab;
			bool success = d.read_huffman_table(dtab);
			if (!success)
			{
				assert(0);
				printf("Failed 2\n");
				return false;
			}

			for (uint32_t i = 0; i < num_codes; i++)
			{
				uint32_t s = d.decode_huffman(dtab);
				if (s != syms[i])
				{
					assert(0);
					printf("Failed 4\n");
					return false;
				}
			}

		}
		return true;
	}

	void palette_index_reorderer::init(uint32_t num_indices, const uint32_t *pIndices, uint32_t num_syms, pEntry_dist_func pDist_func, void *pCtx, float dist_func_weight)
	{
		assert((num_syms > 0) && (num_indices > 0));
		assert((dist_func_weight >= 0.0f) && (dist_func_weight <= 1.0f));

		clear();

		m_remap_table.resize(num_syms);
		m_entries_picked.reserve(num_syms);
		m_total_count_to_picked.resize(num_syms);

		if (num_indices <= 1)
			return;

		prepare_hist(num_syms, num_indices, pIndices);
		find_initial(num_syms);

		while (m_entries_to_do.size())
		{
			// Find the best entry to move into the picked list.
			uint32_t best_entry;
			double best_count;
			find_next_entry(best_entry, best_count, pDist_func, pCtx, dist_func_weight);

			// We now have chosen an entry to place in the picked list, now determine which side it goes on.
			const uint32_t entry_to_move = m_entries_to_do[best_entry];
								
			float side = pick_side(num_syms, entry_to_move, pDist_func, pCtx, dist_func_weight);
								
			// Put entry_to_move either on the "left" or "right" side of the picked entries
			if (side <= 0)
				m_entries_picked.push_back(entry_to_move);
			else
				m_entries_picked.insert(m_entries_picked.begin(), entry_to_move);

			// Erase best_entry from the todo list
			m_entries_to_do.erase(m_entries_to_do.begin() + best_entry);

			// We've just moved best_entry to the picked list, so now we need to update m_total_count_to_picked[] to factor the additional count to best_entry
			for (uint32_t i = 0; i < m_entries_to_do.size(); i++)
				m_total_count_to_picked[m_entries_to_do[i]] += get_hist(m_entries_to_do[i], entry_to_move, num_syms);
		}

		for (uint32_t i = 0; i < num_syms; i++)
			m_remap_table[m_entries_picked[i]] = i;
	}

	void palette_index_reorderer::prepare_hist(uint32_t num_syms, uint32_t num_indices, const uint32_t *pIndices)
	{
		m_hist.resize(0);
		m_hist.resize(num_syms * num_syms);

		for (uint32_t i = 0; i < num_indices; i++)
		{
			const uint32_t idx = pIndices[i];
			inc_hist(idx, (i < (num_indices - 1)) ? pIndices[i + 1] : -1, num_syms);
			inc_hist(idx, (i > 0) ? pIndices[i - 1] : -1, num_syms);
		}
	}

	void palette_index_reorderer::find_initial(uint32_t num_syms)
	{
		uint32_t max_count = 0, max_index = 0;
		for (uint32_t i = 0; i < num_syms * num_syms; i++)
			if (m_hist[i] > max_count)
				max_count = m_hist[i], max_index = i;

		uint32_t a = max_index / num_syms, b = max_index % num_syms;

		m_entries_picked.push_back(a);
		m_entries_picked.push_back(b);

		for (uint32_t i = 0; i < num_syms; i++)
			if ((i != b) && (i != a))
				m_entries_to_do.push_back(i);

		for (uint32_t i = 0; i < m_entries_to_do.size(); i++)
			for (uint32_t j = 0; j < m_entries_picked.size(); j++)
				m_total_count_to_picked[m_entries_to_do[i]] += get_hist(m_entries_to_do[i], m_entries_picked[j], num_syms);
	}

	void palette_index_reorderer::find_next_entry(uint32_t &best_entry, double &best_count, pEntry_dist_func pDist_func, void *pCtx, float dist_func_weight)
	{
		best_entry = 0;
		best_count = 0;

		for (uint32_t i = 0; i < m_entries_to_do.size(); i++)
		{
			const uint32_t u = m_entries_to_do[i];
			double total_count = m_total_count_to_picked[u];

			if (pDist_func)
			{
				float w = maximum<float>((*pDist_func)(u, m_entries_picked.front(), pCtx), (*pDist_func)(u, m_entries_picked.back(), pCtx));
				assert((w >= 0.0f) && (w <= 1.0f));
				total_count = (total_count + 1.0f) * lerp(1.0f - dist_func_weight, 1.0f + dist_func_weight, w);
			}

			if (total_count <= best_count)
				continue;

			best_entry = i;
			best_count = total_count;
		}
	}

	float palette_index_reorderer::pick_side(uint32_t num_syms, uint32_t entry_to_move, pEntry_dist_func pDist_func, void *pCtx, float dist_func_weight)
	{
		float which_side = 0;

		int l_count = 0, r_count = 0;
		for (uint32_t j = 0; j < m_entries_picked.size(); j++)
		{
			const int count = get_hist(entry_to_move, m_entries_picked[j], num_syms), r = ((int)m_entries_picked.size() + 1 - 2 * (j + 1));
			which_side += static_cast<float>(r * count);
			if (r >= 0)
				l_count += r * count;
			else
				r_count += -r * count;
		}

		if (pDist_func)
		{
			float w_left = lerp(1.0f - dist_func_weight, 1.0f + dist_func_weight, (*pDist_func)(entry_to_move, m_entries_picked.front(), pCtx));
			float w_right = lerp(1.0f - dist_func_weight, 1.0f + dist_func_weight, (*pDist_func)(entry_to_move, m_entries_picked.back(), pCtx));
			which_side = w_left * l_count - w_right * r_count;
		}
		return which_side;
	}

	void image_metrics::calc(const image &a, const image &b, uint32_t first_chan, uint32_t total_chans, bool avg_comp_error, bool use_601_luma)
	{
		assert((first_chan < 4U) && (first_chan + total_chans <= 4U));
						
		const uint32_t width = std::min(a.get_width(), b.get_width());
		const uint32_t height = std::min(a.get_height(), b.get_height());
						
		double hist[256];
		clear_obj(hist);

		for (uint32_t y = 0; y < height; y++)
		{
			for (uint32_t x = 0; x < width; x++)
			{
				const color_rgba &ca = a(x, y), &cb = b(x, y);

				if (total_chans)
				{
					for (uint32_t c = 0; c < total_chans; c++)
						hist[iabs(ca[first_chan + c] - cb[first_chan + c])]++;
				}
				else
				{
					if (use_601_luma)
						hist[iabs(ca.get_601_luma() - cb.get_601_luma())]++;
					else
						hist[iabs(ca.get_709_luma() - cb.get_709_luma())]++;
				}
			}
		}

		m_max = 0;
		double sum = 0.0f, sum2 = 0.0f;
		for (uint32_t i = 0; i < 256; i++)
		{
			if (hist[i])
			{
				m_max = std::max<float>(m_max, (float)i);
				double v = i * hist[i];
				sum += v;
				sum2 += i * v;
			}
		}

		double total_values = (double)width * (double)height;
		if (avg_comp_error)
			total_values *= (double)clamp<uint32_t>(total_chans, 1, 4);

		m_mean = (float)clamp<double>(sum / total_values, 0.0f, 255.0);
		m_mean_squared = (float)clamp<double>(sum2 / total_values, 0.0f, 255.0 * 255.0);
		m_rms = (float)sqrt(m_mean_squared);
		m_psnr = m_rms ? (float)clamp<double>(log10(255.0 / m_rms) * 20.0, 0.0f, 300.0f) : 1e+10f;
	}

	void fill_buffer_with_random_bytes(void *pBuf, size_t size, uint32_t seed)
	{
		rand r(seed);

		uint8_t *pDst = static_cast<uint8_t *>(pBuf);

		while (size >= sizeof(uint32_t))
		{
			*(uint32_t *)pDst = r.urand32();
			pDst += sizeof(uint32_t);
			size -= sizeof(uint32_t);
		}

		while (size)
		{
			*pDst++ = r.byte();
			size--;
		}
	}

	uint32_t hash_hsieh(const uint8_t *pBuf, size_t len)
	{
		if (!pBuf || !len) 
			return 0;

		uint32_t h = static_cast<uint32_t>(len);

		const uint32_t bytes_left = len & 3;
		len >>= 2;

		while (len--)
		{
			const uint16_t *pWords = reinterpret_cast<const uint16_t *>(pBuf);

			h += pWords[0];
			
			const uint32_t t = (pWords[1] << 11) ^ h;
			h = (h << 16) ^ t;
			
			pBuf += sizeof(uint32_t);
			
			h += h >> 11;
		}

		switch (bytes_left)
		{
		case 1: 
			h += *reinterpret_cast<const signed char*>(pBuf);
			h ^= h << 10;
			h += h >> 1;
			break;
		case 2: 
			h += *reinterpret_cast<const uint16_t *>(pBuf);
			h ^= h << 11;
			h += h >> 17;
			break;
		case 3:
			h += *reinterpret_cast<const uint16_t *>(pBuf);
			h ^= h << 16;
			h ^= (static_cast<signed char>(pBuf[sizeof(uint16_t)])) << 18;
			h += h >> 11;
			break;
		default:
			break;
		}
		
		h ^= h << 3;
		h += h >> 5;
		h ^= h << 4;
		h += h >> 17;
		h ^= h << 25;
		h += h >> 6;

		return h;
	}

	job_pool::job_pool(uint32_t num_threads) : 
		m_kill_flag(false),
		m_num_active_jobs(0)
	{
		assert(num_threads >= 1U);

		debug_printf("job_pool::job_pool: %u total threads\n", num_threads);

		if (num_threads > 1)
		{
			m_threads.resize(num_threads - 1);

			for (int i = 0; i < ((int)num_threads - 1); i++)
			   m_threads[i] = std::thread([this, i] { job_thread(i); });
		}
	}

	job_pool::~job_pool()
	{
		debug_printf("job_pool::~job_pool\n");
		
		// Notify all workers that they need to die right now.
		m_kill_flag = true;
		
		m_has_work.notify_all();

		// Wait for all workers to die.
		for (uint32_t i = 0; i < m_threads.size(); i++)
			m_threads[i].join();
	}
				
	void job_pool::add_job(const std::function<void()>& job)
	{
		std::unique_lock<std::mutex> lock(m_mutex);

		m_queue.emplace_back(job);

		const size_t queue_size = m_queue.size();

		lock.unlock();

		if (queue_size > 1)
			m_has_work.notify_one();
	}

	void job_pool::add_job(std::function<void()>&& job)
	{
		std::unique_lock<std::mutex> lock(m_mutex);

		m_queue.emplace_back(std::move(job));

		const size_t queue_size = m_queue.size();

		lock.unlock();

		if (queue_size > 1)
			m_has_work.notify_one();
	}

	void job_pool::wait_for_all()
	{
		std::unique_lock<std::mutex> lock(m_mutex);

		// Drain the job queue on the calling thread.
		while (!m_queue.empty())
		{
			std::function<void()> job(m_queue.back());
			m_queue.pop_back();

			lock.unlock();

			job();

			lock.lock();
		}

		// The queue is empty, now wait for all active jobs to finish up.
		m_no_more_jobs.wait(lock, [this]{ return !m_num_active_jobs; } );
	}

	void job_pool::job_thread(uint32_t index)
	{
		debug_printf("job_pool::job_thread: starting %u\n", index);
		
		while (true)
		{
			std::unique_lock<std::mutex> lock(m_mutex);

			// Wait for any jobs to be issued.
			m_has_work.wait(lock, [this] { return m_kill_flag || m_queue.size(); } );

			// Check to see if we're supposed to exit.
			if (m_kill_flag)
				break;

			// Get the job and execute it.
			std::function<void()> job(m_queue.back());
			m_queue.pop_back();

			++m_num_active_jobs;

			lock.unlock();

			job();

			lock.lock();

			--m_num_active_jobs;

			// Now check if there are no more jobs remaining. 
			const bool all_done = m_queue.empty() && !m_num_active_jobs;
			
			lock.unlock();

			if (all_done)
				m_no_more_jobs.notify_all();
		}

		debug_printf("job_pool::job_thread: exiting\n");
	}

} // namespace basisu
