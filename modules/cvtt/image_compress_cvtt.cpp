/**************************************************************************/
/*  image_compress_cvtt.cpp                                               */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "image_compress_cvtt.h"

#include "core/object/worker_thread_pool.h"
#include "core/os/os.h"
#include "core/string/print_string.h"
#include "core/templates/safe_refcount.h"

#include <ConvectionKernels.h>

struct CVTTCompressionJobParams {
	bool is_hdr = false;
	bool is_signed = false;
	int bytes_per_pixel = 0;
	cvtt::BC7EncodingPlan bc7_plan;
	cvtt::Options options;
};

struct CVTTCompressionRowTask {
	Vector<uint8_t> in_mm;
	uint8_t *out_mm_bytes = nullptr;
	int y_start = 0;
	int width = 0;
	int height = 0;
};

struct CVTTCompressionJobQueue {
	CVTTCompressionJobParams job_params;
	const CVTTCompressionRowTask *job_tasks = nullptr;
	uint32_t num_tasks = 0;
	SafeNumeric<uint32_t> current_task;
};

static void _digest_row_task(const CVTTCompressionJobParams &p_job_params, const CVTTCompressionRowTask &p_row_task) {
	const uint8_t *in_bytes = p_row_task.in_mm.ptr();
	uint8_t *out_bytes = p_row_task.out_mm_bytes;
	int w = p_row_task.width;
	int h = p_row_task.height;

	int y_start = p_row_task.y_start;
	int y_end = y_start + 4;

	int bytes_per_pixel = p_job_params.bytes_per_pixel;
	bool is_hdr = p_job_params.is_hdr;
	bool is_signed = p_job_params.is_signed;

	cvtt::PixelBlockU8 input_blocks_ldr[cvtt::NumParallelBlocks];
	cvtt::PixelBlockF16 input_blocks_hdr[cvtt::NumParallelBlocks];

	for (int x_start = 0; x_start < w; x_start += 4 * cvtt::NumParallelBlocks) {
		int x_end = x_start + 4 * cvtt::NumParallelBlocks;

		for (int y = y_start; y < y_end; y++) {
			int first_input_element = (y - y_start) * 4;
			const uint8_t *row_start;
			if (y >= h) {
				row_start = in_bytes + (h - 1) * (w * bytes_per_pixel);
			} else {
				row_start = in_bytes + y * (w * bytes_per_pixel);
			}

			for (int x = x_start; x < x_end; x++) {
				const uint8_t *pixel_start;
				if (x >= w) {
					pixel_start = row_start + (w - 1) * bytes_per_pixel;
				} else {
					pixel_start = row_start + x * bytes_per_pixel;
				}

				int block_index = (x - x_start) / 4;
				int block_element = (x - x_start) % 4 + first_input_element;
				if (is_hdr) {
					memcpy(input_blocks_hdr[block_index].m_pixels[block_element], pixel_start, bytes_per_pixel);
					input_blocks_hdr[block_index].m_pixels[block_element][3] = 0x3c00; // 1.0 (unused)
				} else {
					memcpy(input_blocks_ldr[block_index].m_pixels[block_element], pixel_start, bytes_per_pixel);
				}
			}
		}

		uint8_t output_blocks[16 * cvtt::NumParallelBlocks];

		if (is_hdr) {
			if (is_signed) {
				cvtt::Kernels::EncodeBC6HS(output_blocks, input_blocks_hdr, p_job_params.options);
			} else {
				cvtt::Kernels::EncodeBC6HU(output_blocks, input_blocks_hdr, p_job_params.options);
			}
		} else {
			cvtt::Kernels::EncodeBC7(output_blocks, input_blocks_ldr, p_job_params.options, p_job_params.bc7_plan);
		}

		unsigned int num_real_blocks = ((w - x_start) + 3) / 4;
		if (num_real_blocks > cvtt::NumParallelBlocks) {
			num_real_blocks = cvtt::NumParallelBlocks;
		}

		memcpy(out_bytes, output_blocks, 16 * num_real_blocks);
		out_bytes += 16 * num_real_blocks;
	}
}

static void _digest_job_queue(void *p_job_queue, uint32_t p_index) {
	CVTTCompressionJobQueue *job_queue = static_cast<CVTTCompressionJobQueue *>(p_job_queue);
	uint32_t num_tasks = job_queue->num_tasks;
	uint32_t total_threads = WorkerThreadPool::get_singleton()->get_thread_count();
	uint32_t start = p_index * num_tasks / total_threads;
	uint32_t end = (p_index + 1 == total_threads) ? num_tasks : ((p_index + 1) * num_tasks / total_threads);

	for (uint32_t i = start; i < end; i++) {
		_digest_row_task(job_queue->job_params, job_queue->job_tasks[i]);
	}
}

void image_compress_cvtt(Image *p_image, Image::UsedChannels p_channels) {
	uint64_t start_time = OS::get_singleton()->get_ticks_msec();

	if (p_image->is_compressed()) {
		return; //do not compress, already compressed
	}

	int w = (p_image->get_width() + 3) & ~0x03;
	int h = (p_image->get_height() + 3) & ~0x03;

	if (p_image->get_width() != w || p_image->get_height() != h) {
		// Align the image to 4x4 texels.
		p_image->resize(w, h, Image::INTERPOLATE_NEAREST);
	}

	bool is_ldr = (p_image->get_format() <= Image::FORMAT_RGBA8);
	bool is_hdr = (p_image->get_format() >= Image::FORMAT_RF) && (p_image->get_format() <= Image::FORMAT_RGBE9995);

	if (!is_ldr && !is_hdr) {
		return; // Not a usable source format
	}

	cvtt::Options options;
	uint32_t flags = cvtt::Flags::Default;
	flags |= cvtt::Flags::BC7_RespectPunchThrough;
	if (p_channels == Image::USED_CHANNELS_RG) { //guessing this is a normal map
		flags |= cvtt::Flags::Uniform;
	}
	options.flags = flags;

	Image::Format target_format = Image::FORMAT_BPTC_RGBA;

	bool is_signed = false;
	if (is_hdr) {
		if (p_image->get_format() != Image::FORMAT_RGBH) {
			p_image->convert(Image::FORMAT_RGBH);
		}

		is_signed = p_image->detect_signed();
		target_format = is_signed ? Image::FORMAT_BPTC_RGBF : Image::FORMAT_BPTC_RGBFU;
	} else {
		p_image->convert(Image::FORMAT_RGBA8); //still uses RGBA to convert
	}

	Vector<uint8_t> data;
	int64_t target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
	int mm_count = p_image->has_mipmaps() ? Image::get_image_required_mipmaps(w, h, target_format) : 0;
	data.resize(target_size);
	int shift = Image::get_format_pixel_rshift(target_format);

	uint8_t *wb = data.ptrw();

	int64_t dst_ofs = 0;

	CVTTCompressionJobQueue job_queue;
	job_queue.job_params.is_hdr = is_hdr;
	job_queue.job_params.is_signed = is_signed;
	job_queue.job_params.options = options;
	job_queue.job_params.bytes_per_pixel = is_hdr ? 6 : 4;
	cvtt::Kernels::ConfigureBC7EncodingPlanFromQuality(job_queue.job_params.bc7_plan, 5);

	// Amdahl's law (Wikipedia)
	// If a program needs 20 hours to complete using a single thread, but a one-hour portion of the program cannot be parallelized,
	// therefore only the remaining 19 hours (p = 0.95) of execution time can be parallelized, then regardless of how many threads are devoted
	// to a parallelized execution of this program, the minimum execution time cannot be less than one hour.
	//
	// The number of executions with different inputs can be increased while the latency is the same.

	Vector<CVTTCompressionRowTask> tasks;

	for (int i = 0; i <= mm_count; i++) {
		Vector<uint8_t> in_data;
		int width, height;
		Image::get_image_mipmap_offset_and_dimensions(w, h, target_format, i, width, height);

		int bw = width % 4 != 0 ? width + (4 - width % 4) : width;
		int bh = height % 4 != 0 ? height + (4 - height % 4) : height;

		int64_t src_mip_ofs, src_mip_size;
		int src_mip_w, src_mip_h;
		p_image->get_mipmap_offset_size_and_dimensions(i, src_mip_ofs, src_mip_size, src_mip_w, src_mip_h);

		// Pad textures to nearest block by smearing.
		if (width != src_mip_w || height != src_mip_h) {
			const uint8_t *src_mip_read = p_image->ptr() + src_mip_ofs;

			// Reserve the buffer for padded image data.
			int px_size = Image::get_format_pixel_size(p_image->get_format());
			in_data.resize(width * height * px_size);
			uint8_t *ptrw = in_data.ptrw();

			int x = 0, y = 0;
			for (y = 0; y < src_mip_h; y++) {
				for (x = 0; x < src_mip_w; x++) {
					memcpy(ptrw + (width * y + x) * px_size, src_mip_read + (src_mip_w * y + x) * px_size, px_size);
				}

				// First, smear in x.
				for (; x < width; x++) {
					memcpy(ptrw + (width * y + x) * px_size, ptrw + (width * y + x - 1) * px_size, px_size);
				}
			}

			// Then, smear in y.
			for (; y < height; y++) {
				for (x = 0; x < width; x++) {
					memcpy(ptrw + (width * y + x) * px_size, ptrw + (width * y + x - width) * px_size, px_size);
				}
			}
		} else {
			// Create a buffer filled with the source mip layer data.
			in_data.resize(src_mip_size);
			memcpy(in_data.ptrw(), p_image->ptr() + src_mip_ofs, src_mip_size);
		}

		//const uint8_t *in_bytes = &rb[src_ofs];
		uint8_t *out_bytes = &wb[dst_ofs];

		for (int y_start = 0; y_start < height; y_start += 4) {
			CVTTCompressionRowTask row_task;
			row_task.width = width;
			row_task.height = height;
			row_task.y_start = y_start;
			row_task.in_mm = in_data;
			row_task.out_mm_bytes = out_bytes;

			tasks.push_back(row_task);

			out_bytes += 16 * (bw / 4);
		}

		dst_ofs += (MAX(4, bw) * MAX(4, bh)) >> shift;
	}

	const CVTTCompressionRowTask *tasks_rb = tasks.ptr();

	job_queue.job_tasks = &tasks_rb[0];
	job_queue.num_tasks = static_cast<uint32_t>(tasks.size());
	WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_native_group_task(&_digest_job_queue, &job_queue, WorkerThreadPool::get_singleton()->get_thread_count(), -1, true, SNAME("CVTT Compress"));
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);

	p_image->set_data(w, h, p_image->has_mipmaps(), target_format, data);

	print_verbose(vformat("CVTT: Encoding took %d ms.", OS::get_singleton()->get_ticks_msec() - start_time));
}

void image_decompress_cvtt(Image *p_image) {
	Image::Format target_format;
	bool is_signed = false;
	bool is_hdr = false;

	Image::Format input_format = p_image->get_format();

	switch (input_format) {
		case Image::FORMAT_BPTC_RGBA:
			target_format = Image::FORMAT_RGBA8;
			break;
		case Image::FORMAT_BPTC_RGBF:
		case Image::FORMAT_BPTC_RGBFU:
			target_format = Image::FORMAT_RGBH;
			is_signed = (input_format == Image::FORMAT_BPTC_RGBF);
			is_hdr = true;
			break;
		default:
			return; // Invalid input format
	};

	int w = p_image->get_width();
	int h = p_image->get_height();

	const uint8_t *rb = p_image->get_data().ptr();

	Vector<uint8_t> data;
	int64_t target_size = Image::get_image_data_size(w, h, target_format, p_image->has_mipmaps());
	int mm_count = p_image->get_mipmap_count();
	data.resize(target_size);

	uint8_t *wb = data.ptrw();

	int bytes_per_pixel = is_hdr ? 6 : 4;

	int64_t dst_ofs = 0;

	for (int i = 0; i <= mm_count; i++) {
		int64_t src_ofs = p_image->get_mipmap_offset(i);

		const uint8_t *in_bytes = &rb[src_ofs];
		uint8_t *out_bytes = &wb[dst_ofs];

		cvtt::PixelBlockU8 output_blocks_ldr[cvtt::NumParallelBlocks];
		cvtt::PixelBlockF16 output_blocks_hdr[cvtt::NumParallelBlocks];

		for (int y_start = 0; y_start < h; y_start += 4) {
			int y_end = y_start + 4;

			for (int x_start = 0; x_start < w; x_start += 4 * cvtt::NumParallelBlocks) {
				uint8_t input_blocks[16 * cvtt::NumParallelBlocks];
				memset(input_blocks, 0, sizeof(input_blocks));

				unsigned int num_real_blocks = ((w - x_start) + 3) / 4;
				if (num_real_blocks > cvtt::NumParallelBlocks) {
					num_real_blocks = cvtt::NumParallelBlocks;
				}

				memcpy(input_blocks, in_bytes, 16 * num_real_blocks);
				in_bytes += 16 * num_real_blocks;

				int x_end = x_start + 4 * num_real_blocks;

				if (is_hdr) {
					if (is_signed) {
						cvtt::Kernels::DecodeBC6HS(output_blocks_hdr, input_blocks);
					} else {
						cvtt::Kernels::DecodeBC6HU(output_blocks_hdr, input_blocks);
					}
				} else {
					cvtt::Kernels::DecodeBC7(output_blocks_ldr, input_blocks);
				}

				for (int y = y_start; y < y_end; y++) {
					int first_input_element = (y - y_start) * 4;
					uint8_t *row_start;
					if (y >= h) {
						row_start = out_bytes + (h - 1) * (w * bytes_per_pixel);
					} else {
						row_start = out_bytes + y * (w * bytes_per_pixel);
					}

					for (int x = x_start; x < x_end; x++) {
						uint8_t *pixel_start;
						if (x >= w) {
							pixel_start = row_start + (w - 1) * bytes_per_pixel;
						} else {
							pixel_start = row_start + x * bytes_per_pixel;
						}

						int block_index = (x - x_start) / 4;
						int block_element = (x - x_start) % 4 + first_input_element;
						if (is_hdr) {
							memcpy(pixel_start, output_blocks_hdr[block_index].m_pixels[block_element], bytes_per_pixel);
						} else {
							memcpy(pixel_start, output_blocks_ldr[block_index].m_pixels[block_element], bytes_per_pixel);
						}
					}
				}
			}
		}

		dst_ofs += w * h * bytes_per_pixel;
		w >>= 1;
		h >>= 1;
	}
	p_image->set_data(p_image->get_width(), p_image->get_height(), p_image->has_mipmaps(), target_format, data);
}
