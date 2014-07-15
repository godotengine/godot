/*************************************************************************/
/*  test_compression.cpp                                                        */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "test_compression.h"

#include "os/main_loop.h"
#include "os/os.h"
#include "print_string.h"
#include "core/globals.h"

#include "core/io/compression.h"
#include "core/os/file_access.h"
#include "scene/resources/default_theme/font_normal.inc"

namespace TestCompression {

class TestMainLoop : public MainLoop {


	bool quit;

public:
	virtual void input_event(const InputEvent& p_event) {


	}
	virtual bool idle(float p_time) {
		return false;
	}

	virtual void request_quit() {

		quit=true;

	}
	virtual void init() {

		quit=true;
	}
	virtual bool iteration(float p_time) {

		return quit;
	}
	virtual void finish() {

	}


};

void Report(const unsigned char *data, int cnt, String title) {

	print_line("Report for "+title);
	for (int i=0; i<cnt; ++i) {
		print_line("   " + itos(i) + ": " + itos(data[i]));
	}
}

MainLoop* test() {

	print_line("this is test compression");

	Report(_builtin_normal_font_img_data, 10, "Raw");

	print_line("try compress data:");

	const unsigned char* src = _builtin_normal_font_img_data;
	int src_size = _builtin_normal_font_img_width*_builtin_normal_font_img_height*2;
	print_line(" src size is :" + itos(src_size));
	Compression::Mode cmode = Compression::MODE_DEFLATE;
	int dst_size = Compression::get_max_compressed_buffer_size(src_size,cmode);
	print_line(" dst size is :" + itos(dst_size));
	if (dst_size==-1)
		dst_size=1024*1024*2;
	unsigned char* dst = new unsigned char[dst_size];
	int compressed_size = Compression::compress(dst,src,src_size,cmode);
	print_line(" compressed to " + itos(compressed_size) + " bytes.");
	Report(dst, 10, "Compressed:");

	print_line("try de-compress data:");
	const unsigned char* src2 = (const unsigned char*)dst;
	int src_size2 = compressed_size;
	int dst_size2=1024*1024*2;
	unsigned char* dst2 = new unsigned char[dst_size2];
	Compression::decompress(dst2,dst_size2,src2,src_size2,Compression::MODE_DEFLATE);
	Report(dst2, 10, "De-Compressed:");

	delete[] dst;
	delete[] dst2;
	/*
	print_line("try read png");
	for (int i=0; i<6; ++i ) {
		String png = "normal_font_" + itos(i) + "_ga.png";
		FileAccess *input = FileAccess::open(png, FileAccess::READ);
		int len = input->get_len();
		Vector<uint8_t> png_data;
		png_data.resize(len);
		input->get_buffer(&png_data[0], len);
		input->close();
		memdelete(input);

		String zip = png + ".zip";
		Vector<uint8_t> zip_data;
		int max_buffer_size = Compression::get_max_compressed_buffer_size(len, cmode);
		zip_data.resize(max_buffer_size);
		int acutal_compressed = Compression::compress(&zip_data[0],&png_data[0],len,cmode);
		zip_data.resize(acutal_compressed);
		FileAccess *output = FileAccess::open(zip, FileAccess::WRITE);
		output->store_buffer(&zip_data[0],acutal_compressed);
		output->close();
		memdelete(output);
	}
	print_line("try write png");
	for (int i=0; i<6; ++i ) {

		String zip = "normal_font_" + itos(i) + "_ga.png.zip";
		FileAccess *input = FileAccess::open(zip, FileAccess::READ);
		int len = input->get_len();
		Vector<uint8_t> zip_data;
		zip_data.resize(len);
		input->get_buffer(&zip_data[0], len);
		input->close();
		memdelete(input);

		Vector<uint8_t> png_data;
		int png_size = 1024*1024*2;
		png_data.resize(png_size);
		FileAccess *output = FileAccess::open(png_data, FileAccess::WRITE);

		output->store_buffer(&png_data[0],png_size);
		output->close();
		memdelete(output);

		String png = zip + ".png";

	}
	*/
	print_line("test done");

	return memnew( TestMainLoop );
}

}


