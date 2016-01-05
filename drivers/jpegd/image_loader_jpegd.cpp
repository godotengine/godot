/*************************************************/
/*  image_loader_jpg.cpp                         */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "image_loader_jpegd.h"

#include "print_string.h"
#include "os/os.h"
#include "jpgd.h"
#include <string.h>


Error ImageLoaderJPG::load_image(Image *p_image,FileAccess *f) {


	DVector<uint8_t> src_image;
	int src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	DVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0],src_image_len);

	f->close();



	jpgd::jpeg_decoder_mem_stream mem_stream(w.ptr(),src_image_len);

	jpgd::jpeg_decoder decoder(&mem_stream);

	if (decoder.get_error_code() != jpgd::JPGD_SUCCESS) {
		return ERR_CANT_OPEN;
	}

	const int image_width = decoder.get_width();
	const int image_height = decoder.get_height();
	int comps = decoder.get_num_components();
	if (comps==3)
		comps=4; //weird

	if (decoder.begin_decoding() != jpgd::JPGD_SUCCESS)
		return ERR_FILE_CORRUPT;

	const int dst_bpl = image_width * comps;

	DVector<uint8_t> data;

	data.resize(dst_bpl * image_height);

	DVector<uint8_t>::Write dw = data.write();

	jpgd::uint8 *pImage_data = (jpgd::uint8*)dw.ptr();

	for (int y = 0; y < image_height; y++)
	{
		const jpgd::uint8* pScan_line;
		jpgd::uint scan_line_len;
		if (decoder.decode((const void**)&pScan_line, &scan_line_len) != jpgd::JPGD_SUCCESS)
		{
			return ERR_FILE_CORRUPT;
		}

		jpgd::uint8 *pDst = pImage_data + y * dst_bpl;
		memcpy(pDst, pScan_line, dst_bpl);


	}


	//all good

	Image::Format fmt;
	if (comps==1)
		fmt=Image::FORMAT_GRAYSCALE;
	else
		fmt=Image::FORMAT_RGBA;

	dw = DVector<uint8_t>::Write();
	w = DVector<uint8_t>::Write();

	p_image->create(image_width,image_height,0,fmt,data);

	return OK;

}

void ImageLoaderJPG::get_recognized_extensions(List<String> *p_extensions) const {
	
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}


ImageLoaderJPG::ImageLoaderJPG() {


}


