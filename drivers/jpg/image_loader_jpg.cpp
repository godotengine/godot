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

#include "image_loader_jpg.h"

#include "print_string.h"
#include "os/os.h"
#include "drivers/jpg/tinyjpeg.h"


static void* _tinyjpg_alloc(unsigned int amount) {

	return memalloc(amount);
}

static void _tinyjpg_free(void *ptr) {

	memfree(ptr);
}

Error ImageLoaderJPG::load_image(Image *p_image,FileAccess *f) {


	DVector<uint8_t> src_image;
	int src_image_len = f->get_len();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	src_image.resize(src_image_len);

	DVector<uint8_t>::Write w = src_image.write();

	f->get_buffer(&w[0],src_image_len);

	f->close();

	jdec_private* jdec=tinyjpeg_init(_tinyjpg_alloc,_tinyjpg_free);
	ERR_FAIL_COND_V(!jdec,ERR_UNAVAILABLE);

	int ret = tinyjpeg_parse_header(jdec,&w[0],src_image_len);

	if (ret!=0) {
		tinyjpeg_free(jdec);
	}

	ERR_FAIL_COND_V(ret!=0,ERR_FILE_CORRUPT);

	unsigned int width,height;


	tinyjpeg_get_size(jdec,&width,&height);



	DVector<uint8_t> imgdata;
	imgdata.resize(width*height*3);
	DVector<uint8_t>::Write imgdataw = imgdata.write();


	unsigned char *components[1]={&imgdataw[0]};
	tinyjpeg_set_components(jdec,components,1);
	tinyjpeg_decode(jdec,TINYJPEG_FMT_RGB24);
	imgdataw = DVector<uint8_t>::Write();

	Image dst_image(width,height,0,Image::FORMAT_RGB,imgdata);

	tinyjpeg_free(jdec);

	*p_image=dst_image;

	return OK;

}

void ImageLoaderJPG::get_recognized_extensions(List<String> *p_extensions) const {
	
	p_extensions->push_back("jpg");
	p_extensions->push_back("jpeg");
}


ImageLoaderJPG::ImageLoaderJPG() {


}


