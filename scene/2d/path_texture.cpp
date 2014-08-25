#include "path_texture.h"


void PathTexture::set_begin_texture(const Ref<Texture>& p_texture) {

	begin=p_texture;
	update();
}

Ref<Texture> PathTexture::get_begin_texture() const{

	return begin;
}

void PathTexture::set_repeat_texture(const Ref<Texture>& p_texture){

	repeat=p_texture;
	update();

}
Ref<Texture> PathTexture::get_repeat_texture() const{

	return repeat;
}

void PathTexture::set_end_texture(const Ref<Texture>& p_texture){

	end=p_texture;
	update();
}
Ref<Texture> PathTexture::get_end_texture() const{

	return end;
}

void PathTexture::set_subdivisions(int p_amount){

	ERR_FAIL_INDEX(p_amount,32);
	subdivs=p_amount;
	update();

}

int PathTexture::get_subdivisions() const{

	return subdivs;
}

void PathTexture::set_overlap(int p_amount){

	overlap=p_amount;
	update();
}
int PathTexture::get_overlap() const{

	return overlap;
}


PathTexture::PathTexture() {

	overlap=0;
	subdivs=1;
}
