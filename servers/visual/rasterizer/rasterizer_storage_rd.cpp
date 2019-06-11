#include "rasterizer_storage_rd.h"

RID RasterizerStorageRD::texture_2d_create(const Ref<Image> &p_image) {

	return RID();
}

RID RasterizerStorageRD::texture_2d_layered_create(const Vector<Ref<Image> > &p_layers, VS::TextureLayeredType p_layered_type) {

	return RID();
}
RID RasterizerStorageRD::texture_3d_create(const Vector<Ref<Image> > &p_slices) {

	return RID();
}

void RasterizerStorageRD::texture_2d_update_immediate(RID p_texture, const Ref<Image> &p_image, int p_layer) {
}
void RasterizerStorageRD::texture_2d_update(RID p_texture, const Ref<Image> &p_image, int p_layer) {
}
void RasterizerStorageRD::texture_3d_update(RID p_texture, const Ref<Image> &p_image, int p_depth, int p_mipmap) {
}

//these two APIs can be used together or in combination with the others.
RID RasterizerStorageRD::texture_2d_placeholder_create() {

	return RID();
}
RID RasterizerStorageRD::texture_2d_layered_placeholder_create() {

	return RID();
}
RID RasterizerStorageRD::texture_3d_placeholder_create() {

	return RID();
}

Ref<Image> RasterizerStorageRD::texture_2d_get(RID p_texture) const {

	return Ref<Image>();
}
Ref<Image> RasterizerStorageRD::texture_2d_layer_get(RID p_texture, int p_layer) const {

	return Ref<Image>();
}
Ref<Image> RasterizerStorageRD::texture_3d_slice_get(RID p_texture, int p_depth, int p_mipmap) const {

	return Ref<Image>();
}

void RasterizerStorageRD::texture_replace(RID p_texture, RID p_by_texture) {
}
void RasterizerStorageRD::texture_set_size_override(RID p_texture, int p_width, int p_height) {
}

void RasterizerStorageRD::texture_set_path(RID p_texture, const String &p_path) {
}
String RasterizerStorageRD::texture_get_path(RID p_texture) const {
	return String();
}

void RasterizerStorageRD::texture_set_detect_3d_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_set_detect_normal_callback(RID p_texture, VS::TextureDetectCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_set_detect_roughness_callback(RID p_texture, VS::TextureDetectRoughnessCallback p_callback, void *p_userdata) {
}
void RasterizerStorageRD::texture_debug_usage(List<VS::TextureInfo> *r_info) {
}

void RasterizerStorageRD::texture_set_proxy(RID p_proxy, RID p_base) {
}
void RasterizerStorageRD::texture_set_force_redraw_if_visible(RID p_texture, bool p_enable) {
}

Size2 RasterizerStorageRD::texture_size_with_proxy(RID p_proxy) const {
	return Size2();
}
