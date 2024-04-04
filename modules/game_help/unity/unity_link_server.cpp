/**************************************************************************/
/*  editor_file_server.cpp                                                */
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

#include "unity_link_server.h"

#include "core/io/marshalls.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "../logic/body_animator.h"

#define FILESYSTEM_PROTOCOL_VERSION 1
#define PASSWORD_LENGTH 32
#define MAX_FILE_BUFFER_SIZE 100 * 1024 * 1024 // 100mb max file buffer size (description of files to update, compressed).


static uint8_t*  _buffer_to_animationNode(uint8_t* buffer,Ref<CharacterAnimatorNodeBase> &rs,Error &err) ;
// 解析动画节点
static uint8_t* _buffer_to_animationItem(uint8_t* buffer,CharacterAnimatorNodeBase::AnimationItem &animation_item,Error &err) 
{
	
	CharacterAnimatorNodeBase::AnimationItem item;

	bool is_clip = *(bool*)buffer;
	buffer += 1;

	if(is_clip){
		int name_size = *(int*)buffer;
		if(name_size < 0 || name_size > 10240){
			err = ERR_OUT_OF_MEMORY;
			return buffer;
		}
		buffer += 4;
		item.m_Name = String::utf8((const char*)buffer,name_size);
		buffer += name_size;
	}else{
		buffer = _buffer_to_animationNode(buffer,item.m_animation_node,err);
	}

	return buffer;
}
static uint8_t* _buffer_to_animationNode(uint8_t*  buffer,Ref<CharacterAnimatorNodeBase> &rs,Error &err) 
{
	err = OK;
	// 1 代表是动画节点文件
	int type = *(int*)buffer;
	if(type < 0 || type > 10240){
		err = ERR_OUT_OF_MEMORY;
		return buffer;
	}
	if(type < 0 || type > 10240){
		err = ERR_OUT_OF_MEMORY;
		return buffer;
	}
	buffer += 4;
	Ref<CharacterAnimatorNode1D> node1d;
	Ref<CharacterAnimatorNode2D> node2d;
	if(type == 0){
		// 0 代表是CharacterAnimatorNode1D
		node1d .instantiate();
		rs = node1d;
	}else if(type > 1){
		// 1 代表是CharacterAnimatorNode2D
		node2d.instantiate();
		rs = node2d;
		node2d->m_BlendType = (CharacterAnimatorNode2D::BlendType)*buffer;
	}

	int name_length = *(int*)buffer;
	if(name_length < 0 || name_length > 10240){
		err = ERR_OUT_OF_MEMORY;
		return buffer;
	}
	buffer += 4;

	String name = String::utf8((const char*)buffer,name_length);
	rs->set_name(name);
	buffer += name_length;

	int item_count = *(int*)buffer;
	if(item_count < 0 || item_count > 10240){
		err = ERR_OUT_OF_MEMORY;
		return buffer;
	}
	buffer += 4;

	// 读取绑定的属性名称
	int pro_name_len = *(int*)buffer;
	if(pro_name_len < 0 || pro_name_len > 10240){
		err = ERR_OUT_OF_MEMORY;
		return buffer;
	}
	buffer += 4;
	rs->m_PropertyName = String::utf8((const char*)buffer,pro_name_len);
	buffer += pro_name_len;

	if(type == 0){
		// 0 代表是CharacterAnimatorNode1D
		node1d->m_BlendData.m_ChildCount = item_count;
	}else if(type > 1){
		// 1 代表是CharacterAnimatorNode2D
		node2d->m_BlendData.m_ChildCount = item_count;
	}
	for(int i=0;i<item_count;i++){
		float x = *(float*)buffer;
		buffer += 4;
		float y = *(float*)buffer;
		buffer += 4;
		if(type == 0){
			// 0 代表是CharacterAnimatorNode1D
			node1d->m_BlendData.m_ChildThresholdArray.push_back(x);
		}else if(type > 1){
			// 1 代表是CharacterAnimatorNode2D
			node2d->m_BlendData.m_ChildPositionArray.push_back(Vector2(x,y));
		}
		CharacterAnimatorNodeBase::AnimationItem animation_item;
		buffer = _buffer_to_animationItem(buffer,animation_item,err);
		if(err != OK){
			return buffer;
		}
		rs->m_ChildAnimationArray.append(animation_item);

	}

	return buffer;
}

void UnityLinkServer::poll() {
	if (!active) {
		return;
	}

	if (!server->is_connection_available()) {
		return;
	}

	Ref<StreamPeerTCP> tcp_peer = server->take_connection();
	ERR_FAIL_COND(tcp_peer.is_null());
	int size = tcp_peer->get_u32();
	uint8_t* buffer = (uint8_t*)alloca(size);
	if(!tcp_peer->get_data(buffer,size)){
		return;
	}
	if(size == 0){
		return;
	}
	int type = *(int*)buffer;
	buffer += 4;
	int path_size = *(int*)buffer;
	buffer += 4;
	String path = String::utf8((const char*)buffer,path_size);
	buffer += path_size;

	if(!DirAccess::exists("res://" + path))
	{
		// 创建目录
		Ref<DirAccess> dir_access = DirAccess::create(DirAccess::AccessType::ACCESS_RESOURCES);
		dir_access->make_dir_recursive(path);
	}


	if(type == 1){
		// 1 代表是动画节点
		Ref<CharacterAnimatorNodeBase> anima_node;
		Error err;
		buffer = _buffer_to_animationNode(buffer,anima_node,err);
		if(err != OK){
			ERR_FAIL_MSG("UnityLinkServer: create animation node error " + itos(err) + " " + path);
			return;
		}
		String save_path = "res://" + path + "/" + anima_node->get_name() + "_" + anima_node->get_class() + "anim_node.tres";
		ResourceSaver::save(anima_node,save_path);
	}
	else if(type == 2){
		if(on_load_animation.is_null()){
			return;
		}
		// 解析动画文件
		int file_size = size - 8 - path_size;
		Vector<uint8_t> ba = Vector<uint8_t>();
		ba.resize(file_size);
		memcpy(ba.ptrw(),buffer,file_size);
		Ref<Animation> anim;
		anim.instantiate();
		on_load_animation.call(ba,anim);
		String save_path = "res://" + path + "/" + anim->get_class() + "_" + anim->get_name() +  ".anim.tres";
		ResourceSaver::save(anim,save_path);		
	}
	else if(type == 3){
		// 直接存儲的文件，fbx，png。。。。
		
		int file_size = size - 8 - path_size;

		int name_size = *(int*)buffer;
		buffer += 4;
		String name = String::utf8((const char*)buffer,name_size);
		buffer += name_size;

		file_size -= name_size + 4;
		Ref<FileAccess> f = FileAccess::open("res://" + path + "/" + name,FileAccess::WRITE);
		f->store_buffer(buffer,file_size);	
	}

}

void UnityLinkServer::start() {
	if (active) {
		stop();
	}
	port = 9010;
	Error err = server->listen(port);
	ERR_FAIL_COND_MSG(err != OK, "UnityLinkServer: Unable to listen on port " + itos(port));
	active = true;
}

bool UnityLinkServer::is_active() const {
	return active;
}

void UnityLinkServer::stop() {
	if (active) {
		server->stop();
		active = false;
	}
}

UnityLinkServer::UnityLinkServer() {
	server.instantiate();

}

UnityLinkServer::~UnityLinkServer() {
	stop();
}
