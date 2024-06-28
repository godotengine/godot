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
#include "../logic/animator/body_animator.h"
#include "../logic/data_table_manager.h"

#if TOOLS_ENABLED
#include "editor/import/3d/resource_importer_scene.h"
#include "core/io/config_file.h"
#endif

#define FILESYSTEM_PROTOCOL_VERSION 1
#define PASSWORD_LENGTH 32
#define MAX_FILE_BUFFER_SIZE 100 * 1024 * 1024 // 100mb max file buffer size (description of files to update, compressed).

UnityLinkServer * UnityLinkServer::instance = nullptr;
static void  _buffer_to_animationNode(StreamPeerConstBuffer& buffer,Ref<CharacterAnimatorNodeBase> &rs,Error &err) ;
// 解析动画节点
static void _buffer_to_animationItem(StreamPeerConstBuffer& buffer,Ref<CharacterAnimationItem> &animation_item,Error &err) 
{
	err = OK;
	animation_item.instantiate();

	bool is_clip = buffer.get_8();

	if(is_clip){
		int name_size = buffer.get_32();
		if(name_size < 0 || name_size > 10240){
			err = ERR_OUT_OF_MEMORY;
			return ;
		}
		animation_item->animation_name = buffer.get_utf8_string(name_size);;

		int path_size = buffer.get_32();
		if(path_size < 0 || path_size > 10240){
			err = ERR_OUT_OF_MEMORY;
			return ;			
		}

		if(buffer.is_end()){
			err = ERR_OUT_OF_MEMORY;
			return ;
		}
		animation_item->animation_path = buffer.get_utf8_string(path_size);


		
		int bone_map_path_size = buffer.get_32();
		if(bone_map_path_size < 0 || bone_map_path_size > 10240){
			err = ERR_OUT_OF_MEMORY;
			return ;			
		}

		if(buffer.is_end()){
			err = ERR_OUT_OF_MEMORY;
			return ;
		}
		animation_item->bone_map_path = buffer.get_utf8_string(bone_map_path_size);

		animation_item->speed = buffer.get_float();


	}else{
		_buffer_to_animationNode(buffer,animation_item->child_node,err);
	}

	return ;
}
static void _buffer_to_animationNode(StreamPeerConstBuffer&  buffer,Ref<CharacterAnimatorNodeBase> &rs,Error &err) 
{
	err = OK;
	// 1 代表是动画节点文件
	int type = buffer.get_32();
	if(type < 0 || type > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	if(type < 0 || type > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
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
		node2d->blend_type = (CharacterAnimatorNode2D::BlendType)buffer.get_32();
	}

	int name_length = buffer.get_32();
	if(name_length < 0 || name_length > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	if(buffer.is_end()){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	String name = buffer.get_utf8_string(name_length);
	rs->set_name(name);

	int item_count = buffer.get_32();
	if(item_count < 0 || item_count > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	if(buffer.is_end()){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}

	// 读取绑定的属性名称
	int pro_name_len = buffer.get_32();
	if(pro_name_len < 0 || pro_name_len > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	rs->set_black_board_property(buffer.get_utf8_string(pro_name_len));
	pro_name_len = buffer.get_32();
	if(pro_name_len < 0 || pro_name_len > 10240){
		err = ERR_OUT_OF_MEMORY;
		return ;
	}
	rs->set_black_board_property_y( buffer.get_utf8_string(pro_name_len));

	if(type == 0){
		// 0 代表是CharacterAnimatorNode1D
		node1d->blend_data.position_count = item_count;
	}else if(type > 1){
		// 1 代表是CharacterAnimatorNode2D
		node2d->blend_data.position_count = item_count;
	}
	for(int i=0;i<item_count;i++){
		float x = buffer.get_float();
		float y = buffer.get_float();
		if(buffer.is_end()){
			err = ERR_OUT_OF_MEMORY;
			return ;
		}
		if(type == 0){
			// 0 代表是CharacterAnimatorNode1D
			node1d->blend_data.position_array.push_back(x);
		}else if(type > 1){
			// 1 代表是CharacterAnimatorNode2D
			node2d->blend_data.position_array.push_back(Vector2(x,y));
		}
		Ref<CharacterAnimationItem> animation_item;
		_buffer_to_animationItem(buffer,animation_item,err);
		if(err != OK){
			return ;
		}
		rs->_add_animation_item(animation_item);

	}

	return ;
}
static bool read_string(StreamPeerConstBuffer& msg_buffer,String &str)
{
	int len = msg_buffer.get_u32();
	if (len < 0 || len > 10240) {
		return false;
	}
	str = msg_buffer.get_utf8_string(len);
	return true;

}
static bool poll_client(StreamPeerConstBuffer& msg_buffer) {
	int tag = msg_buffer.get_u32();
	int size = msg_buffer.get_u32();
	int type = msg_buffer.get_u32();
	String path;
	if (!read_string(msg_buffer, path))
	{
		ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + path);

	}
	path = path.to_lower();
	path = path.replace("assets", "Assets");
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
		_buffer_to_animationNode(msg_buffer,anima_node,err);
		if(err != OK){
			ERR_FAIL_V_MSG(false,"UnityLinkServer: create animation node error " + itos(err) + " " + path);
		}
		String save_path = "res://" + path + "/" + anima_node->get_name() + "_" + anima_node->get_class() + "anim_node.tres";
		ResourceSaver::save(anima_node,save_path);
	}
	else if(type == 2){
		// 解析骨骼映射文件
		int file_size = msg_buffer.get_32();
		if(file_size < 0 || file_size > 10240){
			ERR_FAIL_V_MSG(false,"UnityLinkServer: create bone map error :" + path);
			
		}
		String file_name = msg_buffer.get_utf8_string(file_size);
		String save_name = file_name.get_basename();

		Ref<CharacterBoneMap> bone_map;
		bone_map.instantiate();
		//bone_map->ref_skeleton_file_path = path + "/" + file_name;
		//bone_map->is_by_sekeleton_file = true;


		String save_path = "res://" + path + "/" + save_name +  ".bone_map.tres";
		ResourceSaver::save(bone_map,save_path);		
		print_line("UnityLinkServer: save bone map :" + save_path);
	}
	else if(type == 3){
		Callable on_load_animation =  DataTableManager::get_singleton()->get_animation_load_cb();
		if(on_load_animation.is_null()){
			return true;
		}
		String name;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}

		// 解析动画文件
		int mirror = msg_buffer.get_32();
		
		int file_size = size - msg_buffer.get_position();
		String yaml_anim = msg_buffer.get_utf8_string(file_size);
		Ref<Animation> anim;
		anim.instantiate();
		Ref<JSON> json = DataTableManager::get_singleton()->parse_yaml(yaml_anim);
		auto data = json->get_data();
		if(data.get_type() == Variant::DICTIONARY)
		{
			Dictionary dict = data;
			if(dict.has("AnimationClip"))
			{
				Dictionary clip = dict["AnimationClip"];
				on_load_animation.call(clip,mirror,anim);
				anim->optimize();
				String save_path = "res://" + path + "/" + name + ".tres";
				ResourceSaver::save(anim,save_path);
				print_line("UnityLinkServer: save animation node :" + save_path);	
			}
			else
			{
				ERR_FAIL_V_MSG(false,"UnityLinkServer: create animation error " + path);
			}

		}	
		else
		{
			ERR_FAIL_V_MSG(false,"UnityLinkServer: create animation error " + path);
		}
	}
	else if(type == 4){
		// 直接存儲的文件，fbx，png。。。。

		String name;
		if (!read_string(msg_buffer, name))
		{
			ERR_FAIL_V_MSG(false, "UnityLinkServer: create animation node error " + itos(ERR_OUT_OF_MEMORY) + " " + name);

		}
		int file_size = size - msg_buffer.get_position();

		Ref<FileAccess> f = FileAccess::open("res://" + path + "/" + name,FileAccess::WRITE);
		f->store_buffer(msg_buffer.get_u8_ptr(),file_size);	
	}
	return true;
}
static Node* import_fbx(const String &p_path)
{
	
#if TOOLS_ENABLED
	HashMap<StringName, Variant> defaults;
	Dictionary base_subresource_settings;
	
	base_subresource_settings.clear();

	Ref<ConfigFile> config;
	config.instantiate();
	Error err = config->load(p_path + ".import");
	if (err == OK) {
		List<String> keys;
		config->get_section_keys("params", &keys);
		for (const String &E : keys) {
			Variant value = config->get_value("params", E);
			if (E == "_subresources") {
				base_subresource_settings = value;
			} else {
				defaults[E] = value;
			}
		}
	}

	return ResourceImporterScene::get_scene_singleton()->pre_import(p_path, defaults); // Use the scene singleton here because we want to see the full thing.
#endif
	return nullptr;

}
// 處理導入的骨架
static Skeleton3D * process_fbx_skeleton(const String &p_path,Node *p_node)
{
	Node * node = p_node->find_child("Skeleton3D");
	Skeleton3D * skeleton = Object::cast_to<Skeleton3D>(node);
	return skeleton;
}
// 保存模型资源
static void save_fbx_res( const String& group_name,const String& sub_path,const Ref<Resource>& p_resource,String& fbx_path, bool is_resource = true)
{
	String export_root_path = "res://Assets/public";
	if (!DirAccess::dir_exists_absolute(export_root_path))
	{
		print_error("Export Root Path不存在:" + export_root_path);
		return ;

	}
	Ref<DirAccess> dir = DirAccess::create_for_path(export_root_path);
	if (!dir->dir_exists(group_name))
	{
		dir->make_dir(group_name);

	}
	dir->change_dir(group_name);
	if (!dir->dir_exists(sub_path))
	{
			dir->make_dir(sub_path);
	}
	dir->change_dir(sub_path);
	fbx_path = export_root_path.path_join(group_name).path_join(sub_path).path_join(p_resource->get_name() + (is_resource ? "tres" :".tscn"));
	ResourceSaver::save(p_resource,fbx_path);
}
static void get_fbx_meshs(Node *p_node,HashMap<String,MeshInstance3D* > &meshs)
{

	for(int i=0;i<p_node->get_child_count();i++)
	{
		Node * child = p_node->get_child(i);
		if(child->get_class() == "MeshInstance3D")
		{
			MeshInstance3D * mesh = Object::cast_to<MeshInstance3D>(child);
			if(!meshs.has(mesh->get_name())){
				meshs[mesh->get_name()] = mesh;
			}
			else{
				String name = mesh->get_name();
				int index = 1;
				while(meshs.has(name +"_"+ itos(index))){
					name = mesh->get_name().str() + "_" + itos(index);
					index++;
				}
				meshs[name] = mesh;
			}
		}
		get_fbx_meshs(child,meshs);
	}
}
// 處理導入的模型
static void process_fbx_mesh(const String &p_path,const String & p_group, Node *p_node)
{
	String name = p_node->get_name();
	// 存儲模型的預製體文件
	Skeleton3D * skeleton = process_fbx_skeleton(p_path,p_node);
	Dictionary bone_map;
	String ske_save_path,bone_map_save_path;
	if(skeleton != nullptr)
	{
		bone_map = skeleton->get_human_bone_mapping();
		skeleton->set_human_bone_mapping(bone_map);

		// 存儲骨架信息
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		packed_scene->pack(skeleton);
		save_fbx_res("skeleton",p_group,packed_scene,ske_save_path,false);
		Ref<CharacterBoneMap> bone_map_ref;
		bone_map_ref.instantiate();
		bone_map_ref->set_bone_map(bone_map);
		save_fbx_res("skeleton",p_group,bone_map_ref,bone_map_save_path,true);
	}
	HashMap<String,MeshInstance3D* > meshs;
	// 便利存儲模型文件
	get_fbx_meshs(p_node,meshs);

	Ref<CharacterBodyPrefab> prefab;
	prefab->set_name("prefab_" + name);
	prefab.instantiate();
	for(auto it = meshs.begin();it != meshs.end();++it){
		Ref<CharacterBodyPart> part;
		part.instantiate();
		MeshInstance3D* mesh = it->value;
		part->init_form_mesh_instance(mesh,bone_map);
		String save_path = p_path + "_" + it->key + ".tres";
		String temp;
		save_fbx_res("mesh", p_group,part,temp,false);
		prefab->parts.push_back(part);
		print_line("UnityLinkServer: save mesh node :" + save_path);	
	}

	// 保存预制体
	save_fbx_res("prefab",p_group,prefab,bone_map_save_path,true);



}
// 處理導入的骨架
static void process_fbx_animation(const String &p_path,Node *p_node)
{

}
// 創建一個預製體
static void create_fbx_prefab(const String &p_path,const LocalVector<String>& p_paths,const String& p_skeleton_path)
{

}

bool UnityLinkServer::ClientPeer::poll()
{
	
	connection->poll();
	if(connection->get_status() != StreamPeerTCP::STATUS_CONNECTED){
		if(connection->get_status() == StreamPeerTCP::STATUS_ERROR || connection->get_status() == StreamPeerTCP::STATUS_NONE){
			return false;
		}
		return true;
	}
	if(connection->get_available_bytes() == 0){
		return true;
	}
	print_line(String("UnityLinkServer: poll") + String(connection->get_connected_host()) + ":" + itos(connection->get_connected_port()));
	if(buffer_size == 0){
		buffer_size = 102400;
		data = memnew_arr(uint8_t, buffer_size);
	}
	bool msg_finish = false;
	while (true) {
		int read = 0;
		Error err = connection->get_partial_data(&data[curr_read_count], 1, read);
		if (err != OK) {
			return false;
		} else if (read != 1) { // Busy, wait until next poll
			return true;
		}
		++curr_read_count;
		if(!is_msg_statred && curr_read_count == 8)
		{
			int msg_size = *(((int *)data)+1);
			// 小於0 或者大於40兆代表無效數據
			if(msg_size < 0 || msg_size > 40960000)
			{
				return false;
			}
			if(buffer_size < msg_size)
			{
				buffer_size = msg_size;
				uint8_t* old = data;
				data = memnew_arr(uint8_t, buffer_size);
				
				*(((int *)data)) = *(((int *)old));
				*(((int *)data)+1) = msg_size;
				memdelete_arr(old);
			}
			is_msg_statred = true;

		}
		if(curr_read_count >= 8 && curr_read_count == get_msg_size())
		{
			msg_finish = true;
			break;
		}
	}
	if(!msg_finish){
		return true;
	}
	print_line(String("UnityLinkServer: msg_finish") + String(connection->get_connected_host()) + ":" + itos(connection->get_connected_port()) + " msg size:" + itos(get_msg_size()) + " curr_read_count:" + itos(curr_read_count));
	is_msg_statred = false;
	if(!process_msg()){
		curr_read_count = 0;
		return false;
	}
	curr_read_count = 0;
	return true;
}


bool UnityLinkServer::ClientPeer::process_msg()
{
	StreamPeerConstBuffer buffer;
	buffer.set_data_array(data, get_msg_size());
	return poll_client(buffer);
}
void UnityLinkServer::poll() {
	if (!active) {
		return;
	}

	if (!server->is_connection_available()) {
		return;
	}
	// 接收客户端
	while(true)
	{
		Ref<StreamPeerTCP> tcp_peer = server->take_connection();
		if(tcp_peer.is_null()){
			break;
		}
		// Got a connection!
		tcp_peer->set_no_delay(true);
		print_line(String("UnityLinkServer: Got a connection->") + String(tcp_peer->get_connected_host()) + ":" + itos(tcp_peer->get_connected_port()));
		Ref<ClientPeer> peer ;
		peer.instantiate();
		peer->connection = tcp_peer;
		clients.push_back(peer);

	}
	// 遍历客户端
	auto it = clients.begin();
	while(it != clients.end())
	{
		Ref<ClientPeer>& peer = *it;
		if(!peer->poll()){
			print_line(String("UnityLinkServer: disconnect") + String(peer->connection->get_connected_host()) + ":" + itos(peer->connection->get_connected_port()));
			it = clients.erase(it);
			continue;
		}
		++it;
	}

}

void UnityLinkServer::start() {
	if (active) {
		stop();
	}
	port = 9010;
	Error err = server->listen(port);
	ERR_FAIL_COND_MSG(err != OK, String("UnityLinkServer: Unable to listen on port ") + itos(port));
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
	instance = this;
	server.instantiate();

}

UnityLinkServer::~UnityLinkServer() {
	instance = nullptr;
	stop();
}


