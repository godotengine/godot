#include "scene_distribution_interface.h"

#include "scene_multiplayer.h"
#include "core/io/marshalls.h"
#include "scene/main/multiplayer_api.h"

#include <stdio.h>
#include <fstream>

#include "modules/gltf/gltf_document.h"

void SceneDistributionInterface::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("request_glb", "glb_name"), &SceneDistributionInterface::request_glb);
	ClassDB::bind_method(D_METHOD("set_own_peer_as_glb_creator"), &SceneDistributionInterface::set_own_peer_as_glb_creator);

	ADD_SIGNAL(MethodInfo("_check_glb_existence", PropertyInfo(Variant::STRING, "p_path"), PropertyInfo(Variant::INT, "id")));
	ADD_SIGNAL(MethodInfo("_set_glb_creator_peer", PropertyInfo(Variant::INT, "peer")));
}



void SceneDistributionInterface::request_glb(const String& glb_name)
{
	printf("SceneDistributionInterface::request_glb ->%s\n", glb_name.ascii().get_data());

	//check if we got a glb_creator_peer
	if (multiplayer->get_distributor()->glb_creator_peer > 0) {
		printf("we got a glb_creator_peer, it is %d\n", glb_creator_peer);

		int packet_len = SceneMultiplayer::SYS_CMD_SIZE + (glb_name.size() * 4);
		printf("packet_len:%d\n", packet_len);

		std::vector<uint8_t> buf(packet_len, 0);
		buf[0] = SceneMultiplayer::NETWORK_COMMAND_SYS;
		buf[1] = SceneMultiplayer::SYS_COMMAND_REQUEST_GLB;
		multiplayer->get_multiplayer_peer()->set_transfer_channel(0);
		multiplayer->get_multiplayer_peer()->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
		encode_uint32(multiplayer->get_unique_id(), &buf[2]);

		encode_cstring(glb_name.utf8().get_data(), &buf[6]);

		//send our request to glb_creator_peer
		multiplayer->get_multiplayer_peer()->set_target_peer(multiplayer->get_distributor()->glb_creator_peer);
		multiplayer->get_multiplayer_peer()->put_packet(buf.data(), packet_len);
	}
	else
		printf("we got NO glb_creator_peer\n");

}

void SceneDistributionInterface::distribute_glb(const String& p_path, int id)
{
	printf("SceneDistributionInterface::distribute_glb\n");
	CharString a = p_path.ascii();
	printf(a.get_data());

	//load glb file into PackedByteArray
	Ref<GLTFDocument> gltf;
	gltf.instantiate();
	Ref<GLTFState> gltf_state;
	gltf_state.instantiate();

	String externally_created_glb_storage_path_file = externally_created_glb_storage_path + p_path;
	Error err;
	PackedByteArray glb_file_PBA;
	Ref<FileAccess> f = FileAccess::open(externally_created_glb_storage_path_file, FileAccess::READ, &err);

	Vector<uint8_t> data;

	if (err != OK) {
		printf("open external glb error\n");
	}
	else if (f.is_valid()) {
		
		data.resize(f->get_length());
		printf("glb file length:%lld\n", f->get_length());
		f->get_buffer(data.ptrw(), f->get_length());

		glb_file_PBA.resize(f->get_length());
		//f->get_buffer(glb_file_PBA.ptrw(), f->get_length());  //does not work
		
		for (int i = 0; i < f->get_length(); i++) {
			glb_file_PBA.set(i,data[i]);
		}
	}

	//write it to own user://
	String save_path = "user://" + p_path.replace(".glb", ".gltf");

	
	gltf->append_from_buffer(glb_file_PBA, "base_path?", gltf_state);
	gltf->write_to_filesystem(gltf_state, save_path);

	//send it to the clients and MultiplayerSpawner.add_spawnable_scene
	int packet_len = SceneMultiplayer::SYS_CMD_SIZE + 4 + 4 + p_path.size() + glb_file_PBA.size() ;
	printf("packet_len:%d\n", packet_len);

	std::vector<uint8_t> buf(packet_len, 0);
	buf[0] = SceneMultiplayer::NETWORK_COMMAND_SYS;
	buf[1] = SceneMultiplayer::SYS_COMMAND_DISTRIBUTE_GLB;
	multiplayer->get_multiplayer_peer()->set_transfer_channel(0);
	multiplayer->get_multiplayer_peer()->set_transfer_mode(MultiplayerPeer::TRANSFER_MODE_RELIABLE);
	encode_uint32(multiplayer->get_unique_id(), &buf[2]);

	//encode length of glb file name
	encode_uint32(p_path.length(), &buf[6]);
	//encode size of glb_file_PBA
	encode_uint32(glb_file_PBA.size(), &buf[10]);
	//encode glb file name
	encode_cstring(p_path.ascii().get_data(), &buf[14]);

	for (int i = 0; i < glb_file_PBA.size(); i++) {
		buf[p_path.length() + SceneMultiplayer::SYS_CMD_SIZE + 4 + 4 + i] = glb_file_PBA[i];
	}


	printf("buf-out: %x", decode_uint32( &buf[p_path.length() + SceneMultiplayer::SYS_CMD_SIZE + 4 + 4]) );

	//err = encode_variant(p_custom_features, &buf[0], glb_file_PBA.size(), false);
	//ERR_FAIL_COND_V(err != OK, err);
	
	

	//send
	int own_peer = multiplayer->get_unique_id();
	for (const int& P : multiplayer->get_connected_peers()) {
		if (P != own_peer) {
			multiplayer->get_multiplayer_peer()->set_target_peer(P);
			multiplayer->get_multiplayer_peer()->put_packet(buf.data(), packet_len);
		}
	}
}

void SceneDistributionInterface::set_distribute_glb_result(int peer, int result, String file_name)
{
	printf("SceneDistributionInterface::set_distribute_glb_result\n");
	printf("id:%d  result:%d file_name:%s", peer, result, file_name.ascii().get_data());
}

void SceneDistributionInterface::set_glb_existence_info(int peer, int result, String file_name)
{
	printf("SceneDistributionInterface::set_glb_existence_info\n");
	printf("peer:%d file_name:%s\n", peer, file_name.ascii().get_data());

	//set peer as id, and file_name as payload ??
	if (!collect_distribute_glb_result_peers.has(file_name.ascii().get_data())) {
		collect_distribute_glb_result_struct c = {peer, result};
		collect_distribute_glb_result_peers.insert(file_name.ascii().get_data(), c);
	}
}

void SceneDistributionInterface::set_own_peer_as_glb_creator()
{
	printf("SceneDistributionInterface::set_own_peer_as_glb_creator\n");

	//set own peer as glb creator
	glb_creator_peer = multiplayer->get_unique_id();

	//tell all other that we are a glb creator
	emit_signal(SNAME("_set_glb_creator_peer"), glb_creator_peer);

}

HashSet<String> SceneDistributionInterface::get_requested_glb_files()
{
	return requested_glb_files;
}

void SceneDistributionInterface::request_to_externally_create_glb(const String& glb_name)
{
	printf("SceneDistributionInterface::request_to_externally_create_glb %s\n", glb_name.ascii().get_data());

	FILE* fp;
	int status;
	char path[200];


	fp = _popen(externally_create_glb_script.ascii().get_data(), "r");
	if (fp == NULL)
		printf("_popen-open error\n");


	while (fgets(path, 200, fp) != NULL)
		printf("%s", path);


	status = _pclose(fp);
	if (status == -1) {
		/* Error reported by pclose() */
		printf("_pclose error\n");
	}
	else {
		/* Use macros described under wait() to inspect `status' in order
		   to determine success/failure of command executed by popen() */
		printf("_pclose no error\n");
	}
}

void SceneDistributionInterface::check_if_externally_created_glb_was_created()
{
	//printf("SceneDistributionInterface::check_if_externally_created_glb_was_created");

	HashSet<String>::Iterator it;
	for (it = requested_glb_files.begin(); it; ++it) {
		//printf("it is:%s\n", it->ascii().get_data() );
	
		//check in a pre-specified directory, if there is a file with a name, we got in
		//our requested_glb_files hashset
		String s = externally_created_glb_storage_path + *it;
		//printf("s is:%s\n", s.ascii().get_data() );
		std::ifstream f(s.ascii().get_data());
		if (f.good()) {
			printf("Found requested glb\n");

			//remove glb-name from requested_glb_files
			requested_glb_files.remove(it);

			//distribute it now
			distribute_glb(it->ascii().get_data(), 1);
		}
	}
}

void SceneDistributionInterface::set_glb_as_requested(const String& glb_name)
{
	//check if someone already requested this glb_name file in the past
	if (requested_glb_files.has(glb_name.ascii().get_data())) {
		printf("glb file %s is already requested\n", glb_name.ascii().get_data());
	}
	else {
		printf("glb file %s inserted to requested HashSet\n", glb_name.ascii().get_data());
		requested_glb_files.insert(glb_name.ascii().get_data());
	}
}
