#ifndef SCENE_DISTRIBUTION_INTERFACE_H
#define SCENE_DISTRIBUTION_INTERFACE_H

#include "core/object/ref_counted.h"
#include <vector>


class SceneMultiplayer;

class SceneDistributionInterface : public RefCounted {
	GDCLASS(SceneDistributionInterface, RefCounted);

private:
	SceneMultiplayer* multiplayer = nullptr;

	static void _bind_methods();
	//The directory where external programs save the created glb file
	String externally_created_glb_storage_path = "C:/Users/inflo/Documents/godot-workspace/godot-data/requested_glb/";
	//The script to call, to create a glb file
	String externally_create_glb_script = "C:/Users/inflo/Documents/godot-workspace/godot-data/scripts/create_glb.bat";

	//here save requested glb files, if they are created and distributed, they are removed. Pending requests
	HashSet<String> requested_glb_files;

	struct collect_distribute_glb_result_struct {
		int peer;
		int result = -1;
	};
	HashMap<String, collect_distribute_glb_result_struct> collect_distribute_glb_result_peers;

public:
	//the peer that is able to create glb files with external tools
	int glb_creator_peer = -1;

	void request_glb(const String& glb_name);
	void set_glb_as_requested(const String& glb_name);

	void distribute_glb(const String& p_path, int id);
	void set_distribute_glb_result(int peer, int result, String file_name);
	void set_glb_existence_info(int peer, int result, String file_name);

	void set_own_peer_as_glb_creator();
	HashSet<String> get_requested_glb_files();
	void request_to_externally_create_glb(const String& glb_name);
	void check_if_externally_created_glb_was_created();

	//will be called in multiplayer constructor
	SceneDistributionInterface(SceneMultiplayer* p_multiplayer) {
		multiplayer = p_multiplayer;
	}
};

#endif // SCENE_DISTRIBUTION_INTERFACE_H
