#include "foliage_manager.h"

namespace Foliage
{
    void FoliageManager::load(String file_name)
    {
        Ref<Resource> ref = ResourceLoader::load(file_name);
        if(ref.is_valid())
            init(ref);
    }
    void FoliageManager::init(Ref<FoliageMapConfig> map_config)
    {
        FoliageEngine::get_singleton().init(map_config);
    }
    void FoliageManager::update()
    {
        FoliageEngine::get_singleton().update();
    }
    void FoliageManager::set_camera(Camera3D* p_camera)
    {
        FoliageEngine::get_singleton().set_camera(p_camera);
    }
    void FoliageManager::_notification(int p_what) 
    {
        switch (p_what) {
            case NOTIFICATION_READY: {
                //LOG(INFO, "NOTIFICATION_READY");
                //__ready();
                break;
            }

            case NOTIFICATION_PROCESS: {
                update();
                break;
            }

            case NOTIFICATION_PREDELETE: {
                //(INFO, "NOTIFICATION_PREDELETE");
                // _clear();
                break;
            }

            case NOTIFICATION_ENTER_TREE: {
                //LOG(INFO, "NOTIFICATION_ENTER_TREE");
                //_initialize();
                break;
            }

            case NOTIFICATION_EXIT_TREE: {
               // LOG(INFO, "NOTIFICATION_EXIT_TREE");
                //_clear();
                break;
            }

            case NOTIFICATION_ENTER_WORLD: {
             //   LOG(INFO, "NOTIFICATION_ENTER_WORLD");
                //_is_inside_world = true;
                //_update_instances();
                break;
            }

            case NOTIFICATION_TRANSFORM_CHANGED: {
                //LOG(INFO, "NOTIFICATION_TRANSFORM_CHANGED");
                break;
            }

            case NOTIFICATION_EXIT_WORLD: {
               // LOG(INFO, "NOTIFICATION_EXIT_WORLD");
                //_is_inside_world = false;
                break;
            }

            case NOTIFICATION_VISIBILITY_CHANGED: {
                //LOG(INFO, "NOTIFICATION_VISIBILITY_CHANGED");
                //_update_instances();
                break;
            }

            case NOTIFICATION_EDITOR_PRE_SAVE: {
                //LOG(INFO, "NOTIFICATION_EDITOR_PRE_SAVE");
                // if (!_storage.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid storage. Skipping");
                // } else {
                //     _storage->save();
                // }
                // if (!_material.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid material. Skipping");
                // } else {
                //     _material->save();
                // }
                // if (!_texture_list.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid texture list. Skipping");
                // } else {
                //     _texture_list->save();
                // }
                break;
            }

            case NOTIFICATION_EDITOR_POST_SAVE: {
                //LOG(INFO, "NOTIFICATION_EDITOR_POST_SAVE");
                break;
            }
        }

    }
        
    void FoliageManager::clear()
    {
        FoliageEngine::get_singleton().clear();
    }
    FoliageManager::FoliageManager()
    {

    }
    FoliageManager::~FoliageManager()
    {
    }

}