#include "path_manager.h"
static PathManager* singleton = nullptr;

PathManager::PathManager()
{
    singleton = this;
}
PathManager::~PathManager()
{
    if(singleton == this)
    {
        singleton = nullptr;
    }
}
PathManager *PathManager::get_singleton()
{
    static PathManager singleton;
    return &singleton;
}