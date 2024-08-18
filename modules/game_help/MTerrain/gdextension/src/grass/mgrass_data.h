#ifndef MGRASS_DATA
#define MGRASS_DATA

#include "core/io/resource.h"
#include "core/variant/variant.h"
#include "core/templates/hash_map.h"

#include "../mconfig.h"

using namespace godot;

struct MGrassUndoData {
    uint8_t* data=nullptr;
    uint8_t* backup_data=nullptr;
    void free(){
        memdelete_arr(data);
        if(backup_data!=nullptr){
            memdelete_arr(backup_data);
        }
    }
};

class MGrassData : public Resource {
    GDCLASS(MGrassData,Resource);

    protected:
    static void _bind_methods();

    public:
    MGrassData();
    ~MGrassData();
    PackedByteArray data;
    PackedByteArray backup_data;
    int current_undo_id=0;
    int lowest_undo_id=0;
    HashMap<int,MGrassUndoData> undo_data;
    int density_index=2;
    float density=1;

    void set_data(const PackedByteArray& d);
    const PackedByteArray& get_data();
    void set_backup_data(const PackedByteArray& d);
    const PackedByteArray& get_backup_data();
    void set_density(int input);
    int get_density();

    bool backup_exist();
    void backup_create();
    void backup_merge();
    void backup_restore();

    void check_undo(); // register a stage for undo
    void clear_undo();
    void undo();

};
#endif