#ifndef MNAVIGATIONDATA
#define MNAVIGATIONDATA

#include "core/io/resource.h"
#include "core/variant/variant.h"



class MNavigationMeshData : public Resource {
    GDCLASS(MNavigationMeshData, Resource);

    protected:
    static void _bind_methods();

    public:
    PackedByteArray data;
    bool on_all_at_creation = true;

    void set_data(const PackedByteArray& d);
    PackedByteArray get_data();

    void set_on_all_at_creation(bool input);
    bool get_on_all_at_creation();
};


#endif