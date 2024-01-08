//
// Created by Harris.Lu on 2024/1/8.
//

#ifndef GODOT2024_CESIUM3DTILESET_H
#define GODOT2024_CESIUM3DTILESET_H

#include "scene/3d/node_3d.h"
#include "core/variant/variant.h"
#include "scene/main/http_request.h"

namespace Cesium {

    class Cesium3DTileset : public Node3D {
        GDCLASS(Cesium3DTileset, Node3D);

        HTTPRequest* request = nullptr;

    public:
        struct TilesetOptions {
            String url;
        };

    private:
        TilesetOptions _options;

        void _on_request_completed(int p_status, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_data);

    public:
        Cesium3DTileset();
        ~Cesium3DTileset();

        void set_url(String p_url);
        String get_url() const { return _options.url; }

    protected:
        void _notification(int p_what);
        static void _bind_methods();

    };

} // Cesium

#endif //GODOT2024_CESIUM3DTILESET_H
