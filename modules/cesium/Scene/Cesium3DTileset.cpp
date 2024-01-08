//
// Created by Harris.Lu on 2024/1/8.
//

#include "Cesium3DTileset.h"

#include "core/error/error_macros.h"

namespace Cesium {
    Cesium3DTileset::Cesium3DTileset() {
        request = memnew(HTTPRequest);
        if(!request->is_connected("request_completed", callable_mp(this, &Cesium3DTileset::_on_request_completed))) {
            request->connect("request_completed", callable_mp(this, &Cesium3DTileset::_on_request_completed));
        }
    }

    Cesium3DTileset::~Cesium3DTileset() {
    }

    //http://data.mars3d.cn/3dtiles/qx-dyt/tileset.json
    void Cesium3DTileset::set_url(String p_url) {
        if(p_url.is_empty()) {
            return;
        }
        // fromUrl
        _options.url = p_url;

        auto error = request->request(p_url);
        if (error != OK) {
            WARN_PRINT("request error: " + String::num(error));
            return;
        }
    }

    /**
     * 解析tileset.json
     * @param p_status
     * @param p_code
     * @param p_headers
     * @param p_data
     */
    void Cesium3DTileset::_on_request_completed(int p_status, int p_code, const PackedStringArray &p_headers, const PackedByteArray &p_data) {
        if(p_status != HTTPRequest::RESULT_SUCCESS || p_code != 200) {
            WARN_PRINT("request error: " + String::num(p_status));
            return;
        }

        String response_json;
        {
            const uint8_t *r = p_data.ptr();
            response_json.parse_utf8((const char *)r, p_data.size());
        }

        WARN_PRINT(response_json);
        JSON json;
        Error err = json.parse(response_json);
        if (err != OK) {
            WARN_PRINT("parse error: " + String::num(err));
            return;
        }


    }

    void Cesium3DTileset::_bind_methods() {
        ClassDB::bind_method(D_METHOD("set_url", "url"), &Cesium3DTileset::set_url);
        ClassDB::bind_method(D_METHOD("get_url"), &Cesium3DTileset::get_url);

        ADD_GROUP("Cesium", "cesium_");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "url", PROPERTY_HINT_NONE), "set_url", "get_url");
    }

    void Cesium3DTileset::_notification(int p_what) {
        switch (p_what) {
            case NOTIFICATION_ENTER_TREE: {
                this->add_child(request);
                break;
            }
        }

    }
} // Cesium