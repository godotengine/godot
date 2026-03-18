/**************************************************************************/
/*  test_painterly_viewport_copy.cpp                                      */
/*  Regression test for stylized viewport copy scaling                    */
/**************************************************************************/

#include "test_macros.h"

#include "../renderer/gaussian_splat_renderer.h"

#include "servers/rendering/rendering_device.h"
#include "servers/rendering_server.h"

#ifdef TESTS_ENABLED

namespace TestGaussianSplatting {

TEST_CASE("[GaussianSplatting] Painterly viewport copy handles downscale") {
    REQUIRE_GPU_DEVICE();

    RenderingServer *rs = RenderingServer::get_singleton();
    CHECK(rs != nullptr);
    if (rs == nullptr) {
        return;
    }

    rd = rs->create_local_rendering_device();
    CHECK(rd != nullptr);
    if (rd == nullptr) {
        return;
    }

    const Size2i viewport_size(1280, 720);
    const Size2i stylized_size(640, 360);

    RD::TextureFormat source_format;
    source_format.width = stylized_size.x;
    source_format.height = stylized_size.y;
    source_format.depth = 1;
    source_format.array_layers = 1;
    source_format.mipmaps = 1;
    source_format.texture_type = RD::TEXTURE_TYPE_2D;
    source_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    source_format.usage_bits = RD::TEXTURE_USAGE_SAMPLING_BIT |
            RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_FROM_BIT;

    RD::TextureFormat destination_format;
    destination_format.width = viewport_size.x;
    destination_format.height = viewport_size.y;
    destination_format.depth = 1;
    destination_format.array_layers = 1;
    destination_format.mipmaps = 1;
    destination_format.texture_type = RD::TEXTURE_TYPE_2D;
    destination_format.format = RD::DATA_FORMAT_R8G8B8A8_UNORM;
    destination_format.usage_bits = RD::TEXTURE_USAGE_STORAGE_BIT |
            RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT |
            RD::TEXTURE_USAGE_CAN_COPY_TO_BIT |
            RD::TEXTURE_USAGE_SAMPLING_BIT;

    RID stylized_texture = rd->texture_create(source_format, RD::TextureView());
    rd->set_resource_name(stylized_texture, "GS_Test_PainterlyViewport_StylizedTexture");
    RID viewport_texture = rd->texture_create(destination_format, RD::TextureView());
    rd->set_resource_name(viewport_texture, "GS_Test_PainterlyViewport_ViewportTexture");

    CHECK(stylized_texture.is_valid());
    CHECK(viewport_texture.is_valid());
    if (!stylized_texture.is_valid() || !viewport_texture.is_valid()) {
        if (stylized_texture.is_valid()) {
            rd->free(stylized_texture);
        }
        if (viewport_texture.is_valid()) {
            rd->free(viewport_texture);
        }
        memdelete(rd);
        return;
    }

    Ref<GaussianSplatRenderer> renderer;
    renderer.instantiate(rd);
    renderer->set_painterly_internal_scale(0.5f);
    renderer->test_override_rendering_device(rd);

    bool copy_result = renderer->test_copy_final_output(stylized_texture, viewport_texture, viewport_size);
    CHECK(copy_result);
    CHECK(renderer->was_last_viewport_copy_successful());
    CHECK(renderer->get_last_viewport_copy_source_size() == stylized_size);
    CHECK(renderer->get_last_viewport_copy_dest_size() == stylized_size);

    rd->free(stylized_texture);
    rd->free(viewport_texture);
    memdelete(rd);
}

} // namespace TestGaussianSplatting

#endif // TESTS_ENABLED
