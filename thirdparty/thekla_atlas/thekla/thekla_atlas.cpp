
#include "thekla_atlas.h"

#include <cfloat>
// -- GODOT start --
#include <stdio.h>
// -- GODOT end --

#include "nvmesh/halfedge/Edge.h"
#include "nvmesh/halfedge/Mesh.h"
#include "nvmesh/halfedge/Face.h"
#include "nvmesh/halfedge/Vertex.h"
#include "nvmesh/param/Atlas.h"

#include "nvmath/Vector.inl"
#include "nvmath/ftoi.h"

#include "nvcore/Array.inl"


using namespace Thekla;
using namespace nv;


inline Atlas_Output_Mesh * set_error(Atlas_Error * error, Atlas_Error code) {
    if (error) *error = code;
    return NULL;
}



static void input_to_mesh(const Atlas_Input_Mesh * input, HalfEdge::Mesh * mesh, Atlas_Error * error) {

    Array<uint> canonicalMap;
    canonicalMap.reserve(input->vertex_count);

    for (int i = 0; i < input->vertex_count; i++) {
        const Atlas_Input_Vertex & input_vertex = input->vertex_array[i];
        const float * pos = input_vertex.position;
        const float * nor = input_vertex.normal;
        const float * tex = input_vertex.uv;

        HalfEdge::Vertex * vertex = mesh->addVertex(Vector3(pos[0], pos[1], pos[2]));
        vertex->nor.set(nor[0], nor[1], nor[2]);
        vertex->tex.set(tex[0], tex[1]);

        canonicalMap.append(input_vertex.first_colocal);
    }

    mesh->linkColocalsWithCanonicalMap(canonicalMap);


    const int face_count = input->face_count;

    int non_manifold_faces = 0;
    for (int i = 0; i < face_count; i++) {
        const Atlas_Input_Face & input_face = input->face_array[i];

        int v0 = input_face.vertex_index[0];
        int v1 = input_face.vertex_index[1];
        int v2 = input_face.vertex_index[2];

        HalfEdge::Face * face = mesh->addFace(v0, v1, v2);
        if (face != NULL) {
            face->material = input_face.material_index;
        }
        else {
            non_manifold_faces++;
        }
    }

    mesh->linkBoundary();

    if (non_manifold_faces != 0 && error != NULL) {
        *error = Atlas_Error_Invalid_Mesh_Non_Manifold;
    }
}

static Atlas_Output_Mesh * mesh_atlas_to_output(const HalfEdge::Mesh * mesh, const Atlas & atlas, Atlas_Error * error) {

    Atlas_Output_Mesh * output = new Atlas_Output_Mesh;

    const MeshCharts * charts = atlas.meshAt(0);

    // Allocate vertices.
    const int vertex_count = charts->vertexCount();
    output->vertex_count = vertex_count;
    output->vertex_array = new Atlas_Output_Vertex[vertex_count];

    int w = 0;
    int h = 0;

    // Output vertices.
    const int chart_count = charts->chartCount();
    for (int i = 0; i < chart_count; i++) {
        const Chart * chart = charts->chartAt(i);
        uint vertexOffset = charts->vertexCountBeforeChartAt(i);

        const uint chart_vertex_count = chart->vertexCount();
        for (uint v = 0; v < chart_vertex_count; v++) {
            Atlas_Output_Vertex & output_vertex = output->vertex_array[vertexOffset + v]; 

            uint original_vertex = chart->mapChartVertexToOriginalVertex(v);
            output_vertex.xref = original_vertex;

            Vector2 uv = chart->chartMesh()->vertexAt(v)->tex;
            output_vertex.uv[0] = uv.x;
            output_vertex.uv[1] = uv.y;
            w = max(w, ftoi_ceil(uv.x));
            h = max(h, ftoi_ceil(uv.y));
        }
    }

    const int face_count = mesh->faceCount();
    output->index_count = face_count * 3;
    output->index_array = new int[face_count * 3];

    // -- GODOT start --
    int face_ofs = 0;
    // Set face indices.
    for (int f = 0; f < face_count; f++) {
        uint c = charts->faceChartAt(f);
        uint i = charts->faceIndexWithinChartAt(f);
        uint vertexOffset = charts->vertexCountBeforeChartAt(c);

        const Chart * chart = charts->chartAt(c);
        nvDebugCheck(chart->faceAt(i) == f);

        if (i >= chart->chartMesh()->faceCount()) {
            printf("WARNING: Faces may be missing in the final vertex, which could not be packed\n");
            continue;
        }

        const HalfEdge::Face * face = chart->chartMesh()->faceAt(i);
        const HalfEdge::Edge * edge = face->edge;

        //output->index_array[3*f+0] = vertexOffset + edge->vertex->id;
        //output->index_array[3*f+1] = vertexOffset + edge->next->vertex->id;
        //output->index_array[3*f+2] = vertexOffset + edge->next->next->vertex->id;
        output->index_array[3 * face_ofs + 0] = vertexOffset + edge->vertex->id;
        output->index_array[3 * face_ofs + 1] = vertexOffset + edge->next->vertex->id;
        output->index_array[3 * face_ofs + 2] = vertexOffset + edge->next->next->vertex->id;
        face_ofs++;
    }

    output->index_count = face_ofs * 3;
    // -- GODOT end --

    *error = Atlas_Error_Success;
    output->atlas_width = w;
    output->atlas_height = h;

    return output;
}


void Thekla::atlas_set_default_options(Atlas_Options * options) {
    if (options != NULL) {
        // These are the default values we use on The Witness.

        options->charter = Atlas_Charter_Default;
        options->charter_options.witness.proxy_fit_metric_weight = 2.0f;
        options->charter_options.witness.roundness_metric_weight = 0.01f;
        options->charter_options.witness.straightness_metric_weight = 6.0f;
        options->charter_options.witness.normal_seam_metric_weight = 4.0f;
        options->charter_options.witness.texture_seam_metric_weight = 0.5f;
        options->charter_options.witness.max_chart_area = FLT_MAX;
        options->charter_options.witness.max_boundary_length = FLT_MAX;

        options->mapper = Atlas_Mapper_Default;

        options->packer = Atlas_Packer_Default;
        options->packer_options.witness.packing_quality = 0;
        options->packer_options.witness.texel_area = 8;
        options->packer_options.witness.block_align = true;
        options->packer_options.witness.conservative = false;
    }
}


Atlas_Output_Mesh * Thekla::atlas_generate(const Atlas_Input_Mesh * input, const Atlas_Options * options, Atlas_Error * error) {
    // Validate args.
    if (input == NULL || options == NULL || error == NULL) return set_error(error, Atlas_Error_Invalid_Args);

    // Validate options.
    if (options->charter != Atlas_Charter_Witness) {
        return set_error(error, Atlas_Error_Invalid_Options);
    }
    if (options->charter == Atlas_Charter_Witness) {
        // @@ Validate input options!
    }

    if (options->mapper != Atlas_Mapper_LSCM) {
        return set_error(error, Atlas_Error_Invalid_Options);
    }
    if (options->mapper == Atlas_Mapper_LSCM) {
        // No options.
    }

    if (options->packer != Atlas_Packer_Witness) {
        return set_error(error, Atlas_Error_Invalid_Options);
    }
    if (options->packer == Atlas_Packer_Witness) {
        // @@ Validate input options!
    }

    // Validate input mesh.
    for (int i = 0; i < input->face_count; i++) {
        int v0 = input->face_array[i].vertex_index[0];
        int v1 = input->face_array[i].vertex_index[1];
        int v2 = input->face_array[i].vertex_index[2];

        if (v0 < 0 || v0 >= input->vertex_count || 
            v1 < 0 || v1 >= input->vertex_count || 
            v2 < 0 || v2 >= input->vertex_count)
        {
            return set_error(error, Atlas_Error_Invalid_Mesh);
        }
    }


    // Build half edge mesh.
    AutoPtr<HalfEdge::Mesh> mesh(new HalfEdge::Mesh);

    input_to_mesh(input, mesh.ptr(), error);

    if (*error == Atlas_Error_Invalid_Mesh) {
        return NULL;
    }

    Atlas atlas;

    // Charter.
    if (options->charter == Atlas_Charter_Extract) {
        return set_error(error, Atlas_Error_Not_Implemented);
    }
    else if (options->charter == Atlas_Charter_Witness) {
        SegmentationSettings segmentation_settings;
        segmentation_settings.proxyFitMetricWeight = options->charter_options.witness.proxy_fit_metric_weight;
        segmentation_settings.roundnessMetricWeight = options->charter_options.witness.roundness_metric_weight;
        segmentation_settings.straightnessMetricWeight = options->charter_options.witness.straightness_metric_weight;
        segmentation_settings.normalSeamMetricWeight = options->charter_options.witness.normal_seam_metric_weight;
        segmentation_settings.textureSeamMetricWeight = options->charter_options.witness.texture_seam_metric_weight;
        segmentation_settings.maxChartArea = options->charter_options.witness.max_chart_area;
        segmentation_settings.maxBoundaryLength = options->charter_options.witness.max_boundary_length;

        Array<uint> uncharted_materials;
        atlas.computeCharts(mesh.ptr(), segmentation_settings, uncharted_materials);
    }
    
    if (atlas.hasFailed())
        return NULL;

    // Mapper.
    if (options->mapper == Atlas_Mapper_LSCM) {
        atlas.parameterizeCharts();
    }

    if (atlas.hasFailed())
        return NULL;

    // Packer.
    if (options->packer == Atlas_Packer_Witness) {
        int packing_quality = options->packer_options.witness.packing_quality;
        float texel_area = options->packer_options.witness.texel_area;
        int block_align = options->packer_options.witness.block_align;
        int conservative = options->packer_options.witness.conservative;

        /*float utilization =*/ atlas.packCharts(packing_quality, texel_area, block_align, conservative);
    }
    
    if (atlas.hasFailed())
        return NULL;


    // Build output mesh.
    return mesh_atlas_to_output(mesh.ptr(), atlas, error);
}


void Thekla::atlas_free(Atlas_Output_Mesh * output) {
    if (output != NULL) {
        delete [] output->vertex_array;
        delete [] output->index_array;
        delete output;
    }
}

