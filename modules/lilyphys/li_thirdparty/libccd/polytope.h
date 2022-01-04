/***
 * libccd
 * ---------------------------------
 * Copyright (c)2010 Daniel Fiser <danfis@danfis.cz>
 *
 *
 *  This file is part of libccd.
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see accompanying file BDS-LICENSE for details or see
 *  <http://www.opensource.org/licenses/bsd-license.php>.
 *
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#ifndef __CCD_POLYTOPE_H__
#define __CCD_POLYTOPE_H__

#include <stdlib.h>
#include <stdio.h>
#include "support.h"
#include "list.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define CCD_PT_VERTEX 1
#define CCD_PT_EDGE   2
#define CCD_PT_FACE   3


#define __CCD_PT_EL \
    int type;           /*! type of element */ \
    ccd_real_t dist;        /*! distance from origin */ \
    ccd_vec3_t witness; /*! witness point of projection of origin */ \
    ccd_list_t list;    /*! list of elements of same type */

/**
 * General polytope element.
 * Could be vertex, edge or triangle.
 */
struct _ccd_pt_el_t {
    __CCD_PT_EL
};
typedef struct _ccd_pt_el_t ccd_pt_el_t;

struct _ccd_pt_edge_t;
struct _ccd_pt_face_t;

/**
 * Polytope's vertex.
 */
struct _ccd_pt_vertex_t {
    __CCD_PT_EL

    int id;
    ccd_support_t v;
    ccd_list_t edges; //!< List of edges
};
typedef struct _ccd_pt_vertex_t ccd_pt_vertex_t;

/**
 * Polytope's edge.
 */
struct _ccd_pt_edge_t {
    __CCD_PT_EL

    ccd_pt_vertex_t *vertex[2]; //!< Reference to vertices
    struct _ccd_pt_face_t *faces[2]; //!< Reference to faces

    ccd_list_t vertex_list[2]; //!< List items in vertices' lists
};
typedef struct _ccd_pt_edge_t ccd_pt_edge_t;

/**
 * Polytope's triangle faces.
 */
struct _ccd_pt_face_t {
    __CCD_PT_EL

    ccd_pt_edge_t *edge[3]; //!< Reference to surrounding edges
};
typedef struct _ccd_pt_face_t ccd_pt_face_t;


/**
 * Struct containing polytope.
 */
struct _ccd_pt_t {
    ccd_list_t vertices; //!< List of vertices
    ccd_list_t edges; //!< List of edges
    ccd_list_t faces; //!< List of faces

    ccd_pt_el_t *nearest;
    ccd_real_t nearest_dist;
    int nearest_type;
};
typedef struct _ccd_pt_t ccd_pt_t;


CCD_EXPORT void ccdPtInit(ccd_pt_t *pt);
CCD_EXPORT void ccdPtDestroy(ccd_pt_t *pt);

/**
 * Returns vertices surrounding given triangle face.
 */
_ccd_inline void ccdPtFaceVec3(const ccd_pt_face_t *face,
                               ccd_vec3_t **a,
                               ccd_vec3_t **b,
                               ccd_vec3_t **c);
_ccd_inline void ccdPtFaceVertices(const ccd_pt_face_t *face,
                                   ccd_pt_vertex_t **a,
                                   ccd_pt_vertex_t **b,
                                   ccd_pt_vertex_t **c);
_ccd_inline void ccdPtFaceEdges(const ccd_pt_face_t *f,
                                ccd_pt_edge_t **a,
                                ccd_pt_edge_t **b,
                                ccd_pt_edge_t **c);

_ccd_inline void ccdPtEdgeVec3(const ccd_pt_edge_t *e,
                               ccd_vec3_t **a,
                               ccd_vec3_t **b);
_ccd_inline void ccdPtEdgeVertices(const ccd_pt_edge_t *e,
                                   ccd_pt_vertex_t **a,
                                   ccd_pt_vertex_t **b);
_ccd_inline void ccdPtEdgeFaces(const ccd_pt_edge_t *e,
                                ccd_pt_face_t **f1,
                                ccd_pt_face_t **f2);


/**
 * Adds vertex to polytope and returns pointer to newly created vertex.
 */
CCD_EXPORT ccd_pt_vertex_t *ccdPtAddVertex(ccd_pt_t *pt, const ccd_support_t *v);
_ccd_inline ccd_pt_vertex_t *ccdPtAddVertexCoords(ccd_pt_t *pt,
                                                  ccd_real_t x, ccd_real_t y, ccd_real_t z);

/**
 * Adds edge to polytope.
 */
CCD_EXPORT ccd_pt_edge_t *ccdPtAddEdge(ccd_pt_t *pt, ccd_pt_vertex_t *v1,
                                          ccd_pt_vertex_t *v2);

/**
 * Adds face to polytope.
 */
CCD_EXPORT ccd_pt_face_t *ccdPtAddFace(ccd_pt_t *pt, ccd_pt_edge_t *e1,
                                          ccd_pt_edge_t *e2,
                                          ccd_pt_edge_t *e3);

/**
 * Deletes vertex from polytope.
 * Returns 0 on success, -1 otherwise.
 */
_ccd_inline int ccdPtDelVertex(ccd_pt_t *pt, ccd_pt_vertex_t *);
_ccd_inline int ccdPtDelEdge(ccd_pt_t *pt, ccd_pt_edge_t *);
_ccd_inline int ccdPtDelFace(ccd_pt_t *pt, ccd_pt_face_t *);


/**
 * Recompute distances from origin for all elements in pt.
 */
CCD_EXPORT void ccdPtRecomputeDistances(ccd_pt_t *pt);

/**
 * Returns nearest element to origin.
 */
CCD_EXPORT ccd_pt_el_t *ccdPtNearest(ccd_pt_t *pt);


CCD_EXPORT void ccdPtDumpSVT(ccd_pt_t *pt, const char *fn);
CCD_EXPORT void ccdPtDumpSVT2(ccd_pt_t *pt, FILE *);


/**** INLINES ****/
_ccd_inline ccd_pt_vertex_t *ccdPtAddVertexCoords(ccd_pt_t *pt,
                                                  ccd_real_t x, ccd_real_t y, ccd_real_t z)
{
    ccd_support_t s;
    ccdVec3Set(&s.v, x, y, z);
    return ccdPtAddVertex(pt, &s);
}

_ccd_inline int ccdPtDelVertex(ccd_pt_t *pt, ccd_pt_vertex_t *v)
{
    // test if any edge is connected to this vertex
    if (!ccdListEmpty(&v->edges))
        return -1;

    // delete vertex from main list
    ccdListDel(&v->list);

    if ((void *)pt->nearest == (void *)v){
        pt->nearest = NULL;
    }

    free(v);
    return 0;
}

_ccd_inline int ccdPtDelEdge(ccd_pt_t *pt, ccd_pt_edge_t *e)
{
    // text if any face is connected to this edge (faces[] is always
    // aligned to lower indices)
    if (e->faces[0] != NULL)
        return -1;

    // disconnect edge from lists of edges in vertex struct
    ccdListDel(&e->vertex_list[0]);
    ccdListDel(&e->vertex_list[1]);

    // disconnect edge from main list
    ccdListDel(&e->list);

    if ((void *)pt->nearest == (void *)e){
        pt->nearest = NULL;
    }

    free(e);
    return 0;
}

_ccd_inline int ccdPtDelFace(ccd_pt_t *pt, ccd_pt_face_t *f)
{
    ccd_pt_edge_t *e;
    size_t i;

    // remove face from edges' recerence lists
    for (i = 0; i < 3; i++){
        e = f->edge[i];
        if (e->faces[0] == f){
            e->faces[0] = e->faces[1];
        }
        e->faces[1] = NULL;
    }

    // remove face from list of all faces
    ccdListDel(&f->list);

    if ((void *)pt->nearest == (void *)f){
        pt->nearest = NULL;
    }

    free(f);
    return 0;
}

_ccd_inline void ccdPtFaceVec3(const ccd_pt_face_t *face,
                               ccd_vec3_t **a,
                               ccd_vec3_t **b,
                               ccd_vec3_t **c)
{
    *a = &face->edge[0]->vertex[0]->v.v;
    *b = &face->edge[0]->vertex[1]->v.v;

    if (face->edge[1]->vertex[0] != face->edge[0]->vertex[0]
            && face->edge[1]->vertex[0] != face->edge[0]->vertex[1]){
        *c = &face->edge[1]->vertex[0]->v.v;
    }else{
        *c = &face->edge[1]->vertex[1]->v.v;
    }
}

_ccd_inline void ccdPtFaceVertices(const ccd_pt_face_t *face,
                                   ccd_pt_vertex_t **a,
                                   ccd_pt_vertex_t **b,
                                   ccd_pt_vertex_t **c)
{
    *a = face->edge[0]->vertex[0];
    *b = face->edge[0]->vertex[1];

    if (face->edge[1]->vertex[0] != face->edge[0]->vertex[0]
            && face->edge[1]->vertex[0] != face->edge[0]->vertex[1]){
        *c = face->edge[1]->vertex[0];
    }else{
        *c = face->edge[1]->vertex[1];
    }
}

_ccd_inline void ccdPtFaceEdges(const ccd_pt_face_t *f,
                                ccd_pt_edge_t **a,
                                ccd_pt_edge_t **b,
                                ccd_pt_edge_t **c)
{
    *a = f->edge[0];
    *b = f->edge[1];
    *c = f->edge[2];
}

_ccd_inline void ccdPtEdgeVec3(const ccd_pt_edge_t *e,
                               ccd_vec3_t **a,
                               ccd_vec3_t **b)
{
    *a = &e->vertex[0]->v.v;
    *b = &e->vertex[1]->v.v;
}

_ccd_inline void ccdPtEdgeVertices(const ccd_pt_edge_t *e,
                                   ccd_pt_vertex_t **a,
                                   ccd_pt_vertex_t **b)
{
    *a = e->vertex[0];
    *b = e->vertex[1];
}

_ccd_inline void ccdPtEdgeFaces(const ccd_pt_edge_t *e,
                                ccd_pt_face_t **f1,
                                ccd_pt_face_t **f2)
{
    *f1 = e->faces[0];
    *f2 = e->faces[1];
}


#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __CCD_POLYTOPE_H__ */
