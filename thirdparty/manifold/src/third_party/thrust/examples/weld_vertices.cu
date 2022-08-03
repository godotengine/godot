#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>

#include <iostream>

/*
 * This example "welds" triangle vertices together by taking as
 * input "triangle soup" and eliminating redundant vertex positions
 * and shared edges.  A connected mesh is the result.
 * 
 *
 * Input: 9 vertices representing a mesh with 3 triangles
 *  
 *  Mesh              Vertices 
 *    ------           (2)      (5)--(4)    (8)      
 *    | \ 2| \          | \       \   |      | \
 *    |  \ |  \   <->   |  \       \  |      |  \
 *    | 0 \| 1 \        |   \       \ |      |   \
 *    -----------      (0)--(1)      (3)    (6)--(7)
 *
 *   (vertex 1 equals vertex 3, vertex 2 equals vertex 5, ...)
 *
 * Output: mesh representation with 5 vertices and 9 indices
 *
 *  Vertices            Indices
 *   (1)--(3)            [(0,2,1),
 *    | \  | \            (2,3,1), 
 *    |  \ |  \           (2,4,3)]
 *    |   \|   \
 *   (0)--(2)--(4)
 */

// define a 2d float vector
typedef thrust::tuple<float,float> vec2;

int main(void)
{
    // allocate memory for input mesh representation
    thrust::device_vector<vec2> input(9);

    input[0] = vec2(0,0);  // First Triangle
    input[1] = vec2(1,0);
    input[2] = vec2(0,1);
    input[3] = vec2(1,0);  // Second Triangle
    input[4] = vec2(1,1);
    input[5] = vec2(0,1);
    input[6] = vec2(1,0);  // Third Triangle
    input[7] = vec2(2,0);
    input[8] = vec2(1,1);

    // allocate space for output mesh representation
    thrust::device_vector<vec2>         vertices = input;
    thrust::device_vector<unsigned int> indices(input.size());

    // sort vertices to bring duplicates together
    thrust::sort(vertices.begin(), vertices.end());

    // find unique vertices and erase redundancies
    vertices.erase(thrust::unique(vertices.begin(), vertices.end()), vertices.end());

    // find index of each input vertex in the list of unique vertices
    thrust::lower_bound(vertices.begin(), vertices.end(),
                        input.begin(), input.end(),
                        indices.begin());

    // print output mesh representation
    std::cout << "Output Representation" << std::endl;
    for(size_t i = 0; i < vertices.size(); i++)
    {
        vec2 v = vertices[i];
        std::cout << " vertices[" << i << "] = (" << thrust::get<0>(v) << "," << thrust::get<1>(v) << ")" << std::endl;
    }
    for(size_t i = 0; i < indices.size(); i++)
    {
        std::cout << " indices[" << i << "] = " << indices[i] << std::endl;
    }

    return 0;
}

