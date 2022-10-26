void BoxMesh(float width, float height, float OLwidth)
{
    MeshFilter mf = GetComponent<MeshFilter>();
    Mesh mesh = new Mesh();
    mf.mesh = mesh;

    MeshFilter mfOL = GetComponent<MeshFilter>();
    Mesh meshOL = new Mesh();
    mfOL.mesh = meshOL;

    //Verticies
    Vector3[] verticies = new Vector3[4]
    {
        new Vector3(0,0,0), new Vector3(0, height, 0), new Vector3(width, height, 0), new Vector3(width, 0, 0)
    };

    //Verticies Outline
    Vector3[] verticiesOL = new Vector3[4]
    {
        new Vector3(-OLwidth,-OLwidth,0), new Vector3(-OLwidth, height + OLwidth, 0), new Vector3(width + OLwidth, height + OLwidth, 0), new Vector3(width + OLwidth, -OLwidth, 0)
    };

    //Triangles
    int[] tri = new int[6];

    tri[0] = 0;
    tri[1] = 1;
    tri[2] = 3;

    tri[3] = 1;
    tri[4] = 2;
    tri[5] = 3;

    //normals
    
    Vector3[] normals = new Vector3[4];

    normals[0] = -Vector3.forward;
    normals[1] = -Vector3.forward;
    normals[2] = -Vector3.forward;
    normals[3] = -Vector3.forward;

    //UVs
    
    Vector2[] uv = new Vector2[4];
    
    uv[0] = new Vector2(0, 0);
    uv[1] = new Vector2(0, 1);
    uv[2] = new Vector2(1, 1);
    uv[3] = new Vector2(1, 0);
    
    //initialise
    mesh.vertices = verticies;
    mesh.triangles = tri;
    mesh.normals = normals;
    mesh.uv = uv;
    
    meshOL.vertices = verticiesOL;
    meshOL.triangles = tri;
    meshOL.normals = normals;
    meshOL.uv = uv;

    //setting up collider
    polyCollider.pathCount = 1;

    Vector2[] path = new Vector2[4]
    {
        new Vector3(-OLwidth,-OLwidth,0), new Vector3(-OLwidth, height + OLwidth, 0), new Vector3(width + OLwidth, height + OLwidth, 0), new Vector3(width + OLwidth, -OLwidth, 0)
    };

    polyCollider.SetPath(0, path);

}
