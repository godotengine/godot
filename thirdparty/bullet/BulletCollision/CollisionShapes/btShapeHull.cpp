/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

//btShapeHull was implemented by John McCutchan.

#include "btShapeHull.h"
#include "LinearMath/btConvexHull.h"

#define NUM_UNITSPHERE_POINTS 42
#define NUM_UNITSPHERE_POINTS_HIGHRES 256

btShapeHull::btShapeHull(const btConvexShape* shape)
{
	m_shape = shape;
	m_vertices.clear();
	m_indices.clear();
	m_numIndices = 0;
}

btShapeHull::~btShapeHull()
{
	m_indices.clear();
	m_vertices.clear();
}

bool btShapeHull::buildHull(btScalar /*margin*/, int highres)
{
	
	int numSampleDirections = highres ? NUM_UNITSPHERE_POINTS_HIGHRES : NUM_UNITSPHERE_POINTS;
	btVector3 supportPoints[NUM_UNITSPHERE_POINTS_HIGHRES + MAX_PREFERRED_PENETRATION_DIRECTIONS * 2];
	int i;
	for (i = 0; i < numSampleDirections; i++)
	{
		supportPoints[i] = m_shape->localGetSupportingVertex(getUnitSpherePoints(highres)[i]);
	}

	int numPDA = m_shape->getNumPreferredPenetrationDirections();
	if (numPDA)
	{
		for (int s = 0; s < numPDA; s++)
		{
			btVector3 norm;
			m_shape->getPreferredPenetrationDirection(s, norm);
			supportPoints[i++] = m_shape->localGetSupportingVertex(norm);
			numSampleDirections++;
		}
	}
	HullDesc hd;
	hd.mFlags = QF_TRIANGLES;
	hd.mVcount = static_cast<unsigned int>(numSampleDirections);

#ifdef BT_USE_DOUBLE_PRECISION
	hd.mVertices = &supportPoints[0];
	hd.mVertexStride = sizeof(btVector3);
#else
	hd.mVertices = &supportPoints[0];
	hd.mVertexStride = sizeof(btVector3);
#endif

	HullLibrary hl;
	HullResult hr;
	if (hl.CreateConvexHull(hd, hr) == QE_FAIL)
	{
		return false;
	}

	m_vertices.resize(static_cast<int>(hr.mNumOutputVertices));

	for (i = 0; i < static_cast<int>(hr.mNumOutputVertices); i++)
	{
		m_vertices[i] = hr.m_OutputVertices[i];
	}
	m_numIndices = hr.mNumIndices;
	m_indices.resize(static_cast<int>(m_numIndices));
	for (i = 0; i < static_cast<int>(m_numIndices); i++)
	{
		m_indices[i] = hr.m_Indices[i];
	}

	// free temporary hull result that we just copied
	hl.ReleaseResult(hr);

	return true;
}

int btShapeHull::numTriangles() const
{
	return static_cast<int>(m_numIndices / 3);
}

int btShapeHull::numVertices() const
{
	return m_vertices.size();
}

int btShapeHull::numIndices() const
{
	return static_cast<int>(m_numIndices);
}

btVector3* btShapeHull::getUnitSpherePoints(int highres)
{
	static btVector3 sUnitSpherePointsHighres[NUM_UNITSPHERE_POINTS_HIGHRES + MAX_PREFERRED_PENETRATION_DIRECTIONS * 2] =
		{
			btVector3(btScalar(0.997604), btScalar(0.067004), btScalar(0.017144)),
			btVector3(btScalar(0.984139), btScalar(-0.086784), btScalar(-0.154427)),
			btVector3(btScalar(0.971065), btScalar(0.124164), btScalar(-0.203224)),
			btVector3(btScalar(0.955844), btScalar(0.291173), btScalar(-0.037704)),
			btVector3(btScalar(0.957405), btScalar(0.212238), btScalar(0.195157)),
			btVector3(btScalar(0.971650), btScalar(-0.012709), btScalar(0.235561)),
			btVector3(btScalar(0.984920), btScalar(-0.161831), btScalar(0.059695)),
			btVector3(btScalar(0.946673), btScalar(-0.299288), btScalar(-0.117536)),
			btVector3(btScalar(0.922670), btScalar(-0.219186), btScalar(-0.317019)),
			btVector3(btScalar(0.928134), btScalar(-0.007265), btScalar(-0.371867)),
			btVector3(btScalar(0.875642), btScalar(0.198434), btScalar(-0.439988)),
			btVector3(btScalar(0.908035), btScalar(0.325975), btScalar(-0.262562)),
			btVector3(btScalar(0.864519), btScalar(0.488706), btScalar(-0.116755)),
			btVector3(btScalar(0.893009), btScalar(0.428046), btScalar(0.137185)),
			btVector3(btScalar(0.857494), btScalar(0.362137), btScalar(0.364776)),
			btVector3(btScalar(0.900815), btScalar(0.132524), btScalar(0.412987)),
			btVector3(btScalar(0.934964), btScalar(-0.241739), btScalar(0.259179)),
			btVector3(btScalar(0.894570), btScalar(-0.103504), btScalar(0.434263)),
			btVector3(btScalar(0.922085), btScalar(-0.376668), btScalar(0.086241)),
			btVector3(btScalar(0.862177), btScalar(-0.499154), btScalar(-0.085330)),
			btVector3(btScalar(0.861982), btScalar(-0.420218), btScalar(-0.282861)),
			btVector3(btScalar(0.818076), btScalar(-0.328256), btScalar(-0.471804)),
			btVector3(btScalar(0.762657), btScalar(-0.179329), btScalar(-0.621124)),
			btVector3(btScalar(0.826857), btScalar(0.019760), btScalar(-0.561786)),
			btVector3(btScalar(0.731434), btScalar(0.206599), btScalar(-0.649817)),
			btVector3(btScalar(0.769486), btScalar(0.379052), btScalar(-0.513770)),
			btVector3(btScalar(0.796806), btScalar(0.507176), btScalar(-0.328145)),
			btVector3(btScalar(0.679722), btScalar(0.684101), btScalar(-0.264123)),
			btVector3(btScalar(0.786854), btScalar(0.614886), btScalar(0.050912)),
			btVector3(btScalar(0.769486), btScalar(0.571141), btScalar(0.285139)),
			btVector3(btScalar(0.707432), btScalar(0.492789), btScalar(0.506288)),
			btVector3(btScalar(0.774560), btScalar(0.268037), btScalar(0.572652)),
			btVector3(btScalar(0.796220), btScalar(0.031230), btScalar(0.604077)),
			btVector3(btScalar(0.837395), btScalar(-0.320285), btScalar(0.442461)),
			btVector3(btScalar(0.848127), btScalar(-0.450548), btScalar(0.278307)),
			btVector3(btScalar(0.775536), btScalar(-0.206354), btScalar(0.596465)),
			btVector3(btScalar(0.816320), btScalar(-0.567007), btScalar(0.109469)),
			btVector3(btScalar(0.741191), btScalar(-0.668690), btScalar(-0.056832)),
			btVector3(btScalar(0.755632), btScalar(-0.602975), btScalar(-0.254949)),
			btVector3(btScalar(0.720311), btScalar(-0.521318), btScalar(-0.457165)),
			btVector3(btScalar(0.670746), btScalar(-0.386583), btScalar(-0.632835)),
			btVector3(btScalar(0.587031), btScalar(-0.219769), btScalar(-0.778836)),
			btVector3(btScalar(0.676015), btScalar(-0.003182), btScalar(-0.736676)),
			btVector3(btScalar(0.566932), btScalar(0.186963), btScalar(-0.802064)),
			btVector3(btScalar(0.618254), btScalar(0.398105), btScalar(-0.677533)),
			btVector3(btScalar(0.653964), btScalar(0.575224), btScalar(-0.490933)),
			btVector3(btScalar(0.525367), btScalar(0.743205), btScalar(-0.414028)),
			btVector3(btScalar(0.506439), btScalar(0.836528), btScalar(-0.208885)),
			btVector3(btScalar(0.651427), btScalar(0.756426), btScalar(-0.056247)),
			btVector3(btScalar(0.641670), btScalar(0.745149), btScalar(0.180908)),
			btVector3(btScalar(0.602643), btScalar(0.687211), btScalar(0.405180)),
			btVector3(btScalar(0.516586), btScalar(0.596999), btScalar(0.613447)),
			btVector3(btScalar(0.602252), btScalar(0.387801), btScalar(0.697573)),
			btVector3(btScalar(0.646549), btScalar(0.153911), btScalar(0.746956)),
			btVector3(btScalar(0.650842), btScalar(-0.087756), btScalar(0.753983)),
			btVector3(btScalar(0.740411), btScalar(-0.497404), btScalar(0.451830)),
			btVector3(btScalar(0.726946), btScalar(-0.619890), btScalar(0.295093)),
			btVector3(btScalar(0.637768), btScalar(-0.313092), btScalar(0.703624)),
			btVector3(btScalar(0.678942), btScalar(-0.722934), btScalar(0.126645)),
			btVector3(btScalar(0.489072), btScalar(-0.867195), btScalar(-0.092942)),
			btVector3(btScalar(0.622742), btScalar(-0.757541), btScalar(-0.194636)),
			btVector3(btScalar(0.596788), btScalar(-0.693576), btScalar(-0.403098)),
			btVector3(btScalar(0.550150), btScalar(-0.582172), btScalar(-0.598287)),
			btVector3(btScalar(0.474436), btScalar(-0.429745), btScalar(-0.768101)),
			btVector3(btScalar(0.372574), btScalar(-0.246016), btScalar(-0.894583)),
			btVector3(btScalar(0.480095), btScalar(-0.026513), btScalar(-0.876626)),
			btVector3(btScalar(0.352474), btScalar(0.177242), btScalar(-0.918787)),
			btVector3(btScalar(0.441848), btScalar(0.374386), btScalar(-0.814946)),
			btVector3(btScalar(0.492389), btScalar(0.582223), btScalar(-0.646693)),
			btVector3(btScalar(0.343498), btScalar(0.866080), btScalar(-0.362693)),
			btVector3(btScalar(0.362036), btScalar(0.745149), btScalar(-0.559639)),
			btVector3(btScalar(0.334131), btScalar(0.937044), btScalar(-0.099774)),
			btVector3(btScalar(0.486925), btScalar(0.871718), btScalar(0.052473)),
			btVector3(btScalar(0.452776), btScalar(0.845665), btScalar(0.281820)),
			btVector3(btScalar(0.399503), btScalar(0.771785), btScalar(0.494576)),
			btVector3(btScalar(0.296469), btScalar(0.673018), btScalar(0.677469)),
			btVector3(btScalar(0.392088), btScalar(0.479179), btScalar(0.785213)),
			btVector3(btScalar(0.452190), btScalar(0.252094), btScalar(0.855286)),
			btVector3(btScalar(0.478339), btScalar(0.013149), btScalar(0.877928)),
			btVector3(btScalar(0.481656), btScalar(-0.219380), btScalar(0.848259)),
			btVector3(btScalar(0.615327), btScalar(-0.494293), btScalar(0.613837)),
			btVector3(btScalar(0.594642), btScalar(-0.650414), btScalar(0.472325)),
			btVector3(btScalar(0.562249), btScalar(-0.771345), btScalar(0.297631)),
			btVector3(btScalar(0.467411), btScalar(-0.437133), btScalar(0.768231)),
			btVector3(btScalar(0.519513), btScalar(-0.847947), btScalar(0.103808)),
			btVector3(btScalar(0.297640), btScalar(-0.938159), btScalar(-0.176288)),
			btVector3(btScalar(0.446727), btScalar(-0.838615), btScalar(-0.311359)),
			btVector3(btScalar(0.331790), btScalar(-0.942437), btScalar(0.040762)),
			btVector3(btScalar(0.413358), btScalar(-0.748403), btScalar(-0.518259)),
			btVector3(btScalar(0.347596), btScalar(-0.621640), btScalar(-0.701737)),
			btVector3(btScalar(0.249831), btScalar(-0.456186), btScalar(-0.853984)),
			btVector3(btScalar(0.131772), btScalar(-0.262931), btScalar(-0.955678)),
			btVector3(btScalar(0.247099), btScalar(-0.042261), btScalar(-0.967975)),
			btVector3(btScalar(0.113624), btScalar(0.165965), btScalar(-0.979491)),
			btVector3(btScalar(0.217438), btScalar(0.374580), btScalar(-0.901220)),
			btVector3(btScalar(0.307983), btScalar(0.554615), btScalar(-0.772786)),
			btVector3(btScalar(0.166702), btScalar(0.953181), btScalar(-0.252021)),
			btVector3(btScalar(0.172751), btScalar(0.844499), btScalar(-0.506743)),
			btVector3(btScalar(0.177630), btScalar(0.711125), btScalar(-0.679876)),
			btVector3(btScalar(0.120064), btScalar(0.992260), btScalar(-0.030482)),
			btVector3(btScalar(0.289640), btScalar(0.949098), btScalar(0.122546)),
			btVector3(btScalar(0.239879), btScalar(0.909047), btScalar(0.340377)),
			btVector3(btScalar(0.181142), btScalar(0.821363), btScalar(0.540641)),
			btVector3(btScalar(0.066986), btScalar(0.719097), btScalar(0.691327)),
			btVector3(btScalar(0.156750), btScalar(0.545478), btScalar(0.823079)),
			btVector3(btScalar(0.236172), btScalar(0.342306), btScalar(0.909353)),
			btVector3(btScalar(0.277541), btScalar(0.112693), btScalar(0.953856)),
			btVector3(btScalar(0.295299), btScalar(-0.121974), btScalar(0.947415)),
			btVector3(btScalar(0.287883), btScalar(-0.349254), btScalar(0.891591)),
			btVector3(btScalar(0.437165), btScalar(-0.634666), btScalar(0.636869)),
			btVector3(btScalar(0.407113), btScalar(-0.784954), btScalar(0.466664)),
			btVector3(btScalar(0.375111), btScalar(-0.888193), btScalar(0.264839)),
			btVector3(btScalar(0.275394), btScalar(-0.560591), btScalar(0.780723)),
			btVector3(btScalar(0.122015), btScalar(-0.992209), btScalar(-0.024821)),
			btVector3(btScalar(0.087866), btScalar(-0.966156), btScalar(-0.241676)),
			btVector3(btScalar(0.239489), btScalar(-0.885665), btScalar(-0.397437)),
			btVector3(btScalar(0.167287), btScalar(-0.965184), btScalar(0.200817)),
			btVector3(btScalar(0.201632), btScalar(-0.776789), btScalar(-0.596335)),
			btVector3(btScalar(0.122015), btScalar(-0.637971), btScalar(-0.760098)),
			btVector3(btScalar(0.008054), btScalar(-0.464741), btScalar(-0.885214)),
			btVector3(btScalar(-0.116054), btScalar(-0.271096), btScalar(-0.955482)),
			btVector3(btScalar(-0.000727), btScalar(-0.056065), btScalar(-0.998424)),
			btVector3(btScalar(-0.134007), btScalar(0.152939), btScalar(-0.978905)),
			btVector3(btScalar(-0.025900), btScalar(0.366026), btScalar(-0.930108)),
			btVector3(btScalar(0.081231), btScalar(0.557337), btScalar(-0.826072)),
			btVector3(btScalar(-0.002874), btScalar(0.917213), btScalar(-0.398023)),
			btVector3(btScalar(-0.050683), btScalar(0.981761), btScalar(-0.182534)),
			btVector3(btScalar(-0.040536), btScalar(0.710153), btScalar(-0.702713)),
			btVector3(btScalar(-0.139081), btScalar(0.827973), btScalar(-0.543048)),
			btVector3(btScalar(-0.101029), btScalar(0.994010), btScalar(0.041152)),
			btVector3(btScalar(0.069328), btScalar(0.978067), btScalar(0.196133)),
			btVector3(btScalar(0.023860), btScalar(0.911380), btScalar(0.410645)),
			btVector3(btScalar(-0.153521), btScalar(0.736789), btScalar(0.658145)),
			btVector3(btScalar(-0.070002), btScalar(0.591750), btScalar(0.802780)),
			btVector3(btScalar(0.002590), btScalar(0.312948), btScalar(0.949562)),
			btVector3(btScalar(0.090988), btScalar(-0.020680), btScalar(0.995627)),
			btVector3(btScalar(0.088842), btScalar(-0.250099), btScalar(0.964006)),
			btVector3(btScalar(0.083378), btScalar(-0.470185), btScalar(0.878318)),
			btVector3(btScalar(0.240074), btScalar(-0.749764), btScalar(0.616374)),
			btVector3(btScalar(0.210803), btScalar(-0.885860), btScalar(0.412987)),
			btVector3(btScalar(0.077524), btScalar(-0.660524), btScalar(0.746565)),
			btVector3(btScalar(-0.096736), btScalar(-0.990070), btScalar(-0.100945)),
			btVector3(btScalar(-0.052634), btScalar(-0.990264), btScalar(0.127426)),
			btVector3(btScalar(-0.106102), btScalar(-0.938354), btScalar(-0.328340)),
			btVector3(btScalar(0.013323), btScalar(-0.863112), btScalar(-0.504596)),
			btVector3(btScalar(-0.002093), btScalar(-0.936993), btScalar(0.349161)),
			btVector3(btScalar(-0.106297), btScalar(-0.636610), btScalar(-0.763612)),
			btVector3(btScalar(-0.229430), btScalar(-0.463769), btScalar(-0.855546)),
			btVector3(btScalar(-0.245236), btScalar(-0.066175), btScalar(-0.966999)),
			btVector3(btScalar(-0.351587), btScalar(-0.270513), btScalar(-0.896145)),
			btVector3(btScalar(-0.370906), btScalar(0.133108), btScalar(-0.918982)),
			btVector3(btScalar(-0.264360), btScalar(0.346000), btScalar(-0.900049)),
			btVector3(btScalar(-0.151375), btScalar(0.543728), btScalar(-0.825291)),
			btVector3(btScalar(-0.218697), btScalar(0.912741), btScalar(-0.344346)),
			btVector3(btScalar(-0.274507), btScalar(0.953764), btScalar(-0.121635)),
			btVector3(btScalar(-0.259677), btScalar(0.692266), btScalar(-0.673044)),
			btVector3(btScalar(-0.350416), btScalar(0.798810), btScalar(-0.488786)),
			btVector3(btScalar(-0.320170), btScalar(0.941127), btScalar(0.108297)),
			btVector3(btScalar(-0.147667), btScalar(0.952792), btScalar(0.265034)),
			btVector3(btScalar(-0.188061), btScalar(0.860636), btScalar(0.472910)),
			btVector3(btScalar(-0.370906), btScalar(0.739900), btScalar(0.560941)),
			btVector3(btScalar(-0.297143), btScalar(0.585334), btScalar(0.754178)),
			btVector3(btScalar(-0.189622), btScalar(0.428241), btScalar(0.883393)),
			btVector3(btScalar(-0.091272), btScalar(0.098695), btScalar(0.990747)),
			btVector3(btScalar(-0.256945), btScalar(0.228375), btScalar(0.938827)),
			btVector3(btScalar(-0.111761), btScalar(-0.133251), btScalar(0.984696)),
			btVector3(btScalar(-0.118006), btScalar(-0.356253), btScalar(0.926725)),
			btVector3(btScalar(-0.119372), btScalar(-0.563896), btScalar(0.817029)),
			btVector3(btScalar(0.041228), btScalar(-0.833949), btScalar(0.550010)),
			btVector3(btScalar(-0.121909), btScalar(-0.736543), btScalar(0.665172)),
			btVector3(btScalar(-0.307681), btScalar(-0.931160), btScalar(-0.195026)),
			btVector3(btScalar(-0.283679), btScalar(-0.957990), btScalar(0.041348)),
			btVector3(btScalar(-0.227284), btScalar(-0.935243), btScalar(0.270890)),
			btVector3(btScalar(-0.293436), btScalar(-0.858252), btScalar(-0.420860)),
			btVector3(btScalar(-0.175767), btScalar(-0.780677), btScalar(-0.599262)),
			btVector3(btScalar(-0.170108), btScalar(-0.858835), btScalar(0.482865)),
			btVector3(btScalar(-0.332854), btScalar(-0.635055), btScalar(-0.696857)),
			btVector3(btScalar(-0.447791), btScalar(-0.445299), btScalar(-0.775128)),
			btVector3(btScalar(-0.470622), btScalar(-0.074146), btScalar(-0.879164)),
			btVector3(btScalar(-0.639417), btScalar(-0.340505), btScalar(-0.689049)),
			btVector3(btScalar(-0.598438), btScalar(0.104722), btScalar(-0.794256)),
			btVector3(btScalar(-0.488575), btScalar(0.307699), btScalar(-0.816313)),
			btVector3(btScalar(-0.379882), btScalar(0.513592), btScalar(-0.769077)),
			btVector3(btScalar(-0.425740), btScalar(0.862775), btScalar(-0.272516)),
			btVector3(btScalar(-0.480769), btScalar(0.875412), btScalar(-0.048439)),
			btVector3(btScalar(-0.467890), btScalar(0.648716), btScalar(-0.600043)),
			btVector3(btScalar(-0.543799), btScalar(0.730956), btScalar(-0.411881)),
			btVector3(btScalar(-0.516284), btScalar(0.838277), btScalar(0.174076)),
			btVector3(btScalar(-0.353343), btScalar(0.876384), btScalar(0.326519)),
			btVector3(btScalar(-0.572875), btScalar(0.614497), btScalar(0.542007)),
			btVector3(btScalar(-0.503600), btScalar(0.497261), btScalar(0.706161)),
			btVector3(btScalar(-0.530920), btScalar(0.754870), btScalar(0.384685)),
			btVector3(btScalar(-0.395884), btScalar(0.366414), btScalar(0.841818)),
			btVector3(btScalar(-0.300656), btScalar(0.001678), btScalar(0.953661)),
			btVector3(btScalar(-0.461060), btScalar(0.146912), btScalar(0.875000)),
			btVector3(btScalar(-0.315486), btScalar(-0.232212), btScalar(0.919893)),
			btVector3(btScalar(-0.323682), btScalar(-0.449187), btScalar(0.832644)),
			btVector3(btScalar(-0.318999), btScalar(-0.639527), btScalar(0.699134)),
			btVector3(btScalar(-0.496771), btScalar(-0.866029), btScalar(-0.055271)),
			btVector3(btScalar(-0.496771), btScalar(-0.816257), btScalar(-0.294377)),
			btVector3(btScalar(-0.456377), btScalar(-0.869528), btScalar(0.188130)),
			btVector3(btScalar(-0.380858), btScalar(-0.827144), btScalar(0.412792)),
			btVector3(btScalar(-0.449352), btScalar(-0.727405), btScalar(-0.518259)),
			btVector3(btScalar(-0.570533), btScalar(-0.551064), btScalar(-0.608632)),
			btVector3(btScalar(-0.656394), btScalar(-0.118280), btScalar(-0.744874)),
			btVector3(btScalar(-0.756696), btScalar(-0.438105), btScalar(-0.484882)),
			btVector3(btScalar(-0.801773), btScalar(-0.204798), btScalar(-0.561005)),
			btVector3(btScalar(-0.785186), btScalar(0.038618), btScalar(-0.617805)),
			btVector3(btScalar(-0.709082), btScalar(0.262399), btScalar(-0.654306)),
			btVector3(btScalar(-0.583412), btScalar(0.462265), btScalar(-0.667383)),
			btVector3(btScalar(-0.616001), btScalar(0.761286), btScalar(-0.201272)),
			btVector3(btScalar(-0.660687), btScalar(0.750204), btScalar(0.020072)),
			btVector3(btScalar(-0.744987), btScalar(0.435823), btScalar(-0.504791)),
			btVector3(btScalar(-0.713765), btScalar(0.605554), btScalar(-0.351373)),
			btVector3(btScalar(-0.686251), btScalar(0.687600), btScalar(0.236927)),
			btVector3(btScalar(-0.680201), btScalar(0.429407), btScalar(0.593732)),
			btVector3(btScalar(-0.733474), btScalar(0.546450), btScalar(0.403814)),
			btVector3(btScalar(-0.591023), btScalar(0.292923), btScalar(0.751445)),
			btVector3(btScalar(-0.500283), btScalar(-0.080757), btScalar(0.861922)),
			btVector3(btScalar(-0.643710), btScalar(0.070115), btScalar(0.761985)),
			btVector3(btScalar(-0.506332), btScalar(-0.308425), btScalar(0.805122)),
			btVector3(btScalar(-0.503015), btScalar(-0.509847), btScalar(0.697573)),
			btVector3(btScalar(-0.482525), btScalar(-0.682105), btScalar(0.549229)),
			btVector3(btScalar(-0.680396), btScalar(-0.716323), btScalar(-0.153451)),
			btVector3(btScalar(-0.658346), btScalar(-0.746264), btScalar(0.097562)),
			btVector3(btScalar(-0.653272), btScalar(-0.646915), btScalar(-0.392948)),
			btVector3(btScalar(-0.590828), btScalar(-0.732655), btScalar(0.337645)),
			btVector3(btScalar(-0.819140), btScalar(-0.518013), btScalar(-0.246166)),
			btVector3(btScalar(-0.900513), btScalar(-0.282178), btScalar(-0.330487)),
			btVector3(btScalar(-0.914953), btScalar(-0.028652), btScalar(-0.402122)),
			btVector3(btScalar(-0.859924), btScalar(0.220209), btScalar(-0.459898)),
			btVector3(btScalar(-0.777185), btScalar(0.613720), btScalar(-0.137836)),
			btVector3(btScalar(-0.805285), btScalar(0.586889), btScalar(0.082728)),
			btVector3(btScalar(-0.872413), btScalar(0.406077), btScalar(-0.271735)),
			btVector3(btScalar(-0.859339), btScalar(0.448072), btScalar(0.246101)),
			btVector3(btScalar(-0.757671), btScalar(0.216320), btScalar(0.615594)),
			btVector3(btScalar(-0.826165), btScalar(0.348139), btScalar(0.442851)),
			btVector3(btScalar(-0.671810), btScalar(-0.162803), btScalar(0.722557)),
			btVector3(btScalar(-0.796504), btScalar(-0.004543), btScalar(0.604468)),
			btVector3(btScalar(-0.676298), btScalar(-0.378223), btScalar(0.631794)),
			btVector3(btScalar(-0.668883), btScalar(-0.558258), btScalar(0.490673)),
			btVector3(btScalar(-0.821287), btScalar(-0.570118), btScalar(0.006994)),
			btVector3(btScalar(-0.767428), btScalar(-0.587810), btScalar(0.255470)),
			btVector3(btScalar(-0.933296), btScalar(-0.349837), btScalar(-0.079865)),
			btVector3(btScalar(-0.982667), btScalar(-0.100393), btScalar(-0.155208)),
			btVector3(btScalar(-0.961396), btScalar(0.160910), btScalar(-0.222938)),
			btVector3(btScalar(-0.934858), btScalar(0.354555), btScalar(-0.006864)),
			btVector3(btScalar(-0.941687), btScalar(0.229736), btScalar(0.245711)),
			btVector3(btScalar(-0.884317), btScalar(0.131552), btScalar(0.447536)),
			btVector3(btScalar(-0.810359), btScalar(-0.219769), btScalar(0.542788)),
			btVector3(btScalar(-0.915929), btScalar(-0.210048), btScalar(0.341743)),
			btVector3(btScalar(-0.816799), btScalar(-0.407192), btScalar(0.408303)),
			btVector3(btScalar(-0.903050), btScalar(-0.392416), btScalar(0.174076)),
			btVector3(btScalar(-0.980325), btScalar(-0.170969), btScalar(0.096586)),
			btVector3(btScalar(-0.995936), btScalar(0.084891), btScalar(0.029441)),
			btVector3(btScalar(-0.960031), btScalar(0.002650), btScalar(0.279283)),
		};
	static btVector3 sUnitSpherePoints[NUM_UNITSPHERE_POINTS + MAX_PREFERRED_PENETRATION_DIRECTIONS * 2] =
		{
			btVector3(btScalar(0.000000), btScalar(-0.000000), btScalar(-1.000000)),
			btVector3(btScalar(0.723608), btScalar(-0.525725), btScalar(-0.447219)),
			btVector3(btScalar(-0.276388), btScalar(-0.850649), btScalar(-0.447219)),
			btVector3(btScalar(-0.894426), btScalar(-0.000000), btScalar(-0.447216)),
			btVector3(btScalar(-0.276388), btScalar(0.850649), btScalar(-0.447220)),
			btVector3(btScalar(0.723608), btScalar(0.525725), btScalar(-0.447219)),
			btVector3(btScalar(0.276388), btScalar(-0.850649), btScalar(0.447220)),
			btVector3(btScalar(-0.723608), btScalar(-0.525725), btScalar(0.447219)),
			btVector3(btScalar(-0.723608), btScalar(0.525725), btScalar(0.447219)),
			btVector3(btScalar(0.276388), btScalar(0.850649), btScalar(0.447219)),
			btVector3(btScalar(0.894426), btScalar(0.000000), btScalar(0.447216)),
			btVector3(btScalar(-0.000000), btScalar(0.000000), btScalar(1.000000)),
			btVector3(btScalar(0.425323), btScalar(-0.309011), btScalar(-0.850654)),
			btVector3(btScalar(-0.162456), btScalar(-0.499995), btScalar(-0.850654)),
			btVector3(btScalar(0.262869), btScalar(-0.809012), btScalar(-0.525738)),
			btVector3(btScalar(0.425323), btScalar(0.309011), btScalar(-0.850654)),
			btVector3(btScalar(0.850648), btScalar(-0.000000), btScalar(-0.525736)),
			btVector3(btScalar(-0.525730), btScalar(-0.000000), btScalar(-0.850652)),
			btVector3(btScalar(-0.688190), btScalar(-0.499997), btScalar(-0.525736)),
			btVector3(btScalar(-0.162456), btScalar(0.499995), btScalar(-0.850654)),
			btVector3(btScalar(-0.688190), btScalar(0.499997), btScalar(-0.525736)),
			btVector3(btScalar(0.262869), btScalar(0.809012), btScalar(-0.525738)),
			btVector3(btScalar(0.951058), btScalar(0.309013), btScalar(0.000000)),
			btVector3(btScalar(0.951058), btScalar(-0.309013), btScalar(0.000000)),
			btVector3(btScalar(0.587786), btScalar(-0.809017), btScalar(0.000000)),
			btVector3(btScalar(0.000000), btScalar(-1.000000), btScalar(0.000000)),
			btVector3(btScalar(-0.587786), btScalar(-0.809017), btScalar(0.000000)),
			btVector3(btScalar(-0.951058), btScalar(-0.309013), btScalar(-0.000000)),
			btVector3(btScalar(-0.951058), btScalar(0.309013), btScalar(-0.000000)),
			btVector3(btScalar(-0.587786), btScalar(0.809017), btScalar(-0.000000)),
			btVector3(btScalar(-0.000000), btScalar(1.000000), btScalar(-0.000000)),
			btVector3(btScalar(0.587786), btScalar(0.809017), btScalar(-0.000000)),
			btVector3(btScalar(0.688190), btScalar(-0.499997), btScalar(0.525736)),
			btVector3(btScalar(-0.262869), btScalar(-0.809012), btScalar(0.525738)),
			btVector3(btScalar(-0.850648), btScalar(0.000000), btScalar(0.525736)),
			btVector3(btScalar(-0.262869), btScalar(0.809012), btScalar(0.525738)),
			btVector3(btScalar(0.688190), btScalar(0.499997), btScalar(0.525736)),
			btVector3(btScalar(0.525730), btScalar(0.000000), btScalar(0.850652)),
			btVector3(btScalar(0.162456), btScalar(-0.499995), btScalar(0.850654)),
			btVector3(btScalar(-0.425323), btScalar(-0.309011), btScalar(0.850654)),
			btVector3(btScalar(-0.425323), btScalar(0.309011), btScalar(0.850654)),
			btVector3(btScalar(0.162456), btScalar(0.499995), btScalar(0.850654))};
	if (highres)
		return sUnitSpherePointsHighres;
	return sUnitSpherePoints;
}
