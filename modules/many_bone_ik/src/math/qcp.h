/**************************************************************************/
/*  qcp.h                                                                 */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef QCP_H
#define QCP_H

#include "core/math/basis.h"
#include "core/math/vector3.h"
#include "core/variant/variant.h"

/**
 * Implementation of the Quaternionf-Based Characteristic Polynomial algorithm
 * for RMSD and Superposition calculations.
 * <p>
 * Usage:
 * <p>
 * The input consists of 2 Vector3 arrays of equal length. The input
 * coordinates are not changed.
 *
 * <pre>
 *    Vector3[] x = ...
 *    Vector3[] y = ...
 *    SuperPositionQCP qcp = new SuperPositionQCP();
 *    qcp.set(x, y);
 * </pre>
 * <p>
 * or with weighting factors [0 - 1]]
 *
 * <pre>
 *    double[] weights = ...
 *    qcp.set(x, y, weights);
 * </pre>
 * <p>
 * For maximum efficiency, create a SuperPositionQCP object once and reuse it.
 * <p>
 * A. Calculate rmsd only
 *
 * <pre>
 * double rmsd = qcp.getRmsd();
 * </pre>
 * <p>
 * B. Calculate a 4x4 transformation (Quaternionation and translation) matrix
 *
 * <pre>
 * Matrix4f Quaterniontrans = qcp.getTransformationMatrix();
 * </pre>
 * <p>
 * C. Get transformated points (y superposed onto the reference x)
 *
 * <pre>
 * Vector3[] ySuperposed = qcp.getTransformedCoordinates();
 * </pre>
 * <p>
 * Citations:
 * <p>
 * Liu P, Agrafiotis DK, & Theobald DL (2011) Reply to comment on: "Fast
 * determination of the optimal Quaternionation matrix for macromolecular
 * superpositions." Journal of Computational Chemistry 32(1):185-186.
 * [http://dx.doi.org/10.1002/jcc.21606]
 * <p>
 * Liu P, Agrafiotis DK, & Theobald DL (2010) "Fast determination of the optimal
 * Quaternionation matrix for macromolecular superpositions." Journal of Computational
 * Chemistry 31(7):1561-1563. [http://dx.doi.org/10.1002/jcc.21439]
 * <p>
 * Douglas L Theobald (2005) "Rapid calculation of RMSDs using a
 * quaternion-based characteristic polynomial." Acta Crystallogr A
 * 61(4):478-480. [http://dx.doi.org/10.1107/S0108767305015266 ]
 * <p>
 * This is an adoption of the original C code QCPQuaternion 1.4 (2012, October 10) to
 * Java. The original C source code is available from
 * http://theobald.brandeis.edu/qcp/ and was developed by
 * <p>
 * Douglas L. Theobald Department of Biochemistry MS 009 Brandeis University 415
 * South St Waltham, MA 02453 USA
 * <p>
 * dtheobald@brandeis.edu
 * <p>
 * Pu Liu Johnson & Johnson Pharmaceutical Research and Development, L.L.C. 665
 * Stockton Drive Exton, PA 19341 USA
 * <p>
 * pliu24@its.jnj.com
 * <p>
 *
 * @author Douglas L. Theobald (original C code)
 * @author Pu Liu (original C code)
 * @author Peter Rose (adopted to Java)
 * @author Aleix Lafita (adopted to Java)
 * @author Eron Gjoni (adopted to EWB IK)
 */
class QCP {
	double evec_prec = static_cast<double>(1E-6);
	double eval_prec = static_cast<double>(1E-11);

	PackedVector3Array target;

	PackedVector3Array moved;

	Vector<real_t> weight;
	double w_sum = 0;

	Vector3 target_center;
	Vector3 moved_center;

	double e0 = 0;
	double rmsd = 0;
	double Sxy = 0, Sxz = 0, Syx = 0, Syz = 0, Szx = 0, Szy = 0;
	double SxxpSyy = 0, Szz = 0, mxEigenV = 0, SyzmSzy = 0, SxzmSzx = 0, SxymSyx = 0;
	double SxxmSyy = 0, SxypSyx = 0, SxzpSzx = 0;
	double Syy = 0, Sxx = 0, SyzpSzy = 0;
	bool rmsd_calculated = false;
	bool transformation_calculated = false;
	bool inner_product_calculated = false;

	/**
	 * Calculates the RMSD value for superposition of y onto x. This requires the
	 * coordinates to be precentered.
	 *
	 * @param x
	 *            3f points of reference coordinate set
	 * @param y
	 *            3f points of coordinate set for superposition
	 */
	void calculate_rmsd(PackedVector3Array &x, PackedVector3Array &y);

	/**
	 * Calculates the inner product between two coordinate sets x and y (optionally
	 * weighted, if weights set through
	 * {@link #set(Vector3[], Vector3[], double[])}). It also calculates an upper
	 * bound of the most positive root of the key matrix.
	 * http://theobald.brandeis.edu/qcp/qcpQuaternion.c
	 *
	 * @param coords1
	 * @param coords2
	 * @return
	 */
	void inner_product(PackedVector3Array &coords1, PackedVector3Array &coords2);

	void calculate_rmsd(double r_length);

	void set(PackedVector3Array &r_target, PackedVector3Array &r_moved);

	Quaternion calculate_rotation();

	/**
	 * Sets the two input coordinate arrays and weight array. All input arrays must
	 * be of equal length. Input coordinates are not modified.
	 *
	 * @param fixed
	 *            3f points of reference coordinate set
	 * @param moved
	 *            3f points of coordinate set for superposition
	 * @param weight
	 *            a weight in the inclusive range [0,1] for each point
	 */
	void set(PackedVector3Array &p_moved, PackedVector3Array &p_target, Vector<real_t> &p_weight, bool p_translate);

	static void translate(Vector3 r_translate, PackedVector3Array &r_x);

	double get_rmsd(PackedVector3Array &r_fixed, PackedVector3Array &r_moved);
	Vector3 move_to_weighted_center(PackedVector3Array &r_to_center, Vector<real_t> &r_weight);

public:
	/**
	 * Constructor with option to set the precision values.
	 *
	 * @param evec_prec
	 *            required eigenvector precision
	 * @param eval_prec
	 *            required eigenvalue precision
	 */
	QCP(double p_evec_prec, double p_eval_prec);

	/**
	 * Return the RMSD of the superposition of input coordinate set y onto x. Note,
	 * this is the faster way to calculate an RMSD without actually superposing the
	 * two sets. The calculation is performed "lazy", meaning calculations are only
	 * performed if necessary.
	 *
	 * @return root mean square deviation for superposition of y onto x
	 */
	double get_rmsd();

	/**
	 * Weighted superposition.
	 *
	 * @param moved
	 * @param target
	 * @param weight array of weights for each equivalent point position
	 * @param translate translate
	 * @return
	 */
	Quaternion weighted_superpose(PackedVector3Array &p_moved, PackedVector3Array &p_target, Vector<real_t> &p_weight, bool translate);

	Quaternion get_rotation();
	Vector3 get_translation();
};

#endif // QCP_H
