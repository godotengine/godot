/// @ref gtx_matrix_factorisation

namespace glm
{
	template <length_t C, length_t R, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<C, R, T, Q> flipud(mat<C, R, T, Q> const& in)
	{
		mat<R, C, T, Q> tin = transpose(in);
		tin = fliplr(tin);
		mat<C, R, T, Q> out = transpose(tin);

		return out;
	}

	template <length_t C, length_t R, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER mat<C, R, T, Q> fliplr(mat<C, R, T, Q> const& in)
	{
		mat<C, R, T, Q> out;
		for (length_t i = 0; i < C; i++)
		{
			out[i] = in[(C - i) - 1];
		}

		return out;
	}

	template <length_t C, length_t R, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER void qr_decompose(mat<C, R, T, Q> const& in, mat<(C < R ? C : R), R, T, Q>& q, mat<C, (C < R ? C : R), T, Q>& r)
	{
		// Uses modified Gram-Schmidt method
		// Source: https://en.wikipedia.org/wiki/Gram–Schmidt_process
		// And https://en.wikipedia.org/wiki/QR_decomposition

		//For all the linearly independs columns of the input...
		// (there can be no more linearly independents columns than there are rows.)
		for (length_t i = 0; i < (C < R ? C : R); i++)
		{
			//Copy in Q the input's i-th column.
			q[i] = in[i];

			//j = [0,i[
			// Make that column orthogonal to all the previous ones by substracting to it the non-orthogonal projection of all the previous columns.
			// Also: Fill the zero elements of R
			for (length_t j = 0; j < i; j++)
			{
				q[i] -= dot(q[i], q[j])*q[j];
				r[j][i] = 0;
			}

			//Now, Q i-th column is orthogonal to all the previous columns. Normalize it.
			q[i] = normalize(q[i]);

			//j = [i,C[
			//Finally, compute the corresponding coefficients of R by computing the projection of the resulting column on the other columns of the input.
			for (length_t j = i; j < C; j++)
			{
				r[j][i] = dot(in[j], q[i]);
			}
		}
	}

	template <length_t C, length_t R, typename T, qualifier Q>
	GLM_FUNC_QUALIFIER void rq_decompose(mat<C, R, T, Q> const& in, mat<(C < R ? C : R), R, T, Q>& r, mat<C, (C < R ? C : R), T, Q>& q)
	{
		// From https://en.wikipedia.org/wiki/QR_decomposition:
		// The RQ decomposition transforms a matrix A into the product of an upper triangular matrix R (also known as right-triangular) and an orthogonal matrix Q. The only difference from QR decomposition is the order of these matrices.
		// QR decomposition is Gram–Schmidt orthogonalization of columns of A, started from the first column.
		// RQ decomposition is Gram–Schmidt orthogonalization of rows of A, started from the last row.

		mat<R, C, T, Q> tin = transpose(in);
		tin = fliplr(tin);

		mat<R, (C < R ? C : R), T, Q> tr;
		mat<(C < R ? C : R), C, T, Q> tq;
		qr_decompose(tin, tq, tr);

		tr = fliplr(tr);
		r = transpose(tr);
		r = fliplr(r);

		tq = fliplr(tq);
		q = transpose(tq);
	}
} //namespace glm
