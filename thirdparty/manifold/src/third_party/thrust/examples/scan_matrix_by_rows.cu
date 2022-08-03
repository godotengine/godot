#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <assert.h>

// We have a matrix stored in a `thrust::device_vector`. We want to perform a
// scan on each row of a matrix.

__host__
void scan_matrix_by_rows0(thrust::device_vector<int>& u, int n, int m) {
  // Here, we launch a separate scan for each row in the matrix. This works,
  // but each kernel only does a small amount of work. It would be better if we
  // could launch one big kernel for the entire matrix.
  for (int i = 0; i < n; ++i)
    thrust::inclusive_scan(u.begin() + m * i, u.begin() + m * (i + 1),
                           u.begin() + m * i);
}

// We can batch the operation using `thrust::inclusive_scan_by_key`, which
// scans each group of consecutive equal keys. All we need to do is generate
// the right key sequence. We want the keys for elements on the same row to
// be identical.

// So first, we define an unary function object which takes the index of an
// element and returns the row that it belongs to.

struct which_row : thrust::unary_function<int, int> {
  int row_length;

  __host__ __device__
  which_row(int row_length_) : row_length(row_length_) {}

  __host__ __device__
  int operator()(int idx) const {
    return idx / row_length;
  }
};

__host__
void scan_matrix_by_rows1(thrust::device_vector<int>& u, int n, int m) {
  // This `thrust::counting_iterator` represents the index of the element.
  thrust::counting_iterator<int> c_first(0);

  // We construct a `thrust::transform_iterator` which applies the `which_row`
  // function object to the index of each element.
  thrust::transform_iterator<which_row, thrust::counting_iterator<int> >
    t_first(c_first, which_row(m));

  // Finally, we use our `thrust::transform_iterator` as the key sequence to
  // `thrust::inclusive_scan_by_key`.
  thrust::inclusive_scan_by_key(t_first, t_first + n * m, u.begin(), u.begin());
}

int main() {
  int const n = 4;
  int const m = 5;

  thrust::device_vector<int> u0(n * m);
  thrust::sequence(u0.begin(), u0.end());
  scan_matrix_by_rows0(u0, n, m);

  thrust::device_vector<int> u1(n * m);
  thrust::sequence(u1.begin(), u1.end());
  scan_matrix_by_rows1(u1, n, m);

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      assert(u0[j + m * i] == u1[j + m * i]);
}

