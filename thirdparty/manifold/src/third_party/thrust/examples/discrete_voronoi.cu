#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "include/timer.h"

// Compute an approximate Voronoi Diagram with a Jump Flooding Algorithm (JFA)
//
// References
//   http://en.wikipedia.org/wiki/Voronoi_diagram
//   http://www.comp.nus.edu.sg/~tants/jfa.html
//   http://www.utdallas.edu/~guodongrong/Papers/Dissertation.pdf
//
// Thanks to David Coeurjolly for contributing this example



// minFunctor
// Tuple  = <seeds,seeds + k,seeds + m*k, seeds - k, 
//           seeds - m*k, seeds+ k+m*k,seeds + k-m*k,
//           seeds- k+m*k,seeds - k+m*k, i>
struct minFunctor
{
  int m, n, k;
  
  __host__ __device__
  minFunctor(int m, int n, int k)
    : m(m), n(n), k(k) {}


  //To decide I have to change my current Voronoi site
  __host__ __device__
      int minVoro(int x_i, int y_i, int p, int q)
      {    
          if (q == m*n)
              return p;

          // coordinates of points p and q
          int y_q =  q / m;
          int x_q =  q - y_q * m;
          int y_p =  p / m;
          int x_p =  p - y_p * m;
        
          // squared distances
          int d_iq = (x_i-x_q) * (x_i-x_q) + (y_i-y_q) * (y_i-y_q);
          int d_ip = (x_i-x_p) * (x_i-x_p) + (y_i-y_p) * (y_i-y_p);

          if (d_iq < d_ip)
              return q;  // q is closer
          else
              return p;
      }

  //For each point p+{-k,0,k}, we keep the Site with minimum distance
  template <typename Tuple>
  __host__ __device__
  int operator()(const Tuple &t)
  {
      //Current point and site
      int i = thrust::get<9>(t);
      int v = thrust::get<0>(t);

      //Current point coordinates
      int y = i / m;    
      int x = i - y * m;

      if (x >= k)
      {
          v = minVoro(x, y, v, thrust::get<3>(t));

          if (y >= k)
              v = minVoro(x, y, v, thrust::get<8>(t));

          if (y + k < n)
              v = minVoro(x, y, v, thrust::get<7>(t));
      }

      if (x + k < m)
      { 
          v = minVoro(x, y, v, thrust::get<1>(t));

          if (y >= k)
              v = minVoro(x, y, v, thrust::get<6>(t));
          if (y + k < n)
              v = minVoro(x, y, v, thrust::get<5>(t));
      }

      if (y >= k)
          v = minVoro(x, y, v, thrust::get<4>(t));
      if (y + k < n)
          v = minVoro(x, y, v, thrust::get<2>(t));

      //global return
      return v;
  }
};



// print an M-by-N array
template <typename T>
void print(int m, int n, const thrust::device_vector<T>& d_data)
{
    thrust::host_vector<T> h_data = d_data;

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
            std::cout << std::setw(4) << h_data[i * n + j] << " ";
        std::cout << "\n";
    }
}


void generate_random_sites(thrust::host_vector<int> &t, int Nb, int m, int n)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, m * n - 1);

  for(int k = 0; k < Nb; k++)
  {
      int index = dist(rng);
      t[index] = index + 1;
  }
}

//Export the tab to PGM image format
void vector_to_pgm(thrust::host_vector<int> &t, int m, int n, const char *out)
{
    assert(static_cast<int>(t.size()) == m * n &&
           "Vector size does not match image dims.");

    std::fstream f(out, std::fstream::out);
    f << "P2\n";
    f << m << " " << n << "\n";
    f << "253\n";

    //Hash function to map values to [0,255]
    auto to_grey_level = [](int in_value) -> int
    {
        return (71 * in_value) % 253;
    };

    for (int value : t)
    {
      f << to_grey_level(value) << " ";
    }
    f << "\n";
    f.close();
}

/************Main Jfa loop********************/
// Perform a jump with step k
void jfa(thrust::device_vector<int>& in,thrust::device_vector<int>& out, unsigned int k, int m, int n)
{
   thrust::transform(
        thrust::make_zip_iterator(
            thrust::make_tuple(in.begin(), 
                               in.begin() + k, 
                               in.begin() + m*k, 
                               in.begin() - k, 
                               in.begin() - m*k, 
                               in.begin() + k+m*k,
                               in.begin() + k-m*k,
                               in.begin() - k+m*k,
                               in.begin() - k-m*k,
                               thrust::counting_iterator<int>(0))),
        thrust::make_zip_iterator(
            thrust::make_tuple(in.begin(), 
				    		   in.begin() + k, 
                               in.begin() + m*k, 
                               in.begin() - k, 
                               in.begin() - m*k, 
                               in.begin() + k+m*k,
                               in.begin() + k-m*k,
                               in.begin() - k+m*k,
                               in.begin() - k-m*k,
                               thrust::counting_iterator<int>(0)))+ n*m,
        out.begin(),
        minFunctor(m,n,k));
}
/********************************************/

void display_time(timer& t)
{
  std::cout << "  ( "<< 1e3 * t.elapsed() << "ms )" << std::endl;
}

int main(void)
{
  int m = 2048; // number of rows
  int n = 2048; // number of columns  
  int s = 1000; // number of sites
  
  timer t;
 
  //Host vector to encode a 2D image
  std::cout << "[Inititialize " << m << "x" << n << " Image]" << std::endl;
  t.restart();
  thrust::host_vector<int> seeds_host(m*n, m*n);
  generate_random_sites(seeds_host,s,m,n);
  display_time(t);
  
  std::cout<<"[Copy to Device]" << std::endl;
  t.restart();
  thrust::device_vector<int> seeds = seeds_host;
  thrust::device_vector<int> temp(seeds);
  display_time(t);

  //JFA+1  : before entering the log(n) loop, we perform a jump with k=1
  std::cout<<"[JFA stepping]" << std::endl;
  t.restart();
  jfa(seeds,temp,1,m,n);
  seeds.swap(temp);
 
  //JFA : main loop with k=n/2, n/4, ..., 1
  for(int k = thrust::max(m,n) / 2; k > 0; k /= 2)
  {
    jfa(seeds,temp,k,m,n);
    seeds.swap(temp);
  }

  display_time(t);
  std::cout <<"  ( " <<  seeds.size() / (1e6 * t.elapsed()) << " MPixel/s ) " << std::endl;
  
  std::cout << "[Device to Host Copy]" << std::endl;
  t.restart();
  seeds_host = seeds;
  display_time(t);
  
  std::cout << "[PGM Export]" << std::endl;
  t.restart();
  vector_to_pgm(seeds_host, m, n, "discrete_voronoi.pgm");
  display_time(t);

  return 0;
}

