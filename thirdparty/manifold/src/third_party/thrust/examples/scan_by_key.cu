#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <iostream>

// BinaryPredicate for the head flag segment representation
// equivalent to thrust::not2(thrust::project2nd<int,int>()));
template <typename HeadFlagType>
struct head_flag_predicate 
    : public thrust::binary_function<HeadFlagType,HeadFlagType,bool>
{
    __host__ __device__
    bool operator()(HeadFlagType, HeadFlagType right) const
    {
        return !right;
    }
};

template <typename Vector>
void print(const Vector& v)
{
  for(size_t i = 0; i < v.size(); i++)
    std::cout << v[i] << " ";
  std::cout << "\n";
}

int main(void)
{
    int keys[]   = {0,0,0,1,1,2,2,2,2,3,4,4,5,5,5};  // segments represented with keys
    int flags[]  = {1,0,0,1,0,1,0,0,0,1,1,0,1,0,0};  // segments represented with head flags
    int values[] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2};  // values corresponding to each key

    int N = sizeof(keys) / sizeof(int); // number of elements

    // copy input data to device
    thrust::device_vector<int> d_keys  (keys,   keys   + N);
    thrust::device_vector<int> d_flags (flags,  flags  + N);
    thrust::device_vector<int> d_values(values, values + N);
    
    // allocate storage for output
    thrust::device_vector<int> d_output(N);

    // inclusive scan using keys
    thrust::inclusive_scan_by_key
      (d_keys.begin(), d_keys.end(),
       d_values.begin(),
       d_output.begin());
   
    std::cout << "Inclusive Segmented Scan w/ Key Sequence\n";
    std::cout << " keys          : ";  print(d_keys);
    std::cout << " input values  : ";  print(d_values);
    std::cout << " output values : ";  print(d_output);
    
    // inclusive scan using head flags
    thrust::inclusive_scan_by_key
      (d_flags.begin(), d_flags.end(),
       d_values.begin(), 
       d_output.begin(),
       head_flag_predicate<int>());
    
    std::cout << "\nInclusive Segmented Scan w/ Head Flag Sequence\n";
    std::cout << " head flags    : ";  print(d_flags);
    std::cout << " input values  : ";  print(d_values);
    std::cout << " output values : ";  print(d_output);
    
    // exclusive scan using keys
    thrust::exclusive_scan_by_key
      (d_keys.begin(), d_keys.end(),
       d_values.begin(),
       d_output.begin());
   
    std::cout << "\nExclusive Segmented Scan w/ Key Sequence\n";
    std::cout << " keys          : ";  print(d_keys);
    std::cout << " input values  : ";  print(d_values);
    std::cout << " output values : ";  print(d_output);
    
    // exclusive scan using head flags
    thrust::exclusive_scan_by_key
      (d_flags.begin(), d_flags.end(),
       d_values.begin(), 
       d_output.begin(),
       0,
       head_flag_predicate<int>());
    
    std::cout << "\nExclusive Segmented Scan w/ Head Flag Sequence\n";
    std::cout << " head flags    : ";  print(d_flags);
    std::cout << " input values  : ";  print(d_values);
    std::cout << " output values : ";  print(d_output);


    return 0;
}

