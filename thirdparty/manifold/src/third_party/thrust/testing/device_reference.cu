#include <unittest/unittest.h>
#include <thrust/device_vector.h>
#include <thrust/device_reference.h>

void TestDeviceReferenceConstructorFromDeviceReference(void)
{
    typedef int T;

    thrust::device_vector<T> v(1,0);
    thrust::device_reference<T> ref = v[0];

    // ref equals the object at v[0]
    ASSERT_EQUAL(v[0], ref);

    // the address of ref equals the address of v[0]
    ASSERT_EQUAL(&v[0], &ref);

    // modifying v[0] modifies ref
    v[0] = 13;
    ASSERT_EQUAL(13, ref);
    ASSERT_EQUAL(v[0], ref);

    // modifying ref modifies v[0]
    ref = 7;
    ASSERT_EQUAL(7, v[0]);
    ASSERT_EQUAL(v[0], ref);
}
DECLARE_UNITTEST(TestDeviceReferenceConstructorFromDeviceReference);

void TestDeviceReferenceConstructorFromDevicePointer(void)
{
    typedef int T;

    thrust::device_vector<T> v(1,0);
    thrust::device_ptr<T> ptr = &v[0];
    thrust::device_reference<T> ref(ptr);

    // ref equals the object pointed to by ptr
    ASSERT_EQUAL(*ptr, ref);

    // the address of ref equals ptr
    ASSERT_EQUAL(ptr, &ref);

    // modifying *ptr modifies ref
    *ptr = 13;
    ASSERT_EQUAL(13, ref);
    ASSERT_EQUAL(v[0], ref);

    // modifying ref modifies *ptr
    ref = 7;
    ASSERT_EQUAL(7, *ptr);
    ASSERT_EQUAL(v[0], ref);
}
DECLARE_UNITTEST(TestDeviceReferenceConstructorFromDevicePointer);

void TestDeviceReferenceAssignmentFromDeviceReference(void)
{
    // test same types
    typedef int T0;
    thrust::device_vector<T0> v0(2,0);
    thrust::device_reference<T0> ref0 = v0[0];
    thrust::device_reference<T0> ref1 = v0[1];

    ref0 = 13;

    ref1 = ref0;

    // ref1 equals 13
    ASSERT_EQUAL(13, ref1);
    ASSERT_EQUAL(ref0, ref1);

    // test different types
    typedef float T1;
    thrust::device_vector<T1> v1(1,0.0f);
    thrust::device_reference<T1> ref2 = v1[0];

    ref2 = ref1;

    // ref2 equals 13.0f
    ASSERT_EQUAL(13.0f, ref2);
    ASSERT_EQUAL(ref0, ref2);
    ASSERT_EQUAL(ref1, ref2);
}
DECLARE_UNITTEST(TestDeviceReferenceAssignmentFromDeviceReference);

void TestDeviceReferenceManipulation(void)
{
    typedef int T1;

    thrust::device_vector<T1> v(1,0);
    thrust::device_ptr<T1> ptr = &v[0];
    thrust::device_reference<T1> ref(ptr);

    // reset
    ref = 0;

    // test prefix increment
    ++ref;
    ASSERT_EQUAL(1, ref);
    ASSERT_EQUAL(1, *ptr);
    ASSERT_EQUAL(1, v[0]);

    // reset
    ref = 0;

    // test postfix increment
    T1 x1 = ref++;
    ASSERT_EQUAL(0, x1);
    ASSERT_EQUAL(1, ref);
    ASSERT_EQUAL(1, *ptr);
    ASSERT_EQUAL(1, v[0]);

    // reset
    ref = 0;

    // test addition-assignment
    ref += 5;
    ASSERT_EQUAL(5, ref);
    ASSERT_EQUAL(5, *ptr);
    ASSERT_EQUAL(5, v[0]);

    // reset
    ref = 0;

    // test prefix decrement
    --ref;
    ASSERT_EQUAL(-1, ref);
    ASSERT_EQUAL(-1, *ptr);
    ASSERT_EQUAL(-1, v[0]);

    // reset
    ref = 0;

    // test subtraction-assignment
    ref -= 5;
    ASSERT_EQUAL(-5, ref);
    ASSERT_EQUAL(-5, *ptr);
    ASSERT_EQUAL(-5, v[0]);

    // reset
    ref = 1;

    // test multiply-assignment
    ref *= 5;
    ASSERT_EQUAL(5, ref);
    ASSERT_EQUAL(5, *ptr);
    ASSERT_EQUAL(5, v[0]);

    // reset
    ref = 5;

    // test divide-assignment
    ref /= 5;
    ASSERT_EQUAL(1, ref);
    ASSERT_EQUAL(1, *ptr);
    ASSERT_EQUAL(1, v[0]);

    // reset
    ref = 5;

    // test modulus-assignment
    ref %= 5;
    ASSERT_EQUAL(0, ref);
    ASSERT_EQUAL(0, *ptr);
    ASSERT_EQUAL(0, v[0]);

    // reset
    ref = 1;

    // test left shift-assignment
    ref <<= 1;
    ASSERT_EQUAL(2, ref);
    ASSERT_EQUAL(2, *ptr);
    ASSERT_EQUAL(2, v[0]);

    // reset
    ref = 2;

    // test right shift-assignment
    ref >>= 1;
    ASSERT_EQUAL(1, ref);
    ASSERT_EQUAL(1, *ptr);
    ASSERT_EQUAL(1, v[0]);

    // reset
    ref = 0;

    // test OR-assignment
    ref |= 1;
    ASSERT_EQUAL(1, ref);
    ASSERT_EQUAL(1, *ptr);
    ASSERT_EQUAL(1, v[0]);

    // reset
    ref = 1;

    // test XOR-assignment
    ref ^= 1;
    ASSERT_EQUAL(0, ref);
    ASSERT_EQUAL(0, *ptr);
    ASSERT_EQUAL(0, v[0]);

    // test equality of const references
    thrust::device_reference<const T1> ref1 = v[0];
    ASSERT_EQUAL(true, ref1 == ref);
}
DECLARE_UNITTEST(TestDeviceReferenceManipulation);

void TestDeviceReferenceSwap(void)
{
  typedef int T;

  thrust::device_vector<T> v(2);
  thrust::device_reference<T> ref1 = v.front();
  thrust::device_reference<T> ref2 = v.back();

  ref1 = 7;
  ref2 = 13;

  // test thrust::swap()
  thrust::swap(ref1, ref2);
  ASSERT_EQUAL(13, ref1);
  ASSERT_EQUAL(7, ref2);

  // test .swap()
  ref1.swap(ref2);
  ASSERT_EQUAL(7, ref1);
  ASSERT_EQUAL(13, ref2);
}
DECLARE_UNITTEST(TestDeviceReferenceSwap);

