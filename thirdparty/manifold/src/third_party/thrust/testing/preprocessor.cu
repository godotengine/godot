#include <unittest/unittest.h>
#include <string>
#include <thrust/detail/preprocessor.h>

void test_pp_stringize()
{
  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(int))
  , "int"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(hello world))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(hello  world))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE( hello  world))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(hello  world ))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE( hello  world ))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(hello
                                    world))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE("hello world"))
  , "\"hello world\""
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE('hello world'))
  , "'hello world'"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE($%!&<->))
  , "$%!&<->"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE($%!&""<->))
  , "$%!&\"\"<->"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_STRINGIZE))
  , "THRUST_PP_STRINGIZE"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_STRINGIZE(int)))
  , "\"int\""
  );
}
DECLARE_UNITTEST(test_pp_stringize);

void test_pp_cat2()
{
  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(i, nt)))
  , "int"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello, world)))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello , world)))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2( hello, world)))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello,  world)))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello, world )))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello,
                                                   world )))
  , "helloworld"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(hello world, from thrust!)))
  , "hello worldfrom thrust!"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_CAT2(-, >)))
  , "->"
  );
}
DECLARE_UNITTEST(test_pp_cat2);

#define THRUST_TEST_PP_EXPAND_TARGET() success

#define THRUST_TEST_PP_EXPAND_ARGS() ()

void test_pp_expand()
{
  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(int)))
  , "int"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(hello world)))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(hello  world)))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND( hello  world)))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(hello  world )))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND( hello  world )))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(hello
                                    world)))
  , "hello world"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND("hello world")))
  , "\"hello world\""
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND('hello world')))
  , "'hello world'"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND($%!&<->)))
  , "$%!&<->"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND($%!&""<->)))
  , "$%!&\"\"<->"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(THRUST_PP_EXPAND)))
  , "THRUST_PP_EXPAND"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(THRUST_PP_EXPAND(int))))
  , "int"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(
      THRUST_PP_CAT2(THRUST_TEST_, PP_EXPAND_TARGET)()
    )))
  , "success"
  );

  ASSERT_EQUAL(
    std::string(THRUST_PP_STRINGIZE(THRUST_PP_EXPAND(
      THRUST_TEST_PP_EXPAND_TARGET THRUST_TEST_PP_EXPAND_ARGS()
    )))
  , "success"
  );
}
DECLARE_UNITTEST(test_pp_expand);

#undef THRUST_TEST_PP_EXPAND_TARGET

#undef THRUST_TEST_PP_EXPAND_ARGS

void test_pp_arity()
{
  ASSERT_EQUAL(
    THRUST_PP_ARITY()
  , 0
  );

  /* This bash script was used to generate these tests:

    for arity in {0..62}
    do
      echo "  ASSERT_EQUAL("
      echo "    THRUST_PP_ARITY("
      echo "      `bash -c \"echo {0..${arity}} | tr ' ' ,\"`"
      echo "    )"
      echo "  , $((${arity} + 1))"
      echo "  );"
      echo
    done
  */

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0
    )
  , 1
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1
    )
  , 2
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2
    )
  , 3
  );
ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3
    )
  , 4
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4
    )
  , 5
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5
    )
  , 6
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6
    )
  , 7
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7
    )
  , 8
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8
    )
  , 9
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9
    )
  , 10
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10
    )
  , 11
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11
    )
  , 12
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12
    )
  , 13
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13
    )
  , 14
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
    )
  , 15
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    )
  , 16
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
    )
  , 17
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17
    )
  , 18
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
    )
  , 19
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19
    )
  , 20
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    )
  , 21
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
    )
  , 22
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
    )
  , 23
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
    )
  , 24
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
    )
  , 25
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25
    )
  , 26
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26
    )
  , 27
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27
    )
  , 28
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28
    )
  , 29
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29
    )
  , 30
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30
    )
  , 31
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    )
  , 32
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
    )
  , 33
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
    )
  , 34
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34
    )
  , 35
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35
    )
  , 36
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36
    )
  , 37
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
    )
  , 38
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
    )
  , 39
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39
    )
  , 40
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40
    )
  , 41
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41
    )
  , 42
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42
    )
  , 43
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43
    )
  , 44
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44
    )
  , 45
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45
    )
  , 46
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46
    )
  , 47
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47
    )
  , 48
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48
    )
  , 49
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49
    )
  , 50
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50
    )
  , 51
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51
    )
  , 52
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52
    )
  , 53
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53
    )
  , 54
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
    )
  , 55
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55
    )
  , 56
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56
    )
  , 57
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57
    )
  , 58
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58
    )
  , 59
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59
    )
  , 60
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60
    )
  , 61
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
    )
  , 62
  );

  ASSERT_EQUAL(
    THRUST_PP_ARITY(
      0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62
    )
  , 63
  );
}
DECLARE_UNITTEST(test_pp_arity);

#define THRUST_TEST_PP_DISPATCH_PLUS(...)                                     \
  THRUST_PP_DISPATCH(THRUST_TEST_PP_DISPATCH_PLUS, __VA_ARGS__)               \
  /**/
#define THRUST_TEST_PP_DISPATCH_PLUS0()        0
#define THRUST_TEST_PP_DISPATCH_PLUS1(x)       x
#define THRUST_TEST_PP_DISPATCH_PLUS2(x, y)    x + y
#define THRUST_TEST_PP_DISPATCH_PLUS3(x, y, z) x + y + z

void test_pp_dispatch()
{
  ASSERT_EQUAL(
    THRUST_TEST_PP_DISPATCH_PLUS()
  , 0
  );

  ASSERT_EQUAL(
    THRUST_TEST_PP_DISPATCH_PLUS(0)
  , 0
  );

  ASSERT_EQUAL(
    THRUST_TEST_PP_DISPATCH_PLUS(1, 2)
  , 3
  );

  ASSERT_EQUAL(
    THRUST_TEST_PP_DISPATCH_PLUS(1, 2, 3)
  , 6
  );
}
DECLARE_UNITTEST(test_pp_dispatch);

#undef THRUST_TEST_PP_DISPATCH_PLUS
#undef THRUST_TEST_PP_DISPATCH_PLUS0
#undef THRUST_TEST_PP_DISPATCH_PLUS1
#undef THRUST_TEST_PP_DISPATCH_PLUS2
#undef THRUST_TEST_PP_DISPATCH_PLUS3

