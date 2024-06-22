# Report for Assignment 1

## Project chosen
Name: Godot Engine

URL: https://github.com/godotengine/godot.git

Number of lines of code and the tool used to count it: 2326 KLOC, tested by lizard with command `lizard -l cpp`

Programming language: C++
## Coverage measurement

### Existing tool
name: OpenCppCoverage

The way it was executed: After the development version of Godot compile( enabling it to be tested and be with debugged symbols),
run OpenCppCoverage withe command line like this: OpenCppCoverage --  ./bin/godot.windows.editor.dev.x86_64.exe --test
the execuatable will run with all the tests and the OpenCppCoverage will check which part of the code is covered.

Because it is a really big project(it contains millions of lines of code), so we focus on the one part of the project which is the scene part, so we specificlly measure the coverage
of that part with OpenCppCoverage.

![image](https://github.com/SiyuanHong/godot/assets/113177812/9202d40c-1cda-4ff4-b845-fa7c9eaf8603)

### Your own coverage tool
Siyuan Hong

  Function 1

  name: set_global_rotation

  a link to the commit: https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy

  screenshot:

i identified three branches in this function(branch 0, 1,2):

![image](https://github.com/SiyuanHong/godot/assets/113177812/7ff70bbb-2e1f-4541-a038-69965651f022)

 here you can see the first three branches are not reached in the original tests.

  Function 2

  name:get_rotation

  a link to the commit: https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy

  screenshot:

 i identified two branches in this function(branch 3,4):

![image](https://github.com/SiyuanHong/godot/assets/113177812/7ff70bbb-2e1f-4541-a038-69965651f022)

they are not reached in the original test.

Ruizhe Tao

Function 1

name: set_global_skew

a link to the commit: https://github.com/SiyuanHong/godot/commit/069884f925777869f8bf04b8f5257e045245dfa0

screenshot:

2 branches with unique ids are identified in this function:

![skew_uncovered](https://github.com/SiyuanHong/godot/assets/50838626/ba783274-4cb1-4709-b297-b55f52c24516)

Function 2

name: set_global_scale

a link to the commit: https://github.com/SiyuanHong/godot/commit/069884f925777869f8bf04b8f5257e045245dfa0

screenshot:

2 branches with unique ids are identified in this function:

![scale_uncovered](https://github.com/SiyuanHong/godot/assets/50838626/16a5f470-637e-423c-aed3-433ac9b14c95)

Jiarui Pan

Function 1

Name: get_relative_transform_to_parent (https://github.com/SiyuanHong/godot/commit/8f3368781a76d6c3722a1ea3a8791a623f629722)

Screenshot:

![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/before%20test1.png)

Four branches are identified as 0, 1, 2 and 3, contained by coverageDataOfPjrs. At this stage none of these branches are reached by the exisiting tests.

Function 2

Name: set_current (https://github.com/SiyuanHong/godot/commit/df84fe5b13f9e87a36027155b0462e0ad4b5f1cd)

Screenshot:

![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/before%20test2.png)

Three branches are identified as 0, 1 and 2. To seperate, container is differently named as coverageDataOfPjrs2. At this stage none of these branches are reached by the exisiting tests.

## Coverage improvement

### Individual tests
Siyuan Hong

test1:

a link to the commit:
     https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy

old result:
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/5d6d3622-1eed-482c-a810-3847174bed22)


new result:
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/751b8bfe-1a7c-46d7-b379-a9f0ca1c455a)

     the coverage improved by 100%.

     comment: because previously, this function is not covered by any tese case, so i just write a new test case to set one node with a rotation degree and see how does that function works.
     the condition branches include node with a parent node and without parent node, so i just write a parent node and childe node attached to it, and call set_global_rotation separately,
     then all the three branches are reached.

test2:

a link to the commit:
     https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy

old result:
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/7ff70bbb-2e1f-4541-a038-69965651f022)

new result:
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/751b8bfe-1a7c-46d7-b379-a9f0ca1c455a)

     the coverage improved by 50%.

     comment: because these two functions are highly correlated, so i merged the two test cases into just one, but it indeed tested two functions.
     because the system is complicated, and the arrtibute of "dirty" is protected, hard to figure out how to make its value as dirty, so this test case only
     cover the first branch.

Ruizhe Tao

Test 1

a link to the commit: https://github.com/SiyuanHong/godot/commit/069884f925777869f8bf04b8f5257e045245dfa0

old result: \
![skew_uncovered](https://github.com/SiyuanHong/godot/assets/50838626/f43d6649-cc66-4220-b24e-f156599bae2f)

new result: \
![skew_covered](https://github.com/SiyuanHong/godot/assets/50838626/1246d93a-3152-492f-9b5c-6e02c5ce0a49)

	The coverage improved by 100%.
 	Since this function is not tested in the original project, a new test dedicated to this function was made.
  	By creating two nodes, parent node and child node, and assign the child node as the child of the parent node.
   	Use these two node objects to invoke `set_global_skew` function, both branches will be reached, as the condition
	checks if the node has a parent node.

Test 2

a link to the commit: https://github.com/SiyuanHong/godot/commit/069884f925777869f8bf04b8f5257e045245dfa0

old result: \
![scale_uncovered](https://github.com/SiyuanHong/godot/assets/50838626/7fa8a1b2-9d23-432d-8418-6e8a94b9f457)

new result: \
![scale_covered](https://github.com/SiyuanHong/godot/assets/50838626/db7ec6c7-885c-4d2c-8153-be25b8cee24b)

	The coverage improved by 100%.
 	This function is very similar to the previous one, and it is also not tested in the original project. A new test
  	dedicated to this function was made.
  	By creating two nodes, parent node and child node, and assign the child node as the child of the parent node.
   	Use these two node objects to invoke `set_global_scale` function, both branches will be reached, as the condition
	checks if the node has a parent node.

Jiarui Pan

Test1

a link to the commit: https://github.com/SiyuanHong/godot/commit/8f3368781a76d6c3722a1ea3a8791a623f629722

old results:

![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/before%20test1.png)


new result:
![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/after%20test1.png)

     The coverage improved by 100%.

     Comment: Initially, no tests are responsible for this function, so a new testcase is created. According to the code, three cases diverge, thus, the testcase consists of three subcases: p_parent == this, p_parent == parent_2d and the rest. For each condition, content of the tests differs: for the first one, check if the transform by get_relative_transform_to_parent(node) returns the same as Transform2D() when p_parent is the node itself; for the second one, by linking the node to its parent node, check if the transform of the parent node is the same as child node; similarly, for the third one, by linking the node to its parent node and its grandparent node, check if the total transform of the node and its parent node is the same as the grandparent node. By do these, all branches are reached by the tests and coverage is improved.

Test2

a link to the commit: https://github.com/SiyuanHong/godot/commit/df84fe5b13f9e87a36027155b0462e0ad4b5f1cd

old results:

![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/before%20test2.png)


new result:
![image](https://github.com/SiyuanHong/godot/blob/pjr/screenshots/after%20test2.png)

     The coverage improved by 100%.

     Comment: Initially, no tests are responsible for this function, so a new testcase is created. According to the code, two cases diverge, thus, the testcase consists of two subcases: when p_enabled and else. For the first condition, when set_current is true, the test checks if the function together with make_current are called while clear_current is not; likewise, the second subcase checks if the function together with clear_current are called while make_current is not. By do these, all branches are reached by the tests and coverage is improved.


### Overall


## Statement of individual contributions
Siyuan Hong: write methods for function instrumentation; deal with function set_global_rotation and get_rotation

Ruizhe Tao: write function instrumentation for `set_global_skew` and `set_global_scale`, and implement tests for these two functions to cover all branches

Jiarui Pan: completed implementation of tests for get_relative_transform_to_parent and set_current with the coverage measurement and improvement
