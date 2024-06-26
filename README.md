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
Siyuan Hong: 

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
      
## Coverage improvement

### Individual tests
Siyuan Hong

test1:

a link to commit:
     https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy
	 
old results: 
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/5d6d3622-1eed-482c-a810-3847174bed22)

	 
new results :
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/751b8bfe-1a7c-46d7-b379-a9f0ca1c455a)
	 
     the coverage rate improved by 100%.
		 
     comment: because previously, this function is not covered by any tese case, so i just write a new test case to set one node with a rotation degree and see how does that function works.
     the condition branches include node with a parent node and without parent node, so i just write a parent node and childe node attached to it, and call set_global_rotation separately,
     then all the three branches are reached. 
    
test2:

a link to commit: 
     https://github.com/godotengine/godot/compare/master...SiyuanHong:godot:hsy
	 
old results: 
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/7ff70bbb-2e1f-4541-a038-69965651f022)
	 
new results :
     ![image](https://github.com/SiyuanHong/godot/assets/113177812/751b8bfe-1a7c-46d7-b379-a9f0ca1c455a)
	 
     the coverage rate improved by 50%.
		 
     comment: because these two functions are highly correlated, so i merged the two test cases into just one, but it indeed tested two functions.
     because the system is complicated, and the arrtibute of "dirty" is protected, hard to figure out how to make its value as dirty, so this test case only
     cover the first branch.
  
  ### Overall

  
  ## Statement of individual contributions
Siyuan Hong: write methods for function instrumentation; deal with function set_global_rotation and get_rotation
     
