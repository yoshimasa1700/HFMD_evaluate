# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/2.8.10.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/2.8.10.2/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/local/Cellar/cmake/2.8.10.2/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/masahiko/Programs/HFMD_evaluate

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/masahiko/Programs/HFMD_evaluate

# Include any dependencies generated for this target.
include CMakeFiles/learning.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/learning.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/learning.dir/flags.make

CMakeFiles/learning.dir/learning.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/learning.cpp.o: learning.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/learning.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/learning.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/learning.cpp

CMakeFiles/learning.dir/learning.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/learning.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/learning.cpp > CMakeFiles/learning.dir/learning.cpp.i

CMakeFiles/learning.dir/learning.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/learning.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/learning.cpp -o CMakeFiles/learning.dir/learning.cpp.s

CMakeFiles/learning.dir/learning.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/learning.cpp.o.requires

CMakeFiles/learning.dir/learning.cpp.o.provides: CMakeFiles/learning.dir/learning.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/learning.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/learning.cpp.o.provides

CMakeFiles/learning.dir/learning.cpp.o.provides.build: CMakeFiles/learning.dir/learning.cpp.o

CMakeFiles/learning.dir/CRForest.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/CRForest.cpp.o: CRForest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/CRForest.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/CRForest.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp

CMakeFiles/learning.dir/CRForest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/CRForest.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp > CMakeFiles/learning.dir/CRForest.cpp.i

CMakeFiles/learning.dir/CRForest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/CRForest.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/CRForest.cpp -o CMakeFiles/learning.dir/CRForest.cpp.s

CMakeFiles/learning.dir/CRForest.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/CRForest.cpp.o.requires

CMakeFiles/learning.dir/CRForest.cpp.o.provides: CMakeFiles/learning.dir/CRForest.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/CRForest.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/CRForest.cpp.o.provides

CMakeFiles/learning.dir/CRForest.cpp.o.provides.build: CMakeFiles/learning.dir/CRForest.cpp.o

CMakeFiles/learning.dir/util.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/util.cpp.o: util.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/util.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/util.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/util.cpp

CMakeFiles/learning.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/util.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/util.cpp > CMakeFiles/learning.dir/util.cpp.i

CMakeFiles/learning.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/util.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/util.cpp -o CMakeFiles/learning.dir/util.cpp.s

CMakeFiles/learning.dir/util.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/util.cpp.o.requires

CMakeFiles/learning.dir/util.cpp.o.provides: CMakeFiles/learning.dir/util.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/util.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/util.cpp.o.provides

CMakeFiles/learning.dir/util.cpp.o.provides.build: CMakeFiles/learning.dir/util.cpp.o

CMakeFiles/learning.dir/CRTree.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/CRTree.cpp.o: CRTree.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/CRTree.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/CRTree.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp

CMakeFiles/learning.dir/CRTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/CRTree.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp > CMakeFiles/learning.dir/CRTree.cpp.i

CMakeFiles/learning.dir/CRTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/CRTree.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/CRTree.cpp -o CMakeFiles/learning.dir/CRTree.cpp.s

CMakeFiles/learning.dir/CRTree.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/CRTree.cpp.o.requires

CMakeFiles/learning.dir/CRTree.cpp.o.provides: CMakeFiles/learning.dir/CRTree.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/CRTree.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/CRTree.cpp.o.provides

CMakeFiles/learning.dir/CRTree.cpp.o.provides.build: CMakeFiles/learning.dir/CRTree.cpp.o

CMakeFiles/learning.dir/HoG.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/HoG.cpp.o: HoG.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/HoG.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/HoG.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp

CMakeFiles/learning.dir/HoG.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/HoG.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp > CMakeFiles/learning.dir/HoG.cpp.i

CMakeFiles/learning.dir/HoG.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/HoG.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/HoG.cpp -o CMakeFiles/learning.dir/HoG.cpp.s

CMakeFiles/learning.dir/HoG.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/HoG.cpp.o.requires

CMakeFiles/learning.dir/HoG.cpp.o.provides: CMakeFiles/learning.dir/HoG.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/HoG.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/HoG.cpp.o.provides

CMakeFiles/learning.dir/HoG.cpp.o.provides.build: CMakeFiles/learning.dir/HoG.cpp.o

CMakeFiles/learning.dir/detectionresult.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/detectionresult.cpp.o: detectionresult.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles $(CMAKE_PROGRESS_6)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/detectionresult.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/detectionresult.cpp.o -c /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp

CMakeFiles/learning.dir/detectionresult.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/detectionresult.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp > CMakeFiles/learning.dir/detectionresult.cpp.i

CMakeFiles/learning.dir/detectionresult.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/detectionresult.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /Users/masahiko/Programs/HFMD_evaluate/detectionresult.cpp -o CMakeFiles/learning.dir/detectionresult.cpp.s

CMakeFiles/learning.dir/detectionresult.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/detectionresult.cpp.o.requires

CMakeFiles/learning.dir/detectionresult.cpp.o.provides: CMakeFiles/learning.dir/detectionresult.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/detectionresult.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/detectionresult.cpp.o.provides

CMakeFiles/learning.dir/detectionresult.cpp.o.provides.build: CMakeFiles/learning.dir/detectionresult.cpp.o

# Object files for target learning
learning_OBJECTS = \
"CMakeFiles/learning.dir/learning.cpp.o" \
"CMakeFiles/learning.dir/CRForest.cpp.o" \
"CMakeFiles/learning.dir/util.cpp.o" \
"CMakeFiles/learning.dir/CRTree.cpp.o" \
"CMakeFiles/learning.dir/HoG.cpp.o" \
"CMakeFiles/learning.dir/detectionresult.cpp.o"

# External object files for target learning
learning_EXTERNAL_OBJECTS =

learning-1.0.0: CMakeFiles/learning.dir/learning.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/CRForest.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/util.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/CRTree.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/HoG.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/detectionresult.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/build.make
learning-1.0.0: /usr/local/lib/libopencv_calib3d.dylib
learning-1.0.0: /usr/local/lib/libopencv_contrib.dylib
learning-1.0.0: /usr/local/lib/libopencv_core.dylib
learning-1.0.0: /usr/local/lib/libopencv_features2d.dylib
learning-1.0.0: /usr/local/lib/libopencv_flann.dylib
learning-1.0.0: /usr/local/lib/libopencv_gpu.dylib
learning-1.0.0: /usr/local/lib/libopencv_highgui.dylib
learning-1.0.0: /usr/local/lib/libopencv_imgproc.dylib
learning-1.0.0: /usr/local/lib/libopencv_legacy.dylib
learning-1.0.0: /usr/local/lib/libopencv_ml.dylib
learning-1.0.0: /usr/local/lib/libopencv_nonfree.dylib
learning-1.0.0: /usr/local/lib/libopencv_objdetect.dylib
learning-1.0.0: /usr/local/lib/libopencv_ocl.dylib
learning-1.0.0: /usr/local/lib/libopencv_photo.dylib
learning-1.0.0: /usr/local/lib/libopencv_stitching.dylib
learning-1.0.0: /usr/local/lib/libopencv_superres.dylib
learning-1.0.0: /usr/local/lib/libopencv_ts.dylib
learning-1.0.0: /usr/local/lib/libopencv_video.dylib
learning-1.0.0: /usr/local/lib/libopencv_videostab.dylib
learning-1.0.0: CMakeFiles/learning.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable learning"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/learning.dir/link.txt --verbose=$(VERBOSE)
	$(CMAKE_COMMAND) -E cmake_symlink_executable learning-1.0.0 learning

learning: learning-1.0.0

# Rule to build all files generated by this target.
CMakeFiles/learning.dir/build: learning
.PHONY : CMakeFiles/learning.dir/build

CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/learning.cpp.o.requires
CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/CRForest.cpp.o.requires
CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/util.cpp.o.requires
CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/CRTree.cpp.o.requires
CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/HoG.cpp.o.requires
CMakeFiles/learning.dir/requires: CMakeFiles/learning.dir/detectionresult.cpp.o.requires
.PHONY : CMakeFiles/learning.dir/requires

CMakeFiles/learning.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/learning.dir/cmake_clean.cmake
.PHONY : CMakeFiles/learning.dir/clean

CMakeFiles/learning.dir/depend:
	cd /Users/masahiko/Programs/HFMD_evaluate && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate /Users/masahiko/Programs/HFMD_evaluate/CMakeFiles/learning.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/learning.dir/depend

