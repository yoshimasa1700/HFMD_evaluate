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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/masahiko/Program/objectPoseEstimation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/masahiko/Program/objectPoseEstimation

# Include any dependencies generated for this target.
include CMakeFiles/learning.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/learning.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/learning.dir/flags.make

CMakeFiles/learning.dir/learning.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/learning.cpp.o: learning.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/masahiko/Program/objectPoseEstimation/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/learning.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/learning.cpp.o -c /home/masahiko/Program/objectPoseEstimation/learning.cpp

CMakeFiles/learning.dir/learning.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/learning.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/masahiko/Program/objectPoseEstimation/learning.cpp > CMakeFiles/learning.dir/learning.cpp.i

CMakeFiles/learning.dir/learning.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/learning.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/masahiko/Program/objectPoseEstimation/learning.cpp -o CMakeFiles/learning.dir/learning.cpp.s

CMakeFiles/learning.dir/learning.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/learning.cpp.o.requires

CMakeFiles/learning.dir/learning.cpp.o.provides: CMakeFiles/learning.dir/learning.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/learning.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/learning.cpp.o.provides

CMakeFiles/learning.dir/learning.cpp.o.provides.build: CMakeFiles/learning.dir/learning.cpp.o

CMakeFiles/learning.dir/CRForest.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/CRForest.cpp.o: CRForest.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/masahiko/Program/objectPoseEstimation/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/CRForest.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/CRForest.cpp.o -c /home/masahiko/Program/objectPoseEstimation/CRForest.cpp

CMakeFiles/learning.dir/CRForest.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/CRForest.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/masahiko/Program/objectPoseEstimation/CRForest.cpp > CMakeFiles/learning.dir/CRForest.cpp.i

CMakeFiles/learning.dir/CRForest.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/CRForest.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/masahiko/Program/objectPoseEstimation/CRForest.cpp -o CMakeFiles/learning.dir/CRForest.cpp.s

CMakeFiles/learning.dir/CRForest.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/CRForest.cpp.o.requires

CMakeFiles/learning.dir/CRForest.cpp.o.provides: CMakeFiles/learning.dir/CRForest.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/CRForest.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/CRForest.cpp.o.provides

CMakeFiles/learning.dir/CRForest.cpp.o.provides.build: CMakeFiles/learning.dir/CRForest.cpp.o

CMakeFiles/learning.dir/util.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/util.cpp.o: util.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/masahiko/Program/objectPoseEstimation/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/util.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/util.cpp.o -c /home/masahiko/Program/objectPoseEstimation/util.cpp

CMakeFiles/learning.dir/util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/util.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/masahiko/Program/objectPoseEstimation/util.cpp > CMakeFiles/learning.dir/util.cpp.i

CMakeFiles/learning.dir/util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/util.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/masahiko/Program/objectPoseEstimation/util.cpp -o CMakeFiles/learning.dir/util.cpp.s

CMakeFiles/learning.dir/util.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/util.cpp.o.requires

CMakeFiles/learning.dir/util.cpp.o.provides: CMakeFiles/learning.dir/util.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/util.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/util.cpp.o.provides

CMakeFiles/learning.dir/util.cpp.o.provides.build: CMakeFiles/learning.dir/util.cpp.o

CMakeFiles/learning.dir/CRTree.cpp.o: CMakeFiles/learning.dir/flags.make
CMakeFiles/learning.dir/CRTree.cpp.o: CRTree.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/masahiko/Program/objectPoseEstimation/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/learning.dir/CRTree.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/learning.dir/CRTree.cpp.o -c /home/masahiko/Program/objectPoseEstimation/CRTree.cpp

CMakeFiles/learning.dir/CRTree.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/learning.dir/CRTree.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/masahiko/Program/objectPoseEstimation/CRTree.cpp > CMakeFiles/learning.dir/CRTree.cpp.i

CMakeFiles/learning.dir/CRTree.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/learning.dir/CRTree.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/masahiko/Program/objectPoseEstimation/CRTree.cpp -o CMakeFiles/learning.dir/CRTree.cpp.s

CMakeFiles/learning.dir/CRTree.cpp.o.requires:
.PHONY : CMakeFiles/learning.dir/CRTree.cpp.o.requires

CMakeFiles/learning.dir/CRTree.cpp.o.provides: CMakeFiles/learning.dir/CRTree.cpp.o.requires
	$(MAKE) -f CMakeFiles/learning.dir/build.make CMakeFiles/learning.dir/CRTree.cpp.o.provides.build
.PHONY : CMakeFiles/learning.dir/CRTree.cpp.o.provides

CMakeFiles/learning.dir/CRTree.cpp.o.provides.build: CMakeFiles/learning.dir/CRTree.cpp.o

# Object files for target learning
learning_OBJECTS = \
"CMakeFiles/learning.dir/learning.cpp.o" \
"CMakeFiles/learning.dir/CRForest.cpp.o" \
"CMakeFiles/learning.dir/util.cpp.o" \
"CMakeFiles/learning.dir/CRTree.cpp.o"

# External object files for target learning
learning_EXTERNAL_OBJECTS =

learning-1.0.0: CMakeFiles/learning.dir/learning.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/CRForest.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/util.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/CRTree.cpp.o
learning-1.0.0: CMakeFiles/learning.dir/build.make
learning-1.0.0: /usr/lib/libopencv_calib3d.so
learning-1.0.0: /usr/lib/libopencv_contrib.so
learning-1.0.0: /usr/lib/libopencv_core.so
learning-1.0.0: /usr/lib/libopencv_features2d.so
learning-1.0.0: /usr/lib/libopencv_flann.so
learning-1.0.0: /usr/lib/libopencv_gpu.so
learning-1.0.0: /usr/lib/libopencv_highgui.so
learning-1.0.0: /usr/lib/libopencv_imgproc.so
learning-1.0.0: /usr/lib/libopencv_legacy.so
learning-1.0.0: /usr/lib/libopencv_ml.so
learning-1.0.0: /usr/lib/libopencv_nonfree.so
learning-1.0.0: /usr/lib/libopencv_objdetect.so
learning-1.0.0: /usr/lib/libopencv_ocl.so
learning-1.0.0: /usr/lib/libopencv_photo.so
learning-1.0.0: /usr/lib/libopencv_stitching.so
learning-1.0.0: /usr/lib/libopencv_ts.so
learning-1.0.0: /usr/lib/libopencv_video.so
learning-1.0.0: /usr/lib/libopencv_videostab.so
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
.PHONY : CMakeFiles/learning.dir/requires

CMakeFiles/learning.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/learning.dir/cmake_clean.cmake
.PHONY : CMakeFiles/learning.dir/clean

CMakeFiles/learning.dir/depend:
	cd /home/masahiko/Program/objectPoseEstimation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/masahiko/Program/objectPoseEstimation /home/masahiko/Program/objectPoseEstimation /home/masahiko/Program/objectPoseEstimation /home/masahiko/Program/objectPoseEstimation /home/masahiko/Program/objectPoseEstimation/CMakeFiles/learning.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/learning.dir/depend

