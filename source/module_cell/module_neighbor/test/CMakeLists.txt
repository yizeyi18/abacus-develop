remove_definitions(-D__DEEPKS)
remove_definitions(-D__CUDA)
#include "module_/.h"
remove_definitions(-D__ROCM)
remove_definitions(-D__EXX)

AddTest(
  TARGET cell_neighbor_sltk_atom
  SOURCES sltk_atom_test.cpp ../sltk_atom.cpp
)

AddTest(
  TARGET cell_neighbor_sltk_grid
  LIBS parameter ${math_libs} base device cell_info 
  SOURCES sltk_grid_test.cpp ../sltk_grid.cpp ../sltk_atom.cpp
    ../../../module_io/output.cpp
)

AddTest(
  TARGET cell_neighbor_sltk_atom_arrange
  LIBS parameter ${math_libs} base device cell_info 
  SOURCES sltk_atom_arrange_test.cpp  ../sltk_atom_arrange.cpp ../sltk_grid_driver.cpp ../sltk_grid.cpp
  ../sltk_atom.cpp
  ../../../module_io/output.cpp
)