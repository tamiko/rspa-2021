ADD_LIBRARY(grendel SHARED
  cell_problem.cc
  discretization.cc
  output_helper.cc
  )
DEAL_II_SETUP_TARGET(grendel)
