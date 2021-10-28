#ifndef OUTPUT_HELPER_TEMPLATE_H
#define OUTPUT_HELPER_TEMPLATE_H

#include "output_helper.h"
#include "to_vector_function.template.h"

#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace grendel
{

  template <int dim>
  void output(const CellProblem<dim> &cell_problem, const std::string &name)
  {
    dealii::DataOut<dim> data_out;
    const auto &dof_handler = cell_problem.dof_handler();
    data_out.attach_dof_handler(dof_handler);

    for (unsigned int i = 0; i < cell_problem.eigenfunctions().size(); ++i) {

      std::string label = "re_eigenvector_";
      if (i < 10)
        label += "0";
      label += std::to_string(i);

      data_out.add_data_vector(cell_problem.eigenfunctions()[i], label,
                               dealii::DataOut<dim>::type_dof_data);
    }

    const auto &triangulation = dof_handler.get_triangulation();
    dealii::Vector<double> material_ids(triangulation.n_active_cells());

    for (const auto &cell : triangulation.active_cell_iterators())
      material_ids[cell->active_cell_index()] = cell->material_id();
    data_out.add_data_vector(material_ids, "material_ids");

    const auto &discretization = cell_problem.discretization();

    data_out.build_patches(discretization.mapping());
    std::ofstream output(name + ".vtk");
    data_out.write_vtk(output);
  }

} /* namespace grendel */

#endif /* OUTPUT_HELPER_TEMPLATE_H */
