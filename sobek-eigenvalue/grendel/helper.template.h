#ifndef HELPER_TEMPLATE_H
#define HELPER_TEMPLATE_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>

namespace grendel
{
  template <class DH>
  void pin_a_dof(const DH &dof_handler,
                 dealii::AffineConstraints<double> &affine_constraints,
                 unsigned int component = 0)
  {
    Assert(dof_handler.get_triangulation().n_levels() >= 2,
           dealii::ExcNotImplemented());

    for (auto it = dof_handler.begin_active(); it != dof_handler.end(); ++it) {

      if (it->at_boundary() == false) {

        std::vector<unsigned int> local_dof_indices(it->get_fe().dofs_per_cell);
        it->get_dof_indices(local_dof_indices);
        Assert(local_dof_indices.size() != 0, dealii::ExcInternalError());

        const unsigned int dof =
            it->get_fe().component_to_system_index(component, 0);
        affine_constraints.add_line(local_dof_indices[dof]);
        affine_constraints.set_inhomogeneity(local_dof_indices[dof], 0.);

        return;
      }
    }
    AssertThrow(false, dealii::ExcInternalError());

    return;
  }


  typedef double value_type;


  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_curl(const dealii::Tensor<1, dim, Number> &tensor,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    return cross_product_3d(normal, tensor);
  }


  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_curl(const dealii::Tensor<1, 1, Number> &cross,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    return cross_product_2d(normal) * cross[0];
  }


  template <int dim, typename Number>
  dealii::Tensor<1, dim, Number>
  tangential_part(const dealii::Tensor<1, dim, Number> &tensor,
                  const dealii::Tensor<1, dim, Number> &normal)
  {
    dealii::Tensor<1, dim, Number> result;
    switch (dim) {
    case 2:
      result[0] = normal[1] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
      result[1] = -normal[0] * (tensor[0] * normal[1] - tensor[1] * normal[0]);
      break;
    case 3:
      result = cross_product_3d(cross_product_3d(normal, tensor), normal);
      break;
    }
    return result;
  }


} /* namespace grendel */

#endif /* HELPER_TEMPLATE_H */
