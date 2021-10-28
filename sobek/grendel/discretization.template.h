#ifndef DISCRETIZATION_TEMPLATE_H
#define DISCRETIZATION_TEMPLATE_H

#include "discretization.h"

#include "to_vector_function.template.h"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_nedelec.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;

  template <int dim>
  Discretization<dim>::Discretization(const std::string &subsection)
      : dealii::ParameterAcceptor(subsection)
  {
    ParameterAcceptor::parse_parameters_call_back.connect(
        std::bind(&Discretization::parse_parameters_callback, this));

    geometry_ = "circle";
    add_parameter(
        "geometry", geometry_,
        "Geometry to use, valid options are \"circle\", and \"ribbons\"",
        ParameterAcceptor::prm, Patterns::Selection("circle|ribbons"));

    radius_ = 0.4;
    add_parameter("radius", radius_, "Radius of inner circle");

    refinement_ = 7;
    add_parameter("initial refinement",
                  refinement_,
                  "Initial refinement of the geometry");

    order_mapping_ = 1;
    add_parameter("order mapping", order_mapping_, "Order of the mapping");

    order_finite_element_ = 1;
    add_parameter("order finite element",
                  order_finite_element_,
                  "Order of the finite element space");

    order_quadrature_ = 3;
    add_parameter(
        "order quadrature", order_quadrature_, "Order of the quadrature rule");
  }


  template <int dim>
  void Discretization<dim>::parse_parameters_callback()
  {
    /*
     * Create the Triangulation. We have $\Omega = [0,1]^dim$. Colorize
     * boundaries for periodic boundary conditions and mark material IDs
     * accordingly.
     */

    if (!triangulation_)
      triangulation_.reset(new Triangulation<dim>);

    auto &triangulation = *triangulation_;
    triangulation.clear();

    if (geometry_ == "circle") {
      const auto center = Point<dim>(0.5, 0.5);

      AssertThrow(radius_ < sqrt(0.5), ExcMessage("Ups, radius too large"));

      Triangulation<dim> tria1;
      GridGenerator::hyper_ball(tria1, center, radius_, true);

      const std::vector<Point<dim>> vertices{
          {0.0, 0.0},                                             // 0
          {0.0, 0.5 - sqrt(0.5) * radius_},                       // 1
          {0.0, 0.5 + sqrt(0.5) * radius_},                       // 2
          {0.0, 1.0},                                             // 3

          {0.5 - sqrt(0.5) * radius_, 0.0},                       // 4
          {0.5 - sqrt(0.5) * radius_, 0.5 - sqrt(0.5) * radius_}, // 5
          {0.5 - sqrt(0.5) * radius_, 0.5 + sqrt(0.5) * radius_}, // 6
          {0.5 - sqrt(0.5) * radius_, 1.0},                       // 7

          {0.5 + sqrt(0.5) * radius_, 0.0},                       // 8
          {0.5 + sqrt(0.5) * radius_, 0.5 - sqrt(0.5) * radius_}, // 9
          {0.5 + sqrt(0.5) * radius_, 0.5 + sqrt(0.5) * radius_}, // 10
          {0.5 + sqrt(0.5) * radius_, 1.0},                       // 11

          {1.0, 0.0},                                             // 12
          {1.0, 0.5 - sqrt(0.5) * radius_},                       // 13
          {1.0, 0.5 + sqrt(0.5) * radius_},                       // 14
          {1.0, 1.0},                                             // 15
      };

      std::vector<CellData<dim>> cells(8);
      cells[0].vertices = {0, 4, 1, 5};
      cells[1].vertices = {1, 5, 2, 6};
      cells[2].vertices = {2, 6, 3, 7};
      cells[3].vertices = {4, 8, 5, 9};
      cells[4].vertices = {6, 10, 7, 11};
      cells[5].vertices = {8, 12, 9, 13};
      cells[6].vertices = {9, 13, 10, 14};
      cells[7].vertices = {10, 14, 11, 15};

      Triangulation<dim> tria2;
      tria2.create_triangulation(vertices, cells, SubCellData());

      GridGenerator::merge_triangulations(tria1, tria2, triangulation);

      for (auto face : triangulation.active_face_iterators()) {
        const auto center = face->center();
        constexpr double eps = 1.0e-6;
        if (center[0] < eps) {
          face->set_boundary_id(0);
        } else if (center[0] > 1.0 - eps) {
          face->set_boundary_id(2);
        } else if (center[1] < eps) {
          face->set_boundary_id(1);
        } else if (center[1] > 1.0 - eps) {
          face->set_boundary_id(3);
        }
      }

      triangulation.set_all_manifold_ids(1);
      triangulation.set_all_manifold_ids_on_boundary(0);

      for (auto cell : triangulation.active_cell_iterators()) {
        const auto distance = (cell->center() - center).norm();

        if (distance < radius_ / sqrt(2.))
          cell->set_material_id(2);
        else
          cell->set_material_id(1);

        if (distance < 1.0e-6)
          cell->set_manifold_id(numbers::flat_manifold_id);
      }

      triangulation.set_manifold(1, SphericalManifold<dim>(center));

      GridTools::transform(
          [](const auto point) {
            dealii::Point<dim> new_point = point;
            new_point[0] *= 1.000001;
            return new_point;
          },
          triangulation);

      triangulation.refine_global(refinement_);

    } else if (geometry_ == "ribbons") {

      AssertThrow(radius_ < sqrt(0.5), ExcMessage("Ups, radius too large"));

      const std::vector<Point<dim>> vertices{
          {0.00, 0.00},           // 0
          {0.50 - radius_, 0.00}, // 1
          {0.50, 0.00},           // 2
          {0.50 + radius_, 0.00}, // 3
          {1.00, 0.00},           // 4
          {0.00, 0.25},           // 5
          {0.50 - radius_, 0.25}, // 6
          {0.50, 0.25},           // 7
          {0.50 + radius_, 0.25}, // 8
          {1.00, 0.25},           // 9
          {0.00, 0.50},           // 10
          {0.50 - radius_, 0.50}, // 11
          {0.50, 0.50},           // 12
          {0.50 + radius_, 0.50}, // 13
          {1.00, 0.50},           // 14
          {0.00, 0.75},           // 15
          {0.50 - radius_, 0.75}, // 16
          {0.50, 0.75},           // 17
          {0.50 + radius_, 0.75}, // 18
          {1.00, 0.75},           // 19
          {0.00, 1.00},           // 20
          {0.50 - radius_, 1.00}, // 21
          {0.50, 1.00},           // 22
          {0.50 + radius_, 1.00}, // 23
          {1.00, 1.00},           // 24
      };

      std::vector<CellData<dim>> cells(16);
      cells[0].vertices = {0, 1, 5, 6};
      cells[1].vertices = {1, 2, 6, 7};
      cells[2].vertices = {2, 3, 7, 8};
      cells[3].vertices = {3, 4, 8, 9};
      cells[4].vertices = {5, 6, 10, 11};
      cells[5].vertices = {6, 7, 11, 12};
      cells[6].vertices = {7, 8, 12, 13};
      cells[7].vertices = {8, 9, 13, 14};
      cells[8].vertices = {10, 11, 15, 16};
      cells[9].vertices = {11, 12, 16, 17};
      cells[10].vertices = {12, 13, 17, 18};
      cells[11].vertices = {13, 14, 18, 19};
      cells[12].vertices = {15, 16, 20, 21};
      cells[13].vertices = {16, 17, 21, 22};
      cells[14].vertices = {17, 18, 22, 23};
      cells[15].vertices = {18, 19, 23, 24};

      triangulation.create_triangulation(vertices, cells, SubCellData());

      for (auto face : triangulation.active_face_iterators()) {
        const auto center = face->center();
        constexpr double eps = 1.0e-6;
        if (center[0] < eps) {
          face->set_boundary_id(0);
        } else if (center[0] > 1.0 - eps) {
          face->set_boundary_id(2);
        } else if (center[1] < eps) {
          face->set_boundary_id(1);
        } else if (center[1] > 1.0 - eps) {
          face->set_boundary_id(3);
        }
      }

      const auto center = Point<dim>(0.5, 0.5);
      for (auto cell : triangulation.active_cell_iterators()) {
        const auto position = cell->center();
        const auto distance = std::abs((position - center)[0]);

        if (distance < radius_ && position[1] > 0.5)
          cell->set_material_id(2);
        else if (distance < radius_ && position[1] < 0.5)
          cell->set_material_id(1);
        else
          cell->set_material_id(3);
      }

      triangulation.refine_global(refinement_);

    } else {

      AssertThrow(false, ExcMessage("not implemented"));
    }

    /*
     * Populate the rest:
     */

    mapping_.reset(new MappingQ<dim>(order_mapping_));

    finite_element_.reset(
        new FESystem<dim>(FE_Q<dim>(order_finite_element_), 2));

    finite_element_ho_.reset(
        new FESystem<dim>(FE_Q<dim>(order_finite_element_ + 1), 2));

    quadrature_.reset(new QGauss<dim>(order_quadrature_));
  }

} /* namespace grendel */

#endif /* DISCRETIZATION_TEMPLATE_H */
