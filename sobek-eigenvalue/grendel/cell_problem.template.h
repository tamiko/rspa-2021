#ifndef CELL_PROBLEM_TEMPLATE_H
#define CELL_PROBLEM_TEMPLATE_H

#include "cell_problem.h"
#include "helper.template.h"

#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools.h>

namespace grendel
{
  using namespace dealii;


  template <int dim>
  CellProblem<dim>::CellProblem(
      const grendel::Discretization<dim> &discretization,
      const std::string &subsection /*= "CellProblem"*/)
      : ParameterAcceptor(subsection)
      , discretization_(&discretization)
  {
    n_eigenvalues_ = 10;
    moebius_a_ = 1.;
    moebius_b_ = 0.;
    moebius_c_ = 0.;
    moebius_d_ = 1.;

    add_parameter("number of eigenvalues", n_eigenvalues_, "");
    add_parameter("moebius a", moebius_a_, "Moebius transform: parameter a");
    add_parameter("moebius b", moebius_b_, "Moebius transform: parameter b");
    add_parameter("moebius c", moebius_c_, "Moebius transform: parameter c");
    add_parameter("moebius d", moebius_d_, "Moebius transform: parameter d");
  }


  template <int dim>
  void CellProblem<dim>::setup_system()
  {
    deallog << "CellProblem<dim>::setup_system()" << std::endl;

    dof_handler_.initialize(discretization_->triangulation(),
                            discretization_->finite_element());

    deallog << "        " << dof_handler_.n_dofs() << " DoFs" << std::endl;

    DoFRenumbering::Cuthill_McKee(dof_handler_);

    setup_constraints();

    DynamicSparsityPattern c_sparsity(dof_handler_.n_dofs(),
                                      dof_handler_.n_dofs());
    DoFTools::make_sparsity_pattern(
        dof_handler_, c_sparsity, affine_constraints_, false);
    sparsity_pattern_.copy_from(c_sparsity);

    stiffness_matrix_.reinit(sparsity_pattern_);
    mass_matrix_.reinit(sparsity_pattern_);

    eigenfunctions_.resize(n_eigenvalues_);

    IndexSet n_dofs = dof_handler_.locally_owned_dofs();

    for (unsigned int i = 0; i < n_eigenvalues_; ++i) {
      eigenfunctions_[i].reinit(n_dofs, MPI_COMM_WORLD);
    }

    eigenvalues_.resize(n_eigenvalues_, 0.);

    factors_.resize(dim * (n_eigenvalues_ + 1), 0.);
    normalization_.resize(n_eigenvalues_, 0.);
  }


  template <int dim>
  void CellProblem<dim>::setup_constraints()
  {
    affine_constraints_.clear();

    pin_a_dof(dof_handler_, affine_constraints_, 0);

    for (int i = 0; i < dim; ++i)
      DoFTools::make_periodicity_constraints(dof_handler_,
                                             /*b_id1    */ i,
                                             /*b_id2    */ dim + i,
                                             /*direction*/ i,
                                             affine_constraints_);

    DoFTools::make_hanging_node_constraints(dof_handler_, affine_constraints_);

    affine_constraints_.close();
  }


  // internal data for parallelized assembly
  namespace
  {
    template <int dim>
    class AssemblyScratchData
    {
    public:
      AssemblyScratchData(const AssemblyScratchData<dim> &assembly_scratch_data)
          : AssemblyScratchData(assembly_scratch_data.discretization_)
      {
      }

      AssemblyScratchData(const grendel::Discretization<dim> &discretization)
          : discretization_(discretization)
          , fe_values_(discretization_.mapping(),
                       discretization_.finite_element(),
                       discretization_.quadrature(),
                       update_values | update_gradients |
                           update_quadrature_points | update_JxW_values)
          , face_quadrature_(3) // FIXME
          , fe_face_values_(discretization_.mapping(),
                            discretization_.finite_element(),
                            face_quadrature_,
                            update_values | update_gradients |
                                update_quadrature_points |
                                update_normal_vectors | update_JxW_values)
      {
      }

      const grendel::Discretization<dim> &discretization_;
      FEValues<dim> fe_values_;
      const QGauss<dim - 1> face_quadrature_;
      FEFaceValues<dim> fe_face_values_;
    };

    template <int dim>
    class AssemblyCopyData
    {
    public:
      std::vector<types::global_dof_index> local_dof_indices_;
      FullMatrix<double> cell_stiffness_matrix_;
      FullMatrix<double> cell_mass_matrix_;

      std::vector<double> factors_;
      std::vector<double> normalization_;
    };

  } /* anonymous namespace */


  template <int dim>
  void CellProblem<dim>::assemble_system()
  {
    deallog << "CellProblem<dim>::assemble_system()" << std::endl;

    stiffness_matrix_ = 0.;
    mass_matrix_ = 0.;

    const unsigned int dofs_per_cell =
        this->discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = this->discretization_->quadrature().size();

    /* The local, per-cell assembly routine: */
    auto local_assemble_system = [&](const auto &cell,
                                     auto &scratch,
                                     auto &copy) {

      auto &cell_stiffness_matrix = copy.cell_stiffness_matrix_;
      auto &cell_mass_matrix = copy.cell_mass_matrix_;
      auto &fe_face_values = scratch.fe_face_values_;
      auto &fe_values = scratch.fe_values_;
      auto &local_dof_indices = copy.local_dof_indices_;

      cell_stiffness_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_mass_matrix.reinit(dofs_per_cell, dofs_per_cell);

      fe_values.reinit(cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      FEValuesViews::Scalar<dim> fe_real(fe_values, 0);

      const auto id = cell->material_id();

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        const auto JxW = fe_values.JxW(q_point);

        // index j for ansatz space, index i for test space

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {

          const auto grad = fe_real.gradient(i, q_point);
          const auto grad_JxW = grad * JxW;

          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const auto grad = fe_real.gradient(j, q_point);
            const auto M = grad_JxW * grad;

            cell_stiffness_matrix(i, j) += moebius_b_ * M;
            cell_mass_matrix(i, j) += moebius_d_ * M;
          } /* for j */
        }   /* for i */
      }     /* for q */

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if(face->at_boundary())
          continue;

        auto neighbor_id = cell->neighbor(f)->material_id();
        const bool at_interface =
            (id == 1 && neighbor_id == 2) || (id == 2 && neighbor_id == 1);
        if (!at_interface)
          continue;

        fe_face_values.reinit(cell, f);

        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        FEValuesViews::Scalar<dim> fe_real(fe_face_values, 0);

        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {

          const auto normal = fe_face_values.normal_vector(q_point);
          const auto JxW = fe_face_values.JxW(q_point);

          // index j for ansatz space, index i for test space

          for (unsigned int i = 0; i < dofs_per_cell; ++i) {

            const auto grad = fe_real.gradient(i, q_point);
            const auto grad_t_JxW = tangential_part(grad, normal) * JxW;

            for (unsigned int j = 0; j < dofs_per_cell; ++j) {
              const auto grad = fe_real.gradient(j, q_point);
              const auto grad_t = tangential_part(grad, normal);

              const auto S = 0.5 * (grad_t_JxW * grad_t);

              cell_stiffness_matrix(i, j) += moebius_a_ * S;
              cell_mass_matrix(i, j) += moebius_c_ * S;
            } /* for j */
          }   /* for i */
        }     /* for q */
      }       /* for loop over faces*/
    };

    /* The local-to-global copy routine: */
    auto copy_local_to_global = [this](const auto &copy) {
      auto &cell_stiffness_matrix = copy.cell_stiffness_matrix_;
      auto &cell_mass_matrix = copy.cell_mass_matrix_;
      auto &local_dof_indices = copy.local_dof_indices_;

      affine_constraints_.distribute_local_to_global(
          cell_stiffness_matrix, local_dof_indices, stiffness_matrix_);

      affine_constraints_.distribute_local_to_global(
          cell_mass_matrix, local_dof_indices, mass_matrix_);
    };

    /* And run a workstream to assemble the matrix: */

    WorkStream::run(dof_handler_.begin_active(),
                    dof_handler_.end(),
                    local_assemble_system,
                    copy_local_to_global,
                    AssemblyScratchData<dim>(*this->discretization_),
                    AssemblyCopyData<dim>());

    stiffness_matrix_.compress(VectorOperation::add);
    mass_matrix_.compress(VectorOperation::add);
  }

  template <int dim>
  void CellProblem<dim>::solve()
  {
    deallog << "CellProblem<dim>::solve()" << std::endl;

    SolverControl solver_control(10000, 1.0e-8, true, true);
    SLEPcWrappers::SolverKrylovSchur eigensolver(solver_control);

    eigensolver.set_which_eigenpairs(EPS_SMALLEST_MAGNITUDE);
    eigensolver.set_problem_type(EPS_GHEP);

    eigensolver.solve(stiffness_matrix_,
                      mass_matrix_,
                      eigenvalues_,
                      eigenfunctions_,
                      n_eigenvalues_);
  }

  template <int dim>
  void CellProblem<dim>::compute_functionals()
  {
    deallog << "CellProblem<dim>::assemble_system()" << std::endl;

    const unsigned int dofs_per_cell =
        this->discretization_->finite_element().dofs_per_cell;

    const unsigned int n_q_points = this->discretization_->quadrature().size();

    /* The local, per-cell integration routine: */
    auto local_integrate = [&](const auto &cell, auto &scratch, auto &copy) {

      auto &factors = copy.factors_;
      factors = std::vector<double>(dim * (n_eigenvalues_ + 1), 0.);

      auto &normalization = copy.normalization_;
      normalization = std::vector<double>(n_eigenvalues_, 0.);

      auto &fe_values = scratch.fe_values_;
      auto &local_dof_indices = copy.local_dof_indices_;

      fe_values.reinit(cell);
      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      FEValuesViews::Scalar<dim> fe_real(fe_values, 0);

      const auto id = cell->material_id();

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

        const auto JxW = fe_values.JxW(q_point);

        Tensor<1, dim, double> grad_phi;

        for (unsigned int k = 0; k < n_eigenvalues_; ++k) {
          for (unsigned int i = 0; i < dofs_per_cell; ++i) {
            const double val = eigenfunctions_[k](local_dof_indices[i]);
            grad_phi += fe_real.gradient(i, q_point) * val;
          }
          normalization[k] += std::abs(grad_phi * grad_phi) * JxW;
        }
      } /* for q */

      auto &fe_face_values = scratch.fe_face_values_;

      for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
        const auto face = cell->face(f);

        if(face->at_boundary())
          continue;

        auto neighbor_id = cell->neighbor(f)->material_id();
        const bool at_interface =
            (id == 1 && neighbor_id == 2) || (id == 2 && neighbor_id == 1);
        if (!at_interface)
          continue;

        fe_face_values.reinit(cell, f);

        const unsigned int n_face_q_points = scratch.face_quadrature_.size();

        FEValuesViews::Scalar<dim> fe_real(fe_face_values, 0);

        for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point) {

          const auto normal = fe_face_values.normal_vector(q_point);
          const auto JxW = fe_face_values.JxW(q_point);

          for (unsigned int d = 0; d < dim; ++d) {
            dealii::Tensor<1, dim, double> direction;
            direction[d] = 1.;

            const auto direction_t_JxW = tangential_part(direction, normal) * JxW;

            factors[dim * n_eigenvalues_ + d] +=
                0.5 * direction * direction_t_JxW;

            for (unsigned int k = 0; k < n_eigenvalues_; ++k) {

              Tensor<1, dim, double> grad_t_phi;

              for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                const double val = eigenfunctions_[k](local_dof_indices[i]);

                const auto grad = fe_real.gradient(i, q_point) * val;
                grad_t_phi += tangential_part(grad, normal);
              }

              factors[dim * k + d] += 0.5 * direction_t_JxW * grad_t_phi;
            }
          } /* d */
        }   /* for q */
      }     /* for loop over faces*/
    };

    /* The local-to-global copy routine: */
    auto sum_up = [this](const auto &copy) {
      for (unsigned int k = 0; k < dim * (n_eigenvalues_ + 1); ++k)
        factors_[k] += copy.factors_[k];

      for (unsigned int k = 0; k < n_eigenvalues_; ++k)
        normalization_[k] += copy.normalization_[k];
    };

    /* And run a workstream to assemble the matrix: */

    WorkStream::run(dof_handler_.begin_active(),
                    dof_handler_.end(),
                    local_integrate,
                    sum_up,
                    AssemblyScratchData<dim>(*this->discretization_),
                    AssemblyCopyData<dim>());
  }


  template <int dim>
  void CellProblem<dim>::output_results()
  {
    for (unsigned int i = 0; i < n_eigenvalues_; ++i) {
      auto transformed = eigenvalues_[i];
      const auto original = (moebius_d_ * transformed - moebius_b_) /
                            (moebius_a_ - moebius_c_ * transformed);

      deallog << i << "th eigenvalue:\t(( " << transformed << "))  -->  "
              << original;

      deallog << "\t\t[factor:";

      for (unsigned int d = 0; d < dim; ++d)
        deallog << " " << factors_[dim * i + d];

      deallog << "][normalization: " << std::sqrt(normalization_[i]) << "]"
              << std::endl;
    }

    deallog << std::endl << "[ measure:";
    for (unsigned int d = 0; d < dim; ++d)
      deallog << " " << factors_[dim * n_eigenvalues_ + d];
    deallog << " ]" << std::endl;
  }

} /* namespace grendel */

#endif /* CELL_PROBLEM_TEMPLATE_H */
