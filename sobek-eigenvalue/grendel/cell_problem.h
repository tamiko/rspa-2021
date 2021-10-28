#ifndef CELL_PROBLEM_H
#define CELL_PROBLEM_H

#include "boilerplate.h"
#include "discretization.h"

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

namespace grendel
{

  template <int dim>
  class CellProblem : public dealii::ParameterAcceptor
  {
  public:
    CellProblem(
        const grendel::Discretization<dim> &discretization,
        const std::string &subsection = "CellProblem");

    virtual ~CellProblem() final = default;

    /* Interface for computation: */

    virtual void run()
    {
      setup_system();
      assemble_system();
      solve();
      compute_functionals();
      output_results();
    }

    virtual void setup_system();
    virtual void assemble_system();
    virtual void solve();
    virtual void compute_functionals();
    virtual void output_results();

  protected:
    virtual void setup_constraints();

    unsigned int n_eigenvalues_;

    double moebius_a_;
    double moebius_b_;
    double moebius_c_;
    double moebius_d_;

    dealii::SmartPointer<const grendel::Discretization<dim>> discretization_;
    A_RO(discretization)

    /* Implementation: */

    dealii::DoFHandler<dim> dof_handler_;
    A_RO(dof_handler)

    dealii::SparsityPattern sparsity_pattern_;
    A_RO(sparsity_pattern)

    dealii::AffineConstraints<double> affine_constraints_;
    A_RO(affine_constraints)

    dealii::PETScWrappers::SparseMatrix stiffness_matrix_;
    A_RO(stiffness_matrix)

    dealii::PETScWrappers::SparseMatrix mass_matrix_;
    A_RO(mass_matrix)

    std::vector<dealii::PETScWrappers::MPI::Vector> eigenfunctions_;
    A_RO(eigenfunctions)

    std::vector<double> eigenvalues_;
    A_RO(eigenvalues)

    std::vector<double> factors_;
    std::vector<double> normalization_;
  };

} /* namespace grendel */

#endif /* CELL_PROBLEM_H */
