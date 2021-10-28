#ifndef SIMPLELOOP_TEMPLATE_H
#define SIMPLELOOP_TEMPLATE_H

#include "simpleloop.h"
#include <output_helper.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/revision.h>
#include <deal.II/grid/grid_out.h>

#include <iomanip>

namespace sobek
{
  using namespace dealii;
  using namespace grendel;

  template <int dim>
  SimpleLoop<dim>::SimpleLoop()
      : ParameterAcceptor("A - Base Parameter")
      , discretization("C - Discretization")
      , cell_problem(discretization, "D - CellProblem")
  {
    base_name_ = "test";
    add_parameter("basename", base_name_, "base name for all output files");
  }


  template <int dim>
  void SimpleLoop<dim>::run()
  {
    /*
     * Prepare output:
     */

    deallog.pop();
    deallog.push(DEAL_II_GIT_SHORTREV "+" SOBEK_GIT_SHORTREV);
#ifdef DEBUG
    deallog.depth_console(5);
    deallog.push("DEBUG");
#else
    deallog.depth_console(4);
#endif
    deallog << "[Init] Initiating Flux Capacitor... [ OK ]" << std::endl;
    deallog << "[Init] Bringing Warp Core online... [ OK ]" << std::endl;

    deallog << "[Init] Reading parameters and allocating objects... "
            << std::flush;

    ParameterAcceptor::initialize("sobek.prm");

    deallog << "[ OK ]" << std::endl;

    /* Print out parameters to a prm file: */
    {
      std::ofstream output(base_name_ + "-parameter.prm");
      ParameterAcceptor::prm.print_parameters(output, ParameterHandler::Text);
    }

    /* Prepare deallog: */
    std::ofstream filestream(base_name_ + "-deallog.log");
    deallog.attach(filestream);

    /* Print out parameters to deallog as well: */
    deallog << "SimpleLoop<dim>::run()" << std::endl;
    ParameterAcceptor::prm.log_parameters(deallog);

    deallog.push(base_name_);

    /* clang-format off */
    /* Print out some info about current library and program versions: */
    deallog << "###" << std::endl;
    deallog << "#" << std::endl;
    deallog << "# deal.II version " << std::setw(8) << DEAL_II_PACKAGE_VERSION
            << "  -  " << DEAL_II_GIT_REVISION << std::endl;
    deallog << "# sobek   version " << std::setw(8) << SOBEK_VERSION
            << "  -  " << SOBEK_GIT_REVISION << std::endl;
    deallog << "#" << std::endl;
    deallog << "###" << std::endl;
    /* clang-format on */

    deallog << "        output primal triangulation" << std::endl;
    std::ofstream output(base_name_ + "-triangulation.inp");
    GridOut().write_ucd(discretization.triangulation(), output);

    cell_problem.run();

    grendel::output<dim>(cell_problem, base_name_ + "-solution");

    deallog.pop();
    deallog.detach();
  }



} /* namespace sobek */

#endif /* SIMPLELOOP_TEMPLATE_H */
