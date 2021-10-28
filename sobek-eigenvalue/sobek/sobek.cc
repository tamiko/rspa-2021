#include "simpleloop.h"

#include <deal.II/base/utilities.h>

#include <fstream>

int main(int argc, char **argv)
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  sobek::SimpleLoop<DIM> simple_loop;

  /*
   * If necessary, create empty parameter file and exit:
   */

  const auto filename = "sobek.prm";
  if (!std::ifstream(filename)) {
    std::ofstream file(filename);
    dealii::ParameterAcceptor::prm.print_parameters(
        file, dealii::ParameterHandler::OutputStyle::Text);
    return 0;
  }

  simple_loop.run();

  return 0;
}
