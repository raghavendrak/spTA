#include "gputucker/cmdline_opts.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

#include "gputucker/constants.hpp"
#include "gputucker/helper.hpp"
namespace supertensor {
namespace gputucker {

CommandLineOptions::CommandLineOptions(): _input_path(""),
                                          _order(3),
                                          _rank(10),
                                          _gpu_count(1),
                                          _max_iter(constants::kMaxIteration),
                                          _tol(constants::kLambda),
                                          _output_dir("") {
  Initialize();
}

CommandLineOptions::~CommandLineOptions() {}

/*
* Initialize the command line options
* @return void
*/
void CommandLineOptions::Initialize() {
  po::options_description options("Program Options");

  options.add_options()
    ("help,h", "Display help menu.")
    ("input,i", po::value<std::string>(&this->_input_path),"Input tensor path")
    ("order,o", po::value<int>(&this->_order), "Order")
    ("rank,r",  po::value<int>(&this->_rank)->default_value(10), "Rank")
    ("gpus,g",  po::value<int>(&this->_gpu_count)->default_value(1), "The number of GPUs")
    ("max-iter,m", po::value<int>(&this->_max_iter)->default_value(constants::kMaxIteration),
     "Maximum number of decomposition iterations")
    ("tol,t", po::value<double>(&this->_tol)->default_value(constants::kLambda),
     "Convergence tolerance on observed-entry fit change")
    ("output-dir", po::value<std::string>(&this->_output_dir)->default_value(""),
     "Optional directory to write final factors, core, and summary metrics");

  this->_options.add(options);
}

/*
* Parse the command line options
* @param argc - The number of arguments
* @param argv - The arguments
* @return ReturnStatus - The return status
*/
CommandLineOptions::ReturnStatus CommandLineOptions::Parse(int argc, char *argv[]) {
  ReturnStatus ret = OPTS_SUCCESS;

  po::variables_map var_map;

  try {
    // Parse the command line options
    po::store(po::parse_command_line(argc, argv, this->_options), var_map);
    po::notify(var_map);

    // Help option
    if (var_map.count("help")) {
      std::cout << this->_options << std::endl;
      return OPTS_HELP;
    }

    // Enforce an input file argument every time
    if (!(0 < var_map.count("input"))) {
      std::cout << CYN "[ERROR] Input file must be specified!!!" RESET << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    } else {
      // Strip whitespaces from front/back of filename string
      boost::algorithm::trim(this->_input_path);
      ret = ValidateFile() ? OPTS_SUCCESS : OPTS_FAILURE;
    }

    // Enforce an order argument every time
    if (!(0 < var_map.count("order"))) {
      std::cout << CYN "[ERROR] Tensor order must be specified!!!" << std::endl;
      std::cout << this->_options << std::endl;
      return OPTS_FAILURE;
    }

    // We can check if a rank is defaulted
    if (!var_map["rank"].defaulted()) {
      std::cout << "[WARNING] Default value for User-Value overwritten to " << this->_rank << std::endl;
    }

    // We can check if the number of GPUs is defaulted
    if (!var_map["gpus"].defaulted()) {
      std::cout << "[WARNING] Default value for GPU count overwritten to " << this->_gpu_count << std::endl;
    }

    if (!var_map["max-iter"].defaulted()) {
      std::cout << "[WARNING] Default value for max iteration overwritten to " << this->_max_iter << std::endl;
    }

    if (!var_map["tol"].defaulted()) {
      std::cout << "[WARNING] Default value for tolerance overwritten to " << this->_tol << std::endl;
    }

  } catch (std::exception &e) {
    std::cout << "[ERROR] Parsing error : " << e.what() << std::endl;
    ret = OPTS_FAILURE;
  } catch (...) {
    std::cout << "[ERROR] Parsing error (unknown type)." << std::endl;
    ret = OPTS_FAILURE;
  }

  return ret;
}

/*
* Validate the input file
* @return bool - True if the file is valid, false otherwise
*/
bool CommandLineOptions::ValidateFile() {
  if (!boost::filesystem::is_regular_file(this->_input_path)) {
    std::cout << CYN "[ERROR] Input file provided does not exist [" << this->_input_path << "]" << std::endl;
    return false;
  }
  return true;
}

}  // namespace gputucker
}  // namespace supertensor
