/*
    This file is part of ParTI!.

    ParTI! is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    ParTI! is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with ParTI!.
    If not, see <http://www.gnu.org/licenses/>.
*/

#include <memory>
#include <ParTI/algorithm.hpp>
#include <ParTI/argparse.hpp>
#include <ParTI/cfile.hpp>
#include <ParTI/timer.hpp>
#include <ParTI/session.hpp>
#include <ParTI/sptensor.hpp>
#include <ParTI/utils.hpp>

using namespace pti;

int main(int argc, char const* argv[]) {
    bool dense_format = false;
    size_t limit = 10;
    std::string output;
    int device = 0;
    int solve_device = -1;
    Scalar tol = static_cast<Scalar>(1.0e-5);
    int max_iters = 50;
    ParamDefinition defs[] = {
        { "-d",             PARAM_BOOL,   { &dense_format } },
        { "--dense-format", PARAM_BOOL,   { &dense_format } },
        { "-l",             PARAM_SIZET,  { &limit } },
        { "--limit",        PARAM_SIZET,  { &limit } },
        { "-o",             PARAM_STRING, { &output } },
        { "--output",       PARAM_STRING, { &output } },
        { "--dev",          PARAM_INT,    { &device } },
        { "--solve-dev",    PARAM_INT,    { &solve_device } },
        { "-t",             PARAM_SCALAR, { &tol } },
        { "--tol",          PARAM_SCALAR, { &tol } },
        { "-m",             PARAM_INT,    { &max_iters } },
        { "--max-iters",    PARAM_INT,    { &max_iters } },
        { ptiEndParamDefinition }
    };
    std::vector<char const*> args = parse_args(argc, argv, defs);

    if(args.size() < 3) {
        std::printf("Usage: %s [OPTIONS] X R1 R2 ... dimorder1 dimorder2 ...\n\n", argv[0]);
        std::printf("Options:\n");
        std::printf("\t-d, --dense-format\tPrint tensor in dense format instead of sparse format.\n");
        std::printf("\t-l, --limit\t\tLimit the number of elements to print [Default: 10].\n");
        std::printf("\t-o, --output\t\tWrite the result to a file\n");
        std::printf("\t--dev\t\tComputing device\n");
        std::printf("\t--solve-dev\t\tDevice used only for factor solves [Default: same as --dev].\n");
        std::printf("\t-t, --tol\t\tConvergence tolerance [Default: 1e-5].\n");
        std::printf("\t-m, --max-iters\t\tMaximum HOOI iterations [Default: 50].\n");
        std::printf("\n");
        return 1;
    }

    // session.print_devices();
    Device* dev = session.devices[device];
    if(dynamic_cast<CudaDevice*>(dev) != nullptr) {
        std::printf("Using CUDA for calculation.\n");
    } else if(dynamic_cast<CpuDevice*>(dev) != nullptr) {
        std::printf("Using CPU for calculation.\n");
    } else {
        std::printf("Unknown device type.\n");
        return 1;
    }

    if(solve_device >= 0 && static_cast<size_t>(solve_device) >= session.devices.size()) {
        std::printf("Invalid solve device id.\n");
        return 1;
    }
    Device* solve_dev = (solve_device >= 0) ? session.devices[solve_device] : dev;
    if(dynamic_cast<CudaDevice*>(solve_dev) != nullptr) {
        std::printf("Using CUDA for factor solves.\n");
    } else if(dynamic_cast<CpuDevice*>(solve_dev) != nullptr) {
        std::printf("Using CPU for factor solves.\n");
    } else {
        std::printf("Unknown solve device type.\n");
        return 1;
    }

    CFile fX(args[0], "r");
    SparseTensor X = SparseTensor::load(fX, 1);
    fX.fclose();

    std::printf("X = %s\n\n", X.to_string(!dense_format, limit).c_str());

    if(args.size() < X.nmodes * 2 + 1) {
        std::fprintf(stderr, "Error: Insufficient arguments\n");
        return 1;
    }
    std::unique_ptr<size_t[]> R(new size_t[X.nmodes]);
    std::unique_ptr<size_t[]> dimorder(new size_t[X.nmodes]);
    for(size_t i = 1; i <= X.nmodes; ++i) {
        R[i - 1] = strtonum(std::strtoull, args[i], 0);
        dimorder[i - 1] = strtonum(std::strtoull, args[i + X.nmodes], 0);
    }

    std::printf("Y = tucker_decomposition(X, [%s], [%s], tol=%g, maxiters=%d);\n",
        array_to_string(R.get(), X.nmodes).c_str(),
        array_to_string(dimorder.get(), X.nmodes).c_str(),
        static_cast<double>(tol),
        max_iters);

    Timer timer_tucker(cpu);
    timer_tucker.start();
    SparseTensor Y = tucker_decomposition(
        X, R.get(), dimorder.get(), dev, solve_dev, TUCKER_INIT_NVECS,
        static_cast<double>(tol), static_cast<unsigned>(max_iters));
    timer_tucker.stop();
    timer_tucker.print_elapsed_time("Tucker Decomp");

    std::printf("Y = %s\n", Y.to_string(!dense_format, limit).c_str());

    if(output.length() != 0) {
        CFile fY(output.c_str(), "w");
        X.dump(fY, 1);
    }

    return 0;
}
