#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <unordered_map>

namespace fs = std::filesystem;

// Linearize multi-dimensional block index (p_0,...,p_{R-1}) to single index
std::size_t linearizeBlockIndex(const std::vector<std::size_t> &p,
                                const std::vector<std::size_t> &blocksPerDim) {
    std::size_t idx = 0;
    std::size_t stride = 1;
    int R = static_cast<int>(p.size());

    // row-major: last dimension varies fastest
    for (int d = R - 1; d >= 0; --d) {
        idx += p[d] * stride;
        stride *= blocksPerDim[d];
    }
    return idx;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <directory_with_tns_files> S1 S2 S3 [S4 S5]\n";
        return 1;
    }

    std::string dirPath = argv[1];

    // Read block sizes S_k from command line
    std::vector<std::size_t> blockSizes;
    for (int i = 2; i < argc; ++i) {
        std::size_t s = std::stoull(argv[i]);
        if (s == 0) {
            std::cerr << "Block sizes must be positive.\n";
            return 1;
        }
        blockSizes.push_back(s);
    }

    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        std::cerr << "Error: " << dirPath << " is not a directory.\n";
        return 1;
    }

    // Open log file to dump all results
    std::ofstream logout("sub_tsr_nz_cnt.log");
    if (!logout) {
        std::cerr << "Error: could not open log file sub_tsr_nz_cnt.log\n";
        return 1;
    }

    // Iterate over all .tns files in directory
    for (const auto &entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() != ".tns") continue;

        const std::string filePath = entry.path().string();
        std::ifstream fin(filePath);
        if (!fin) {
            std::cerr << "Could not open file: " << filePath << "\n";
            continue;
        }

        std::size_t order;
        if (!(fin >> order)) {
            std::cerr << "Failed to read tensor order from " << filePath << "\n";
            continue;
        }

        if (order == 0 || order > 5) {
            std::cerr << "Skipping " << filePath
                      << " (unsupported order " << order << ")\n";
            continue;
        }

        std::vector<std::size_t> dims(order);
        for (std::size_t r = 0; r < order; ++r) {
            if (!(fin >> dims[r])) {
                std::cerr << "Failed to read dimension sizes from "
                          << filePath << "\n";
                goto next_file;
            }
        }

        {
            // Prepare per-dimension block sizes S_k (reuse last if fewer provided)
            std::vector<std::size_t> S(order);
            for (std::size_t r = 0; r < order; ++r) {
                if (r < blockSizes.size()) {
                    S[r] = blockSizes[r];
                } else {
                    S[r] = blockSizes.back();
                }
            }

            // Number of blocks per dimension: ceil(I_k / S_k)
            std::vector<std::size_t> blocksPerDim(order);
            std::size_t totalBlocks = 1;
            for (std::size_t r = 0; r < order; ++r) {
                blocksPerDim[r] = (dims[r] + S[r] - 1) / S[r];
                totalBlocks *= blocksPerDim[r];
            }

            // Use sparse map for block counts to avoid huge dense allocations.
            // Only blocks that actually receive nonzeros are stored.
            std::unordered_map<std::size_t, std::size_t> blockCounts;

            // Read all entries: i1 ... iR val
            while (true) {
                std::vector<std::size_t> idx(order);
                for (std::size_t r = 0; r < order; ++r) {
                    if (!(fin >> idx[r])) {
                        // normal EOF
                        goto done_entries;
                    }
                }
                double val;
                if (!(fin >> val)) {
                    // malformed line
                    std::cerr << "Malformed entry in " << filePath << "\n";
                    break;
                }

                // Count all entries (they should all be nonzero in COO)
                if (val == 0.0) {
                    continue; // skip zeros if they appear
                }

                // Compute block indices p_r = floor((i_r - 1) / S_r), 0-based
                std::vector<std::size_t> p(order);
                for (std::size_t r = 0; r < order; ++r) {
                    std::size_t coord0 = idx[r] - 1; // convert 1-based -> 0-based
                    std::size_t pr = coord0 / S[r];
                    if (pr >= blocksPerDim[r]) {
                        // Clamp to last block in case of out-of-range indices
                        pr = blocksPerDim[r] - 1;
                    }
                    p[r] = pr;
                }

                std::size_t linIdx = linearizeBlockIndex(p, blocksPerDim);
                ++blockCounts[linIdx];
            }

        done_entries:
            // Output histogram for this tensor
            logout << "File: " << filePath << "\n";
            logout << "Order: " << order << "\n";
            logout << "Dims: ";
            for (std::size_t r = 0; r < order; ++r) {
                logout << dims[r] << (r + 1 < order ? " " : "\n");
            }
            logout << "Blocks per dim: ";
            for (std::size_t r = 0; r < order; ++r) {
                logout << blocksPerDim[r] << (r + 1 < order ? " " : "\n");
            }

            logout << "BlockIndex  NonZeros\n";
            // Dump only blocks that have at least one nonzero
            for (const auto &kv : blockCounts) {
                logout << kv.first << " " << kv.second << "\n";
            }
            logout << "\n";
        }

    next_file:
        fin.close();
        continue;
    }

    return 0;
}