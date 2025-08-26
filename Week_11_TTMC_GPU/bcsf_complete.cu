/**
 * B-CSF MTTKRP Complete Implementation
 * 
 * This file contains a complete implementation of the B-CSF (Balanced-CSF) 
 * MTTKRP (Matricized Tensor Times Khatri-Rao Product) algorithm for sparse tensors.
 * 
 * Based on the original B-CSF implementation by Israt Nisa et al.
 * Ohio State University
 */

#include <fstream>
#include <stdio.h>
#include <algorithm>
#include <iterator>
#include <utility>  
#include <math.h> 
#include <omp.h>
#include <cuda.h>
#include <vector>
#include <unordered_map>
#include <map>
#include <boost/functional/hash.hpp>
#include <bits/stdc++.h>  
#include <time.h>
#include <sys/time.h>
#include <iomanip> 
#include <iostream>

using namespace std;

// Data type definitions
#define DTYPE float
#define ITYPE size_t

// CUDA error checking
inline cudaError_t checkCuda(cudaError_t result, int s){
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error in line : %s - %d\n", cudaGetErrorString(result), s);
    assert(result == cudaSuccess);
  }
  return result;
}

// CUDA timer functions
void cuda_timer_start(cudaEvent_t start){
  checkCuda(cudaEventRecord(start), __LINE__);
}

void cuda_timer_stop(cudaEvent_t start, cudaEvent_t stop, float &mili){
  checkCuda(cudaEventRecord(stop), __LINE__);
  cudaEventSynchronize(stop);
  checkCuda(cudaEventElapsedTime(&mili, start, stop), __LINE__);
  cudaDeviceSynchronize();
}

// Utility function for timing
inline double seconds(){
  struct timeval tp;
  gettimeofday(&tp, NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// Data structures
class Matrix{
  public:
  ITYPE nRows;
  ITYPE nCols;
  DTYPE *vals;
};

class Tensor{
  public:
  ITYPE ndims;
  ITYPE *dims;
  ITYPE totNnz;
  ITYPE nFibers;
  ITYPE *accessK;
  ITYPE *fbrLikeSlcInds;
  bool switchBC = false;
  std::vector<ITYPE> modeOrder;
  std::vector<ITYPE> fbrCount;
  std::vector<vector<ITYPE>> inds;
  std::vector<DTYPE> vals;
  std::vector<vector<ITYPE>> fbrPtr;
  std::vector<vector<ITYPE>> fbrIdx;
  std::vector<vector<ITYPE>> slcMapperBin;
  ITYPE *nnzPerSlice;
  ITYPE *fiberPerSlice;
  ITYPE *nnzPerFiber;
  ITYPE *denseSlcPtr;
  ITYPE *partPerNnz;
  ITYPE *totnnzPerPart;
  unordered_map<pair<ITYPE, ITYPE>, ITYPE, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl;
};

class TiledTensor{
  public:
  ITYPE ndims;
  ITYPE *dims;
  ITYPE totNnz;
  ITYPE nFibers;
  ITYPE *accessK;
  ITYPE *fbrLikeSlcInds;
  std::vector<ITYPE> modeOrder;
  std::vector<ITYPE> fbrCount;
  std::vector<vector<ITYPE>> inds;
  std::vector<DTYPE> vals;
  std::vector<vector<ITYPE>> fbrPtr;
  std::vector<vector<ITYPE>> fbrIdx;
  std::vector<vector<ITYPE>> slcMapperBin;
  ITYPE *nnzPerSlice;
  ITYPE *fiberPerSlice;
  ITYPE *nnzPerFiber;
  ITYPE *denseSlcPtr;
  ITYPE *partPerNnz;
  ITYPE *totnnzPerPart;
  unordered_map<pair<ITYPE, ITYPE>, int, boost::hash<pair<ITYPE, ITYPE>>> fbrHashTbl;
};

class Options {
  public:
  ITYPE R = 32;
  ITYPE S = 32;  // Added S parameter for TTMC
  ITYPE R3 = 32; // Added R3 parameter for TTMC
  ITYPE mode = 0;
  ITYPE impType = 1;
  ITYPE warpPerSlice = 4;
  ITYPE nTile = 1;
  ITYPE tileSize;
  ITYPE gridSize = 512;
  ITYPE TBsize = 512;
  ITYPE MIfbTh = 1;
  bool verbose = false;
  bool correctness = false;
  bool isTTMC = false; // Added isTTMC flag
  string inFileName;
  string outFileName;
  ITYPE nBin = 10;
  std::string m0 = "012";
  std::string m1 = "120";
  std::string m2 = "201";
  ITYPE fbrThreashold = 99999999;

  void print() {
    std::cout << "R = " << R << '\n';
    std::cout << "S = " << S << '\n';
    std::cout << "R3 = " << R3 << '\n';
    std::cout << "mode = " << mode << '\n';
    std::cout << "impType = " << impType << '\n';
    std::cout << "warpPerSlice = " << warpPerSlice << '\n';
    std::cout << "nTiles = " << nTile << '\n';
    std::cout << "verbose = " << verbose << '\n';
    std::cout << "isTTMC = " << isTTMC << '\n';

    if(inFileName.empty()){
      cout << "Provide input file path. Program will exit." << endl;
      exit(0);
    }
    else{
      std::cout << "input file name = " << inFileName << '\n';
    }

    if(!outFileName.empty())
    std::cout << "output file name = " << outFileName << '\n';
  }
};

// Utility functions
inline void print_help_and_exit() {
  printf("options:\n\
  -R rank/feature : set the rank (default 32)\n\
  -S S parameter : set the S parameter for TTMC (default 32)\n\
  -3 R3 parameter : set the R3 parameter for TTMC (default 32)\n\
  -m mode : set the mode of MTTKRP (default 0)\n\
  -t implementation type: 1: COO CPU, 2: HCSR CPU, 3: COO GPU 4: HCSR GPU 8: B-CSF 10: HB-CSF (default 1) \n\
  -f fiber-splitting threshold: set the maximum length (nnz) for each fiber. Longer fibers will be split (default inf)\n\
  -w warp per slice: set number of WARPs assign to per slice  (default 4)\n\
  -i input file name: e.g., ../dataset/delicious.tns \n\
  -o output file name: if not set not output file will be written\n\
  -T TTMC flag: set to 1 to enable TTMC mode (default 0)\n");
    
  exit(1);
}

inline Options parse_cmd_options(int argc, char **argv) {
  Options param;
  int i;
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
    break;
    if (++i >= argc){
      print_help_and_exit();
    }
  
    switch (argv[i - 1][1]) {
      case 'R':
      param.R = atoi(argv[i]);
      break;
      case 'S':
      param.S = atoi(argv[i]);
      break;
      case '3':
      param.R3 = atoi(argv[i]);
      break;
      case 'm':
      param.mode = atoi(argv[i]);
      break;
      case 't':
      param.impType = atoi(argv[i]);
      break;
      case 'w':
      param.warpPerSlice = atoi(argv[i]);
      break;
      case 'l':
      param.nTile = atoi(argv[i]);
      break;
      case 'f':
      param.fbrThreashold = atoi(argv[i]);
      break;
      case 'b':
      param.TBsize = atoi(argv[i]);
      break;
      case 'h':
      param.MIfbTh = atoi(argv[i]);
      break;
      case 'g':
      param.gridSize = atoi(argv[i]);
      break;
      case 'v':
      if(atoi(argv[i]) == 1)
      param.verbose = true;
      else
      param.verbose = false;
      break;
      case 'c':
      if(atoi(argv[i]) == 1)
      param.correctness = true;
      else
      param.correctness = false;
      break;
      case 'T':
      if(atoi(argv[i]) == 1)
      param.isTTMC = true;
      else
      param.isTTMC = false;
      break;
      case 'i':
      param.inFileName = argv[i];
      break;
      case 'o':
      param.outFileName = argv[i];
      break;
      case 'p':
      param.m0 = argv[i];
      break;
      case 'q':
      param.m1 = argv[i];
      break;
      case 'r':
      param.m2 = argv[i];
      break;
      default:
      fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
      print_help_and_exit();
      break;
    }
  }
  
  if (i > argc){
    cout << "weird " << argc << endl;
    print_help_and_exit();
  }
  return param;
}

inline int load_tensor(Tensor &X, const Options &Opt){
  if(Opt.verbose)
  cout << endl << "Loading tensor.." << endl;
  
  string filename = Opt.inFileName;
  ITYPE index;
  DTYPE vid=0;
  ITYPE switchMode = 0;
  bool switchBC =  false;

  ifstream fp(filename);
  if(fp.fail()){
    cout << filename << " does not exist!" << endl;
    exit(0);
  }

  fp >> X.ndims;
  X.dims = new ITYPE[X.ndims];

  for (int i = 0; i < X.ndims; ++i){
    fp >> X.dims[i];
    X.inds.push_back(std::vector<ITYPE>());
  }

  int mode1 = (1 + Opt.mode) % X.ndims;
  int mode2 = (2 + Opt.mode) % X.ndims;

  if( X.dims[mode1] > X.dims[mode2]) switchBC = true;

  for (int i = 0; i < X.ndims; ++i){
    if(i > 0 && switchBC){
      if(i == 1)
      switchMode = 2;
      else if(i == 2)
      switchMode = 1;
    }
    else
    switchMode = i;
    X.modeOrder.push_back((switchMode + Opt.mode) % X.ndims);
  }

  while(fp >> index) {
    X.inds[0].push_back(index-1);
    for (int i = 1; i < X.ndims; ++i) {
      fp >> index;
      X.inds[i].push_back(index-1);
    }
    fp >> vid;
    X.vals.push_back(vid);
  }
  X.totNnz = X.vals.size();
  return 0;
}

inline bool sort_pred(tuple <ITYPE, ITYPE, ITYPE, DTYPE> left,
tuple <ITYPE, ITYPE, ITYPE, DTYPE> right) {
  if (get<0>(left) != get<0>(right))
  return (get<0>(left) < get<0>(right));
  return (get<1>(left) < get<1>(right));
}

inline int sort_COOtensor(Tensor &X){
  const ITYPE mode0 = X.modeOrder[0];
  const ITYPE mode1 = X.modeOrder[1];
  const ITYPE mode2 = X.modeOrder[2];

  vector < tuple <ITYPE, ITYPE, ITYPE, DTYPE> > items;
  tuple <ITYPE, ITYPE, ITYPE, DTYPE> ap;

  for (long idx = 0; idx < X.totNnz; ++idx) {
    ap=std::make_tuple(X.inds[mode0][idx], X.inds[mode1][idx], X.inds[mode2][idx], X.vals[idx]);
    items.push_back(ap);
  }

  std::sort(items.begin(), items.end(), sort_pred);

  for (long idx = 0; idx < X.totNnz; ++idx) {
    X.inds[mode0][idx] = get<0>(items[idx]);
    X.inds[mode1][idx] = get<1>(items[idx]);
    X.inds[mode2][idx] = get<2>(items[idx]);
    X.vals[idx] = get<3>(items[idx]);
  }
  return 0;
}

inline int create_mats(const Tensor &X, Matrix *U, const Options &Opt, bool ata){
  ITYPE mode;
  ITYPE R = Opt.R;
  for (int m = 0; m < X.ndims; ++m){
    mode = X.modeOrder[m];
    U[mode].nRows =  X.dims[mode];
    U[mode].nCols =  R;
    if(Opt.isTTMC && m == Opt.mode){
      if(X.ndims == 3){
        U[mode].nCols = R * Opt.S;
      }
      else{
        U[mode].nCols = R * Opt.S * Opt.R3;
      }
    }
    if(ata)
    U[mode].nCols = U[mode].nRows;
    U[mode].vals = (DTYPE*)malloc(U[mode].nRows * U[mode].nCols * sizeof(DTYPE));
  }
  return 0;
}

inline int randomize_mats(const Tensor &X, Matrix *U, const Options &Opt){
  ITYPE mode;
  for (int m = 0; m < X.ndims; ++m){
    mode = X.modeOrder[m];
    srand48(0L);
    for(long r = 0; r < U[mode].nRows; ++r){
      for(long c = 0; c < U[mode].nCols; ++c)
      U[mode].vals[r * U[mode].nCols + c] =  mode + .5;
    }
  }
  return 0;
}

inline int zero_mat(const Tensor &X, Matrix *U, ITYPE mode){
  for(long r = 0; r < U[mode].nRows; ++r){
    for(long c = 0; c < U[mode].nCols; ++c)
    U[mode].vals[r * U[mode].nCols +c] = 0;
  }
  return 0;
}

inline void write_output(Matrix *U, ITYPE mode, string outFile){
  ofstream fp(outFile);
  fp << U[mode].nRows << " x " << U[mode].nCols << " matrix" << endl;
  fp << std::fixed;
  for (int i = 0; i < U[mode].nRows; ++i) {
    for (int j = 0; j < U[mode].nCols; ++j) {
      fp << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
    }
    fp << endl;
  }
}

inline void print_matrix(Matrix *U, ITYPE mode){
  cout << U[mode].nRows << " x " << U[mode].nCols << " matrix" << endl;
  cout << std::fixed;
  int rows_upper_bound = (U[mode].nRows > 5) ? 5 : U[mode].nRows;
  for (int i = 0; i <  rows_upper_bound; ++i) {
    for (int j = 0; j < 5; ++j) {
      cout << std::setprecision(2) << U[mode].vals[i * U[mode].nCols + j] << "\t" ;
    }
    cout << endl;
  }
}

inline void correctness_check(DTYPE *out, DTYPE *COOout, int nr, int nc){
  long mismatch = 0;
  DTYPE maxDiff = 0;
  DTYPE precision = 0.1;
  cout << std::fixed;
  for (int i = 0; i < nr; ++i){
    for (int j = 0; j < nc; ++j){
      DTYPE diff = abs(out[i * nc + j] - COOout[i * nc + j])/abs(COOout[i * nc + j]);
      if( diff > precision){
        if(diff > maxDiff)
        maxDiff = diff;
        if(mismatch < 2 )
        cout << "mismatch at (" << i <<"," << j <<") got: " << out[i * nc +j] << " exp: " << COOout[i * nc +j] << endl;
        mismatch++;
      }
    }
  }

  if(mismatch == 0)
  cout << "Correctness pass!" << endl;
  else{
    cout <<  mismatch <<" mismatches found at " << precision << " precision" << endl;
    cout << "Maximum diff " << maxDiff << endl;
  }
}

// COO CPU implementations for correctness checking
int MTTKRP_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt){
  int *curMode = new int [X.ndims];
  ITYPE R = Opt.R;

  for (int m = 0; m < X.ndims; ++m)
  curMode[m] = (m + Opt.mode) % X.ndims;
      
  ITYPE mode0 = curMode[0];
  ITYPE mode1 = curMode[1];
  ITYPE mode2 = curMode[2];
  
  for(ITYPE x=0; x<X.totNnz; ++x) {
    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];
  
    for(ITYPE r=0; r<R; ++r) {
      tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r];
      U[mode0].vals[idx0 * R + r] += tmp_val;
    }
  }
  
  delete[] curMode;
  return 0;
}
int TTMC_COO_CPU(const Tensor &X, Matrix *U, const Options &Opt){
  int *curMode = new int [X.ndims];
  ITYPE R = Opt.R;
  ITYPE S = Opt.S;
  for (int m = 0; m < X.ndims; ++m)
  curMode[m] = (m + Opt.mode) % X.ndims;
      
  ITYPE mode0 = curMode[0];
  ITYPE mode1 = curMode[1];
  ITYPE mode2 = curMode[2];
  
  for(ITYPE x=0; x<X.totNnz; ++x) {
    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];
  
    for(ITYPE r=0; r<R; ++r) {
      for(ITYPE s=0; s<S; ++s) {
        tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * S + s];
        U[mode0].vals[idx0 * R * S + r * S + s] += tmp_val;
      }
    }
  }
  
  delete[] curMode;
  return 0;
}

int MTTKRP_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt){
  ITYPE R = Opt.R;
  ITYPE mode0 = X.modeOrder[0];
  ITYPE mode1 = X.modeOrder[1];
  ITYPE mode2 = X.modeOrder[2];
  ITYPE mode3 = X.modeOrder[3];
  
  for(ITYPE x=0; x<X.totNnz; ++x) {
    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];
    ITYPE idx3 = X.inds[mode3][x];
  
    for(ITYPE r=0; r<R; ++r) {
      tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * R + r] * U[mode3].vals[idx3 * R + r];
      U[mode0].vals[idx0 * R + r] += tmp_val;
    }
  }
  return 0;
}
int TTMC_COO_CPU_4D(const Tensor &X, Matrix *U, const Options &Opt){
  ITYPE R = Opt.R;
  ITYPE S = Opt.S;
  ITYPE T = Opt.R3;

  ITYPE mode0 = X.modeOrder[0];
  ITYPE mode1 = X.modeOrder[1];
  ITYPE mode2 = X.modeOrder[2];
  ITYPE mode3 = X.modeOrder[3];
  
  for(ITYPE x=0; x<X.totNnz; ++x) {
    DTYPE tmp_val = 0;
    ITYPE idx0 = X.inds[mode0][x];
    ITYPE idx1 = X.inds[mode1][x];
    ITYPE idx2 = X.inds[mode2][x];
    ITYPE idx3 = X.inds[mode3][x];
  
    for(ITYPE r=0; r<R; ++r) {
      for(ITYPE s=0; s<S; ++s) {
        for(ITYPE t=0; t<T; ++t) {
          tmp_val = X.vals[x] * U[mode1].vals[idx1 * R + r] * U[mode2].vals[idx2 * S + s] * U[mode3].vals[idx3 * T + t];
          U[mode0].vals[idx0 * R * S * T + r * S * T + s * T + t] += tmp_val;
        }
      }
    }
  }
  return 0;
}

// HCSR creation and tiling functions
inline int create_HCSR(Tensor &X, const Options &Opt){
  ITYPE fbrThreashold = Opt.fbrThreashold;
  if(Opt.impType == 12 )
  fbrThreashold = 99999999;

  for (int i = 0; i < X.ndims - 1; ++i){
    X.fbrPtr.push_back(std::vector<ITYPE>());
    X.fbrIdx.push_back(std::vector<ITYPE>());
  }

  std::vector<ITYPE> prevId(X.ndims-1);
  std::vector<ITYPE> fbrId(X.ndims-1);

  for (int i = 0; i < X.ndims-1; ++i){
    prevId[i] =  X.inds[X.modeOrder[i]][0];
    X.fbrPtr[i].push_back(0);
    X.fbrIdx[i].push_back(prevId[i]);
    X.fbrPtr[i].reserve(X.totNnz);
    X.fbrIdx[i].reserve(X.totNnz);
  }

  int idx = 1 ;
  
  while(idx < X.totNnz) {
    for (int i = 0; i < X.ndims-1; ++i)
    fbrId[i] = X.inds[X.modeOrder[i]][idx];

    ITYPE fiberNnz = 1;
    bool sameFbr = true;

    for (int i = 0; i < X.ndims-1; ++i) {
      if(fbrId[i] != prevId[i])
      sameFbr = false;
    }
  
    while( sameFbr && idx < X.totNnz && fiberNnz < fbrThreashold){
      ++idx;
      fiberNnz++;
      for (int i = 0; i < X.ndims-1; ++i) {
        fbrId[i] = X.inds[X.modeOrder[i]][idx];
        if(fbrId[i] != prevId[i])
        sameFbr = false;
      }
    }

    if(idx == X.totNnz)
    break;

    X.fbrPtr[X.ndims-2].push_back(idx);
    X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

    for (int i = X.ndims - 3; i > -1 ; --i) {
      bool diffFbr = false;
      int iDim = i;
      while(iDim > -1){
        if( fbrId[iDim] != prevId[iDim]){
          diffFbr = true;
        }
        iDim--;
      }
      if(diffFbr){
        X.fbrIdx[i].push_back(fbrId[i]);
        X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size()) - 1);
      }
    }

    for (int i = 0; i < X.ndims-1; ++i)
    prevId[i] =  fbrId[i];

    ++idx;
    fiberNnz = 1;
  }
  X.fbrPtr[X.ndims-2].push_back(idx);
  X.fbrIdx[X.ndims-2].push_back(fbrId[X.ndims-2]);

  for (int i = X.ndims - 3; i > -1 ; --i)
  X.fbrPtr[i].push_back((ITYPE)(X.fbrPtr[i+1].size() - 1 ));
  
  X.nFibers = X.fbrPtr[1].size();

  for (int i =0; i <  2 ;i++)
  X.inds[X.modeOrder[i]].resize(0);

  return 0;
}

inline int make_KTiling(const Tensor &X, TiledTensor *TiledX, const Options &Opt){
  ITYPE mode0 = X.modeOrder[0];
  ITYPE mode1 = X.modeOrder[1];
  ITYPE mode2 = X.modeOrder[2];
  ITYPE mode3 = ((X.ndims == 4) ? X.modeOrder[3] : 0) ;
  
  for (int tile = 0; tile < Opt.nTile; ++tile){
    TiledX[tile].ndims = X.ndims;
    TiledX[tile].dims = new ITYPE[TiledX[tile].ndims];
  
    for (int i = 0; i < X.ndims; ++i){
      TiledX[tile].inds.push_back(std::vector<ITYPE>());
      TiledX[tile].dims[i] = X.dims[i];
      TiledX[tile].modeOrder.push_back(X.modeOrder[i]);
    }
  }

  int tile = 0;
  for (int idx = 0; idx < X.totNnz; ++idx){
    tile = ((TiledX[0].ndims == 3) ? X.inds[mode2][idx]/Opt.tileSize : X.inds[mode3][idx]/Opt.tileSize) ;

    for (int i = 0; i < X.ndims; ++i)  {
      TiledX[tile].inds[i].push_back(X.inds[i][idx]);
    }
    TiledX[tile].vals.push_back(X.vals[idx]);
  }
  
  for (int tile = 0; tile < Opt.nTile; ++tile){
    TiledX[tile].totNnz = TiledX[tile].vals.size();
  }
  return 0;
}

inline int create_TiledHCSR(TiledTensor *TiledX, const Options &Opt, int tile){
  ITYPE fbrThreashold = Opt.fbrThreashold;

  for (int i = 0; i < TiledX[tile].ndims - 1; ++i){
    TiledX[tile].fbrPtr.push_back(std::vector<ITYPE>());
    TiledX[tile].fbrIdx.push_back(std::vector<ITYPE>());
  }
  
  ITYPE mode0 = TiledX[tile].modeOrder[0];
  ITYPE mode1 = TiledX[tile].modeOrder[1];
  ITYPE mode2 = TiledX[tile].modeOrder[2];

  std::vector<ITYPE> prevId(TiledX[tile].ndims-1);
  std::vector<ITYPE> fbrId(TiledX[tile].ndims-1);

  for (int i = 0; i < TiledX[tile].ndims-1; ++i){
    prevId[i] =  TiledX[tile].inds[TiledX[tile].modeOrder[i]][0];
    TiledX[tile].fbrPtr[i].push_back(0);
    TiledX[tile].fbrIdx[i].push_back(prevId[i]);
  }
  
  int idx = 1 ;
  
  while(idx < TiledX[tile].totNnz) {
    for (int i = 0; i < TiledX[tile].ndims-1; ++i)
    fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];
  
    ITYPE fiberNnz = 1;
    bool sameFbr = true;

    for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
      if(fbrId[i] != prevId[i])
      sameFbr = false;
    }
      
    while( sameFbr && idx < TiledX[tile].totNnz && fiberNnz < fbrThreashold){
      ++idx;
      fiberNnz++;
      for (int i = 0; i < TiledX[tile].ndims-1; ++i) {
        fbrId[i] = TiledX[tile].inds[TiledX[tile].modeOrder[i]][idx];
        if(fbrId[i] != prevId[i])
        sameFbr = false;
      }
    }

    if(idx == TiledX[tile].totNnz)
    break;

    TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
    TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

    for (int i = TiledX[tile].ndims - 3; i > -1 ; --i) {
      bool diffFbr = false;
      int iDim = i;
      while(iDim > -1){
        if( fbrId[iDim] != prevId[iDim]){
          diffFbr = true;
        }
        iDim--;
      }
      if(diffFbr){
        TiledX[tile].fbrIdx[i].push_back(fbrId[i]);
        TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size()) - 1);
      }
    }
  
    for (int i = 0; i < TiledX[tile].ndims-1; ++i)
    prevId[i] =  fbrId[i];

    ++idx;
    fiberNnz = 1;
  }
  TiledX[tile].fbrPtr[TiledX[tile].ndims-2].push_back(idx);
  TiledX[tile].fbrIdx[TiledX[tile].ndims-2].push_back(fbrId[TiledX[tile].ndims-2]);

  for (int i = TiledX[tile].ndims - 3; i > -1 ; --i)
  TiledX[tile].fbrPtr[i].push_back((ITYPE)(TiledX[tile].fbrPtr[i+1].size() - 1 ));
  
  TiledX[tile].nFibers = TiledX[tile].fbrPtr[1].size();
  return 0;
}

inline int make_TiledBin(TiledTensor *TiledX, const Options & Opt, int tile){
  ITYPE THREADLOAD = 2;
  ITYPE TB = 512;
  std::vector<ITYPE> UB;
  std::vector<ITYPE> LB;

  for (int i = 0; i < Opt.nBin; i++) {
    TiledX[tile].slcMapperBin.push_back(std::vector<ITYPE>());
    UB.push_back((1 << i) * THREADLOAD + 1);
    LB.push_back(UB[i] >> 1);
  }
  

  LB[0] = 0;   UB[0] = 3;  // 1 WARP
  LB[1] = 2;   UB[1] = 5;  // 2 WARP
  LB[2] = 4;   UB[2] = 9;  // 4 WARP
  LB[3] = 8;   UB[3] = 17; // 8 WARP
  LB[4] = 16;   UB[4] = 1025;  // 16 WARP = 1 TB

  LB[5] = 0;   UB[5] = 4 * TB + 1; // 32 WARP =2 TB
  LB[6] = 4 * TB;   UB[6] = 8 * TB + 1; // 64 WARP =4 TB
  LB[7] = 8 * TB;   UB[7] = 16 * TB + 1; // 128 WARP =8 TB
  LB[8] = 16 * TB;   UB[8] = 32 * TB + 1;  // 256 WARP = 16 TB
  LB[9] = 32 * TB ;   UB[9] = TiledX[tile].totNnz + 1;  // 512 WARP = 32 TB

  UB[Opt.nBin - 1] = TiledX[tile].totNnz + 1;
  if(Opt.verbose)
  cout << "Merged all bins for smaller tiles, added nFiber as bin info" << endl;

  UB[0] = 1025; //merging first 5 bin

  for(ITYPE slc = 0; slc < TiledX[tile].fbrIdx[0].size(); ++slc) {
    int nnzSlc = 0;
    nnzSlc += TiledX[tile].fbrPtr[0][slc+1] - TiledX[tile].fbrPtr[0][slc];

    for (int fbrS = TiledX[tile].fbrPtr[0][slc]; fbrS < TiledX[tile].fbrPtr[0][slc+1]; ++fbrS){
      
      if(TiledX[tile].ndims == 3)
      nnzSlc += TiledX[tile].fbrPtr[1][fbrS+1] - TiledX[tile].fbrPtr[1][fbrS];
    
      else if(TiledX[tile].ndims == 4){
        for (int fbr = TiledX[tile].fbrPtr[1][fbrS]; fbr < TiledX[tile].fbrPtr[1][fbrS+1]; ++fbr){
          nnzSlc += TiledX[tile].fbrPtr[2][fbr+1] - TiledX[tile].fbrPtr[2][fbr];
        }
      }
    }
      
    for (int bin = 0; bin < Opt.nBin; ++bin) {
      if (nnzSlc > LB[bin] && nnzSlc < UB[bin]) {
        TiledX[tile].slcMapperBin[bin].push_back(slc);
        break;
      }
    }
  }

  if(Opt.verbose){
    for (int bin = 0; bin < Opt.nBin; ++bin)
    cout << "Bin "<<bin << ": " << TiledX[tile].slcMapperBin[bin].size() << endl;
  }
  return 0;
}

// CUDA kernels for B-CSF MTTKRP
__global__ void mttkrp_HCSR_kernel_smllBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + tId);
  unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;
  unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5)
  DTYPE tmp = 0, tmp_val;
      
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc]; //i_ptr
    unsigned int idx0 = dfbrIdx0[mappedSlc]; //i
    int fb_st = fbrPtr0[mappedSlc];
    int fb_end = fbrPtr0[mappedSlc+1];

      // j is parallelized across warps
      //fbr = j_ptr
    for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
      tmp_val = 0;
      //serial k loop
      //x = k_ptr
      for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

        unsigned int idx2 = dInds2[x];    //k
          // r is parallelized across threads in a warp
        for(unsigned int r=laneId; r<R; r+=32) {
          tmp_val += vals[x] * dU2[idx2 * R + r];
        }
      }
      //idx1 = j
      unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
      for(unsigned int r=laneId; r<R; r+=32) {
        tmp += tmp_val * dU1[idx1 * R + r] ;
      }
    }

    for(unsigned int r=laneId; r<R; r+=32) {
      atomicAdd(&dU0[idx0 * R + r], tmp);
    }
  }
}

__global__ void mttkrp_HCSR_kernel_smllBin_4D(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds3, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, DTYPE *dU3,
ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){

  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + tId);
  unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;
  unsigned int slc = gId >> (5 + logOfWPC);
  DTYPE outbuffer = 0, tmp_val = 0, outbuffer1 = 0;
      
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];
    unsigned int idx0 = dfbrIdx0[mappedSlc];

    for (int fbrS = fbrPtr0[mappedSlc]; fbrS < fbrPtr0[mappedSlc+1]; fbrS++){
      
      unsigned int idx1 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];
      outbuffer1 = 0;
      
      for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
        ITYPE idx2 = fbrIdx2[fbr];
        tmp_val = 0;
      
        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {

          unsigned int idx3 = dInds3[x];
          for(unsigned int r=laneId; r<R; r+=32)
          tmp_val += vals[x] * dU3[idx3 * R + r];
        }
        for(unsigned int r=laneId; r<R; r+=32)
        outbuffer1 += tmp_val * dU2[idx2 * R + r] ;
      }
      for(unsigned int r=laneId; r<R; r+=32)
      outbuffer += outbuffer1 * dU1[idx1 * R + r] ;
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      atomicAdd(&dU0[idx0 * R + r], outbuffer);
    }
  }
}

__global__ void mttkrp_HCSR_kernel_hvyBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
  
  unsigned int laneId = threadIdx.x & 31;
  unsigned int workId = threadIdx.x >> 5;
  unsigned int slc = blockIdx.x >> logOfTPS;
  unsigned int localBId = blockIdx.x & (TbPerSlc -1);
  
  DTYPE tmp = 0, tmp_val;
      
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];
    unsigned int idx0 = dfbrIdx0[mappedSlc];
    unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];
    unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS;
    unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
    unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

    for (int fbr = fb_st + workId; fbr < fb_end && fbr < fbrPtr0[mappedSlc+1] ; fbr+=warpPerSlice){
      tmp_val = 0;
      
      for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

        unsigned int idx2 = dInds2[x];
        for(unsigned int r=laneId; r<R; r+=32) {
          tmp_val += vals[x] * dU2[idx2 * R + r];
        }
      }
      unsigned int idx1 = fbrIdx1[fbr];//dInds1[fbrPtr1[fbr]];
      for(unsigned int r=laneId; r<R; r+=32) {
        tmp += tmp_val * dU1[idx1 * R + r] ;
          // // atomicAdd(&dU0[idx0 * R + r], tmp);
      }
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      atomicAdd(&dU0[idx0 * R + r], tmp);
    }
  }
}

__global__ void mttkrp_HCSR_kernel_hvyBin_4D(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds3, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, DTYPE *dU3,
ITYPE mode, ITYPE R, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
  
  unsigned int laneId = threadIdx.x & 31;
  unsigned int workId = threadIdx.x >> 5;
  unsigned int slc = blockIdx.x >> logOfTPS;
  unsigned int localBId = blockIdx.x & (TbPerSlc -1);
  
  DTYPE outbuffer = 0, tmp_val = 0, outbuffer1 = 0;
      
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];
    unsigned int idx0 = dfbrIdx0[mappedSlc];
    unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];
    unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS;
    unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
    unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;

    for (int fbrS = fb_st; fbrS < fb_end && fbrS < fbrPtr0[mappedSlc+1] ; fbrS++){
      unsigned int idx1 = fbrIdx1[fbrS];// dInds1[fbrPtr1[fbr]];
      outbuffer1 = 0;

      for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){
        ITYPE idx2 = fbrIdx2[fbr];
        tmp_val = 0;
      
        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) {

          unsigned int idx3 = dInds3[x];
          for(unsigned int r=laneId; r<R; r+=32)
          tmp_val += vals[x] * dU3[idx3 * R + r];
        }
        for(unsigned int r=laneId; r<R; r+=32)
        outbuffer1 += tmp_val * dU2[idx2 * R + r] ;
      }
      for(unsigned int r=laneId; r<R; r+=32)
      outbuffer += outbuffer1 * dU1[idx1 * R + r] ;
    }
    for(unsigned int r=laneId; r<R; r+=32) {
      atomicAdd(&dU0[idx0 * R + r], outbuffer);
    }
  }
}

// CUDA kernels for B-CSF TTMC
__global__ void ttmc_HCSR_kernel_smllBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE S, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + tId);
  unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;
  unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5)
  DTYPE tmp = 0, tmp_val;
  
  
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc]; //i_ptr
    unsigned int idx0 = dfbrIdx0[mappedSlc]; //i
    int fb_st = fbrPtr0[mappedSlc];
    int fb_end = fbrPtr0[mappedSlc+1];

  // j is parallelized across warps
  //fbr = j_ptr
    for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
      tmp_val = 0;
    //serial k loop
    //x = k_ptr
      for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

        unsigned int idx2 = dInds2[x];    //k
      // s is parallelized across threads in a warp
        for(unsigned int s=laneId; s<S; s+=32) {
          tmp_val += vals[x] * dU2[idx2 * S + s];
        }
      }
    //idx1 = j
      unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
      for(unsigned int r=0; r<R; ++r) {
        for(unsigned int s=laneId; s<S; s+=32) {
    //each thread should have a bufer of length R if  we want to touch global memory only once
      //   tmp += tmp_val * dU1[idx1 * R + r] ;     
          atomicAdd(&dU0[idx0 * R * S + r * S + s], tmp_val * dU1[idx1 * R + r]);
        }
      }
    }
  // for(unsigned int r=0; r<R; ++r) {  
  //   for(unsigned int s=laneId; s<S; s+=32) {  
  //     atomicAdd(&dU0[idx0 * R * S + r * S + s], tmp);      
  //   }
  // }
  }
}

__global__ void ttmc_HCSR_kernel_hvyBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE S, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
  
  unsigned int laneId = threadIdx.x & 31;
  unsigned int workId = threadIdx.x >> 5;
  unsigned int slc = blockIdx.x >> logOfTPS;
  unsigned int localBId = blockIdx.x & (TbPerSlc -1);
  
  // DTYPE tmp = 0, tmp_val;
          
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];
    unsigned int idx0 = dfbrIdx0[mappedSlc];
    unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];
    unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS;
    unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
    unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;
    fb_end = (fb_end > fbrPtr0[mappedSlc+1]) ? fbrPtr0[mappedSlc+1] : fb_end;

    extern __shared__ DTYPE buf[];
    for(int buf_idx = threadIdx.x; buf_idx < R * S; buf_idx += blockDim.x){
      buf[warpPerSlice * S + buf_idx] = 0.0;
    }
    
  // j is parallelized across warps
  //fbr = j_ptr
  for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
    // tmp_val = 0;
    for(int buf_idx = workId * S + laneId; buf_idx < (workId + 1)* S; buf_idx += 32){
      buf[buf_idx] = 0.0;
    }
    //serial k loop
    //x = k_ptr
    for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

      unsigned int idx2 = dInds2[x];    //k
      // s is parallelized across threads in a warp
      for(unsigned int s=laneId; s<S; s+=32) {
        buf[workId * S + s] += vals[x] * dU2[idx2 * S + s];
      }
    }
    //idx1 = j
    unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
    for(unsigned int r=0; r<R; ++r) {
      for(unsigned int s=laneId; s<S; s+=32) {
        atomicAdd(&buf[warpPerSlice * S + r * S + s], buf[workId * S + s] * dU1[idx1 * R + r]);
      }
    }
  }
  __syncthreads();

  for(unsigned int r=workId; r<R; r+= warpPerSlice) {  
    for(unsigned int s=laneId; s<S; s+=32) {  
      atomicAdd(&dU0[idx0 * R * S + r * S + s], buf[warpPerSlice * S + r * S + s]);      
    }
  }
  // for(unsigned int r=0; r<R; ++r) {  
  //   for(unsigned int s=laneId; s<S; s+=32) {  
  //     atomicAdd(&dU0[idx0 * R * S + r * S + s], tmp);      
  //   }
  // } 
  }
}

__global__ void ttmc_HCSR_kernel_hvyBin_4D(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds3, ITYPE *fbrPtr0,
  ITYPE *fbrPtr1, ITYPE *fbrIdx1, ITYPE *fbrPtr2, ITYPE *fbrIdx2, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2, DTYPE *dU3,
  ITYPE mode, ITYPE R, ITYPE S, ITYPE T, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){
  extern __shared__ DTYPE buf[];
    
  unsigned int laneId = threadIdx.x & 31;
  unsigned int workId = threadIdx.x >> 5;
  unsigned int slc = blockIdx.x >> logOfTPS;
  unsigned int localBId = blockIdx.x & (TbPerSlc -1);
  
  DTYPE outbuffer = 0, tmp_val = 0, outbuffer1 = 0;
      
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];// i_ptr
    unsigned int idx0 = dfbrIdx0[mappedSlc];// i
    unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];
    unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS;
    unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
    unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;
    fb_end = (fb_end > fbrPtr0[mappedSlc+1]) ? fbrPtr0[mappedSlc+1] : fb_end;

    //serial j loop
    for (int fbrS = fb_st; fbrS < fb_end && fbrS < fbrPtr0[mappedSlc+1] ; fbrS++){
      unsigned int idx1 = fbrIdx1[fbrS];// j
      // outbuffer1 = 0;
      for(int buf_idx = threadIdx.x; buf_idx <  S * T; buf_idx += blockDim.x){
        buf[buf_idx] = 0.0;
      }
      __syncthreads();

      // k_ptr parallelized across warps
      for (int fbr = fbrPtr1[fbrS] + workId; fbr < fbrPtr1[fbrS+1]; fbr+=warpPerSlice){ // k_ptr
        ITYPE idx2 = fbrIdx2[fbr];// k
        tmp_val = 0;
        for(int buf_idx = laneId; buf_idx <  T; buf_idx += 32){
          buf[S*T +  workId * T + buf_idx] = 0.0;
        }

        // serial l loop
        for(unsigned int x = fbrPtr2[fbr]; x < fbrPtr2[fbr+1]; ++x) { // l_ptr

          unsigned int idx3 = dInds3[x];// l
          // t parallelized across threads in a warp
          for(unsigned int t=laneId; t<T; t+=32)
            buf[S*T +  workId * T + t] += vals[x] * dU3[idx3 * T + t];
        }
        // __syncthreads();
        for(unsigned int s=0; s<S; ++s){
          for(unsigned int t=laneId; t<T; t+=32){
            atomicAdd(&buf[s*T + t], buf[S*T +  workId * T + t] * dU2[idx2 * S + s] );
          }
        }
        // outbuffer1 += buf[S*T +  workId * T + s] * dU2[idx2 * S + s] ;
      }
      __syncthreads();

      for(unsigned int r=workId; r<R; r+=warpPerSlice){
        for(unsigned int s=0; s<S; ++s){
          for(unsigned int t=laneId; t<T; t+=32){
            atomicAdd(&dU0[idx0 * R * S * T + r * S * T + s * T + t], buf[s*T + t] * dU1[idx1 * R + r] );
          }
        }
      }
      // outbuffer += outbuffer1 * dU1[idx1 * R + r] ;
    }
    // for(unsigned int r=laneId; r<R; r+=32) {
    //   atomicAdd(&dU0[idx0 * R + r], outbuffer);
    // }
  }
}

// Main B-CSF-TTMC GPU implementation
int TTMC_B_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
  // Allocate and memcpy GPU memory 
  ITYPE *dInds2, *dInds3, *dfbrPtr0, *dfbrIdx0, *dfbrPtr1, *dfbrIdx1, *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin, *dFbrLikeSlcInds;
  DTYPE *dVals;
  ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0,  dFbrIdxLoc =0, dBinLoc = 0, dFbrLoc2 =0;
  ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

  ITYPE mode0 = TiledX[0].modeOrder[0];
  ITYPE mode1 = TiledX[0].modeOrder[1];
  ITYPE mode2 = TiledX[0].modeOrder[2];
  ITYPE mode3 =((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

  for (int tile = 0; tile < Opt.nTile; ++tile){
    totNnz += TiledX[tile].totNnz;
    totSlcPtr += TiledX[tile].fbrPtr[0].size() ;
    totSlcIdx += TiledX[tile].fbrIdx[0].size() ;
    totFbrPtr += TiledX[tile].fbrPtr[1].size() ;
    totFbrIdx += TiledX[tile].fbrIdx[1].size() ;
    totFbrPtr2 += ((TiledX[tile].ndims == 4) ? TiledX[tile].fbrPtr[2].size() : 0) ;
  }

  double t0 = seconds();
  checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrPtr0, totSlcPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrIdx0, totSlcIdx * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dSlcMapperBin, totSlcPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrPtr1, totFbrPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrIdx1, totFbrIdx * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dFbrLikeSlcInds, totFbrIdx * sizeof(ITYPE)), 0);

  if(TiledX[0].ndims == 3)
  checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);

  if(TiledX[0].ndims == 4){
    checkCuda(cudaMalloc((void**) &dFbrIdx2, totFbrPtr2 * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dFbrPtr2, totFbrPtr2 * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dInds3, totNnz * sizeof(ITYPE)), 0);
  }

  // cuda memcopy for tiled parts
  for (int tile = 0; tile < Opt.nTile; ++tile){
    if(tile > 0) {
      dLoc += TiledX[tile-1].totNnz;
      dSlcLoc += TiledX[tile - 1].fbrPtr[0].size();
      dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size();
      dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
      dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
      dFbrLoc2 += ((TiledX[tile].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
    }

    checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr0 + dSlcLoc, &(TiledX[tile].fbrPtr[0][0]), TiledX[tile].fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx0 + dSlcIdxLoc, &(TiledX[tile].fbrIdx[0][0]), TiledX[tile].fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr1 + dFbrLoc, &(TiledX[tile].fbrPtr[1][0]), TiledX[tile].fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx1 + dFbrIdxLoc, &(TiledX[tile].fbrIdx[1][0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      
    if(Opt.impType == 14)
    checkCuda(cudaMemcpy(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[tile].fbrLikeSlcInds[0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
  
    if(TiledX[tile].ndims == 3)
    checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[TiledX[tile].modeOrder[2]][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

    if(TiledX[tile].ndims == 4){
      checkCuda(cudaMemcpy(dFbrPtr2 + dFbrLoc2, &(TiledX[tile].fbrPtr[2][0]), TiledX[tile].fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      checkCuda(cudaMemcpy(dFbrIdx2 + dFbrLoc2, &(TiledX[tile].fbrIdx[2][0]), TiledX[tile].fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      checkCuda(cudaMemcpy(dInds3 + dLoc, &(TiledX[tile].inds[mode3][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    }

    dBinLoc = 0;
    for (int bin = 0; bin < Opt.nBin; ++bin){
      if(bin > 0)
      dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
      checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(TiledX[tile].slcMapperBin[bin][0]), TiledX[tile].slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    }
  }
  float tnsMemcpyTime = seconds() - t0;

  t0 = seconds();
  // Matrices
  DTYPE *dU0, *dU1, *dU2, *dU3;
  // U0 will be output tensor of size I x R x S
  
  checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

  cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
  checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
  checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);

  float mtxMemcpyTime = seconds() - t0;
  
  if(Opt.verbose){
    cout << "Tensor memory copy time: " << tnsMemcpyTime * 1000 << " ms" << endl;
    cout << "Matrix memory copy time: " << mtxMemcpyTime * 1000 << " ms" << endl;
    cout << "Total memory allocation and copy time: " << (tnsMemcpyTime + mtxMemcpyTime) * 1000 << " ms" << endl;
  }
  
  if(TiledX[0].ndims == 4){
    checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
    checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
  }

  // BLOCK and GRID
  int BLOCKSIZE = 512;
  unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32);

  if(Opt.warpPerSlice * 32 > BLOCKSIZE){
    cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
    exit(0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t streams[Opt.nBin];
  float mili = 0, GPUTime = 0, CPUtimer = 0, allModeGPUTime = 0;

  int smallBinEndsAt = 5;

  // Warp per slice and threadblock per size 
  int *warpPerSlc = new int[Opt.nBin];
  int *logOfWarpPerSlc = new int[Opt.nBin];
  int *TbPerSlc = new int[Opt.nBin];
  int *logOfTbPerSlc = new int[Opt.nBin];

  for (int bin = 0; bin < Opt.nBin ; ++bin){
    TbPerSlc[bin] = 1;
    warpPerSlc[bin] = ((bin > 0) ? 2 << (bin - 1) : 1);
      
    if(warpPerSlc[bin] > 16)
    warpPerSlc[bin] = 16;

    logOfWarpPerSlc[bin] = log2(warpPerSlc[bin]);

    TbPerSlc[bin] = 1;
    logOfTbPerSlc[bin] = 0;
      
    if (bin >= smallBinEndsAt){
      TbPerSlc[bin] = 1 << (bin - smallBinEndsAt + 1);
      if(TbPerSlc[bin] > 32) TbPerSlc[bin] = 32;
      logOfTbPerSlc[bin] = log2(TbPerSlc[bin]);
      warpPerSlc[bin] = 16;
      logOfWarpPerSlc[bin] = 4;
    }
      
    if(Opt.verbose ){
      cout << "  Bin " << bin << ": " << warpPerSlc[bin] << " warps/slice, "
      << TbPerSlc[bin] << " threadblocks/slice" << endl;
    }
  }

  int slcPerTb = 1;
  dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

  for (int bin = 0; bin < Opt.nBin; ++bin)
  cudaStreamCreate(&streams[bin]);
  
  if(Opt.verbose)
  cout << "Created " << Opt.nBin << " CUDA streams for parallel execution" << endl;

  // MTTKRP on Opt.mode
  int MTTKRPmode = mode0;
      
  for (int tile = 0; tile < Opt.nTile; ++tile){
    dBinLoc = 0;
      
    if(tile > 0) {
      dLoc += TiledX[tile-1].totNnz;
      dSlcLoc += TiledX[tile - 1].fbrPtr[0].size();
      dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size();
      dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
      dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
      dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
    }

    BLOCKSIZE = 512;
    dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

    int smallBinEndsAt = 5;
    int slcPerTb = 0;

    double t0 = seconds();
    cuda_timer_start(start);
      
      // Process small bins
    for (int bin = 0; bin < Opt.nBin ; ++bin){
      ITYPE shSize = 0;
      if(bin < smallBinEndsAt){
        dBinLoc += ((bin > 0) ? TiledX[tile].slcMapperBin[bin-1].size() : 0);
        grid.x = ( TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

        if(Opt.verbose && TiledX[tile].slcMapperBin[bin].size() > 0)
        cout << "    Small bin " << bin << ": " << TiledX[tile].slcMapperBin[bin].size() << " slices, grid size: " << grid.x << endl;

        if(TiledX[0].ndims == 3){
          // shSize = warpPerSlc[bin] * Opt.S * sizeof(DTYPE) + Opt.R * Opt.S * sizeof(DTYPE);
          ttmc_HCSR_kernel_smllBin<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
            dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(),
            dU0, dU1, dU2, Opt.mode, Opt.R, Opt.S, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]);
        }
          // else
        // ttmc_HCSR_kernel_smllBin_4D<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
        // dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(), 
        // dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]); 
      }
      // Processing heavy bin
      else{
        dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
        grid.x = (TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
          
        if(Opt.verbose && TiledX[tile].slcMapperBin[bin].size() > 0)
        cout << "    Heavy bin " << bin << ": " << TiledX[tile].slcMapperBin[bin].size() << " slices, grid size: " << grid.x << endl;
          
        if(TiledX[0].ndims == 3){
          shSize = warpPerSlc[bin] * Opt.S * sizeof(DTYPE) + Opt.R * Opt.S * sizeof(DTYPE);
          ttmc_HCSR_kernel_hvyBin<<<grid, block, shSize, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
          dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(),
          dU0, dU1, dU2, Opt.mode, Opt.R, Opt.S, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]);
        }
        else{
          shSize = Opt.S * Opt.R3 * sizeof(DTYPE) + warpPerSlc[bin] * Opt.R3 * sizeof(DTYPE);
          ttmc_HCSR_kernel_hvyBin_4D<<<grid, block, shSize, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc, 
          dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(), 
          dU0, dU1, dU2, dU3, Opt.mode, Opt.R, Opt.S, Opt.R3, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]); 
        }
      }
    }
  
    cuda_timer_stop(start, stop, mili);
    CPUtimer += seconds() - t0;
    GPUTime += mili;

    if(Opt.verbose){
      cout << "Tile: " << tile << " - time: " << mili << " ms";
      cout << " nnz: " << TiledX[tile].totNnz << " nFibers: "
      << TiledX[tile].fbrPtr[1].size() << " nSlc " << TiledX[tile].fbrIdx[0].size();
      cout << endl;
    }
  }
  
  allModeGPUTime += GPUTime;
  cout << "B-CSF-GPU-mode " << MTTKRPmode <<" :" << GPUTime << " ms" << endl;

  for (int bin = 0; bin < Opt.nBin; ++bin)
  cudaStreamDestroy(streams[bin]);

  // check correctness
  checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);

  cudaFree(dVals);
  cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
  cudaFree(dfbrIdx0); cudaFree(dInds2); cudaFree(dInds3);
  cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
  cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);
  cudaFree(dFbrLikeSlcInds);

  return 0;
}

// Main B-CSF-MTTKRP GPU implementation
int MTTKRP_B_HCSR_GPU(TiledTensor *TiledX, Matrix *U, const Options &Opt){
  /* Allocate and memcpy GPU memory */
  ITYPE *dInds2, *dInds3, *dfbrPtr0, *dfbrIdx0, *dfbrPtr1, *dfbrIdx1, *dFbrPtr2, *dFbrIdx2, *dSlcMapperBin, *dFbrLikeSlcInds;
  DTYPE *dVals;
  ITYPE dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0, dFbrLoc =0,  dFbrIdxLoc =0, dBinLoc = 0, dFbrLoc2 =0;
  ITYPE totNnz = 0, totSlcPtr = 0, totSlcIdx = 0, totFbrPtr = 0, totFbrIdx = 0, totFbrPtr2 = 0;

  ITYPE mode0 = TiledX[0].modeOrder[0];
  ITYPE mode1 = TiledX[0].modeOrder[1];
  ITYPE mode2 = TiledX[0].modeOrder[2];
  ITYPE mode3 =((TiledX[0].ndims == 4) ? TiledX[0].modeOrder[3] : 0) ;

  for (int tile = 0; tile < Opt.nTile; ++tile){
    totNnz += TiledX[tile].totNnz;
    totSlcPtr += TiledX[tile].fbrPtr[0].size() ;
    totSlcIdx += TiledX[tile].fbrIdx[0].size() ;
    totFbrPtr += TiledX[tile].fbrPtr[1].size() ;
    totFbrIdx += TiledX[tile].fbrIdx[1].size() ;
    totFbrPtr2 += ((TiledX[tile].ndims == 4) ? TiledX[tile].fbrPtr[2].size() : 0) ;
  }

  double t0 = seconds();
  checkCuda(cudaMalloc((void**) &dVals, totNnz * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrPtr0, totSlcPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrIdx0, totSlcIdx * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dSlcMapperBin, totSlcPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrPtr1, totFbrPtr * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dfbrIdx1, totFbrIdx * sizeof(ITYPE)), 0);
  checkCuda(cudaMalloc((void**) &dFbrLikeSlcInds, totFbrIdx * sizeof(ITYPE)), 0);

  if(TiledX[0].ndims == 3)
  checkCuda(cudaMalloc((void**) &dInds2, totNnz * sizeof(ITYPE)), 0);

  if(TiledX[0].ndims == 4){
    checkCuda(cudaMalloc((void**) &dFbrIdx2, totFbrPtr2 * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dFbrPtr2, totFbrPtr2 * sizeof(ITYPE)), 0);
    checkCuda(cudaMalloc((void**) &dInds3, totNnz * sizeof(ITYPE)), 0);
  }

  /* cuda memcopy for tiled parts*/
  for (int tile = 0; tile < Opt.nTile; ++tile){
    if(tile > 0) {
      dLoc += TiledX[tile-1].totNnz;
      dSlcLoc += TiledX[tile - 1].fbrPtr[0].size();
      dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size();
      dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
      dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
      dFbrLoc2 += ((TiledX[tile].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
    }

    checkCuda(cudaMemcpy(dVals + dLoc, &(TiledX[tile].vals[0]), TiledX[tile].totNnz * sizeof(DTYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr0 + dSlcLoc, &(TiledX[tile].fbrPtr[0][0]), TiledX[tile].fbrPtr[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx0 + dSlcIdxLoc, &(TiledX[tile].fbrIdx[0][0]), TiledX[tile].fbrIdx[0].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrPtr1 + dFbrLoc, &(TiledX[tile].fbrPtr[1][0]), TiledX[tile].fbrPtr[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    checkCuda(cudaMemcpy(dfbrIdx1 + dFbrIdxLoc, &(TiledX[tile].fbrIdx[1][0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      
    if(Opt.impType == 14)
    checkCuda(cudaMemcpy(dFbrLikeSlcInds + dFbrIdxLoc, &(TiledX[tile].fbrLikeSlcInds[0]), TiledX[tile].fbrIdx[1].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
  
    if(TiledX[tile].ndims == 3)
    checkCuda(cudaMemcpy(dInds2 + dLoc, &(TiledX[tile].inds[TiledX[tile].modeOrder[2]][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);

    if(TiledX[tile].ndims == 4){
      checkCuda(cudaMemcpy(dFbrPtr2 + dFbrLoc2, &(TiledX[tile].fbrPtr[2][0]), TiledX[tile].fbrPtr[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      checkCuda(cudaMemcpy(dFbrIdx2 + dFbrLoc2, &(TiledX[tile].fbrIdx[2][0]), TiledX[tile].fbrIdx[2].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
      checkCuda(cudaMemcpy(dInds3 + dLoc, &(TiledX[tile].inds[mode3][0]), TiledX[tile].totNnz * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    }

    dBinLoc = 0;
    for (int bin = 0; bin < Opt.nBin; ++bin){
      if(bin > 0)
      dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
      checkCuda(cudaMemcpy(dSlcMapperBin + dSlcIdxLoc + dBinLoc, &(TiledX[tile].slcMapperBin[bin][0]), TiledX[tile].slcMapperBin[bin].size() * sizeof(ITYPE),cudaMemcpyHostToDevice), 0);
    }
  }
  float tnsMemcpyTime = seconds() - t0;

  t0 = seconds();
  // Matrices
  DTYPE *dU0, *dU1, *dU2, *dU3;
  checkCuda(cudaMalloc((void**) &dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dU1, U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE)), 0);
  checkCuda(cudaMalloc((void**) &dU2, U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE)), 0);

  cudaMemset(dU0, 0,  U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE));
  checkCuda(cudaMemcpy(dU1, &(U[mode1].vals[0]), U[mode1].nRows * U[mode1].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
  checkCuda(cudaMemcpy(dU2, &(U[mode2].vals[0]), U[mode2].nRows * U[mode2].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);

  float mtxMemcpyTime = seconds() - t0;
  
  if(Opt.verbose){
    cout << "Tensor memory copy time: " << tnsMemcpyTime * 1000 << " ms" << endl;
    cout << "Matrix memory copy time: " << mtxMemcpyTime * 1000 << " ms" << endl;
    cout << "Total memory allocation and copy time: " << (tnsMemcpyTime + mtxMemcpyTime) * 1000 << " ms" << endl;
  }
  
  if(TiledX[0].ndims == 4){
    checkCuda(cudaMalloc((void**) &dU3, U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE)), 0);
    checkCuda(cudaMemcpy(dU3, &(U[mode3].vals[0]), U[mode3].nRows * U[mode3].nCols * sizeof(DTYPE), cudaMemcpyHostToDevice), 0);
  }

  // BLOCK and GRID
  int BLOCKSIZE = 512;
  unsigned int rowInATB = BLOCKSIZE / (Opt.warpPerSlice*32);

  if(Opt.warpPerSlice * 32 > BLOCKSIZE){
    cout << "BLOCKSIZE is smaller than work per slice! Increase BLOCKSIZE." << endl;
    exit(0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t streams[Opt.nBin];
  float mili = 0, GPUTime = 0, CPUtimer = 0, allModeGPUTime = 0;

  int smallBinEndsAt = 5;

  /* Warp per slice and threadblock per size */
  int *warpPerSlc = new int[Opt.nBin];
  int *logOfWarpPerSlc = new int[Opt.nBin];
  int *TbPerSlc = new int[Opt.nBin];
  int *logOfTbPerSlc = new int[Opt.nBin];

  for (int bin = 0; bin < Opt.nBin ; ++bin){
    TbPerSlc[bin] = 1;
    warpPerSlc[bin] = ((bin > 0) ? 2 << (bin - 1) : 1);
      
    if(warpPerSlc[bin] > 16)
    warpPerSlc[bin] = 16;

    logOfWarpPerSlc[bin] = log2(warpPerSlc[bin]);

    TbPerSlc[bin] = 1;
    logOfTbPerSlc[bin] = 0;
      
    if (bin >= smallBinEndsAt){
      TbPerSlc[bin] = 1 << (bin - smallBinEndsAt + 1);
      if(TbPerSlc[bin] > 32) TbPerSlc[bin] = 32;
      logOfTbPerSlc[bin] = log2(TbPerSlc[bin]);
      warpPerSlc[bin] = 16;
      logOfWarpPerSlc[bin] = 4;
    }
      
    if(Opt.verbose ){
      cout << "  Bin " << bin << ": " << warpPerSlc[bin] << " warps/slice, "
      << TbPerSlc[bin] << " threadblocks/slice" << endl;
    }
  }

  int slcPerTb = 1;
  dLoc = 0, dSlcLoc = 0, dSlcIdxLoc = 0; dFbrLoc =0, dFbrIdxLoc = 0, dFbrLoc2= 0;

  for (int bin = 0; bin < Opt.nBin; ++bin)
  cudaStreamCreate(&streams[bin]);
  
  if(Opt.verbose)
  cout << "Created " << Opt.nBin << " CUDA streams for parallel execution" << endl;

  /*MTTKRP on Opt.mode*/
  int MTTKRPmode = mode0;
      
  for (int tile = 0; tile < Opt.nTile; ++tile){
    dBinLoc = 0;
      
    if(tile > 0) {
      dLoc += TiledX[tile-1].totNnz;
      dSlcLoc += TiledX[tile - 1].fbrPtr[0].size();
      dSlcIdxLoc += TiledX[tile - 1].fbrIdx[0].size();
      dFbrLoc += TiledX[tile - 1].fbrPtr[1].size();
      dFbrIdxLoc += TiledX[tile - 1].fbrIdx[1].size();
      dFbrLoc2 += ((TiledX[0].ndims == 4) ? TiledX[tile - 1].fbrPtr[2].size() : 0) ;
    }

    BLOCKSIZE = 512;
    dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);

    int smallBinEndsAt = 5;
    int slcPerTb = 0;

    double t0 = seconds();
    cuda_timer_start(start);
      
      // Process small bins
    for (int bin = 0; bin < Opt.nBin ; ++bin){
      if(bin < smallBinEndsAt){
        ITYPE shSize = 0;
        dBinLoc += ((bin > 0) ? TiledX[tile].slcMapperBin[bin-1].size() : 0);
        grid.x = ( TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;

        if(Opt.verbose && TiledX[tile].slcMapperBin[bin].size() > 0)
        cout << "    Small bin " << bin << ": " << TiledX[tile].slcMapperBin[bin].size() << " slices, grid size: " << grid.x << endl;

        if(TiledX[0].ndims == 3)
        mttkrp_HCSR_kernel_smllBin<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
        dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(),
        dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]);
        else
        mttkrp_HCSR_kernel_smllBin_4D<<<grid, block, shSize , streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
        dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(),
        dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin], TbPerSlc[bin], logOfTbPerSlc[bin]);
      }
      // Processing heavy bin
      else{
        dBinLoc += TiledX[tile].slcMapperBin[bin-1].size();
        grid.x = (TbPerSlc[bin] * warpPerSlc[bin] * 32 * TiledX[tile].slcMapperBin[bin].size() + BLOCKSIZE - 1) / BLOCKSIZE;
          
        if(Opt.verbose && TiledX[tile].slcMapperBin[bin].size() > 0)
        cout << "    Heavy bin " << bin << ": " << TiledX[tile].slcMapperBin[bin].size() << " slices, grid size: " << grid.x << endl;
          
        if(TiledX[0].ndims == 3)
        mttkrp_HCSR_kernel_hvyBin<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
        dInds2 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc,  dfbrIdx1 + dFbrLoc, TiledX[tile].slcMapperBin[bin].size(),
        dU0, dU1, dU2, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]);
        else
        mttkrp_HCSR_kernel_hvyBin_4D<<<grid, block, 0, streams[bin]>>>(dVals + dLoc, dfbrIdx0 + dSlcIdxLoc, dSlcMapperBin + dSlcIdxLoc + dBinLoc,
        dInds3 + dLoc, dfbrPtr0 + dSlcLoc, dfbrPtr1 + dFbrLoc, dfbrIdx1 + dFbrIdxLoc, dFbrPtr2 + dFbrLoc2, dFbrIdx2 + dFbrLoc2, TiledX[tile].slcMapperBin[bin].size(),
        dU0, dU1, dU2, dU3, Opt.mode, Opt.R, warpPerSlc[bin], logOfWarpPerSlc[bin],  TbPerSlc[bin], logOfTbPerSlc[bin]);
      }
    }
  
    cuda_timer_stop(start, stop, mili);
    CPUtimer += seconds() - t0;
    GPUTime += mili;

    if(Opt.verbose){
      cout << "Tile: " << tile << " - time: " << mili << " ms";
      cout << " nnz: " << TiledX[tile].totNnz << " nFibers: "
      << TiledX[tile].fbrPtr[1].size() << " nSlc " << TiledX[tile].fbrIdx[0].size();
      cout << endl;
    }
  }
  
  allModeGPUTime += GPUTime;
  cout << "B-CSF-GPU-mode " << MTTKRPmode <<" :" << GPUTime << " ms" << endl;

  for (int bin = 0; bin < Opt.nBin; ++bin)
  cudaStreamDestroy(streams[bin]);

  // check correctness
  checkCuda(cudaMemcpy(&U[mode0].vals[0], dU0, U[mode0].nRows * U[mode0].nCols * sizeof(DTYPE), cudaMemcpyDeviceToHost), 0);

  cudaFree(dVals);
  cudaFree(dU0); cudaFree(dU1); cudaFree(dU2); cudaFree(dU3);
  cudaFree(dfbrIdx0); cudaFree(dInds2); cudaFree(dInds3);
  cudaFree(dfbrIdx0); cudaFree(dfbrIdx1); cudaFree(dFbrIdx2);
  cudaFree(dfbrPtr0); cudaFree(dfbrPtr1); cudaFree(dFbrPtr2);
  cudaFree(dFbrLikeSlcInds);

  return 0;
}

// Main function
int main(int argc, char* argv[]){
  Options Opt = parse_cmd_options(argc, argv);

  Tensor X;
  load_tensor(X, Opt);
  sort_COOtensor(X);
  
  TiledTensor TiledX[Opt.nTile];
    
  Matrix *U = new Matrix[X.ndims];
  create_mats(X, U, Opt, false);
  randomize_mats(X, U, Opt);
  zero_mat(X, U, Opt.mode);

  if(Opt.verbose){
    if(Opt.isTTMC){
      cout << endl << "Starting TTMC..." << endl;
    }
    else{
      cout << endl << "Starting MTTKRP..." << endl;
    }
    cout << "Tensor dimensions: ";
    for(int i = 0; i < X.ndims; i++){
      cout << X.dims[i];
      if(i < X.ndims-1) cout << " x ";
    }
    cout << endl;
    cout << "Total non-zeros: " << X.totNnz << endl;
    cout << "Rank R: " << Opt.R << endl;
    cout << "Rank S: " << Opt.S << endl;
    cout << "Rank R3: " << Opt.R3 << endl;
    cout << "Mode: " << Opt.mode << endl;
    cout << "Number of tiles: " << Opt.nTile << endl;
    // cout << "Warp per slice: " << Opt.warpPerSlice << endl;
    cout << "Number of bins: " << Opt.nBin << endl;
    // cout << "TTMC mode: " << (Opt.isTTMC ? "enabled" : "disabled") << endl;
  }
  
  // B-CSF implementation (type 8)
  if(Opt.impType == 8){
    create_HCSR(X, Opt);

    int tilingMode = X.modeOrder[X.ndims -1];

      // make tile fit in shared
    if(Opt.impType == 9){
      Opt.tileSize = 192;
      Opt.nTile = (X.dims[tilingMode] + Opt.tileSize - 1)/Opt.tileSize;
    }
    else
    Opt.tileSize = (X.dims[tilingMode] + Opt.nTile - 1)/Opt.nTile;
      
    if(Opt.nTile > X.dims[tilingMode]){
      cout << "Number of tiles ("<< Opt.nTile << ") should be as minimum as K's dimension (" << X.dims[tilingMode]  << "). Exiting."<< endl ;
      exit(0);
    }

      // split X into tiles based on K indices
    if(Opt.verbose)
    cout << "Creating tiles with tile size: " << Opt.tileSize << endl;
    make_KTiling(X, TiledX, Opt);
      
      // create HCSR for each tile
    if(Opt.verbose)
    cout << "Creating HCSR for tiles..." << endl;
    for (int tile = 0; tile < Opt.nTile; ++tile){
      if(TiledX[tile].totNnz > 0){
        create_TiledHCSR(TiledX, Opt, tile);
        if(Opt.verbose)
        cout << "  Tile " << tile << ": " << TiledX[tile].totNnz << " nnz, "
        << TiledX[tile].fbrPtr[0].size() << " slices, "
        << TiledX[tile].fbrPtr[1].size() << " fibers" << endl;
      }
    }

      // Split tiles into bins according to nnz in slice
    if(Opt.verbose)
    cout << "Creating bins for load balancing..." << endl;
    for (int tile = 0; tile < Opt.nTile; ++tile){
      if(TiledX[tile].totNnz > 0)
      make_TiledBin(TiledX, Opt, tile);
    }

    if(Opt.verbose)
    cout << "Sorted mode: " << X.modeOrder[0] << " " << X.modeOrder[1] << " " <<X.modeOrder[2] << endl;
    if(Opt.verbose)
    cout << "Starting GPU computation..." << endl;
      
    if(Opt.isTTMC){
      TTMC_B_HCSR_GPU(TiledX, U, Opt);
    }
    else{
      MTTKRP_B_HCSR_GPU(TiledX, U, Opt);
    }
    if(Opt.verbose)
    cout << "GPU computation completed successfully!" << endl;
  }
  else {
    cout << "This implementation is for B-CSF (type 8) only. Use -t 8 to run B-CSF." << endl;
    cout << "Available types:" << endl;
    cout << "  1: COO CPU" << endl;
    cout << "  2: HCSR CPU" << endl;
    cout << "  3: COO GPU" << endl;
    cout << "  4: HCSR GPU" << endl;
    cout << "  8: B-CSF (this implementation)" << endl;
    cout << "  10: HB-CSF" << endl;
  }

  // Correctness checking
  if(Opt.correctness){
    if (Opt.impType == 1) {
      cout << "Already running COO seq on CPU!" << endl;
      exit(0);
    }
      
    int mode = Opt.mode;
    int nr = U[mode].nRows;
    int nc = U[mode].nCols;
    DTYPE *out = (DTYPE*)malloc(nr * nc * sizeof(DTYPE));
    memcpy(out, U[mode].vals, nr*nc * sizeof(DTYPE));
      
    if(Opt.verbose){
      cout << "B-CSF result matrix (first 5x5 elements):" << endl;
      print_matrix(U, mode);
    }

      // Reset matrices and run COO CPU for comparison
    // randomize_mats(X, U, Opt);
    zero_mat(X, U, mode);

    if(Opt.verbose)
    cout << "Running COO CPU for correctness check on mode " << mode << endl;
      
    double t0 = seconds();
    if(Opt.isTTMC){
      (X.ndims == 3) ? TTMC_COO_CPU(X, U, Opt) : TTMC_COO_CPU_4D(X, U, Opt);
    }
    else{
      ((X.ndims == 3) ? MTTKRP_COO_CPU(X, U, Opt) : MTTKRP_COO_CPU_4D(X, U, Opt));
    }
    double cpu_time = seconds() - t0;
      
    if(Opt.verbose){
      cout << "COO CPU computation time: " << cpu_time * 1000 << " ms" << endl;
      cout << "COO CPU result matrix (first 5x5 elements):" << endl;
      print_matrix(U, mode);
    }
      
      // Compare results
    correctness_check(out, U[mode].vals, nr, nc);
      
    free(out);
  }

  if(!Opt.outFileName.empty()){
    if(Opt.verbose)
    cout << "Writing output to: " << Opt.outFileName << endl;
    write_output(U, Opt.mode, Opt.outFileName);
  }

  // Cleanup
  if(Opt.verbose)
  cout << "Cleaning up memory..." << endl;
  for (int m = 0; m < X.ndims; ++m){
    free(U[m].vals);
  }
  delete[] U;
  delete[] X.dims;

  if(Opt.verbose){
    if(Opt.isTTMC){
      cout << "TTMC computation completed successfully!" << endl;
    }
    else{
      cout << "MTTKRP computation completed successfully!" << endl;
    }
  }

  return 0;
}

//with buffer
/*
__global__ void ttmc_HCSR_kernel_smllBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE S, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int LogOfTPS){
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + tId);
  unsigned int workId = (tId & ((1 << (5 + logOfWPC)) - 1)) >> 5;
  unsigned int slc = gId >> (5 + logOfWPC); // 5: minimum 1 WARP (2^5)
//DTYPE tmp = 0, tmp_val;

  if(slc < nSlices)
  {
    extern __shared__ DTYPE buf[];
    unsigned int mappedSlc = dSlcMapperBin[slc]; //i_ptr
    unsigned int idx0 = dfbrIdx0[mappedSlc]; //i
    int fb_st = fbrPtr0[mappedSlc];
    int fb_end = fbrPtr0[mappedSlc+1];

    for(int buf_idx = threadIdx.x; buf_idx < R * S; buf_idx += blockDim.x){
      buf[warpPerSlice * S + buf_idx] = 0.0;
    }
    __syncthreads();
  // j is parallelized across warps
  //fbr = j_ptr
    for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
  // tmp_val = 0;
      for(int buf_idx = workId * S + laneId; buf_idx < (workId + 1)* S; buf_idx += 32){
        buf[buf_idx] = 0.0;
      }
  //serial k loop
  //x = k_ptr
      for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

        unsigned int idx2 = dInds2[x];    //k
    // s is parallelized across threads in a warp
        for(unsigned int s=laneId; s<S; s+=32) {
      // tmp_val += vals[x] * dU2[idx2 * S + s]; 
          buf[workId * S + s] += vals[x] * dU2[idx2 * S + s];
        }
      }
  //idx1 = j
      unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
      for(unsigned int r=0; r<R; ++r) {
        for(unsigned int s=laneId; s<S; s+=32) {
      //each thread should have a bufer of length R if  we want to touch global memory only once
    //   tmp += tmp_val * dU1[idx1 * R + r] ;     
      // atomicAdd(&dU0[idx0 * R * S + r * S + s], tmp_val * dU1[idx1 * R + r]); 
          atomicAdd(&buf[warpPerSlice * S + r * S + s], buf[workId * S + s] * dU1[idx1 * R + r]);
        }
      }
    }
    __syncthreads();

    for(unsigned int r_offset=0; r_offset<R; r_offset+= warpPerSlice) {
      unsigned int r = r_offset + workId;
      if(r < R){
        for(unsigned int s=laneId; s<S; s+=32) {
          atomicAdd(&dU0[idx0 * R * S + r * S + s], buf[warpPerSlice * S + r * S + s]);
        }
      }
    }
  }
}

__global__ void ttmc_HCSR_kernel_hvyBin(DTYPE * vals, ITYPE *dfbrIdx0, ITYPE *dSlcMapperBin, ITYPE *dInds2, ITYPE *fbrPtr0,
ITYPE *fbrPtr1, ITYPE *fbrIdx1, unsigned int nSlices, DTYPE *dU0, DTYPE * dU1, DTYPE *dU2,
ITYPE mode, ITYPE R, ITYPE S, ITYPE warpPerSlice, int logOfWPC, int TbPerSlc, int logOfTPS){

  unsigned int laneId = threadIdx.x & 31;
  unsigned int workId = threadIdx.x >> 5;
  unsigned int slc = blockIdx.x >> logOfTPS;
  unsigned int localBId = blockIdx.x & (TbPerSlc -1);

//   DTYPE tmp = 0, tmp_val;
        
  if(slc < nSlices){
    unsigned int mappedSlc = dSlcMapperBin[slc];
    unsigned int idx0 = dfbrIdx0[mappedSlc];
    unsigned int nFbr = fbrPtr0[mappedSlc+1] - fbrPtr0[mappedSlc];
    unsigned int fbrPerTb = (nFbr + TbPerSlc - 1 ) >> logOfTPS;
    unsigned int fb_st = fbrPtr0[mappedSlc] + localBId * fbrPerTb ;
    unsigned int fb_end = fbrPtr0[mappedSlc] + (localBId + 1) * fbrPerTb ;
    fb_end = (fb_end > fbrPtr0[mappedSlc+1]) ? fbrPtr0[mappedSlc+1] : fb_end;

    extern __shared__ DTYPE buf[];
    for(int buf_idx = threadIdx.x; buf_idx < R * S; buf_idx += blockDim.x){
      buf[warpPerSlice * S + buf_idx] = 0.0;
    }

  // j is parallelized across warps
  //fbr = j_ptr
    for (int fbr = fb_st + workId; fbr < fb_end; fbr+=warpPerSlice){
  // tmp_val = 0;
      for(int buf_idx = workId * S + laneId; buf_idx < (workId + 1)* S; buf_idx += 32){
        buf[buf_idx] = 0.0;
      }
  //serial k loop
  //x = k_ptr
      for(unsigned int x = fbrPtr1[fbr]; x < fbrPtr1[fbr+1]; ++x) {

        unsigned int idx2 = dInds2[x];    //k
    // s is parallelized across threads in a warp
        for(unsigned int s=laneId; s<S; s+=32) {
      // tmp_val += vals[x] * dU2[idx2 * S + s]; 
          buf[workId * S + s] += vals[x] * dU2[idx2 * S + s];
        }
      }
  //idx1 = j
      unsigned int idx1 = fbrIdx1[fbr];// dInds1[fbrPtr1[fbr]];
      for(unsigned int r=0; r<R; ++r) {
        for(unsigned int s=laneId; s<S; s+=32) {
      // atomicAdd(&dU0[idx0 * R * S + r * S + s], tmp_val * dU1[idx1 * R + r]);     
          atomicAdd(&buf[warpPerSlice * S + r * S + s], buf[workId * S + s] * dU1[idx1 * R + r]);
        }
      }
    }
    __syncthreads();

    for(unsigned int r=workId; r<R; r+= warpPerSlice) {
      for(unsigned int s=laneId; s<S; s+=32) {
        atomicAdd(&dU0[idx0 * R * S + r * S + s], buf[warpPerSlice * S + r * S + s]);
      }
    }
  
  }
}
*/