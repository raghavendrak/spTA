#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>

using namespace std;

// Function to read a vector of integers from a string
vector<int> readVectorInt(const string& line) {
    vector<int> result;
    stringstream ss(line);
    int val;
    while (ss >> val) {
        result.push_back(val);
    }
    return result;
}

// Function to read a vector of doubles from a string
vector<double> readVectorDouble(const string& line) {
    vector<double> result;
    stringstream ss(line);
    double val;
    while (ss >> val) {
        result.push_back(val);
    }
    return result;
}


// Function to read a CSF tensor from the csf.txt file
void readCSFTensor(const string& filename, int*& mode_0_ptr, int*& mode_0_idx, 
                   int*& mode_1_ptr, int*& mode_1_idx, int*& mode_2_ptr, 
                   int*& mode_2_idx, double*& values) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open CSF file: " + filename);
    }

    string line;

    // Read mode_0_ptr
    getline(file, line);
    auto temp_mode_0_ptr = readVectorInt(line);
    mode_0_ptr = new int[temp_mode_0_ptr.size()];
    copy(temp_mode_0_ptr.begin(), temp_mode_0_ptr.end(), mode_0_ptr);

    // Read mode_0_idx
    getline(file, line);
    auto temp_mode_0_idx = readVectorInt(line);
    mode_0_idx = new int[temp_mode_0_idx.size()];
    copy(temp_mode_0_idx.begin(), temp_mode_0_idx.end(), mode_0_idx);

    // Read mode_1_ptr
    getline(file, line);
    auto temp_mode_1_ptr = readVectorInt(line);
    mode_1_ptr = new int[temp_mode_1_ptr.size()];
    copy(temp_mode_1_ptr.begin(), temp_mode_1_ptr.end(), mode_1_ptr);

    // Read mode_1_idx
    getline(file, line);
    auto temp_mode_1_idx = readVectorInt(line);
    mode_1_idx = new int[temp_mode_1_idx.size()];
    copy(temp_mode_1_idx.begin(), temp_mode_1_idx.end(), mode_1_idx);

    // Read mode_2_ptr
    getline(file, line);
    auto temp_mode_2_ptr = readVectorInt(line);
    mode_2_ptr = new int[temp_mode_2_ptr.size()];
    copy(temp_mode_2_ptr.begin(), temp_mode_2_ptr.end(), mode_2_ptr);

    // Read mode_2_idx
    getline(file, line);
    auto temp_mode_2_idx = readVectorInt(line);
    mode_2_idx = new int[temp_mode_2_idx.size()];
    copy(temp_mode_2_idx.begin(), temp_mode_2_idx.end(), mode_2_idx);

    // Read values
    getline(file, line);
    auto temp_values = readVectorDouble(line);
    values = new double[temp_values.size()];
    copy(temp_values.begin(), temp_values.end(), values);

    file.close();
}

// Function to read a matrix from a file 
void readMatrix(const string& filename, int& rows, int& cols, double*& arr) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open matrix file: " + filename);
    }

    // Read the entire file into a vector
    arr = new double[rows*cols];
    double value;
    
    int count = 0;
    while (file >> value) {
        if (count < rows * cols) {
            arr[count++] = value;
        } else {
            throw runtime_error("More values in the file than expected.");
        }
    }
    
    // Close the file
    file.close();

    // Total size = rows * cols
    if (count % cols != 0) {
        throw runtime_error("Mismatch between total number of elements and specified column count.");
    }
}

// Function to perform contraction based on the inputs
void performContraction(const int* mode_0_ptr, const int* mode_0_idx,
                        const int* mode_1_ptr, const int* mode_1_idx,
                        const int* mode_2_ptr, const int* mode_2_idx,
                        const double* values, const double* arr_A, 
                        const double* arr_B, unordered_map<char, int>& mp_char, 
                        unordered_map<char, int>& mp_dim, const string& contraction, 
                        double*& arr_O, const int arr_A_size, const int arr_B_size, const int arr_O_size) {
    
    // Traverse through CSF tensor pointer and indices arays for all modes
    for (int i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
        int i = mode_0_idx[i_ptr];                         // Index in the mode 'i'
        mp_char['i'] = i;                                  // Map char 'i' to current index in the contraction

        for (int j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
            int j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF
            mp_char['j'] = j;                              // Map char 'j' to current index in the contraction

            for (int k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                int k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF
                mp_char['k'] = k;                          // Map char 'k' to current index in the contraction

                double value = values[k_ptr];              // CSF value for the above i, j, k

                // Iterate over the matrix dimensions 
                for (int r = 0; r < mp_dim['r']; ++r) {
                    mp_char['r'] = r;                      // Map 'r' to the current index

                    for (int s = 0; s < mp_dim['s']; ++s) {
                        mp_char['s'] = s;                  // Map 's' to the current index

                        // Compute linearized indices for matrices A, B based on the contraction string
                        int index_A = mp_char[contraction[4]] * mp_dim[contraction[5]] + mp_char[contraction[5]];
                        int index_B = mp_char[contraction[7]] * mp_dim[contraction[8]] + mp_char[contraction[8]];

                        // For mode-2 linearised output [Here, for the contraction ijk,kr,js->isr, output tensor is of 'isr']:
                        // int index_O = mp_char[contraction[11]] * mp_dim[contraction[12]] * mp_dim[contraction[13]]
                        //             + mp_char[contraction[12]] * mp_dim[contraction[13]]
                        //             + mp_char[contraction[13]];

                        // For mode-0 linearised output [Here, for the contraction ijk,kr,js->isr, output tensor is of 'isr']:
                        int index_O = mp_char[contraction[13]] * mp_dim[contraction[11]] * mp_dim[contraction[12]]
                                    + mp_char[contraction[11]] * mp_dim[contraction[12]]
                                    + mp_char[contraction[12]];

                        // Debug print
                        // cout << "i : " << i << " " << "j : " << j << " " << "k : " << k << " " << "value : " << value << " " << "r : " << r << " s : " << s << endl;
                        // cout << "Index_A: " << index_A << ", Index_B: " << index_B << ", Index_O: " << index_O << endl;

                        // Perform contraction
                        if (index_O >= 0 && index_O < arr_O_size && 
                            index_A >= 0 && index_A < arr_A_size && 
                            index_B >= 0 && index_B < arr_B_size) {
                            arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];
                        } else {   // Debug print
                            cout << "Index out of bounds! A: " << index_A << ", B: " << index_B << ", O: " << index_O << endl;
                            cout << "mp_char[i] : " << mp_char[contraction[11]] << " mp_char[s] : " << mp_char[contraction[12]]  << " mp_char[r] : " << mp_char[contraction[13]] << endl;
                            cout << "mp_dim[s] : " << mp_dim[contraction[12]] << "mp_dim[r] : " << mp_dim[contraction[13]] << endl; 
                        }
                    }
                }
            }
        }
    }
}



int main() {
    // Input tensor dimensions (l * m * n)
    int l, m, n;
    cout << "Enter the dimensions of the input tensor (l * m * n): ";
    cin >> l >> m >> n;

    int* mode_0_ptr = nullptr;
    int* mode_0_idx = nullptr;
    int* mode_1_ptr = nullptr;
    int* mode_1_idx = nullptr;
    int* mode_2_ptr = nullptr;
    int* mode_2_idx = nullptr;
    double* values = nullptr;

    // Read the CSF tensor from csf_1.txt
    readCSFTensor("csf_1.txt", mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, 
                  mode_2_ptr, mode_2_idx, values);

    // Read input matrices A (n * f1) and B (m * f2)
    int f1, f2;
    cout << "Enter the column dimension of input matrices A and B respectively (f1) and (f2): ";
    cin >> f1 >> f2;

    double* arr_A = nullptr;
    double* arr_B = nullptr;

    readMatrix("input_matrix_A.txt", n, f1, arr_A);
    readMatrix("input_matrix_B.txt", m, f2, arr_B);

    int arr_A_size = n*f1;
    int arr_B_size = m*f2;
    

    // Create maps for character and dimensions
    unordered_map<char, int> mp_char, mp_dim;
    mp_dim['i'] = l;
    mp_dim['j'] = m;
    mp_dim['k'] = n;
    mp_dim['r'] = f1;
    mp_dim['s'] = f2;

    // Enter the contraction string (e.g., "ijk,kr,js->isr")
    string contraction;
    cout << "Enter the contraction you want to perform (e.g., ijk,kr,js->isr): ";
    cin >> contraction;
    cout << "Contraction String : " << contraction << endl;

    // Calculate the dimensions of the output tensor based on the contraction indices
    int output_size = mp_dim[contraction[11]] * mp_dim[contraction[12]] * mp_dim[contraction[13]];

    double* arr_O = new double[output_size]();                   // The () in new double[output_size]() initializes the array to zero

    // Perform contraction
    performContraction(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                       values, arr_A, arr_B, mp_char, mp_dim, contraction, arr_O, arr_A_size, arr_B_size, output_size);


    // Write the output array to output.txt
    ofstream output_file("output.txt");
    if (!output_file.is_open()) {
        throw runtime_error("Unable to open output file.");
    }

    for (int i = 0; i < output_size; i++) {
        double val = arr_O[i];
        output_file << val << " ";
    }
    output_file.close();

    cout << "Output written to output.txt" << endl;

    // Memory clean up
    delete[] mode_0_ptr;
    delete[] mode_0_idx;
    delete[] mode_1_ptr;
    delete[] mode_1_idx;
    delete[] mode_2_ptr;
    delete[] mode_2_idx;
    delete[] values;
    delete[] arr_A;
    delete[] arr_B;
    delete[] arr_O;

    return 0;
}
