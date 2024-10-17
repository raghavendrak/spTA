#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <cstring>  // Add this line to resolve 'memset' error

using namespace std;
using namespace std::chrono;

// Function to read a vector of integers from a string
vector<int64_t> readVectorInt(const string& line) {
    vector<int64_t> result;
    stringstream ss(line);
    int64_t val;
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
void readCSFTensor(const string& filename, int64_t*& mode_0_ptr, int64_t*& mode_0_idx, 
                   int64_t*& mode_1_ptr, int64_t*& mode_1_idx, int64_t*& mode_2_ptr, 
                   int64_t*& mode_2_idx, double*& values) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open CSF file: " + filename);
    }

    string line;

    // Read mode_0_ptr
    getline(file, line);
    auto temp_mode_0_ptr = readVectorInt(line);
    mode_0_ptr = new int64_t[temp_mode_0_ptr.size()];
    copy(temp_mode_0_ptr.begin(), temp_mode_0_ptr.end(), mode_0_ptr);

    // Read mode_0_idx
    getline(file, line);
    auto temp_mode_0_idx = readVectorInt(line);
    mode_0_idx = new int64_t[temp_mode_0_idx.size()];
    copy(temp_mode_0_idx.begin(), temp_mode_0_idx.end(), mode_0_idx);

    // Read mode_1_ptr
    getline(file, line);
    auto temp_mode_1_ptr = readVectorInt(line);
    mode_1_ptr = new int64_t[temp_mode_1_ptr.size()];
    copy(temp_mode_1_ptr.begin(), temp_mode_1_ptr.end(), mode_1_ptr);

    // Read mode_1_idx
    getline(file, line);
    auto temp_mode_1_idx = readVectorInt(line);
    mode_1_idx = new int64_t[temp_mode_1_idx.size()];
    copy(temp_mode_1_idx.begin(), temp_mode_1_idx.end(), mode_1_idx);

    // Read mode_2_ptr
    getline(file, line);
    auto temp_mode_2_ptr = readVectorInt(line);
    mode_2_ptr = new int64_t[temp_mode_2_ptr.size()];
    copy(temp_mode_2_ptr.begin(), temp_mode_2_ptr.end(), mode_2_ptr);

    // Read mode_2_idx
    getline(file, line);
    auto temp_mode_2_idx = readVectorInt(line);
    mode_2_idx = new int64_t[temp_mode_2_idx.size()];
    copy(temp_mode_2_idx.begin(), temp_mode_2_idx.end(), mode_2_idx);

    // Read values
    getline(file, line);
    auto temp_values = readVectorDouble(line);
    values = new double[temp_values.size()];
    copy(temp_values.begin(), temp_values.end(), values);

    file.close();
}

// Function to read a matrix from a file 
void readMatrix(const string& filename, int64_t& rows, int64_t& cols, double*& arr) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Unable to open matrix file: " + filename);
    }

    // Read the entire file into a vector
    arr = new double[rows * cols];
    double value;
    
    int64_t count = 0;
    while (file >> value) {
        if (count < rows * cols) {
            arr[count++] = value;
        } else {
            throw runtime_error("More values in the file than expected.");
        }
    }
    
    // Close the file
    file.close();

    if (count % cols != 0) {
        throw runtime_error("Mismatch between total number of elements and specified column count.");
    }
}

// Function to perform contraction based on the inputs using 5 for loops
void performContraction_1(const int64_t* mode_0_ptr, const int64_t* mode_0_idx,
                        const int64_t* mode_1_ptr, const int64_t* mode_1_idx,
                        const int64_t* mode_2_ptr, const int64_t* mode_2_idx,
                        const double* values, const double* arr_A, const double* arr_B,  
                        double*& arr_O, const int64_t arr_A_size, const int64_t arr_B_size, const int64_t arr_O_size, const int64_t contraction, 
                        const int64_t l, const int64_t m, const int64_t n, const int64_t f1, const int64_t f2) {
    if(contraction == 0){
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {

                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_A = j * f1 + r;
                            int64_t index_B = k * f2 + s;

                            // For mode-1 linearized output 
                            int64_t index_O = s * l * f1 + i * f1 + r;

                            // Perform contraction
                            arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];
                            
                        }
                    }
                }
            }
        }
    }
    if(contraction == 1){
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {

                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_A = i * f1 + r;
                            int64_t index_B = k * f2 + s;

                            // For mode-1 linearized output 
                            int64_t index_O = s * m * f1 + r * m + j;

                            // Perform contraction
                            arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];            
                        }
                    }
                }
            }
        }
    }
    if(contraction == 2){
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    // Iterate over the matrix dimensions 
                    for (int64_t r = 0; r < f1; ++r) {

                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A, B based on the contraction string
                            int64_t index_A = i * f1 + r;
                            int64_t index_B = j * f2 + s;

                            // For mode-1 linearized output
                            int64_t index_O = s * n * f1 + r * n + k;

                            // Perform contraction
                            arr_O[index_O] += value * arr_A[index_A] * arr_B[index_B];            
                        }
                    }
                }
            }
        }
    }
}



// Function to perform contraction based on the inputs using 4 for loops
void performContraction_2(const int64_t* mode_0_ptr, const int64_t* mode_0_idx,
                        const int64_t* mode_1_ptr, const int64_t* mode_1_idx,
                        const int64_t* mode_2_ptr, const int64_t* mode_2_idx,
                        const double* values, const double* arr_A, const double* arr_B,  
                        double*& arr_O, const int64_t arr_A_size, const int64_t arr_B_size, const int64_t arr_O_size, const int64_t contraction, 
                        const int64_t l, const int64_t m, const int64_t n, const int64_t f1, const int64_t f2) {
    

    if(contraction == 0){
        double* buffer = new double[f2];    // buffer for mode-s

        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                         // Index in the mode 'i'
            
            
            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF
                
                memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0
                
                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = k * f2 + s;

                        buffer[s] += value * arr_B[index_B];                        
                    }
                }

                for (int64_t r = 0; r < f1; ++r) {

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_A = j * f1 + r;

                        // For mode-1 linearized output
                        int64_t index_O = s * l * f1 + i * f1 + r;

                        // Perform contraction
                        arr_O[index_O] += buffer[s] * arr_A[index_A];              
                    }
                }
            }
        }

        delete [] buffer;
    }
    if(contraction == 1){
        double* buffer = new double[f2];    // buffer for mode-s
        
        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                         // Index in the mode 'i'

            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                     // Index for 'j' mode in CSF

                memset(buffer, 0, f2 * sizeof(double));             // Set the entire memory block to 0

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                 // Index for 'k' mode in CSF

                    double value = values[k_ptr];                  // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = k * f2 + s;

                        // Perform contraction
                        buffer[s] += value * arr_B[index_B];            
                    }
                }


                for (int64_t r = 0; r < f1; ++r) {

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices A, B based on the contraction string
                        int64_t index_A = i * f1 + r;
                        // int64_t index_B = k * f2 + s;

                        // For mode-1 linearized output 
                        int64_t index_O = s * m * f1 + r * m + j;

                        // Perform contraction
                        arr_O[index_O] += buffer[s] * arr_A[index_A];              
                    }
                }
            }
        }

        delete [] buffer;
    }
    if(contraction == 2){
        double* buffer = new double[n*f2];    // buffer for mode-k and mode-s
        int64_t* k_buffer = new int64_t[n];   // buffer for k-indices


        // Traverse through CSF tensor pointer and indices arrays for all modes
        for (int64_t i_ptr = 0; i_ptr < mode_0_ptr[1]; ++i_ptr) {
            int64_t i = mode_0_idx[i_ptr];                          // Index in the mode 'i'

            memset(buffer, 0, n * f2 * sizeof(double));             // Set the entire memory block to 0
            memset(k_buffer, 0, n * sizeof(int64_t));               // Set the entire memory block to 0
            for (int64_t j_ptr = mode_1_ptr[i_ptr]; j_ptr < mode_1_ptr[i_ptr + 1]; ++j_ptr) {
                int64_t j = mode_1_idx[j_ptr];                      // Index for 'j' mode in CSF

                for (int64_t k_ptr = mode_2_ptr[j_ptr]; k_ptr < mode_2_ptr[j_ptr + 1]; ++k_ptr) {
                    int64_t k = mode_2_idx[k_ptr];                  // Index for 'k' mode in CSF
                    k_buffer[k]++;

                    double value = values[k_ptr];                   // CSF value for the above i, j, k

                    for (int64_t s = 0; s < f2; ++s) {

                        // Compute linearized indices for matrices B based on the contraction string
                        int64_t index_B = j * f2 + s;

                        int64_t index_buf = k * f2 + s; 

                        buffer[index_buf] += value * arr_B[index_B];          
                    }
                }
            }


            for (int64_t z = 0; z < n ; ++z) {
                int64_t k = z;

                while(k_buffer[z] != 0){
                    for (int64_t r = 0; r < f1; ++r) {

                        for (int64_t s = 0; s < f2; ++s) {

                            // Compute linearized indices for matrices A based on the contraction string
                            int64_t index_A = i * f1 + r;

                            // For mode-1 linearized output 
                            int64_t index_O = s * n * f1 + r * n + k;

                            int64_t index_buf = k * f2 + s; 

                            arr_O[index_O] += buffer[index_buf] * arr_A[index_A] ;        
                        }
                    }

                    k_buffer[z]--;
                }
            }
        }

        delete [] buffer;
        delete [] k_buffer;
    }
}

/*
NOTE : Make sure to change the input matrix files as per the chosen contraction : 

For contraction "0" for ijk,jr,ks→irs, l = 3, m = 3, n = 5, f1 = 2, f2 = 2 : 

output1.txt : 0 0 2 2 4 4 0 0 2 2 4 4 
output2.txt : 0 0 2 2 4 4 0 0 2 2 4 4 

Your Contraction Choice : ijk,jr,ks→irs
Time taken by contraction: 2451 nanoseconds
Output written to output_1.txt
Time taken by contraction: 3237 nanoseconds
Output written to output_2.txt

For contraction "1" for ijk,ir,ks→rjs l = 3, m = 3, n = 5, f1 = 2, f2 = 2 : 

output1.txt : 0 5 1 0 5 1 0 5 1 0 5 1  
output2.txt : 0 5 1 0 5 1 0 5 1 0 5 1  

Your Contraction Choice : ijk,ir,ks→rjs
Time taken by contraction: 2306 nanoseconds
Output written to output_1.txt
Time taken by contraction: 2502 nanoseconds
Output written to output_2.txt


For contraction "2" for ijk,ir,js→rsk l = 3, m = 3, n = 5, f1 = 2, f2 = 2 : 

output1.txt : 0 2 2 1 1 0 2 2 1 1 0 2 2 1 1 0 2 2 1 1 
output2.txt : 0 2 2 1 1 0 2 2 1 1 0 2 2 1 1 0 2 2 1 1 

Your Contraction Choice : ijk,ir,js→rsk
Time taken by contraction: 2615 nanoseconds
Output written to output_1.txt
Time taken by contraction: 4708 nanoseconds
Output written to output_2.txt
*/



int main() {
    // Input tensor dimensions (l * m * n)
    int64_t l, m, n;
    cout << "Enter the dimensions of the input tensor (l * m * n): ";
    cin >> l >> m >> n;

    int64_t* mode_0_ptr = nullptr;
    int64_t* mode_0_idx = nullptr;
    int64_t* mode_1_ptr = nullptr;
    int64_t* mode_1_idx = nullptr;
    int64_t* mode_2_ptr = nullptr;
    int64_t* mode_2_idx = nullptr;
    double* values = nullptr;

    // Read the CSF tensor from csf_1.txt
    readCSFTensor("csf_1.txt", mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, 
                  mode_2_ptr, mode_2_idx, values);

    // Read input matrices A (n * f1) and B (m * f2)
    int64_t f1, f2;
    cout << "Enter the column dimension of input matrices A and B respectively (f1) and (f2): ";
    cin >> f1 >> f2;

    double* arr_A = nullptr;
    double* arr_B = nullptr;

    // Enter the choice of contraction
    int64_t contraction;
    cout << "Choose the contraction, \"0\" for ijk,jr,ks→irs, \"1\" for ijk,ir,ks→rjs, \"2\" for ijk,ir,js→rsk " << endl;
    cin >> contraction;
    if(contraction == 0){
        cout << "Your Contraction Choice : ijk,jr,ks→irs" << endl; 
    }
    else if(contraction == 1){
        cout << "Your Contraction Choice : ijk,ir,ks→rjs" << endl;
    }
    else if(contraction == 2){
        cout << "Your Contraction Choice : ijk,ir,js→rsk" << endl;
    }

    int64_t arr_A_rows = 0;
    int64_t arr_B_rows = 0;

    if(contraction == 0) {
        arr_A_rows = m;
        arr_B_rows = n;
    } else if(contraction == 1) {
        arr_A_rows = l;
        arr_B_rows = n;
    } else if(contraction == 2) {
        arr_A_rows = l;
        arr_B_rows = m;
    }

    readMatrix("input_matrix_A.txt", arr_A_rows, f1, arr_A);
    readMatrix("input_matrix_B.txt", arr_B_rows, f2, arr_B);

    int64_t arr_A_size = arr_A_rows * f1;
    int64_t arr_B_size = arr_B_rows * f2;

    int64_t output_sizes[3];
    output_sizes[0] = l * f1 * f2;
    output_sizes[1] = f1 * m * f2;
    output_sizes[2] = f1 * f2 * n;

    int64_t arr_0_size = output_sizes[contraction];

    double* arr_O = new double[arr_0_size](); // The () initializes the array to zero

    // Record start time
    auto start_1 = high_resolution_clock::now();

    // Performing TTMC contraction using 5 for loops
    performContraction_1(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                       values, arr_A, arr_B, arr_O, arr_A_size, arr_B_size, arr_0_size, contraction, l, m, n, f1, f2);

    // Record end time
    auto end_1 = high_resolution_clock::now();
    auto duration_1 = duration_cast<nanoseconds>(end_1 - start_1);
    cout << "Time taken by contraction: " << duration_1.count() << " nanoseconds" << endl;

    // Write the output array to output.txt
    ofstream output_file_1("output_1.txt");
    if (!output_file_1.is_open()) {
        throw runtime_error("Unable to open output file.");
    }

    for (int64_t i = 0; i < arr_0_size; i++) {
        output_file_1 << arr_O[i] << " ";
    }
    output_file_1.close();

    cout << "Output written to output_1.txt" << endl;

    delete[] arr_O;




    // Performing TTMC contraction using 4 for loops
    arr_O = new double[arr_0_size](); // The () initializes the array to zero

    // Record start time
    auto start_2 = high_resolution_clock::now();

    // Perform contraction using 4 for loops
    performContraction_2(mode_0_ptr, mode_0_idx, mode_1_ptr, mode_1_idx, mode_2_ptr, mode_2_idx, 
                       values, arr_A, arr_B, arr_O, arr_A_size, arr_B_size, arr_0_size, contraction, l, m, n, f1, f2);

    // Record end time
    auto end_2 = high_resolution_clock::now();
    auto duration_2 = duration_cast<nanoseconds>(end_2 - start_2);
    cout << "Time taken by contraction: " << duration_2.count() << " nanoseconds" << endl;

    // Write the output array to output.txt
    ofstream output_file_2("output_2.txt");
    if (!output_file_2.is_open()) {
        throw runtime_error("Unable to open output file.");
    }

    for (int64_t i = 0; i < arr_0_size; i++) {
        output_file_2 << arr_O[i] << " ";
    }
    output_file_2.close();

    cout << "Output written to output_2.txt" << endl;

    delete[] arr_O;




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

    return 0;
}

