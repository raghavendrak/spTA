# TTMC Method 1 on CPU [contains 5 'for' loops]

Here, we perform TTMC Contraction of input Tensor 'T' with matrix 'A' and 'B' resulting in output tensor 'O' using 5 'for' loops.

ttmc_method_1_cpu.cpp performs any TTMC contraction by taking csf_1.txt as input, it contains tensor elements in CSF Format and input_matrix_A.txt, input_matrix_B.txt contains the input matrices and in a 2D-Array Format.

The csf_1.txt file contains the non-zero elements of a tensor in CSF Format.

The input_matrix_A.txt file contains the matrix 'A' elements.

The input_matrix_B.txt file contains the matrix 'B' elements.

We should also give the contraction string, which we want to perform as input. (eg. : ijk,kr,js->isr)

The contraction is performed by linearising the input matrices 'A' and 'B'.

Finally, after the contraction, the outputs are stored to the array 'arr_O', which is mode-0 linearised version of the output tensor 'O'.

And then, the output array 'arr_O' is written to the output.txt file.



