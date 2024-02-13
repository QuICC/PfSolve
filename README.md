# PfSolve - Multi-API GPU runtime autotuned collection of kernels for matrix operations based on the platform for code generation used in the VkFFT library.

PfSolve currently contains the following algorithms:

- Bi-/Tri- diagonal matrix solvers based on a Parallel Cyclic Reduction + Thomas algorithm implemented with single-warp GPU programming approach. See: docs/pasc23_Tolmachev.pdf for reference. This code can be fused with banded matrix multiplications to achieve one step of Jones Worland algorithm from https://www.research-collection.ethz.ch/handle/20.500.11850/505302.
- Block/Copy/Scaling/dgbmv optimized BLAS routines. 
- Finite Difference solver as a demonstration of a single-warp GPU programming approach. See: https://www.youtube.com/watch?v=lHlFPqlOezo and docs/report_Tolmachev_2023.pdf for reference.
- Draft of a non-uniform FFT extension for VkFFT - single-warp runtime grid-optimization approach for kernel convolution stage of nuFFT. See docs/report_Tolmachev_2023.pdf for reference.

## Project structure

Project is designed similarly to VkFFT v1.3 and reuses a lot of code from it (section IV-B https://ieeexplore.ieee.org/document/10036080). The code reused is the union-container abstracted codelets for required math operators (vkFFT_MathUtils/pfSolve_MathUtils), application/plan/kernel-levels structure, API management.

Bi-/Tri- diagonal matrix solvers use pfSolve_PlanManagement/pfSolve_Plans/pfSolve_Plan_JonesWorland as a plan entry point, where configuration of code generator happens. Then kernel layout is given in pfSolve_CodeGen/pfSolve_KernelsLevel2/pfSolve_JonesWorland. This layout uses pfSolve_MatVecMul and pfSolve_TridiagonalSolve routines from Level 1, which are the respective algorithms codes. They use math functions from pfSolve_MathUtils file and memory operations defined in pfSolve_KernelsLevel0.

## Kernels reuse and specialization constants

pfSolve reduces the number of compiled files by caching the generated binaries for further reuse and using the specialization constants for providing values that have low impact on kernel optimization at runtime. The first part uses pfSolve_AppCollectionManager to create a map collection of kernels that is checked against if a particular system that is required during solution has already been generated. The specialization constants can select the following parameters in each kernel at execution time (with no additional binaries required): 

- offsetM - offset of the M matrix in the buffer (constants for bidiagonal backsolve)
- offsetV - offset of the V matrix in the buffer (constants for matrix-vector multiplication)
- offsetSolution - offset of the input/output block of vectors to be solved against
- inputZeropad[2] - specify range of values to be zero in the input
- outputZeropad[2] - specify range of values to be zero in the output 
- inputBufferStride - specify the stride of the M and V matrix buffers
- outputBufferStride - specify the stride of the input/output block of vectors buffers
- batch - specify number of batched systems to solve
- long double scaleC - specify the scalar to multiply some of the values (depends on the algorithm)

## Build instructions and test suite 

The testing suite and its CMakeLists is almost entirely based on the VkFFT test suite. Can be launched with -pfsolve X flag, where X is the id of required test. Current tests 0: double-double emulation of quad precision single step of JW, 1: regular double precision single step of JW. results are verified against higher precision in mpir or __float128 (in gcc).

## Description of type bitfields (in decimal)

block%10        =   1 or other - all operations are done in real number format \
                    2 - read real part of complex number, write it as real number \
                    3 - read imaginary part of complex number, write it as real number \
                    4 - read complex number in strided format, write in packed complex format (XXIIXXII - XIXIXIXI) \
                    5 - read real number, write it in real part of complex number \
                    6 - read real number, write it in imaginary part of complex number \
                    7 - read complex number in packed complex format, write complex number in strided format for batched polynomial processing (XIXIXIXI - XXIIXXII) \
(block/10) % 10 =   2 - add the results of the operator to the value in the outputBuffer \
(block/100) % 10=   1 - ScaleC operator \
                    2 - ScaleD operator \
                    3 - ScaleSphLaplA operator \
                    4 - ScaleSphLaplB operator \

jw_type%10      =   1 - do not perform Tridiagonal Solve in kernel \
                    2 - do not perform MatVecMul in kernel (not used, use 3) \
                    3 - special version where MatVecMul matrix is diagonal (used in applyTriSolve - there MatVecMul is used to get identity values in bidiagonal matrix diagonal) \
