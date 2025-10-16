#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <cstddef>


using namespace std;


void transposeAVX2(double* out, const double* in, size_t rows, size_t cols)
{
    const size_t blockSize = 4; // 4 double = 256 бит

    size_t i, j;
    for (i = 0; i + blockSize <= rows; i += blockSize) {
        for (j = 0; j + blockSize <= cols; j += blockSize) {

            // Загружаем блок 4×4
            __m256d row0 = _mm256_loadu_pd(&in[(i+0)*cols + j]); // a0 a1 a2 a3
            __m256d row1 = _mm256_loadu_pd(&in[(i+1)*cols + j]); // b0 b1 b2 b3
            __m256d row2 = _mm256_loadu_pd(&in[(i+2)*cols + j]); // c0 c1 c2 c3
            __m256d row3 = _mm256_loadu_pd(&in[(i+3)*cols + j]); // d0 d1 d2 d3

            // Разбиваем и переставляем пары строк
            __m256d t0 = _mm256_unpacklo_pd(row0, row1); // a0 b0 a1 b1
            __m256d t1 = _mm256_unpackhi_pd(row0, row1); // a2 b2 a3 b3
            __m256d t2 = _mm256_unpacklo_pd(row2, row3); // c0 d0 c1 d1
            __m256d t3 = _mm256_unpackhi_pd(row2, row3); // c2 d2 c3 d3

            // Собираем верх и низ блока
            __m256d r0 = _mm256_permute2f128_pd(t0, t2, 0x20); // a0 b0 c0 d0
            __m256d r1 = _mm256_permute2f128_pd(t1, t3, 0x20); // a2 b2 c2 d2
            __m256d r2 = _mm256_permute2f128_pd(t0, t2, 0x31); // a1 b1 c1 d1
            __m256d r3 = _mm256_permute2f128_pd(t1, t3, 0x31); // a3 b3 c3 d3

            // Сохраняем транспонированный блок
            _mm256_storeu_pd(&out[(j+0)*rows + i], r0); // a0 b0 c0 d0
            _mm256_storeu_pd(&out[(j+1)*rows + i], r1); // a1 b1 c1 d1
            _mm256_storeu_pd(&out[(j+2)*rows + i], r2); // a2 b2 c2 d2
            _mm256_storeu_pd(&out[(j+3)*rows + i], r3); // a3 b3 c3 d3
        }


        // хвостовые столбцы справа
        for (; j < cols; ++j) {
            for (size_t ii = 0; ii < blockSize && (i+ii)<rows; ++ii)
                out[j*rows + (i+ii)] = in[(i+ii)*cols + j];
        }
    }

    // хвостовые строки снизу
    for (; i < rows; ++i) {
        for (j = 0; j < cols; ++j)
            out[j*rows + i] = in[i*cols + j];
    }
}


void matrixPrint(double *matrix, size_t rows, size_t cols) {
    cout << "Matrix:\n";
    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++)
            cout << matrix[r * cols + c] << ' ';
        cout << endl;
    }
}


void matrixMul(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){
    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            double accum = 0;
            for (size_t i = 0; i < sharedDim; i++)
                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
            result[r1 * colsB + c2] = accum;
        }
}

void matrixMulAVX2(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){

    std::vector<double> TransposeT(colsB * sharedDim, 0.0);

    if (sharedDim % 4 == 0 && colsB % 4 == 0)
        transposeAVX2(TransposeT.data(), matB, colsB, sharedDim);
    else {
        for (size_t r = 0; r < sharedDim; ++r)
            for (size_t c = 0; c < colsB; ++c)
                TransposeT[c * sharedDim + r] = matB[r * colsB + c];
    }

    const double *matT = TransposeT.data();
    double temp[2];

    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m256d sum_vec = _mm256_setzero_pd();
            size_t k = 0;

            // Умножаем блоками по 4 double
            for (; k + 3 < sharedDim; k += 4) {
                __m256d x = _mm256_loadu_pd(&matA[r1 * sharedDim + k]);
                __m256d y = _mm256_loadu_pd(&matT[c2 * sharedDim + k]);
                sum_vec = _mm256_add_pd(sum_vec, _mm256_mul_pd(x, y));
            }

            // Суммируем 4 элемента внутри регистра
            __m128d low = _mm256_castpd256_pd128(sum_vec);
            __m128d high = _mm256_extractf128_pd(sum_vec, 1);
            __m128d sum128 = _mm_add_pd(low, high);

            _mm_storeu_pd(temp, sum128);
            double result_val = temp[0] + temp[1];

            // Хвостовые элементы
            for (; k < sharedDim; ++k) {
                result_val += matA[r1 * sharedDim + k] * matT[c2 * sharedDim + k];
            }

            result[r1 * colsB + c2] = result_val;
        }
}



int main() {
    cout << "Test 1:" << endl;
    std::size_t rowsA = 2, sharedDim = 3, colsB = 2;
    std::vector<double> A(rowsA * sharedDim, 1.0);
    std::vector<double> B(sharedDim * colsB, 2.0);
    std::vector<double> R(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);

    cout << "Test 2:" << endl;
    rowsA = 2; sharedDim = 3; colsB = 2;
    A = std::vector<double>({ 1.0, 2.0, -3.0,
                                -2.0, 13.0, -2.0});
    B = std::vector<double>({3.0, 4.0,
                                5.0, -1.0,
                                4.0, 4.0});
    R = std::vector<double>(rowsA * colsB, 0.0);

    matrixMul(R.data(), A.data(), B.data(), sharedDim, rowsA, colsB);
    matrixPrint(R.data(), rowsA, colsB);

    rowsA = 12; sharedDim = 12; colsB = 12;
    A = std::vector<double>(rowsA * sharedDim, 1.0);
    B = std::vector<double>(colsB * sharedDim);
    std::vector<double> R1 = std::vector<double>(rowsA * colsB, 0.0);
    std::vector<double> R2 = std::vector<double>(rowsA * colsB, 0.0);

    for (size_t i = 0; i < colsB * sharedDim; ++i) {
        B[i] = (double) i;
    }

    matrixMul(R1.data(), A.data(), B.data(), sharedDim, rowsA, colsB);

    matrixPrint(R1.data(), rowsA, colsB);

    matrixMulAVX2(R2.data(), A.data(), B.data(), sharedDim, rowsA, colsB);

    matrixPrint(R2.data(), rowsA, colsB);

    return 0;
}