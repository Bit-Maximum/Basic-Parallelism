#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <cstddef>


using namespace std;

void matrixMul(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){
    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            double accum = 0;
            for (size_t i = 0; i < sharedDim; i++)
                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
            result[r1 * colsB + c2] = accum;
        }
}

void matrixMulAVX512(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){

    if ((rowsA * colsB) % sizeof(__m512d) == 0) {
        cout << "(rowsA * colsB) % sizeof(__m512d) == " << (rowsA * colsB) % sizeof(__m512d) << endl;
        cout << "Unprocessable entity" << endl;
        return;
    }

    for (size_t r1 = 0; r1 < rowsA; r1++)
        for (size_t c2 = 0; c2 < colsB; c2++) {
            __m512d x = _mm512_load_pd(&matA[r1 * sharedDim]);
            __m512d y = _mm512_load_pd(&matB[c2 * sharedDim]);
//
//            for (size_t i = 0; i < sharedDim; i++)
//                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
//            result[r1 * colsB + c2] = accum;
        }
}

void matrixPrint(double *matrix, size_t rows, size_t cols) {
    cout << "Matrix:\n";
    for (size_t r = 0; r < rows; r++){
        for (size_t c = 0; c < cols; c++)
            cout << matrix[r * cols + c] << ' ';
        cout << '\n';
    }
}


void transposeAVX512(double* out, const double* in, size_t rows, size_t cols)
{
    const size_t blockSize = 8; // 8 double = 512 бит

    size_t i, j;
    for (i = 0; i + blockSize <= rows; i += blockSize) {
        for (j = 0; j + blockSize <= cols; j += blockSize) {

            // Загружаем блок 8×8 из входной матрицы
            __m512d row0 = _mm512_loadu_pd(&in[(i + 0) * cols + j]);  // row0 = a0 a1 a2 a3 a4 a5 a6 a7
            __m512d row1 = _mm512_loadu_pd(&in[(i + 1) * cols + j]);  // row1 = b0 b1 b2 b3 b4 b5 b6 b7
            __m512d row2 = _mm512_loadu_pd(&in[(i + 2) * cols + j]);
            __m512d row3 = _mm512_loadu_pd(&in[(i + 3) * cols + j]);
            __m512d row4 = _mm512_loadu_pd(&in[(i + 4) * cols + j]);
            __m512d row5 = _mm512_loadu_pd(&in[(i + 5) * cols + j]);
            __m512d row6 = _mm512_loadu_pd(&in[(i + 6) * cols + j]);
            __m512d row7 = _mm512_loadu_pd(&in[(i + 7) * cols + j]);

            // Разбиваем и переставляем пары строк (unpack)
            // unpacklo — чередуем первые 2 double каждой строки
            // unpackhi — чередуем последние 2 double каждой строки
            __m512d t0 = _mm512_unpacklo_pd(row0, row1);  // t0 = a0 b0 a1 b1 a2 b2 a3 b3
            __m512d t1 = _mm512_unpackhi_pd(row0, row1);  // t1 = a4 b4 a5 b5 a6 b6 a7 b7
            __m512d t2 = _mm512_unpacklo_pd(row2, row3);  // t2: c0 d0 c1 d1 c2 d2 c3 d3
            __m512d t3 = _mm512_unpackhi_pd(row2, row3);  // t3: c4 d4 c5 d5 c6 d6 c7 d7
            __m512d t4 = _mm512_unpacklo_pd(row4, row5);  // t4: e0 f0 e1 f1 g0 h0 g1 h1
            __m512d t5 = _mm512_unpackhi_pd(row4, row5);  // t5: e4 f4 e5 f5 g2 h2 g3 h3
            __m512d t6 = _mm512_unpacklo_pd(row6, row7);  // t6: g0 h0 g1 h1 g2 h2 g3 h3
            __m512d t7 = _mm512_unpackhi_pd(row6, row7);  // t7: g4 h4 g5 h5 g6 h6 g7 h7

            // Первый shuffle: объединяем пары блоков 4×4 (128-bit lanes)
            // 0x44 = берем нижние половины из обоих регистров
            // 0xEE = берем верхние половины из обоих регистров
            __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x44);  // s0: a0 b0 a1 b1 c0 d0 c1 d1
            __m512d s1 = _mm512_shuffle_f64x2(t0, t2, 0xEE);  // s1: a2 b2 a3 b3 c2 d2 c3 d3
            __m512d s2 = _mm512_shuffle_f64x2(t1, t3, 0x44);  // s2: a4 b4 a5 b5 c4 d4 c5 d5
            __m512d s3 = _mm512_shuffle_f64x2(t1, t3, 0xEE);  // s3: a6 b6 a7 b7 c6 d6 c7 d7
            __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x44);  // s4: e0 f0 e1 f1 g0 h0 g1 h1
            __m512d s5 = _mm512_shuffle_f64x2(t4, t6, 0xEE);  // s5: e2 f2 e3 f3 g2 h2 g3 h3
            __m512d s6 = _mm512_shuffle_f64x2(t5, t7, 0x44);  // s6: e4 f4 e5 f5 g4 h4 g5 h5
            __m512d s7 = _mm512_shuffle_f64x2(t5, t7, 0xEE);  // s7: e6 f6 e7 f7 g6 h6 g7 h7

            // Второй shuffle: собираем верх и низ блока 8x8
            // 0x88 = берем первые два 128-bit блока из верхней и нижней половин (верхняя часть блока)
            // 0xDD = берем последние два 128-bit блока из верхней и нижней половин (нижняя часть блока)
            // s0: [a0 b0] | [a1 b1] | [c0 d0] | [c1 d1]
            // s4: [e0 f0] | [e1 f1] | [g0 h0] | [g1 h1]
            __m512d r0 = _mm512_shuffle_f64x2(s0, s4, 0x88);  // r0: a0 b0 c0 d0 e0 f0 g0 h0
            __m512d r1 = _mm512_shuffle_f64x2(s1, s5, 0x88);  // r1: a1 b1 c1 d1 e1 f1 g1 h1
            __m512d r2 = _mm512_shuffle_f64x2(s2, s6, 0x88);  // r2: a2 b2 c2 d2 e2 f2 g2 h2
            __m512d r3 = _mm512_shuffle_f64x2(s3, s7, 0x88);  // r3: a3 b3 c3 d3 e3 f3 g3 h3
            __m512d r4 = _mm512_shuffle_f64x2(s0, s4, 0xDD);  // r4: a4 b4 c4 d4 e4 f4 g4 h4
            __m512d r5 = _mm512_shuffle_f64x2(s1, s5, 0xDD);  // r5: a5 b5 c5 d5 e5 f5 g5 h5
            __m512d r6 = _mm512_shuffle_f64x2(s2, s6, 0xDD);  // r6: a6 b6 c6 d6 e6 f6 g6 h6
            __m512d r7 = _mm512_shuffle_f64x2(s3, s7, 0xDD);  // r7: a7 b7 c7 d7 e7 f7 g7 h7

            // Сохраняем транспонированный блок
            _mm512_storeu_pd(&out[(j + 0) * rows + i], r0);
            _mm512_storeu_pd(&out[(j + 1) * rows + i], r1);
            _mm512_storeu_pd(&out[(j + 2) * rows + i], r2);
            _mm512_storeu_pd(&out[(j + 3) * rows + i], r3);
            _mm512_storeu_pd(&out[(j + 4) * rows + i], r4);
            _mm512_storeu_pd(&out[(j + 5) * rows + i], r5);
            _mm512_storeu_pd(&out[(j + 6) * rows + i], r6);
            _mm512_storeu_pd(&out[(j + 7) * rows + i], r7);
        }

        // Обработка хвостовых столбцов справа (если cols не кратен 8)
        for (; j < cols; ++j) {
            for (size_t ii = 0; ii < blockSize; ++ii) {
                out[j * rows + (i + ii)] = in[(i + ii) * cols + j];
            }
        }
    }

    // Обработка хвостовых строк снизу (если rows не кратен 8)
    for (; i < rows; ++i) {
        for (j = 0; j < cols; ++j) {
            out[j * rows + i] = in[i * cols + j];
        }
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

    rowsA = 10; sharedDim = 10; colsB = 10;
    A = std::vector<double>(rowsA * sharedDim);
    B = std::vector<double>(colsB * sharedDim);
//    R = std::vector<double>(rowsA * colsB, 0.0);

    for (size_t i = 0; i < rowsA * sharedDim; ++i) {
        A[i] = (double) i;
    }

    for (size_t i = 0; i < colsB * sharedDim; ++i) {
        B[i] = (double) i;
    }

    matrixPrint(A.data(), rowsA, sharedDim);
    matrixPrint(B.data(), colsB, sharedDim);

    return 0;
}