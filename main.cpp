#include <chrono>
#include <cstdlib>
#include <iostream>
#include <omp.h>

#define N (1u << 24)


double av_omp_3(const double* V, size_t n) {
    unsigned P = omp_get_num_procs();
    unsigned T;
    double* r = static_cast<double*>(calloc(sizeof(double), P));

#pragma omp parallel shared(T)
    {
        unsigned
                t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
        }

        double output = 0.0;
        for (size_t i = t; i < n; i += T)
            output += V[i];
        r[t] = output;
    }

    double sum = 0.0;
    for (size_t i = 0; i < P; i++)
        sum += r[i];

    return sum / n;
}

struct sum_t {
    double v;
    char padding[64 - sizeof(double)];
};

double av_omp_4(const double* V, size_t n) {
    unsigned P = omp_get_num_procs();
    unsigned T;
    struct sum_t* r = static_cast<sum_t*>(calloc(64, P));

#pragma omp parallel shared(T)
    {
        unsigned
                t = omp_get_thread_num();
#pragma omp single
        {
            T = omp_get_num_threads();
        }

        double output = 0.0;
        for (size_t i = t; i < n; i += T)
            output += V[i];
        r[t].v = output;
    }

    double sum = 0.0;
    for (size_t i = 0; i < P; i++)
        sum += r[i].v;

    return sum / n;
}


int main() {
    double* p = static_cast<double*>(malloc(N * sizeof(double)));

    for (size_t i = 0; i < N; i++) {
        p[i] = (double)i;
    }

    auto t4 = std::chrono::steady_clock::now();
    std::cout << "Average: " << av_omp_3(p, N) << std::endl;
    std::cout << "Time:    " <<
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t4).count() <<
              std::endl << std::endl;

    auto t5 = std::chrono::steady_clock::now();
    std::cout << "Average: " << av_omp_4(p, N) << std::endl;
    std::cout << "Time:    " <<
              std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t5).count() <<
              std::endl << std::endl;

    free(p);
    return 0;
}