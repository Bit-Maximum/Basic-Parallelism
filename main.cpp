#include <iostream>
#include <memory>
#include <omp.h>
#include <chrono>

#define N (1 << 23)

using namespace std;



double average(const double* V, size_t n) {
    double sum = 0.0;
    for(size_t i = 0; i < n; i++) {
        sum += V[i];
    }
    return sum / (double) n;
}


double average_omp(const double* V, size_t n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < n; i++) {
        sum += V[i];
    }
    return sum / (double) n;
}



int main() {

    const size_t SIZE = N;
    auto data = std::make_unique<double[]>(N);

    for (size_t i = 0; i < SIZE; ++i){
        data[i] = (double) i;
//        cout << data[i] << " ";
    }

    auto t1 = std::chrono::steady_clock::now();
    double r1 = average(data.get(), SIZE);
    auto t2 = std::chrono::steady_clock::now();
    cout << "Sync AVG: " << r1 << endl;
    cout << "Duration of synchronous calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;

    t1 = std::chrono::steady_clock::now();
    double r2 = average_omp(data.get(), SIZE);
    t2 = std::chrono::steady_clock::now();
    cout << "Parallel AVG: " << r2 << endl;
    cout << "Duration of parallel calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;


    return 0;
}
