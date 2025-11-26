#include "vector_mod.h"
#include "mod_ops.h"
#include "num_threads.h"
#include <vector>
#include <barrier>
#include <condition_variable>
#include <thread>


class barrier {
    unsigned current_barier_id = 0;
    const unsigned T_max;
    unsigned T;
    std::mutex mtx;
    std::condition_variable cv;
public:

    explicit barrier(unsigned threads): T_max(threads), T(T_max) {}

    void arrive_and_wait() {
        std::unique_lock lock {mtx};
        if(--T) {
            auto inner_id = current_barier_id;
            while(inner_id == current_barier_id) {
                cv.wait(lock);
            }
        }
        else {
            T = T_max;
            current_barier_id++;
            cv.notify_all();
        }
    }
};

IntegerWord pow_mod(IntegerWord base, IntegerWord power, IntegerWord mod) {
	IntegerWord result = 1;
	while (power > 0) {
		if (power % 2 != 0) {
			result = mul_mod(result, base, mod);
		}
		power >>= 1;
		base = mul_mod(base, base, mod);
	}
	return result;
}

IntegerWord word_pow_mod(size_t power, IntegerWord mod) {
	return pow_mod((-mod) % mod, power, mod);
}

struct thread_range {
	std::size_t start, end;
};
thread_range vector_thread_range(size_t total_words, unsigned total_threads, unsigned thread_num) {
	auto remains_size = total_words % total_threads;
	auto chunk_size = total_words / total_threads;
	if (thread_num < remains_size)
        remains_size = ++chunk_size * thread_num;
	else
        remains_size += chunk_size * thread_num;
	size_t end = remains_size + chunk_size;
	return thread_range(remains_size, end);
}

struct partial_result_t {
	alignas(std::hardware_destructive_interference_size) IntegerWord value;
};

IntegerWord vector_mod(const IntegerWord* V, std::size_t N, IntegerWord mod) {
	long long total_threads = get_num_threads();
    // Made threads -1 because first thread is already running
	std::vector<std::thread> threads(total_threads - 1);
	std::vector<partial_result_t> partial_results(total_threads);
	std::barrier<> bar(total_threads);

	auto thread_lambda = [V, N, total_threads, mod, &partial_results, &bar](unsigned t) {
		auto [b, e] = vector_thread_range(N, total_threads, t);

		IntegerWord sum = 0;
		// Схема Хорнера
		for (std::size_t i = e; b < i;) {
			//sum = (sum * x + V[end-1-i]) % mod;
			sum = add_mod(times_word(sum, mod), V[--i], mod); // то же самое, но без переполнения
		}
		partial_results[t].value = sum;
		for (size_t i = 1, ii = 2; i < total_threads; i = ii, ii += ii) {
			bar.arrive_and_wait();
			if (t % ii == 0 && t + i < total_threads) {
				auto neighbor = vector_thread_range(N, total_threads, t + i);
				partial_results[t].value = add_mod(partial_results[t].value, mul_mod(partial_results[t + i].value, word_pow_mod(neighbor.start - b, mod), mod), mod);
			}
		}
	};
	// s готова
	for (std::size_t i = 1; i < total_threads; ++i) {
		threads[i - 1] = std::thread(thread_lambda, i);
	}
	thread_lambda(0);

	for (auto& i : threads) {
		i.join();
	}
	return partial_results[0].value;
}