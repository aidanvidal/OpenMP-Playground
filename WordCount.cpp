#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <omp.h>

#define DEFAULT_NUM_THREADS 8 // This is how many cores my machine has

// Read entire file into a string
std::string read_entire_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Check if char is part of a word
bool is_word_char(char c) {
    return std::isalnum(c) || c == '\'';
}

// Tokenize into lowercase words
std::vector<std::string> tokenize(const std::string& text) {
    std::vector<std::string> words;
    std::string word;
    for (char c : text) {
        if (is_word_char(c)) {
            word += std::tolower(c);
        } else if (!word.empty()) {
            words.push_back(word);
            word.clear();
        }
    }
    if (!word.empty()) words.push_back(word);
    return words;
}

// Output top-N results
void print_top_words(const std::unordered_map<std::string, int>& map, int top_n = 10) {
    std::vector<std::pair<std::string, int>> word_list(map.begin(), map.end());
    std::sort(word_list.begin(), word_list.end(),
              [](const auto& a, const auto& b) { return b.second < a.second; });

    std::cout << "Top " << top_n << " words:\n";
    for (int i = 0; i < top_n && i < word_list.size(); ++i) {
        std::cout << word_list[i].first << ": " << word_list[i].second << "\n";
    }
}

// === SERIAL VERSION ===
void run_serial(const std::vector<std::string>& words) {
    std::unordered_map<std::string, int> word_count;
    for (const std::string& word : words) {
        word_count[word]++;
    }

    print_top_words(word_count);
}

// === PARALLEL VERSION ===
void run_parallel(const std::vector<std::string>& words, int num_threads) {
    size_t total_words = words.size();
    std::vector<std::unordered_map<std::string, int>> thread_maps(num_threads);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        size_t chunk_size = total_words / num_threads;
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? total_words : start + chunk_size;

        for (size_t i = start; i < end; ++i) {
            thread_maps[tid][words[i]]++;
        }
    }

    // Reduce phase
    std::unordered_map<std::string, int> global_map;
    for (const auto& local_map : thread_maps) {
        for (const auto& [word, count] : local_map) {
            global_map[word] += count;
        }
    }

    print_top_words(global_map);
}

// === MAIN ===
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <mode: serial|parallel> [num_threads]\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string mode = argv[2];
    int num_threads = (argc >= 4) ? std::stoi(argv[3]) : DEFAULT_NUM_THREADS;

    std::string content;
    try {
        content = read_entire_file(filename);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    std::vector<std::string> words = tokenize(content);

    double start = omp_get_wtime();
    if (mode == "serial") {
        run_serial(words);
    } else if (mode == "parallel") {
        run_parallel(words, num_threads);
    } else {
        std::cerr << "Unknown mode: " << mode << " (use 'serial' or 'parallel')\n";
        return 1;
    }
    double end = omp_get_wtime();
    std::cout << "Execution time for " << mode << ": " << (end - start) << " seconds\n";

    return 0;
}