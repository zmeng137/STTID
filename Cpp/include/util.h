// util.h - Some general-purpose tools
#ifndef UTIL_H
#define UTIL_H

#include "core.h"
#include <mkl/mkl_spblas.h>

namespace util{

// Print matrix [rmask, cmask]
template<class T>
void PrintMatWindow(T* matrix, size_t row, size_t col,  
                    std::tuple<int,int> rmask, std::tuple<int,int> cmask)
{
    if (std::get<0>(rmask) < 0 || std::get<1>(rmask) >= row ||
        std::get<0>(cmask) < 0 || std::get<1>(cmask) >= col) {
            throw std::invalid_argument("Invalid input row or column mask.");
    }
    for (int i = std::get<0>(rmask); i <= std::get<1>(rmask); ++i) {
        for (int j = std::get<0>(cmask); j <= std::get<1>(cmask); ++j) {
            std::cout << matrix[i * col + j] << " ";        
        }
        std::cout << "\n";
    }
    return;
}

// Print a 1-dimensional array
template<class T>
void Print1DArray(T* array, size_t N) 
{
    std::cout << "[" << array[0];
    for (size_t i = 1; i < N; ++i) {
        std::cout << ", " << array[i];
    }
    std::cout << "]" << std::endl;
    return;
}

// Random array generator
template<class T>
void generateRandomArray(T* array, int size, T minValue, T maxValue) {
    // Create a random number generator and distribution
    std::random_device rd;  // Seed source
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<> dis(minValue, maxValue);
    // Fill the vector with random numbers
    for (int i = 0; i < size; ++i) {
        array[i] = dis(gen);
    }
    return;
}

class Timer
{
public:

  Timer(std::string label);
  ~Timer();
  static void summarize(std::ostream& os=std::cout);

private:

  std::string label_;
  std::chrono::_V2::system_clock::time_point t_start_;
  static std::map<std::string, double> times_;
  static std::map<std::string, double> squared_times_;
  static std::map<std::string, double> max_time_;
  static std::map<std::string, double> min_time_;
  static std::map<std::string, int> counts_;

}; 

template<size_t Order>
bool read_sparse_tensor(const std::string& filename, 
                       std::array<size_t*, Order>& indices,
                       std::array<size_t,  Order>& dimensions,
                       double* values, size_t num_entries, 
                       bool binary,    size_t idx_offset) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "err 1\n";
        return false;
    }

    // Initialize arrays for each dimension
    for (size_t i = 0; i < Order; ++i) {
        indices[i] = new size_t[num_entries];
    }

    std::string line;
    size_t entry = 0;
    
    while (std::getline(file, line) && entry < num_entries) {
        std::istringstream iss(line);
        
        // Read indices for each dimension
        for (size_t dim = 0; dim < Order; ++dim) {
            if (iss >> indices[dim][entry]) {
                // Index offset 
                indices[dim][entry] -= idx_offset;
            } else {
                // Cleanup on error
                for (size_t i = 0; i < Order; ++i) {
                    delete[] indices[i];
                }
                std::cout << "err 1\n";
                return false;
            }
        }
        
        // Read the value
        if (!binary) {
            if (!(iss >> values[entry])) {
            // Cleanup on error
                for (size_t i = 0; i < Order; ++i) {
                    delete[] indices[i];
                }
                std::cout << "err 1\n";
                return false;
            }
        } 
        
        ++entry;
    }

    // If the tensor is binary (such as wiki knowledge graph), the value array is directly initialized by 1
    if (binary) {
        std::fill(values, values + entry, 1.0);
    }

    // Identify the tensor size by finding the maximum mode indices 
    for (size_t i = 0; i < Order; ++i) {
        dimensions[i] = *std::max_element(indices[i], indices[i] + entry) + idx_offset;
    }

    file.close();
    return true;
}

#define CHECK_MKL_ERROR(val) checkMKL((val), #val, __FILE__, __LINE__)
inline void checkMKL(sparse_status_t err, const char* const func, const char* const file, int const line)
{
	if (err != SPARSE_STATUS_SUCCESS) 
		fprintf(stderr, "MKL Error at: %s: %d\n Error Code %d %s\n", file, line, err, func);
}

class HostMemoryLogger
{
private:
    std::ofstream log_file;
    std::thread logger_thread;
    bool running = true;
    int sample_interval_ms;

    // Get current timestamp as string
    std::string get_timestamp()
    {
        auto now = std::chrono::system_clock::now();
        auto now_time = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf;
        localtime_r(&now_time, &tm_buf);

        char buffer[25];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
        return std::string(buffer);
    }

    // Extract memory value (in KB) from a line of /proc/self/status
    long extract_memory_value(const std::string &line)
    {
        size_t pos = line.find(":");
        if (pos != std::string::npos)
        {
            std::string value_str = line.substr(pos + 1);
            // Remove "kB" and whitespace
            value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
                                           [](unsigned char c)
                                           { return std::isalpha(c) || std::isspace(c); }),
                            value_str.end());
            return std::stol(value_str);
        }
        return 0;
    }

    void logging_loop()
    {
        // Write CSV header
        log_file << "Timestamp,VmRSS (MB),VmSize (MB),VmPeak (MB),VmHWM (MB)" << std::endl;

        while (running)
        {
            std::ifstream status("/proc/self/status");
            std::string line;

            long vm_rss = 0;  // Resident set size (physical memory used)
            long vm_size = 0; // Virtual memory size
            long vm_peak = 0; // Peak virtual memory size
            long vm_hwm = 0;  // Peak resident set size

            while (std::getline(status, line))
            {
                if (line.find("VmRSS") != std::string::npos)
                {
                    vm_rss = extract_memory_value(line);
                }
                else if (line.find("VmSize") != std::string::npos)
                {
                    vm_size = extract_memory_value(line);
                }
                else if (line.find("VmPeak") != std::string::npos)
                {
                    vm_peak = extract_memory_value(line);
                }
                else if (line.find("VmHWM") != std::string::npos)
                {
                    vm_hwm = extract_memory_value(line);
                }
            }

            // Convert KB to MB for readability
            double vm_rss_mb = vm_rss / 1024.0;
            double vm_size_mb = vm_size / 1024.0;
            double vm_peak_mb = vm_peak / 1024.0;
            double vm_hwm_mb = vm_hwm / 1024.0;

            // Write to CSV file
            log_file << get_timestamp() << ","
                     << std::fixed << std::setprecision(2) << vm_rss_mb << ","
                     << vm_size_mb << ","
                     << vm_peak_mb << ","
                     << vm_hwm_mb << std::endl;

            // Ensure data is written to disk
            log_file.flush();

            std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms));
        }
    }

public:
    HostMemoryLogger(const std::string &filename, int interval_ms = 1000)
        : sample_interval_ms(interval_ms)
    {
        log_file.open(filename);
        if (!log_file.is_open())
        {
            std::cerr << "Failed to open log file: " << filename << std::endl;
            return;
        }

        // Write comment header with explanation of fields
        log_file << "# Program Memory Logger" << std::endl;
        log_file << "# VmRSS: Resident Set Size - amount of physical memory used by the process" << std::endl;
        log_file << "# VmSize: Virtual Memory Size - total virtual memory allocated to the process" << std::endl;
        log_file << "# VmPeak: Peak Virtual Memory Size - maximum VmSize observed" << std::endl;
        log_file << "# VmHWM: High Water Mark - peak resident set size" << std::endl;

        logger_thread = std::thread(&HostMemoryLogger::logging_loop, this);
        std::cout << "Program Memory Logger started with interval: "
                  << interval_ms << "ms" << std::endl;
    }

    ~HostMemoryLogger()
    {
        running = false;
        if (logger_thread.joinable())
        {
            logger_thread.join();
        }
        if (log_file.is_open())
        {
            log_file.close();
        }
    }

    // Get current memory usage snapshot
    static void printCurrentMemoryUsage()
    {
        std::ifstream status("/proc/self/status");
        std::string line;

        long vm_rss = 0;
        long vm_size = 0;

        while (std::getline(status, line))
        {
            if (line.find("VmRSS") != std::string::npos)
            {
                size_t pos = line.find(":");
                if (pos != std::string::npos)
                {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
                                                   [](unsigned char c)
                                                   { return std::isalpha(c) || std::isspace(c); }),
                                    value_str.end());
                    vm_rss = std::stol(value_str);
                }
            }
            else if (line.find("VmSize") != std::string::npos)
            {
                size_t pos = line.find(":");
                if (pos != std::string::npos)
                {
                    std::string value_str = line.substr(pos + 1);
                    value_str.erase(std::remove_if(value_str.begin(), value_str.end(),
                                                   [](unsigned char c)
                                                   { return std::isalpha(c) || std::isspace(c); }),
                                    value_str.end());
                    vm_size = std::stol(value_str);
                }
            }
        }

        std::cout << "Current Memory Usage:" << std::endl;
        std::cout << "  Physical (RSS): " << vm_rss / 1024.0 << " MB" << std::endl;
        std::cout << "  Virtual (VSZ): " << vm_size / 1024.0 << " MB" << std::endl;
    }
};




}

#endif

