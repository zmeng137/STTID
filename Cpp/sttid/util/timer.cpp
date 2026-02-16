#include "util.h"

// static members of a class must be defined
// somewhere in an object file, otherwise you
// will get linker errors (undefined reference)
std::map<std::string, int> util::Timer::counts_;
std::map<std::string, double> util::Timer::times_;
std::map<std::string, double> util::Timer::squared_times_;
std::map<std::string, double> util::Timer::max_time_;
std::map<std::string, double> util::Timer::min_time_;

util::Timer::Timer(std::string label)
  : label_(label) {
    t_start_ = std::chrono::high_resolution_clock::now();
}

util::Timer::~Timer()
{
    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_diff = std::chrono::duration<double>(t_end - t_start_).count();
    times_[label_] += t_diff;
    squared_times_[label_] += (t_diff)*(t_diff);
    counts_[label_]++;
    if(max_time_[label_]==0.0){ //If this is the first time calling the max_time_ map for this label:
      max_time_[label_] += t_diff;
      min_time_[label_] += t_diff;
    } else { //If the max_time_ map has already been called for this label:
      max_time_[label_] = (t_diff>max_time_[label_])?t_diff:max_time_[label_];
      min_time_[label_] = (t_diff<min_time_[label_])?t_diff:min_time_[label_];
    }
}

void util::Timer::summarize(std::ostream& os)
{
	os << std::endl;
	std::cout << std::setw(25) << std::left << "label" << "\t,"
	<< std::setw(10) << "mean time (ms)" << "\t,"
	<< std::setw(5) << "calls"			 << "\t,"
	<< std::setw(10) << "total time (s)" << "\t,"
	<< std::setw(10) << "max time (ms)"  << "\t,"
	<< std::setw(10) << "min time (ms)"  << "\t,"
	<< std::setw(10) << "std time (ms)"  << std::endl;

	for (auto [label, time]: times_)
	{
		int count = counts_[label];
		double sigma = sqrt(squared_times_[label]/count-time*time/double(count*count));
		std::cout << std::fixed << std::setprecision(4) << std::setw(25) 
		<< std::left 	 << label.substr(0, 35) 		<< "\t,"
		<< std::setw(10) << time/double(count)*1000.0 	<< "\t,"
		<< std::setw(5) << count 						<< "\t,"
		<< std::setw(10) << time 						<< "\t,"
		<< std::setw(10) << max_time_[label]*1000.0 	<< "\t,"
		<< std::setw(10) << min_time_[label]*1000.0 	<< "\t,"
		<< std::setw(10) << sigma*1000.0 				<< std::endl;
	}
	os << std::endl;
}