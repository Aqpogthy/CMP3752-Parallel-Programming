#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
// The start of this and most of the framework is gathered from tutorial 2
void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.ppm)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test_large.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned char> image_input(image_filename.c_str());
		CImgDisplay disp_input(image_input, "input");

		//host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		// Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef int mytype;

		float nr_bins = 256;

		std::vector<mytype> B(nr_bins);

		//device - buffers

		size_t histogram_size = nr_bins * sizeof(mytype);

		cl::Buffer Histogram(context, CL_MEM_READ_WRITE, histogram_size);

		cl::Buffer CumulativeHistogram(context, CL_MEM_READ_WRITE, histogram_size);

		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());

		cl::Buffer NormalisedHistogram(context, CL_MEM_READ_WRITE, histogram_size);
		
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		
		float SF = nr_bins / image_input.size(); //sets the scale factor which will be used to normalise and scale the cumulative histogram

		
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);//Copy images to device memory
		queue.enqueueFillBuffer(Histogram, 0, 0, histogram_size);
		queue.enqueueFillBuffer(CumulativeHistogram, 0, 0, histogram_size);
		queue.enqueueFillBuffer(NormalisedHistogram, 0, 0, histogram_size);


		//Setup and execute the kernel (i.e. device code)
		/*cl::Kernel HistogramKernel = cl::Kernel(program, "intensityHistogram");
		HistogramKernel.setArg(0, dev_image_input);
		HistogramKernel.setArg(1, Histogram);*/

		cl::Kernel LocalHistogramKernel = cl::Kernel(program, "intensityHistogramLocal");
		cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0]; // get device
		//std::cout << LocalHistogramKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>	(device) << endl; // get info
		size_t local_size = LocalHistogramKernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
		LocalHistogramKernel.setArg(0, dev_image_input);
		LocalHistogramKernel.setArg(1, cl::Local(local_size * sizeof(mytype) * nr_bins));
		LocalHistogramKernel.setArg(2, Histogram);
		LocalHistogramKernel.setArg(3, nr_bins);


		cl::Kernel CumulativeHistogramKernel = cl::Kernel(program, "cumulativeHistogram");
		CumulativeHistogramKernel.setArg(0, Histogram);
		CumulativeHistogramKernel.setArg(1, CumulativeHistogram);

		cl::Kernel NormaliseKernel = cl::Kernel(program, "NormaliseAndScale");
		NormaliseKernel.setArg(0, CumulativeHistogram);
		NormaliseKernel.setArg(1, NormalisedHistogram);
		NormaliseKernel.setArg(2, SF);
		
		cl::Kernel BackProjectionKernel = cl::Kernel(program, "backProjection");
		BackProjectionKernel.setArg(0, dev_image_input);
		BackProjectionKernel.setArg(1, NormalisedHistogram);
		BackProjectionKernel.setArg(2, dev_image_output);

		cl::Event prof_event1;
		cl::Event prof_event2;
		cl::Event prof_event3;
		cl::Event prof_event4;

		//queue.enqueueNDRangeKernel(HistogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event1);
		queue.enqueueNDRangeKernel(LocalHistogramKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event1);//just comment this out and uncomment the one above out to swap to global mem histogram kernel
		queue.enqueueNDRangeKernel(CumulativeHistogramKernel, cl::NullRange, cl::NDRange(histogram_size), cl::NullRange, NULL, &prof_event2);
		queue.enqueueNDRangeKernel(NormaliseKernel, cl::NullRange, cl::NDRange(histogram_size), cl::NullRange, NULL, &prof_event3);
		queue.enqueueNDRangeKernel(BackProjectionKernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &prof_event4);
		
		//Copy the result from device to host and display
		vector<unsigned char> output_buffer(image_input.size());
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, image_input.size(), &output_buffer.data()[0]);
		CImg<unsigned char> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");

		typedef int mytype2;
		cout << endl;
		std::vector<mytype2> H((int)nr_bins);
		queue.enqueueReadBuffer(Histogram, CL_TRUE, 0, histogram_size, &H[0]);
		std::cout << "Histogram = " << H << std::endl;
		std::cout << "Histogram kernel execution time [ns]: " << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event1, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::vector<mytype2> CH((int)nr_bins);
		queue.enqueueReadBuffer(CumulativeHistogram, CL_TRUE, 0, histogram_size, &CH[0]);
		std::cout << "Cumulative Histogram = " << CH << std::endl;
		std::cout << "Cumulative Histogram kernel execution time [ns]: " << prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::vector<mytype2> NH((int)nr_bins);
		queue.enqueueReadBuffer(NormalisedHistogram, CL_TRUE, 0, histogram_size, &NH[0]);
		std::cout << "NH = " << NH << std::endl;
		std::cout << "Normalised Histogram kernel execution time [ns]: " << prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		std::cout << "Back Projection kernel execution time [ns]: " << prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << endl;
		cout << endl;

		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err.what() << std::endl;
	}

	return 0;
}
