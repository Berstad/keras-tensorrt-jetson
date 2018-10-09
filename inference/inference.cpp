#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>

#include "NvInfer.h"

#include "TFRTEngine.h"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;

#define INPUT_C 3
#define INPUT_H 299
#define INPUT_W 299

int main(int argc, char** argv) {
	std::cout << "Building engine..." << std::endl;
	int BATCH_SIZE = 1;
	auto build_start = std::chrono::high_resolution_clock::now();
	TFRTEngine engine = TFRTEngine();
	engine.addInput("input_1_1", DimsCHW(INPUT_C, INPUT_H, INPUT_W), sizeof(float));
	engine.addOutput("dense_1_1/Softmax", sizeof(float));

	bool success = engine.loadUff("./models/graph.pb.uff", (size_t) BATCH_SIZE, nvinfer1::DataType::kFLOAT);

	if (success) 
	{
		std::cout << engine.engineSummary() << std::endl;
	
	
		VideoCapture cap("testvideo.mpg"); // open our test video
	    const char* gst =  "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)I420, framerate=(fraction)120/1 ! \
				nvvidconv flip-method=6 ! video/x-raw, format=(string)BGRx ! \
				videoconvert ! video/x-raw, format=(string)BGR ! \
				appsink";
	    // VideoCapture cap(gst);
	
	    if(!cap.isOpened())  // check if we succeeded
	        return -1;
		std::cout << "Video stream opened!" << std::endl;
	    float num_frames = 0;
	    float totalMs = 0;
	    float pre_totalMs = 0;
	
		/* Allocate memory for predictions */
	   	std::vector<std::vector<void*>> batch(BATCH_SIZE);
	   	for (int b; b < BATCH_SIZE; b++) {
	   		batch[b].push_back(new unsigned char[INPUT_C * INPUT_H * INPUT_W * 4]);
	   	}
	   	float* data = new float[BATCH_SIZE*INPUT_C*INPUT_H*INPUT_W];

		std::cout << "Memory allocated for predictions!" << std::endl;
	   	auto build_end = std::chrono::high_resolution_clock::now();
	   	int build_ms = std::chrono::duration<float, std::milli>(build_end - build_start).count();
	   	float build_time = build_ms/1000;
	   	std::cout << "Time to build: " << build_time << std::endl;
	    for(;;)
	    {
	    	// Start timer
	    	auto t_start = std::chrono::high_resolution_clock::now();
	    	
	    	// New input frame
	        cv::Mat frame;
	        cap >> frame; // get a new frame from video
	        
	        // Check if video is over
	        if(frame.empty())
	        {
	        	cv::waitKey(1);
	        	break;
	        }
	        
	        // Convert color from BGRA to BGR
	        if (frame.channels() == 4)
	            cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
	
	        // Scale frame
	        cv::Size size(INPUT_H, INPUT_W);
	        cv::Mat frame_scaled;	
	        cv::resize(frame,frame_scaled,size);
	
			// Convert to tensor
			float pixelMean[3]{ 127.5f, 127.5f, 127.5f }; //Shortcut, change in future
			cv::Mat_<cv::Vec3f>::iterator it;
			unsigned volChl = INPUT_H*INPUT_W;
			for (int c = 0; c < INPUT_C; ++c)                              
			{
				cv::Mat_<cv::Vec3b>::iterator it = frame_scaled.begin<cv::Vec3b>();	//cv::Vec3f not working - reason still unknown...
				// the color image to input should be in BGR order
				for (unsigned j = 0; j < volChl; ++j)
				{
					//OpenCV read in frame as BGR format, by default, thus need only deduct the mean value
					data[c*volChl + j] = float((*it)[c]) - pixelMean[c];
					it++;
				}
			}
	
	
	        // Make input batch
	        std::vector<std::vector<void*>> batch(BATCH_SIZE);
	       	//for (int b; b < BATCH_SIZE; b++) {
	        batch[0].push_back(data);
	        //}
	      	
			auto t_mid = std::chrono::high_resolution_clock::now();
	      	// Do predictions and delete them
			std::vector<std::vector<void*>> outputs = engine.predict(batch);
	 		// End timer
	    	auto t_end = std::chrono::high_resolution_clock::now();
	   		//for (auto i: outputs[0])
	  		//	std::cout << i << ' ';
	    	for (int b = 0; b < outputs.size(); b++){
	    		// std::cout << outputs[b][0][0] << " , " << outputs[b][0][1] << '\n';
	    		delete outputs[b][0];
	    	}
	    	int ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
	    	totalMs += ms;
	    	num_frames += 1;
	    	int pre_ms = std::chrono::duration<float, std::milli>(t_mid - t_start).count();
	    	pre_totalMs += pre_ms;
	    	std::cout << "Frame number: " << num_frames << " | Time elapsed: " << ms << "ms | Time preprocessing: " << pre_ms << "ms" << std::endl;
	    }
	    float avg_fps = num_frames/(totalMs/1000);
	    float perc_pre = (pre_totalMs/totalMs)*100;
	    std::cout << "Total over " << num_frames << " runs is " << totalMs
	    				<< " ms, average FPS is: " << avg_fps  << std::endl;
	   	std::cout << "Total time preprocessing: " << pre_totalMs << "ms, or " << perc_pre << "% of total time." << std::endl; 
	    // the camera will be deinitialized automatically in VideoCapture destructor
	}
    return 0;
}
