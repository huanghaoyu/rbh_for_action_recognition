#ifndef RBH_H_
#define RBH_H_

#include "common.h"
#include "log.h"
#include "frame_reader.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.h>

struct Rbh
{
    static const int dctGridStep = 8;
    Mat spatialVarianceMap;
    Mat dcMap;
    Mat verticalVarianceMap;
    Mat horizontalVarianceMap;

    void Update(Frame& frame)
    {
        if(frame.dctMap.empty())
        	return;

        spatialVarianceMap = Mat::zeros(frame.RawImage.rows/dctGridStep, frame.RawImage.cols/dctGridStep, CV_32FC1);
        dcMap = Mat::zeros(frame.RawImage.rows/dctGridStep, frame.RawImage.cols/dctGridStep, CV_32FC1);
        verticalVarianceMap = Mat::zeros(frame.RawImage.rows/dctGridStep, frame.RawImage.cols/dctGridStep, CV_32FC1);
        horizontalVarianceMap = Mat::zeros(frame.RawImage.rows/dctGridStep, frame.RawImage.cols/dctGridStep, CV_32FC1);

        // spatial variance
        for(int blk_j = 0; blk_j < frame.RawImage.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < frame.RawImage.cols/dctGridStep; ++blk_i)
            {
            	float max = -1000.0;
                for(int j = 1; j < dctGridStep; ++j)
                    for(int i = 1; i < dctGridStep; ++i)
                    	if(frame.dctMap.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+i) > max)
                    		max = frame.dctMap.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+i);
                spatialVarianceMap.at<float>(blk_j, blk_i) = max;
            }
        }

        // dc
        for(int blk_j = 0; blk_j < frame.RawImage.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < frame.RawImage.cols/dctGridStep; ++blk_i)
            {
                dcMap.at<float>(blk_j, blk_i) = frame.dctMap.at<float>(blk_j*dctGridStep+0, blk_i*dctGridStep+0);
            }
        }

        // vertical variance
        for(int blk_j = 0; blk_j < frame.RawImage.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < frame.RawImage.cols/dctGridStep; ++blk_i)
            {
            	float max = -1000.0;
                for(int j = 1; j < dctGridStep; ++j)
                	if(frame.dctMap.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+0) > max)
                		max = frame.dctMap.at<float>(blk_j*dctGridStep+j, blk_i*dctGridStep+0);
                verticalVarianceMap.at<float>(blk_j, blk_i) = max;
            }
        }

        // horizontal variance
        for(int blk_j = 0; blk_j < frame.RawImage.rows/dctGridStep; ++blk_j)
        {
            for(int blk_i = 0; blk_i < frame.RawImage.cols/dctGridStep; ++blk_i)
            {
            	float max = -1000.0;
                for(int i = 1; i < dctGridStep; ++i)
                	if(frame.dctMap.at<float>(blk_j*dctGridStep+0, blk_i*dctGridStep+i) > max)
                		max = frame.dctMap.at<float>(blk_j*dctGridStep+0, blk_i*dctGridStep+i);
                horizontalVarianceMap.at<float>(blk_j, blk_i) = max;
            }
        }

        frame.spatialVarianceMap = spatialVarianceMap.clone();
        frame.dcMap = dcMap.clone();
        frame.verticalVarianceMap = verticalVarianceMap.clone();
        frame.horizontalVarianceMap = horizontalVarianceMap.clone();
    }
};

#endif /* RBH_H_ */
