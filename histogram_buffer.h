#include <vector>
#include <utility>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "integral_transform.h"
#include "diag.h"

using namespace cv;
using namespace std;

#ifndef __HISTOGRAM_BUFFER_H__
#define __HISTOGRAM_BUFFER_H__

struct HistogramBuffer
{
	vector<pair<Mat, Mat > > currentStack;
	vector<Mat> gluedIntegralTransforms;
	DescInfo descInfo;
	int tStride;

	HistogramBuffer(DescInfo descInfo, int tStride) : 
		descInfo(descInfo),
		tStride(tStride)
	{
		gluedIntegralTransforms.resize(descInfo.ntCells);
	}

	void AddUpCurrentStack()
	{
		Mat cumulativeIntegralTransform;
		for(int i = 0; i < currentStack.size(); i++)
		{
			Mat integralTransform = BuildOrientationIntegralTransform(descInfo, currentStack[i].first, currentStack[i].second);
			if(i == 0)
				cumulativeIntegralTransform = integralTransform;
			else
				cumulativeIntegralTransform += integralTransform;
		}

		rotate(gluedIntegralTransforms.begin(), ++gluedIntegralTransforms.begin(), gluedIntegralTransforms.end());
		gluedIntegralTransforms.back() = cumulativeIntegralTransform / tStride;
		currentStack.clear();
	}

	void QueryPatchDescriptor(Rect rect, float* res)
	{
		descInfo.ResetPatchDescriptorBuffer(res);
		for(int iT = 0; iT < descInfo.ntCells; iT++)
			ComputeDescriptor(gluedIntegralTransforms[iT], rect, descInfo, res + iT*descInfo.dim);
	}

	void Update(Mat dx, Mat dy)
	{
		currentStack.push_back(make_pair(dx, dy));	
	}
};

struct HofMbhBuffer
{
	Size frameSizeAfterInterpolation;
	bool print;
	bool AreDescriptorsReady;
	vector<int> effectiveFrameIndices;
	int tStride;
	int ntCells;
	double fScale;

	HistogramBuffer hog;
	HistogramBuffer hof;
	HistogramBuffer mbhX;
	HistogramBuffer mbhY;
    HistogramBuffer spatialVariance;
    HistogramBuffer dc;
    HistogramBuffer verticalVariance;
    HistogramBuffer horizontalVariance;

	Mat patchDescriptor;

	float* hog_patchDescriptor;
	float* hof_patchDescriptor;
	float* mbhX_patchDescriptor;
	float* mbhY_patchDescriptor;
    float* spatialVariance_patchDescriptor;
    float* dc_patchDescriptor;
    float* verticalVariance_patchDescriptor;
    float* horizontalVariance_patchDescriptor;

    DescInfo hogInfo, hofInfo, mbhInfo, spatialVarianceInfo, dcInfo, verticalVarianceInfo, horizontalVarianceInfo;

    void CreatePatchDescriptorPlaceholder(DescInfo& hogInfo, DescInfo& hofInfo, DescInfo& mbhInfo, DescInfo& spatialVarianceInfo, DescInfo& dcInfo,
                                          DescInfo& verticalVarianceInfo, DescInfo& horizontalVarianceInfo)
	{
		int size = (hogInfo.enabled ? hogInfo.fullDim : 0) 
			+ (hofInfo.enabled ? hofInfo.fullDim : 0)
            + (mbhInfo.enabled ? 2*mbhInfo.fullDim : 0)
            + (spatialVarianceInfo.enabled ? spatialVarianceInfo.fullDim : 0)
            + (dcInfo.enabled ? dcInfo.fullDim : 0)
            + (verticalVarianceInfo.enabled ? verticalVarianceInfo.fullDim : 0)
            + (horizontalVarianceInfo.enabled ? horizontalVarianceInfo.fullDim : 0);
		patchDescriptor.create(1, size, CV_32F);
		float* begin = patchDescriptor.ptr<float>();

		int used = 0;
		if(hogInfo.enabled)
		{
			hog_patchDescriptor = begin + used;
			used += hogInfo.fullDim;
		}
		if(hofInfo.enabled)
		{
			hof_patchDescriptor = begin + used;
			used += hofInfo.fullDim;
		}
		if(mbhInfo.enabled)
		{
			mbhX_patchDescriptor = begin + used;
			used += mbhInfo.fullDim;

			mbhY_patchDescriptor = begin + used;
			used += mbhInfo.fullDim;
		}
        if(spatialVarianceInfo.enabled)
        {
            spatialVariance_patchDescriptor = begin + used;
            used += spatialVarianceInfo.fullDim;
        }
        if(dcInfo.enabled)
        {
            dc_patchDescriptor = begin + used;
            used += dcInfo.fullDim;
        }
        if(verticalVarianceInfo.enabled)
        {
            verticalVariance_patchDescriptor = begin + used;
            used += verticalVarianceInfo.fullDim;
        }
        if(horizontalVarianceInfo.enabled)
        {
            horizontalVariance_patchDescriptor = begin + used;
            used += horizontalVarianceInfo.fullDim;
        }
	}

	HofMbhBuffer(
		DescInfo hogInfo, 
		DescInfo hofInfo, 
		DescInfo mbhInfo,
        DescInfo spatialVarianceInfo,
        DescInfo dcInfo,
        DescInfo verticalVarianceInfo,
        DescInfo horizontalVarianceInfo,
		int ntCells, 
		int tStride, 
		Size frameSizeAfterInterpolation, 
		double fScale, 
		bool print = false)
		: 
		frameSizeAfterInterpolation(frameSizeAfterInterpolation), 
		ntCells(ntCells),
		tStride(tStride),
		fScale(fScale),
		print(print),

		hof(hofInfo, tStride),
		mbhX(mbhInfo, tStride),
		mbhY(mbhInfo, tStride),
		hog(hogInfo, tStride),
        spatialVariance(spatialVarianceInfo, tStride),
        dc(dcInfo, tStride),
        verticalVariance(verticalVarianceInfo, tStride),
        horizontalVariance(horizontalVarianceInfo, tStride),

		hog_patchDescriptor(NULL), 
		hof_patchDescriptor(NULL),
		mbhX_patchDescriptor(NULL),
		mbhY_patchDescriptor(NULL),
        spatialVariance_patchDescriptor(NULL),
        dc_patchDescriptor(NULL),
        verticalVariance_patchDescriptor(NULL),
        horizontalVariance_patchDescriptor(NULL),

		hogInfo(hogInfo),
		hofInfo(hofInfo),
		mbhInfo(mbhInfo),
        spatialVarianceInfo(spatialVarianceInfo),
        dcInfo(dcInfo),
        verticalVarianceInfo(verticalVarianceInfo),
        horizontalVarianceInfo(horizontalVarianceInfo),
		AreDescriptorsReady(false)
	{
        CreatePatchDescriptorPlaceholder(hogInfo, hofInfo, mbhInfo, spatialVarianceInfo, dcInfo,
                                         verticalVarianceInfo, horizontalVarianceInfo);
	}

	void Update(Frame& frame)
	{
		if(hofInfo.enabled)
		{
			TIMERS.HofComputation.Start();
			hof.Update(frame.Dx, frame.Dy);
			TIMERS.HofComputation.Stop();
		}

		if(mbhInfo.enabled)
		{
			TIMERS.MbhComputation.Start();
			Mat flowXdX, flowXdY, flowYdX, flowYdY;
			Sobel(frame.Dx, flowXdX, CV_32F, 1, 0, 1);
			Sobel(frame.Dx, flowXdY, CV_32F, 0, 1, 1);
			Sobel(frame.Dy, flowYdX, CV_32F, 1, 0, 1);
			Sobel(frame.Dy, flowYdY, CV_32F, 0, 1, 1);
			mbhX.Update(flowXdX, flowXdY);
			mbhY.Update(flowYdX, flowYdY);
			TIMERS.MbhComputation.Stop();
		}

		if(hogInfo.enabled)
		{
			TIMERS.HogComputation.Start();
            Mat dx, dy;
            Sobel(frame.RawImage, dx, CV_32F, 1, 0, 1);
            Sobel(frame.RawImage, dy, CV_32F, 0, 1, 1);
			hog.Update(dx, dy);
			TIMERS.HogComputation.Stop();
		}

        if(spatialVarianceInfo.enabled)
        {
            TIMERS.SpatialVarianceComputation.Start();
            Mat dx, dy;
            Sobel(frame.spatialVarianceMap, dx, CV_32F, 1, 0, 1);
            Sobel(frame.spatialVarianceMap, dy, CV_32F, 0, 1, 1);
            spatialVariance.Update(dx, dy);
            TIMERS.SpatialVarianceComputation.Stop();
        }

        if(dcInfo.enabled)
        {
            TIMERS.DcComputation.Start();
            Mat dx, dy;
            Sobel(frame.dcMap, dx, CV_32F, 1, 0, 1);
            Sobel(frame.dcMap, dy, CV_32F, 0, 1, 1);
            dc.Update(dx, dy);
            TIMERS.DcComputation.Stop();
        }

        if(verticalVarianceInfo.enabled)
        {
            TIMERS.VerticalVarianceComputation.Start();
            Mat dx, dy;
            Sobel(frame.verticalVarianceMap, dx, CV_32F, 1, 0, 1);
            Sobel(frame.verticalVarianceMap, dy, CV_32F, 0, 1, 1);
            verticalVariance.Update(dx, dy);
            TIMERS.VerticalVarianceComputation.Stop();
        }

        if(horizontalVarianceInfo.enabled)
        {
            TIMERS.HorizontalVarianceComputation.Start();
            Mat dx, dy;
            Sobel(frame.horizontalVarianceMap, dx, CV_32F, 1, 0, 1);
            Sobel(frame.horizontalVarianceMap, dy, CV_32F, 0, 1, 1);
            horizontalVariance.Update(dx, dy);
            TIMERS.HorizontalVarianceComputation.Stop();
        }

		effectiveFrameIndices.push_back(frame.PTS);
		AreDescriptorsReady = false;
		if(effectiveFrameIndices.size() % tStride == 0)
		{
			if(hofInfo.enabled)
			{
				TIMERS.HofComputation.Start();
				hof.AddUpCurrentStack();
				TIMERS.HofComputation.Stop();
			}

			if(mbhInfo.enabled)
			{
				TIMERS.MbhComputation.Start();
				mbhX.AddUpCurrentStack();
				mbhY.AddUpCurrentStack();
				TIMERS.MbhComputation.Stop();
			}

			if(hogInfo.enabled)
			{
				TIMERS.HogComputation.Start();
				hog.AddUpCurrentStack();
				TIMERS.HogComputation.Stop();
			}

            if(spatialVarianceInfo.enabled)
            {
                TIMERS.SpatialVarianceComputation.Start();
                spatialVariance.AddUpCurrentStack();
                TIMERS.SpatialVarianceComputation.Stop();
            }

            if(dcInfo.enabled)
            {
                TIMERS.DcComputation.Start();
                dc.AddUpCurrentStack();
                TIMERS.DcComputation.Stop();
            }

            if(verticalVarianceInfo.enabled)
            {
                TIMERS.VerticalVarianceComputation.Start();
                verticalVariance.AddUpCurrentStack();
                TIMERS.VerticalVarianceComputation.Stop();
            }

            if(horizontalVarianceInfo.enabled)
            {
                TIMERS.HorizontalVarianceComputation.Start();
                horizontalVariance.AddUpCurrentStack();
                TIMERS.HorizontalVarianceComputation.Stop();
            }

			AreDescriptorsReady = effectiveFrameIndices.size() >= ntCells * tStride;
		}
	}

	void PrintFileHeader()
	{
//		printf("#descr = ");
//		if(hogInfo.enabled)
//			printf("hog (%d) ", hogInfo.fullDim);
//		if(hofInfo.enabled)
//			printf("hof (%d) ", hofInfo.fullDim);
//		if(mbhInfo.enabled)
//			printf("mbh (%d + %d)", mbhInfo.fullDim, mbhInfo.fullDim);
//		printf("\n#x\ty\tpts\tStartPTS\tEndPTS\tXoffset\tYoffset\tPatchWidth\tPatchHeight\tdescr\n");
	}

	void PrintPatchDescriptorHeader(Rect rect)
	{
//		int firstFrame = effectiveFrameIndices[effectiveFrameIndices.size()-ntCells*tStride];
//		int lastFrame = effectiveFrameIndices.back();
//		Point patchCenter(rect.x + rect.width/2, rect.y + rect.height/2);
//		printf("%.2lf\t%.2lf\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t",
//			double(patchCenter.x) / frameSizeAfterInterpolation.width,
//			double(patchCenter.y) / frameSizeAfterInterpolation.height,
//			(firstFrame + lastFrame)/2,
//			firstFrame,
//			lastFrame,
//			int(rect.x / fScale),
//			int(rect.y / fScale),
//			int(rect.width / fScale),
//			int(rect.height / fScale));
	}

	void PrintPatchDescriptor(Rect rect)
	{
		TIMERS.DescriptorQuerying.Start();
		if(hofInfo.enabled)
		{
			TIMERS.HofQuerying.Start();
			hof.QueryPatchDescriptor(rect, hof_patchDescriptor);
			TIMERS.HofQuerying.Stop();
		}
		if(mbhInfo.enabled)
		{
			TIMERS.MbhQuerying.Start();
			mbhX.QueryPatchDescriptor(rect, mbhX_patchDescriptor);
			mbhY.QueryPatchDescriptor(rect, mbhY_patchDescriptor);
			TIMERS.MbhQuerying.Stop();
		}
		if(hogInfo.enabled)
		{
			TIMERS.HogQuerying.Start();
			hog.QueryPatchDescriptor(rect, hog_patchDescriptor);
			TIMERS.HogQuerying.Stop();
		}
        if(spatialVarianceInfo.enabled)
        {
            TIMERS.SpatialVarianceQuerying.Start();
            spatialVariance.QueryPatchDescriptor(rect, spatialVariance_patchDescriptor);
            TIMERS.SpatialVarianceQuerying.Stop();
        }
        if(dcInfo.enabled)
        {
            TIMERS.DcQuerying.Start();
            dc.QueryPatchDescriptor(rect, dc_patchDescriptor);
            TIMERS.DcQuerying.Stop();
        }
        if(verticalVarianceInfo.enabled)
        {
            TIMERS.VerticalVarianceQuerying.Start();
            verticalVariance.QueryPatchDescriptor(rect, verticalVariance_patchDescriptor);
            TIMERS.VerticalVarianceQuerying.Stop();
        }
        if(horizontalVarianceInfo.enabled)
        {
            TIMERS.HorizontalVarianceQuerying.Start();
            horizontalVariance.QueryPatchDescriptor(rect, horizontalVariance_patchDescriptor);
            TIMERS.HorizontalVarianceQuerying.Stop();
        }
		TIMERS.DescriptorQuerying.Stop();
		
		if(print)
		{
			TIMERS.Writing.Start();
			PrintPatchDescriptorHeader(rect);
			PrintFloatArray(patchDescriptor);
			
			printf("\n");
			TIMERS.Writing.Stop();
			
		}
	}

	void PrintFullDescriptor(int blockWidth, int blockHeight, int xStride, int yStride)
	{
		for(int xOffset = 0; xOffset + blockWidth < frameSizeAfterInterpolation.width; xOffset += xStride)
		{
			for(int yOffset = 0; yOffset + blockHeight < frameSizeAfterInterpolation.height; yOffset += yStride)
			{
				Rect rect(xOffset, yOffset, blockWidth, blockHeight);
				PrintPatchDescriptor(rect);
			}
		}
	}
};

#endif
