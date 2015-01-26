#include <cstdio>
#include <ctime>

#include "timing.h"

#ifndef __DIAG_H__
#define __DIAG_H__

struct Diag
{
	Timer HogComputation;
	Timer HofComputation;
	Timer MbhComputation;
    Timer SpatialVarianceComputation;
    Timer DcComputation;
    Timer VerticalVarianceComputation;
    Timer HorizontalVarianceComputation;

	Timer InterpolationHOFMBH;
	Timer InterpolationHOG;

	Timer DescriptorComputation;
	Timer DescriptorQuerying;

	Timer HofQuerying;
	Timer MbhQuerying;
	Timer HogQuerying;
    Timer SpatialVarianceQuerying;
    Timer DcQuerying;
    Timer VerticalVarianceQuerying;
    Timer HorizontalVarianceQuerying;

	Timer Everything;
	Timer Reading;
	Timer ReadingAndDecoding;
	Timer Writing;

	int CallsComputeDescriptor;
	int SkippedFrames;

	Diag() : CallsComputeDescriptor(0), SkippedFrames(0) {}

	void Print(int frameCount)
	{
		log("Reading (sec):\t%.2lf", Reading.TotalInSeconds());
		log("Decoding (sec):\t%.2lf", ReadingAndDecoding.TotalInSeconds() - Reading.TotalInSeconds());

        log("Interp (sec):\t%.2lf", InterpolationHOFMBH.TotalInSeconds() + InterpolationHOG.TotalInSeconds());
		log("Interp.HOFMBH (sec):\t%.2lf", InterpolationHOFMBH.TotalInSeconds());
		
		log("IntHist (sec):\t%.2lf", DescriptorComputation.TotalInSeconds());
		log("IntHist.HOG (sec):\t%.2lf", HogComputation.TotalInSeconds());
		log("IntHist.HOF (sec):\t%.2lf", HofComputation.TotalInSeconds());
		log("IntHist.MBH (sec):\t%.2lf", MbhComputation.TotalInSeconds());
        log("IntHist.SpatialVariance (sec):\t%.2lf", SpatialVarianceComputation.TotalInSeconds());
        log("IntHist.DC (sec):\t%.2lf", DcComputation.TotalInSeconds());
        log("IntHist.VerticalVariance (sec):\t%.2lf", VerticalVarianceComputation.TotalInSeconds());
        log("IntHist.HorizontalVariance (sec):\t%.2lf", HorizontalVarianceComputation.TotalInSeconds());

		log("Interp.HOG (sec):\t%.2lf", InterpolationHOG.TotalInSeconds());
		
		log("Desc (sec):\t%.2lf", DescriptorQuerying.TotalInSeconds());
		log("Desc.HOG (sec):\t%.2lf", HogQuerying.TotalInSeconds());
		log("Desc.HOF (sec):\t%.2lf", HofQuerying.TotalInSeconds());
        log("Desc.MBH (sec):\t%.2lf", MbhQuerying.TotalInSeconds());
        log("Desc.SpatialVariance (sec):\t%.2lf", SpatialVarianceQuerying.TotalInSeconds());
        log("Desc.DC (sec):\t%.2lf", DcQuerying.TotalInSeconds());
        log("Desc.VerticalVariance (sec):\t%.2lf", VerticalVarianceQuerying.TotalInSeconds());
        log("Desc.HorizontalVariance (sec):\t%.2lf", HorizontalVarianceQuerying.TotalInSeconds());

		log("Writing (sec):\t%.2lf", Writing.TotalInSeconds());

		double totalWithoutWriting = Everything.TotalInSeconds() - Writing.TotalInSeconds();
		log("Total (sec):\t%.2lf", totalWithoutWriting);
		log("Total (with writing, sec):\t%.2lf", Everything.TotalInSeconds());

		log("Fps:\t%.2lf", frameCount / totalWithoutWriting);
		log("Calls.ComputeDescriptor:\t%d", CallsComputeDescriptor);
		log("Frames:\t%d", frameCount);
		log("Frames.Skipped:\t%d", SkippedFrames);
	}
} TIMERS;

#endif
