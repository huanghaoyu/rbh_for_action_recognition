#include <string>
#include <cstdio>
#include <algorithm>
#include <opencv/cv.h>
#include <opencv/cxcore.h>

#include "motion_vector_file_utils.h"
#include "log.h"
#include "frame_reader.h"
#include "histogram_buffer.h"
#include "desc_info.h"
#include "options.h"
#include "diag.h"
#include "rbh.h"

#include <iterator>
#include <vector>

using namespace std;
using namespace cv;

//void WriteData(Mat& matData)
//{
//    if(matData.empty())
//    {
//        cout<<"Mat empty!"<<endl;
//        return;
//    }
//    for(int r=0; r<matData.rows; ++r)
//    {
//        for(int c=0; c<matData.cols; ++c)
//        {
//            cout<<matData.at<float>(r,c)<<'\t';
//        }
//        cout<<endl;
//    }
//}

int main(int argc, char* argv[])
{
	Options opts(argc, argv);

	const int nt_cell = 3;
	const int tStride = 5;
	vector<Size> patchSizes;
	patchSizes.push_back(Size(32, 32));
	patchSizes.push_back(Size(48, 48));

	DescInfo hofInfo(8+1, true, nt_cell, opts.HofEnabled);
	DescInfo mbhInfo(8, false, nt_cell, opts.MbhEnabled);
	DescInfo hogInfo(8, false, nt_cell, opts.HogEnabled);
    DescInfo spatialVarianceInfo(8, false, nt_cell, opts.SpatialVarianceEnabled);
    DescInfo dcInfo(8, false, nt_cell, opts.DcEnabled);
    DescInfo verticalVarianceInfo(8, false, nt_cell, opts.VerticalVarianceEnabled);
    DescInfo horizontalVarianceInfo(8, false, nt_cell, opts.HorizontalVarianceEnabled);

	TIMERS.Reading.Start();
    FrameReader rdr(opts.VideoPath, hogInfo.enabled);
	TIMERS.Reading.Stop();

//    VideoCapture videoCapture(opts.VideoPath);
//    Mat cap;
//    int frameNum = int(videoCapture.get(CV_CAP_PROP_FRAME_COUNT));
//    frameNum -= 3;
//    int frameIndex = 1;

	Size frameSizeAfterInterpolation = 
		opts.Interpolation
			? Size(2*rdr.DownsampledFrameSize.width - 1, 2*rdr.DownsampledFrameSize.height - 1)
			: rdr.DownsampledFrameSize;
	int cellSize = rdr.OriginalFrameSize.width / frameSizeAfterInterpolation.width;
	double fscale = 1 / 8.0;

	log("Frame count:\t%d", rdr.FrameCount);
	log("Original frame size:\t%dx%d", rdr.OriginalFrameSize.width, rdr.OriginalFrameSize.height);
	log("Downsampled:\t%dx%d", rdr.DownsampledFrameSize.width, rdr.DownsampledFrameSize.height);
    log("After interpolation:\t%dx%d", frameSizeAfterInterpolation.width, frameSizeAfterInterpolation.height);
	log("CellSize:\t%d", cellSize);

    HofMbhBuffer buffer(hogInfo, hofInfo, mbhInfo, spatialVarianceInfo, dcInfo, verticalVarianceInfo, horizontalVarianceInfo,
                        nt_cell, tStride, frameSizeAfterInterpolation, fscale, true);
    buffer.PrintFileHeader();

    Rbh rbh;

	TIMERS.Everything.Start();
	while(true)
	{
//        if(frameIndex > frameNum)
//            break;
        Frame frame = rdr.Read();
//        videoCapture.operator >>(cap);
        if(frame.PTS == -1 /*|| cap.empty()*/)
			break;
//        frame.RawImage = cap.clone();
//        ++frameIndex;

		log("#read frame pts=%d, mvs=%s, type=%c", frame.PTS, frame.NoMotionVectors ? "no" : "yes", frame.PictType);

		if(opts.GoodPts.empty() || count(opts.GoodPts.begin(), opts.GoodPts.end(), frame.PTS) == 1)
		{
			TIMERS.DescriptorComputation.Start();
			
            if(frame.NoMotionVectors || frame.RawImage.empty())
			{
				TIMERS.SkippedFrames++;
				continue;
			}

            rbh.Update(frame);
            frame.Interpolate(frameSizeAfterInterpolation, fscale);
            buffer.Update(frame);
			TIMERS.DescriptorComputation.Stop();
		
            if(buffer.AreDescriptorsReady)
            {
                for(int k = 0; k < patchSizes.size(); k++)
                {
                    int blockWidth = patchSizes[k].width / cellSize;
                    int blockHeight = patchSizes[k].height / cellSize;
                    int xStride = opts.Dense ? 1 : blockWidth / 2;
                    int yStride = opts.Dense ? 1 : blockHeight / 2;
                    buffer.PrintFullDescriptor(blockWidth, blockHeight, xStride, yStride, rdr.FrameCount);
                }
                buffer.t++;
            }
		}
	}
    TIMERS.Everything.Stop();
	TIMERS.Print(rdr.FrameCount);
 }
