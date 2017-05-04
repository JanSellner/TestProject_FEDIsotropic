#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "fed.h"
#include <iomanip>
#include <iostream>

static const float stabilityExplicit = 0.25f;   // Maxmim stability from the explicit scheme which is still stable
static const double lambda = 10;                // Controls the conductivity function; higher lambda -> more edges get blurred
static const int M = 10;                        // Number of FED cycles
static const float T = 100;                     // Total diffusion time

static double conductivity(const double magnitude)
{
    return 1 / (1 + std::pow(magnitude, 2) / std::pow(lambda, 2));
}

static cv::Mat conductivityImage(const cv::Mat& img)
{
    cv::Mat imgGradX, imgGradY;
    cv::Sobel(img, imgGradX, CV_64FC1, 1, 0);
    cv::Sobel(img, imgGradY, CV_64FC1, 0, 1);

    cv::Mat imgCond(img.rows, img.cols, CV_64FC1);
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            const double gradNorm = std::sqrt(imgGradX.at<double>(row, col) * imgGradX.at<double>(row, col) +
                                              imgGradY.at<double>(row, col) * imgGradY.at<double>(row, col));

            imgCond.at<double>(row, col) = conductivity(gradNorm);
        }
    }

    return imgCond;
}

static cv::Mat FEDInnerStep(const cv::Mat& img, const cv::Mat& cond, const double stepsize)
{
    // Copy input signal vector
    cv::Mat imgCopy = img.clone();

    // Apply the computation pattern to each image location
    for (int row = 0; row < img.rows; row++)
    {
        for (int col = 0; col < img.cols; col++)
        {
            double xLeft = 0;
            double xRight = 0;
            double yTop = 0;
            double yBottom = 0;

            if (col > 0)
            {
                // 3 <--> 4
                xLeft = (cond.at<double>(row, col - 1) + cond.at<double>(row, col)) * (img.at<double>(row, col - 1) - img.at<double>(row, col));
            }
            if (col < img.cols - 1)
            {
                // 4 <--> 5
                xRight = (cond.at<double>(row, col) + cond.at<double>(row, col + 1)) * (img.at<double>(row, col + 1) - img.at<double>(row, col));
            }
            if (row > 0)
            {
                // 1 <--> 4
                yTop = (cond.at<double>(row - 1, col) + cond.at<double>(row, col)) * (img.at<double>(row - 1, col) - img.at<double>(row, col));
            }
            if (row < img.rows - 1)
            {
                // 4 <--> 7
                yBottom = (cond.at<double>(row, col) + cond.at<double>(row + 1, col)) * (img.at<double>(row + 1, col) - img.at<double>(row, col));
            }

            // Update the current pixel location with the conductivity based derivative information and the varying step size
            imgCopy.at<double>(row, col) = 0.5 * stepsize * (xLeft + xRight + yTop + yBottom);
        }
    }

    // Update old image
    return img + imgCopy;
}

void main()
{
    cv::Mat img = cv::imread("GaussianScaleSpace_TrissQuarterResolution.jpg", CV_8UC1);
    cv::resize(img, img, cv::Size(700, (700.0 / img.cols) * img.rows), 0, 0, cv::INTER_CUBIC);
    cv::Mat imgDouble;
    img.convertTo(imgDouble, CV_64FC1);

    // Retrieve the different step sizes from the FED library; the same steps are used in all FED cycles
    std::vector<float> taus;
    const int n = fed_tau_by_process_time(T, M, stabilityExplicit, true, taus);

    // Meta information
    double sumDiffusionTime = 0.0;
    int iteration = 0;

    // T=0 is the original image
    std::stringstream stream;
    stream << "T=" << std::setfill('0') << std::setw(3) << iteration << ".png";
    cv::imwrite(stream.str(), imgDouble);   // Write image from the last iteration the current folder

    // Print array with the cumulated diffusion time
    std::cout << "cumDiffusionTime = [" << sumDiffusionTime;

    // M FED cycles
    for (size_t i = 0; i < M; ++i)
    {
        // Calculate the conductivity image again before each cycle starts
        cv::Mat cond = conductivityImage(imgDouble);

        // n inner FED steps
        for (size_t j = 0; j < n; ++j)
        {
            imgDouble = FEDInnerStep(imgDouble, cond, taus[j]);
            
            sumDiffusionTime += taus[j];
            ++iteration;

            stream.str("");
            stream << "T=" << std::setfill('0') << std::setw(3) << iteration << ".png";
            cv::imwrite(stream.str(), imgDouble);

            std::cout << ", " << sumDiffusionTime;
        }
    }

    std::cout << "];" << std::endl;

    cv::Mat imgFED;
    imgDouble.convertTo(imgFED, CV_8UC1);

    // Compare with standard Gaussian using the same diffusion time
    cv::Mat imgGauss;
    const double sigma = std::sqrt(2 * T);
    cv::GaussianBlur(img, imgGauss, cv::Size(), sigma);
    cv::imwrite("GaussBlurred.png", imgGauss);
}
