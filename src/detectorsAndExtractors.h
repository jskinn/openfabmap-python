#ifndef DETECTORS_AND_EXTRACTORS_H
#define DETECTORS_AND_EXTRACTORS_H

#include <memory>
#include <Python.h>
#include <boost/python.hpp>
//#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace pyof2 {
    cv::Ptr<cv::FeatureDetector> generateDetector(const boost::python::dict &settings);
    cv::Ptr<cv::DescriptorExtractor> generateExtractor(const boost::python::dict &settings);
}

#endif // DETECTORS_AND_EXTRACTORS_H
