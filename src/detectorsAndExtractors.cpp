#include <opencv2/core/core.hpp>
#ifdef OPENCV2P4
#include <opencv2/nonfree/nonfree.hpp>
#endif
#include "detectorsAndExtractors.h"

// ------------------- DETECTORS -------------------

cv::Ptr<cv::FeatureDetector> createSTAR(const boost::python::dict& settings)
{
    int maxSize = 32;
    int responseThreshold = 30;
    int lineThreshold = 10;
    int lineBinarized = 8;
    int suppressNonmaxSize = 5;
    
    if (settings.has_key("MaxSize")) {
        maxSize = boost::python::extract<int>(settings.get("MaxSize"));
    }
    if (settings.has_key("Response")) {
        responseThreshold = boost::python::extract<int>(settings.get("Response"));
    }
    if (settings.has_key("LineThreshold")) {
        lineThreshold = boost::python::extract<int>(settings.get("LineThreshold"));
    }
    if (settings.has_key("LineBinarized")) {
        lineBinarized = boost::python::extract<int>(settings.get("LineBinarized"));
    }
    if (settings.has_key("Suppression")) {
        suppressNonmaxSize = boost::python::extract<int>(settings.get("Suppression"));
    }
    
    return cv::makePtr<cv::StarFeatureDetector>(maxSize, responseThreshold, lineThreshold, lineBinarized, suppressNonmaxSize);
}

cv::Ptr<cv::FeatureDetector> createFAST(const boost::python::dict& settings)
{
    int threshold = 10;
    bool nonmaxSuppression = true;
    
    if (settings.has_key("Threshold")) {
        threshold = boost::python::extract<int>(settings.get("Threshold"));
    }
    if (settings.has_key("NonMaxSuppression")) {
        nonmaxSuppression = boost::python::extract<bool>(settings.get("NonMaxSuppression"));
    }
    
    return cv::makePtr<cv::FastFeatureDetector>(threshold, nonmaxSuppression);
}

cv::Ptr<cv::FeatureDetector> createSURF(const boost::python::dict& settings)
{
    double hessianThreshold = 400;
    int nOctaves = 4;
    int nOctaveLayers = 2;
    bool extended = true;
    bool upright = false;
    
    if (settings.has_key("HessianThreshold")) {
        hessianThreshold = boost::python::extract<double>(settings.get("HessianThreshold"));
    }
    if (settings.has_key("NumOctaves")) {
        nOctaves = boost::python::extract<int>(settings.get("NumOctaves"));
    }
    if (settings.has_key("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.has_key("Extended")) {
        extended = boost::python::extract<bool>(settings.get("Extended"));
    }
    if (settings.has_key("Upright")) {
        upright = boost::python::extract<bool>(settings.get("Upright"));
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SURF>(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
#else
    return cv::makePtr<cv::SurfFeatureDetector>(hessianThreshold, nOctaves, nOctaveLayers, upright);
#endif
}

cv::Ptr<cv::FeatureDetector> createSIFT(const boost::python::dict& settings)
{
    int numFeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    
    if (settings.has_key("NumFeatures")) {
        numFeatures = boost::python::extract<int>(settings.get("NumFeatures"));
    }
    if (settings.has_key("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.has_key("ContrastThreshold")) {
        contrastThreshold = boost::python::extract<double>(settings.get("ContrastThreshold"));
    }
    if (settings.has_key("EdgeThreshold")) {
        edgeThreshold = boost::python::extract<double>(settings.get("EdgeThreshold"));
    }
    if (settings.has_key("Sigma")) {
        sigma = boost::python::extract<double>(settings.get("Sigma"));
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SIFT>(numFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
    return cv::makePtr<cv::SiftFeatureDetector>(contrastThreshold, edgeThreshold);
#endif
}

cv::Ptr<cv::FeatureDetector> createMSER(const boost::python::dict& settings)
{
    int delta = 5;
    int minArea = 60;
    int maxArea = 14400;
    double maxVariation = 0.25;
    double minDiversity = 0.2;
    double maxEvolution = 200;
    double areaThreshold = 1.01;
    double minMargin = 0.003;
    int edgeBlurSize = 5;
    
    if (settings.has_key("Delta")) {
        delta = boost::python::extract<int>(settings.get("Delta"));
    }
    if (settings.has_key("MinArea")) {
        minArea = boost::python::extract<int>(settings.get("MinArea"));
    }
    if (settings.has_key("MaxArea")) {
        maxArea = boost::python::extract<int>(settings.get("MaxArea"));
    }
    if (settings.has_key("MaxVariation")) {
        maxVariation = boost::python::extract<double>(settings.get("MaxVariation"));
    }
    if (settings.has_key("MinDiversity")) {
        minDiversity = boost::python::extract<double>(settings.get("MinDiversity"));
    }
    if (settings.has_key("MaxEvolution")) {
        maxEvolution = boost::python::extract<double>(settings.get("MaxEvolution"));
    }
    if (settings.has_key("AreaThreshold")) {
        areaThreshold = boost::python::extract<double>(settings.get("AreaThreshold"));
    }
    if (settings.has_key("MinMargin")) {
        minMargin = boost::python::extract<double>(settings.get("MinMargin"));
    }
    if (settings.has_key("EdgeBlurSize")) {
        edgeBlurSize = boost::python::extract<int>(settings.get("EdgeBlurSize"));
    }
    
    return cv::makePtr<cv::MserFeatureDetector>(delta, minArea, maxArea, maxVariation, minDiversity, maxEvolution, areaThreshold, minMargin, edgeBlurSize);
}

/**
 * Generates a feature detector based on options in the settings dict.
 * Does some fiddling for the setttings structure.
 * Will work with no settings specified, defaults to a STAR detector in STATIC detector mode.
 * Individual detector settings default to as in the OpenCV documentation, or as in the
 * sample openFABMAP settings where no OpenCV default.
 * 
 * @param settings A Python dict of settings, the full settings object.
 * @return A cv::FeatureDetector pointer, as a cv::Ptr (for OpenCV compatibility)
 */
cv::Ptr<cv::FeatureDetector> pyof2::generateDetector(const boost::python::dict &settings) {
    // Get the feature settings
    boost::python::dict featureOptions;
    if (settings.has_key("FeatureOptions"))
    {
        featureOptions = boost::python::extract<boost::python::dict>(settings.get("FeatureOptions"));
    }
    
    // Read the settings, with default values.
    std::string detectorMode = "STATIC";
    std::string detectorType = "STAR";
    if (featureOptions.has_key("DetectorMode")) {
        detectorMode = boost::python::extract<std::string>(featureOptions.get("DetectorMode"));
    }
    if (featureOptions.has_key("DetectorType")) {
        detectorType = boost::python::extract<std::string>(featureOptions.get("DetectorType"));
    }
    
    // 
    if(detectorMode == "ADAPTIVE") {

        if(detectorType != "STAR" && detectorType != "SURF" && detectorType != "FAST") {
            //Adaptive Detectors only work with STAR, SURF and FAST
            detectorType = "STAR";
        }
        
        // Get the settings for adaptive features
        boost::python::dict adaptiveOptions;
        if (featureOptions.has_key("Adaptive"))
        {
            adaptiveOptions = boost::python::extract<boost::python::dict>(featureOptions.get("Adaptive"));
        }
        
        // Defaults from the OpenCV documentation
        int minFeatures = 400;
        int maxFeatures = 500;
        int maxIters = 5;
        if (adaptiveOptions.has_key("MinFeatures")) {
            minFeatures = boost::python::extract<int>(adaptiveOptions.get("MinFeatures"));
        }
        if (adaptiveOptions.has_key("MaxFeatures")) {
            maxFeatures = boost::python::extract<int>(adaptiveOptions.get("MaxFeatures"));
        }
        if (adaptiveOptions.has_key("MaxIters")) {
            maxIters = boost::python::extract<int>(adaptiveOptions.get("MaxIters"));
        }
        return cv::makePtr<cv::DynamicAdaptedFeatureDetector>(cv::AdjusterAdapter::create(detectorType), minFeatures, maxFeatures, maxIters);

    } else {
        boost::python::dict detectorOptions;
        if(detectorType == "FAST") {
            if (featureOptions.has_key("FastDetector"))
            {
                detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("FastDetector"));
            }
            return createFAST(detectorOptions);
        } else if(detectorType == "SURF") {
            if (featureOptions.has_key("SurfDetector"))
            {
                detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("SurfDetector"));
            }
            return createSURF(detectorOptions);
        } else if(detectorType == "SIFT") {
            if (featureOptions.has_key("SiftDetector"))
            {
                detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("SiftDetector"));
            }
            return createSIFT(detectorOptions);
        } else if(detectorType == "MSER") {
            if (featureOptions.has_key("MSERDetector"))
            {
                detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("MSERDetector"));
            }
            return createMSER(detectorOptions);
        } else {
            if (featureOptions.has_key("StarDetector"))
            {
                detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("StarDetector"));
            }
            return createSTAR(detectorOptions);
        }
    }
}

// ------------------- EXTRACTORS -------------------

cv::Ptr<cv::DescriptorExtractor> createSIFTExtractor(const boost::python::dict& settings)
{
    int numFeatures = 0;
    int nOctaveLayers = 3;
    double contrastThreshold = 0.04;
    double edgeThreshold = 10;
    double sigma = 1.6;
    
    if (settings.has_key("NumFeatures")) {
        numFeatures = boost::python::extract<int>(settings.get("NumFeatures"));
    }
    if (settings.has_key("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.has_key("ContrastThreshold")) {
        contrastThreshold = boost::python::extract<double>(settings.get("ContrastThreshold"));
    }
    if (settings.has_key("EdgeThreshold")) {
        edgeThreshold = boost::python::extract<double>(settings.get("EdgeThreshold"));
    }
    if (settings.has_key("Sigma")) {
        sigma = boost::python::extract<double>(settings.get("Sigma"));
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SIFT>(numFeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
#else
    return cv::makePtr<cv::SiftDescriptorExtractor>();
#endif
}

cv::Ptr<cv::DescriptorExtractor> createSURFExtractor(const boost::python::dict& settings)
{
    double hessianThreshold = 400;
    int nOctaves = 4;
    int nOctaveLayers = 2;
    bool extended = true;
    bool upright = false;
    
    if (settings.has_key("HessianThreshold")) {
        hessianThreshold = boost::python::extract<double>(settings.get("HessianThreshold"));
    }
    if (settings.has_key("NumOctaves")) {
        nOctaves = boost::python::extract<int>(settings.get("NumOctaves"));
    }
    if (settings.has_key("NumOctaveLayers")) {
        nOctaveLayers = boost::python::extract<int>(settings.get("NumOctaveLayers"));
    }
    if (settings.has_key("Extended")) {
        extended = boost::python::extract<bool>(settings.get("Extended"));
    }
    if (settings.has_key("Upright")) {
        upright = boost::python::extract<bool>(settings.get("Upright"));
    }
    
#ifdef OPENCV2P4
    return cv::makePtr<cv::SURF>(hessianThreshold, nOctaves, nOctaveLayers, extended, upright);
#else
    return cv::makePtr<cv::SurfDescriptorExtractor>(nOctaves, nOctaveLayers, extended, upright);
#endif
}

/**
 * Generates a feature detector based on options in the settings file
 * 
 * @param settings A python dict of settings,
 * @return 
 */
cv::Ptr<cv::DescriptorExtractor> pyof2::generateExtractor(const boost::python::dict &settings)
{
    // Get the feature settings
    boost::python::dict featureOptions;
    if (settings.has_key("FeatureOptions"))
    {
        featureOptions = boost::python::extract<boost::python::dict>(settings.get("FeatureOptions"));
    }
    
    std::string extractorType = "SURF";
    if (featureOptions.has_key("ExtractorType")) {
        extractorType = boost::python::extract<std::string>(featureOptions.get("ExtractorType"));
    }
    
    boost::python::dict detectorOptions;
    if(extractorType == "SIFT") {
        if (featureOptions.has_key("SiftDetector"))
        {
            detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("SiftDetector"));
        }
        return createSIFTExtractor(detectorOptions);
    } else {
        if (featureOptions.has_key("SurfDetector"))
        {
            detectorOptions = boost::python::extract<boost::python::dict>(featureOptions.get("SurfDetector"));
        }
        return createSURFExtractor(detectorOptions);
    }
}