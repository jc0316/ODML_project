/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The view controller that selects an image and makes a prediction using Vision and Core ML.
*/

import Vision
import UIKit
import Dispatch

class MainViewController: UIViewController {
    var firstRun = true

    /// A predictor instance that uses Vision and Core ML to generate prediction strings from a photo.
    let imagePredictor = ImagePredictor()

    /// The largest number of predictions the main view controller displays the user.
    let predictionsToShow = 1

    // MARK: Main storyboard outlets
    @IBOutlet weak var startupPrompts: UIStackView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
    
    var testingBool = true
}

extension MainViewController {
    // MARK: Main storyboard actions
    /// The method the storyboard calls when the user one-finger taps the screen.
    @IBAction func singleTap() {
        // Show options for the source picker only if the camera is available.
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            present(photoPicker, animated: false)
            return
        }

        present(cameraPicker, animated: false)
    }

    /// The method the storyboard calls when the user two-finger taps the screen.
    @IBAction func doubleTap() {
        present(photoPicker, animated: false)
    }
    
    @IBAction func testModel(_ sender: Any) {
        print("Test Button Pressed")
        
        let defaultConfig = MLModelConfiguration()

        // Create an instance of the image classifier's wrapper class.
        
        let imageClassifierWrapper = try? model_original_int8_static_linear(configuration: defaultConfig)
        //let imageClassifierWrapper = try? MobileNet(configuration: defaultConfig)

        guard let imageClassifier = imageClassifierWrapper else {
            fatalError("App failed to create an image classifier model instance.")
        }

        // Get the underlying model instance.
        let imageClassifierModel = imageClassifier.model

        // Create a Vision instance using the image classifier's model instance.
        guard let imageClassifierVisionModel = try? VNCoreMLModel(for: imageClassifierModel) else {
            fatalError("App failed to create a `VNCoreMLModel` instance.")
        }
        
        var inferenceTime: [Double] = []
        
        var correctPredictions: Int = 0
        var totalPredictions: Int = 0
        
        let ImageTypes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"]
        for type in ImageTypes {
            for number in 1...3000 {
                let imageName = type + String(number)
                if let testImage = UIImage(named: imageName) {
                    print("Image Loaded: \(imageName)")
                    
                    classifyImage(testImage, model: imageClassifierVisionModel, correctString: type) { result, time, error in
                            if let error = error {
                                // Handle error case
                                print("Error:", error.localizedDescription)
                            } else if let result = result, let time = time {
                                // Handle success case
                                print("Classification result:", result)
                                print("Computation time:", time, "seconds")
                            } else {
                                // Handle unexpected cases if needed
                                print("Unexpected result with no errors.")
                            }
                            if result!.contains(type){
                                correctPredictions += 1
                            }
                            totalPredictions += 1
                            inferenceTime.append(time!)
                    }
                
                } else {
                    print("Image Failed to Load: \(imageName)")
                }
            }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 10.0) {
            //call any function
            print("Inference Time:")
            print(inferenceTime)
            print("Model Accuracy:")
            print(Double(correctPredictions)/Double(totalPredictions))
        }
    }
    
    func saveListToFile(list: [String], fileName: String) {
        let content = "[" + list.joined(separator: ",") + "]"
        
        if let documentDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = documentDirectory.appendingPathComponent(fileName).appendingPathExtension("txt")
            
            do {
                try content.write(to: fileURL, atomically: true, encoding: .utf8)
                print("File saved at: \(fileURL.path)")
            } catch {
                print("Error saving file: \(error)")
            }
        }
    }
}


extension MainViewController {
    // MARK: Main storyboard updates
    /// Updates the storyboard's image view.
    /// - Parameter image: An image.
    func updateImage(_ image: UIImage) {
        DispatchQueue.main.async {
            self.imageView.image = image
        }
    }

    /// Updates the storyboard's prediction label.
    /// - Parameter message: A prediction or message string.
    /// - Tag: updatePredictionLabel
    func updatePredictionLabel(_ message: String) {
        DispatchQueue.main.async {
            self.predictionLabel.text = message
        }

        if firstRun {
            DispatchQueue.main.async {
                self.firstRun = false
                self.predictionLabel.superview?.isHidden = false
                self.startupPrompts.isHidden = true
            }
        }
    }
    /// Notifies the view controller when a user selects a photo in the camera picker or photo library picker.
    /// - Parameter photo: A photo from the camera or photo library.
    func userSelectedPhoto(_ photo: UIImage) {
        updateImage(photo)
        updatePredictionLabel("Making predictions for the photo...")
        self.classifyImage(photo)
    }

}

extension MainViewController {
    // MARK: Image prediction methods
    /// Sends a photo to the Image Predictor to get a prediction of its content.
    /// - Parameter image: A photo.
    private func classifyImage(_ image: UIImage) {
        do {
            try self.imagePredictor.makePredictions(for: image,
                                                    completionHandler: imagePredictionHandler)
        } catch {
            print("Vision was unable to make a prediction...\n\n\(error.localizedDescription)")
        }
    }

    /// The method the Image Predictor calls when its image classifier model generates a prediction.
    /// - Parameter predictions: An array of predictions.
    /// - Tag: imagePredictionHandler
    private func imagePredictionHandler(_ predictions: [ImagePredictor.Prediction]?) {
        guard let predictions = predictions else {
            updatePredictionLabel("No predictions. (Check console log.)")
            return
        }

        let formattedPredictions = formatPredictions(predictions)

        let predictionString = formattedPredictions.joined(separator: "\n")
        updatePredictionLabel(predictionString)
    }

    /// Converts a prediction's observations into human-readable strings.
    /// - Parameter observations: The classification observations from a Vision request.
    /// - Tag: formatPredictions
    private func formatPredictions(_ predictions: [ImagePredictor.Prediction]) -> [String] {
        // Vision sorts the classifications in descending confidence order.
        let topPredictions: [String] = predictions.prefix(predictionsToShow).map { prediction in
            var name = prediction.classification

            // For classifications with more than one name, keep the one before the first comma.
            if let firstComma = name.firstIndex(of: ",") {
                name = String(name.prefix(upTo: firstComma))
            }

            return "\(name)"
        }

        return topPredictions
    }
}

extension MainViewController {
    
    
    func classifyImage(_ image: UIImage, model: VNCoreMLModel, correctString: String, completion: @escaping (String?, TimeInterval?, Error?) -> Void) {
        // Convert UIImage to CIImage
        guard let ciImage = CIImage(image: image) else {
            completion(nil, nil, NSError(domain: "ImageConversionError", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not convert UIImage to CIImage"]))
            return
        }
        
        // Start time measurement
        let startTime = Date()
        
        // Create a VNCoreMLRequest with the provided model
        let request = VNCoreMLRequest(model: model) { (request, error) in
            // End time measurement
            let endTime = Date()
            let computationTime = endTime.timeIntervalSince(startTime)
            
            if let error = error {
                completion(nil, computationTime, error)
                return
            }
            
            // Process classification results
            guard let results = request.results as? [VNClassificationObservation], let topResult = results.first else {
                completion(nil, computationTime, NSError(domain: "ClassificationError", code: -1, userInfo: [NSLocalizedDescriptionKey: "No classification results found"]))
                return
            }
            
            // Return the top classification result along with computation time
            completion(topResult.identifier, computationTime, nil)
        }
        
        request.imageCropAndScaleOption = .centerCrop
        
        // Create a VNImageRequestHandler and perform the request
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        DispatchQueue.global().async {
            do {
                try handler.perform([request])
            } catch {
                let endTime = Date()
                let computationTime = endTime.timeIntervalSince(startTime)
                completion(nil, computationTime, error)
            }
        }
    }
}
