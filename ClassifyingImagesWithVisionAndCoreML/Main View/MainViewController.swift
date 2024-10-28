/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The view controller that selects an image and makes a prediction using Vision and Core ML.
*/

import SwiftUI
import UIKit
import Dispatch
import LLM

class MainViewController: UIViewController {
    var firstRun = true
    let imagePredictor = ImagePredictor()
    let predictionsToShow = 1
    
    // MARK: Main storyboard outlets
    @IBOutlet weak var startupPrompts: UIStackView!
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var predictionLabel: UILabel!
    @IBOutlet weak var containerView: UIView!
    
    var testingBool = true
    
    // LLM-related properties
    private var botHostingController: UIHostingController<BotContainerView>?
    private var bot: Bot?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupLLMView()
    }
    
    private func setupLLMView() {
        // Create a container view that combines the bot view with any additional UI elements
        let botContainerView = BotContainerView()
        
        // Create a hosting controller to bridge SwiftUI to UIKit
        let hostingController = UIHostingController(rootView: botContainerView)
        
        // Add as child view controller
        addChild(hostingController)
        
        // Configure the frame and add to view hierarchy
        hostingController.view.frame = view.bounds
        hostingController.view.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(hostingController.view)
        
        // Complete the parent-child relationship
        hostingController.didMove(toParent: self)
        
        // Store reference
        self.botHostingController = hostingController
        
        // Ensure the LLM view doesn't cover other important UI elements
        //view.bringSubviewToFront(imageView)
        //view.bringSubviewToFront(predictionLabel)
        if let promptsView = startupPrompts {
            view.bringSubviewToFront(promptsView)
        }
    }
}

// SwiftUI container view to manage the Bot and ContentView
struct BotContainerView: View {
    @StateObject private var viewModel = BotContainerViewModel()
    
    var body: some View {
        VStack {
            Spacer() // Push the bot view to the bottom
            if let bot = viewModel.bot {
                BotView(bot)
                    .frame(height: 200) // Adjust height as needed
            } else {
                ProgressView(value: viewModel.progress) {
                    Text("Loading Hugging Face model...")
                } currentValueLabel: {
                    Text(String(format: "%.2f%%", viewModel.progress * 100))
                }
                .padding()
            }
        }
        .onAppear {
            viewModel.initializeBot()
        }
    }
}

// ViewModel to manage the Bot's state
class BotContainerViewModel: ObservableObject {
    @Published var bot: Bot?
    @Published var progress: CGFloat = 0
    
    func updateProgress(_ progress: Double) {
        self.progress = CGFloat(progress)
    }
    
    func initializeBot() {
        Task {
            let newBot = await Bot(updateProgress)
            await MainActor.run {
                self.bot = newBot
            }
        }
    }
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
        
        var inferenceTime: [Double] = []
        var inferenceTimeString: [String] = []
        
        let ImageTypes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "space", "nothing", "del"]
        for type in ImageTypes {
            for number in 1...3000 {
                let imageName = type + String(number)
                if let testImage = UIImage(named: imageName) {
                    print("Image Loaded: \(imageName)")
                    let startTime = DispatchTime.now()
                    self.classifyImage(testImage)
                    let endTime = DispatchTime.now()
                    
                    let nanoseconds = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
                    
                    // Measure inference latency
                    inferenceTime.append(Double(nanoseconds)/1_000_000_000.0)
                    inferenceTimeString.append(String(Double(nanoseconds)/1_000_000_000.0))
                    
                } else {
                    print("Image Failed to Load: \(imageName)")
                }
            }
        }
        print(inferenceTime)
        saveListToFile(list: inferenceTimeString, fileName: "BaseModelInferenceTime")
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
