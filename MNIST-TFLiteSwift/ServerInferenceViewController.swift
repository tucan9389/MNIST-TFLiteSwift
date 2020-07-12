//
//  ServerInferenceViewController.swift
//  MNIST-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/07/12.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import UIKit
import Alamofire

class ServerInferenceViewController: UIViewController {
    // MARK: - UI
    @IBOutlet weak var drawView: DrawView?
    @IBOutlet weak var predictLabel: UILabel?
    
    let context = CIContext()
    var pixelBuffer: CVPixelBuffer?

    // MARK: - VC Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        
        predictLabel?.text = ""
        
        setupPixelBuffer()
    }
    
    func setupPixelBuffer() {
        // Set the pixel buffer dimensions - Remember it's grayscale
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue] as CFDictionary
        CVPixelBufferCreate(kCFAllocatorDefault, 28, 28, kCVPixelFormatType_OneComponent8, attrs, &pixelBuffer)
    }
    
    @IBAction func clear(_ sender: Any) {
        drawView?.lines = []
        drawView?.setNeedsDisplay()
        predictLabel?.text = ""
    }
    
    @IBAction func classify(_ sender: Any) {
        guard let drawView = drawView, let predictLabel = predictLabel else { return }
        
        predictLabel.font = .systemFont(ofSize: 32, weight: .heavy)
        predictLabel.text = "Inference..."
        
        // get drawn image
        let viewContext = drawView.getViewContext()
        
        guard let cgImage = viewContext?.makeImage() else { return }
        
        let uiImage = UIImage(cgImage: cgImage)
        let imageData = uiImage.jpegData(compressionQuality: 1.0)
        
        AF.upload(multipartFormData: { (multipartFormData) in
            guard let imageData = imageData else { return }
            multipartFormData.append(imageData, withName: "mnist_image", fileName: "mnist_file.jpg", mimeType: "image/jpg")
        }, to: "http://127.0.0.1:5000/mnist")
        .validate()
        .response { response in
            guard let data = response.data,
                  let observation = try? JSONDecoder().decode(MNISTClassificationObservation.self, from: data) else {
                assert(false, "Error: cannot parse the result")
                return
            }
            
            print(observation.predictedNumber)
            predictLabel.font = .systemFont(ofSize: 100, weight: .heavy)
            predictLabel.text = "\(observation.predictedNumber)"
        }
    }
}

struct MNISTClassificationObservation: Decodable {
    let fileName: String
    let predictedNumber: Int
    
    private enum CodingKeys: String, CodingKey {
        case fileName = "filename"
        case predictedNumber = "pred_number"
    }
}
