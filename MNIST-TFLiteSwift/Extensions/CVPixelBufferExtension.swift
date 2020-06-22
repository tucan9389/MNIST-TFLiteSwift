/*
 * Copyright Doyoung Gwak 2020
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//
//  CVPixelBufferExtension.swift
//  PoseEstimation-TFLiteSwift
//
//  Created by Doyoung Gwak on 2020/03/16.
//  Copyright © 2020 Doyoung Gwak. All rights reserved.
//

import Accelerate
import Foundation

extension CVPixelBuffer {
    var size: CGSize {
        return CGSize(width: CVPixelBufferGetWidth(self), height: CVPixelBufferGetHeight(self))
    }
    
    /// Returns a new `CVPixelBuffer` created by taking the self area and resizing it to the
    /// specified target size. Aspect ratios of source image and destination image are expected to be
    /// same.
    ///
    /// - Parameters:
    ///   - from: Source area of image to be cropped and resized.
    ///   - to: Size to scale the image to(i.e. image size used while training the model).
    /// - Returns: The cropped and resized image of itself.
    func resize(from source: CGRect, to size: CGSize) -> CVPixelBuffer? {
        let inputImageRowBytes = CVPixelBufferGetBytesPerRow(self)
        let imageChannels = 4
        
        CVPixelBufferLockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(self, CVPixelBufferLockFlags(rawValue: 0)) }
        
        // Finds the address of the upper leftmost pixel of the source area.
        guard
            let inputBaseAddress = CVPixelBufferGetBaseAddress(self)?.advanced(
                by: Int(source.minY) * inputImageRowBytes + Int(source.minX) * imageChannels)
            else {
                return nil
        }
        
        // Crops given area as vImage Buffer.
        var croppedImage = vImage_Buffer(
            data: inputBaseAddress, height: UInt(source.height), width: UInt(source.width),
            rowBytes: inputImageRowBytes)
        
        let resultRowBytes = Int(size.width) * imageChannels
        guard let resultAddress = malloc(Int(size.height) * resultRowBytes) else {
            return nil
        }
        
        // Allocates a vacant vImage buffer for resized image.
        var resizedImage = vImage_Buffer(
            data: resultAddress,
            height: UInt(size.height), width: UInt(size.width),
            rowBytes: resultRowBytes
        )
        
        // Performs the scale operation on cropped image and stores it in result image buffer.
        guard vImageScale_ARGB8888(&croppedImage, &resizedImage, nil, vImage_Flags(0)) == kvImageNoError
            else {
                return nil
        }
        
        let releaseCallBack: CVPixelBufferReleaseBytesCallback = { mutablePointer, pointer in
            if let pointer = pointer {
                free(UnsafeMutableRawPointer(mutating: pointer))
            }
        }
        
        var result: CVPixelBuffer?
        
        // Converts the thumbnail vImage buffer to CVPixelBuffer
        let conversionStatus = CVPixelBufferCreateWithBytes(
            nil,
            Int(size.width), Int(size.height),
            CVPixelBufferGetPixelFormatType(self),
            resultAddress,
            resultRowBytes,
            releaseCallBack,
            nil,
            nil,
            &result
        )
        
        guard conversionStatus == kCVReturnSuccess else {
            free(resultAddress)
            return nil
        }
        
        return result
    }
    
    /// Returns the RGB `Data` representation of the given image buffer with the specified
    /// `byteCount`.
    ///
    /// - Parameters:
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    func rgbData(byteCount: Int, isNormalized: Bool = false, isModelQuantized: Bool) -> Data? {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
        let destinationBytesPerRow = 1 * width
        
        // Assign input image to `sourceBuffer` to convert it.
        var sourceBuffer = vImage_Buffer(
            data: sourceData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: sourceBytesPerRow
        )
        
        // Make `destinationBuffer` and `destinationData` for its data to be assigned.
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            os_log("Error: out of memory", type: .error)
            return nil
        }
        defer { free(destinationData) }
        var destinationBuffer = vImage_Buffer(
            data: destinationData,
            height: vImagePixelCount(height),
            width: vImagePixelCount(width),
            rowBytes: destinationBytesPerRow)
        
        // Convert image type.
        switch CVPixelBufferGetPixelFormatType(self) {
        case kCVPixelFormatType_32BGRA:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_32ARGB:
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        case kCVPixelFormatType_OneComponent8:
            //            vImageConvert_Fto16Q12(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
            destinationBuffer = sourceBuffer
        default:
            os_log("The type of this image is not supported.", type: .error)
            return nil
        }
        
        // Make `Data` with converted image.
        let imageByteData = Data(
            bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        
        if isModelQuantized { return imageByteData }
        
        let imageBytes = [UInt8](imageByteData)
        print(imageBytes.count)
        let bytes: [Float32]
        if isNormalized {
            bytes = imageBytes.map { Float32($0) / 255.0 } // normalization
        } else {
            bytes = imageBytes.map { Float32($0) } // not normalization
        }
        return Data(copyingBufferOf: bytes)
    }
    
//    func grayData(isNormalized: Bool = false, isModelQuantized: Bool) -> Data? {
//        CVPixelBufferLockBaseAddress(self, .readOnly)
//        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
//        guard let sourceData = CVPixelBufferGetBaseAddress(self) else {
//            return nil
//        }
//
//        let width = CVPixelBufferGetWidth(self)
//        let height = CVPixelBufferGetHeight(self)
//        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(self)
//        let destinationBytesPerRow = 1 * width
//
//        // Assign input image to `sourceBuffer` to convert it.
//        var sourceBuffer = vImage_Buffer(
//            data: sourceData,
//            height: vImagePixelCount(height),
//            width: vImagePixelCount(width),
//            rowBytes: 28//sourceBytesPerRow
//        )
//
//        //let baseAddress = CVPixelBufferGetBaseAddress(self)
//        // let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
//        //let buffer = baseAddress!.assumingMemoryBound(to: UInt8.self)
//
//        //            // Make `destinationBuffer` and `destinationData` for its data to be assigned.
//        //            guard let destinationData = malloc(height * destinationBytesPerRow) else {
//        //                os_log("Error: out of memory", type: .error)
//        //                return nil
//        //            }
//        //            defer { free(destinationData) }
//        //            var destinationBuffer = vImage_Buffer(
//        //                data: destinationData,
//        //                height: vImagePixelCount(height),
//        //                width: vImagePixelCount(width),
//        //                rowBytes: destinationBytesPerRow)
//        //
//        //            // Convert image type.
//        //            switch CVPixelBufferGetPixelFormatType(self) {
//        //            case kCVPixelFormatType_OneComponent8:
//        //    //            vImageConvert_Fto16Q12(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
//        //                destinationBuffer = sourceBuffer
//        //            default:
//        //                os_log("The type of this image is not supported.", type: .error)
//        //                return nil
//        //            }
//
//        // Make `Data` with converted image.
//        let imageByteData = Data(
//            bytes: sourceBuffer.data, count: sourceBuffer.rowBytes * height)
//
//        if isModelQuantized { return imageByteData }
//
//        let imageBytes = [UInt8](imageByteData)
//        print(imageBytes.count)
//        let bytes: [Float32]
//        var maxv: Float32 = -1.0
//        var minv: Float32 = 1.0
//        if isNormalized {
//            bytes = imageBytes.map {
//                maxv = max(Float32($0), maxv)
//                minv = min(Float32($0), minv)
//                return Float32($0) / 255.0
//            } // normalization
//
//            for i in 0..<28 {
//                var str = ""
//                for j in 0..<28 {
//                    str += String(format: "% 3d, ", imageBytes[i + j*28])
//                }
//                print(str)
//            }
//        } else {
//            bytes = imageBytes.map { Float32($0) } // not normalization
//        }
//        return Data(copyingBufferOf: bytes)
//    }
    
    func grayData(isNormalized: Bool = false, isModelQuantized: Bool) -> Data? {
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }
        guard let baseAddress = CVPixelBufferGetBaseAddress(self) else { return nil }
        
        let width = CVPixelBufferGetWidth(self)
        let height = CVPixelBufferGetHeight(self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(self)
        
        let buffer = baseAddress.assumingMemoryBound(to: UInt8.self)
        var pixelData = [UInt8](repeating: 0, count: width * height)
        pixelData = pixelData.enumerated().map { buffer[$0.offset] }
        
        let bytes: [Float32]
        if isNormalized {
            bytes = pixelData.map { Float32($0) / 255.0 }
        } else {
            bytes = pixelData.map { Float32($0) }
        }
        
        
        
        
        for i in 0..<width {
            var str = ""
            for j in 0..<height {
                str += String(format: "% 3d, ", pixelData[i*28 + j])
            }
            print(str)
        }
        print()
        print()
        
        let px = self
        
        return Data(copyingBufferOf: bytes)
    }
    
    func pixelData() -> [UInt8]? {
        let size = self.size
        let dataSize = size.width * size.height * 1
        var pixelData = [UInt8](repeating: 0, count: Int(dataSize))
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let context = CGContext(data: &pixelData,
                                width: Int(size.width),
                                height: Int(size.height),
                                bitsPerComponent: 8,
                                bytesPerRow: 4 * Int(size.width),
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        // guard let cgImage = self.cgImage else { return nil }
        // context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        // context.draw

        return pixelData
    }
}
