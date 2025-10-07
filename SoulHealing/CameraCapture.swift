//
//  CameraCapture.swift
//  MusicTest
//
//  Created by Andy Lyu on 10/7/25.
//

import Foundation
import AVFoundation
import AppKit
import SwiftUI

struct CameraCaptureView: NSViewRepresentable {
    var onFrameCaptured: ((NSImage) -> Void)? = nil
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        view.wantsLayer = true
        
        AVCaptureDevice.requestAccess(for: .video) { granted in
            if granted {
                DispatchQueue.main.async {
                    context.coordinator.setupCaptureSession(view: view, onFrameCaptured: onFrameCaptured)
                }
            } else {
                print("Camera access denied")
            }
        }
        
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        context.coordinator.previewLayer?.frame = nsView.bounds
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
        var captureSession: AVCaptureSession?
        var previewLayer: AVCaptureVideoPreviewLayer?
        var lastCaptureTime: Date = .distantPast
        var onFrameCaptured: ((NSImage) -> Void)?
        
        func setupCaptureSession(view: NSView, onFrameCaptured: ((NSImage) -> Void)?) {
            self.onFrameCaptured = onFrameCaptured
            
            let session = AVCaptureSession()
            session.sessionPreset = .high
            
            guard let device = AVCaptureDevice.default(for: .video) else {
                print("No camera found")
                return
            }
            
            do {
                let input = try AVCaptureDeviceInput(device: device)
                if session.canAddInput(input) {
                    session.addInput(input)
                }
            } catch {
                print("Error setting up camera input: \(error)")
                return
            }
            
            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
            ]
            output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
            if session.canAddOutput(output) {
                session.addOutput(output)
            }
            
            let layer = AVCaptureVideoPreviewLayer(session: session)
            layer.videoGravity = .resizeAspectFill
            layer.frame = view.bounds
            view.layer?.addSublayer(layer)
            
            self.previewLayer = layer
            self.captureSession = session
            
            DispatchQueue.global(qos: .userInitiated).async {
                session.startRunning()
            }
        }
    
        func captureOutput(_ output: AVCaptureOutput,
                           didOutput sampleBuffer: CMSampleBuffer,
                           from connection: AVCaptureConnection) {
            
            let now = Date()
            if now.timeIntervalSince(lastCaptureTime) < 1 {
                return
            }
            lastCaptureTime = now
            
            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            let ciImage = CIImage(cvPixelBuffer: imageBuffer)
            let rep = NSCIImageRep(ciImage: ciImage)
            let nsImage = NSImage(size: rep.size)
            nsImage.addRepresentation(rep)
            
            DispatchQueue.main.async {
                self.onFrameCaptured?(nsImage)
            }
        }
    }
}
