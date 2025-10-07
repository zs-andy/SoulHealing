import SwiftUI
import AVFoundation
import MLXLMCommon

struct ContentView: View {
    @StateObject private var llmModel = LLMModel()
    @State private var vlmModel = FastVLMModel()
    @State private var fastVLMOutput: String = ""
    @State private var isLLMLoaded = false
    @State private var count = 0
    @State private var currentSongName: String = ""
    @State private var audioPlayer: AVAudioPlayer?
    
    private let systemPrompt: String = "请你观察画面，一句话描述画面中的人物。要求必须描述人物的情绪，需要对负面情绪更加敏感"
    
    private func fastVLM(_ uiImage: NSImage) {
        guard let cgImage = uiImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            print("FastVLM: Failed to get CGImage")
            return
        }
        let ciImage = CIImage(cgImage: cgImage)
        
        Task {
            let userInput = UserInput(
                prompt: .text(systemPrompt),
                images: [.ciImage(ciImage)]
            )
            _ = await vlmModel.generate(userInput)
        }
    }
    
    private func playSong(named songName: String) {
        print("Playing song: \(songName)")
        guard let url = Bundle.main.url(forResource: songName, withExtension: "mp3") else {
            print("Could not find song: \(songName)")
            return
        }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
        } catch {
            print("Could not play song: \(error)")
        }
    }
    
    var body: some View {
        VStack {
            CameraCaptureView(onFrameCaptured: fastVLM)
                .frame(width: 640, height: 480)
            
            VStack(alignment: .leading) {
                Text(fastVLMOutput)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
                Text(llmModel.apiOutput)
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(8)
            }
            .padding()
        }
        .onChange(of: vlmModel.output) { _, newValue in
            fastVLMOutput = newValue
            count += 1
            if isLLMLoaded && !newValue.isEmpty && count >= 30{
                Task {
                    await llmModel.sendMessage(fastVLMOutput: newValue)
                    count = 0
                }
            }
        }
        .onChange(of: llmModel.apiOutput) { _, newValue in
            if !newValue.isEmpty && newValue != currentSongName {
                currentSongName = newValue
                playSong(named: newValue)
            }
        }
        .task {
            await vlmModel.load()
            await llmModel.initializeLLM()
            isLLMLoaded = true
        }
    }
}

#Preview {
    ContentView()
}
