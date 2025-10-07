//
//  LLMModel.swift
//  MusicTest
//
//  Created by Andy Lyu on 10/7/25.
//

import Foundation
import Combine

@MainActor
class LLMModel: ObservableObject {
    @Published var apiOutput: String = ""
    @Published var isLoading: Bool = false
    
    private let apiKey = "YOUR_API_KEY"
    private let baseURL = "https://api.siliconflow.cn/v1/chat/completions"
    private let model = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    
    private var conversationHistory: [ChatMessage] = [
        ChatMessage(role: "system", content: "你是一位心理疗愈师，请根据输入的画面描述和用户情绪从以下歌曲列表中选择一首能改善用户负面情绪的歌曲(比较重要)，如果用户是正面情绪选择情绪对应的歌曲，不同情绪对应不同歌曲。只需要输出歌曲名称！{BeforeDawn(平静，缓慢，适合助眠), coolest(快速 有激情 兴奋), 蜜桃物语(放松 欢快)}，括号内是歌曲的情绪。")
    ]
    
    // Initialize the LLM with system prompt
    func initializeLLM() async {
        isLoading = true
        defer { isLoading = false }
        guard let url = URL(string: baseURL) else {
            apiOutput = "Invalid URL"
            return
        }
        
        let requestBody = ChatRequest(
            model: model,
            messages: conversationHistory,
            temperature: 0.8,
            stream: false
        )
        
        guard let bodyData = try? JSONEncoder().encode(requestBody) else {
            apiOutput = "Failed to encode request body"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = bodyData
        request.timeoutInterval = 30
        
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            
            // Try to decode as error response first
            if let errorResponse = try? JSONDecoder().decode(ChatResponse.self, from: data),
               let error = errorResponse.error {
                apiOutput = "初始化错误: \(error.message)"
                return
            }
            // Decode as success response
            let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
            if let reply = decoded.choices?.first?.message.content {
                apiOutput = "\(reply)"
                conversationHistory.append(ChatMessage(role: "user", content: reply))
            } else {
                apiOutput = "LLM初始化完成"
            }
        } catch {
            apiOutput = "初始化失败: \(error.localizedDescription)"
        }
    }
    
    // Send message to LLM with FastVLM output
    func sendMessage(fastVLMOutput: String) async {
        guard !fastVLMOutput.isEmpty else { return }
        
        isLoading = true
        defer { isLoading = false }
        
        // Add user message to conversation history
        let userMessage = ChatMessage(role: "user", content: fastVLMOutput)
        conversationHistory.append(userMessage)
        
        guard let url = URL(string: baseURL) else {
            apiOutput = "Invalid URL"
            return
        }
        
        let requestBody = ChatRequest(
            model: model,
            messages: conversationHistory,
            temperature: 0.5,
            stream: false
        )
        
        guard let bodyData = try? JSONEncoder().encode(requestBody) else {
            apiOutput = "Failed to encode request body"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.httpBody = bodyData
        request.timeoutInterval = 30
        
        do {
            let (data, _) = try await URLSession.shared.data(for: request)
            
            // Try to decode as error response first
            if let errorResponse = try? JSONDecoder().decode(ChatResponse.self, from: data),
               let error = errorResponse.error {
                apiOutput = "API错误: \(error.message)"
                return
            }
            let decoded = try JSONDecoder().decode(ChatResponse.self, from: data)
            if let reply = decoded.choices?.first?.message.content {
                DispatchQueue.main.async {
                    self.apiOutput = reply
                    self.conversationHistory.append(ChatMessage(role: "user", content: reply))
                }
            } else {
                DispatchQueue.main.async {
                    self.apiOutput = "未收到有效响应"
                }
            }
        } catch {
            apiOutput = "请求失败: \(error.localizedDescription)"
        }
    }
}
