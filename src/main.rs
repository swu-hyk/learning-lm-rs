mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod chat;
mod api;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};
use std::fs;
use crate::chat::Chat;
use crate::config::LlamaConfigJson;
use crate::params::ChatParams;
use safetensors::SafeTensors;

fn print_banner() {
    println!("\n🤖 Welcome to My Rust4LLM Console");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Available modes:");
    println!("1. Story - Creative writing mode");
    println!("2. Chat - Interactive conversation mode");
    println!("3. API - Start API service");
    println!("4. Exit");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

fn get_user_input(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    input.trim().to_string()
}

fn chat_mode() {
    // 加载聊天模型
    println!("\n📚 Loading chat model...");
    let config = LlamaConfigJson::from_file("models/chat/config.json".into());
    
    // 创建tokenizer
    let tokenizer = Tokenizer::from_file("models/chat/tokenizer.json").unwrap();
    
    let model_data = fs::read("models/chat/model.safetensors").unwrap();
    let params = ChatParams::from_safetensors(&SafeTensors::deserialize(&model_data).unwrap(), &config);
    let mut chat = Chat::new(params, tokenizer, config);
    
    // 设置聊天模式的系统提示
    chat.set_system_message(
        "You are a helpful AI assistant. Please provide clear, concise, and accurate responses. \
        Be friendly and professional.".to_string()
    );
    
    println!("✨ Chat mode initialized! Type 'exit' to return to main menu.");
    println!("💡 Type 'clear' to clear chat history.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    loop {
        let input = get_user_input("\nYou: ");
        if input.to_lowercase() == "exit" {
            break;
        } else if input.to_lowercase() == "clear" {
            chat.clear_history();
            println!("🧹 Chat history cleared!");
            continue;
        }
        
        print!("Assistant: ");
        io::stdout().flush().unwrap();
        let _response = chat.chat(&input);  
        println!("\n"); 
    }
}

fn story_mode() {
    // 加载故事模型
    println!("\n📚 Loading story model...");
    let model_dir = PathBuf::from("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    println!("✨ Story mode initialized! Type 'exit' to return to main menu.");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    loop {
        let input = get_user_input("\nStory beginning: ");
        if input.to_lowercase() == "exit" {
            break;
        }
        
        print!("\nGenerating story continuation...\n\n");
        io::stdout().flush().unwrap();
        
        // 使用 model.rs 中的 generate 函数
        let tokens = tokenizer.encode(input.as_str(), true).unwrap();
        let input_ids = tokens.get_ids();
        let output_ids = llama.generate(
            input_ids,
            1000,   
            0.95,  
            40,    
            0.8, 
        );
        
        print!("{}", input);  
        println!("{}\n", tokenizer.decode(&output_ids[input_ids.len()..], true).unwrap()); 
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    }
}

fn api_mode() {
    println!("\n🌐 Starting API service...");
    
    // 提供可以访问story模型和chat模型的API服务
    match crate::api::start_api_server() {
        Ok(_) => println!("API服务已停止"),
        Err(e) => println!("API服务启动失败: {}", e),
    }
}

fn main() {
    loop {
        print_banner();
        
        match get_user_input("Please select a mode (1-4): ").as_str() {
            "1" => story_mode(),
            "2" => chat_mode(),
            "3" => api_mode(),
            "4" => {
                println!("\n👋 Goodbye!");
                break;
            }
            _ => println!("\n❌ Invalid choice. Please try again."),
        }
    }
}
